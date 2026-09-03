# cache_activations.py
"""
Step 2: Residual Stream Activation Caching & Baseline Logit Evaluation
======================================================================
Extracts and caches resid_post activations across all layers at the final
prompt token position for the 50-item Looking-Glass dataset.

Features:
  1. Assistant Prefill formatting via chat templates (continue_final_message=True)
     to ensure single-token target evaluation without conversational preamble.
  2. Base model fallback (raw completion strings without chat tokens).
  3. Memory-efficient caching: stores only the final token slice [layers, d_model].
  4. Complete metadata tracking: saves activations, baseline logit differences,
     and token IDs to a single PyTorch archive for downstream probing and patching.
"""

import os
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils
from looking_glass_dataset import LOOKING_GLASS_DATASET


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Options: "meta-llama/Meta-Llama-3.1-8B-Instruct", "google/gemma-2-9b-it", "google/gemma-2-2b-it"
MODEL_NAME = "google/gemma-2-9b-it" 
OUTPUT_DIR = "./cached_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{MODEL_NAME.split('/')[-1]}_cached_acts.pt")

CONDITIONS = [
    "c1_flattery_false",
    "c2_auditor_false",
    "c3_flattery_true",
    "c4_neutral"
]


# ---------------------------------------------------------------------------
# Prompt Formatting & Assistant Prefill Helper
# ---------------------------------------------------------------------------
def format_prompt_with_prefill(raw_text: str, tokenizer, is_instruct: bool,
                               condition: str = "") -> str:
    """
    Formats a prompt for single-token logit evaluation.

    For instruct models, uses assistant-turn prefill so the model continues
    the completion stem directly without conversational preamble.

    Structure:
      C1/C2/C3 — "<Claim>. <Persona>." goes into user turn;
                  "<Completion Stem>" goes into assistant prefill.
      C4       — Entire neutral stem goes into assistant prefill with a
                  minimal, unbiased user turn (no persona/accuracy cues).
      Base     — Raw text, no chat template.
    """
    if not is_instruct:
        return raw_text

    if condition == "c4_neutral":
        # C4 prompts are single flowing statements with no period-space
        # boundary.  Use a minimal user turn that carries zero persona or
        # epistemic signal — just enough to satisfy the chat template.
        user_turn = "Continue:"
        assistant_stem = raw_text
    elif ". " in raw_text:
        # C1/C2/C3: split at the last period-space to separate
        # claim+persona (user turn) from completion stem (assistant prefill)
        user_body, assistant_stem = raw_text.rsplit(". ", 1)
        user_turn = user_body + "."
    else:
        # Safety fallback — should not be reached with the v2 dataset
        user_turn = "Continue:"
        assistant_stem = raw_text

    messages = [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": assistant_stem}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True
    )
    return formatted_prompt


# ---------------------------------------------------------------------------
# Main Caching Pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")

    # 1. Load Model
    print(f"[Model] Loading {MODEL_NAME} in bfloat16...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        torch_dtype=torch.bfloat16
    )
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    is_instruct = any(x in MODEL_NAME.lower() for x in ["instruct", "-it"])

    print(f"[Config] Model has {n_layers} layers, d_model={d_model}, is_instruct={is_instruct}")

    # 2. Hook Names Setup for resid_post across all layers
    hook_names = [utils.get_act_name("resid_post", l) for l in range(n_layers)]

    n_items = len(LOOKING_GLASS_DATASET)
    n_conds = len(CONDITIONS)

    # Tensor storage: [n_conditions, n_items, n_layers, d_model]
    cached_acts = torch.zeros((n_conds, n_items, n_layers, d_model), dtype=torch.float32)
    cached_logit_diffs = torch.zeros((n_conds, n_items), dtype=torch.float32)
    cached_logits_true = torch.zeros((n_conds, n_items), dtype=torch.float32)
    cached_logits_false = torch.zeros((n_conds, n_items), dtype=torch.float32)

    metadata = []

    print("[Pipeline] Caching residual stream activations across 50 items x 4 conditions...")

    # 3. Execution Loop
    for item_idx, item in enumerate(tqdm(LOOKING_GLASS_DATASET, desc="Items")):
        # Resolve single-token IDs[cite: 1]
        tok_true_id = model.to_single_token(item["tok_true"])
        tok_false_id = model.to_single_token(item["tok_false"])

        item_meta = {
            "id": item["id"],
            "tok_true": item["tok_true"],
            "tok_false": item["tok_false"],
            "tok_true_id": tok_true_id,
            "tok_false_id": tok_false_id,
        }
        metadata.append(item_meta)

        for cond_idx, cond_key in enumerate(CONDITIONS):
            raw_text = item[cond_key]
            prompt = format_prompt_with_prefill(
                raw_text, model.tokenizer, is_instruct, condition=cond_key
            )

            # Pre-tokenize to avoid double-BOS: apply_chat_template already
            # inserts <bos>/<|begin_of_text|> for instruct models, so we must
            # suppress TransformerLens's default BOS prepend for those.
            # Base models get no chat template, so they DO need the BOS.
            tokens = model.to_tokens(prompt, prepend_bos=(not is_instruct))

            # Forward pass with activation caching filter
            with torch.no_grad():
                logits, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name in hook_names,
                    return_type="logits"
                )

            # Extract logits at final token position
            last_token_logits = logits[0, -1, :]
            l_true = last_token_logits[tok_true_id].item()
            l_false = last_token_logits[tok_false_id].item()
            l_diff = l_true - l_false

            cached_logit_diffs[cond_idx, item_idx] = l_diff
            cached_logits_true[cond_idx, item_idx] = l_true
            cached_logits_false[cond_idx, item_idx] = l_false

            # Extract resid_post at final token position across all layers
            for l in range(n_layers):
                act_layer = cache[hook_names[l]][0, -1, :].detach().cpu().to(torch.float32)
                cached_acts[cond_idx, item_idx, l, :] = act_layer

            # Free activation cache to prevent VRAM accumulation
            del logits, cache

    # 4. Save to Disk
    payload = {
        "activations": cached_acts,             # [4, 50, n_layers, d_model]
        "logit_diffs": cached_logit_diffs,     # [4, 50]
        "logits_true": cached_logits_true,     # [4, 50]
        "logits_false": cached_logits_false,   # [4, 50]
        "metadata": metadata,
        "conditions": CONDITIONS,
        "model_name": MODEL_NAME,
        "n_layers": n_layers,
        "d_model": d_model,
    }

    torch.save(payload, OUTPUT_FILE)
    print(f"\n[Done] Successfully saved cached states to {OUTPUT_FILE}")

    # 5. Print Baseline Summary
    print("\n--- Baseline Mean Logit Differences (Logit(True) - Logit(False)) ---")
    for cond_idx, cond_name in enumerate(CONDITIONS):
        mean_diff = cached_logit_diffs[cond_idx].mean().item()
        std_diff = cached_logit_diffs[cond_idx].std().item()
        print(f"  {cond_name:<20s}: Mean = {mean_diff:+.3f} (± {std_diff:.3f})")


if __name__ == "__main__":
    main()