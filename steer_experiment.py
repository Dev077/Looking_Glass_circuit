# steer_experiment.py
"""
Step 5: Causal Activation Steering & Sycophancy-to-Truth Flipping
================================================================
Scale-agnostic version: alpha values are expressed as fractions of the
mean residual stream norm at the injection layer, so the same script
produces meaningful interventions on models from 2B to 27B+.

Sweeps norm-relative alpha across:
  - v_lg_diff_means (Difference-in-means)
  - v_lg_probe      (Un-scaled probe normal vector)
  - v_rand          (Random Gaussian noise control of matched norm)
"""

import os
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils

from lg_dataset import LOOKING_GLASS_DATASET


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-2-2b-it"
CACHE_DIR = "./cached_data"
VECTORS_FILE = os.path.join(CACHE_DIR, "extracted_vectors.pt")
CACHED_ACTS_FILE = os.path.join(CACHE_DIR, f"{MODEL_NAME.split('/')[-1]}_cached_acts.pt")
OUTPUT_PLOT = os.path.join(CACHE_DIR, "causal_steering_curve.png")

# Alpha as FRACTION of mean residual norm at injection layer.
# 0.0 = no intervention, 0.1 = 10% of residual norm, etc.
# This range works across model scales because it's relative.
ALPHA_FRACS = np.linspace(-0.15, 0.45, 13)
OVERRIDE_INJECTION_LAYER = None

# ---------------------------------------------------------------------------
# 1. Load Artifacts & Model
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 1. Load Artifacts & Model
# ---------------------------------------------------------------------------
if not os.path.exists(VECTORS_FILE):
    raise FileNotFoundError(f"Vectors file not found at {VECTORS_FILE}. Run train_probes.py first.")

print(f"[Loading] Reading vectors from {VECTORS_FILE}...")
vector_data = torch.load(VECTORS_FILE, map_location="cpu", weights_only=False)

# Select injection layer: explicit override if set, otherwise the probe's peak
if OVERRIDE_INJECTION_LAYER is not None:
    peak_layer = OVERRIDE_INJECTION_LAYER
    print(f"[Target] Manual override active: steering at Layer {peak_layer}")
else:
    peak_layer = vector_data["peak_layer_lg"]
    print(f"[Target] Using peak layer from probe: Layer {peak_layer}")

v_lg_diff = vector_data["v_lg_diff_means"]
v_lg_probe = vector_data["v_lg_probe"]

# Load cached activations to compute residual norm at injection layer
print(f"[Loading] Reading cached activations for norm calibration...")
cached_payload = torch.load(CACHED_ACTS_FILE, map_location="cpu", weights_only=False)
cached_logit_diffs = cached_payload["logit_diffs"]
cond_names = cached_payload["conditions"]

# Compute mean residual stream norm at peak layer across all C1 items
# This sets the scale for alpha: alpha_actual = alpha_frac * resid_norm
resid_norms = cached_payload["activations"][0, :, peak_layer, :].float().norm(dim=-1)
mean_resid_norm = resid_norms.mean().item()

# Convert fractional alphas to absolute values
ALPHA_VALS = ALPHA_FRACS * mean_resid_norm

print(f"[Scaling] Mean residual norm at layer {peak_layer}: {mean_resid_norm:.1f}")
print(f"[Scaling] Alpha fractions: [{ALPHA_FRACS[0]:.2f}, ..., {ALPHA_FRACS[-1]:.2f}]")
print(f"[Scaling] Effective alpha range: [{ALPHA_VALS[0]:.1f}, ..., {ALPHA_VALS[-1]:.1f}]")

# Baselines from cache
c1_base_mean = cached_logit_diffs[cond_names.index("c1_flattery_false")].mean().item()
c2_base_mean = cached_logit_diffs[cond_names.index("c2_auditor_false")].mean().item()
del cached_payload  # free RAM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using: {device}")

print(f"[Model] Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_NAME,
    device=device,
    dtype=torch.bfloat16
)
is_instruct = any(x in MODEL_NAME.lower() for x in ["instruct", "-it"])
model_dtype = model.cfg.dtype

# Prepare unit steering vectors in model device and dtype
v_diff_t = v_lg_diff.to(device=device, dtype=model_dtype)
v_diff_t = v_diff_t / torch.norm(v_diff_t)

v_probe_t = v_lg_probe.to(device=device, dtype=model_dtype)
v_probe_t = v_probe_t / torch.norm(v_probe_t)

# Negative Control: Random Gaussian Vector with matched unit norm
torch.manual_seed(42)
v_rand_t = torch.randn_like(v_diff_t)
v_rand_t = v_rand_t / torch.norm(v_rand_t)

hook_name = utils.get_act_name("resid_post", peak_layer)
n_layers = model.cfg.n_layers
print(f"[Target] Intervening at Layer {peak_layer}/{n_layers} ('{hook_name}')")


# ---------------------------------------------------------------------------
# 2. Prompt Formatting Helper
# ---------------------------------------------------------------------------
def format_prompt(raw_text: str) -> str:
    if not is_instruct:
        return raw_text
    user_body, assistant_stem = raw_text.rsplit(". ", 1)
    assistant_role = "model" if "gemma" in MODEL_NAME.lower() else "assistant"
    messages = [
        {"role": "user", "content": user_body + "."},
        {"role": assistant_role, "content": assistant_stem}
    ]
    return model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True
    )


# ---------------------------------------------------------------------------
# 3. Hook Function for Residual Injection
# ---------------------------------------------------------------------------
def activation_steering_hook(resid_post, hook, steer_vec, alpha):
    """Injects alpha * steer_vec into resid_post at the final token position."""
    res = resid_post.clone()
    res[:, -1, :] = res[:, -1, :] + alpha * steer_vec
    return res


# ---------------------------------------------------------------------------
# 4. Run Steering Intervention Across Dataset
# ---------------------------------------------------------------------------
n_items = len(LOOKING_GLASS_DATASET)
n_alphas = len(ALPHA_VALS)

results_diff = np.zeros((n_alphas, n_items))
results_probe = np.zeros((n_alphas, n_items))
results_rand = np.zeros((n_alphas, n_items))

print(f"\n[Steering] {n_alphas} alpha values × {n_items} items (norm-relative scaling)...")

for item_idx, item in enumerate(tqdm(LOOKING_GLASS_DATASET, desc="C1 Steer")):
    tok_true_id = model.to_single_token(item["tok_true"])
    tok_false_id = model.to_single_token(item["tok_false"])

    prompt_str = format_prompt(item["c1_flattery_false"])
    tokens = model.to_tokens(prompt_str, prepend_bos=(not is_instruct))

    for a_idx, alpha in enumerate(ALPHA_VALS):
        # 1. Diff-in-Means Steering
        hook_diff = lambda act, hook, a=alpha: activation_steering_hook(act, hook, v_diff_t, a)
        with torch.no_grad():
            logits_diff = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_diff)],
                return_type="logits"
            )[0, -1, :]
            results_diff[a_idx, item_idx] = (logits_diff[tok_true_id] - logits_diff[tok_false_id]).item()

        # 2. Probe Weight Steering
        hook_probe = lambda act, hook, a=alpha: activation_steering_hook(act, hook, v_probe_t, a)
        with torch.no_grad():
            logits_probe = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_probe)],
                return_type="logits"
            )[0, -1, :]
            results_probe[a_idx, item_idx] = (logits_probe[tok_true_id] - logits_probe[tok_false_id]).item()

        # 3. Random Vector Baseline
        hook_rand = lambda act, hook, a=alpha: activation_steering_hook(act, hook, v_rand_t, a)
        with torch.no_grad():
            logits_rand = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_rand)],
                return_type="logits"
            )[0, -1, :]
            results_rand[a_idx, item_idx] = (logits_rand[tok_true_id] - logits_rand[tok_false_id]).item()


# ---------------------------------------------------------------------------
# 4b. Reverse Control: Steer C2 (Auditor) Toward Sycophancy
# ---------------------------------------------------------------------------
results_c2_reverse = np.zeros((n_alphas, n_items))

print(f"\n[Reverse] Steering C2 (Auditor) — negative alpha should induce sycophancy...")

for item_idx, item in enumerate(tqdm(LOOKING_GLASS_DATASET, desc="C2 Reverse")):
    tok_true_id = model.to_single_token(item["tok_true"])
    tok_false_id = model.to_single_token(item["tok_false"])

    prompt_str = format_prompt(item["c2_auditor_false"])
    tokens = model.to_tokens(prompt_str, prepend_bos=(not is_instruct))

    for a_idx, alpha in enumerate(ALPHA_VALS):
        hook_fn = lambda act, hook, a=alpha: activation_steering_hook(act, hook, v_diff_t, a)
        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_fn)],
                return_type="logits"
            )[0, -1, :]
            results_c2_reverse[a_idx, item_idx] = (logits[tok_true_id] - logits[tok_false_id]).item()


# ---------------------------------------------------------------------------
# 5. Sanity Check
# ---------------------------------------------------------------------------
alpha_zero_idx = int(np.argmin(np.abs(ALPHA_VALS)))
alpha_zero_mean = results_diff[alpha_zero_idx].mean()
drift = abs(alpha_zero_mean - c1_base_mean)
print(f"\n[Sanity] alpha≈0 LogitDiff: {alpha_zero_mean:+.4f} | Cached C1: {c1_base_mean:+.4f} | Drift: {drift:.4f}")
if drift > 0.5:
    print("[WARNING] alpha≈0 drifted from baseline.")
else:
    print("[OK] Baseline matches.")


# ---------------------------------------------------------------------------
# 6. Summary Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 105)
print(f"{'α frac':>7} {'α abs':>8} | {'Diff-Means Δ':>13} {'Acc%':>6} | {'Probe Δ':>13} {'Acc%':>6} | {'Random Δ':>13} {'Acc%':>6}")
print("-" * 105)
for a_idx in range(n_alphas):
    af = ALPHA_FRACS[a_idx]
    aa = ALPHA_VALS[a_idx]
    m_diff = results_diff[a_idx].mean()
    m_probe = results_probe[a_idx].mean()
    m_rand = results_rand[a_idx].mean()
    acc_diff = (results_diff[a_idx] > 0).mean() * 100
    acc_probe = (results_probe[a_idx] > 0).mean() * 100
    acc_rand = (results_rand[a_idx] > 0).mean() * 100
    print(f"{af:+7.3f} {aa:+8.1f} | {m_diff:+13.3f} {acc_diff:5.1f}% | {m_probe:+13.3f} {acc_probe:5.1f}% | {m_rand:+13.3f} {acc_rand:5.1f}%")
print("=" * 105)


# ---------------------------------------------------------------------------
# 7. Publication Plot
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Use fractional alpha for x-axis (model-agnostic)
mean_diff = results_diff.mean(axis=1)
sem_diff = results_diff.std(axis=1) / np.sqrt(n_items)

mean_probe = results_probe.mean(axis=1)
sem_probe = results_probe.std(axis=1) / np.sqrt(n_items)

mean_rand = results_rand.mean(axis=1)
sem_rand = results_rand.std(axis=1) / np.sqrt(n_items)

mean_c2r = results_c2_reverse.mean(axis=1)
sem_c2r = results_c2_reverse.std(axis=1) / np.sqrt(n_items)

ax.plot(ALPHA_FRACS, mean_diff, label="C1 Steer: $\\mathbf{v}_{LG}$ (Diff-Means)", color="#1f77b4", linewidth=2.5, marker="o")
ax.fill_between(ALPHA_FRACS, mean_diff - sem_diff, mean_diff + sem_diff, color="#1f77b4", alpha=0.15)

ax.plot(ALPHA_FRACS, mean_probe, label="C1 Steer: $\\mathbf{v}_{LG}$ (Probe)", color="#9467bd", linewidth=2.0, marker="s", linestyle="--")
ax.fill_between(ALPHA_FRACS, mean_probe - sem_probe, mean_probe + sem_probe, color="#9467bd", alpha=0.12)

ax.plot(ALPHA_FRACS, mean_c2r, label="C2 Reverse: $\\mathbf{v}_{LG}$ (Diff-Means)", color="#ff7f0e", linewidth=2.0, marker="D", linestyle="-.")
ax.fill_between(ALPHA_FRACS, mean_c2r - sem_c2r, mean_c2r + sem_c2r, color="#ff7f0e", alpha=0.12)

ax.plot(ALPHA_FRACS, mean_rand, label="C1 Steer: Random $\\mathbf{v}_{rand}$", color="#7f7f7f", linewidth=1.8, linestyle=":", marker="x")

ax.axhline(c1_base_mean, color="#d62728", linestyle="--", linewidth=1.2, label=f"Unsteered C1 ({c1_base_mean:+.2f})")
ax.axhline(c2_base_mean, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"Unsteered C2 ({c2_base_mean:+.2f})")
ax.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)

ax.set_title(
    f"Bidirectional Causal Steering via $\\mathbf{{v}}_{{LG}}$\n"
    f"({MODEL_NAME}, Layer {peak_layer}/{n_layers}, "
    f"$\\|\\mathbf{{h}}\\|$ = {mean_resid_norm:.0f})",
    fontsize=12, fontweight="bold", pad=12
)
ax.set_xlabel("Steering Multiplier $\\alpha$ (fraction of $\\|\\mathbf{h}\\|$)", fontsize=11)
ax.set_ylabel("Mean Logit Difference: Logit(True) $-$ Logit(False)", fontsize=11)
ax.legend(loc="best", frameon=True, fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
print(f"\n[Done] Causal steering figure saved to {OUTPUT_PLOT}")