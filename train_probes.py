# train_probes.py
"""
Step 3: Layer-Wise Linear Probing, Subspace Orthogonality & Temporal Hierarchy
=============================================================================
Loads cached residual stream activations and:
  1. Trains factorially isolated 5-fold cross-validated linear probes:
       - Probe LG   (v_LG)   : C1 (Flattery) vs C2 (Auditor)   [Claim held constant: False]
       - Probe User (v_user) : C1 (False Claim) vs C3 (True)   [Persona held constant: Flattery]
  2. Measures Ground Truth Recall across layers via Logit Lens vocabulary projections.
  3. Evaluates subspace cosine orthogonality between peak feature directions.
  4. Saves extracted steering vectors (v_LG, v_user) to disk for Step 4.
  5. Renders and saves publication-quality temporal hierarchy curves.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-2-2b-it"
CACHE_DIR = "./cached_data"
CACHE_FILE = os.path.join(CACHE_DIR, f"{MODEL_NAME.split('/')[-1]}_cached_acts.pt")
OUTPUT_PLOT = os.path.join(CACHE_DIR, "temporal_hierarchy.png")
VECTORS_FILE = os.path.join(CACHE_DIR, "extracted_vectors.pt")

RANDOM_SEED = 42
N_SPLITS = 5


# ---------------------------------------------------------------------------
# 1. Load Cached Activations & Metadata
# ---------------------------------------------------------------------------
if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(
        f"Cache file not found at {CACHE_FILE}. Run Step 2 (cache_activations.py) first."
    )

print(f"[Loading] Reading cached activations from {CACHE_FILE}...")
payload = torch.load(CACHE_FILE, map_location="cpu")

# activations shape: [n_conditions, n_items, n_layers, d_model]
activations = payload["activations"]
conditions = payload["conditions"]
metadata = payload["metadata"]
n_layers = payload["n_layers"]
d_model = payload["d_model"]

cond_to_idx = {name: i for i, name in enumerate(conditions)}
idx_c1 = cond_to_idx["c1_flattery_false"]
idx_c2 = cond_to_idx["c2_auditor_false"]
idx_c3 = cond_to_idx["c3_flattery_true"]
idx_c4 = cond_to_idx["c4_neutral"]

n_items = activations.shape[1]
print(f"[Config] Loaded {n_items} items across {n_layers} layers (d_model = {d_model}).")


# ---------------------------------------------------------------------------
# 2. Train Factorially Disentangled Linear Probes
# ---------------------------------------------------------------------------
def evaluate_layer_probe(X_pos, X_neg, n_permutations=100):
    """
    Evaluates 5-fold cross-validated accuracy across all layers.
    X_pos, X_neg: [n_items, n_layers, d_model]
    Returns:
        acc_means  [n_layers]  — real CV accuracy
        acc_stds   [n_layers]  — real CV std
        perm_means [n_layers]  — permutation null baseline (shuffled labels)
        weights    [n_layers, d_model] — unit-norm probe direction in ORIGINAL activation space
    """
    n_samples = X_pos.shape[0]
    y = np.array([1] * n_samples + [0] * n_samples)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    acc_means = np.zeros(n_layers)
    acc_stds = np.zeros(n_layers)
    perm_means = np.zeros(n_layers)
    weights = np.zeros((n_layers, d_model))

    rng = np.random.RandomState(RANDOM_SEED)

    for l in range(n_layers):
        X_l = np.concatenate([X_pos[:, l, :].numpy(), X_neg[:, l, :].numpy()], axis=0)

        # Pipeline standardizes features inside CV folds to prevent data leakage
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED))
        ])

        scores = cross_val_score(pipe, X_l, y, cv=cv, scoring="accuracy")
        acc_means[l] = scores.mean()
        acc_stds[l] = scores.std()

        # Permutation null: same pipeline on shuffled labels
        perm_scores = []
        for _ in range(n_permutations):
            y_perm = rng.permutation(y)
            perm_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED))
            ])
            ps = cross_val_score(perm_pipe, X_l, y_perm, cv=N_SPLITS, scoring="accuracy")
            perm_scores.append(ps.mean())
        perm_means[l] = np.mean(perm_scores)

        # Fit on full data to extract directional vector
        pipe.fit(X_l, y)
        # CRITICAL: un-scale weights from StandardScaler space back to original
        # activation space before normalization. coef_ is in scaled space;
        # dividing by scale_ recovers the direction in the residual stream.
        w_scaled = pipe.named_steps["clf"].coef_[0]
        scale = pipe.named_steps["scaler"].scale_
        w_original = w_scaled / scale
        weights[l] = w_original / (np.linalg.norm(w_original) + 1e-8)

    return acc_means, acc_stds, perm_means, weights


print("\n[Probing] Training Probe LG (v_LG): C2 (Auditor) vs C1 (Flattery)...")
# C1: False claim + Flattery (0) vs C2: False claim + Auditor (1)
acc_lg, std_lg, perm_lg, weights_lg = evaluate_layer_probe(
    X_pos=activations[idx_c2],
    X_neg=activations[idx_c1]
)

print("[Probing] Training Probe User (v_user): C3 (True Claim) vs C1 (False Claim)...")
# C1: False claim + Flattery (0) vs C3: True claim + Flattery (1)
acc_user, std_user, perm_user, weights_user = evaluate_layer_probe(
    X_pos=activations[idx_c3],
    X_neg=activations[idx_c1]
)


# ---------------------------------------------------------------------------
# 3. Ground Truth Recall Trajectory (Logit Lens / Vocabulary Alignment)
# ---------------------------------------------------------------------------
# Across diverse facts, each item has distinct target tokens.
# We evaluate ground truth emergence by projecting residual states onto
# the unembedding difference: (W_U[tok_true] - W_U[tok_false]).
print("\n[Logit Lens] Measuring Ground Truth Recall across layers...")
fact_recall_neutral = np.zeros(n_layers)
fact_recall_sycophancy = np.zeros(n_layers)

try:
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Model] Loading unembedding matrix from {MODEL_NAME} ({device})...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, torch_dtype=torch.bfloat16)
    model_dtype = model.cfg.dtype
    W_U = model.W_U.detach().cpu().to(torch.float32)  # [d_model, d_vocab]

    # Pre-build the per-item unembedding difference vectors: [n_items, d_model]
    d_tok_all = torch.stack([
        W_U[:, meta["tok_true_id"]] - W_U[:, meta["tok_false_id"]]
        for meta in metadata
    ])  # [n_items, d_model]

    with torch.no_grad():
        for l in range(n_layers):
            # Batch all 50 items at once: [n_items, d_model]
            h_c4_batch = activations[idx_c4, :, l, :].to(device=device, dtype=model_dtype)
            h_c1_batch = activations[idx_c1, :, l, :].to(device=device, dtype=model_dtype)

            # Apply final LayerNorm in one batched call
            h_c4_normed = model.ln_final(h_c4_batch).cpu().to(torch.float32)  # [n_items, d_model]
            h_c1_normed = model.ln_final(h_c1_batch).cpu().to(torch.float32)

            # Vectorized dot product: element-wise multiply then sum over d_model
            dots_c4 = (h_c4_normed * d_tok_all).sum(dim=1)  # [n_items]
            dots_c1 = (h_c1_normed * d_tok_all).sum(dim=1)

            fact_recall_neutral[l] = (dots_c4 > 0).float().mean().item()
            fact_recall_sycophancy[l] = (dots_c1 > 0).float().mean().item()

    has_logit_lens = True
except Exception as e:
    print(f"[Warning] Could not load model for Logit Lens ({e}). Using baseline logit stats.")
    has_logit_lens = False


# ---------------------------------------------------------------------------
# 4. Subspace Orthogonality & Vector Extraction
# ---------------------------------------------------------------------------
peak_layer_lg = int(np.argmax(acc_lg))
peak_layer_user = int(np.argmax(acc_user))
later_peak = max(peak_layer_lg, peak_layer_user)

v_lg_peak = weights_lg[peak_layer_lg]
v_user_peak = weights_user[peak_layer_user]

# Same-layer comparison (both at the later peak) for a cleaner orthogonality test
v_lg_at_later = weights_lg[later_peak]
v_user_at_later = weights_user[later_peak]

# Difference-in-Means vectors in original activation space (for steering)
diff_means_lg = (
    activations[idx_c2, :, peak_layer_lg, :].mean(dim=0) -
    activations[idx_c1, :, peak_layer_lg, :].mean(dim=0)
).numpy()
diff_means_lg /= (np.linalg.norm(diff_means_lg) + 1e-8)

diff_means_user = (
    activations[idx_c3, :, peak_layer_user, :].mean(dim=0) -
    activations[idx_c1, :, peak_layer_user, :].mean(dim=0)
).numpy()
diff_means_user /= (np.linalg.norm(diff_means_user) + 1e-8)

# Cosine similarities
cos_probe_cross = np.dot(v_lg_peak, v_user_peak)          # cross-layer (each at own peak)
cos_probe_same  = np.dot(v_lg_at_later, v_user_at_later)  # same layer
cos_diff_means  = np.dot(diff_means_lg, diff_means_user)  # diff-means sanity check

print("\n" + "=" * 65)
print("GEOMETRIC & TEMPORAL PROBING RESULTS")
print("=" * 65)
print(f"Probe LG   (v_LG)   Peak: Layer {peak_layer_lg:2d} | Accuracy: {acc_lg[peak_layer_lg]*100:5.1f}% (± {std_lg[peak_layer_lg]*100:.1f}%) | Perm null: {perm_lg[peak_layer_lg]*100:.1f}%")
print(f"Probe User (v_user) Peak: Layer {peak_layer_user:2d} | Accuracy: {acc_user[peak_layer_user]*100:5.1f}% (± {std_user[peak_layer_user]*100:.1f}%) | Perm null: {perm_user[peak_layer_user]*100:.1f}%")
print(f"\nCosine Similarity (cross-layer, each at own peak):  |cos| = {abs(cos_probe_cross):.4f}")
print(f"Cosine Similarity (same layer {later_peak}):              |cos| = {abs(cos_probe_same):.4f}")
print(f"Cosine Similarity (diff-means vectors):              |cos| = {abs(cos_diff_means):.4f}")

if abs(cos_probe_same) < 0.20:
    print("[Verification] SUB-SPACES ARE ORTHOGONAL (|cos| < 0.20). Hypothesis supported.")
else:
    print("[Verification] Subspaces exhibit linear correlation. Check confounding phrases.")

# Save vectors for Step 4
torch.save({
    "peak_layer_lg": peak_layer_lg,
    "peak_layer_user": peak_layer_user,
    "v_lg_probe": torch.tensor(v_lg_peak, dtype=torch.float32),
    "v_user_probe": torch.tensor(v_user_peak, dtype=torch.float32),
    "v_lg_diff_means": torch.tensor(diff_means_lg, dtype=torch.float32),
    "v_user_diff_means": torch.tensor(diff_means_user, dtype=torch.float32),
    "cos_probe_cross": cos_probe_cross,
    "cos_probe_same": cos_probe_same,
    "cos_diff_means": cos_diff_means,
    "model_name": MODEL_NAME,
}, VECTORS_FILE)
print(f"[Saved] Extracted steering vectors written to {VECTORS_FILE}")


# ---------------------------------------------------------------------------
# 5. Plot Publication-Quality Temporal Hierarchy Curves
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

layers = np.arange(n_layers)

# 1. Probe User (Claim)
ax.plot(layers, acc_user, label="User Claim ($v_{user}$)", color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
ax.fill_between(layers, acc_user - std_user, acc_user + std_user, color="#1f77b4", alpha=0.15)

# 2. Probe Looking-Glass (Persona Expectation)
ax.plot(layers, acc_lg, label="Looking-Glass ($v_{LG}$)", color="#d62728", linewidth=2.5, marker="s", markersize=4)
ax.fill_between(layers, acc_lg - std_lg, acc_lg + std_lg, color="#d62728", alpha=0.15)

# 3. Permutation null baselines (p >> n sanity check)
ax.plot(layers, perm_user, label="Permutation Null (User)", color="#1f77b4", linewidth=1.0, linestyle=":", alpha=0.5)
ax.plot(layers, perm_lg, label="Permutation Null (LG)", color="#d62728", linewidth=1.0, linestyle=":", alpha=0.5)

# 4. Ground Truth Recall
if has_logit_lens:
    ax.plot(layers, fact_recall_neutral, label="Factual Recall (Neutral)", color="#2ca02c", linewidth=2.0, linestyle="--")
    ax.plot(layers, fact_recall_sycophancy, label="Factual Recall (Sycophancy Trap)", color="#ff7f0e", linewidth=2.0, linestyle="--")

# Chance level
ax.axhline(0.50, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="Chance (50%)")

# Mark peak layers
ax.axvline(peak_layer_user, color="#1f77b4", linestyle=":", alpha=0.6)
ax.axvline(peak_layer_lg, color="#d62728", linestyle=":", alpha=0.6)

ax.set_title(f"Temporal Hierarchy of Representation ({MODEL_NAME})", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Layer Depth ($l$)", fontsize=12)
ax.set_ylabel("Linear Separability / Recall Accuracy", fontsize=12)
ax.set_xlim(0, n_layers - 1)
ax.set_ylim(0.35, 1.05)
ax.legend(loc="lower right", frameon=True, fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
print(f"[Done] Temporal hierarchy plot successfully saved to {OUTPUT_PLOT}")