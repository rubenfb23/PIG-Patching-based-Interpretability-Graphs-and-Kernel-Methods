"""Stage-by-stage numerical snapshot of the PIG pipeline.

Uses the in-repo ToyHookedModel (no downloads required).
All outputs are also saved to audit/stage_snapshot.txt.
"""

from __future__ import annotations

import sys
import io
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ── numpy print options ────────────────────────────────────────────────────────
np.set_printoptions(precision=3, suppress=True)

# ── tee output to file ─────────────────────────────────────────────────────────
OUTPUT_PATH = Path(__file__).parent / "stage_snapshot.txt"


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
    def flush(self):
        for f in self.files:
            f.flush()


_out_file = open(OUTPUT_PATH, "w")
_tee = Tee(sys.stdout, _out_file)
sys.stdout = _tee  # type: ignore[assignment]


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Prompts
# ═══════════════════════════════════════════════════════════════════════════════
section("STAGE 1 — Prompts  ( x_cln, x_crp, y_star, slice_label )")

from pig.prompts import create_ioi_dataset, SliceLabel

N_EXAMPLES = 4  # per slice

pairs_ns = create_ioi_dataset(N_EXAMPLES, corruption="name_swap", seed=42)
pairs_ab = create_ioi_dataset(N_EXAMPLES, corruption="abba",      seed=42)
all_pairs = pairs_ns + pairs_ab

slice_ns = pairs_ns[0].slice_label
slice_ab = pairs_ab[0].slice_label

print(f"\nSlice labels used:  {slice_ns}  |  {slice_ab}")
print(f"Total prompt pairs: {len(all_pairs)} ({N_EXAMPLES} per slice)\n")

for label, pairs in [("name_swap", pairs_ns[:2]), ("abba", pairs_ab[:2])]:
    print(f"--- slice: ioi:{label} (first 2 examples) ---")
    for i, p in enumerate(pairs):
        print(f"  [{i}] x_cln  = {repr(p.x_cln)}")
        print(f"       x_crp  = {repr(p.x_crp)}")
        print(f"       y_star = {repr(p.y_star)}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Baseline observable  O_i^base = O(f_θ(x_crp))
# ═══════════════════════════════════════════════════════════════════════════════
section("STAGE 2 — Baseline observable  O_i^base = O( f_θ(x_crp) )")

from pig.model import create_model
from pig.prompts import get_prompt_pair_distractor

model = create_model("toy_transformer", device="cpu")
print(f"\nModel: {model.model_name}  n_layers={model.n_layers}  "
      f"n_heads={model.n_heads}  d_model={model.d_model}")
print(f"Observable: logit margin  target - distractor  (y_distractor from meta)")

print("\nBaseline scores (base_score = O(x_crp)) — first 4 per slice:")
for label, pairs in [("name_swap", pairs_ns), ("abba", pairs_ab)]:
    scores = []
    for p in pairs:
        dist = get_prompt_pair_distractor(p)
        score = model.score(p.x_crp, p.y_star, distractor_token=dist)
        scores.append(score)
    arr = np.array(scores)
    print(f"  ioi:{label}  base_scores = {arr}")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 & 4 — Patched observable + Patch-effect tensor E_u^(i)
# ═══════════════════════════════════════════════════════════════════════════════
section("STAGE 3 & 4 — Patched observable O_{i,u}^patch  and  E_u^(i) = O_patch - O_base")

from pig.patching import compute_patch_effects

print("\nComputing patch effects (residual stream nodes only) ...")
dataset = compute_patch_effects(
    model,
    all_pairs,
    cache_dir=None,
    show_progress=True,
    node_types=["res"],
)

print(f"\nPatchEffectDataset: {len(dataset)} tensors")

# Show shape of one tensor
t0 = dataset[0]
print(f"\nFirst tensor shape (effects): {t0.effects.shape}  "
      f"= [n_layers={t0.num_layers}, n_tokens={t0.num_tokens}, "
      f"n_components={t0.num_components}]")
print(f"  base_score  = {t0.base_score:.4f}")
print(f"  clean_score = {t0.clean_score:.4f}")
print(f"  token_labels = {t0.token_labels}")

# Show flat [N_s, |V|] patch-effect matrix for the name_swap slice (first 4 rows)
print(f"\nFlat effect matrix for slice '{slice_ns}'  shape [N_s, |V|]:")
E_ns = dataset.get_effect_matrix(slice_ns)
n_tokens_ns = dataset.get_common_dimensions(slice_ns)[1]
n_layers    = dataset.get_common_dimensions(slice_ns)[0]
n_comp      = dataset.get_common_dimensions(slice_ns)[2]
V = n_layers * n_tokens_ns * n_comp

print(f"  Shape: {E_ns.shape}  (N_s={E_ns.shape[0]}, |V|={E_ns.shape[1]})")
print(f"  Node ordering: (layer * n_tokens + token) * n_components + comp_idx")
print(f"  (With res-only: comp_idx=0 always, so index = layer*{n_tokens_ns} + token)")
print(f"\n  First 4 rows, first 12 columns (columns = node index 0..11):")

# Build node labels for first 12 columns
node_labels = []
for layer in range(n_layers):
    for tok in range(n_tokens_ns):
        node_labels.append(f"L{layer}T{tok}")
short_labels = node_labels[:12]
header = "  " + "  ".join(f"{lbl:>7}" for lbl in short_labels)
print(header)
for row_idx in range(min(4, E_ns.shape[0])):
    row_str = "  ".join(f"{v:7.3f}" for v in E_ns[row_idx, :12])
    print(f"  {row_str}")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Correlation (co-influence) matrix  C_s[u,v]
# ═══════════════════════════════════════════════════════════════════════════════
section("STAGE 5 — Co-influence correlation matrix  C_s[u,v] = corr_i(E_u^(i), E_v^(i))")

from pig.graph import GraphBuilder

builder = GraphBuilder(k=5, enforce_direction=True)

# Compute raw correlation matrix (before sparsification) for name_swap slice
corr_ns = builder._compute_correlation_matrix(E_ns)

print(f"\nCorrelation matrix for slice '{slice_ns}'  shape: {corr_ns.shape}")
print(f"(Signed Pearson correlation, denominator N={E_ns.shape[0]}, not N-1)")

# Print full matrix if small enough, else top-left block
max_show = min(corr_ns.shape[0], 12)
print(f"\n  First {max_show}x{max_show} block:")
# header
hdr = "          " + "".join(f" {node_labels[j]:>7}" for j in range(max_show))
print(hdr)
for i in range(max_show):
    row_str = "".join(f" {corr_ns[i,j]:7.3f}" for j in range(max_show))
    print(f"  {node_labels[i]:>7}  {row_str}")

print(f"\n  Matrix diagonal (should be 1.0): {np.diag(corr_ns)[:max_show]}")
print(f"  Range: [{corr_ns.min():.3f}, {corr_ns.max():.3f}]")
print(f"\nNote: Stage 5' (direct-influence / paired patching) — NOT implemented")
print(f"Note: Stage 5'' (partial correlation)               — NOT implemented")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Sparse graph (top-k per source with direction constraint)
# ═══════════════════════════════════════════════════════════════════════════════
section("STAGE 6 — Sparse graph  G_s = (V, E_s, w_s)  after top-k + direction constraint")

from pig.graph import build_graphs, create_graph_builder
from pig.embeddings import compute_wl_features_from_list

# Build per-slice canonical graphs (one graph per slice label)
builder_default = create_graph_builder("correlation_topk", k=5, enforce_direction=True)
g_ns = builder_default.build_from_slice(dataset, slice_ns)
g_ab = builder_default.build_from_slice(dataset, slice_ab)

# Also build per-example graphs for the classifier
per_ex_graphs = builder_default.build_per_example(dataset)
# per_ex_graphs is list of (PatchInfluenceGraph, SliceLabel)

print(f"\nSparsification rule: top-k per source node, k={builder_default.k}")
print(f"Direction constraint: enforce_direction={builder_default.enforce_direction}")
print(f"  Order rule: src < dst  iff  (layer, token, type_order) is lexicographically less")
print(f"  Type order: att=0, mlp=1, res=2")

print(f"\nCanonical graph  slice '{slice_ns}':")
print(f"  Nodes: {g_ns.num_nodes}  Edges: {g_ns.num_edges}")
print(f"  Edge list (src_idx, dst_idx, weight) — first 10:")
for e in g_ns.get_edge_list()[:10]:
    src_n = g_ns.nodes[e[0]]
    dst_n = g_ns.nodes[e[1]]
    print(f"    ({e[0]:3d} L{src_n.layer}T{src_n.token}.{src_n.node_type}, "
          f"{e[1]:3d} L{dst_n.layer}T{dst_n.token}.{dst_n.node_type}, "
          f"w={e[2]:+.4f})")

print(f"\nCanonical graph  slice '{slice_ab}':")
print(f"  Nodes: {g_ab.num_nodes}  Edges: {g_ab.num_edges}")
print(f"  Edge list — first 10:")
for e in g_ab.get_edge_list()[:10]:
    src_n = g_ab.nodes[e[0]]
    dst_n = g_ab.nodes[e[1]]
    print(f"    ({e[0]:3d} L{src_n.layer}T{src_n.token}.{src_n.node_type}, "
          f"{e[1]:3d} L{dst_n.layer}T{dst_n.token}.{dst_n.node_type}, "
          f"w={e[2]:+.4f})")

print(f"\nPer-example graphs: {len(per_ex_graphs)} total "
      f"({sum(1 for _, l in per_ex_graphs if l == slice_ns)} × {slice_ns}, "
      f"{sum(1 for _, l in per_ex_graphs if l == slice_ab)} × {slice_ab})")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — WL features
# ═══════════════════════════════════════════════════════════════════════════════
section("STAGE 7 — WL feature matrix  X[s, (h, l)] = |{v : l^(h)(v) = l}|")

WL_DEPTH = 3  # default in WLEncoder

# Use per-example graphs for a richer feature matrix
fm = compute_wl_features_from_list(per_ex_graphs, depth=WL_DEPTH)
X = fm.to_matrix()

non_zero_cols = np.where(X.any(axis=0))[0]
print(f"\nWL depth (iterations): {WL_DEPTH}")
print(f"Feature matrix X shape: {X.shape}  = [n_graphs={X.shape[0]}, n_features={X.shape[1]}]")
print(f"Non-zero columns: {len(non_zero_cols)} out of {X.shape[1]}")

# Show first 5 rows × up to 12 informative columns
info_cols = non_zero_cols[:12]
print(f"\nFirst 5 rows × {len(info_cols)} informative columns:")
# truncated vocab labels
vocab_short = [fm.vocabulary[c][:12] for c in info_cols]
print("  " + "  ".join(f"{v:>12}" for v in vocab_short))
for row_idx in range(min(5, X.shape[0])):
    lbl = str(fm.slice_labels[row_idx])
    row_str = "  ".join(f"{X[row_idx, c]:12.0f}" for c in info_cols)
    print(f"  [{lbl[:18]:>18}]  {row_str}")

print(f"\nSlice label ordering in X:")
for i, sl in enumerate(fm.slice_labels):
    print(f"  row {i}: {sl}")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — Kernel matrices
# ═══════════════════════════════════════════════════════════════════════════════
section("STAGE 8 — Kernel matrices  K[s,t]")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from pig.quantum import (
    compute_quantum_kernel_matrix,
    QuantumFeatureMap,
    QuantumFeatureReducer,
)

S = X.shape[0]

# ── 8a. Linear kernel (on StandardScaler-normalized X) ──────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
K_lin = (X_scaled @ X_scaled.T) / X.shape[1]  # normalize by d

print(f"\n[8a] Linear kernel  K_lin[s,t] = x̂_s · x̂_t  (shape {K_lin.shape})")
print(f"     (features standardized by StandardScaler before dot product)")
print(f"     gamma not applicable")
print(np.array2string(K_lin.astype(np.float32), precision=3, suppress_small=True))

# ── 8b. RBF kernel ────────────────────────────────────────────────────────────
# gamma="scale" in sklearn = 1 / (n_features * X.var())
gamma_scale = 1.0 / (X.shape[1] * X.var()) if X.var() > 0 else 1.0
K_rbf = rbf_kernel(X_scaled, gamma=gamma_scale)

print(f"\n[8b] RBF kernel  K_rbf[s,t] = exp(-gamma * ||x̂_s - x̂_t||²)  (shape {K_rbf.shape})")
print(f"     gamma (sklearn 'scale') = 1/(n_features * Var[X_scaled]) = {gamma_scale:.6f}")
print(np.array2string(K_rbf.astype(np.float32), precision=3, suppress_small=True))

# ── 8c. Quantum fidelity kernel ───────────────────────────────────────────────
N_QUBITS = 4
DEPTH    = 2

# Need n_features >= n_qubits for PCA reducer
if X.shape[1] >= N_QUBITS:
    K_q, reducer = compute_quantum_kernel_matrix(
        fm,
        n_qubits=N_QUBITS,
        depth=DEPTH,
        shots=None,   # exact statevector simulation
        reduction="pca",
        random_state=42,
    )
    print(f"\n[8c] Quantum fidelity kernel  K_q[s,t] = |<0|U(z_s)†U(z_t)|0>|²  "
          f"(shape {K_q.shape})")
    print(f"     n_qubits={N_QUBITS}  depth={DEPTH}  shots=None (exact)")
    print(f"     reduction: PCA to {N_QUBITS} components then QuantumFeatureMap")
    print(f"     Circuit: H gates → depth × (Rz(z_i), Rx(z_i), CZ chain)")
    print(np.array2string(K_q.astype(np.float32), precision=3, suppress_small=True))
else:
    print(f"\n[8c] Quantum kernel skipped: n_features={X.shape[1]} < n_qubits={N_QUBITS}")
    K_q = None

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — Classification / grouping
# ═══════════════════════════════════════════════════════════════════════════════
section("STAGE 9 — Classification  (SVM on precomputed kernel / cross-validation)")

from pig.kernels import ClassicalKernelClassifier
from pig.quantum import QuantumKernelClassifier

y_labels = fm.slice_labels

# ── 9a. Classical linear kernel SVM ──────────────────────────────────────────
clf_lin = ClassicalKernelClassifier(kernel="linear", normalize=True, random_state=42)
try:
    cv_lin = clf_lin.cross_validate(X, y_labels, cv=2)
    print(f"\n[9a] Linear kernel SVM CV (cv=2):")
    print(f"     accuracy_mean = {cv_lin['accuracy_mean']:.3f}  "
          f"± {cv_lin['accuracy_std']:.3f}  ({cv_lin['cv_folds']} folds)")
except Exception as e:
    print(f"\n[9a] Linear kernel SVM CV failed: {e}")

# ── 9b. RBF kernel SVM ───────────────────────────────────────────────────────
clf_rbf = ClassicalKernelClassifier(kernel="rbf", normalize=True, random_state=42)
try:
    cv_rbf = clf_rbf.cross_validate(X, y_labels, cv=2)
    print(f"\n[9b] RBF kernel SVM CV (cv=2):")
    print(f"     accuracy_mean = {cv_rbf['accuracy_mean']:.3f}  "
          f"± {cv_rbf['accuracy_std']:.3f}  ({cv_rbf['cv_folds']} folds)")
except Exception as e:
    print(f"\n[9b] RBF kernel SVM CV failed: {e}")

# ── 9c. Quantum kernel SVM ────────────────────────────────────────────────────
if K_q is not None:
    clf_q = QuantumKernelClassifier(random_state=42)
    try:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform([str(l) for l in y_labels])
        cv_q = clf_q.cross_validate(K_q, y_labels, cv=2)
        print(f"\n[9c] Quantum kernel SVM CV (cv=2):")
        print(f"     accuracy_mean = {cv_q['accuracy_mean']:.3f}  "
              f"± {cv_q['accuracy_std']:.3f}  ({cv_q['cv_folds']} folds)")
    except Exception as e:
        print(f"\n[9c] Quantum kernel SVM CV failed: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
section("PIPELINE SUMMARY")
print(f"""
Stage 1  | {len(all_pairs)} PromptPair objects in {len(set(str(p.slice_label) for p in all_pairs))} slices
Stage 2  | Baseline observable: logit margin O = logit(y*) - logit(y_distractor)
Stage 3  | Patched observable matrix:  shape [{N_EXAMPLES}, {model.get_num_layers()}, T, 1]  (res-only)
Stage 4  | Patch-effect tensor E:      shape [{N_EXAMPLES}, {V}] after flatten
Stage 5  | Correlation matrix C_s:     shape [{V}, {V}]  (signed Pearson, denom N)
Stage 5' | Direct-influence DI:        NOT IMPLEMENTED in graph builders
Stage 5''| Partial correlation PC:     NOT IMPLEMENTED
Stage 6  | Sparse graph G_s:           top-k={builder_default.k} per source, direction enforced
Stage 7  | WL feature matrix X:        shape {X.shape}  depth={WL_DEPTH}
Stage 8  | Linear kernel K_lin:        shape {K_lin.shape}
         | RBF kernel K_rbf:           shape {K_rbf.shape}  gamma={gamma_scale:.4f}
         | Quantum kernel K_q:         {'shape ' + str(K_q.shape) if K_q is not None else 'SKIPPED'}  n_qubits={N_QUBITS} depth={DEPTH}
Stage 9  | SVM cross-validation on per-example graphs
""")

# close file
sys.stdout = sys.__stdout__
_out_file.close()
print(f"Saved stage snapshot to: {OUTPUT_PATH}")
