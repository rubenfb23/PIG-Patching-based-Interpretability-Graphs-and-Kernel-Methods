"""Quick standalone testbed for causal-graph metrics on synthetic patch effects.

This script is intentionally self-contained and lightweight:
- no transformers
- no torch
- only numpy + scipy
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def generate_mock_data(
    num_examples: int = 100,
    num_nodes: int = 3,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic patching-style effects and clean/corrupted scores.

    Returns:
        E: shape (num_examples, num_nodes)
        clean_scores: shape (num_examples,)
        corrupted_scores: shape (num_examples,)
    """
    if num_examples < 4:
        raise ValueError("num_examples must be >= 4")
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")

    rng = np.random.default_rng(seed)

    # Baseline gap per example: clean - corrupted.
    delta = rng.normal(loc=2.0, scale=0.45, size=num_examples)
    # Avoid near-zero denominator in normalization.
    tiny = np.abs(delta) < 0.15
    delta[tiny] = np.sign(delta[tiny]) * 0.15

    corrupted_scores = rng.normal(loc=-0.2, scale=0.8, size=num_examples)
    clean_scores = corrupted_scores + delta

    # Latent factors induce correlation structure between nodes.
    shared_factor = rng.normal(0.0, 1.0, size=num_examples)
    secondary_factor = rng.normal(0.0, 1.0, size=num_examples)

    load_shared = rng.uniform(0.35, 0.95, size=num_nodes)
    # Make first two nodes more alike for a realistic "co-influence" pattern.
    if num_nodes >= 2:
        load_shared[1] = load_shared[0] + rng.normal(0.0, 0.03)

    load_secondary = rng.normal(0.0, 0.55, size=num_nodes)

    E = np.empty((num_examples, num_nodes), dtype=np.float64)
    for j in range(num_nodes):
        E[:, j] = (
            delta * (0.20 + 0.38 * np.abs(load_shared[j]))
            + 0.55 * load_shared[j] * shared_factor
            + 0.30 * load_secondary[j] * secondary_factor
            + rng.normal(0.0, 0.22, size=num_examples)
        )

    return E, clean_scores, corrupted_scores


def load_minimal_real_data(
    cache_dir: str | Path = ".cache/patch_effects",
    corruption: str | None = None,
    component_size: int = 14,
    num_examples: int = 32,
    num_nodes: int = 3,
    prefer_positive_delta: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load a minimal MVP subset from cached patch-effect JSON files.

    Strategy:
    - filter one corruption type (or all) + one component width
    - keep only the first `num_examples` files (deterministic order)
    - prefer examples with clean > corrupted when available
    - flatten effects and keep `num_nodes` columns that are as de-correlated as possible
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_path}")

    records: list[dict] = []
    for path in sorted(cache_path.glob("*.json")):
        payload = json.loads(path.read_text())
        effects = np.array(payload["effects"], dtype=np.float64)
        slice_info = payload.get("prompt_pair", {}).get("slice", {})
        corr = slice_info.get("corruption")
        corr_ok = corruption is None or corr == corruption
        if corr_ok and effects.shape[2] == component_size:
            payload["_source_file"] = str(path)
            records.append(payload)

    if not records:
        raise ValueError(
            f"No cached records found for corruption={corruption!r}, "
            f"component_size={component_size}."
        )

    # Prefer clean>corrupted examples to keep restoration interpretation cleaner.
    selected_records = records
    if prefer_positive_delta:
        positives = [rec for rec in records if rec["clean_score"] > rec["base_score"]]
        negatives = [rec for rec in records if rec["clean_score"] <= rec["base_score"]]
        if len(positives) >= num_examples:
            selected_records = positives[:num_examples]
        else:
            selected_records = positives + negatives[: max(0, num_examples - len(positives))]

    if len(selected_records) < num_examples:
        raise ValueError(
            f"Not enough cached examples for corruption={corruption!r}, "
            f"component_size={component_size}. Found={len(selected_records)}, "
            f"required={num_examples}."
        )

    selected = selected_records[:num_examples]
    min_tokens = min(np.array(rec["effects"]).shape[1] for rec in selected)
    full_E = np.stack(
        [
            np.array(rec["effects"], dtype=np.float64)[:, :min_tokens, :].reshape(-1)
            for rec in selected
        ],
        axis=0,
    )

    if full_E.shape[1] < num_nodes:
        raise ValueError(
            f"Requested num_nodes={num_nodes} but only {full_E.shape[1]} features exist."
        )

    node_idx = select_less_correlated_nodes(full_E, num_nodes=num_nodes)
    E = full_E[:, node_idx]

    clean_scores = np.array([rec["clean_score"] for rec in selected], dtype=np.float64)
    corrupted_scores = np.array([rec["base_score"] for rec in selected], dtype=np.float64)
    pos_count = int(np.sum(clean_scores > corrupted_scores))
    meta = {
        "corruption_filter": corruption,
        "component_size": component_size,
        "num_candidates": len(records),
        "num_selected": len(selected),
        "num_positive_selected": pos_count,
        "num_negative_selected": int(len(selected) - pos_count),
        "min_tokens": int(min_tokens),
    }
    return E, clean_scores, corrupted_scores, node_idx, meta


def select_less_correlated_nodes(full_E: np.ndarray, num_nodes: int) -> np.ndarray:
    """Greedy node selection: high variance first, then low max-|corr| with selected."""
    if full_E.ndim != 2:
        raise ValueError("full_E must be 2D")
    n, m = full_E.shape
    if num_nodes > m:
        raise ValueError(f"num_nodes={num_nodes} exceeds available features={m}")

    variances = full_E.var(axis=0)
    order_var = np.argsort(variances)[::-1]
    # Restrict pool to informative features to avoid selecting near-constant columns.
    pool_size = min(m, max(200, 25 * num_nodes))
    candidate_pool = order_var[:pool_size]

    X = full_E - full_E.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1)
    valid = std > 1e-12
    if not np.any(valid):
        return order_var[:num_nodes]
    candidate_pool = np.array([idx for idx in candidate_pool if valid[idx]], dtype=int)
    if candidate_pool.size < num_nodes:
        fallback = np.array([idx for idx in order_var if valid[idx]], dtype=int)
        candidate_pool = fallback[: max(num_nodes, fallback.size)]

    Xn = np.zeros_like(X)
    Xn[:, valid] = X[:, valid] / std[valid]
    corr = (Xn.T @ Xn) / max(1, (n - 1))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    selected = [int(candidate_pool[0])]
    remaining = set(int(i) for i in candidate_pool if i != selected[0])
    while len(selected) < num_nodes and remaining:
        best = None
        best_key = None
        for idx in remaining:
            max_abs_corr = float(np.max(np.abs(corr[idx, selected])))
            key = (max_abs_corr, -float(variances[idx]))
            if best_key is None or key < best_key:
                best_key = key
                best = idx
        selected.append(int(best))
        remaining.remove(int(best))

    return np.array(selected, dtype=int)


def restoration_normalization(
    E: np.ndarray,
    clean_scores: np.ndarray,
    corrupted_scores: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Task 1: Normalize patch effects by baseline gap magnitude.

    Delta_i = clean_i - corrupted_i
    E_hat_iu = E_iu / (|Delta_i| + epsilon)
    R_u = mean_i(E_hat_iu)
    """
    if E.ndim != 2:
        raise ValueError("E must be a 2D array")

    num_examples = E.shape[0]
    if clean_scores.shape != (num_examples,) or corrupted_scores.shape != (num_examples,):
        raise ValueError("clean_scores and corrupted_scores must match E.shape[0]")

    delta = clean_scores - corrupted_scores
    denom = np.abs(delta) + epsilon
    E_hat = E / denom[:, None]
    R = E_hat.mean(axis=0)
    return E_hat, R, delta


def build_similarity_graph(
    E_hat: np.ndarray,
    min_weight: float = 0.10,
    top_k: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Task 2: Build Pearson-correlation adjacency + sparsification.

    Filters:
    1) |weight| < min_weight -> 0
    2) Per row (node), keep top_k outgoing edges by |weight|
    """
    if E_hat.ndim != 2:
        raise ValueError("E_hat must be a 2D array")

    num_nodes = E_hat.shape[1]
    if num_nodes < 2:
        raise ValueError("Need at least 2 nodes to build edges")

    # Robust Pearson-like correlation that handles constant columns as zero-corr.
    X = E_hat - E_hat.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1)
    valid = std > 1e-12
    Xn = np.zeros_like(X)
    Xn[:, valid] = X[:, valid] / std[valid]
    denom = max(1, X.shape[0] - 1)
    corr = (Xn.T @ Xn) / denom
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)

    sparse = corr.copy()
    sparse[np.abs(sparse) < min_weight] = 0.0

    if top_k is not None and top_k > 0:
        k = min(top_k, num_nodes - 1)
        pruned = np.zeros_like(sparse)

        all_idx = np.arange(num_nodes)
        for i in range(num_nodes):
            candidates = all_idx != i
            idx = all_idx[candidates]
            row = sparse[i, idx]

            nonzero_mask = np.abs(row) > 0
            idx_nz = idx[nonzero_mask]
            val_nz = row[nonzero_mask]

            if idx_nz.size == 0:
                continue

            if idx_nz.size <= k:
                keep_idx = idx_nz
            else:
                top_local = np.argpartition(np.abs(val_nz), -k)[-k:]
                keep_idx = idx_nz[top_local]

            pruned[i, keep_idx] = sparse[i, keep_idx]

        sparse = pruned

    return sparse, corr


def split_half_reproducibility(
    E_hat: np.ndarray,
    min_weight: float = 0.10,
    top_k: int = 2,
    seed: int = 0,
) -> tuple[float, float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Task 3: Split-half graph stability via Spearman correlation.

    Steps:
    1) Random split of examples into halves A/B
    2) Build G_A and G_B with Task 2
    3) Spearman correlation between flattened off-diagonal edge weights
    """
    n = E_hat.shape[0]
    if n < 6:
        raise ValueError("Need at least 6 examples for split-half reproducibility")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half = n // 2
    idx_a, idx_b = perm[:half], perm[half:]

    G_A_sparse, G_A_dense = build_similarity_graph(
        E_hat[idx_a],
        min_weight=min_weight,
        top_k=top_k,
    )
    G_B_sparse, G_B_dense = build_similarity_graph(
        E_hat[idx_b],
        min_weight=min_weight,
        top_k=top_k,
    )

    # Stability uses dense edges to avoid rank artifacts from thresholding/top-k ties.
    off_diag = ~np.eye(G_A_dense.shape[0], dtype=bool)
    wA = G_A_dense[off_diag]
    wB = G_B_dense[off_diag]
    num_edges = int(wA.size)

    if np.allclose(wA, wA[0]) or np.allclose(wB, wB[0]):
        rho, pvalue = 0.0, 1.0
    else:
        rho, pvalue = spearmanr(wA, wB)
        if np.isnan(rho):
            rho = 0.0
        if np.isnan(pvalue):
            pvalue = 1.0

    return (
        float(rho),
        float(pvalue),
        num_edges,
        G_A_sparse,
        G_B_sparse,
        G_A_dense,
        G_B_dense,
    )


def simulate_joint_restoration(
    R_u: float,
    R_v: float,
    noise_std: float = 0.02,
    seed: int = 123,
) -> float:
    """Task 4 helper: synthetic R({u,v}) from individual restorations + small noise."""
    rng = np.random.default_rng(seed)
    return float(R_u + R_v + rng.normal(0.0, noise_std))


def mediation_metric(R_uv: float, R_v: float) -> float:
    """Task 4 metric: M(u -> v) = R({u,v}) - R({v})."""
    return float(R_uv - R_v)


def restoration_fraction(
    y_patch: np.ndarray,
    y_base: np.ndarray,
    y_clean: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Mean restoration fraction for output-level interventions."""
    return float(np.mean((y_patch - y_base) / (np.abs(y_clean - y_base) + epsilon)))


def run_controlled_causal_demo(
    num_examples: int = 256,
    seed: int = 0,
    epsilon: float = 1e-6,
) -> dict:
    """Simple controlled simulator to test Level A/B/C with real interventions.

    DAG:
      u -> v, u -> w, v -> w, (v,w) -> y
    """
    rng = np.random.default_rng(seed)

    # Structural coefficients (known ground truth).
    a_uv = 1.15
    a_uw = 0.30
    a_vw = 1.05
    b_vy = 0.80
    b_wy = 1.10

    z = rng.normal(0.0, 1.0, size=num_examples)
    e_u = rng.normal(0.0, 0.30, size=num_examples)
    e_v = rng.normal(0.0, 0.25, size=num_examples)
    e_w = rng.normal(0.0, 0.25, size=num_examples)
    e_y = rng.normal(0.0, 0.20, size=num_examples)
    corruption_shift = rng.normal(1.0, 0.20, size=num_examples)

    u_clean = z + e_u
    u_base = z - corruption_shift + e_u

    def forward(
        *,
        patch_u: bool = False,
        patch_v: bool = False,
        patch_w: bool = False,
        clamp_v_base: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u = u_clean if patch_u else u_base
        v = a_uv * u + e_v
        if patch_v:
            v = a_uv * u_clean + e_v
        if clamp_v_base:
            v = a_uv * u_base + e_v

        w = a_uw * u + a_vw * v + e_w
        if patch_w:
            w = a_uw * u_clean + a_vw * (a_uv * u_clean + e_v) + e_w

        y = b_vy * v + b_wy * w + e_y
        return u, v, w, y

    u0, v0, w0, y0 = forward()  # corrupted baseline
    uc, vc, wc, yc = forward(patch_u=True, patch_v=True, patch_w=True)  # clean baseline

    # Level A: influence matrix I(src -> tgt) from activation shifts.
    node_names = ["u", "v", "w"]
    base_acts = {"u": u0, "v": v0, "w": w0}
    clean_acts = {"u": uc, "v": vc, "w": wc}
    patched_src = {
        "u": dict(zip(node_names, forward(patch_u=True)[:3], strict=False)),
        "v": dict(zip(node_names, forward(patch_v=True)[:3], strict=False)),
        "w": dict(zip(node_names, forward(patch_w=True)[:3], strict=False)),
    }

    I = np.zeros((3, 3), dtype=np.float64)
    for i, src in enumerate(node_names):
        for j, tgt in enumerate(node_names):
            num = np.abs(patched_src[src][tgt] - base_acts[tgt])
            den = np.abs(clean_acts[tgt] - base_acts[tgt]) + epsilon
            I[i, j] = float(np.mean(num / den))

    # Level B: true interventional output restoration, not simulated by formula.
    _, _, _, y_u = forward(patch_u=True)
    _, _, _, y_v = forward(patch_v=True)
    _, _, _, y_uv = forward(patch_u=True, patch_v=True)
    R_u = restoration_fraction(y_u, y0, yc, epsilon=epsilon)
    R_v = restoration_fraction(y_v, y0, yc, epsilon=epsilon)
    R_uv = restoration_fraction(y_uv, y0, yc, epsilon=epsilon)
    M_uv = R_uv - R_v

    # Level C: edge ablation/path blocking for candidate u -> v.
    _, _, _, y_u_block_v = forward(patch_u=True, clamp_v_base=True)
    R_u_block_v = restoration_fraction(y_u_block_v, y0, yc, epsilon=epsilon)
    necessity_u_v = R_u - R_u_block_v
    retained_ratio = float(R_u_block_v / (R_u + epsilon))

    return {
        "node_names": node_names,
        "I": I,
        "R_u": float(R_u),
        "R_v": float(R_v),
        "R_uv": float(R_uv),
        "M_u_to_v": float(M_uv),
        "R_u_block_v": float(R_u_block_v),
        "necessity_u_to_v": float(necessity_u_v),
        "retained_ratio_when_block_v": retained_ratio,
        "coefficients": {
            "a_uv": a_uv,
            "a_uw": a_uw,
            "a_vw": a_vw,
            "b_vy": b_vy,
            "b_wy": b_wy,
        },
    }


def save_controlled_causal_demo(output_dir: str | Path, demo: dict) -> tuple[Path, Path]:
    """Persist controlled A/B/C demo outputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "controlled_causal_demo.npz"
    json_path = out / "controlled_causal_demo.json"

    np.savez_compressed(
        npz_path,
        I=np.array(demo["I"], dtype=np.float64),
        node_names=np.array(demo["node_names"], dtype=object),
    )
    demo_json = dict(demo)
    demo_json["I"] = np.array(demo["I"], dtype=float).tolist()
    json_path.write_text(json.dumps(demo_json, indent=2), encoding="utf-8")
    return npz_path, json_path


def interpret_mediation(M: float, near_zero: float = 0.05) -> str:
    """Human-readable interpretation for mediation score."""
    if abs(M) <= near_zero:
        return "M cerca de cero: v posiblemente media gran parte del efecto de u."
    if M > near_zero:
        return "M > 0 y significativo: u y v parecen actuar en paralelo (sinergia)."
    return "M < 0: posible interferencia o efecto supresor entre u y v."


def print_nonzero_edges(adj: np.ndarray, title: str) -> None:
    """Pretty-print non-zero directed edges."""
    print(title)
    num_nodes = adj.shape[0]
    any_edge = False
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and adj[i, j] != 0.0:
                any_edge = True
                print(f"  u{i} -> u{j}: {adj[i, j]:+.4f}")
    if not any_edge:
        print("  (sin aristas tras sparsification)")


def save_results(
    output_dir: str | Path,
    *,
    data_source: str,
    selection_meta: dict,
    selected_idx: np.ndarray,
    min_weight: float,
    top_k: int,
    E: np.ndarray,
    clean_scores: np.ndarray,
    corrupted_scores: np.ndarray,
    E_hat: np.ndarray,
    R: np.ndarray,
    delta: np.ndarray,
    adj_dense: np.ndarray,
    adj_sparse: np.ndarray,
    G_A_sparse: np.ndarray,
    G_B_sparse: np.ndarray,
    G_A_dense: np.ndarray,
    G_B_dense: np.ndarray,
    rho: float,
    pvalue: float,
    num_stability_edges: int,
    u: int,
    v: int,
    R_u: float,
    R_v: float,
    R_uv: float,
    M_uv: float,
    controlled_causal_summary: dict | None = None,
) -> tuple[Path, Path]:
    """Persist MVP artifacts to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    npz_path = out / "results.npz"
    np.savez_compressed(
        npz_path,
        E=E,
        clean_scores=clean_scores,
        corrupted_scores=corrupted_scores,
        E_hat=E_hat,
        R=R,
        delta=delta,
        adj_dense=adj_dense,
        adj_sparse=adj_sparse,
        G_A_sparse=G_A_sparse,
        G_B_sparse=G_B_sparse,
        G_A_dense=G_A_dense,
        G_B_dense=G_B_dense,
        selected_idx=selected_idx,
    )

    summary = {
        "data_source": data_source,
        "selection_meta": selection_meta,
        "num_examples": int(E.shape[0]),
        "num_nodes": int(E.shape[1]),
        "selected_idx": selected_idx.astype(int).tolist(),
        "min_weight": float(min_weight),
        "top_k": int(top_k),
        "delta_stats": {
            "min": float(delta.min()),
            "max": float(delta.max()),
            "mean": float(delta.mean()),
        },
        "R": R.astype(float).tolist(),
        "split_half": {
            "rho": float(rho),
            "pvalue": float(pvalue),
            "num_edges": int(num_stability_edges),
            "method": "spearman on dense off-diagonal edge weights",
        },
        "mediation": {
            "u": int(u),
            "v": int(v),
            "R_u": float(R_u),
            "R_v": float(R_v),
            "R_uv": float(R_uv),
            "M_uv": float(M_uv),
            "interpretation": interpret_mediation(M_uv),
        },
    }
    if controlled_causal_summary is not None:
        summary["controlled_causal_demo"] = controlled_causal_summary
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return npz_path, summary_path


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    # -------------------------------
    # 0) Minimal MVP data
    # -------------------------------
    num_examples = 20
    num_nodes = 3

    # Prefer tiny real subset from cached GPT-2 run.
    # Fallback to synthetic mock data if cache is unavailable.
    try:
        E, clean_scores, corrupted_scores, selected_idx, selection_meta = load_minimal_real_data(
            cache_dir=".cache/patch_effects",
            corruption=None,
            component_size=14,
            num_examples=num_examples,
            num_nodes=num_nodes,
            prefer_positive_delta=True,
        )
        data_source = "cache real (.cache/patch_effects, subset MVP)"
    except Exception:
        E, clean_scores, corrupted_scores = generate_mock_data(
            num_examples=num_examples,
            num_nodes=num_nodes,
            seed=7,
        )
        selected_idx = np.arange(num_nodes)
        selection_meta = {
            "corruption_filter": None,
            "component_size": None,
            "num_candidates": num_examples,
            "num_selected": num_examples,
            "num_positive_selected": int(np.sum(clean_scores > corrupted_scores)),
            "num_negative_selected": int(np.sum(clean_scores <= corrupted_scores)),
            "min_tokens": None,
        }
        data_source = "sintético (fallback)"

    print("=" * 80)
    print("Datos de entrada")
    print("=" * 80)
    print(f"Fuente: {data_source}")
    print(f"E shape: {E.shape}")
    print(f"Indices de nodos usados en E: {selected_idx.tolist()}")
    print(
        "Ejemplos seleccionados: "
        f"{selection_meta['num_selected']} "
        f"(clean>corrupted: {selection_meta['num_positive_selected']}, "
        f"clean<=corrupted: {selection_meta['num_negative_selected']})"
    )
    print(f"clean_scores shape: {clean_scores.shape}")
    print(f"corrupted_scores shape: {corrupted_scores.shape}")

    # -------------------------------
    # 1) Restoration normalization
    # -------------------------------
    E_hat, R, delta = restoration_normalization(E, clean_scores, corrupted_scores)

    print("\n" + "=" * 80)
    print("Tarea 1: Restoration normalization")
    print("=" * 80)
    print(f"E_hat shape: {E_hat.shape}")
    print(
        "Delta stats -> "
        f"min: {delta.min():+.4f}, max: {delta.max():+.4f}, mean: {delta.mean():+.4f}"
    )
    print("R_u (fracción de restauración media por nodo):")
    for u, val in enumerate(R):
        print(f"  R[u{u}] = {val:+.4f}")

    # -------------------------------
    # 2) Graph construction + sparsification
    # -------------------------------
    min_weight = 0.10
    top_k = 2
    adj_sparse, adj_dense = build_similarity_graph(
        E_hat,
        min_weight=min_weight,
        top_k=top_k,
    )

    print("\n" + "=" * 80)
    print("Tarea 2: Grafo de similitud + sparsification")
    print("=" * 80)
    print(f"Parámetros: min_weight={min_weight}, top_k={top_k}")
    print("Matriz de correlación densa (Pearson):")
    print(adj_dense)
    print("Matriz de adyacencia tras filtros:")
    print(adj_sparse)
    print_nonzero_edges(adj_sparse, "Aristas no nulas:")

    # -------------------------------
    # 3) Split-half reproducibility
    # -------------------------------
    rho, pvalue, n_edges, G_A_sparse, G_B_sparse, G_A_dense, G_B_dense = split_half_reproducibility(
        E_hat,
        min_weight=min_weight,
        top_k=top_k,
        seed=99,
    )

    print("\n" + "=" * 80)
    print("Tarea 3: Split-half reproducibility")
    print("=" * 80)
    print(f"Spearman rho entre pesos de aristas densas (G_A vs G_B): {rho:+.4f}")
    print(f"p-value: {pvalue:.4g}")
    print(f"num_edges usadas en estabilidad: {n_edges}")
    print_nonzero_edges(G_A_sparse, "Aristas en G_A (sparse):")
    print_nonzero_edges(G_B_sparse, "Aristas en G_B (sparse):")

    # -------------------------------
    # 4) Causal mediation (simulated)
    # -------------------------------
    u, v = 0, 1
    R_u = float(R[u])
    R_v = float(R[v])
    R_uv = simulate_joint_restoration(R_u, R_v, noise_std=0.02, seed=11)
    M_uv = mediation_metric(R_uv, R_v)

    print("\n" + "=" * 80)
    print("Tarea 4: Métrica de mediación causal (simulada)")
    print("=" * 80)
    print(f"Nodos elegidos: u={u}, v={v}")
    print(f"R({{u}}) = {R_u:+.4f}")
    print(f"R({{v}}) = {R_v:+.4f}")
    print(f"R({{u,v}}) simulado = {R_uv:+.4f}")
    print(f"M(u -> v) = R({{u,v}}) - R({{v}}) = {M_uv:+.4f}")
    print("Interpretación:")
    print(f"  {interpret_mediation(M_uv)}")

    # -------------------------------
    # 5) Controlled causal demo (A/B/C)
    # -------------------------------
    demo = run_controlled_causal_demo(num_examples=256, seed=123, epsilon=1e-6)
    demo_nodes = demo["node_names"]
    demo_I = np.array(demo["I"], dtype=np.float64)

    print("\n" + "=" * 80)
    print("Tarea 5: Niveles causales A/B/C (simulación controlada)")
    print("=" * 80)
    print("Level A - Influencia directa I(src -> tgt):")
    header = "        " + " ".join(f"{name:>10s}" for name in demo_nodes)
    print(header)
    for i, src in enumerate(demo_nodes):
        row = " ".join(f"{demo_I[i, j]:>+10.3f}" for j in range(len(demo_nodes)))
        print(f"  {src:>4s} {row}")

    print("\nLevel B - Mediación interventional real (output):")
    print(f"  R({{u}}) = {demo['R_u']:+.4f}")
    print(f"  R({{v}}) = {demo['R_v']:+.4f}")
    print(f"  R({{u,v}}) = {demo['R_uv']:+.4f}")
    print(f"  M(u->v) = R({{u,v}}) - R({{v}}) = {demo['M_u_to_v']:+.4f}")

    print("\nLevel C - Path blocking para edge u->v:")
    print(f"  R({{u}}) = {demo['R_u']:+.4f}")
    print(f"  R({{u, clamp(v=base)}}) = {demo['R_u_block_v']:+.4f}")
    print(f"  Necessity(u->v) = R({{u}}) - R({{u, clamp_v}}) = {demo['necessity_u_to_v']:+.4f}")
    print(f"  Retained ratio al bloquear v = {demo['retained_ratio_when_block_v']:+.4f}")

    controlled_summary = {
        "node_names": demo_nodes,
        "level_a_I": np.array(demo["I"], dtype=float).tolist(),
        "level_b": {
            "R_u": float(demo["R_u"]),
            "R_v": float(demo["R_v"]),
            "R_uv": float(demo["R_uv"]),
            "M_u_to_v": float(demo["M_u_to_v"]),
        },
        "level_c": {
            "R_u_block_v": float(demo["R_u_block_v"]),
            "necessity_u_to_v": float(demo["necessity_u_to_v"]),
            "retained_ratio_when_block_v": float(demo["retained_ratio_when_block_v"]),
        },
        "coefficients": demo["coefficients"],
    }
    demo_npz_path, demo_json_path = save_controlled_causal_demo("outputs/test_graphs_metrics", demo)

    # -------------------------------
    # 6) Persist MVP outputs
    # -------------------------------
    npz_path, summary_path = save_results(
        "outputs/test_graphs_metrics",
        data_source=data_source,
        selection_meta=selection_meta,
        selected_idx=selected_idx,
        min_weight=min_weight,
        top_k=top_k,
        E=E,
        clean_scores=clean_scores,
        corrupted_scores=corrupted_scores,
        E_hat=E_hat,
        R=R,
        delta=delta,
        adj_dense=adj_dense,
        adj_sparse=adj_sparse,
        G_A_sparse=G_A_sparse,
        G_B_sparse=G_B_sparse,
        G_A_dense=G_A_dense,
        G_B_dense=G_B_dense,
        rho=rho,
        pvalue=pvalue,
        num_stability_edges=n_edges,
        u=u,
        v=v,
        R_u=R_u,
        R_v=R_v,
        R_uv=R_uv,
        M_uv=M_uv,
        controlled_causal_summary=controlled_summary,
    )
    print("\n" + "=" * 80)
    print("Artefactos guardados")
    print("=" * 80)
    print(f"NPZ: {npz_path}")
    print(f"Resumen JSON: {summary_path}")
    print(f"Demo causal NPZ: {demo_npz_path}")
    print(f"Demo causal JSON: {demo_json_path}")


if __name__ == "__main__":
    main()
