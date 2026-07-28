#!/usr/bin/env python3
"""Full evaluation pipeline for PIG paper revision.

Runs all required studies on distilgpt2 with seeds {7, 42, 123}:
1. Scalability ablations (top-k + hash dim) + graph persistence
2. Null controls (label shuffle, node perm, sign flip, etc.)
3. Held-out generalization (baseline per seed)
4. Circuit overlap / localization sanity check

Uses example-disjoint train/test splits (80 train / 20 test per corruption).
"""
import json
import sys
from pathlib import Path

import hashlib
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pig.graph import Edge, Node, PatchInfluenceGraph, build_bootstrap_slice_graphs, create_graph_builder
from pig.graph_features import compute_fixed_layout_features_from_list
from pig.kernels import ClassicalKernelClassifier
from pig.model import create_model
from pig.patching import compute_patch_effects
from pig.prompts import SliceLabel, PromptPair, create_ioi_dataset

MODEL = "distilgpt2"
SEEDS = (7, 42, 123)
N_EXAMPLES = 100
N_BOOTSTRAP = 32
BASE_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "paper_revision_study"


# =================== Graph serialization ===================

def _graphs_to_dicts(graphs_with_labels):
    return [{"graph": {"nodes": [{"layer": n.layer, "token": n.token, "node_type": n.node_type} for n in g.nodes],
                       "edges": [{"src": e.src, "dst": e.dst, "weight": e.weight} for e in g.edges],
                       "metadata": g.metadata},
             "label": str(l)} for g, l in graphs_with_labels]


def _dicts_to_graphs(data_list):
    result = []
    for entry in data_list:
        g = entry["graph"]
        nodes = [Node(layer=n["layer"], token=n["token"], node_type=n.get("node_type", "res")) for n in g["nodes"]]
        edges = [Edge(src=e["src"], dst=e["dst"], weight=e["weight"]) for e in g["edges"]]
        graph = PatchInfluenceGraph(nodes=nodes, edges=edges, metadata=g.get("metadata", {}))
        parts = entry["label"].split(":")
        label = SliceLabel(task=parts[0], corruption=parts[1]) if len(parts) == 2 else SliceLabel(task="ioi", corruption=entry["label"])
        result.append((graph, label))
    return result


# =================== Feature helpers ===================

def _hash_features(matrix, dim=1024):
    n_graphs, n_features = matrix.shape
    hashed = np.zeros((n_graphs, dim), dtype=np.float64)
    for j in range(n_features):
        key = f"{j}:{dim}"
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % dim
        hashed[:, h] += matrix[:, j]
    return hashed


def _coarse_count(matrix, feature_names):
    """Aggregate fixed-layout features into coarse buckets."""
    buckets = {}
    for j, name in enumerate(feature_names):
        if "->" not in name:
            continue
        parts = name.split("->")
        src_parts = parts[0].split(":")
        dst_parts = parts[1].split(":")
        if len(src_parts) < 2 or len(dst_parts) < 2:
            continue
        try:
            src_layer = int(src_parts[0].replace("L", ""))
        except ValueError:
            continue
        dst_layer = int(dst_parts[0].replace("L", ""))
        src_token = int(src_parts[1].replace("T", ""))
        dst_token = int(dst_parts[1].replace("T", ""))
        node_type_key = f"{src_parts[2]}->{dst_parts[2]}"
        delta_layer = abs(dst_layer - src_layer)
        delta_token = abs(dst_token - src_token)
        layer_bucket = min(delta_layer // 2, 5)
        token_bucket = min(delta_token // 2, 5)
        sign_bucket = 1 if matrix[:, j].mean() > 0 else 0
        bucket_key = f"{node_type_key}_l{layer_bucket}_t{token_bucket}_s{sign_bucket}"
        buckets.setdefault(bucket_key, []).append(j)
    bucket_keys = sorted(buckets.keys())
    n_graphs = matrix.shape[0]
    n_buckets = len(bucket_keys)
    X = np.zeros((n_graphs, n_buckets), dtype=np.float64)
    for i in range(n_graphs):
        for bk, cols in buckets.items():
            for j in cols:
                X[i, bucket_keys.index(bk)] += matrix[i, j]
    return X, bucket_keys


def _get_labels(graphs_with_labels):
    label_map = {"name_swap": 0, "abba": 1}
    return np.array([label_map.get(str(l).split(":")[-1], 0) for _, l in graphs_with_labels], dtype=np.int32)


def _eval(X, y, Xt, yt):
    clf = ClassicalKernelClassifier(kernel="linear", random_state=42, normalize=True)
    clf.fit(X, y)
    preds = clf.predict(Xt)
    if isinstance(yt, np.ndarray):
        return float((preds == yt).mean())
    return float(sum(1 for p, t in zip(preds, yt) if p == int(t.corruption == "abba")) / len(yt))


def _prep_rep(X, rep, feat_names=None):
    """Return (X_rep, feature_names) for a representation."""
    if rep == "fixed_layout_signed":
        return np.sign(X), feat_names
    if rep == "fixed_layout_weighted":
        return X, feat_names
    if rep == "hashed_fixed_layout_signed":
        return _hash_features(X), None
    if rep == "coarse_count":
        if not feat_names or all("->" not in fn for fn in feat_names):
            # No directional features to coarse-grain; fall back to raw
            return X, feat_names
        return _coarse_count(X, feat_names)
    return X, feat_names


def evaluate_rep(X, y, Xt, yt, rep, feat_names=None):
    if X.shape[1] == 0 or Xt.shape[1] == 0:
        return float("nan")
    Xr, fn = _prep_rep(X, rep, feat_names)
    Xtr, _ = _prep_rep(Xt, rep, feat_names)
    return _eval(Xr, y, Xtr, yt)


# =================== Core graph building ===================

def _build_graphs_for_examples(model, pairs, k, seed, corruption="name_swap"):
    """Build bootstrap graphs from example pairs."""
    prompts = [PromptPair(x_cln=p[0], x_crp=p[1], y_star=p[2],
                          slice_label=SliceLabel(task="ioi", corruption=corruption))
               for p in pairs]
    pe = compute_patch_effects(model=model, prompt_pairs=prompts, node_types=["res"])
    gb = create_graph_builder("correlation_topk", k=k, enforce_direction=True)
    return build_bootstrap_slice_graphs(pe, gb, graphs_per_slice=N_BOOTSTRAP,
                                         sample_fraction=0.75, min_examples=3, seed=seed)


# =================== 1. Scalability Ablations ===================

def run_scalability_ablation(output_dir):
    print("\n" + "=" * 60)
    print("1. SCALABILITY ABLATIONS")
    print("=" * 60)

    model = create_model(MODEL)
    graphs_per_seed = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        ds_swap = create_ioi_dataset(n_examples=N_EXAMPLES, corruption='name_swap', seed=seed)
        ds_abba = create_ioi_dataset(n_examples=N_EXAMPLES, corruption='abba', seed=seed)
        # Tag pairs with corruption for proper separation after stratified split
        tagged_pairs = ([(p.x_cln, p.x_crp, p.y_star, "name_swap") for p in ds_swap] +
                        [(p.x_cln, p.x_crp, p.y_star, "abba") for p in ds_abba])
        targets = [p.y_star for p in ds_swap] + [p.y_star for p in ds_abba]
        pairs_train, pairs_test = train_test_split(tagged_pairs, train_size=80,
                                                     random_state=seed, stratify=targets)

        tr_swap = [p[:3] for p in pairs_train if p[3] == "name_swap"]
        tr_abba = [p[:3] for p in pairs_train if p[3] == "abba"]
        te_swap = [p[:3] for p in pairs_test if p[3] == "name_swap"]
        te_abba = [p[:3] for p in pairs_test if p[3] == "abba"]

        # Build graphs per-corruption so SliceLabels match the data
        g_tr_swap = _build_graphs_for_examples(model, tr_swap, k=5, seed=seed, corruption="name_swap") if tr_swap else []
        g_tr_abba = _build_graphs_for_examples(model, tr_abba, k=5, seed=seed, corruption="abba") if tr_abba else []
        g_te_swap = _build_graphs_for_examples(model, te_swap, k=5, seed=seed, corruption="name_swap") if te_swap else []
        g_te_abba = _build_graphs_for_examples(model, te_abba, k=5, seed=seed, corruption="abba") if te_abba else []

        graphs = g_tr_swap + g_tr_abba
        graphs_test = g_te_swap + g_te_abba
        graphs_per_seed[seed] = (graphs, graphs_test)

        # Compute features on combined set for consistent columns, then split
        combined_all = graphs + graphs_test
        cfeats_all = compute_fixed_layout_features_from_list(combined_all)
        n_all = len(graphs)
        cmat_all = cfeats_all.to_matrix()
        y = _get_labels(graphs)
        yt = _get_labels(graphs_test)
        fn = cfeats_all.feature_names

        print(f"  Train: {len(graphs)} graphs (name_swap={sum(1 for l in graphs if 'name_swap' in str(l[1]))}, "
              f"abba={sum(1 for l in graphs if 'abba' in str(l[1]))})")

        # Top-k ablation (also per-corruption)
        print("  Top-k:")
        for k in [1, 3, 5, 10]:
            tk_swap = _build_graphs_for_examples(model, tr_swap, k=k, seed=seed, corruption="name_swap") if tr_swap else []
            tk_abba = _build_graphs_for_examples(model, tr_abba, k=k, seed=seed, corruption="abba") if tr_abba else []
            tkt_swap = _build_graphs_for_examples(model, te_swap, k=k, seed=seed, corruption="name_swap") if te_swap else []
            tkt_abba = _build_graphs_for_examples(model, te_abba, k=k, seed=seed, corruption="abba") if te_abba else []
            tk = tk_swap + tk_abba
            tkt = tkt_swap + tkt_abba

            # Compute features on combined set for consistent columns, then split
            combined = tk + tkt
            cfeats = compute_fixed_layout_features_from_list(combined)
            n_tr = len(tk)
            cmat = cfeats.to_matrix()
            X_tr = cmat[:n_tr]
            X_te = cmat[n_tr:]
            fn_k = cfeats.feature_names

            accs = {}
            for rep in ["fixed_layout_signed", "hashed_fixed_layout_signed", "coarse_count"]:
                accs[rep] = evaluate_rep(X_tr, y, X_te, yt, rep, fn_k)
            print(f"    k={k}: fsig={accs['fixed_layout_signed']:.4f} "
                  f"hsig={accs['hashed_fixed_layout_signed']:.4f} "
                  f"coarse={accs['coarse_count']:.4f}")

        # Hash dimension ablation
        print("  Hash dim:")
        for d in [128, 256, 512, 1024]:
            Xh = _hash_features(cmat_all[:n_all], d)
            Xht = _hash_features(cmat_all[n_all:], d)
            acc = _eval(Xh, y, Xht, yt)
            print(f"    d={d}: {acc:.4f}")

    # Persist graphs for downstream scripts
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    for seed, (gr, gt) in graphs_per_seed.items():
        with open(graphs_dir / f"train_seed{seed}.json", "w") as f:
            json.dump(_graphs_to_dicts(gr), f)
        with open(graphs_dir / f"test_seed{seed}.json", "w") as f:
            json.dump(_graphs_to_dicts(gt), f)
    print(f"\n  Graphs saved to {graphs_dir}")
    return graphs_per_seed


# =================== 2. Null Controls ===================

def run_null_controls(graphs_per_seed, output_dir):
    print("\n" + "=" * 60)
    print("2. NULL CONTROLS")
    print("=" * 60)

    results = {}
    for seed in sorted(graphs_per_seed.keys()):
        gr, gt = graphs_per_seed[seed]
        all_graphs = gr + gt
        feats = compute_fixed_layout_features_from_list(all_graphs)
        X = feats.to_matrix()
        y = _get_labels(all_graphs)
        n = len(gr)
        Xtr, Xte = X[:n], X[n:]
        ytr, yte = y[:n], y[n:]
        fn = feats.feature_names

        print(f"\n  Seed {seed}:")
        for rep in ["fixed_layout_signed", "hashed_fixed_layout_signed"]:
            baseline = evaluate_rep(Xtr, ytr, Xte, yte, rep, fn)

            # Label shuffle
            rng = np.random.default_rng(42)
            lperm = rng.permutation(len(all_graphs))
            new_labels = [l for _, l in all_graphs]
            shuffled_labels = [new_labels[i] for i in lperm]
            new_feats = compute_fixed_layout_features_from_list(
                [(g, l) for (g, _), l in zip(all_graphs, shuffled_labels)])
            S = new_feats.to_matrix()
            s_shuf = evaluate_rep(S[:n], np.array(shuffled_labels[:n], dtype=np.int32),
                                  S[n:], np.array(shuffled_labels[n:], dtype=np.int32), rep, fn)

            # Sign flip
            rng2 = np.random.default_rng(42)
            flipped = []
            for g, l in all_graphs:
                ne = [Edge(e.src, e.dst, float(e.weight * int(rng2.choice([-1, 1])))) for e in g.edges]
                flipped.append((PatchInfluenceGraph(nodes=g.nodes, edges=ne, metadata={**g.metadata, "null_control": "sign_flip"}), l))
            sf = compute_fixed_layout_features_from_list(flipped)
            sf_acc = evaluate_rep(sf.to_matrix()[:n], ytr, sf.to_matrix()[n:], yte, rep, fn)

            # Node permutation
            rng3 = np.random.default_rng(42)
            permuted = []
            for g, l in all_graphs:
                nc = len(g.nodes)
                pm = rng3.permutation(nc)
                ne = [Edge(int(pm[e.src]), int(pm[e.dst]), e.weight) for e in g.edges]
                nn = [g.nodes[i] for i in pm]
                permuted.append((PatchInfluenceGraph(nodes=nn, edges=ne, metadata={**g.metadata, "null_control": "node_perm"}), l))
            pf = compute_fixed_layout_features_from_list(permuted)
            np_acc = evaluate_rep(pf.to_matrix()[:n], ytr, pf.to_matrix()[n:], yte, rep, fn)

            # Random patch effects
            rng4 = np.random.default_rng(42)
            rands = []
            for g, l in all_graphs:
                ne = [Edge(e.src, e.dst, float(rng4.standard_normal())) for e in g.edges]
                rands.append((PatchInfluenceGraph(nodes=g.nodes, edges=ne, metadata={**g.metadata, "null_control": "random_patch"}), l))
            rf = compute_fixed_layout_features_from_list(rands)
            rf_acc = evaluate_rep(rf.to_matrix()[:n], ytr, rf.to_matrix()[n:], yte, rep, fn)

            accs = {"baseline": baseline, "label_shuffle": s_shuf, "sign_flip": sf_acc,
                    "node_permutation": np_acc, "random_patch_effects": rf_acc}
            results[f"seed{seed}_{rep}"] = accs
            print(f"    {rep}: base={baseline:.4f} lshuf={s_shuf:.4f} sflip={sf_acc:.4f} np={np_acc:.4f} rand={rf_acc:.4f}")

    out_dir = output_dir / "null_controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# =================== 3. Held-out Generalization ===================

def run_held_out_generalization(graphs_per_seed, output_dir):
    print("\n" + "=" * 60)
    print("3. HELD-OUT GENERALIZATION")
    print("=" * 60)

    baseline_accs = {}
    for seed in sorted(graphs_per_seed.keys()):
        gr, gt = graphs_per_seed[seed]
        feats = compute_fixed_layout_features_from_list(gr)
        tfeats = compute_fixed_layout_features_from_list(gt)
        y, yt = _get_labels(gr), _get_labels(gt)
        fn = feats.feature_names
        acc = evaluate_rep(feats.to_matrix(), y, tfeats.to_matrix(), yt, "fixed_layout_signed", fn)
        baseline_accs[seed] = acc
        print(f"  Seed {seed}: {acc:.4f}")

    out_dir = output_dir / "held_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(baseline_accs, f, indent=2)
    return baseline_accs


# =================== 4. Circuit Overlap ===================

def run_circuit_overlap(graphs_per_seed, output_dir):
    print("\n" + "=" * 60)
    print("4. CIRCUIT OVERLAP / LOCALIZATION")
    print("=" * 60)

    results = {}
    ioi_layers = list(range(5, 11))

    all_graphs = []
    for seed in sorted(graphs_per_seed.keys()):
        gr, gt = graphs_per_seed[seed]
        all_graphs.extend(gr[:8])

    node_counts = {}
    for g, _ in all_graphs:
        for edge in g.edges:
            src = g.nodes[edge.src]
            dst = g.nodes[edge.dst]
            key = (src.layer, src.token, src.node_type)
            node_counts[key] = node_counts.get(key, 0) + 1

    sorted_nodes = sorted(node_counts.items(), key=lambda x: -x[1])[:50]
    print("\n  Top nodes (IOI circuit layers 5-10 marked *):")
    for (layer, token, ntype), freq in sorted_nodes[:15]:
        mark = " *" if layer in ioi_layers else ""
        print(f"    L{layer} T{token} {ntype}: freq={freq}{mark}")

    circuit_count = sum(1 for (l, t, nt) in node_counts if l in ioi_layers)
    total = len(node_counts)
    results["top_nodes"] = [{"layer": l, "token": t, "type": nt, "freq": f}
                            for (l, t, nt), f in sorted_nodes[:20]]
    results["circuit_fraction"] = f"{circuit_count}/{total}"
    print(f"\n  Circuit node fraction: {circuit_count}/{total}")

    out_dir = output_dir / "circuit_overlap"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# =================== Main ===================

def main():
    output_dir = BASE_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scalability ablation + graph generation
    graphs_per_seed = run_scalability_ablation(output_dir)

    # 2. Null controls (uses graphs from step 1)
    null_results = run_null_controls(graphs_per_seed, output_dir)

    # 3. Held-out generalization
    ho_results = run_held_out_generalization(graphs_per_seed, output_dir)

    # 4. Circuit overlap
    co_results = run_circuit_overlap(graphs_per_seed, output_dir)

    # Summary
    summary = {
        "n_graphs_per_seed": {str(k): {"train": len(gr), "test": len(gt)} for k, (gr, gt) in graphs_per_seed.items()},
        "null_controls": null_results,
        "held_out": ho_results,
        "circuit_overlap": co_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL STUDIES COMPLETE")
    print("=" * 60)
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
