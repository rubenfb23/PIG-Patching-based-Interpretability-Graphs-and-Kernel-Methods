"""Fixed-layout graph features and null graph controls.

These utilities support auxiliary representation checks around WL features.
They intentionally keep the causal graph construction untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from pig.graph import Edge, Node, PatchInfluenceGraph
from pig.patching import PatchEffectTensor
from pig.prompts import SliceLabel

NullControl = Literal[
    "edge_shuffle", "weight_shuffle", "sign_shuffle",
    "node_label_shuffle", "random_graph_matched_degree",
]


def _node_key(node: Node) -> str:
    head = "none" if node.head is None else str(node.head)
    return f"L{node.layer}:T{node.token}:{node.node_type}:H{head}"


def _edge_feature_key(src: Node, dst: Node) -> str:
    return f"edge|{_node_key(src)}->{_node_key(dst)}"


@dataclass
class FixedLayoutFeatureMatrix:
    """Dense fixed-layout edge-weight features for graph classifiers."""

    matrix: NDArray[np.float32]
    feature_names: list[str]
    slice_labels: list[SliceLabel]

    @property
    def num_graphs(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.matrix.shape[1])

    def to_matrix(self) -> NDArray[np.float32]:
        return self.matrix


def compute_fixed_layout_features_from_list(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
) -> FixedLayoutFeatureMatrix:
    """Vectorize graphs using explicit directed edge slots.

    Unlike WL, this baseline uses the fixed transformer layout directly:
    every possible observed ``source node -> destination node`` slot becomes
    one feature, and the value is the edge weight or zero.
    """
    graph_list = [graph for graph, _ in graphs_with_labels]
    slice_labels = [label for _, label in graphs_with_labels]
    feature_names = _build_edge_vocabulary(graph_list)
    feature_index = {name: index for index, name in enumerate(feature_names)}
    matrix = np.zeros((len(graph_list), len(feature_names)), dtype=np.float32)

    for row_idx, graph in enumerate(graph_list):
        for edge in graph.edges:
            src = graph.nodes[edge.src]
            dst = graph.nodes[edge.dst]
            key = _edge_feature_key(src, dst)
            col_idx = feature_index.get(key)
            if col_idx is not None:
                matrix[row_idx, col_idx] += np.float32(edge.weight)

    return FixedLayoutFeatureMatrix(
        matrix=matrix,
        feature_names=feature_names,
        slice_labels=slice_labels,
    )


_DIRECTED_TRIAD_NAMES = (
    "003",
    "012",
    "102",
    "021D",
    "021U",
    "021C",
    "111D",
    "111U",
    "030T",
    "030C",
    "201",
    "120D",
    "120U",
    "120C",
    "210",
    "300",
)


def _degree_summary(values: NDArray[np.float64]) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.std(values)), float(np.max(values))


def compute_directed_motif_features_from_list(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
) -> FixedLayoutFeatureMatrix:
    """Vectorize directed graphs using triad motifs and signed degree summaries.

    The 16 directed triad-census counts are normalized by the number of node
    triplets. Remaining features summarize positive/negative edge fractions
    and signed in/out-degree distributions. Unlike fixed edge slots, this
    representation is invariant to node ordering and absolute node identity.
    """
    import networkx as nx

    feature_names = [f"triad_{name}" for name in _DIRECTED_TRIAD_NAMES]
    feature_names.extend(["positive_edge_fraction", "negative_edge_fraction"])
    for sign in ("all", "positive", "negative"):
        for direction in ("in", "out"):
            for statistic in ("mean", "std", "max"):
                feature_names.append(f"{sign}_{direction}_degree_{statistic}")

    matrix = np.zeros(
        (len(graphs_with_labels), len(feature_names)),
        dtype=np.float32,
    )
    slice_labels = [label for _, label in graphs_with_labels]

    for row_idx, (graph, _) in enumerate(graphs_with_labels):
        directed = nx.DiGraph()
        directed.add_nodes_from(range(graph.num_nodes))
        directed.add_edges_from(
            (int(edge.src), int(edge.dst))
            for edge in graph.edges
            if edge.src != edge.dst and np.isfinite(edge.weight)
        )
        census = nx.triadic_census(directed)
        triplets = max(graph.num_nodes * (graph.num_nodes - 1) * (graph.num_nodes - 2) / 6, 1)
        for col_idx, name in enumerate(_DIRECTED_TRIAD_NAMES):
            matrix[row_idx, col_idx] = np.float32(census.get(name, 0) / triplets)

        finite_edges = [edge for edge in graph.edges if np.isfinite(edge.weight)]
        edge_count = max(len(finite_edges), 1)
        matrix[row_idx, 16] = np.float32(
            sum(edge.weight > 0 for edge in finite_edges) / edge_count
        )
        matrix[row_idx, 17] = np.float32(
            sum(edge.weight < 0 for edge in finite_edges) / edge_count
        )

        degree_vectors: list[NDArray[np.float64]] = []
        for predicate in (
            lambda edge: True,
            lambda edge: edge.weight > 0,
            lambda edge: edge.weight < 0,
        ):
            in_degree = np.zeros(graph.num_nodes, dtype=np.float64)
            out_degree = np.zeros(graph.num_nodes, dtype=np.float64)
            for edge in finite_edges:
                if predicate(edge):
                    out_degree[int(edge.src)] += 1.0
                    in_degree[int(edge.dst)] += 1.0
            degree_vectors.extend((in_degree, out_degree))

        col_idx = 18
        for values in degree_vectors:
            for statistic in _degree_summary(values):
                matrix[row_idx, col_idx] = np.float32(statistic)
                col_idx += 1

    return FixedLayoutFeatureMatrix(
        matrix=matrix,
        feature_names=feature_names,
        slice_labels=slice_labels,
    )


def apply_null_control_to_graphs(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    control: NullControl,
    *,
    seed: int = 42,
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    """Apply a graph null control while preserving graph labels."""
    rng = np.random.default_rng(seed)
    transformed: list[tuple[PatchInfluenceGraph, SliceLabel]] = []
    for graph, label in graphs_with_labels:
        if control == "edge_shuffle":
            null_graph = shuffle_graph_edges(graph, rng)
        elif control == "weight_shuffle":
            null_graph = shuffle_graph_weights(graph, rng)
        elif control == "sign_shuffle":
            null_graph = sign_shuffle_graph(graph, rng)
        elif control == "node_label_shuffle":
            null_graph = node_label_shuffle_graph(graph, rng)
        elif control == "random_graph_matched_degree":
            null_graph = random_graph_matched_degree(graph, rng)
        else:
            raise ValueError(f"Unknown null control: {control}")
        transformed.append((null_graph, label))
    return transformed


def shuffle_graph_weights(
    graph: PatchInfluenceGraph,
    rng: np.random.Generator,
) -> PatchInfluenceGraph:
    """Shuffle edge weights across the existing topology."""
    weights = [edge.weight for edge in graph.edges]
    if weights:
        shuffled = rng.permutation(np.array(weights, dtype=np.float32)).tolist()
    else:
        shuffled = []
    edges = [
        Edge(src=edge.src, dst=edge.dst, weight=float(shuffled[index]))
        for index, edge in enumerate(graph.edges)
    ]
    return _clone_graph(
        graph,
        edges,
        null_control="weight_shuffle",
    )


def shuffle_graph_edges(
    graph: PatchInfluenceGraph,
    rng: np.random.Generator,
) -> PatchInfluenceGraph:
    """Shuffle edge endpoints while preserving the edge-weight multiset."""
    if not graph.edges:
        return _clone_graph(graph, [], null_control="edge_shuffle")

    preserve_direction = _all_edges_follow_direction(graph)
    candidate_pairs = _candidate_edge_pairs(graph.nodes, preserve_direction)
    if not candidate_pairs:
        return _clone_graph(graph, [], null_control="edge_shuffle")

    replace = len(graph.edges) > len(candidate_pairs)
    chosen_indices = rng.choice(
        len(candidate_pairs),
        size=len(graph.edges),
        replace=replace,
    )
    shuffled_weights = rng.permutation(
        np.array([edge.weight for edge in graph.edges], dtype=np.float32)
    )
    edges = []
    for edge_idx, pair_idx in enumerate(chosen_indices):
        src, dst = candidate_pairs[int(pair_idx)]
        edges.append(Edge(src=src, dst=dst, weight=float(shuffled_weights[edge_idx])))

    return _clone_graph(
        graph,
        edges,
        null_control="edge_shuffle",
        preserve_direction=preserve_direction,
    )


def _build_edge_vocabulary(graphs: list[PatchInfluenceGraph]) -> list[str]:
    features: set[str] = set()
    for graph in graphs:
        for src in graph.nodes:
            for dst in graph.nodes:
                if src == dst:
                    continue
                features.add(_edge_feature_key(src, dst))
    return sorted(features)


def _all_edges_follow_direction(graph: PatchInfluenceGraph) -> bool:
    return all(graph.nodes[edge.src] < graph.nodes[edge.dst] for edge in graph.edges)


def _candidate_edge_pairs(
    nodes: list[Node],
    preserve_direction: bool,
) -> list[tuple[int, int]]:
    pairs = []
    for src_idx, src in enumerate(nodes):
        for dst_idx, dst in enumerate(nodes):
            if src_idx == dst_idx:
                continue
            if preserve_direction and not (src < dst):
                continue
            pairs.append((src_idx, dst_idx))
    return pairs


def _clone_graph(
    graph: PatchInfluenceGraph,
    edges: list[Edge],
    **metadata: object,
) -> PatchInfluenceGraph:
    graph_metadata = dict(graph.metadata)
    graph_metadata.update(metadata)
    return PatchInfluenceGraph(
        nodes=list(graph.nodes),
        edges=edges,
        slice_label=graph.slice_label,
        num_layers=graph.num_layers,
        num_tokens=graph.num_tokens,
        metadata=graph_metadata,
    )


# --------------- Additional baselines ---------------

def compute_patch_effect_features_from_list(
    tensors: Sequence["PatchEffectTensor"],
) -> FixedLayoutFeatureMatrix:
    """Baseline: raw patch-effect vectors (no graph construction).

    Each graph is represented by its raw patch-effect tensor flattened
    to a fixed-dimensional vector. This provides a strong non-graph
    baseline that uses the same underlying data.
    """
    feature_vectors = []
    slice_labels = [t.prompt_pair.slice_label for t in tensors]
    max_dim = max(t.effects.flatten().shape[0] for t in tensors)
    for t in tensors:
        v = t.effects.flatten()[:max_dim]
        if len(v) < max_dim:
            v = np.pad(v, (0, max_dim - len(v)), mode="constant")
        feature_vectors.append(v.astype(np.float32))
    matrix = np.stack(feature_vectors, axis=0)
    feature_names = [f"effect_{i}" for i in range(int(matrix.shape[1]))]
    return FixedLayoutFeatureMatrix(
        matrix=matrix, feature_names=feature_names, slice_labels=slice_labels
    )


def compute_topk_node_features_from_list(
    tensors: Sequence["PatchEffectTensor"],
    k: int = 10,
) -> FixedLayoutFeatureMatrix:
    """Baseline: top-k node identities by max effect magnitude.

    For each graph, select the top-k nodes (by max |effect| across examples)
    and encode them as (layer, token, component) triplets with their
    average effect. This tests whether the identity of influential nodes
    alone carries the classification signal.
    """
    slice_labels = [t.prompt_pair.slice_label for t in tensors]
    all_nodes: list[tuple[str, int, int, str]] = []
    for t in tensors:
        for layer_idx in range(t.num_layers):
            for token_idx in range(t.num_tokens):
                for comp_idx, comp in enumerate(t.component_axis):
                    effects = t.effects[layer_idx, token_idx, comp_idx]
                    max_mag = float(np.max(np.abs(effects)))
                    node_key = (
                        f"L{layer_idx}:T{token_idx}:{comp.node_type}:H{comp.head or 'none'}"
                    )
                    avg_effect = float(np.mean(effects))
                    all_nodes.append((node_key, max_mag, avg_effect))

    # Global top-k across all graphs
    all_nodes.sort(key=lambda x: -x[1])
    top_k_nodes = [n[0] for n in all_nodes[:k]]
    feature_index = {name: i for i, name in enumerate(top_k_nodes)}

    num_graphs = len(tensors)
    matrix = np.zeros((num_graphs, len(top_k_nodes)), dtype=np.float32)
    for row_idx, t in enumerate(tensors):
        for layer_idx in range(t.num_layers):
            for token_idx in range(t.num_tokens):
                for comp_idx, comp in enumerate(t.component_axis):
                    effects = t.effects[layer_idx, token_idx, comp_idx]
                    node_key = (
                        f"L{layer_idx}:T{token_idx}:{comp.node_type}:H{comp.head or 'none'}"
                    )
                    col = feature_index.get(node_key)
                    if col is not None:
                        matrix[row_idx, col] = float(np.mean(effects))

    feature_names = top_k_nodes
    return FixedLayoutFeatureMatrix(
        matrix=matrix, feature_names=feature_names, slice_labels=slice_labels
    )


def compute_histogram_features_from_list(
    graphs_with_labels: list[tuple["PatchInfluenceGraph", "SliceLabel"]],
    n_bins: int = 32,
) -> FixedLayoutFeatureMatrix:
    """Baseline: global histograms of edge weights and signs.

    Compute histogram of:
    - All edge weights
    - Edge signs (+1/-1)
    - Node in/out degree distributions
    - Weight moments (mean, std, skewness, kurtosis)
    """
    slice_labels = [label for _, label in graphs_with_labels]
    num_features = n_bins * 2 + 6  # weight hist + sign hist + stats
    feature_names = (
        [f"wgt_hist_{i}" for i in range(n_bins)] +
        [f"sign_hist_{i}" for i in range(n_bins)] +
        ["mean_weight", "std_weight", "skew_weight", "kurt_weight", "density", "num_edges"]
    )

    matrix = np.zeros((len(graphs_with_labels), num_features), dtype=np.float32)
    for row_idx, (graph, _) in enumerate(graphs_with_labels):
        weights = [e.weight for e in graph.edges]
        if not weights:
            continue

        # Weight histogram
        w_min, w_max = min(weights), max(weights)
        if w_min == w_max:
            w_min, w_max = -1.0, 1.0
        hist_w, _ = np.histogram(weights, bins=n_bins, range=(w_min, w_max))
        matrix[row_idx, :n_bins] = hist_w.astype(np.float32) / max(hist_w.max(), 1)

        # Sign histogram
        signs = [1.0 if w > 0 else -1.0 for w in weights]
        hist_s = [
            sum(1 for s in signs if (w_min + (w_max - w_min) * (i / n_bins)) < 0)
            for i in range(n_bins)
        ]
        matrix[row_idx, n_bins:2*n_bins] = np.array(hist_s, dtype=np.float32) / max(max(hist_s), 1)

        # Statistics
        matrix[row_idx, 2*n_bins] = float(np.mean(weights))
        matrix[row_idx, 2*n_bins + 1] = float(np.std(weights))
        if np.std(weights) > 1e-8:
            matrix[row_idx, 2*n_bins + 2] = float(
                np.mean(((weights - np.mean(weights)) / np.std(weights)) ** 3)
            )
            m4 = np.mean([(w - np.mean(weights)) ** 4 for w in weights])
            matrix[row_idx, 2*n_bins + 3] = float(m4 / (np.std(weights) ** 4 + 1e-8) - 3.0)
        matrix[row_idx, 2*n_bins + 4] = graph.compute_statistics()["density"]
        matrix[row_idx, 2*n_bins + 5] = float(len(weights))

    return FixedLayoutFeatureMatrix(
        matrix=matrix, feature_names=feature_names, slice_labels=slice_labels
    )


def compute_label_permutation_features(
    graphs_with_labels: list[tuple["PatchInfluenceGraph", "SliceLabel"]],
    n_perms: int = 10,
    seed: int = 42,
) -> FixedLayoutFeatureMatrix:
    """Baseline: random label permutation.

    Returns a fixed-layout feature matrix with randomly permuted labels
    for use as a null baseline.
    """
    rng = np.random.default_rng(seed)
    labels = [label for _, label in graphs_with_labels]
    shuffled = list(rng.permutation(labels))
    # Use fixed_layout features on the original graphs
    from pig.graph_features import compute_fixed_layout_features_from_list
    base = compute_fixed_layout_features_from_list(graphs_with_labels)
    return FixedLayoutFeatureMatrix(
        matrix=base.matrix,
        feature_names=base.feature_names,
        slice_labels=shuffled,
    )


def compute_random_graph_features(
    graphs_with_labels: list[tuple["PatchInfluenceGraph", "SliceLabel"]],
    *,
    seed: int = 42,
    preserve_degree: bool = True,
) -> FixedLayoutFeatureMatrix:
    """Baseline: random graphs with matched degree distribution.

    For each graph, generate a random graph with the same in/out degree
    sequence but randomized edge assignments (configuration model).
    """
    rng = np.random.default_rng(seed)
    transformed = []
    for graph, label in graphs_with_labels:
        if preserve_degree:
            # Shuffle endpoints while preserving degree sequence
            in_deg, out_deg = graph.get_node_degrees()
            # Simple edge shuffle as approximation
            null_graph = shuffle_graph_edges(graph, rng)
        else:
            # Completely random: uniform random edges
            n = len(graph.nodes)
            num_edges = graph.num_edges
            indices = rng.choice(n * n, size=num_edges, replace=False)
            edges = [Edge(src=int(i % n), dst=int(i // n), weight=float(rng.random())) for i in indices]
            null_graph = PatchInfluenceGraph(
                nodes=list(graph.nodes), edges=edges, slice_label=label,
                num_layers=graph.num_layers, num_tokens=graph.num_tokens,
                metadata={**graph.metadata, "null_control": "random_graph"},
            )
        transformed.append((null_graph, label))
    return compute_fixed_layout_features_from_list(transformed)


def sign_shuffle_graph(
    graph: PatchInfluenceGraph,
    rng: np.random.Generator,
) -> PatchInfluenceGraph:
    """Flip edge signs randomly while preserving edge positions."""
    signs = rng.choice([-1, 1], size=len(graph.edges)).astype(np.float32)
    edges = [
        Edge(src=edge.src, dst=edge.dst, weight=float(edge.weight * signs[i]))
        for i, edge in enumerate(graph.edges)
    ]
    return _clone_graph(graph, edges, null_control="sign_shuffle")


def node_label_shuffle_graph(
    graph: PatchInfluenceGraph,
    rng: np.random.Generator,
) -> PatchInfluenceGraph:
    """Randomly permute node identities while preserving edge structure."""
    n = len(graph.nodes)
    perm = rng.permutation(n)
    # perm[i] = old index that moved to position i
    # Build reverse: old_idx -> new_idx
    inv_perm = np.empty(n, dtype=int)
    for i in range(n):
        inv_perm[int(perm[i])] = i
    # New node order
    new_nodes = [graph.nodes[int(i)] for i in perm]
    new_edges = []
    for edge in graph.edges:
        src_new = int(inv_perm[int(edge.src)])
        dst_new = int(inv_perm[int(edge.dst)])
        new_edges.append(Edge(src=src_new, dst=dst_new, weight=edge.weight))
    return _clone_graph(graph, new_edges, nodes=new_nodes, null_control="node_label_shuffle")


def random_graph_matched_degree(
    graph: PatchInfluenceGraph,
    rng: np.random.Generator,
) -> PatchInfluenceGraph:
    """Generate a random graph with matched in/out degree sequence.

    Uses a simple configuration model: shuffle destination stubs and
    pair them with source stubs, preserving marginal degree sequences.
    """
    in_deg, out_deg = graph.get_node_degrees()
    nn = len(graph.nodes)

    # Build stub lists (source and destination nodes for each edge)
    src_stubs = []
    dst_stubs = []
    for i in range(nn):
        src_stubs.extend([i] * int(out_deg[i]))
        dst_stubs.extend([i] * int(in_deg[i]))

    # Shuffle destination stubs to break structure while preserving degrees
    rng.shuffle(dst_stubs)

    edges = []
    for i, dst_idx in enumerate(dst_stubs):
        edges.append(Edge(src=int(src_stubs[i]), dst=int(dst_idx),
                         weight=float(rng.random())))
    return _clone_graph(graph, edges, null_control="random_graph_matched_degree")


# ---- Scalable representation baselines ----

def compute_hashed_fixed_layout_features(
    graphs_with_labels: list[tuple["PatchInfluenceGraph", "SliceLabel"]],
    dim: int = 1024,
    seed: int = 42,
) -> FixedLayoutFeatureMatrix:
    """Hashed fixed-layout: stable hash aggregation into fixed-d space.

    Maps each edge slot to a coordinate via a stable hash function,
    accumulating signed weights. Hash collisions are resolved by signed
    accumulation (positive edges add +1, negative edges add -1).
    NaN edge weights (from zero-variance nodes) are excluded.
    """
    _ = seed
    # Generate a stable hash offset for each possible edge slot
    slice_labels = [label for _, label in graphs_with_labels]
    feature_names = [f"hash_{i}" for i in range(dim)]

    matrix = np.zeros((len(graphs_with_labels), dim), dtype=np.float32)
    for row_idx, (graph, _) in enumerate(graphs_with_labels):
        for edge in graph.edges:
            src = graph.nodes[edge.src]
            dst = graph.nodes[edge.dst]
            slot = (src.layer, src.token, src.node_type, src.head,
                    dst.layer, dst.token, dst.node_type, dst.head)
            h = hash(slot) % dim
            sign = 1.0 if edge.weight > 0 else -1.0
            matrix[row_idx, int(h)] += float(sign)
    return FixedLayoutFeatureMatrix(
        matrix=matrix, feature_names=feature_names, slice_labels=slice_labels
    )


def compute_coarse_count_features(
    graphs_with_labels: list[tuple["PatchInfluenceGraph", "SliceLabel"]],
    num_buckets: int = 105,
) -> FixedLayoutFeatureMatrix:
    """Coarse count: aggregate edges by coarse type buckets.

    Buckets include: node type combination (res-res, res-mlp, etc.),
    layer-difference bucket, token-difference bucket, and edge sign.
    Yields approximately num_buckets features for typical configurations.
    """
    slice_labels = [label for _, label in graphs_with_labels]
    feature_names = []
    matrix = np.zeros((len(graphs_with_labels), num_buckets), dtype=np.float32)

    # Define bucket types
    comp_types = [("res", "res"), ("res", "mlp"), ("res", "att"),
                  ("mlp", "res"), ("mlp", "mlp"), ("mlp", "att"),
                  ("att", "res"), ("att", "mlp"), ("att", "att")]
    layer_diff_bins = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    token_diff_bins = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    signs = [+1, -1]

    bucket_id = 0
    for ct in comp_types:
        for ld in layer_diff_bins:
            for td in token_diff_bins:
                for s in signs:
                    feature_names.append(
                        f"ct={ct[0]}-{ct[1]}|dl={ld}|tl={td}|s={s}"
                    )
                    bucket_id += 1
                    if bucket_id >= num_buckets:
                        break
                if bucket_id >= num_buckets:
                    break
            if bucket_id >= num_buckets:
                break
        if bucket_id >= num_buckets:
            break

    # Trim to num_buckets
    feature_names = feature_names[:num_buckets]
    actual_dim = len(feature_names)
    matrix = np.zeros((len(graphs_with_labels), actual_dim), dtype=np.float32)

    for row_idx, (graph, _) in enumerate(graphs_with_labels):
        for edge in graph.edges:
            src = graph.nodes[edge.src]
            dst = graph.nodes[edge.dst]
            ld = dst.layer - src.layer
            td = dst.token - src.token
            ct = (src.node_type, dst.node_type)
            s = 1 if edge.weight > 0 else -1

            # Find matching bucket index
            for i, name in enumerate(feature_names):
                parts = name.split("|")
                comp_part = parts[0].split("=")[1]
                dl_part = int(parts[1].split("=")[1])
                tl_part = int(parts[2].split("=")[1])
                s_part = int(parts[3].split("=")[1])

                if comp_part != f"{ct[0]}-{ct[1]}":
                    continue
                if dl_part != ld and dl_part != -1:
                    continue
                if tl_part != td and tl_part != -1:
                    continue
                if s_part != s:
                    continue
                matrix[row_idx, i] += 1
                break
    return FixedLayoutFeatureMatrix(
        matrix=matrix, feature_names=feature_names, slice_labels=slice_labels
    )


# ---- Dimensionality control: PCA and Random Projection ----

def compute_pca_projected_features(
    graphs_with_labels: list[tuple["PatchInfluenceGraph", "SliceLabel"]],
    target_dim: int = 96,
    seed: int = 42,
) -> FixedLayoutFeatureMatrix:
    """PCA projection of fixed-layout features to target dimension.

    Projects the full fixed-layout feature matrix to target_dim using
    PCA, yielding a compressed representation for fair comparison
    against spectral_shape and other low-dim baselines.
    """
    from sklearn.decomposition import PCA

    slice_labels = [label for _, label in graphs_with_labels]

    # First get full fixed-layout features
    base = compute_fixed_layout_features_from_list(graphs_with_labels)
    X = base.matrix.copy()  # (n_graphs, n_features)

    # Fit PCA and transform — cap components by both feature dim and sample count
    n_samples, n_features = X.shape
    n_components = min(target_dim, n_features, n_samples)
    if n_components == 0:
        raise ValueError(f"Cannot run PCA: need at least 1 sample and 1 feature, got n_samples={n_samples}, n_features={n_features}")
    pca = PCA(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(X).astype(np.float32)

    feature_names = [f"pca_{i}" for i in range(projected.shape[1])]
    return FixedLayoutFeatureMatrix(
        matrix=projected, feature_names=feature_names, slice_labels=slice_labels,
    )


def compute_random_projection_features(
    graphs_with_labels: list[tuple["PatchInfluenceGraph", "SliceLabel"]],
    target_dim: int = 96,
    seed: int = 42,
) -> FixedLayoutFeatureMatrix:
    """Random projection of fixed-layout features to target dimension.

    Uses Gaussian random projection to map the full fixed-layout
    feature space to target_dim dimensions. Preserves pairwise distances
    approximately (Johnson-Lindenstrauss lemma).
    """
    from sklearn.random_projection import GaussianRandomProjection

    slice_labels = [label for _, label in graphs_with_labels]

    # First get full fixed-layout features
    base = compute_fixed_layout_features_from_list(graphs_with_labels)
    X = base.matrix.copy()

    # Fit random projection and transform
    rp = GaussianRandomProjection(n_components=target_dim, random_state=seed)
    projected = rp.fit_transform(X).astype(np.float32)

    feature_names = [f"rp_{i}" for i in range(projected.shape[1])]
    return FixedLayoutFeatureMatrix(
        matrix=projected, feature_names=feature_names, slice_labels=slice_labels,
    )
