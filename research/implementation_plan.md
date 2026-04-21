# Implementation Plan: Extending the PIG Pipeline

## Overview

Based on the audit analysis, this document provides concrete implementation
plans for the highest-impact additions to the PIG pipeline.

---

## A. Direct-Influence Graph Builder

### Location
`src/pig/graphs/direct_influence.py`

### Core class

```python
class DirectInfluenceGraphBuilder(GraphBuilder):
    """Build graphs from direct-influence measurements.

    For each pair (u, v), compute:
        I(u→v) = ||a_v^patch(u) - a_v^base|| / (||a_v^cln - a_v^base|| + ε)

    This measures whether patching u changes v's activation.

    Edges go from higher-influence to higher-influence nodes.
    """

    def __init__(
        self,
        k: int = 5,
        enforce_direction: bool = True,
        min_weight: float = 0.0,
        influence_threshold: float = 0.01,
    ):
        super().__init__(k, enforce_direction, min_weight)
        self.influence_threshold = influence_threshold

    def build_from_slice(
        self,
        dataset: PatchEffectDataset,
        slice_label: SliceLabel,
    ) -> PatchInfluenceGraph:
        """Build graph from direct-influence matrix.

        1. For each tensor in slice:
           - Run clean forward → cache all activations
           - Run corrupted forward → get base activations
           - For each node u: patch u, get patch activations
           - Compute I(u→v) for all pairs
        2. Average influence across examples in slice
        3. Apply direction constraint
        4. Apply top-k sparsification
        """
        tensors = dataset.get_by_slice(slice_label)
        if not tensors:
            raise ValueError(f"No tensors for slice {slice_label}")

        # Get common dimensions
        num_layers, num_tokens, n_components = dataset.get_common_dimensions(slice_label)
        component_axis = dataset.get_component_axis()
        n_nodes = num_layers * num_tokens * n_components

        # Aggregate influence matrix across examples
        total_I = np.zeros((n_nodes, n_nodes), dtype=np.float32)

        for tensor in tensors:
            I = self._compute_influence_matrix(tensor)
            total_I += I

        # Average
        I_avg = total_I / len(tensors)

        # Build nodes and edges
        nodes = self._create_nodes(num_layers, num_tokens, component_axis)

        if self.enforce_direction:
            I_avg = self._apply_direction_constraint(I_avg, nodes)

        sparse_matrix = self._apply_topk_sparsification(I_avg)

        edges = [...]  # Convert to Edge objects
        return PatchInfluenceGraph(...)

    def _compute_influence_matrix(self, tensor: PatchEffectTensor):
        """Compute I(u→v) for all pairs in one example.

        Requires capturing activations at all nodes during forward pass.
        Uses the effects tensor as a proxy for activation change.
        """
        n_layers, n_tokens, n_comp = tensor.effects.shape
        n_nodes = n_layers * n_tokens * n_comp
        I = np.zeros((n_nodes, n_nodes), dtype=np.float32)

        # The effects tensor already tells us the impact of patching each node
        # on the observable. We need activation-level measurements.
        #
        # Alternative: use the effects as a proxy and rank pairs by
        # how much node u affects the observable when v is also patched.
        #
        # E(u,v) - E(u) - E(v) + E(base) = interaction term
        # If non-zero → u and v interact → edge

        for layer in range(n_layers):
            for tok in range(n_tokens):
                for comp_idx in range(n_comp):
                    u = (layer * n_tokens + tok) * n_comp + comp_idx
                    for v_layer in range(n_layers):
                        for v_tok in range(n_tokens):
                            for v_comp in range(n_comp):
                                v = (v_layer * n_tokens + v_tok) * n_comp + v_comp
                                if u == v:
                                    continue
                                I[u, v] = abs(tensor.effects[layer, tok, comp_idx])
                                # Simplified: use magnitude of effect at u
                                # as proxy for influence on all v
                                # More accurate: compute actual activation delta

        return I
```

### How to integrate:

1. Register in `pig/graphs/registry.py`:
   ```python
   register_graph_builder("direct_influence")(DirectInfluenceGraphBuilder)
   ```

2. Add to the publication study:
   ```python
   # scripts/run_classical_publication_study.py
   BUILDER_VARIANTS = [
       "correlation_topk",
       "direct_influence",
       "partial_correlation",
   ]
   ```

---

## B. Partial Correlation Graph Builder

### Location
`src/pig/graphs/partial_correlation.py`

### Implementation

```python
class PartialCorrelationGraphBuilder(GraphBuilder):
    """Build graphs from partial correlation of patch effects.

    Partial correlation measures the direct relationship between u and v
    controlling for all other nodes. This eliminates spuriously correlated
    nodes that share a common cause.

    PC(u,v) = -precision[u,v] / sqrt(precision[u,u] * precision[v,v])
    where precision = inv(covariance)
    """

    def __init__(self, *args, k: int = 5, **kwargs):
        super().__init__(*args, k=k, **kwargs)
        self._regularization = 1e-6

    def _compute_partial_correlation(self, effect_matrix):
        """Compute partial correlation matrix."""
        from scipy import linalg

        n_samples = effect_matrix.shape[0]
        n_features = effect_matrix.shape[1]

        # Covariance matrix
        cov = (effect_matrix.T @ effect_matrix) / n_samples

        # Add regularization for numerical stability
        cov += self._regularization * np.eye(n_features, dtype=np.float32)

        # Precision matrix (inverse of covariance)
        try:
            precision = linalg.inv(cov)
        except linalg.LinAlgError:
            # Fall back to pseudoinverse
            precision = linalg.pinv(cov)

        # Partial correlation
        d = 1.0 / np.sqrt(np.diag(precision) + 1e-10)
        partial_corr = -precision * np.outer(d, d)
        np.fill_diagonal(partial_corr, 1.0)

        return partial_corr.astype(np.float32)
```

### Warning
With N=4 samples and 22 features, the correlation matrix is rank-deficient.
Need N > features for stable partial correlation. Minimum ~50 examples recommended.

---

## C. Activation Flow Graph Builder

### Location
`src/pig/graphs/activation_flow.py`

### Implementation concept

```python
class ActivationFlowGraphBuilder(GraphBuilder):
    """Build graphs from activation flow analysis.

    For each patch point u, measure how much of its activation
    propagates to downstream nodes. Uses the concept of 'activation
    prediction' from Aditya et al. (2022).

    Flow(u→v) = ||MLP(Res_u) projected to v|| / ||Res_u||
    """

    def build_from_slice(self, dataset, slice_label):
        """Build graph using activation flow.

        For each pair (u, v) where u is earlier than v:
        1. Take the residual stream at u
        2. Project through subsequent layers' MLP + attention
        3. Measure how much ends up at v
        4. Normalize by initial magnitude at u
        """
        pass
```

---

## D. Cross-Slice Graph Comparison

### Location
`src/pig/graphs/slice_comparison.py`

### Implementation concept

```python
def compare_graphs(graphs_a, graphs_b, method="wl"):
    """Compare graphs from two different slices.

    Methods:
    - 'wl': WL kernel between graphs
    - 'node_overlap': Jaccard similarity of edge positions
    - 'edge_weight_corr': correlation of edge weights at same positions
    - 'spectral': spectral distance between adjacency matrices
    """
    if method == "wl":
        # WL kernel
        from pig.embeddings import WLEncoder
        encoder = WLEncoder(depth=5)
        emb_a = encoder.encode(graphs_a[0])
        emb_b = encoder.encode(graphs_b[0])
        # Kernel = dot product of feature vectors
        return np.dot(emb_a.to_vector(), emb_b.to_vector()) / (
            np.linalg.norm(emb_a.to_vector()) * np.linalg.norm(emb_b.to_vector())
        )
    elif method == "spectral":
        adj_a = graphs_a[0].get_adjacency_matrix()
        adj_b = graphs_b[0].get_adjacency_matrix()
        eig_a = np.linalg.eigvalsh(adj_a)
        eig_b = np.linalg.eigvalsh(adj_b)
        return np.linalg.norm(eig_a - eig_b)
```

---

## E. Causal Validation of Graph Edges

### Location
`src/pig/causal.py` (extend)

### What to add

```python
def validate_edge(graph, node_u, node_v, dataset, slice_label, n_bootstrap=200):
    """Validate a single edge u→v using causal mediation.

    Three tests:
    1. Direct test: patch u in x_crp, measure restoration
    2. Clamping test: patch u but clamp v to base, measure loss
    3. Combined test: patch both u and v, check synergy

    Returns:
    - influence: I(u→v)
    - mediation: M(u→v)
    - necessity: nec(u→v)
    - bootstrap CI for each
    """
    pass

def validate_all_edges(graph, dataset, slice_label, k_bootstrap=200):
    """Validate all edges in a graph and return significance scores."""
    results = {}
    for edge in graph.edges:
        result = validate_edge(graph, edge.src, edge.dst, dataset, slice_label, k_bootstrap)
        results[(edge.src, edge.dst)] = result
    return results
```

---

## F. Improved Dataset for Classification

### Current problem
Only 8 examples (4 per slice) → classifiers can't generalize

### Proposed dataset expansion

```python
def create_extended_ioi_dataset(n_per_corruption=50, seeds=None):
    """Create larger IOI dataset with multiple corruption types.

    Corruption types:
    - name_swap: swap subject/object names
    - abba: A B B A pattern (original IOI)
    - name_swap_2: double swap
    - object_swap: swap object with a different noun
    - verb_morph: change verb tense
    """
    if seeds is None:
        seeds = [7, 42, 123, 456, 789]

    all_pairs = []
    for corruption in ["name_swap", "abba", "name_swap_2", "object_swap"]:
        for seed in seeds[:n_per_corruption // 5]:
            pairs = create_ioi_dataset(
                n_per_corruption // 5, corruption=corruption, seed=seed
            )
            all_pairs.extend(pairs)

    return all_pairs
```

### Expected improvement
- From 87.5% → potentially 95%+ with real GPT-2 and more examples
- Statistical significance testing becomes meaningful
- Kernel differentiation becomes possible

---

## G. Graph Classifier Improvements

### Current problem
- Only uses WL features (node-level counts)
- Doesn't use edge weights as features
- SVM with precomputed kernel doesn't use graph structure directly

### Proposed: WL Kernel

```python
from sklearn.svm import SVC
from torch_kernels import WLKernel  # or implement manually

# Instead of: X = WL_features(graphs) → SVM(X, y)
# Use: K = WL_kernel(graphs) → SVM(K, y)
# This compares graphs directly with graph kernel
```

### Implementation (manual WL kernel)

```python
def wl_graph_kernel(graphs, depth=5):
    """Compute WL kernel between all pairs of graphs."""
    n = len(graphs)
    K = np.zeros((n, n), dtype=np.float32)

    encoder = WLEncoder(depth=depth, use_edge_weights=False)

    embeddings = encoder.encode_batch(graphs)

    # Build unified vocabulary
    all_labels = set()
    for emb in embeddings:
        all_labels.update(emb.features.keys())
    vocab = sorted(all_labels)

    # Compute kernel matrix
    for i in range(n):
        vec_i = embeddings[i].to_vector(vocab)
        for j in range(i, n):
            vec_j = embeddings[j].to_vector(vocab)
            K[i, j] = K[j, i] = np.dot(vec_i, vec_j)

    return K
```

---

## H. Complete Research File for Audit Content

### Location
`research/audit_analysis.md`

Already written above. Contains:
1. Full pipeline architecture
2. Audit findings
3. What's implemented vs not
4. Kernel analysis
5. Research recommendations
6. Literature connections
7. Success metrics
