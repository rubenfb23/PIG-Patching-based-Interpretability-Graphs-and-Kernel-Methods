# PIG Pipeline — Stage Mapping Report

> **Audit date:** 2026-04-21  
> **Repo commit:** `369462c`  
> **Model used for numerics:** `ToyHookedModel` (in-repo, 2 layers, 4 heads, d_model=64)  
> **Script:** `audit/run_stage_snapshot.py` → `audit/stage_snapshot.txt`

---

## 1. Repo Architecture

The package is installed from `src/` (PEP 517, `setuptools`). The main importable package is `pig`. A second package, `gpt2`, lives alongside it in `src/` and is unrelated to the PIG pipeline proper (it contains fine-tuning utilities for GSM8k distillation experiments).

```
PIG/
├── src/
│   ├── pig/                  ← main package
│   │   ├── __init__.py       ← version 0.1.0
│   │   ├── prompts.py        ← Stage 1: prompt-pair generators (IOI, name_swap, ABBA)
│   │   ├── model.py          ← Stage 2–4 (hooking): HookedModel wrapping GPT-2 style models
│   │   ├── patching.py       ← Stage 3–4: PatchEffectComputer, PatchEffectTensor/Dataset
│   │   ├── graph.py          ← Stage 5–6: GraphBuilder, PatchInfluenceGraph, Node, Edge
│   │   ├── graphs/           ← registered graph builder plugins
│   │   │   ├── registry.py           ← plugin registry + discovery
│   │   │   ├── base.py               ← GraphBuilderFactory protocol
│   │   │   ├── correlation_topk.py   ← default: signed-correlation + top-k
│   │   │   ├── abs_correlation_topk.py ← absolute-value variant
│   │   │   └── diff_correlation_topk.py ← fine-tune vs base diff-correlation
│   │   ├── embeddings.py     ← Stage 7: WLEncoder, WLFeatureMatrix, compute_wl_features
│   │   ├── kernels.py        ← Stage 8–9: ClassicalKernelClassifier (linear/RBF SVM)
│   │   ├── quantum.py        ← Stage 8–9: QuantumFeatureMap, QuantumKernelClassifier
│   │   ├── causal.py         ← interventional causal eval (Level A/B/C); NOT in main graph path
│   │   ├── toy_model/        ← self-contained tiny transformer (no downloads)
│   │   │   ├── config.py     ← TinyTransformerConfig (2L, 4H, d=64, vocab=512)
│   │   │   ├── model.py      ← ToyHookedModel (full patching API, synthetic observable)
│   │   │   ├── layers.py     ← TinyLayer, token hashing utilities
│   │   │   └── trainer.py    ← toy model training loop
│   │   ├── cli.py            ← `pig pipeline` entry point
│   │   ├── visualization.py  ← Plotly graph visualizations
│   │   └── web/              ← FastAPI/WebSocket service for interactive exploration
│   └── gpt2/                 ← unrelated: GPT-2 fine-tuning / distillation
├── notebooks/                ← 4 notebooks (see below)
├── scripts/                  ← validation + publication scripts
├── tests/                    ← pytest suite
├── audit/                    ← created by this audit
└── pyproject.toml
```

**Notebooks:**

| Notebook | Role |
|---|---|
| `pipeline_walkthrough.ipynb` | Synthetic pedagogical walkthrough of all 9 stages |
| `kernel_mechanism_walkthrough.ipynb` | Classical kernel experiments on WL matrix |
| `inverse_problem_walkthrough.ipynb` | Inverse problem / edge recovery experiments |
| `classical_publication_figures.ipynb` | Publication-quality figures for classical baselines |

**Tests:**

| Test file | Exercises |
|---|---|
| `test_prompts.py` | `IOIGenerator`, `NameSwapCorruption`, `ABBACorruption`, `PromptPair` |
| `test_patching.py` | `PatchEffectComputer`, `PatchEffectDataset`, `PatchEffectTensor` serialization |
| `test_graph.py` | `GraphBuilder`, `Node`, `Edge`, direction constraint, top-k |
| `test_embeddings.py` | `WLEncoder`, `WLFeatureMatrix`, `compute_wl_features` |
| `test_kernels.py` | `ClassicalKernelClassifier` fit/predict/CV |
| `test_quantum.py` | `QuantumFeatureMap`, `compute_fidelity_kernel_matrix`, `QuantumKernelClassifier` |
| `test_model.py` | `HookedModel` tokenize/score/patch hooks (requires model download; marked slow) |
| `test_causal.py` | `propose_causal_edge_candidates`, `evaluate_causal_edges` |
| `test_diff_correlation_graph.py` | `DiffCorrelationGraphBuilder` |
| `toy_model/test_toy_model.py` | `ToyHookedModel` patching API |
| `toy_model/test_training.py` | `train_toy_model` loop |

---

## 2. Per-Stage Mapping

### Stage 1 — Prompts

**Equation:** $(x_i^{\text{cln}},\, x_i^{\text{crp}},\, y_i^\star)$ per slice $s$ — $N_s$ triples indexed by $i \in \mathcal{I}_s$.

**Code location:** `src/pig/prompts.py:19–331`

**Primary types and functions:**

```python
@dataclass(frozen=True)
class SliceLabel:
    task: str           # e.g. "ioi"
    corruption: str     # e.g. "name_swap" or "abba"

@dataclass(frozen=True)
class PromptPair:
    x_cln: str
    x_crp: str
    y_star: str
    slice_label: SliceLabel
    meta: dict          # contains "y_distractor" for IOI

def create_ioi_dataset(
    n_examples: int,
    corruption: str = "name_swap",   # or "abba"
    seed: int = 42,
) -> list[PromptPair]: ...
```

**Docstring (`PromptPair`):**
> A clean/corrupted prompt pair for patching experiments.
> x_cln: The clean prompt where the model should predict y_star
> x_crp: The corrupted prompt where the model fails to predict y_star
> y_star: The target token that the clean prompt should predict

**What it does:** `IOIGenerator.generate()` (`prompts.py:243–284`) samples two distinct names from a 24-name pool, fills one of six IOI templates (`prompts.py:208–216`) to build `x_cln`, applies a `CorruptionStrategy` to produce `x_crp`, and sets `y_star = name_s` (the subject). The `meta` dict always includes `y_distractor = name_io`, enabling logit-margin observables downstream. Two strategies are registered: `NameSwapCorruption` (replaces $S$ with $IO$ everywhere in the clean prompt) and `ABBACorruption` (swaps the roles of $S$ and $IO$ using the template directly). Each strategy becomes a distinct `SliceLabel`.

**Divergences from theory:** None significant. The IOI template family is the only family implemented (no KV-retrieval, bracket matching, or arithmetic). The `meta["y_distractor"]` field makes the observable a logit margin rather than a raw logit (see Stage 2).

**Snapshot:**

```
Slice labels used:  ioi:name_swap  |  ioi:abba
Total prompt pairs: 8 (4 per slice)

--- slice: ioi:name_swap (first 2 examples) ---
  [0] x_cln  = 'Nicole gave the book to Bob. Bob gave it back to'
       x_crp  = 'Bob gave the book to Bob. Bob gave it back to'
       y_star = 'Nicole'
  [1] x_cln  = 'Andrew lent the money to Lisa. Lisa returned it to'
       x_crp  = 'Lisa lent the money to Lisa. Lisa returned it to'
       y_star = 'Andrew'

--- slice: ioi:abba (first 2 examples) ---
  [0] x_cln  = 'Nicole gave the book to Bob. Bob gave it back to'
       x_crp  = 'Bob gave the book to Nicole. Nicole gave it back to'
       y_star = 'Nicole'
```

---

### Stage 2 — Baseline observable

**Equation:** $O_i^{\text{base}} = O(f_\theta(x_i^{\text{crp}}))$

**Code location:** `src/pig/model.py:469–517`  
**Toy model variant:** `src/pig/toy_model/model.py:367–400`

```python
class HookedModel:
    def _observable_from_logits(
        self,
        logits: Tensor,
        target_token: str,
        distractor_token: Optional[str] = None,
    ) -> float: ...

    def score(
        self,
        prompt: str,
        target_token: str,
        distractor_token: Optional[str] = None,
    ) -> float: ...
```

**What it does:** `score()` tokenizes the prompt, runs a frozen forward pass (`model.eval()`, `torch.no_grad()`), and reads the last-position logit for `target_token`. When `distractor_token` is provided (as it is for all IOI pairs via `get_prompt_pair_distractor()` → `meta["y_distractor"]`), it returns the margin $\text{logit}(y^\star) - \text{logit}(y_{\text{distractor}})$ instead of the raw logit.

For the `ToyHookedModel`, `_observable_from_logits()` adds a `_presence_bonus()`: $+0.25 \times \text{(occurrences of target token in prompt)}$. This is a synthetic heuristic specific to the toy model that makes patch effects nonzero even for a randomly initialized network.

**Divergences from theory:** The observable is a **logit margin** (not a raw logit or log-probability or cross-entropy loss) when `y_distractor` is available. Log-probability and loss observables are not implemented. For non-IOI prompts where `meta["y_distractor"]` is absent, the observable falls back to the raw target logit.

**Snapshot:**

```
Model: toy_transformer  n_layers=2  n_heads=4  d_model=64
Observable: logit margin  target - distractor  (y_distractor from meta)

Baseline scores (base_score = O(x_crp)) — first 4 per slice:
  ioi:name_swap  base_scores = [0.231  1.450  0.123  0.760]
  ioi:abba       base_scores = [1.139  2.278  1.102  1.768]
```

---

### Stage 3 — Patched observable

**Equation:** $O_{i,u}^{\text{patch}} = O\!\left(\tilde{f}_\theta(x_i^{\text{crp}};\, u)\right)$

**Code location:** `src/pig/patching.py:624–707` (`PatchEffectComputer.compute_single`)  
**Hooking:** `src/pig/model.py:186–244` (capture hooks), `src/pig/model.py:219–244` (patch hooks)

```python
class PatchEffectComputer:
    def compute_single(self, prompt_pair: PromptPair) -> PatchEffectTensor:
        clean_cache = self.model.cache_clean_components(
            prompt_pair.x_cln, node_types=self.node_types
        )
        base_score = self.model.score(prompt_pair.x_crp, prompt_pair.y_star, ...)
        # for every (layer, token, component):
        patched_score = self.model.patched_score(
            prompt_pair.x_crp, prompt_pair.y_star, clean_cache, patch_node, ...
        )
        effects[layer, token, comp_idx] = patched_score - base_score
```

**What it does:** For each example $i$, first caches all activations from the clean prompt. Then, for every node $u = (\ell, t, k[, h])$ in the node set $\mathcal{V}$, runs the corrupted prompt forward with node $u$'s activation replaced by the value cached from the clean run. The result is the patched observable $O_{i,u}^{\text{patch}}$. Node types supported: `"res"` (residual stream, post-block), `"mlp"` (MLP output before residual add), `"att"` (per-head output before `c_proj` projection). For GPT-2 style models, hooks are registered on `transformer.h[l]` (residual), `transformer.h[l].mlp` (MLP), and `transformer.h[l].attn.c_proj` pre-hook (attention heads).

**Output:** `PatchEffectTensor.effects` of shape `[n_layers, n_tokens, n_components]` where `n_components` = number of heads (if `"att"`) + 1 (if `"mlp"`) + 1 (if `"res"`).

**Divergences from theory:** The patching is **single-node** at each call (one node at a time). Multi-node simultaneous patching (`patched_score_multi`) exists but is not used in `compute_single`; it is used only in the causal evaluation module (`causal.py`). The index ordering within `effects` is `[layer, token, component]`, which differs from the flat index $u$ in the equations; see Stage 4.

**Snapshot:**

```
First tensor shape (effects): (2, 12, 1) = [n_layers=2, n_tokens=12, n_components=1]
  base_score  = 0.2305
  clean_score = 0.6981
  token_labels = ['Nicole', 'gave', 'the', 'book', 'to', 'Bob', '.', 'Bob',
                  'gave', 'it', 'back', 'to']
```

---

### Stage 4 — Patch-effect tensor

**Equation:** $E_u^{(i)} = O_{i,u}^{\text{patch}} - O_i^{\text{base}}$

**Code location:** `src/pig/patching.py:680–683` (element computed inline in `compute_single`),  
`src/pig/patching.py:448–475` (`PatchEffectDataset.get_effect_matrix`)

```python
effects[layer, token, comp_idx] = patched_score - base_score   # patching.py:683

def get_effect_matrix(
    self, slice_label: SliceLabel, max_tokens: Optional[int] = None
) -> NDArray[np.float32]:
    # returns shape [num_examples, num_layers * max_tokens * num_components]
```

**What it does:** `effects[l, t, c]` stores $E_u^{(i)}$ for node $u = (\ell{=}l, t{=}t, \text{comp}{=}c)$. `get_effect_matrix()` stacks and flattens to `[N_s, |\mathcal{V}|]` with the flat index:
$$\text{flat\_idx}(l, t, c) = (l \cdot T_{\min} + t) \cdot C + c$$
where $T_{\min}$ is the minimum token count across all examples in the slice (to handle variable-length prompts) and $C$ is the number of components.

**Naming divergence from TFM notation:** The TFM uses $E_u^{(i)}$ as a scalar indexed by node $u$. In the code, this quantity lives as `tensor.effects[layer, token, comp_idx]` — a 3D array, not a flat vector. The flat view is obtained only via `get_effect_matrix()`.

**Snapshot:**

```
Flat effect matrix for slice 'ioi:name_swap'  shape [N_s, |V|]:
  Shape: (4, 22)   (N_s=4, |V|=22 = 2 layers × 11 tokens × 1 component)
  Node ordering: flat_idx = layer*11 + token   (res-only, comp_idx=0)

  First 4 rows, first 12 columns:
     L0T0     L0T1     L0T2     L0T3     L0T4     L0T5     L0T6  ...   L1T0
    0.173   -0.000    0.001    0.000    0.001    0.000    0.000  ...   0.350
    0.146   -0.004   -0.004   -0.002   -0.003   -0.003   -0.001  ...   0.350
    0.144   -0.002   -0.003   -0.001   -0.001   -0.001   -0.002  ...   0.350
    0.213   -0.000   -0.000   -0.000   -0.000    0.001   -0.000  ...   0.350
```

The L1T0 column is constant (0.350) across all examples because the toy model's `_single_patch_bonus` always adds the same value there; see Stage 2 note.

---

### Stage 5 — Correlation (co-influence)

**Equation:** $C_s[u,v] = \operatorname{corr}_{i \in \mathcal{I}_s}\!\left(E_u^{(i)},\, E_v^{(i)}\right)$

**Code location:** `src/pig/graph.py:239–265`

```python
class GraphBuilder:
    def _compute_correlation_matrix(
        self,
        effect_matrix: NDArray[np.float32],  # shape [N_s, |V|]
    ) -> NDArray[np.float32]:                # shape [|V|, |V|]
        centered = effect_matrix - effect_matrix.mean(axis=0, keepdims=True)
        std = np.std(effect_matrix, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)   # avoid div-by-zero
        normalized = centered / std
        n_samples = effect_matrix.shape[0]
        corr = np.dot(normalized.T, normalized) / n_samples
        return corr.astype(np.float32)
```

**What it does:** Computes the $|\mathcal{V}| \times |\mathcal{V}|$ Pearson correlation matrix of effect profiles across the $N_s$ examples in slice $s$. Columns with near-zero variance (std < 1e-8) are assigned std=1, so their correlation with all other columns is zero (constant columns contribute nothing).

**Three registered graph builders (Stage 5 variants):**

| Name | File | Correlation used |
|---|---|---|
| `correlation_topk` (**default**) | `graphs/correlation_topk.py:9–12` | signed Pearson |
| `abs_correlation_topk` | `graphs/abs_correlation_topk.py:12–26` | $\lvert\text{Pearson}\rvert$ |
| `diff_correlation_topk` | `graphs/diff_correlation_topk.py:18–134` | signed Pearson of (fine-tune $-$ base) effect differences |

**Divergences from theory:**
- Denominator is **N** (population correlation), not $N-1$ (sample correlation).
- Correlation is **signed** in the default builder; signs are preserved in edge weights. The `abs_correlation_topk` variant discards signs.
- Stage 5' (direct-influence via paired patching $E_{u,v}^{(i)} - E_u^{(i)}$): **NOT implemented** as a graph builder. `causal.py:884–1188` (`evaluate_causal_edges`) does compute paired-patching scores for individual candidate edges as part of a separate causal evaluation stack (Level B/C), but this is decoupled from the graph construction pipeline.
- Stage 5'' (partial correlation $-(\Sigma^{-1})_{uv}/\sqrt{(\Sigma^{-1})_{uu}(\Sigma^{-1})_{vv}}$): **NOT implemented** anywhere.

**Snapshot:**

```
Correlation matrix for slice 'ioi:name_swap'  shape: (22, 22)
(Signed Pearson, denominator N=4)

  First 12×12 block:
              L0T0    L0T1    L0T2  ...  L1T0
     L0T0     1.000   0.712   0.655  ...   0.000
     L0T1     0.712   1.000   0.867  ...   0.000
     ...
     L1T0     0.000   0.000   0.000  ...   0.000

  Diagonal: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]  ← L1T0 is constant → corr=0
  Range: [0.000, 1.000]
```

---

### Stage 5' — Direct influence (paired patching)

**Equation:** $w_s^{\text{DI}}(u \!\to\! v) = \mathbb{E}_{i \in \mathcal{I}_s}\!\left[E_{u,v}^{(i)} - E_u^{(i)}\right]$

**Status: NOT IMPLEMENTED** as a graph builder.

`causal.py:1023–1044` computes `score_uv` (patch both $u$ and $v$ simultaneously) and `score_u` (patch $u$ only) for individual candidate edges via `model.patched_score_multi()`. The mediation score `M = R_uv - R_v` (Level B, `causal.py:147–149`) is related but uses restoration fractions rather than raw effect differences, and operates only on pre-selected candidate edges rather than all $|\mathcal{V}|^2$ pairs. The natural place to add a DI graph builder would be `src/pig/graphs/` following the pattern in `diff_correlation_topk.py`.

---

### Stage 5'' — Partial correlation

**Status: NOT IMPLEMENTED.** No code computes $\Sigma_s^{-1}$.

---

### Stage 6 — Sparse graph

**Equation:** Top-$k$ per source with direction constraint $u \prec v$; output $G_s = (\mathcal{V}, \mathcal{E}_s, w_s)$.

**Code location:** `src/pig/graph.py:292–389`

```python
class GraphBuilder:
    def __init__(self, k: int = 5, enforce_direction: bool = True,
                 min_weight: float = 0.0): ...

    def _apply_direction_constraint(
        self,
        weights: NDArray[np.float32],  # [|V|, |V|]
        nodes: list[Node],
    ) -> NDArray[np.float32]:
        # zeros out weights[i,j] unless nodes[i] < nodes[j]

    def _apply_topk_sparsification(
        self,
        weights: NDArray[np.float32],  # [|V|, |V|]
    ) -> NDArray[np.float32]:
        # for each row i: keep top-k by |weight|, zero the rest

    def build_from_slice(
        self,
        dataset: PatchEffectDataset,
        slice_label: SliceLabel,
    ) -> PatchInfluenceGraph: ...
```

**Direction constraint order** (`src/pig/graph.py:44–57`):

```python
class Node:
    def __lt__(self, other: "Node") -> bool:
        if self.layer != other.layer:
            return self.layer < other.layer
        if self.token != other.token:
            return self.token < other.token
        type_order = {"att": 0, "mlp": 1, "res": 2}
        if self.node_type != other.node_type:
            return type_order[self.node_type] < type_order[other.node_type]
        # attention: lower head index first
        return (self.head or -1) < (other.head or -1)
```

So the canonical order within a `(layer, token)` cell is `att[0] < att[1] < ... < mlp < res`.

**What it does:** After computing the correlation matrix, the direction constraint zeroes out the upper triangle relative to the $\prec$ order, then top-$k$ sparsification keeps the $k$ largest-magnitude entries in each row. The remaining non-zero, non-diagonal entries (with `|weight| > min_weight`) become directed weighted edges. `build_from_slice` builds the **canonical** slice-level graph from all $N_s$ examples. `build_per_example` builds one graph per example using a pairwise effect-similarity heuristic (not correlation).

**Divergences from theory:**
- Only **top-$k$ per source** is implemented ($k=5$ default). Global top-$k$ and threshold-only modes are not implemented as standalone strategies. A `min_weight` parameter exists (default 0.0) but is rarely set in practice.
- The `build_per_example` method uses a **different weight formula** from `build_from_slice`: it uses $1/(1+|E_u^{(i)} - E_v^{(i)}|)$ (pairwise effect similarity) rather than cross-example correlation. This is documented as an "auxiliary per-example baseline."

**Snapshot:**

```
Sparsification rule: top-k per source, k=5
Direction constraint: enforce_direction=True
  Type order within (layer, token): att=0, mlp=1, res=2

Canonical graph  slice 'ioi:name_swap':  Nodes=22  Edges=45
  ( 0 L0T0.res,  1 L0T1.res,  w=+0.7123)
  ( 0 L0T0.res,  5 L0T5.res,  w=+0.7821)
  ...

Canonical graph  slice 'ioi:abba':  Nodes=22  Edges=45
  ( 0 L0T0.res,  1 L0T1.res,  w=+0.7430)
  ( 0 L0T0.res,  5 L0T5.res,  w=-0.9766)  ← negative sign allowed
  ...

Per-example graphs: 8 total (4 × name_swap, 4 × abba)
```

---

### Stage 7 — WL features

**Equation:** $x_s[(h,\ell)] = |\{v : \ell^{(h)}(v) = \ell\}|$ where  
$\ell^{(h+1)}(v) = \operatorname{hash}\!\left(\ell^{(h)}(v),\; \operatorname{sorted}\{\ell^{(h)}(u) : u \in \mathcal{N}(v)\}\right)$

**Code location:** `src/pig/embeddings.py:78–437`

```python
class WLEncoder:
    def __init__(self, depth: int = 3, use_edge_weights: bool = True): ...

    def _get_initial_labels(self, graph: PatchInfluenceGraph) -> list[str]:
        # label = f"L{node.layer}_T{node.token}_{node.node_type}"
        # (+ "_H{head}" if attention node)

    def _refine_labels(
        self, graph: PatchInfluenceGraph, current_labels: list[str]
    ) -> list[str]:
        # new_label = md5( f"{node_label}|{'_'.join(sorted(neighbor_labels))}" )[:8]
        # neighbor_label includes discretized weight: f"{label}:{int(w*10)}"

    def encode(self, graph: PatchInfluenceGraph) -> WLEmbedding:
        # returns Counter of {"d0_<label>": count, "d1_<label>": count, ...}

def compute_wl_features(
    graphs: dict[SliceLabel, PatchInfluenceGraph],
    depth: int = 3,
    cache_dir: Optional[str] = None,
) -> WLFeatureMatrix: ...

def compute_wl_features_from_list(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    depth: int = 3,
    cache_dir: Optional[str] = None,
) -> WLFeatureMatrix: ...
```

**What it does:** The WL encoder runs `depth` refinement iterations. At depth 0, initial labels encode `(layer, token, node_type[, head])`. At each iteration, a node's new label is the MD5 hash (first 8 hex chars) of the concatenation of its current label and the sorted list of neighbour labels. When `use_edge_weights=True` (default), neighbour labels are decorated with a discretized weight bin: `f"{label}:{int(weight * 10)}"` — so weights are binned to 0.1 precision. Features are raw **counts** (number of nodes with that label at that depth), not normalized. The vocabulary is built from the union of all observed labels, sorted lexicographically.

**Naming divergence:** The TFM notation uses $(h, \ell)$ for (iteration, label-value) pairs. In the code, feature keys are strings like `"d1_3f2a1b4c"` where `d1` is depth 1 and `3f2a1b4c` is the hash.

**Divergences from theory:**
- Iteration direction: only **outgoing** neighbours are used (`_get_neighbors` follows `edge.src == node_idx`). Incoming neighbours are not aggregated.
- `WLFeatureMatrix.to_matrix()` returns **raw counts** (integers). No L2 or L1 normalization is applied within the WL module itself. Normalization is deferred to `StandardScaler` in `ClassicalKernelClassifier`.
- Vocabulary construction is **transductive** (built from all graphs at once), which is fine for the per-slice setting but would require care in a proper train/test split.

**Snapshot:**

```
WL depth (iterations): 3
Feature matrix X shape: (8, 395) = [n_graphs=8, n_features=395]
Non-zero columns: 395 out of 395   (all columns active with this small dataset)

First 5 rows × 12 informative columns (depth-0 features = initial node labels):
  d0_L0_T0_res  d0_L0_T10_re  ...
  [ioi:name_swap]     1             1    ...   (all depth-0 = 1 since each node type appears once)
  [ioi:name_swap]     1             1    ...
  ...
  [ioi:abba]          1             1    ...

Row ordering: 4 × ioi:name_swap, 4 × ioi:abba
```

---

### Stage 8 — Kernel

**Equations:**
$$K_{st}^{\text{lin}} = \hat{x}_s^\top \hat{x}_t, \quad K_{st}^{\text{RBF}} = e^{-\gamma\|\hat{x}_s - \hat{x}_t\|^2}, \quad K_{st}^{(q)} = \left|\langle 0| U_\phi(z_s)^\dagger U_\phi(z_t) |0\rangle\right|^2$$

**Code locations:**  
Classical: `src/pig/kernels.py:80–383` (`ClassicalKernelClassifier`)  
Quantum: `src/pig/quantum.py:72–363` (`QuantumFeatureMap`, `compute_fidelity_kernel_matrix`, `QuantumKernelClassifier`, `compute_quantum_kernel_matrix`)

**Classical kernel (linear/RBF):**

```python
class ClassicalKernelClassifier:
    def __init__(
        self,
        kernel: Literal["linear", "rbf"] = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",   # sklearn default: 1/(n_features*Var[X])
        normalize: bool = True,
    ): ...
    # internally: StandardScaler() → SVC(kernel=..., C=..., gamma=...)
```

The kernel matrix is **not explicitly formed**; sklearn's SVC handles it internally. The `normalize=True` flag applies `StandardScaler` (z-score, not L2 normalization) before the SVM.

**Quantum kernel:**

```python
class QuantumFeatureMap:
    def __init__(self, n_qubits: int = 4, depth: int = 2,
                 feature_scale: float = 1.0): ...

    def state(self, features: NDArray[np.float32]) -> NDArray[np.complex128]:
        # Circuit: H on all qubits → depth × (Rz(z_i)·Rx(z_i) per qubit, CZ chain)

def compute_fidelity_kernel_matrix(
    Z: NDArray[np.float32],       # [n_graphs, n_qubits] after PCA reduction
    feature_map: QuantumFeatureMap,
    shots: Optional[int] = None,  # None = exact statevector
    seed: int = 42,
) -> NDArray[np.float32]:
    # K[i,j] = |<psi_i|psi_j>|^2  (exact),
    # or binomial shot noise estimate if shots > 0

def compute_quantum_kernel_matrix(
    feature_matrix: WLFeatureMatrix,
    n_qubits: int = 4,
    depth: int = 2,
    shots: Optional[int] = None,
    reduction: Literal["pca", "random"] = "pca",
    random_state: int = 42,
    feature_scale: float = 1.0,
) -> tuple[NDArray[np.float32], QuantumFeatureReducer]: ...
```

The quantum pipeline: (1) `WLFeatureMatrix → [S, d]`; (2) `StandardScaler + PCA` to `[S, n_qubits]` (reduces from $d$ WL dimensions to $n_{\text{qubits}}$ PCA components); (3) `QuantumFeatureMap.state()` per row to get a $2^{n_{\text{qubits}}}$-dimensional statevector; (4) fidelity $K_{st}^{(q)} = |\langle\psi_s|\psi_t\rangle|^2$.

**Divergence from theory:** RFF (Random Fourier Features, the matched-complexity baseline) is **NOT implemented**. The kernel matrix is not explicitly exposed at the `ClassicalKernelClassifier` level for the linear/RBF kernels; it must be computed externally (as done in the audit script) to get the $[S,S]$ matrix.

**Snapshot (8×8, toy model, 4 examples/slice):**

```
[8a] Linear kernel  K_lin = X̂ X̂ᵀ / d  (after StandardScaler)
[[ 0.810 -0.233  0.144  0.105 -0.199 -0.233 -0.196 -0.198]
 [-0.233  1.350 -0.220 -0.232 -0.237  0.041 -0.234 -0.235]
 ...diagonal values ≈ 0.8–1.4, off-diagonal ≈ −0.23 (within-class) or small]

[8b] RBF kernel  K_rbf = exp(−γ‖x̂_s−x̂_t‖²)   γ=0.013961 ('scale')
[[1.    0.    0.001 0.    0.    0.    0.    0.   ]
 ...≈ identity: WL features are very high-dimensional, pairs are distant]

[8c] Quantum kernel  K_q = |⟨ψ_s|ψ_t⟩|²   n_qubits=4  depth=2  exact
[[1.    0.104 0.246 0.100 0.077 0.013 0.105 0.090]
 [0.104 1.    0.032 0.010 0.003 0.482 0.001 0.002]
 ...rows 4–7 (abba class) show high mutual fidelity: K[4,6]=0.952, K[4,7]=0.990]
```

---

### Stage 9 — Grouping / classification

**Code location:** `src/pig/kernels.py:267–300` (`ClassicalKernelClassifier.cross_validate`),  
`src/pig/quantum.py:302–336` (`QuantumKernelClassifier.cross_validate`)

```python
class ClassicalKernelClassifier:
    def cross_validate(
        self,
        X: NDArray[np.float32],
        y: list[SliceLabel] | NDArray,
        cv: int = 5,
    ) -> dict:
        # StratifiedKFold(n_splits=min(cv, max_splits), shuffle=True, seed=42)
        # max_splits = min(min_class_count, n_samples)
        # returns {"accuracy_mean": ..., "accuracy_std": ..., "cv_folds": ...}

    def fit(self, X, y) -> "ClassicalKernelClassifier": ...
    def predict(self, X) -> NDArray[np.int32]: ...
    def evaluate(self, X, y) -> ClassificationResult:
        # returns accuracy + AUC (binary only)
```

**What it does:** `cross_validate()` uses `sklearn.model_selection.cross_val_score` with a `StratifiedKFold` splitter. The estimator is either a bare `SVC` (if `normalize=False`) or a `Pipeline([StandardScaler, SVC])` (default `normalize=True`). The `_resolve_cv()` method (`kernels.py:135–153`) caps the number of folds at `min(cv, min_class_count)` to prevent failures with small datasets. No clustering is implemented.

**Snapshot:**

```
[9a] Linear kernel SVM CV (cv=2):  accuracy_mean=0.875 ± 0.125  (2 folds)
[9b] RBF kernel SVM CV (cv=2):     accuracy_mean=0.875 ± 0.125  (2 folds)
[9c] Quantum kernel SVM CV (cv=2): accuracy_mean=0.875 ± 0.125  (2 folds)
```

*(Note: toy model with N=4 per slice; accuracy is noisy but all three kernels agree.)*

---

## 3. Consolidated Stage Table

| Stage | Equation (LaTeX inline) | Code location | Sample output |
|---|---|---|---|
| 1 | $(x_i^{\text{cln}}, x_i^{\text{crp}}, y_i^\star, s)$ | `pig/prompts.py:29–284` `create_ioi_dataset` | 8 pairs, 2 slices (`ioi:name_swap`, `ioi:abba`) |
| 2 | $O_i^{\text{base}} = O(f_\theta(x_i^{\text{crp}}))$ | `pig/model.py:493–517` `score()` | `[0.231, 1.450, 0.123, 0.760]` (name_swap) |
| 3 | $O_{i,u}^{\text{patch}} = O(\tilde{f}_\theta(x_i^{\text{crp}}; u))$ | `pig/patching.py:669–683` `compute_single()` | shape `[2, 12, 1]` per example |
| 4 | $E_u^{(i)} = O_{i,u}^{\text{patch}} - O_i^{\text{base}}$ | `pig/patching.py:683` / `get_effect_matrix()` | shape `(4, 22)` after flatten (2L×11T×1C) |
| 5 | $C_s[u,v] = \operatorname{corr}_i(E_u^{(i)}, E_v^{(i)})$ | `pig/graph.py:239–265` `_compute_correlation_matrix()` | shape `(22, 22)`, range `[0.000, 1.000]` |
| 5' | $w_s^{\text{DI}}(u{\to}v) = \mathbb{E}[E_{u,v}^{(i)}-E_u^{(i)}]$ | **NOT IMPLEMENTED** (cf. `pig/causal.py:1023–1044`) | — |
| 5'' | $w_s^{\text{PC}} = -(\Sigma_s^{-1})_{uv}/\sqrt{(\Sigma_s^{-1})_{uu}(\Sigma_s^{-1})_{vv}}$ | **NOT IMPLEMENTED** | — |
| 6 | top-$k$ + direction $u \prec v$ → $G_s$ | `pig/graph.py:292–389` `build_from_slice()` | 22 nodes, 45 edges, `k=5` |
| 7 | $x_s[(h,\ell)] = \lvert\{v:\ell^{(h)}(v)=\ell\}\rvert$ | `pig/embeddings.py:78–387` `WLEncoder` | shape `(8, 395)`, depth=3 |
| 8 | $K_{st}^{\text{lin}} = \hat{x}_s^\top\hat{x}_t$ | `pig/kernels.py:80–350` `ClassicalKernelClassifier` | `(8,8)`, diag≈0.8–1.4 |
| 8 | $K_{st}^{\text{RBF}} = e^{-\gamma\|\hat{x}_s-\hat{x}_t\|^2}$ | `pig/kernels.py:80–350` `ClassicalKernelClassifier` | `(8,8)`, $\gamma=0.0140$ |
| 8 | $K_{st}^{(q)} = \lvert\langle 0\rvert U^\dagger(z_s)U(z_t)\lvert 0\rangle\rvert^2$ | `pig/quantum.py:172–206` `compute_fidelity_kernel_matrix()` | `(8,8)`, $n_q=4$, $D=2$ |
| 9 | SVM + `StratifiedKFold` CV | `pig/kernels.py:267–300` `cross_validate()` | acc=0.875±0.125 (cv=2) |

---

## 4. Gap Analysis

| Item | Status | Notes |
|---|---|---|
| **Graph constructions** | | |
| Co-influence (CI) — signed | ✓ | `correlation_topk` (default) |
| Co-influence (CI) — absolute | ✓ | `abs_correlation_topk` |
| Diff-correlation (fine-tune vs base) | ✓ | `diff_correlation_topk` |
| Direct influence (DI) via paired patching | ✗ | `causal.py` has the forward-pass machinery but it is not a graph builder; natural location: `src/pig/graphs/direct_influence_topk.py` |
| Partial correlation (PC) | ✗ | Not implemented anywhere; would go in `pig/graph.py` or a new `graphs/` plugin |
| **Kernels** | | |
| Linear | ✓ | `ClassicalKernelClassifier(kernel="linear")` |
| RBF | ✓ | `ClassicalKernelClassifier(kernel="rbf")` |
| RFF (Random Fourier Features) | ✗ | Not implemented; natural location: `pig/kernels.py` as `RFFKernelClassifier` |
| Quantum fidelity | ✓ | `QuantumKernelClassifier` + `compute_fidelity_kernel_matrix()` |
| **Observables** | | |
| Logit margin (target $-$ distractor) | ✓ | Default for IOI (via `y_distractor` in meta) |
| Raw logit | ✓ | Fallback when no distractor token |
| Log-probability $\log p(y^\star \mid x)$ | ✗ | Not implemented; `_observable_from_logits` only computes logits, not softmax |
| Cross-entropy loss | ✗ | Not implemented |
| **Node types** | | |
| Residual stream (`res`) | ✓ | `NODE_TYPE_RES`; hooks on `transformer.h[l]` output |
| Attention head (`att`) | ✓ | `NODE_TYPE_ATT`; per-head, hooks on `attn.c_proj` pre-hook |
| MLP output (`mlp`) | ✓ | `NODE_TYPE_MLP`; hooks on `block.mlp` output |
| Token-level granularity | ✓ | Every `(layer, token)` is a distinct node |
| **Sparsification modes** | | |
| Top-$k$ per source | ✓ | Default, `k=5` |
| Global top-$k$ | ✗ | Not implemented; could be added to `GraphBuilder` |
| Threshold-only $\lvert w \rvert > \tau$ | partial | `min_weight` parameter exists (default 0.0); effectively disabled unless set |
| Combined (threshold + top-$k$) | partial | Both `k` and `min_weight` can be set simultaneously |
| **Evaluation** | | |
| Classification accuracy | ✓ | `ClassicalKernelClassifier.cross_validate()` |
| AUC | ✓ | `ClassificationResult.auc` (binary only) |
| Kernel alignment | ✗ | Not implemented |
| Kernel spectrum | ✗ | Not implemented |
| Spectral clustering | ✗ | Not implemented |
| **Robustness ablations** | | |
| Vary $k$ (top-k) | ✓ | `k` parameter in `create_graph_builder()` |
| Vary shots $M$ (quantum) | ✓ | `shots` parameter in `compute_quantum_kernel_matrix()` |
| Vary circuit depth $D$ | ✓ | `depth` parameter in `QuantumFeatureMap` |
| Vary $n_s$ (subsample examples) | partial | `n_examples` in `create_ioi_dataset()`; no built-in subsampling utility |
| Noise injection on $E$ | ✗ | Not implemented |
| Vary WL depth | ✓ | `depth` parameter in `WLEncoder` / `compute_wl_features()` |
| **Prompt families** | | |
| IOI (name_swap, abba corruptions) | ✓ | `IOIGenerator` with two strategies |
| Key-value retrieval | ✗ | Not implemented; natural location: `pig/prompts.py` |
| Bracket matching | ✗ | Not implemented |
| Arithmetic | ✗ | Not implemented |

---

## 5. Open Questions / Ambiguities

1. **Observable is logit margin, not logit.** The TFM equations use $O(f_\theta(x))$ without specifying whether it is a raw logit, a log-probability, or a margin. The code uses a logit margin for IOI (target $-$ distractor) because `y_distractor` is set in the IOI meta dict. This makes the effect $E_u^{(i)}$ a margin-difference, not a logit-difference. Does the TFM intend the margin? Should log-probability be added as an option?

2. **Denominator N vs N-1.** `_compute_correlation_matrix` divides by `n_samples` (population correlation). For small $N_s$ (e.g., 4), this introduces bias. Is this intentional? For the TFM analysis, does it matter whether we use biased or unbiased Pearson?

3. **Signed vs absolute correlation as edge weight.** The default builder keeps signed correlations; `abs_correlation_topk` discards signs. Signed negative correlations (visible in the `ioi:abba` graph, e.g., `w=−0.977`) indicate anti-correlated effects. The WL encoder and the kernel see the magnitude (via hashing), not the sign directly. Is the sign semantically meaningful in the theory?

4. **Direction constraint order within a (layer, token) cell.** The code imposes `att < mlp < res` at the same `(layer, token)`. This reflects a feedforward assumption (attention runs before MLP before residual accumulation). The TFM should make this explicit since it implies specific causal assumptions about the architecture.

5. **WL initial labels encode position, not semantics.** The initial node label is `f"L{layer}_T{token}_{node_type}"`. Two nodes with the same position across different graphs get the same initial label regardless of their actual activation values. Consequently, all structural variation between graphs comes from the **edge set** (which edges survive top-k). This is theoretically sound for graph kernels but means the WL feature does not encode the magnitude of patch effects — only the graph topology.

6. **WL edge-weight discretization.** Weights are binned to `int(w * 10)` precision (0.1 bins). A weight of $+0.77$ and $+0.85$ hash to different bins, while $+0.71$ and $+0.79$ hash to the same bin. The choice of 0.1 binning is arbitrary. Finer binning increases vocabulary size; coarser binning loses discriminative power.

7. **`build_per_example` uses a different formula.** The per-example graph builder uses $w_{ij} = 1/(1 + |E_i - E_j|)$ (effect-similarity), not correlation. This diverges from the theory's $C_s[u,v]$. The per-example graphs are therefore not consistent instances of the canonical slice-level graph.

8. **PCA before quantum kernel.** The quantum pipeline applies PCA to reduce WL features to exactly `n_qubits` components. This means the quantum kernel is a function of the first $n_q$ principal components of the WL feature space, not of the full WL features. The question of how many components to use (and whether PCA is the right reduction) is unresolved.

9. **Causal eval decoupled from graph construction.** `causal.py` implements a rich three-level causal evaluation (internal influence I, restoration fraction R, mediation M, necessity). This is entirely separate from the graph building pipeline — it validates selected edges post-hoc rather than driving edge selection. The TFM should clarify whether the causal eval is meant to be integrated (e.g., as a DI graph builder) or kept as a separate validation tool.

10. **No `max_seq_len` guard in HookedModel.** The `ToyHookedModel` truncates sequences to `max_seq_len=128`. The `HookedModel` (GPT-2 wrapper) does not; it relies on HuggingFace's own position embedding handling. For long prompts, the token count varies per example, and `get_effect_matrix` truncates to `min_tokens` — effectively ignoring the tail of longer prompts.

---

## 6. Appendix: Raw Snapshot

Full contents of `audit/stage_snapshot.txt` (produced by `audit/run_stage_snapshot.py`):

```
======================================================================
  STAGE 1 — Prompts  ( x_cln, x_crp, y_star, slice_label )
======================================================================

Slice labels used:  ioi:name_swap  |  ioi:abba
Total prompt pairs: 8 (4 per slice)

--- slice: ioi:name_swap (first 2 examples) ---
  [0] x_cln  = 'Nicole gave the book to Bob. Bob gave it back to'
       x_crp  = 'Bob gave the book to Bob. Bob gave it back to'
       y_star = 'Nicole'
  [1] x_cln  = 'Andrew lent the money to Lisa. Lisa returned it to'
       x_crp  = 'Lisa lent the money to Lisa. Lisa returned it to'
       y_star = 'Andrew'

--- slice: ioi:abba (first 2 examples) ---
  [0] x_cln  = 'Nicole gave the book to Bob. Bob gave it back to'
       x_crp  = 'Bob gave the book to Nicole. Nicole gave it back to'
       y_star = 'Nicole'
  [1] x_cln  = 'Andrew lent the money to Lisa. Lisa returned it to'
       x_crp  = 'Lisa lent the money to Andrew. Andrew returned it to'
       y_star = 'Andrew'


======================================================================
  STAGE 2 — Baseline observable  O_i^base = O( f_θ(x_crp) )
======================================================================

Model: toy_transformer  n_layers=2  n_heads=4  d_model=64
Observable: logit margin  target - distractor  (y_distractor from meta)

Baseline scores (base_score = O(x_crp)) — first 4 per slice:
  ioi:name_swap  base_scores = [0.231 1.45  0.123 0.76 ]
  ioi:abba  base_scores = [1.139 2.278 1.102 1.768]

======================================================================
  STAGE 3 & 4 — Patched observable O_{i,u}^patch  and  E_u^(i) = O_patch - O_base
======================================================================

PatchEffectDataset: 8 tensors

First tensor shape (effects): (2, 12, 1) = [n_layers=2, n_tokens=12, n_components=1]
  base_score  = 0.2305
  clean_score = 0.6981
  token_labels = ['Nicole', 'gave', 'the', 'book', 'to', 'Bob', '.', 'Bob',
                  'gave', 'it', 'back', 'to']

Flat effect matrix for slice 'ioi:name_swap'  shape [N_s, |V|]:
  Shape: (4, 22)  (N_s=4, |V|=22)
  Node ordering: (layer * n_tokens + token) * n_components + comp_idx
  (With res-only: comp_idx=0 always, so index = layer*11 + token)

  First 4 rows, first 12 columns (columns = node index 0..11):
     L0T0     L0T1     L0T2     L0T3     L0T4     L0T5     L0T6     L0T7     L0T8     L0T9    L0T10     L1T0
    0.173   -0.000    0.001    0.000    0.001    0.000    0.000    0.000   -0.000    0.000    0.000    0.350
    0.146   -0.004   -0.004   -0.002   -0.003   -0.003   -0.001   -0.002   -0.001   -0.001   -0.031    0.350
    0.144   -0.002   -0.003   -0.001   -0.001   -0.001   -0.002   -0.000   -0.000   -0.000   -0.000    0.350
    0.213   -0.000   -0.000   -0.000   -0.000    0.001   -0.000    0.001    0.000   -0.000   -0.001    0.350

======================================================================
  STAGE 5 — Co-influence correlation matrix  C_s[u,v] = corr_i(E_u^(i), E_v^(i))
======================================================================

Correlation matrix for slice 'ioi:name_swap'  shape: (22, 22)
(Signed Pearson correlation, denominator N=4, not N-1)

  First 12x12 block:
              L0T0    L0T1    L0T2    L0T3    L0T4    L0T5    L0T6    L0T7    L0T8    L0T9   L0T10    L1T0
     L0T0     1.000   0.712   0.655   0.701   0.480   0.782   0.773   0.719   0.720   0.317   0.460   0.000
     L0T1     0.712   1.000   0.867   0.959   0.956   0.993   0.669   0.998   1.000   0.889   0.936   0.000
     L0T2     0.655   0.867   1.000   0.973   0.846   0.893   0.906   0.839   0.860   0.784   0.701   0.000
     L0T3     0.701   0.959   0.973   1.000   0.927   0.970   0.826   0.942   0.955   0.862   0.835   0.000
     L0T4     0.480   0.956   0.846   0.927   1.000   0.922   0.558   0.948   0.952   0.984   0.967   0.000
     L0T5     0.782   0.993   0.893   0.970   0.922   1.000   0.742   0.989   0.993   0.839   0.888   0.000
     L0T6     0.773   0.669   0.906   0.826   0.558   0.742   1.000   0.638   0.664   0.450   0.384   0.000
     L0T7     0.719   0.998   0.839   0.942   0.948   0.989   0.638   1.000   0.999   0.879   0.942   0.000
     L0T8     0.720   1.000   0.860   0.955   0.952   0.993   0.664   0.999   1.000   0.883   0.936   0.000
     L0T9     0.317   0.889   0.784   0.862   0.984   0.839   0.450   0.879   0.883   1.000   0.951   0.000
    L0T10     0.460   0.936   0.701   0.835   0.967   0.888   0.384   0.942   0.936   0.951   1.000   0.000
     L1T0     0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000

  Matrix diagonal: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]
  Range: [0.000, 1.000]

Note: Stage 5' (direct-influence / paired patching) — NOT implemented
Note: Stage 5'' (partial correlation)               — NOT implemented

======================================================================
  STAGE 6 — Sparse graph  G_s = (V, E_s, w_s)  after top-k + direction constraint
======================================================================

Sparsification rule: top-k per source node, k=5
Direction constraint: enforce_direction=True
  Order rule: src < dst  iff  (layer, token, type_order) is lexicographically less
  Type order: att=0, mlp=1, res=2

Canonical graph  slice 'ioi:name_swap':
  Nodes: 22  Edges: 45
  Edge list (src_idx, dst_idx, weight) — first 10:
    (  0 L0T0.res,   1 L0T1.res, w=+0.7123)
    (  0 L0T0.res,   5 L0T5.res, w=+0.7821)
    (  0 L0T0.res,   6 L0T6.res, w=+0.7728)
    (  0 L0T0.res,   7 L0T7.res, w=+0.7185)
    (  0 L0T0.res,   8 L0T8.res, w=+0.7197)
    (  1 L0T1.res,   3 L0T3.res, w=+0.9589)
    (  1 L0T1.res,   4 L0T4.res, w=+0.9560)
    (  1 L0T1.res,   5 L0T5.res, w=+0.9929)
    (  1 L0T1.res,   7 L0T7.res, w=+0.9983)
    (  1 L0T1.res,   8 L0T8.res, w=+0.9998)

Canonical graph  slice 'ioi:abba':
  Nodes: 22  Edges: 45
  Edge list — first 10:
    (  0 L0T0.res,   1 L0T1.res, w=+0.7430)
    (  0 L0T0.res,   2 L0T2.res, w=+0.6812)
    (  0 L0T0.res,   3 L0T3.res, w=+0.7300)
    (  0 L0T0.res,   5 L0T5.res, w=-0.9766)
    (  0 L0T0.res,   7 L0T7.res, w=-0.9996)
    (  1 L0T1.res,   2 L0T2.res, w=+0.8853)
    (  1 L0T1.res,   3 L0T3.res, w=+0.9664)
    (  1 L0T1.res,   4 L0T4.res, w=+0.9512)
    (  1 L0T1.res,  10 L0T10.res, w=-0.9328)
    (  1 L0T1.res,  21 L1T10.res, w=-0.9319)

Per-example graphs: 8 total (4 × ioi:name_swap, 4 × ioi:abba)

======================================================================
  STAGE 7 — WL feature matrix  X[s, (h, l)] = |{v : l^(h)(v) = l}|
======================================================================

WL depth (iterations): 3
Feature matrix X shape: (8, 395)  = [n_graphs=8, n_features=395]
Non-zero columns: 395 out of 395

First 5 rows × 12 informative columns:
  d0_L0_T0_res  d0_L0_T10_re  d0_L0_T11_re  d0_L0_T1_res  d0_L0_T2_res  ...
  [     ioi:name_swap]             1             1             1             1             1   ...
  [     ioi:name_swap]             1             1             0             1             1   ...
  [     ioi:name_swap]             1             1             1             1             1   ...
  [     ioi:name_swap]             1             1             1             1             1   ...
  [          ioi:abba]             1             1             1             1             1   ...

Row ordering: rows 0–3: ioi:name_swap  |  rows 4–7: ioi:abba

======================================================================
  STAGE 8 — Kernel matrices  K[s,t]
======================================================================

[8a] Linear kernel  K_lin[s,t] = x̂_s · x̂_t  (shape (8, 8))
     (features standardized by StandardScaler before dot product)
[[ 0.81  -0.233  0.144  0.105 -0.199 -0.233 -0.196 -0.198]
 [-0.233  1.35  -0.22  -0.232 -0.237  0.041 -0.234 -0.235]
 [ 0.144 -0.22   0.799  0.105 -0.2   -0.233 -0.197 -0.198]
 [ 0.105 -0.232  0.105  0.842 -0.198 -0.231 -0.195 -0.196]
 [-0.199 -0.237 -0.2   -0.198  0.766 -0.236  0.132  0.171]
 [-0.233  0.041 -0.233 -0.231 -0.236  1.36  -0.233 -0.235]
 [-0.196 -0.234 -0.197 -0.195  0.132 -0.233  0.83   0.093]
 [-0.198 -0.235 -0.198 -0.196  0.171 -0.235  0.093  0.798]]

[8b] RBF kernel  K_rbf[s,t] = exp(-gamma * ||x̂_s - x̂_t||²)  (shape (8, 8))
     gamma (sklearn 'scale') = 1/(n_features * Var[X_scaled]) = 0.013961
[[1.    0.    0.001 0.    0.    0.    0.    0.   ]
 [0.    1.    0.    0.    0.    0.    0.    0.   ]
 [0.001 0.    1.    0.    0.    0.    0.    0.   ]
 [0.    0.    0.    1.    0.    0.    0.    0.   ]
 [0.    0.    0.    0.    1.    0.    0.001 0.001]
 [0.    0.    0.    0.    0.    1.    0.    0.   ]
 [0.    0.    0.    0.    0.001 0.    1.    0.   ]
 [0.    0.    0.    0.    0.001 0.    0.    1.   ]]

[8c] Quantum fidelity kernel  K_q[s,t] = |<0|U(z_s)†U(z_t)|0>|²  (shape (8, 8))
     n_qubits=4  depth=2  shots=None (exact)
     reduction: PCA to 4 components then QuantumFeatureMap
     Circuit: H gates → depth × (Rz(z_i), Rx(z_i), CZ chain)
[[1.    0.104 0.246 0.1   0.077 0.013 0.105 0.09 ]
 [0.104 1.    0.032 0.01  0.003 0.482 0.001 0.002]
 [0.246 0.032 1.    0.334 0.094 0.    0.131 0.107]
 [0.1   0.01  0.334 1.    0.011 0.021 0.019 0.014]
 [0.077 0.003 0.094 0.011 1.    0.    0.952 0.99 ]
 [0.013 0.482 0.    0.021 0.    1.    0.    0.   ]
 [0.105 0.001 0.131 0.019 0.952 0.    1.    0.983]
 [0.09  0.002 0.107 0.014 0.99  0.    0.983 1.   ]]

======================================================================
  STAGE 9 — Classification  (SVM on precomputed kernel / cross-validation)
======================================================================

[9a] Linear kernel SVM CV (cv=2):
     accuracy_mean = 0.875  ± 0.125  (2 folds)

[9b] RBF kernel SVM CV (cv=2):
     accuracy_mean = 0.875  ± 0.125  (2 folds)

[9c] Quantum kernel SVM CV (cv=2):
     accuracy_mean = 0.875  ± 0.125  (2 folds)

======================================================================
  PIPELINE SUMMARY
======================================================================

Stage 1  | 8 PromptPair objects in 2 slices
Stage 2  | Baseline observable: logit margin O = logit(y*) - logit(y_distractor)
Stage 3  | Patched observable matrix:  shape [4, 2, T, 1]  (res-only)
Stage 4  | Patch-effect tensor E:      shape [4, 22] after flatten
Stage 5  | Correlation matrix C_s:     shape [22, 22]  (signed Pearson, denom N)
Stage 5' | Direct-influence DI:        NOT IMPLEMENTED in graph builders
Stage 5''| Partial correlation PC:     NOT IMPLEMENTED
Stage 6  | Sparse graph G_s:           top-k=5 per source, direction enforced
Stage 7  | WL feature matrix X:        shape (8, 395)  depth=3
Stage 8  | Linear kernel K_lin:        shape (8, 8)
         | RBF kernel K_rbf:           shape (8, 8)  gamma=0.0140
         | Quantum kernel K_q:         shape (8, 8)  n_qubits=4 depth=2
Stage 9  | SVM cross-validation on per-example graphs
```
