# PIG API Reference

This document provides detailed API documentation for all modules in the PIG (Patching-based Interpretability Graphs) framework.

## Table of Contents

- [pig.model](#pigmodel)
- [pig.toy_model](#pigtoy_model)
- [pig.prompts](#pigprompts)
- [pig.patching](#pigpatching)
- [pig.graph](#piggraph)
- [pig.embeddings](#pigembeddings)
- [pig.kernels](#pigkernels)

---

## pig.model

Model setup with activation capture and patching hooks.

### GPT-2 Architecture (used by `HookedModel`)

```mermaid
flowchart TD
    A[Prompt text] --> B[GPT-2 tokenizer]
    B --> C[input_ids]
    C --> D[token + positional embeddings]
    D --> E[Block 0]
    E --> F[Block 1]
    F --> G[...]
    G --> H[Block n_layer-1]
    H --> I[ln_f]
    I --> J[lm_head]
    J --> K[logits]

    subgraph BLK[Transformer Block i]
        direction TB
        X1[Residual input x]
        X1 --> X2[LayerNorm]
        X2 --> X3[Self-attention QKV]
        X3 --> X4[c_proj]
        X4 --> X5[Residual add]
        X5 --> X6[LayerNorm]
        X6 --> X7[MLP]
        X7 --> X8[Residual add -> block output]
    end

    classDef hook fill:#eef,stroke:#446,stroke-width:1px
    H1[[res node\n(layer, token)]]:::hook
    H2[[mlp node\n(layer, token)]]:::hook
    H3[[att node\n(layer, token, head)\npre-hook at attn.c_proj]]:::hook

    X8 -. capture/patch .-> H1
    X7 -. capture/patch .-> H2
    X4 -. capture/patch .-> H3
```

Operational details in this repo:

- Backed by `AutoModelForCausalLM` (default `gpt2`) with frozen parameters.
- `res` hooks are registered on each `transformer.h[i]` block output.
- `mlp` hooks are registered on `transformer.h[i].mlp` output.
- `att` hooks are forward pre-hooks on `transformer.h[i].attn.c_proj`; tensors are split into per-head vectors using `head_dim = d_model / n_heads`.

### HookedModel

```python
class HookedModel(model_name: str = "gpt2", device: str | None = None)
```

A wrapper around HuggingFace transformers with hooks for activation patching.

**Parameters:**
- `model_name`: Name of the model to load (default: "gpt2")
- `device`: Device to run on ("cuda", "cpu", or None for auto-detect)

**Attributes:**
- `model`: The underlying transformer model
- `tokenizer`: The model's tokenizer
- `device`: The device being used
- `n_layers`: Number of transformer layers
- `d_model`: Model hidden dimension

**Methods:**

#### forward

```python
def forward(self, text: str) -> torch.Tensor
```

Run forward pass and return logits.

**Parameters:**
- `text`: Input text string

**Returns:** Logits tensor of shape `[1, seq_len, vocab_size]`

#### cache_clean

```python
def cache_clean(self, text: str) -> dict[tuple[int, int], torch.Tensor]
```

Cache residual stream activations for all (layer, token) positions.

**Parameters:**
- `text`: Input text to cache activations for

**Returns:** Dictionary mapping `(layer, token)` tuples to activation tensors

#### score

```python
def score(self, text: str, target: str) -> float
```

Compute the logit of a target token given input text.

**Parameters:**
- `text`: Input text (prompt)
- `target`: Target token to score

**Returns:** Logit value for the target token at the final position

#### patched_score

```python
def patched_score(
    self,
    text: str,
    target: str,
    clean_cache: dict[tuple[int, int], torch.Tensor],
    position: tuple[int, int]
) -> float
```

Compute score with one position patched from clean activations.

**Parameters:**
- `text`: Input text (typically corrupted)
- `target`: Target token to score
- `clean_cache`: Cached clean activations from `cache_clean()`
- `position`: `(layer, token)` position to patch

**Returns:** Logit value after patching

---

## pig.toy_model

Tiny in-repo transformer backend used for fast tests and low-resource runs.

### Toy Transformer Architecture (used by `ToyHookedModel`)

```mermaid
flowchart TD
    T1[Prompt text] --> T2[Regex tokenization]
    T2 --> T3[Hashed IDs\nsha256 -> modulo vocab]
    T3 --> T4[input_ids]
    T4 --> T5[token_embedding + position_embedding]
    T5 --> T6[Tiny Layer 0]
    T6 --> T7[Tiny Layer 1]
    T7 --> T8[...]
    T8 --> T9[ln_f]
    T9 --> T10[lm_head]
    T10 --> T11[logits]

    subgraph TBLK[_TinyLayer i (pre-norm)]
        direction TB
        Y1[Residual input x]
        Y1 --> Y2[ln_1]
        Y2 --> Y3[qkv linear]
        Y3 --> Y4[causal masked self-attention]
        Y4 --> Y5[head outputs\n1 x seq_len x n_heads x head_dim]
        Y5 --> Y6[out_proj]
        Y6 --> Y7[Residual add]
        Y7 --> Y8[ln_2]
        Y8 --> Y9[fc_1 -> GELU -> fc_2]
        Y9 --> Y10[Residual add -> block output]
    end

    classDef hook fill:#efe,stroke:#464,stroke-width:1px
    TH1[[att node\n(layer, token, head)]]:::hook
    TH2[[mlp node\n(layer, token)]]:::hook
    TH3[[res node\n(layer, token)]]:::hook

    Y5 -. capture/patch .-> TH1
    Y9 -. capture/patch .-> TH2
    Y10 -. capture/patch .-> TH3
```

Default `TinyTransformerConfig` values:

- `vocab_size=512`, `max_seq_len=128`
- `d_model=64`, `n_layers=2`, `n_heads=4`, `head_dim=16`
- `mlp_dim=128`, `seed=0`

Key classes:

- `TinyTransformerConfig`: dataclass that defines model dimensions.
- `_TinyLayer`: one pre-norm transformer block.
- `ToyHookedModel`: API-compatible backend with `cache_clean_components`, `score`, `patched_score`, and `patched_score_multi`.

---

## pig.prompts

Prompt generation for the Indirect Object Identification (IOI) task.

### SliceLabel

```python
@dataclass
class SliceLabel:
    task: str
    corruption: str
```

Label identifying a slice (task family + corruption type).

**Attributes:**
- `task`: Task identifier (e.g., "ioi")
- `corruption`: Corruption type (e.g., "name_swap", "abba")

### PromptPair

```python
@dataclass
class PromptPair:
    x_cln: str      # Clean prompt
    x_crp: str      # Corrupted prompt
    y_star: str     # Target token
    slice_label: SliceLabel
    meta: dict      # Additional metadata
```

A clean/corrupted prompt pair with target token.

### CorruptionStrategy (Protocol)

```python
class CorruptionStrategy(Protocol):
    def corrupt(self, clean_prompt: str, meta: dict) -> str: ...
```

Protocol for corruption strategies.

### NameSwapCorruption

```python
class NameSwapCorruption:
    def corrupt(self, clean_prompt: str, meta: dict) -> str
```

Replaces S (subject) with IO (indirect object), removing the target from the corrupted prompt.

**Example:**
- Clean: "John gave the book to Mary. Mary gave it back to" → predict "John"
- Corrupt: "John gave the book to John. John gave it back to" → predict "John" less likely

### ABBACorruption

```python
class ABBACorruption:
    def corrupt(self, clean_prompt: str, meta: dict) -> str
```

Creates ABBA pattern where both names are the same.

### IOIGenerator

```python
class IOIGenerator(
    templates: list[str] | None = None,
    names: list[str] | None = None,
    corruption: CorruptionStrategy | None = None,
    seed: int | None = None
)
```

Generator for IOI task prompt pairs.

**Parameters:**
- `templates`: List of template strings (default: built-in IOI templates)
- `names`: List of names to use (default: common English names)
- `corruption`: Corruption strategy (default: NameSwapCorruption)
- `seed`: Random seed for reproducibility

**Methods:**

#### generate

```python
def generate(self) -> PromptPair
```

Generate a single prompt pair.

#### generate_batch

```python
def generate_batch(self, n: int) -> list[PromptPair]
```

Generate multiple prompt pairs.

### create_ioi_dataset

```python
def create_ioi_dataset(
    n_examples: int = 50,
    corruption: str = "name_swap",
    seed: int = 42
) -> list[PromptPair]
```

Convenience function to create an IOI dataset.

**Parameters:**
- `n_examples`: Number of examples to generate
- `corruption`: Corruption type ("name_swap" or "abba")
- `seed`: Random seed

**Returns:** List of PromptPair objects

---

## pig.patching

Patch-effect tensor computation and caching.

### PatchEffectTensor

```python
@dataclass
class PatchEffectTensor:
    effects: NDArray[np.float32]  # Shape: [num_layers, num_tokens, num_components]
    component_axis: list[ComponentSpec]
    prompt_pair: PromptPair
    base_score: float
    clean_score: float
```

Patch effects for a single example.

**Properties:**
- `num_layers`: Number of layers
- `num_tokens`: Number of token positions
- `num_components`: Number of components per token
- `slice_label`: The slice label from the prompt pair

**Methods:**

#### get_significant_positions

```python
def get_significant_positions(
    self,
    threshold: float = 0.1
) -> list[tuple[int, int, int, float]]
```

Get positions with effects above threshold.

**Returns:** List of `(layer, token, component_idx, effect)` tuples

### PatchEffectDataset

```python
class PatchEffectDataset:
    tensors: list[PatchEffectTensor]
```

Collection of patch effect tensors.

**Methods:**

#### add

```python
def add(self, tensor: PatchEffectTensor) -> None
```

Add a tensor to the dataset.

#### get_by_slice

```python
def get_by_slice(self, slice_label: SliceLabel) -> list[PatchEffectTensor]
```

Get all tensors for a specific slice.

#### get_common_dimensions

```python
def get_common_dimensions(self) -> tuple[int, int]
```

Get the minimum (num_layers, num_tokens) across all tensors.

#### get_effect_matrix

```python
def get_effect_matrix(
    self,
    slice_label: SliceLabel,
    max_tokens: int | None = None
) -> NDArray[np.float32]
```

Get stacked effect matrix for a slice.

**Parameters:**
- `slice_label`: The slice to get effects for
- `max_tokens`: Maximum tokens (truncates to this length)

**Returns:** Matrix of shape `[num_examples, num_layers * num_tokens]`

### PatchEffectCache

```python
class PatchEffectCache(cache_dir: Path | str = ".cache/patch_effects")
```

Disk cache for patch effect tensors.

**Methods:**
- `get(prompt_pair: PromptPair) -> PatchEffectTensor | None`
- `put(tensor: PatchEffectTensor) -> None`

### PatchEffectComputer

```python
class PatchEffectComputer(
    model: HookedModel,
    cache: PatchEffectCache | None = None
)
```

Computes patch effects for prompt pairs.

**Methods:**

#### compute_single

```python
def compute_single(self, prompt_pair: PromptPair) -> PatchEffectTensor
```

Compute patch effects for a single prompt pair.

#### compute_batch

```python
def compute_batch(
    self,
    prompt_pairs: list[PromptPair],
    show_progress: bool = False
) -> PatchEffectDataset
```

Compute patch effects for multiple prompt pairs.

### compute_patch_effects

```python
def compute_patch_effects(
    model: HookedModel,
    prompt_pairs: list[PromptPair],
    cache_dir: str | None = None,
    show_progress: bool = False,
    node_types: Sequence[str] | None = None
) -> PatchEffectDataset
```

Convenience function to compute patch effects with optional caching.

---

## pig.graph

Graph construction from patch effects.

### Node

```python
@dataclass
class Node:
    layer: int
    token: int
    node_type: str
    head: int | None = None
```

A node representing a `(layer, token, component)` patch point.

### Edge

```python
@dataclass
class Edge:
    src: int     # Source node index
    dst: int     # Destination node index
    weight: float
```

A directed edge between nodes.

### PatchInfluenceGraph

```python
@dataclass
class PatchInfluenceGraph:
    nodes: list[Node]
    edges: list[Edge]
    slice_label: SliceLabel
```

A directed graph representing patch influence patterns.

**Properties:**
- `num_nodes`: Number of nodes
- `num_edges`: Number of edges

**Methods:**

#### to_adjacency_matrix

```python
def to_adjacency_matrix(self) -> NDArray[np.float32]
```

Convert to dense adjacency matrix.

#### to_edge_list

```python
def to_edge_list(self) -> list[tuple[int, int, float]]
```

Get edges as `(src, dst, weight)` tuples.

#### to_networkx

```python
def to_networkx(self) -> nx.DiGraph
```

Convert to NetworkX directed graph.

### GraphBuilder

```python
class GraphBuilder(
    k: int = 5,
    enforce_direction: bool = True,
    similarity: str = "correlation"
)
```

Builds graphs from patch effect datasets.

**Parameters:**
- `k`: Number of top outgoing edges per node
- `enforce_direction`: Only allow edges from earlier to later positions
- `similarity`: Similarity metric ("correlation" or "cosine")

**Methods:**

#### build_single

```python
def build_single(
    self,
    dataset: PatchEffectDataset,
    slice_label: SliceLabel
) -> PatchInfluenceGraph
```

Build a graph for a single slice.

#### build_all

```python
def build_all(
    self,
    dataset: PatchEffectDataset
) -> dict[SliceLabel, PatchInfluenceGraph]
```

Build graphs for all slices in the dataset.

#### build_per_example

```python
def build_per_example(
    self,
    dataset: PatchEffectDataset
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]
```

Build individual graphs for each example.

---

## pig.embeddings

Weisfeiler-Lehman (WL) graph embeddings.

### WLEmbedding

```python
@dataclass
class WLEmbedding:
    features: dict[str, int]  # Feature label -> count
    graph_id: str
    depth: int
```

WL feature embedding for a single graph.

**Methods:**

#### to_vector

```python
def to_vector(self, vocabulary: list[str]) -> NDArray[np.float32]
```

Convert to dense vector using a vocabulary.

### WLEncoder

```python
class WLEncoder(depth: int = 3, use_edge_weights: bool = True)
```

Weisfeiler-Lehman subtree feature encoder.

**Parameters:**
- `depth`: Number of WL iterations (higher = more complex patterns)
- `use_edge_weights`: Whether to include edge weights in hashing

**Methods:**

#### encode

```python
def encode(self, graph: PatchInfluenceGraph) -> WLEmbedding
```

Compute WL features for a graph.

#### encode_batch

```python
def encode_batch(self, graphs: list[PatchInfluenceGraph]) -> list[WLEmbedding]
```

Encode multiple graphs.

### WLFeatureMatrix

```python
@dataclass
class WLFeatureMatrix:
    embeddings: list[WLEmbedding]
    vocabulary: list[str]
    slice_labels: list[SliceLabel]
```

Feature matrix from WL encoding of multiple graphs.

**Properties:**
- `num_graphs`: Number of graphs
- `num_features`: Number of features in vocabulary

**Methods:**

#### to_matrix

```python
def to_matrix(self) -> NDArray[np.float32]
```

Convert to dense feature matrix of shape `[num_graphs, num_features]`.

### compute_wl_features

```python
def compute_wl_features(
    graphs: dict[SliceLabel, PatchInfluenceGraph],
    depth: int = 3,
    cache_dir: str | None = None
) -> WLFeatureMatrix
```

Compute WL features for a collection of slice graphs.

### compute_wl_features_from_list

```python
def compute_wl_features_from_list(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    depth: int = 3,
    cache_dir: str | None = None
) -> WLFeatureMatrix
```

Compute WL features from a list of `(graph, label)` tuples.

---

## pig.kernels

Classical kernel methods for graph classification.

### ClassificationResult

```python
@dataclass
class ClassificationResult:
    accuracy: float
    auc: float | None  # For binary classification
    predictions: NDArray[np.int32]
    true_labels: NDArray[np.int32]
    kernel_type: str
```

Results from a classification experiment.

### LearningCurveResult

```python
@dataclass
class LearningCurveResult:
    train_sizes: NDArray[np.int32]
    train_scores_mean: NDArray[np.float64]
    train_scores_std: NDArray[np.float64]
    test_scores_mean: NDArray[np.float64]
    test_scores_std: NDArray[np.float64]
    kernel_type: str
```

Results from a learning curve experiment.

### ClassicalKernelClassifier

```python
class ClassicalKernelClassifier(
    kernel: Literal["linear", "rbf"] = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    random_state: int = 42,
    normalize: bool = True
)
```

SVM classifier with classical kernels.

**Parameters:**
- `kernel`: Kernel type ("linear" or "rbf")
- `C`: SVM regularization parameter
- `gamma`: RBF kernel coefficient
- `random_state`: Random seed for reproducibility
- `normalize`: Whether to standardize features

**Methods:**

#### fit

```python
def fit(
    self,
    X: NDArray[np.float32],
    y: list[SliceLabel] | NDArray
) -> ClassicalKernelClassifier
```

Fit the classifier.

#### predict

```python
def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]
```

Predict labels for new samples.

#### predict_proba

```python
def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]
```

Predict class probabilities.

#### evaluate

```python
def evaluate(
    self,
    X: NDArray[np.float32],
    y: list[SliceLabel] | NDArray
) -> ClassificationResult
```

Evaluate the classifier on a test set.

#### cross_validate

```python
def cross_validate(
    self,
    X: NDArray[np.float32],
    y: list[SliceLabel] | NDArray,
    cv: int = 5
) -> dict
```

Perform cross-validation.

**Returns:** Dictionary with keys:
- `accuracy_mean`: Mean CV accuracy
- `accuracy_std`: Standard deviation
- `kernel_type`: Kernel used
- `cv_folds`: Number of folds

#### learning_curve

```python
def learning_curve(
    self,
    X: NDArray[np.float32],
    y: list[SliceLabel] | NDArray,
    train_sizes: NDArray | None = None,
    cv: int = 5
) -> LearningCurveResult
```

Compute learning curve.

### train_classical_baseline

```python
def train_classical_baseline(
    feature_matrix: WLFeatureMatrix,
    kernel: Literal["linear", "rbf"] = "rbf",
    random_state: int = 42
) -> tuple[ClassicalKernelClassifier, dict]
```

Train a classical kernel baseline on WL features.

**Parameters:**
- `feature_matrix`: WL feature matrix from graph embeddings
- `kernel`: Kernel type
- `random_state`: Random seed

**Returns:** Tuple of (fitted classifier, cross-validation results)
