# Patching-based Interpretability Graphs (PIG)

A Python framework for mechanistic interpretability that:

1. Generates interventional datasets via **activation patching**
2. Summarizes patch effects as **sparse directed graphs** (one graph per "slice" of prompts/corruptions)
3. Compares slices using **kernel methods** (classical and quantum) to induce a similarity geometry over circuits

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PIG.git
cd PIG

# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync project + dev dependencies
uv sync --dev

# Run CLI commands without manual venv activation
uv run pig --help
```

If `uv` is not yet in your `PATH`, use:

```bash
~/.local/bin/uv sync --dev
~/.local/bin/uv run pig --help
```

Legacy `pip` workflow is still supported:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .

# For development
pip install -e ".[dev]"
```

## CLI Entry Point

Recommended invocation (no manual venv activation):

```bash
uv run pig pipeline
```

Run with the ultralight in-repo toy model:

```bash
uv run pig pipeline --model-name toy_transformer
```

Direct invocation also works if your environment exposes `pig` in `PATH`:

```bash
pig pipeline
```

If you want to continue even if a stage fails:

```bash
pig pipeline --continue-on-error
```

### Using uv or pdm

Both work with this repo because it is standard `pyproject.toml` + setuptools.

- Recommended: `uv` for speed and simpler day-to-day workflow.
- Use `pdm` if you need stronger dependency-group workflow and lockfile control.

Typical commands:

```bash
# uv
uv sync --dev
uv run pig --help
uv run pig pipeline
uv run pig pipeline --continue-on-error
uv run pig pipeline --model-name toy_transformer
uv run python scripts/compare_model_memory.py --model-a gpt2 --model-b toy_transformer --device cpu

# pdm
pdm install -G dev
pdm run pig pipeline
# or using script alias from pyproject.toml
pdm run pipeline
```

## Quick Start

```python
from pig.model import create_model
from pig.prompts import create_ioi_dataset
from pig.patching import compute_patch_effects
from pig.graph import GraphBuilder
from pig.embeddings import compute_wl_features
from pig.kernels import train_classical_baseline

# 1. Load a model with activation hooks
model = create_model(model_name="gpt2")
# model = create_model(model_name="toy_transformer")

# 2. Generate clean/corrupted prompt pairs
dataset = create_ioi_dataset(n_examples=50, corruption="name_swap", seed=42)

# 3. Compute patch effects for all (layer, token[, component]) positions
# Use node_types=("res", "mlp", "att") for fine-grained components.
effects = compute_patch_effects(model, dataset)

# 4. Build graphs from patch effects
builder = GraphBuilder(k=5, enforce_direction=True)
graphs = builder.build_all(effects)

# 5. Compute WL graph embeddings
features = compute_wl_features(graphs, depth=3)

# 6. Train a classical kernel baseline
classifier, results = train_classical_baseline(features, kernel="rbf")
print(f"Cross-validation accuracy: {results['accuracy_mean']:.2%}")
```

## Architecture

```
src/pig/
├── model.py       # HookedModel: activation capture & patching
├── toy_model.py   # ToyHookedModel: tiny local transformer for fast tests
├── prompts.py     # Prompt generators (IOI task, corruption strategies)
├── patching.py    # Patch-effect tensor computation & caching
├── graph.py       # Graph construction from effects
├── embeddings.py  # Weisfeiler-Lehman graph embeddings
├── kernels.py     # Classical SVM classifiers (linear, RBF)
└── quantum.py     # Quantum feature maps + fidelity kernels
```

## Model Architectures Used in PIG

### GPT-2 (`HookedModel` backend)

```mermaid
flowchart TD
    A[Text prompt] --> B[GPT-2 tokenizer]
    B --> C[input_ids: 1 x seq_len]
    C --> D[token embedding + positional embedding]
    D --> E[Transformer Block 0]
    E --> F[Transformer Block 1]
    F --> G[...]
    G --> H[Transformer Block n_layer-1]
    H --> I[Final LayerNorm ln_f]
    I --> J[lm_head projection to vocab]
    J --> K[logits: 1 x seq_len x vocab_size]

    subgraph BLK[GPT-2 block i: model.transformer.h[i]]
        direction TB
        L1[Input residual stream x]
        L1 --> L2[LayerNorm]
        L2 --> L3[Self-attention QKV]
        L3 --> L4[Head concat -> c_proj]
        L4 --> L5[Residual add]
        L5 --> L6[LayerNorm]
        L6 --> L7[MLP]
        L7 --> L8[Residual add -> block output]
    end

    classDef hook fill:#eef,stroke:#446,stroke-width:1px
    M1[[res hook\nblock output hidden_states]]:::hook
    M2[[mlp hook\nblock.mlp output]]:::hook
    M3[[att hook\npre-hook at block.attn.c_proj\n(per-head slices)]]:::hook

    L8 -. capture/patch .-> M1
    L7 -. capture/patch .-> M2
    L4 -. capture/patch .-> M3
```

Detail captured by PIG patching API:

- `res`: residual stream per `(layer, token)` from each transformer block output.
- `mlp`: MLP output per `(layer, token)` from `block.mlp`.
- `att`: per-head vectors per `(layer, token, head)` at `attn.c_proj` input.

### Toy Transformer (`ToyHookedModel` backend)

```mermaid
flowchart TD
    T0[Text prompt] --> T1[Regex tokenizer]
    T1 --> T2[Hashed token IDs\nsha256 -> modulo vocab]
    T2 --> T3[input_ids: 1 x seq_len]
    T3 --> T4[token_embedding + position_embedding]
    T4 --> T5[Tiny layer 0]
    T5 --> T6[Tiny layer 1]
    T6 --> T7[... up to n_layers-1]
    T7 --> T8[LayerNorm ln_f]
    T8 --> T9[lm_head]
    T9 --> T10[logits: 1 x seq_len x vocab_size]

    subgraph TBLK[_TinyLayer i (pre-norm)]
        direction TB
        U1[Input residual x]
        U1 --> U2[ln_1]
        U2 --> U3[Linear qkv -> q,k,v]
        U3 --> U4[Causal masked attention per head]
        U4 --> U5[Head outputs shape\n1 x seq_len x n_heads x head_dim]
        U5 --> U6[out_proj on flattened heads]
        U6 --> U7[Residual add]
        U7 --> U8[ln_2]
        U8 --> U9[fc_1 -> GELU -> fc_2]
        U9 --> U10[Residual add -> block output]
    end

    classDef hook fill:#efe,stroke:#464,stroke-width:1px
    V1[[att hook\nhead outputs before out_proj]]:::hook
    V2[[mlp hook\nmlp_out before residual add]]:::hook
    V3[[res hook\nblock output x]]:::hook

    U5 -. capture/patch .-> V1
    U9 -. capture/patch .-> V2
    U10 -. capture/patch .-> V3
```

Default toy config (`TinyTransformerConfig`):

- `vocab_size=512`, `max_seq_len=128`
- `d_model=64`, `n_layers=2`, `n_heads=4`, `head_dim=16`
- `mlp_dim=128`

## Core Concepts

### Paired Inputs

For each example, create `(x_clean, x_corrupt)` where the corruption breaks the target behavior:

```python
# Clean: "John gave the book to Mary. Mary gave it back to" → predict "John"
# Corrupt: "Mary gave the book to Mary. Mary gave it back to" → predict "Mary"
```

### Patch Effect

The causal effect of patching node `u = (layer, token)`:

```
E_u = O(patched_forward(x_corrupt; u)) - O(forward(x_corrupt))
```

where `O` is the observable (target token logit).

### Slices

Groups of examples by task family, corruption type, or difficulty. Each slice produces one graph.

### Graph Construction

- **Nodes**: Fixed set of `(layer, token)` positions
- **Edges**: Correlation of effect profiles within a slice
- **Direction**: Edges only from earlier to later positions
- **Sparsity**: Top-k outgoing edges per node

## Modules

### `pig.model` - Model Setup

```python
from pig.model import HookedModel

model = HookedModel(model_name="gpt2", device="cuda")

# Cache clean activations
cache = model.cache_clean("The capital of France is")

# Compute observable
score = model.score("The capital of France is", "Paris")

# Patch and compute
patched = model.patched_score(
    "The capital of Germany is", "Paris", cache, (6, 4)
)
```

### `pig.toy_model` - Ultra-Simple Local Transformer (for tests)

```python
from pig.toy_model import ToyHookedModel, TinyTransformerConfig

toy = ToyHookedModel(
    config=TinyTransformerConfig(
        d_model=32, n_layers=2, n_heads=4, mlp_dim=64, max_seq_len=96
    ),
    device="cpu",
)
```

### `pig.prompts` - Prompt Generation

```python
from pig.prompts import IOIGenerator, create_ioi_dataset

# Using the generator directly
gen = IOIGenerator(seed=42)
pair = gen.generate()
print(pair.x_cln)    # Clean prompt
print(pair.x_crp)    # Corrupted prompt
print(pair.y_star)   # Target token

# Convenience function
dataset = create_ioi_dataset(n_examples=100, corruption="name_swap")
```

### `pig.patching` - Effect Computation

```python
from pig.patching import compute_patch_effects, PatchEffectComputer

# With caching
dataset = compute_patch_effects(
    model, prompt_pairs,
    cache_dir=".cache/effects",
    show_progress=True
)

# Access effects
for tensor in dataset:
    print(tensor.shape)  # (num_layers, num_tokens)
    hotspots = tensor.get_significant_positions(threshold=0.1)
```

### `pig.graph` - Graph Construction

```python
from pig.graph import GraphBuilder

builder = GraphBuilder(k=5, enforce_direction=True)

# Per-slice graphs (correlation-based)
graphs = builder.build_all(dataset)

# Per-example graphs (for classification)
graphs_with_labels = builder.build_per_example(dataset)
```

### `pig.embeddings` - WL Features

```python
from pig.embeddings import compute_wl_features, WLEncoder

# From slice graphs
features = compute_wl_features(graphs, depth=3)
X = features.to_matrix()  # Shape: [num_slices, num_features]

# From per-example graphs
from pig.embeddings import compute_wl_features_from_list
features = compute_wl_features_from_list(graphs_with_labels, depth=3)
```

### `pig.kernels` - Classification

```python
from pig.kernels import ClassicalKernelClassifier, train_classical_baseline

# Quick training with cross-validation
clf, cv_results = train_classical_baseline(features, kernel="rbf")

# Manual control
clf = ClassicalKernelClassifier(kernel="linear", C=1.0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
result = clf.evaluate(X_test, y_test)
```

### `pig.quantum` - Quantum Kernels

```python
from pig.quantum import compute_quantum_kernel_matrix, train_quantum_kernel_baseline

# Compute a quantum kernel matrix from WL features
K, reducer = compute_quantum_kernel_matrix(
    features,
    n_qubits=4,
    depth=2,
    shots=500,
    reduction="pca",
)

# Train an SVM with the precomputed quantum kernel
clf, cv_results, _ = train_quantum_kernel_baseline(
    features,
    n_qubits=4,
    depth=2,
    shots=500,
    reduction="pca",
)
```

## Implementation Status

| Phase | Story | Status |
|-------|-------|--------|
| 1. Foundation | 1.1 Model Setup | ✅ Complete |
| 1. Foundation | 1.2 Prompt Generator | ✅ Complete |
| 2. Patching | 2.1 Patching Hooks | ✅ Complete |
| 2. Patching | 2.2 Effect Tensors | ✅ Complete |
| 3. Graphs | 3.1 Graph Builder | ✅ Complete |
| 4. Classical | 4.1 WL Embeddings | ✅ Complete |
| 4. Classical | 4.2 SVM Baseline | ✅ Complete |
| 5. Quantum | 5.1 Quantum Circuit | ✅ Complete |
| 5. Quantum | 5.2 Fidelity Kernel | ✅ Complete |
| 6. Evaluation | 6.1 Ablations | ⏳ Not Started |
| 6. Evaluation | 6.2 Figures | ⏳ Not Started |
| 7. Packaging | 7.1 Caching | ✅ Complete |
| 7. Packaging | 7.2 CLI | ⏳ Not Started |
| 7. Packaging | 7.3 Tests | ✅ Complete |

## Documentation

- [docs/patching_graphs_v2.pdf](docs/patching_graphs_v2.pdf) - Original slide deck with theoretical foundations
- [docs/api.md](docs/api.md) - API reference
- [PLAN.md](PLAN.md) - Detailed implementation plan with acceptance criteria

## License

See [LICENSE](LICENSE).

## Acknowledgments

This work is tutored by David Olivieri from University of Vigo.

Ruben Fernandez-Boullon
