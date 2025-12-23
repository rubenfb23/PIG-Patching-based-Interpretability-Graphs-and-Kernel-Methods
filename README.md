# Patching-based Interpretability Graphs (PIG)

A Python framework for mechanistic interpretability that:

1. Generates interventional datasets via **activation patching**
2. Summarizes patch effects as **sparse directed graphs** (one graph per "slice" of prompts/corruptions)
3. Compares slices using **kernel methods** (classical and quantum) to induce a similarity geometry over circuits

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation (Model & Prompts) | ✅ Complete |
| 2 | Patching Engine | ✅ Complete |
| 3 | Graph Construction | ✅ Complete |
| 4 | Classical Kernels | ✅ Complete |
| 5 | Quantum Kernels | ✅ Complete |
| 6 | Evaluation & Ablations | 🚧 Planned |
| 7 | Packaging & Reproducibility | 🟡 Partial |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PIG.git
cd PIG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

```python
from pig.model import HookedModel
from pig.prompts import create_ioi_dataset
from pig.patching import compute_patch_effects
from pig.graph import GraphBuilder
from pig.embeddings import compute_wl_features
from pig.kernels import train_classical_baseline

# 1. Load a model with activation hooks
model = HookedModel(model_name="gpt2")

# 2. Generate clean/corrupted prompt pairs
dataset = create_ioi_dataset(n_examples=50, corruption="name_swap", seed=42)

# 3. Compute patch effects for all (layer, token) positions
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

## Development

### Running Validation Scripts

The project includes validation scripts for each implementation phase:

```bash
# Phase 1: Model & Prompts
python scripts/validate_story_1_1.py  # Model setup
python scripts/validate_story_1_2.py  # Prompt generation

# Phase 2: Patching
python scripts/validate_story_2_1.py  # Patching hooks
python scripts/validate_story_2_2.py  # Patch effect tensor

# Phase 3: Graphs
python scripts/validate_story_3_1.py  # Graph construction

# Phase 4: Classical Kernels
python scripts/validate_story_4_1.py  # WL embeddings
python scripts/validate_story_4_2.py  # Classical baseline

# Phase 5: Quantum Kernels
python scripts/validate_story_5_1.py  # Quantum feature map
python scripts/validate_story_5_2.py  # Quantum fidelity kernel
```

### Running Tests

```bash
pytest tests/
```

## Architecture

```
src/pig/
├── model.py       # HookedModel: activation capture & patching
├── prompts.py     # Prompt generators (IOI task, corruption strategies)
├── patching.py    # Patch-effect tensor computation & caching
├── graph.py       # Graph construction from effects
├── embeddings.py  # Weisfeiler-Lehman graph embeddings
├── kernels.py     # Classical SVM classifiers (linear, RBF)
└── quantum.py     # Quantum feature maps + fidelity kernels
```

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

## Running Tests

```bash
# Run all tests (excluding slow model tests)
pytest tests/ --ignore=tests/test_model.py

# Run with model tests (requires GPU)
pytest tests/ -m slow

# Run validation scripts
python scripts/validate_story_1_1.py
python scripts/validate_story_1_2.py
python scripts/validate_story_2_1.py
python scripts/validate_story_2_2.py
python scripts/validate_story_3_1.py
python scripts/validate_story_4_1.py
python scripts/validate_story_4_2.py
python scripts/validate_story_5_1.py
python scripts/validate_story_5_2.py
```

## Documentation

- [docs/patching_graphs_v2.pdf](docs/patching_graphs_v2.pdf) - Original slide deck with theoretical foundations
- [docs/api.md](docs/api.md) - API reference
- [PLAN.md](PLAN.md) - Detailed implementation plan with acceptance criteria

## License

See [LICENSE](LICENSE).

## Acknowledgments

This work is tutored by David Olivieri from University of Vigo.

Ruben Fernandez-Boullon
