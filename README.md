# Patching-based Interpretability Graphs (PIG)

A Python framework for mechanistic interpretability that:

1. Generates interventional datasets via **activation patching**
2. Summarizes patch effects as **sparse directed graphs** (one graph per "slice" of prompts/corruptions)
3. Compares slices using **kernel methods** (classical and quantum) to induce a similarity geometry over circuits

## Current Publication Track

The repo is currently being tightened around the classical publication path:

- Canonical method object: one correlation graph per `slice`
- Main benchmark: `IOI`
- Main claim: cheap correlational proposal plus interventional causal validation
- Auxiliary evidence only: `WL + linear/RBF SVM`, bootstrap-slice WL,
  fixed-layout adjacency baselines, and null controls

For reproducible sweeps over `seed`, `k`, `node_types`, and `num_examples`, use:

```bash
uv run python scripts/run_classical_publication_study.py --model-name toy_transformer
```

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

Pipeline logs are versioned by default (one file per run):

- `outputs/pipeline_logs/pipeline_<model>_<utc>.log`
- Logs are opened in write mode for each run (no cross-run append mixing).

### Causal A/B/C Evaluation (GPT-2, fast subset)

Run interventional causal evaluation on a small subset from cached patch effects:

```bash
uv run pig causal-eval --model-name gpt2
```

The causal flow now uses a strict disjoint split:

- `discovery` subset: proposes candidate edges
- `evaluation` subset: evaluates those candidates (no circular reuse)

Default cache location is now model-scoped:

- `gpt2` -> `.cache/patch_effects/gpt2`
- `toy_transformer` -> `.cache/patch_effects/toy_transformer`
- local checkpoint path (e.g. `outputs/gpt2_gsm8k_distilled`) ->
  `.cache/patch_effects/outputs_gpt2_gsm8k_distilled`

You can also append it to the existing pipeline:

```bash
uv run pig pipeline --model-name gpt2 --with-causal-eval

# Forward causal options through pipeline
uv run pig pipeline --model-name gpt2 --with-causal-eval \
  --causal-no-require-clean-better \
  --causal-overwrite
```

Common overrides:

```bash
# Use your local fine-tuned GPT-2 checkpoint path
uv run pig causal-eval --model-name outputs/gpt2_gsm8k_distilled

# Tune runtime budget
uv run pig causal-eval --num-examples 20 --num-edges 5 --bootstrap-samples 200 --permutation-samples 200

# If you intentionally need legacy cache entries (no modern metadata)
uv run pig causal-eval --model-name gpt2 --allow-legacy-cache

# Disable the clean_score > base_score subset filter (for sensitivity analysis)
uv run pig causal-eval --model-name gpt2 --no-require-clean-better

# Reuse an existing non-empty output directory (explicit opt-in)
uv run pig causal-eval --model-name gpt2 --output-dir outputs/causal_eval/manual_run --overwrite
```

# Classical publication sweep

The publication runner writes per-run JSON plus an aggregate CSV summary:

```bash
uv run python scripts/run_classical_publication_study.py \
  --model-name toy_transformer \
  --seeds 7,42,123 \
  --k-grid 1,3,5,10 \
  --node-types-grid 'res;res,mlp,att'
```

The runner keeps the legacy per-example `WL + SVM` columns and also reports
bootstrap-slice WL, fixed-layout adjacency baselines, and label/topology/weight
null-control deltas in `auxiliary_baselines`.

For causal sensitivity analysis without the `clean_score > base_score` filter:

```bash
uv run python scripts/run_classical_publication_study.py \
  --model-name gpt2 \
  --seeds 7,42,123 \
  --k-grid 5 \
  --num-examples-grid 20 \
  --node-types-grid 'res' \
  --no-causal-require-clean-better
```

Default outputs land in:

- `outputs/classical_publication/<model>_<utc>/results.json`
- `outputs/classical_publication/<model>_<utc>/summary.csv`
- `outputs/classical_publication/<model>_<utc>/runs/*.json`

To aggregate a finished study into a short markdown report with post-filter slice balance:

```bash
uv run python scripts/summarize_classical_publication_study.py \
  outputs/classical_publication/<model>_<utc> \
  --output outputs/classical_publication/<model>_<utc>/summary_report.md
```

Cache safety policy:

- Causal eval filters cache tensors by `model_fingerprint` + `axis_fingerprint`.
- Legacy cache entries without modern metadata are rejected by default (fail-closed).
- `--component-size` is deprecated and only retained as validation.
- Use `--cache-dir` only when you intentionally override model-scoped defaults.
- `--component-size` is auto-derived from `--model-name` + `--node-types` when omitted; incompatible explicit values fail fast.

Statistical policy in causal output:

- Per-edge p-values are reported with multiple-testing corrections (`FDR/BH` and `Bonferroni`).
- `run_metadata` includes a `clean_better_filter_report` with side-by-side summaries:
  - without filter
  - with `clean_score > base_score` filter
  - per-slice effect summaries and confidence intervals

Outputs are versioned by default:

- `outputs/causal_eval/<model>_<utc>_seed<seed>/causal_eval.json`
- `outputs/causal_eval/<model>_<utc>_seed<seed>/causal_eval.npz`
- `outputs/causal_eval/<model>_<utc>_seed<seed>/quicklook_causal.png`
- `outputs/causal_eval/<model>_<utc>_seed<seed>/causal_eval.log`

If `--output-dir` points to a non-empty directory, causal eval fails unless `--overwrite` is set.

You can regenerate the quicklook manually:

```bash
RUN_DIR=outputs/causal_eval/<model>_<utc>_seed<seed>
uv run python scripts/plot_causal_eval.py \
  --json-path "$RUN_DIR/causal_eval.json" \
  --npz-path "$RUN_DIR/causal_eval.npz" \
  --output-path "$RUN_DIR/quicklook_causal.png"
```

### Local 4D Graph Viewer (Web)

You can now run a local interactive viewer backend over WebSocket and connect a
TypeScript 3D client.

1. Generate cache files (toy model by default):

```bash
uv run pig viewer-cache --model-name toy_transformer
```

1. Start backend from repo root:

```bash
uv run pig viewer --cache-dir .cache/patch_effects/toy_transformer --host 127.0.0.1 --port 8765
```

1. In another terminal, start frontend:

```bash
cd web
npm install
npm run dev
```

1. Open `http://127.0.0.1:5173` and explore the graph by slice/layer/token and
edge-weight threshold.

Notes:

- The backend expects JSON tensors produced by `PatchEffectCache` in `--cache-dir`.
- The viewer returns sparse top-k graphs and supports interactive filtering via
    WebSocket (`filter_update` -> `graph_update`).

#### Deploy / run modes

Development mode:

```bash
# terminal 1 (backend)
uv run pig viewer --cache-dir .cache/patch_effects/toy_transformer --host 127.0.0.1 --port 8765

# terminal 2 (frontend)
cd web
npm run dev
```

Production-like local mode:

```bash
cd web
npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

Then open `http://127.0.0.1:4173` while backend is still running on `8765`.

#### Troubleshooting

- `address already in use` on port `8765` means an older backend process is still running.
  - Check: `ss -ltnp '( sport = :8765 )'`
  - Stop process: `kill <PID>`
- If cache generation options changed (e.g. `--node-types`), regenerate cache before launching backend.
- If frontend changes are not visible, hard refresh browser (`Ctrl+Shift+R`).

### Cache migration / rebuild

```bash
# Rebuild modern cache metadata for base GPT-2
uv run pig viewer-cache --model-name gpt2

# Rebuild for toy model
uv run pig viewer-cache --model-name toy_transformer

# Rebuild for a fine-tuned/distilled checkpoint path
uv run pig viewer-cache --model-name outputs/gpt2_gsm8k_distilled
```

## GPT-2 Teacher Distillation (GSM8K)

Install extra packages for this workflow:

```bash
uv pip install datasets accelerate
```

Generate distilled SFT data with `openai/gpt-oss-20b` as teacher:

```bash
uv run python -m gpt2.distill_gsm8k \
  --teacher-model openai/gpt-oss-20b \
  --output-path outputs/gsm8k_distilled_gptoss20b.jsonl \
  --max-examples 2000 \
  --batch-size 2
```

Then fine-tune GPT-2 student with the generated JSONL (`text` field):

```bash
PYTHONPATH=src uv run torchrun --standalone --nproc_per_node=4 \
  -m gpt2.finetuning \
  --data-path outputs/gsm8k_distilled_gptoss20b.jsonl \
  --text-key text \
  --model-name gpt2 \
  --output-dir outputs/gpt2_gsm8k_distilled \
  --seq-len 512 \
  --epochs 3
```

One-shot wrapper (runs both phases in sequence):

```bash
src/gpt2/run_distill_and_finetune.sh
```

By default it uses the full GSM8K train split (`--max-examples -1`).  
For a shorter run:

```bash
src/gpt2/run_distill_and_finetune.sh --max-examples 2000
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
from pig.graph import create_graph_builder
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
builder = create_graph_builder(
    builder_name="correlation_topk",
    k=5,
    enforce_direction=True,
)
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
├── toy_model/     # ToyHookedModel package: tiny local transformer for fast tests
│   ├── __init__.py
│   ├── config.py
│   ├── layers.py
│   ├── model.py
│   └── trainer.py
├── graphs/        # Graph strategy plugins (one file per strategy)
│   ├── base.py
│   ├── registry.py
│   ├── correlation_topk.py
│   └── abs_correlation_topk.py
├── prompts.py     # Prompt generators (IOI task, corruption strategies)
├── patching.py    # Patch-effect tensor computation & caching
├── graph.py       # Graph construction from effects
├── embeddings.py  # Weisfeiler-Lehman graph embeddings
├── kernels.py     # Classical SVM classifiers (linear, RBF)
├── quantum.py     # Quantum feature maps + fidelity kernels
└── visualization.py  # Heatmaps, PCA, and reporting outputs
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
from pig.toy_model import (
    TinyTrainingConfig,
    TinyTransformerConfig,
    ToyHookedModel,
    train_toy_model,
)

toy = ToyHookedModel(
    config=TinyTransformerConfig(
        d_model=32, n_layers=2, n_heads=4, mlp_dim=64, max_seq_len=96
    ),
    device="cpu",
)

history = train_toy_model(
    toy,
    texts=[
        "Alice gave the book to Bob .",
        "Bob gave the book to Alice .",
    ],
    config=TinyTrainingConfig(epochs=3, learning_rate=1e-3),
)
print(history[-1].mean_loss)
```

```bash
# Train directly from a corpus file (one sample per line)
uv run python -m pig.toy_model.trainer \
  --data-file data/toy_corpus.txt \
  --epochs 10 \
  --save-path outputs/toy_model.pt
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
from pig.graph import create_graph_builder, get_available_graph_builders

print(get_available_graph_builders())
# ['abs_correlation_topk', 'correlation_topk', ...]

builder = create_graph_builder(
    builder_name="correlation_topk",
    k=5,
    enforce_direction=True,
)

# Per-slice graphs (correlation-based)
graphs = builder.build_all(dataset)

# Per-example graphs (for classification)
graphs_with_labels = builder.build_per_example(dataset)
```

```bash
# Run the full pipeline with a selected graph strategy
uv run pig pipeline --model-name toy_transformer --graph-builder correlation_topk
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

### `pig.visualization` - Patching Heatmaps (Layer x Token)

```python
from pig.visualization import (
    prepare_patching_heatmap_inputs,
    plot_patching_heatmap_layer_token,
)

E, nodes_df, examples_df = prepare_patching_heatmap_inputs(dataset)

result = plot_patching_heatmap_layer_token(
    E=E,
    nodes_df=nodes_df,
    examples_df=examples_df,
    slice_filter="ioi:name_swap",  # or ["ioi:name_swap", "ioi:abba"]
    component="resid",
    agg="mean",                    # "mean" or "median"
    subset="all",                  # or "clean_correct_corrupted_wrong"
)

print(result["output_png"])
print(result["output_html"])
print(result["output_json"])
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
