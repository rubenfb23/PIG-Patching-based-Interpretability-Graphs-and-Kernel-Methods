# AGENTS RUNBOOK — PIG (Patching-based Interpretability Graphs)

This document is the operational handoff for any coding/research agent joining this repo.
It explains what the project does, what is already implemented, and exactly how to run and extend it safely.

---

## 1) Project mission (what PIG is)

PIG is an end-to-end pipeline that converts activation patching signals into graph representations and compares those graphs with classical and quantum kernels.

Core flow:

1. Generate clean/corrupted prompt pairs (IOI-style task).
2. Measure causal patch effects across transformer internals.
3. Aggregate effects into sparse directed graphs.
4. Embed graphs (WL features) and train kernel classifiers.
5. Compare classical kernels vs quantum fidelity kernel variants.

---

## 2) Current implementation status (repo reality)

Implemented and usable now:

- Model layer with hook-enabled execution (`gpt2` and local toy transformer).
- Prompt generation with at least swap-like and ABBA corruption variants.
- Patching engine and patch-effect dataset container.
- Graph builder with top-k sparsification and optional direction constraint.
- WL embeddings + classical SVM baselines.
- Quantum kernel utilities and quantum baseline routines.
- CLI pipeline runner that executes the full pipeline sequence.
- Artifact generation script for heatmaps and PCA visualization.

Still evolving:

- Broader evaluation/ablation automation and reporting breadth.
- Packaging polish and longer reproducibility reports.

---

## 3) Repo map (actual structure)

Primary package:

- `src/pig/model.py` — model abstraction and hooks.
- `src/pig/toy_model/model.py` — tiny local transformer backend implementation.
- `src/pig/toy_model/config.py` — toy model and training configs.
- `src/pig/toy_model/layers.py` — tiny block and tokenization helpers.
- `src/pig/toy_model/trainer.py` — toy model training utilities.
- `src/pig/toy_model/__init__.py` — public exports for toy model classes.
- `src/pig/prompts.py` — IOI prompt/corruption generators.
- `src/pig/patching.py` — patch effect computation + datasets.
- `src/pig/graph.py` — graph construction from effects.
- `src/pig/embeddings.py` — WL embeddings and feature utilities.
- `src/pig/kernels.py` — classical kernel workflows.
- `src/pig/quantum.py` — quantum kernel/fidelity workflows.
- `src/pig/visualization.py` — output figures (heatmaps/PCA).
- `src/pig/cli.py` — CLI entrypoint (`pig`).

Operational scripts:

- `scripts/generate_pipeline_artifacts.py`.
- `scripts/compare_model_memory.py`.

---

## 4) Environment and install

Python requirement: `>=3.10`.

Recommended setup (`uv`):

```bash
uv sync --dev
uv run pig --help
```

If `uv` is not on PATH:

```bash
~/.local/bin/uv sync --dev
~/.local/bin/uv run pig --help
```

Alternative editable install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 5) How to run the pipeline (canonical)

Main command:

```bash
uv run pig pipeline
```

Model selection:

```bash
uv run pig pipeline --model-name gpt2
uv run pig pipeline --model-name toy_transformer
```

Continue through failures (for diagnosis):

```bash
uv run pig pipeline --continue-on-error
```

What `pig pipeline` does internally (`src/pig/cli.py`):

1. Runs the pipeline scripts in a fixed sequence ending in artifact generation.
2. Sets environment variable `PIG_MODEL_NAME=<model-name>` for each script.
3. Renames fresh output images with model suffix (`_gpt2` or `_toy`) to avoid collisions.
4. Restores previously-suffixed outputs if missing.

---

## 6) Outputs and artifacts

Generated figures are written to `outputs/`.

Typical files include:

- Slice heatmaps (e.g., swap/ABBA corruption views).
- PCA embedding plot (`pca_embeddings*.png`).

Because CLI may suffix files per model, expect names like:

- `..._gpt2.png`
- `..._toy.png`

Do not assume unsuffixed names exist after pipeline runs.

---

## 7) Data model and conceptual contracts

### Prompt pair record

Each example should provide:

- `x_cln` (clean prompt)
- `x_crp` (corrupted prompt)
- `y_star` (target token/string)
- slice metadata (corruption/task label)

### Patch effect

For patch node `u`:

`E_u = O(patched(x_crp; u)) - O(x_crp)`

Large positive `E_u` implies causal restoration for the target behavior.

### Graph construction

- Fixed node set across slices.
- Edge weights from co-variation/correlation of node effects.
- Optional direction constraint (earlier → later only).
- Top-k outgoing edges to standardize sparsity.

### Embedding/kernel stage

- WL feature vectors are the default classical embedding.
- Classical: linear/RBF SVM style flows.
- Quantum: project/reduce features, build fidelity kernel, train with precomputed kernel.

---

## 8) Agent workflow playbook (recommended sequence)

When taking a new task:

1. Read `README.md` for user-facing expectations.
2. Read touched modules in `src/pig/` before editing.
3. Reproduce baseline with:
   - `uv run pig pipeline --model-name toy_transformer` (fast sanity path).
4. Keep changes minimal and pipeline-compatible.
5. If output naming or artifacts are affected, check `src/pig/cli.py` behavior.

---

## 9) Verification before handoff

Before finalizing a change, run:

```bash
uv run pig pipeline --model-name toy_transformer
```

Use `gpt2` pipeline run only when needed due to runtime/resource cost.

---

## 10) Common pitfalls and debugging notes

- Tokenization ambiguity can make `y_star` validation unstable if not normalized.
- Hook point mismatches can silently invalidate patching assumptions.
- Output file collisions are handled by suffixing; don’t hardcode raw filenames.
- Large model runs are expensive; prefer toy model for quick iteration.
- Reproducibility depends on fixed seeds in prompt generation and deterministic config choices.

---

## 11) Extension boundaries (what to do first vs later)

Prioritize first:

- Stability of patch effects.
- Graph comparability across slices.
- Classical baseline robustness.

Expand later:

- Finer node granularity (heads/MLP subcomponents).
- Multi-site/path patching.
- Large-model scaling before small-model pipeline is fully stable.

---

## 12) Fast command cheat sheet

```bash
# Install + verify CLI
uv sync --dev
uv run pig --help

# Full pipeline (fast model)
uv run pig pipeline --model-name toy_transformer

# Full pipeline (gpt2)
uv run pig pipeline --model-name gpt2

# Continue even if one script fails
uv run pig pipeline --continue-on-error

# Generate artifacts directly
uv run python scripts/generate_pipeline_artifacts.py

# Memory comparison utility
uv run python scripts/compare_model_memory.py --model-a gpt2 --model-b toy_transformer --device cpu

# Verification run
uv run pig pipeline --model-name toy_transformer
```

---

## 13) Definition of done for agent contributions

A contribution is complete when:

- The intended behavior is implemented and validated.
- Pipeline still runs for toy model.
- Documentation is updated if interfaces/commands changed.
- No unrelated refactors or scope creep were introduced.
