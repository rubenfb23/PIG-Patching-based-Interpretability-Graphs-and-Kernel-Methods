# PIG: Patching-based Interpretability Graphs

This repo is for building an end-to-end pipeline that turns **mechanistic interpretability interventions (patching)** into a **graph dataset** and then applies **kernel methods** (classical and quantum) to learn a similarity geometry over “circuits”.

The core framing is:

1. **Mechanistic interpretability problem**: identify *which internal components causally influence* a behavior, and how they *compose into circuits*.
2. **Patching** provides causal evidence via interventions.
3. Aggregate patch effects into **graphs** (one graph per “slice” of data).
4. Apply **graph embeddings + kernels** (classical vs quantum) to compare slices and run downstream learning.

---

## What we are trying to build (high level)

An end-to-end system that produces:

- A **fixed local transformer** (frozen weights) that supports many forward passes and activation hooks.
- A **paired prompt generator** that outputs clean/corrupted prompt pairs plus a target token.
- A **patching engine** that runs interventions and produces a patch-effect tensor:
  - per example `i` and per internal node `u`: an effect value `E_u^(i)`.
- A **graph dataset** `{G_s}` where each graph corresponds to a slice `s` (task family / corruption type / prompt family):
  - Same node set across all graphs; edges/weights vary by slice.
  - Standardized sparsity via top-k outgoing edges per node.
- A **kernel learning layer** that computes similarities between graphs:
  - Classical kernels (e.g., WL/spectral/graphlets → linear/RBF).
  - Quantum “swap”: keep graph embeddings fixed, replace the kernel with a fidelity kernel.
- A minimal **evaluation suite**: classification/clustering + robustness/noise ablations.

Key design principle: **freeze complexity early**: one model, one task family, one patch type first; only expand once the pipeline is stable.

---

## Concepts and formal objects

### Model and observable

- Model: a transformer `f_θ` with fixed parameters.
- Prompt: `x = (x_1, …, x_T)` with fixed/controlled context length `T`.
- Observable `O(x)` is a scalar derived from the forward pass (choose one and standardize it):
  - Target-token logit: `O(x) = ℓ_{y*}(x)`
  - Or negative log-prob / loss: `O(x) = -log p_θ(y* | x)`

The observable must be defined unambiguously (tokenization matters) because it is what patching “restores”.

### Clean/corrupted prompt pairs

Each training/evaluation item `i` is:

- Clean prompt `x_i^cln`
- Corrupted prompt `x_i^crp` (designed to break behavior while preserving structure)
- Target token `y_i*` (the token we want the model to put high probability on)
- Slice label `s(i)` (task family / corruption type / prompt family / difficulty bin)

### Nodes (patch points)

A node `u` denotes a patchable internal component at a specific location:

- Layer index `ℓ ∈ {0,…,L-1}`
- Token position `t ∈ {1,…,T}` (often we start with `t = -1` for last token)
- Component kind `k ∈ {res, att, mlp}`
- Head index `h ∈ {1,…,H}` if `k = att`

The initial MVP is **residual stream patching** with nodes `u = (ℓ, t, res)`.

### Patch effect

For an example pair `(x^cln, x^crp)` and a node `u`, define the patched forward pass
`\tilde f_θ(x^crp; u)` as:

- run the corrupted forward pass, except at node `u` replace the activation with the corresponding clean activation.

Patch effect:

`E_u(x^cln, x^crp) = O(\tilde f_θ(x^crp; u)) - O(f_θ(x^crp))`

Interpretation: large positive `E_u` means patching `u` causally restores the desired behavior.

### Slices and graphs

Define slice indices `s ∈ {1,…,S}` by grouping examples (e.g., `task=IOI`, `corruption=swap`).

For each slice `s`, build a directed weighted graph:

- `G_s = (V, E_s, w_s)`
- Node set `V` is fixed across slices (same patch points).
- Edge weights summarize how patch effects co-vary across examples in that slice.

Recommended default edge construction (co-influence):

- For each node `u`, collect the effect profile across examples in slice `s`: `{E_u^(i)}_{i ∈ I_s}`
- Set edge weight (often absolute value is used for sparsification):

`w_s(u → v) = corr( {E_u^(i)}, {E_v^(i)} )`

Optional but recommended constraint:

- Enforce a partial order `u ≺ v` (e.g., earlier layer/token to later) so edges only go forward.

Critical standardization:

- Top-k sparsification: for each `u`, keep only the k outgoing edges with largest `|w_s(u→v)|`.

This yields comparable graph sizes and stabilizes kernels.

---

## Pipeline (execution plan turned into implementation tasks)

This section is written so a new contributor/agent can implement one stage at a time with a clear “done” condition.

### Task 0 — Choose and validate a local model

**Goal**: pick a small model that supports many forward passes and reliable activation hooks.

**MVP choice**: a small HuggingFace model (e.g., GPT-2 sized) run locally, frozen weights.

**Acceptance test**:

- Able to run on the machine (CPU/GPU).
- Can run ~10^4 forward passes in a reasonable session (hardware dependent, but should not be “minutes per forward”).
- Can capture a residual activation at layer `ℓ` and replace it during a corrupted run.
- Observed `Δlogit` (or chosen observable delta) changes non-trivially for at least some `(ℓ,t)`.

### Task 1 — Controlled task family + corruption generator

**Goal**: generate many paired prompts where clean succeeds and corrupted fails.

**MVP task**: IOI-style indirect object identification with a simple swap corruption.

**Outputs per example**:

- `x_cln`, `x_crp`, `y*`, `slice_label`, `meta`

**Acceptance test**:

- Average gap `E[ O(x^cln) - O(x^crp) ] > 0`.
- Generator is reproducible with a fixed seed.

### Task 2 — Implement patching hooks (start with residual stream)

**Goal**: implement `\tilde f_θ(x^crp; u)` for one node type.

**MVP patch type**: `u = (ℓ, t, res)` residual stream patch.

**API shape (suggested)**:

- `cache_clean(prompt_cln) -> cache_cln`
- `score(prompt, y*) -> O(prompt)`
- `patched_score(prompt_crp, y*, cache_cln, u) -> O(\tilde f_θ)`

**Acceptance test**:

- For some `(ℓ,t)`, effect `E_u > 0` for the corrupted prompt.
- Random `(ℓ,t)` gives effects concentrated near 0 on average.

### Task 3 — Compute patch-effect tensor `E_u^(i)` (raw interventional dataset)

**Goal**: for each example `i` and each node `u` in a chosen node set, compute `E_u^(i)`.

**MVP node set**:

- Residual-only nodes on a small subset of layers `L0` and token positions `T0`.

**Storage**:

- Pilot: dense `E^(i)` matrix for small `|L0|×|T0|`.
- Scale: sparse-friendly format (triplets or chunked arrays), compression.

**Acceptance test**:

- Per-example heatmaps show structured hotspots, not uniform noise.
- Effects differ across slices enough to support downstream prediction.

### Task 4 — Build per-slice graphs `{G_s}` from effects

**Goal**: aggregate effects into a sparse directed weighted graph per slice.

**MVP graph recipe**:

- Node list `V0 = {(ℓ,t,res) : ℓ∈L0, t∈T0}`
- For each slice `s`, build effect vectors across examples and compute correlations.
- Apply direction constraint `u ≺ v` by `(ℓ,t)` ordering.
- Apply top-k outgoing sparsification.

**Graph storage (minimal, kernel-friendly)**:

- Fixed node list (implicit indices)
- Edge list: `(u_idx, v_idx, weight)`
- Optional node attributes: `type`, `layer`, `token` (useful for WL)

**Acceptance test**:

- All slices share identical node set and comparable edge budgets (`≤ k|V|`).
- Graph statistics differ across slices (signal).

### Task 5 — Classical baselines (must be strong first)

**Goal**: get a strong classical baseline using graph embeddings + classical kernels.

**MVP baseline**:

- WL subtree features → linear kernel SVM (optionally RBF afterwards).

**Acceptance test**:

- Above-chance classification of slice labels.
- Learning curve: accuracy vs number of graphs behaves sensibly.

### Task 6 — Quantum kernel swap

**Goal**: keep embeddings fixed, replace the similarity kernel with a quantum fidelity kernel.

**MVP approach**:

- Use WL feature vectors `X ∈ R^{S×d}` from Task 5.
- Reduce to `n_q` dimensions via PCA / random projection: `x_s → z_s ∈ R^{n_q}`.
- Quantum feature map circuit `U_φ(z)` with small depth `D`.
- Estimate fidelity kernel entries with `M` shots.
- Train the same downstream model using a precomputed kernel matrix.

**Acceptance test**:

- End-to-end pipeline runs and produces non-trivial results.
- Performance varies sensibly with `M` (shot noise) and `D` (expressivity).

Minimum sweeps to report:

- `M ∈ {200, 500, 2000, 5000}`
- `D ∈ {1, 2, 3}`
- Compare against classical linear/RBF on same splits.

### Task 7 — Robustness and ablations

**Goal**: quantify sensitivity of conclusions to noise and design choices.

Minimum ablations:

- Patch-noise: add noise to `E_u^(i)` (e.g., σ in `{0, 0.05, 0.1, 0.2}` relative scale)
- Data ablation: number of examples per slice `{10, 25, 50, 100}`
- Graph ablation: `k ∈ {1,3,5,10}` and direction constraint on/off
- Quantum ablation: shots `M` and depth `D`

**Acceptance test**:

- Qualitative conclusions are stable across reasonable ranges.
- Quantum/classical differences (if any) appear in specific regimes rather than randomly.

### Task 8 — Packaging + reproducibility (“minimum publishable result”)

**Goal**: make the pipeline reproducible and cache-heavy (because patching is expensive).

Caching layers:

1. Patch effects
2. Graphs
3. Embeddings
4. Kernel matrices

Minimum publishable result (MPR):

- A dataset of patch-influence graphs for a controlled task family.
- A rigorous classical baseline comparison.
- A characterization of when quantum kernels behave differently (sample efficiency / robustness / noise sensitivity).

---

## Interfaces and data artifacts (what files/objects should exist)

Even if the exact code layout changes, we should keep these artifact boundaries stable.

### Artifact: paired prompt dataset

A collection of records:

- `x_cln: str`
- `x_crp: str`
- `y_star: str` (or token id, but standardize)
- `slice: dict` (e.g. `{task:"IOI", corruption:"swap"}`)
- `meta: dict` (names, template parameters)

### Artifact: patch-effect tensor

For each example `i` and node `u`:

- `E_u^(i): float`

MVP representation for residual-only pilot:

- `E^(i)` as matrix shaped `[len(L0), len(T0)]`.

### Artifact: graph dataset

For each slice key `s`:

- Node list `V` (fixed global ordering)
- Edge list `E_s`: list of `(u_idx, v_idx, weight)`
- Node attributes: at least `type`, optionally `layer`, `token`, `head`

### Artifact: embeddings and kernels

- `X`: graph embeddings / feature vectors (`S × d`)
- `K_classical`: kernel matrix (`S × S`)
- `K_quantum`: kernel matrix (`S × S`)

---

## Evaluation checklist (the minimum questions)

- Can we classify/cluster graphs by **task family** or **corruption type**?
- Do kernel choices change **sample efficiency** (accuracy vs #graphs) or **robustness** (noise in patch effects)?
- How do **quantum kernel depth** and **shots** trade off bias/variance?

Minimum figures:

- Accuracy/AUC vs number of graphs
- Accuracy vs patch noise σ
- Accuracy vs graph sparsity k (+ direction constraint toggle)
- Quantum: accuracy vs shots M for each depth D
- Kernel spectrum / eigen-decay comparison (classical vs quantum)
- A 2D visualization (kernel PCA) for classical vs quantum

---

## Non-goals (explicitly out of scope for the MVP)

These are described as extensions in the deck, but are not required for the first stable pipeline:

- Finer node granularity (attention heads and MLPs) beyond residual-only patching.
- Path-level / multi-site patching (paired interventions) beyond single-node patches.
- Alternative graph constructions beyond co-influence correlation (e.g., direct influence mediation edges).
- Scaling to large models before the pipeline is validated on a small one.

---

## Contributor notes (practical pitfalls)

- **Tokenization**: `y*` must be defined in token-space; string-to-token can be ambiguous.
- **Hook points** are architecture-dependent; start with a stable tap (e.g., block output) and document it.
- **Cost control**: patching is expensive; always cache clean activations and intermediate results.
- **Comparability**: keep node set and sparsification rules fixed across slices.
- **Reproducibility**: fix seeds, version prompt templates, record model hash, log configs.

## Suggested repo structure

This project has a clear pipeline shape (prompt pairs -> patch effects -> per-slice graphs -> embeddings/kernels -> evaluation). For a public repo, the best default is to mirror that pipeline in the package layout, make cache boundaries explicit, and keep the runnable entrypoints thin.

Recommended structure (Python-first, pip-installable package):

```text
.
|-- README.md
|-- LICENSE
|-- AGENTS.md
|-- pyproject.toml
|-- src/
|       |-- __init__.py
|       |-- config/                # typed config schema + defaults
|       |-- prompts/               # templates, corruptions, slice grouping
|       |-- model/                 # HF loader, tokenization, observable O(x), hooks
|       |-- patching/              # cache clean activations + patch runs (start: residual)
|       |-- effects/               # compute + store E_u^(i)
|       |-- graphs/                # slice aggregation, corr weights, top-k sparsify
|       |-- embeddings/            # WL baseline first; spectral/graphlets later
|       |-- kernels/               # classical kernels + quantum fidelity kernel swap
|       |-- eval/                  # splits, metrics, plots, experiment runner
|       `-- cli.py                 # optional: one CLI entrypoint (or use scripts/)
|-- scripts/                       # thin stage entrypoints (Task0..Task7)
|-- configs/                       # reproducible experiment YAMLs
|-- docs/                          # latex deck + paper/notes
|-- tests/                         # deterministic unit tests (no GPU required)
|-- examples/                      # tiny, committed example artifacts for quickstart
|   |-- sample_pairs.jsonl
|   `-- sample_graphs.json
|-- data/                          # gitignored: generated prompt pairs + patch effects
|   `-- README.md
`-- artifacts/                     # gitignored: graphs/features/kernels/plots
    `-- README.md
```

Why this is optimized for a public repo:

- Mirrors the scientific story: contributors can navigate by pipeline stage.
- Keeps expensive outputs out of git by default: `data/` (inputs/effects) vs `artifacts/` (graphs/features/kernels/plots).
- Makes reproducibility straightforward: `configs/` define runs; `scripts/` (or `pig` CLI) execute them.
- Keeps CI fast and reliable: unit tests cover slicing/top-k/WL vectorization without depending on model downloads or GPUs.

Public-repo hygiene defaults (recommended):

- Add `data/` and `artifacts/` to `.gitignore`; commit only small `examples/` artifacts.
- Provide one canonical entrypoint for each stage (either `scripts/taskX_*.py` or a single `pig` CLI with subcommands).
- Keep a single environment definition (`pyproject.toml`) and enforce formatting/linting in CI.
