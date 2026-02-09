# PIG Implementation Plan

This document outlines the implementation plan for the Patching-based Interpretability Graphs (PIG) project. Each task is written as a user story for an external developer.

---

## Phase 1: Foundation

### Story 1.1 — Model Setup and Validation

**As a** developer,
**I want to** set up a local transformer model with activation hooks,
**So that** I can run interventions and capture internal activations for patching experiments.

**Acceptance Criteria:**

- [ ] A small HuggingFace model (e.g., GPT-2) runs locally with frozen weights
- [ ] The system can execute ~10,000 forward passes in a reasonable time
- [ ] Residual activations can be captured at any layer `ℓ` and token position `t`
- [ ] Activations can be replaced during a forward pass (hook injection works)
- [ ] Observable delta (Δlogit) changes non-trivially when patching some `(ℓ, t)` positions

**Technical Notes:**

- Use PyTorch + HuggingFace Transformers
- The observable `O(x)` should be the target-token logit `ℓ_{y*}(x)`

---

### Story 1.2 — Paired Prompt Generator

**As a** developer,
**I want to** create a generator that produces clean/corrupted prompt pairs,
**So that** I have controlled inputs where the clean prompt succeeds and the corrupted prompt fails at a target behavior.

**Acceptance Criteria:**

- [ ] Generator outputs tuples: `(x_cln, x_crp, y_star, slice_label, meta)`
- [ ] Average observable gap `E[O(x^cln) - O(x^crp)] > 0` across generated pairs
- [ ] Generator is reproducible with a fixed random seed
- [ ] Supports at least one task family (MVP: IOI-style indirect object identification)
- [ ] Supports at least one corruption type (MVP: name swap)

**Output Schema:**

```python
{
    "x_cln": str,        # Clean prompt
    "x_crp": str,        # Corrupted prompt
    "y_star": str,       # Target token
    "slice": {           # Slice metadata
        "task": str,
        "corruption": str
    },
    "meta": dict         # Additional metadata (names, template params)
}
```

---

## Phase 2: Patching Engine

### Story 2.1 — Residual Stream Patching Hooks

**As a** developer,
**I want to** implement patching hooks that replace activations during corrupted runs,
**So that** I can measure the causal effect of each internal node on model behavior.

**Acceptance Criteria:**

- [ ] Implement `cache_clean(prompt_cln) -> cache_cln` to store clean activations
- [ ] Implement `score(prompt, y_star) -> O(prompt)` to compute the observable
- [ ] Implement `patched_score(prompt_crp, y_star, cache_cln, node_u) -> O(patched)`
- [ ] For some `(ℓ, t)` positions, patch effect `E_u > 0` on corrupted prompts
- [ ] Random `(ℓ, t)` positions show effects concentrated near zero on average

**Technical Notes:**

- MVP patch type: `u = (ℓ, t, res)` for residual stream patching
- Patch effect formula: `E_u = O(patched_forward(x^crp; u)) - O(forward(x^crp))`

---

### Story 2.2 — Patch-Effect Tensor Computation

**As a** developer,
**I want to** compute and store a patch-effect tensor `E_u^(i)` for all examples and nodes,
**So that** I have the raw interventional dataset needed for graph construction.

**Acceptance Criteria:**

- [ ] For each example `i` and node `u`, compute and store `E_u^(i)`
- [ ] MVP storage: dense matrix `E^(i)` shaped `[num_layers, num_tokens]`
- [ ] Per-example heatmaps show structured hotspots (not uniform noise)
- [ ] Effects differ visibly across different slices
- [ ] Caching is implemented to avoid redundant computation

**Storage Format:**

- Pilot: dense NumPy arrays
- Scale: sparse-friendly format (triplets or chunked arrays with compression)

---

## Phase 3: Graph Construction

### Story 3.1 — Per-Slice Graph Builder

**As a** developer,
**I want to** aggregate patch effects into sparse directed weighted graphs per slice,
**So that** I can represent circuit structure in a kernel-friendly format.

**Acceptance Criteria:**

- [ ] Node set `V` is fixed across all slices (same patch points)
- [ ] Edge weights are computed via correlation of effect profiles within each slice
- [ ] Direction constraint enforced: edges only go from earlier `(ℓ, t)` to later
- [ ] Top-k sparsification applied: keep only k outgoing edges per node
- [ ] All slices have comparable edge budgets (`≤ k × |V|`)
- [ ] Graph statistics differ across slices (verifiable signal)

**Graph Schema:**

```python
{
    "nodes": [...],                    # Fixed global ordering
    "edges": [                         # Per-slice edge list
        {"src": int, "dst": int, "weight": float},
        ...
    ],
    "node_attrs": {                    # Optional attributes
        "type": str,                   # "res", "att", "mlp"
        "layer": int,
        "token": int
    }
}
```

---

## Phase 4: Classical Kernels

### Story 4.1 — Graph Embeddings (WL Features)

**As a** developer,
**I want to** compute Weisfeiler-Lehman (WL) subtree feature vectors for each graph,
**So that** I have fixed-dimensional embeddings suitable for kernel methods.

**Acceptance Criteria:**

- [ ] Implement WL subtree hashing with configurable depth
- [ ] Output: feature matrix `X` of shape `[num_slices, num_features]`
- [ ] Features are deterministic given the same input graphs
- [ ] Embedding computation is cached for reuse

---

### Story 4.2 — Classical Kernel Baseline

**As a** developer,
**I want to** train an SVM classifier using classical kernels on graph embeddings,
**So that** I have a strong baseline for comparing against quantum kernels.

**Acceptance Criteria:**

- [ ] Implement linear kernel SVM on WL features
- [ ] Implement RBF kernel SVM on WL features
- [ ] Classification accuracy is above chance for slice label prediction
- [ ] Learning curve (accuracy vs. number of graphs) behaves sensibly
- [ ] Results are reproducible with fixed random seeds

---

## Phase 5: Quantum Kernels

### Story 5.1 — Quantum Feature Map Circuit

**As a** developer,
**I want to** implement a quantum feature map circuit that encodes graph embeddings,
**So that** I can compute quantum fidelity kernels for comparison.

**Acceptance Criteria:**

- [ ] Dimensionality reduction: PCA or random projection to `n_q` qubits
- [ ] Implement parameterized circuit `U_φ(z)` with configurable depth `D`
- [ ] Circuit runs on a quantum simulator (PennyLane recommended)
- [ ] Fidelity estimation works with configurable number of shots `M`

**Technical Notes:**

- Input: WL feature vectors from Story 4.1
- Output: quantum state preparation circuit

---

### Story 5.2 — Quantum Fidelity Kernel

**As a** developer,
**I want to** compute a quantum fidelity kernel matrix from the quantum feature map,
**So that** I can train downstream models and compare against classical kernels.

**Acceptance Criteria:**

- [ ] Compute kernel matrix `K_quantum` of shape `[num_slices, num_slices]`
- [ ] Train SVM using precomputed quantum kernel matrix
- [ ] End-to-end pipeline runs and produces non-trivial results
- [ ] Performance varies sensibly with shots `M` (shot noise effect visible)
- [ ] Performance varies sensibly with depth `D` (expressivity effect visible)

**Required Sweeps:**

- Shots: `M ∈ {200, 500, 2000, 5000}`
- Depth: `D ∈ {1, 2, 3}`

---

## Phase 6: Evaluation and Ablations

### Story 6.1 — Robustness Ablations

**As a** developer,
**I want to** run ablation studies on key hyperparameters and noise levels,
**So that** I can characterize the stability of results and identify regime differences.

**Acceptance Criteria:**

- [ ] Patch-noise ablation: add noise `σ ∈ {0, 0.05, 0.1, 0.2}` to effects
- [ ] Data ablation: vary examples per slice `{10, 25, 50, 100}`
- [ ] Graph ablation: vary sparsity `k ∈ {1, 3, 5, 10}` and direction constraint on/off
- [ ] Quantum ablation: sweep shots `M` and depth `D`
- [ ] Qualitative conclusions are stable across reasonable parameter ranges
- [ ] Quantum/classical differences appear in specific regimes (not randomly)

---

### Story 6.2 — Evaluation Figures and Visualizations

**As a** developer,
**I want to** generate evaluation figures that answer the core research questions,
**So that** results can be communicated clearly in publications.

**Required Figures:**

- [ ] Accuracy/AUC vs. number of graphs (learning curve)
- [ ] Accuracy vs. patch noise `σ` (robustness)
- [ ] Accuracy vs. graph sparsity `k` with direction constraint toggle
- [ ] Quantum: accuracy vs. shots `M` for each depth `D`
- [ ] Kernel spectrum / eigen-decay comparison (classical vs. quantum)
- [ ] 2D visualization (kernel PCA) for classical vs. quantum

---

## Phase 7: Packaging and Reproducibility

### Story 7.1 — Caching Infrastructure

**As a** developer,
**I want to** implement a caching layer for all expensive computations,
**So that** experiments are fast to iterate and fully reproducible.

**Caching Layers:**

- [ ] Patch effects: cache `E_u^(i)` tensors
- [ ] Graphs: cache per-slice graph structures
- [ ] Embeddings: cache WL feature vectors
- [ ] Kernel matrices: cache classical and quantum kernels

---

### Story 7.2 — CLI and Configuration System

**As a** developer,
**I want to** implement a CLI with typed configuration,
**So that** experiments can be run reproducibly from the command line.

**Acceptance Criteria:**

- [ ] Single CLI entrypoint with subcommands for each pipeline stage
- [ ] Configuration via YAML files in `configs/` directory
- [ ] All random seeds are configurable and logged
- [ ] Model version/hash is recorded in outputs
- [ ] Prompt templates are versioned

---

### Story 7.3 — Test Suite

**As a** developer,
**I want to** implement a test suite that validates core functionality,
**So that** changes can be verified without GPU or model downloads.

**Acceptance Criteria:**

- [ ] Unit tests for prompt generation logic
- [ ] Unit tests for graph construction (top-k, direction constraint)
- [ ] Unit tests for WL feature extraction
- [ ] Tests are deterministic and run on CPU only
- [ ] CI pipeline runs tests on every commit

---

## Appendix: Out of Scope for MVP

The following are explicitly deferred to future iterations:

- Finer node granularity (attention heads and MLPs) beyond residual-only patching
- Path-level / multi-site patching (paired interventions)
- Alternative graph constructions beyond co-influence correlation
- Scaling to large models before the pipeline is validated on a small one

---

## Dependency Graph

```
Story 1.1 (Model Setup)
    ↓
Story 1.2 (Prompt Generator)
    ↓
Story 2.1 (Patching Hooks)
    ↓
Story 2.2 (Patch-Effect Tensor)
    ↓
Story 3.1 (Graph Builder)
    ↓
Story 4.1 (WL Embeddings) ────────→ Story 5.1 (Quantum Circuit)
    ↓                                      ↓
Story 4.2 (Classical Baseline) ←──── Story 5.2 (Quantum Kernel)
    ↓                                      ↓
    └──────────────→ Story 6.1 (Ablations) ←───────┘
                            ↓
                    Story 6.2 (Figures)
                            ↓
    ┌───────────────────────┴───────────────────────┐
    ↓                       ↓                       ↓
Story 7.1 (Caching)   Story 7.2 (CLI)        Story 7.3 (Tests)
```

---

## Minimum Publishable Result (MPR)

The MVP is complete when the following are delivered:

1. A dataset of patch-influence graphs for a controlled task family (IOI)
2. A rigorous classical baseline comparison (WL + SVM)
3. A characterization of when quantum kernels behave differently (sample efficiency / robustness / noise sensitivity)
4. Reproducible experiment configs and cached artifacts
