# PIG - Patching-based Interpretability Graphs (PIG)

1) generate a large *interventional* dataset via activation patching,
2) summarize patch effects as a *dataset of sparse graphs* (one graph per “slice” of prompts/corruptions), and
3) compare slices using *kernel methods* (classical and quantum) to induce a similarity geometry over circuits.

## What the deck contains

The slides in [docs/patching_graphs_v2.pdf](docs/patching_graphs_v2.pdf) cover:

- **Problem setting (Mechanistic Interpretability)**: framed as causal structure discovery inside a transformer.
- **Patching (Interventions → effect maps)**: define clean vs corrupted runs and an effect size per patched node.
- **Theoretical formulation**: define the node set (by layer/token/component/head), “slices”, and the resulting graph dataset.
- **Algorithms**:
  - *Algorithm 1*: compute patch effects and build one sparse weighted directed graph per slice.
  - *Algorithm 2*: compute classical kernels on graph embeddings (e.g., WL/spectral/graphlets) and run downstream learning.
  - *Algorithm 3*: “swap” the similarity layer for a quantum fidelity kernel on the same embeddings.
- **Execution plan**: a task-by-task implementation plan with acceptance tests and minimal code skeletons.
- **Evaluation checklist**: suggested plots/ablations (noise, sparsity, shots, depth, spectra, clustering viz).

## Core concepts (as used in the deck)

- **Paired inputs**: for each example $i$, create $(x_i^{\mathrm{cln}}, x_i^{\mathrm{crp}})$ where the corruption breaks the target behavior.
- **Observable** $O(\cdot)$: a scalar readout such as a target logit $\ell_{y^\star}$ or negative log-prob/loss.
- **Node** $u$: an internal intervention point indexed by layer, token position, component type (res/att/mlp), and (optionally) head.
- **Patch effect**: $E_u^{(i)} = O(\tilde f_\theta(x_i^{\mathrm{crp}};u)) - O(f_\theta(x_i^{\mathrm{crp}}))$.
- **Slice** $s$: a group of examples (task family / corruption type / prompt template family / difficulty bin).
- **Graph per slice**: $G_s = (\mathcal{V}, \mathcal{E}_s, w_s)$ with a fixed node set $\mathcal{V}$ shared across slices.
  - Edges use a co-variation rule over patch effects within a slice (e.g., correlation between $\{E_u^{(i)}\}$ and $\{E_v^{(i)}\}$).
  - Sparsification: keep top-$k$ outgoing edges per node by $|w_s(u\to v)|$.

## Repository layout

- [docs/](docs/)
  - [docs/patching_graphs_v2.pdf](docs/patching_graphs_v2.pdf): compiled slide deck
  - [docs/patching_graphs_v2.tex](docs/patching_graphs_v2.tex): LaTeX source
- [LICENSE](LICENSE)

## Notes on implementation status

This repository currently does not ship a reference Python package or CLI; the code blocks in the deck are templates/skeletons intended to guide an implementation (e.g., with PyTorch + HuggingFace for patching; NumPy for graph building; scikit-learn for SVM/KRR; optional PennyLane for quantum kernel simulation).

If you add an implementation later, a natural next step is to introduce a small `src/` (or `pig/`) package and a minimal “run end-to-end on a toy task” script aligned with the deck’s Task 0–Task 6 acceptance tests.

## (Optional) Rebuilding the PDF

The PDF is already included in [docs/patching_graphs_v2.pdf](docs/patching_graphs_v2.pdf). If you choose to rebuild locally, you’ll need a LaTeX distribution with Beamer/TikZ/algorithm2e.

Example commands:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode patching_graphs_v2.tex
```

## Citation

If you want to reference the current state of the project, cite the repository and/or link the slide deck:

- [docs/patching_graphs_v2.pdf](docs/patching_graphs_v2.pdf)

## License

See [LICENSE](LICENSE).
