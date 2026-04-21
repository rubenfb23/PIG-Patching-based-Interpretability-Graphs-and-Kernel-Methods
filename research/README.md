# PIG Research Directory

This directory contains deep research on the PIG (Patching-based Interpretability Graphs) pipeline,
based on analysis of the `audit/` folder and the codebase.

## Files

| File | Description |
|------|-------------|
| [audit_analysis.md](audit_analysis.md) | Comprehensive analysis of the audit output, what's implemented vs missing, kernel comparison, and research roadmap |
| [implementation_plan.md](implementation_plan.md) | Concrete implementation plans for Direct-Influence, Partial Correlation, Activation Flow, and other graph builders |
| [literature_review.md](literature_review.md) | Review of the key papers that form the foundation of the PIG pipeline |

## Key Findings

1. **Stage 5' (Direct-Influence) and 5'' (Partial Correlation) are NOT implemented**
2. **All three kernels (linear, RBF, quantum) give 87.5% accuracy** on the toy dataset — no differentiation
3. **The dataset is too small** (N=8 examples) for meaningful statistical analysis
4. **Correlation-based graph builders measure co-variation, not causation** — Direct-Influence would be better
5. **Quantum kernel uses only 4 qubits** — the Hilbert space is too small to show advantage

## Pipeline Overview

```
Stage 1: Prompts (x_cln, x_crp, y_star)
Stage 2: Baseline observable (corrupted forward)
Stage 3: Patched observable (patched forward)
Stage 4: Patch effects (E = O_patch - O_base)
Stage 5: Correlation matrix (co-influence)
Stage 6: Sparse graph (top-k + direction)
Stage 7: WL features (graph embeddings)
Stage 8: Kernels (linear, RBF, quantum)
Stage 9: SVM classification
```

## Priorities for Next Work

1. Implement Direct-Influence graph builder
2. Implement Partial Correlation graph builder
3. Scale dataset from 8 to 50+ examples
4. Compare graph builders via causal validation
5. Add WL kernel for direct graph comparison
