# Paper Data Audit — 2026-05-02

Scope: `docs/paper/main.tex` numerical claims after checking provenance against
local `outputs/` and the experiment runners.

## Keep as primary results

These values are traceable to `scripts/run_paper_decision_study.py`, which splits
prompt pairs before graph bootstrapping.

Source: `outputs/paper_decision/gpt2_n100_res_k5_spectral_graphlets_20260430/summary.csv`

| Metric | Values by seed | Mean | Pop. std |
|---|---:|---:|---:|
| WL bootstrap linear | 0.7969, 0.5469, 0.9531 | 0.7656 | 0.1673 |
| Fixed layout weighted linear | 0.9531, 0.6562, 1.0000 | 0.8698 | 0.1522 |
| Fixed layout binary linear | 0.9844, 0.8281, 0.9844 | 0.9323 | 0.0737 |
| Fixed layout signed linear | 1.0000, 0.9688, 1.0000 | 0.9896 | 0.0147 |
| Spectral shape linear | 0.5938, 0.5469, 0.6719 | 0.6042 | 0.0516 |
| Graphlet shape linear | 0.5469, 0.6094, 0.7500 | 0.6354 | 0.0849 |

Source: `outputs/paper_decision/gpt2_n100_res_k5_scalable_representations_20260430/summary.csv`

| Metric | Values by seed | Mean | Pop. std |
|---|---:|---:|---:|
| Hashed fixed-layout signed | 0.9688, 0.9531, 0.8906 | 0.9375 | 0.0338 |
| Hashed fixed-layout weighted | 0.8750, 0.9531, 0.9062 | 0.9115 | 0.0321 |
| Coarse count | 0.9219, 0.9375, 0.9219 | 0.9271 | 0.0074 |

Source: `outputs/paper_decision/gpt2_n500_res_k5_scalable_representations_20260430/summary.csv`

| Metric | Mean |
|---|---:|
| Fixed layout signed | 1.0000 |
| Hashed fixed-layout signed | 1.0000 |
| Hashed fixed-layout weighted | 0.9583 |
| Coarse count | 0.9740 |
| WL bootstrap linear | 1.0000 |

## Corrected or removed from the paper

- Removed exact raw patch-effect and top-k node diagnostic numbers from the main
  narrative. They came from per-example diagnostics, not the graph bootstrap
  protocol.
- Removed `global_histogram` from the results table because no strict
  `paper_decision` provenance was found.
- Corrected scalable-row standard deviations:
  - hashed signed: `0.0147 -> 0.0338`
  - hashed weighted: `0.0237 -> 0.0321`
  - coarse count: `0.0147 -> 0.0074`
- Removed sign-shuffle, node-label-shuffle, and label-permutation rows from the
  strict null-control table. The strict runner currently reports edge-shuffle
  and weight-shuffle for the fixed-layout weighted baseline.
- Removed the numeric pilot causal table. The exact values in the draft were
  not traceable to the current GPT-2 `paper_decision` outputs.

## Follow-up experiment to run before claiming raw baselines

`scripts/run_paper_decision_study.py` now includes strict train/test metrics for
`patch_effect_node_vector` and `topk_node_identity`. Re-run the GPT-2 n=100
paper-decision study from cache or full patching output before reporting those
numbers in the paper.
