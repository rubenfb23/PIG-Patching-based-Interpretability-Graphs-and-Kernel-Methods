#!/usr/bin/env bash
# ===========================================================================
# CLMI — Serious experiment run for 4× RTX 3090 (24 GB each)
#
# Grid:
#   seeds       = 5
#   overlaps    = 0.0, 0.25, 0.5, 0.75
#   pairs/ov    = 6
#   ft_modes    = full, lora
#   mitigations = none, freeze_nc, anchor_reg
#   protocols   = ABA, BAB          (both directions for NC)
#
# Total runs  = 5 × 4 × 6 × 2 × 3 × 2 = 1440
# GPUs        = 4 (parallel)  → ~360 runs per GPU
# ETA         ≈ 360 × 4 min / 60 ≈ 24 h
#
# Training params (tuned for convergence on GPT-2 key-value tasks):
#   steps   = 500 per phase (A, B, A2)
#   batch   = 32  (3090 has 24 GB, GPT-2 fits easily with bs=32)
#   lr      = 3e-5 (slightly lower than default for stability)
#   eval    = every 20 steps → 25 eval points per phase (smooth curves)
#   n-keys  = 400, n-values = 400 (default, decent difficulty)
#   k       = 16  (number of key-value mappings per task)
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# Activate venv
if [[ -d .venv ]]; then
    source .venv/bin/activate
fi

# Ensure results dir
mkdir -p results/tables results/figures results/runs

# Remove stale summary to start fresh (comment out if using --resume)
# rm -f results/tables/summary.csv

LOG="results/run_serious_$(date +%Y%m%d_%H%M%S).log"

echo "=== CLMI Serious Run ==="
echo "Log: $LOG"
echo "GPUs: 4"
echo "Total runs: 1440"
echo "Estimated time: ~24 hours"
echo ""

python scripts/run_experiments.py \
    --device cuda \
    --n-gpus 4 \
    --seeds 5 \
    --overlaps 0.0 0.25 0.5 0.75 \
    --n_pairs_per_overlap 6 \
    --ft-modes full lora \
    --mitigations none freeze_nc anchor_reg \
    --protocols ABA BAB \
    --n-keys 400 \
    --n-values 400 \
    --batch-size 32 \
    --learning-rate 3e-5 \
    --max-steps-a 500 \
    --max-steps-b 500 \
    --max-steps-a2 500 \
    --eval-every 20 \
    --continue-on-error \
    --resume \
    2>&1 | tee "$LOG"

echo ""
echo "=== Done. Results in results/tables/summary.csv ==="
echo "Log saved to: $LOG"
