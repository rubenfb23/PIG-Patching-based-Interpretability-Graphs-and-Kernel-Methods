#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run teacher distillation (GSM8K) + student GPT-2 finetuning, sequentially.

Default behavior uses the FULL dataset (max-examples = -1).
If `--distilled-path` already exists and is non-empty, distillation is skipped
and the script goes directly to finetuning.

Usage:
  src/gpt2/run_distill_and_finetune.sh [options]

Options:
  --teacher-model <name>       Teacher model (default: openai/gpt-oss-20b)
  --student-model <name>       Student model (default: gpt2)
  --distilled-path <path>      Output JSONL for distilled data
  --student-output-dir <path>  Output directory for finetuned model
  --batch-size <int>           Teacher generation batch size (default: 24)
  --student-batch-size <int>   Student micro-batch size per GPU (default: 2)
  --student-grad-accum-steps <int> Student gradient accumulation steps (default: 8)
  --student-num-workers <int>  Student DataLoader workers per process (default: 2)
  --student-lr <float>         Student learning rate (default: 5e-5)
  --student-warmup-ratio <float> Student LR warmup ratio (default: 0.03)
  --student-weight-decay <float> Student weight decay (default: 0.1)
  --student-max-grad-norm <float> Student grad clip norm (default: 1.0)
  --dtype <auto|bf16|fp16|fp32> Teacher dtype (default: bf16)
  --attn-implementation <str>  Teacher attention implementation (default: eager)
  --max-gpu-memory-gib <float> Per-GPU cap for teacher load (default: 22)
  --gpu-memory-reserve-gib <float> Reserve per GPU when auto cap (default: 2)
  --cpu-offload-gib <float>    CPU RAM budget for offload (default: 0)
  --offload-folder <path>      Offload folder (default: outputs/offload/gpt_oss_teacher)
  --mxfp4-dequantize | --no-mxfp4-dequantize
                               GPT-OSS dequantize mode (default: enabled)
  --max-examples <int>         Distillation sample cap; -1 means full dataset (default: -1)
  --seq-len <int>              Student training sequence length (default: 512)
  --epochs <int>               Student training epochs (default: 8)
  --nproc-per-node <int>       GPUs/processes for torchrun (default: auto-detect)
  --force-distill              Force re-running distillation even if distilled data exists
  --extra-distill-args "<str>" Extra args passed to gpt2.distill_gsm8k
  --extra-train-args "<str>"   Extra args passed to gpt2.finetuning
  -h, --help                   Show this help

Examples:
  src/gpt2/run_distill_and_finetune.sh
  src/gpt2/run_distill_and_finetune.sh --max-examples 2000
EOF
}

TEACHER_MODEL="openai/gpt-oss-20b"
STUDENT_MODEL="gpt2"
DISTILLED_PATH="outputs/gsm8k_distilled_gptoss20b.jsonl"
STUDENT_OUTPUT_DIR="outputs/gpt2_gsm8k_distilled"
BATCH_SIZE=24
STUDENT_BATCH_SIZE=4
STUDENT_GRAD_ACCUM_STEPS=6
STUDENT_NUM_WORKERS=3
STUDENT_LR="5e-5"
STUDENT_WARMUP_RATIO="0.03"
STUDENT_WEIGHT_DECAY="0.1"
STUDENT_MAX_GRAD_NORM="1.0"
DTYPE="bf16"
ATTN_IMPLEMENTATION="eager"
MAX_GPU_MEMORY_GIB=22
GPU_MEMORY_RESERVE_GIB=2
CPU_OFFLOAD_GIB=0
OFFLOAD_FOLDER="outputs/offload/gpt_oss_teacher"
MXFP4_DEQUANTIZE="--mxfp4-dequantize"
MAX_EXAMPLES=-1
SEQ_LEN=512
EPOCHS=8
NPROC_PER_NODE=""
FORCE_DISTILL=0
EXTRA_DISTILL_ARGS="--max-new-tokens 192 --log-interval 5 --wandb-enabled --wandb-project pig-distill --wandb-run-name distill_full --wandb-tags distill,gsm8k,gptoss20b"
EXTRA_TRAIN_ARGS="--response-only-loss --val-ratio 0.1 --test-ratio 0.1 --test-eval-every-epochs 1 --epoch-compare-num-examples 512 --early-stopping-patience 5 --early-stopping-min-delta 0.002 --early-stopping-warmup-epochs 1 --no-compile --wandb-enabled --wandb-project pig-finetune --wandb-run-name gpt2_gsm8k_response_only --wandb-tags finetune,gpt2,gsm8k,response_only"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher-model)
      TEACHER_MODEL="$2"
      shift 2
      ;;
    --student-model)
      STUDENT_MODEL="$2"
      shift 2
      ;;
    --distilled-path)
      DISTILLED_PATH="$2"
      shift 2
      ;;
    --student-output-dir)
      STUDENT_OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --student-batch-size)
      STUDENT_BATCH_SIZE="$2"
      shift 2
      ;;
    --student-grad-accum-steps)
      STUDENT_GRAD_ACCUM_STEPS="$2"
      shift 2
      ;;
    --student-num-workers)
      STUDENT_NUM_WORKERS="$2"
      shift 2
      ;;
    --student-lr)
      STUDENT_LR="$2"
      shift 2
      ;;
    --student-warmup-ratio)
      STUDENT_WARMUP_RATIO="$2"
      shift 2
      ;;
    --student-weight-decay)
      STUDENT_WEIGHT_DECAY="$2"
      shift 2
      ;;
    --student-max-grad-norm)
      STUDENT_MAX_GRAD_NORM="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --max-examples)
      MAX_EXAMPLES="$2"
      shift 2
      ;;
    --max-gpu-memory-gib)
      MAX_GPU_MEMORY_GIB="$2"
      shift 2
      ;;
    --gpu-memory-reserve-gib)
      GPU_MEMORY_RESERVE_GIB="$2"
      shift 2
      ;;
    --cpu-offload-gib)
      CPU_OFFLOAD_GIB="$2"
      shift 2
      ;;
    --offload-folder)
      OFFLOAD_FOLDER="$2"
      shift 2
      ;;
    --mxfp4-dequantize)
      MXFP4_DEQUANTIZE="--mxfp4-dequantize"
      shift 1
      ;;
    --no-mxfp4-dequantize)
      MXFP4_DEQUANTIZE="--no-mxfp4-dequantize"
      shift 1
      ;;
    --attn-implementation)
      ATTN_IMPLEMENTATION="$2"
      shift 2
      ;;
    --seq-len)
      SEQ_LEN="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --force-distill)
      FORCE_DISTILL=1
      shift 1
      ;;
    --extra-distill-args)
      EXTRA_DISTILL_ARGS="$2"
      shift 2
      ;;
    --extra-train-args)
      EXTRA_TRAIN_ARGS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${NPROC_PER_NODE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE=1
  fi
fi

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  NPROC_PER_NODE=1
fi

if [[ "${STUDENT_BATCH_SIZE}" -lt 1 ]]; then
  echo "[error] --student-batch-size must be >= 1 (got ${STUDENT_BATCH_SIZE})." >&2
  exit 1
fi

if [[ "${STUDENT_GRAD_ACCUM_STEPS}" -lt 1 ]]; then
  echo "[error] --student-grad-accum-steps must be >= 1 (got ${STUDENT_GRAD_ACCUM_STEPS})." >&2
  exit 1
fi

if [[ "${STUDENT_NUM_WORKERS}" -lt 0 ]]; then
  echo "[error] --student-num-workers must be >= 0 (got ${STUDENT_NUM_WORKERS})." >&2
  exit 1
fi

if [[ "${EXTRA_TRAIN_ARGS}" == *"--micro-batch-size"* ]]; then
  echo "[error] Do not pass --micro-batch-size inside --extra-train-args." >&2
  echo "[hint] Use --student-batch-size <int> instead." >&2
  exit 1
fi

if [[ "${EXTRA_TRAIN_ARGS}" == *"--grad-accum-steps"* ]]; then
  echo "[error] Do not pass --grad-accum-steps inside --extra-train-args." >&2
  echo "[hint] Use --student-grad-accum-steps <int> instead." >&2
  exit 1
fi

if [[ "${EXTRA_TRAIN_ARGS}" == *"--num-workers"* ]]; then
  echo "[error] Do not pass --num-workers inside --extra-train-args." >&2
  echo "[hint] Use --student-num-workers <int> instead." >&2
  exit 1
fi

if [[ "${EXTRA_TRAIN_ARGS}" == *"--lr"* ]]; then
  echo "[error] Do not pass --lr inside --extra-train-args." >&2
  echo "[hint] Use --student-lr <float> instead." >&2
  exit 1
fi

if [[ "${EXTRA_TRAIN_ARGS}" == *"--warmup-ratio"* ]]; then
  echo "[error] Do not pass --warmup-ratio inside --extra-train-args." >&2
  echo "[hint] Use --student-warmup-ratio <float> instead." >&2
  exit 1
fi

if [[ "${EXTRA_TRAIN_ARGS}" == *"--weight-decay"* ]]; then
  echo "[error] Do not pass --weight-decay inside --extra-train-args." >&2
  echo "[hint] Use --student-weight-decay <float> instead." >&2
  exit 1
fi

if [[ "${EXTRA_TRAIN_ARGS}" == *"--max-grad-norm"* ]]; then
  echo "[error] Do not pass --max-grad-norm inside --extra-train-args." >&2
  echo "[hint] Use --student-max-grad-norm <float> instead." >&2
  exit 1
fi

mkdir -p "$(dirname "${DISTILLED_PATH}")" "${STUDENT_OUTPUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DISTILLED_LINES=0
if [[ -f "${DISTILLED_PATH}" ]]; then
  DISTILLED_LINES="$(wc -l < "${DISTILLED_PATH}" | tr -d ' ')"
fi

RUN_DISTILL=1
if [[ "${FORCE_DISTILL}" -eq 0 && "${DISTILLED_LINES}" -gt 0 ]]; then
  RUN_DISTILL=0
  echo "[1/2] Reusing existing distilled dataset at ${DISTILLED_PATH} (rows=${DISTILLED_LINES})"
fi

if [[ "${RUN_DISTILL}" -eq 1 ]]; then
  if ! PYTHONPATH=src uv run python - <<'PY'
import importlib.util
ok = all(importlib.util.find_spec(pkg) is not None for pkg in ("datasets", "accelerate"))
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[deps] Installing distillation extras (datasets, accelerate)..."
    uv pip install datasets accelerate
  fi

  # Avoid stale/mismatched metadata from previous interrupted runs.
  rm -f "${DISTILLED_PATH}" "${DISTILLED_PATH}.meta.json"

  echo "[1/2] Distilling GSM8K with teacher=${TEACHER_MODEL} (max-examples=${MAX_EXAMPLES})"
  DISTILL_CMD=(
    uv run python -m gpt2.distill_gsm8k
    --teacher-model "${TEACHER_MODEL}"
    --output-path "${DISTILLED_PATH}"
    --max-examples "${MAX_EXAMPLES}"
    --batch-size "${BATCH_SIZE}"
    --dtype "${DTYPE}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
    --max-gpu-memory-gib "${MAX_GPU_MEMORY_GIB}"
    --gpu-memory-reserve-gib "${GPU_MEMORY_RESERVE_GIB}"
    --cpu-offload-gib "${CPU_OFFLOAD_GIB}"
    --offload-folder "${OFFLOAD_FOLDER}"
    "${MXFP4_DEQUANTIZE}"
  )
  if [[ -n "${EXTRA_DISTILL_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_DISTILL_ARRAY=(${EXTRA_DISTILL_ARGS})
    DISTILL_CMD+=("${EXTRA_DISTILL_ARRAY[@]}")
  fi
  PYTHONPATH=src "${DISTILL_CMD[@]}"

  DISTILLED_LINES="$(wc -l < "${DISTILLED_PATH}" | tr -d ' ')"
fi

if [[ "${DISTILLED_LINES}" -eq 0 ]]; then
  echo "[error] Distilled dataset has 0 rows at ${DISTILLED_PATH}." >&2
  echo "[error] Provide a non-empty pre-distilled file or run distillation first." >&2
  echo "[hint] Example: src/gpt2/run_distill_and_finetune.sh --force-distill" >&2
  exit 1
fi

echo "[info] distilled rows=${DISTILLED_LINES}"
if [[ "${DISTILLED_LINES}" -lt 64 ]]; then
  echo "[warn] Distilled dataset is very small (${DISTILLED_LINES} rows)." >&2
  echo "[warn] Finetuning will run, but quality is likely poor. Increase --max-examples." >&2
fi

echo "[2/2] Finetuning student=${STUDENT_MODEL} (nproc_per_node=${NPROC_PER_NODE})"
echo "[2/2] student micro-batch-size=${STUDENT_BATCH_SIZE}"
echo "[2/2] student num-workers=${STUDENT_NUM_WORKERS}"
echo "[2/2] student grad-accum=${STUDENT_GRAD_ACCUM_STEPS} lr=${STUDENT_LR} warmup=${STUDENT_WARMUP_RATIO} wd=${STUDENT_WEIGHT_DECAY} max-grad-norm=${STUDENT_MAX_GRAD_NORM}"
TRAIN_CMD=(
  uv run torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}"
  -m gpt2.finetuning
  --data-path "${DISTILLED_PATH}"
  --text-key text
  --model-name "${STUDENT_MODEL}"
  --output-dir "${STUDENT_OUTPUT_DIR}"
  --seq-len "${SEQ_LEN}"
  --epochs "${EPOCHS}"
  --micro-batch-size "${STUDENT_BATCH_SIZE}"
  --grad-accum-steps "${STUDENT_GRAD_ACCUM_STEPS}"
  --num-workers "${STUDENT_NUM_WORKERS}"
  --lr "${STUDENT_LR}"
  --warmup-ratio "${STUDENT_WARMUP_RATIO}"
  --weight-decay "${STUDENT_WEIGHT_DECAY}"
  --max-grad-norm "${STUDENT_MAX_GRAD_NORM}"
)
if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TRAIN_ARRAY=(${EXTRA_TRAIN_ARGS})
  TRAIN_CMD+=("${EXTRA_TRAIN_ARRAY[@]}")
fi
PYTHONPATH=src "${TRAIN_CMD[@]}"

LAST_CKPT="${STUDENT_OUTPUT_DIR}/checkpoints/last.pt"
if [[ ! -f "${LAST_CKPT}" ]]; then
  echo "[error] Missing checkpoint: ${LAST_CKPT}" >&2
  exit 1
fi

GLOBAL_STEP="$(
  LAST_CKPT_PATH="${LAST_CKPT}" PYTHONPATH=src uv run python - <<'PY'
import os
import torch

path = os.environ["LAST_CKPT_PATH"]
payload = torch.load(path, map_location="cpu")
print(int(payload.get("global_step", -1)))
PY
)"

if [[ "${GLOBAL_STEP}" -lt 1 ]]; then
  echo "[error] Finetuning completed with global_step=${GLOBAL_STEP} (no optimizer updates)." >&2
  echo "[hint] Try lowering --seq-len and/or set --student-batch-size 1 and --student-grad-accum-steps 1." >&2
  exit 1
fi
echo "[info] finetuning global_step=${GLOBAL_STEP}"

echo "[done] Distillation + finetuning completed."
