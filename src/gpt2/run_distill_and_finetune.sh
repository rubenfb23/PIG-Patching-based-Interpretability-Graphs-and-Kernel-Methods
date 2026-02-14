#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run teacher distillation (GSM8K) + student GPT-2 finetuning, sequentially.

Default behavior uses the FULL dataset (max-examples = -1).

Usage:
  src/gpt2/run_distill_and_finetune.sh [options]

Options:
  --teacher-model <name>       Teacher model (default: openai/gpt-oss-20b)
  --student-model <name>       Student model (default: gpt2)
  --distilled-path <path>      Output JSONL for distilled data
  --student-output-dir <path>  Output directory for finetuned model
  --batch-size <int>           Teacher generation batch size (default: 1)
  --dtype <auto|bf16|fp16|fp32> Teacher dtype (default: bf16)
  --attn-implementation <str>  Teacher attention implementation (default: eager)
  --max-gpu-memory-gib <float> Per-GPU cap for teacher load (default: 0 auto)
  --gpu-memory-reserve-gib <float> Reserve per GPU when auto cap (default: 2)
  --cpu-offload-gib <float>    CPU RAM budget for offload (default: 96)
  --offload-folder <path>      Offload folder (default: outputs/offload/gpt_oss_teacher)
  --mxfp4-dequantize | --no-mxfp4-dequantize
                               GPT-OSS dequantize mode (default: enabled)
  --max-examples <int>         Distillation sample cap; -1 means full dataset (default: -1)
  --seq-len <int>              Student training sequence length (default: 512)
  --epochs <int>               Student training epochs (default: 3)
  --nproc-per-node <int>       GPUs/processes for torchrun (default: auto-detect)
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
BATCH_SIZE=1
DTYPE="bf16"
ATTN_IMPLEMENTATION="eager"
MAX_GPU_MEMORY_GIB=0
GPU_MEMORY_RESERVE_GIB=2
CPU_OFFLOAD_GIB=96
OFFLOAD_FOLDER="outputs/offload/gpt_oss_teacher"
MXFP4_DEQUANTIZE="--mxfp4-dequantize"
MAX_EXAMPLES=-1
SEQ_LEN=512
EPOCHS=3
NPROC_PER_NODE=""
EXTRA_DISTILL_ARGS=""
EXTRA_TRAIN_ARGS=""

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

mkdir -p "$(dirname "${DISTILLED_PATH}")" "${STUDENT_OUTPUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if ! PYTHONPATH=src uv run python - <<'PY'
import importlib.util
ok = all(importlib.util.find_spec(pkg) is not None for pkg in ("datasets", "accelerate"))
raise SystemExit(0 if ok else 1)
PY
then
  echo "[deps] Installing distillation extras (datasets, accelerate)..."
  uv pip install datasets accelerate
fi

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
if [[ "${DISTILLED_LINES}" -eq 0 ]]; then
  echo "[error] Distillation produced 0 rows at ${DISTILLED_PATH}." >&2
  echo "[error] Try increasing --max-new-tokens and keep chat template enabled." >&2
  echo "[hint] Example: --extra-distill-args \"--max-new-tokens 192 --log-interval 5\"" >&2
  exit 1
fi
echo "[info] distilled rows=${DISTILLED_LINES}"

echo "[2/2] Finetuning student=${STUDENT_MODEL} (nproc_per_node=${NPROC_PER_NODE})"
TRAIN_CMD=(
  uv run torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}"
  -m gpt2.finetuning
  --data-path "${DISTILLED_PATH}"
  --text-key text
  --model-name "${STUDENT_MODEL}"
  --output-dir "${STUDENT_OUTPUT_DIR}"
  --seq-len "${SEQ_LEN}"
  --epochs "${EPOCHS}"
)
if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TRAIN_ARRAY=(${EXTRA_TRAIN_ARGS})
  TRAIN_CMD+=("${EXTRA_TRAIN_ARRAY[@]}")
fi
PYTHONPATH=src "${TRAIN_CMD[@]}"

echo "[done] Distillation + finetuning completed."
