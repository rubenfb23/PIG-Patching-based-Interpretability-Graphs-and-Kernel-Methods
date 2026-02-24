#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_CACHE_DIR=".cache/patch_effects/publication/gpt2"
FT_CACHE_DIR=".cache/patch_effects/publication/outputs_gpt2_gsm8k_distilled_lr2e5_acc4_model_final"
HOST="127.0.0.1"
BASE_PORT="8765"
DIFF_PORT="8766"
LOG_DIR="outputs/viewer_logs"
KILL_EXISTING="0"
TAIL_LOGS="0"

usage() {
  cat <<'EOF'
Usage: scripts/run_viewers_diff.sh [options]

Starts two PIG viewer backends in background:
  - base graph on BASE_PORT
  - diff graph on DIFF_PORT (using --base-cache-dir)

Options:
  --base-cache-dir PATH   Base cache dir (default: .cache/patch_effects/publication/gpt2)
  --ft-cache-dir PATH     Fine-tuned cache dir (default: .cache/patch_effects/publication/outputs_gpt2_gsm8k_distilled_lr2e5_acc4_model_final)
  --host HOST             Bind host (default: 127.0.0.1)
  --base-port PORT        Base backend port (default: 8765)
  --diff-port PORT        Diff backend port (default: 8766)
  --log-dir PATH          Log directory (default: outputs/viewer_logs)
  --kill-existing         Kill listeners already running on base/diff ports
  --tail-logs             Follow backend logs in this terminal
  -h, --help              Show help
EOF
}

get_listener_pids() {
  local port="$1"
  ss -ltnp "( sport = :${port} )" 2>/dev/null \
    | awk -F 'pid=' 'NR > 1 && $2 != "" { split($2, a, ","); print a[1] }' \
    | sort -u
}

ensure_port_free_or_kill() {
  local port="$1"
  local pids
  pids="$(get_listener_pids "$port" || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  if [[ "$KILL_EXISTING" == "1" ]]; then
    echo "Killing existing listener(s) on :${port}: ${pids}"
    # shellcheck disable=SC2086
    kill ${pids}
    sleep 0.5
    return 0
  fi

  echo "Port :${port} is already in use by PID(s): ${pids}" >&2
  echo "Use --kill-existing or free the port manually." >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-cache-dir)
      BASE_CACHE_DIR="$2"
      shift 2
      ;;
    --ft-cache-dir)
      FT_CACHE_DIR="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --diff-port)
      DIFF_PORT="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --kill-existing)
      KILL_EXISTING="1"
      shift
      ;;
    --tail-logs)
      TAIL_LOGS="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"

ensure_port_free_or_kill "$BASE_PORT"
ensure_port_free_or_kill "$DIFF_PORT"

mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BASE_LOG="${LOG_DIR}/viewer_base_${STAMP}.log"
DIFF_LOG="${LOG_DIR}/viewer_diff_${STAMP}.log"
PID_FILE="${LOG_DIR}/viewers_${STAMP}.pid"

PYTHONUNBUFFERED=1 uv run python -m pig.web.api \
  --cache-dir "$BASE_CACHE_DIR" \
  --host "$HOST" \
  --port "$BASE_PORT" >"$BASE_LOG" 2>&1 &
BASE_PID=$!

PYTHONUNBUFFERED=1 uv run python -m pig.web.api \
  --cache-dir "$FT_CACHE_DIR" \
  --base-cache-dir "$BASE_CACHE_DIR" \
  --host "$HOST" \
  --port "$DIFF_PORT" >"$DIFF_LOG" 2>&1 &
DIFF_PID=$!

sleep 1

if ! kill -0 "$BASE_PID" 2>/dev/null; then
  echo "Base viewer failed to start. Log: $BASE_LOG" >&2
  tail -n 40 "$BASE_LOG" >&2 || true
  exit 1
fi

if ! kill -0 "$DIFF_PID" 2>/dev/null; then
  echo "Diff viewer failed to start. Log: $DIFF_LOG" >&2
  tail -n 40 "$DIFF_LOG" >&2 || true
  kill "$BASE_PID" 2>/dev/null || true
  exit 1
fi

cat >"$PID_FILE" <<EOF
BASE_PID=${BASE_PID}
DIFF_PID=${DIFF_PID}
EOF

echo "Base viewer: ws://${HOST}:${BASE_PORT} (pid ${BASE_PID})"
echo "Diff viewer: ws://${HOST}:${DIFF_PORT} (pid ${DIFF_PID})"
echo "Logs:"
echo "  ${BASE_LOG}"
echo "  ${DIFF_LOG}"
echo "PID file: ${PID_FILE}"
echo
echo "Frontend:"
echo "  cd web && npm run dev -- --host 127.0.0.1 --port 5173"
echo
echo "Stop:"
echo "  kill ${BASE_PID} ${DIFF_PID}"

if [[ "$TAIL_LOGS" == "1" ]]; then
  echo
  echo "Tailing logs (Ctrl+C to stop tailing; backends keep running):"
  tail -f "$BASE_LOG" "$DIFF_LOG"
fi
