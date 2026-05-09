#!/usr/bin/env bash
#
# scripts/run_openlab_collect.sh — Launch the full bc_v2 collect on Openlab.
#
# Defaults: 25 games × 5 deviations × 4 seeds × 7 strategies + perturbation
# Bucket 4 → 3,125 tasks, ~30–90 min on a 64-worker Openlab box.
#
# Usage:
#   bash scripts/run_openlab_collect.sh                      # full run, defaults
#   bash scripts/run_openlab_collect.sh --workers 32         # tune workers
#   bash scripts/run_openlab_collect.sh --output-name myrun  # custom output dir
#   bash scripts/run_openlab_collect.sh --smoke              # quick 2-game smoke test
#
# After launch:
#   tail -F --pid=$(cat $LOG_DIR/$OUTPUT_NAME/collect.pid) "$LOG_FILE"
#
# The script detaches via setsid + disown so it survives shell logout.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/jihangli/miniconda3/envs/arc312/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3.12 || command -v python3 || command -v python)"
fi

LOG_DIR="$PROJECT_ROOT/Local_Output/Logs"
COLLECT_DIR="$PROJECT_ROOT/Local_Output/Collection_Cache"
mkdir -p "$LOG_DIR" "$COLLECT_DIR"

# ---------- defaults ----------
WORKERS=64
MAX_STEPS=3000
OUTPUT_NAME="staged_v2_gt"
DEVIATIONS="0,25,50,75,100"
SEEDS=4
STRATEGIES="random_full,keyboard_only,click_grid,directional_sustained,edge_sweep,color_targeted,action_id_sweep"
PERTURB_RATES="0.05,0.10,0.15"
PERTURB_SEEDS=4
GAMES=""
SMOKE=0

# ---------- parse args ----------
while [ $# -gt 0 ]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2;;
    --output-name) OUTPUT_NAME="$2"; shift 2;;
    --max-steps) MAX_STEPS="$2"; shift 2;;
    --deviations) DEVIATIONS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --strategies) STRATEGIES="$2"; shift 2;;
    --perturb-rates) PERTURB_RATES="$2"; shift 2;;
    --perturb-seeds) PERTURB_SEEDS="$2"; shift 2;;
    --games) GAMES="$2"; shift 2;;
    --smoke)
      SMOKE=1
      OUTPUT_NAME="gt_warmstart_smoke"
      WORKERS=4
      MAX_STEPS=1500
      DEVIATIONS="0,50,100"
      SEEDS=1
      STRATEGIES="random_full,click_grid,edge_sweep"
      PERTURB_RATES="0.10"
      PERTURB_SEEDS=1
      GAMES="r11l,sp80,cd82"
      shift
      ;;
    --help|-h)
      head -25 "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "[run-collect] unknown option: $1"; exit 1;;
  esac
done

OUTPUT_ROOT="$COLLECT_DIR/$OUTPUT_NAME"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR/$OUTPUT_NAME"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/$OUTPUT_NAME/collect_${STAMP}.log"
PID_FILE="$LOG_DIR/$OUTPUT_NAME/collect.pid"

# ---------- build args ----------
ARGS=(
  -m scripts.collect_gt_warmstart  # use module form so package imports resolve
  --project-root "$PROJECT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --workers "$WORKERS"
  --max-steps "$MAX_STEPS"
  --deviation-points "$DEVIATIONS"
  --seeds-per-deviation "$SEEDS"
  --strategies "$STRATEGIES"
  --perturb-rates "$PERTURB_RATES"
  --perturb-seeds "$PERTURB_SEEDS"
)
if [ -n "$GAMES" ]; then
  ARGS+=(--games "$GAMES")
fi

# Module-style invocation needs the script directly (it's a script, not a package
# module). Use the file path instead.
ARGS=(
  "$PROJECT_ROOT/scripts/collect_gt_warmstart.py"
  --project-root "$PROJECT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --workers "$WORKERS"
  --max-steps "$MAX_STEPS"
  --deviation-points "$DEVIATIONS"
  --seeds-per-deviation "$SEEDS"
  --strategies "$STRATEGIES"
  --perturb-rates "$PERTURB_RATES"
  --perturb-seeds "$PERTURB_SEEDS"
)
if [ -n "$GAMES" ]; then
  ARGS+=(--games "$GAMES")
fi

# Show config
echo "[run-collect] PY=$PY"
echo "[run-collect] output_root=$OUTPUT_ROOT"
echo "[run-collect] workers=$WORKERS max_steps=$MAX_STEPS"
echo "[run-collect] deviations=$DEVIATIONS seeds=$SEEDS"
echo "[run-collect] strategies=$STRATEGIES"
echo "[run-collect] perturb_rates=$PERTURB_RATES perturb_seeds=$PERTURB_SEEDS"
echo "[run-collect] games=${GAMES:-<all 25>}"
echo "[run-collect] log=$LOG_FILE"

if [ "$SMOKE" -eq 1 ]; then
  # Foreground for smoke (so user sees output immediately and validates).
  echo "[run-collect] SMOKE MODE — running foreground"
  "$PY" "${ARGS[@]}"
  exit $?
fi

# Background launch with full detach.
echo "[run-collect] launching in background (setsid + disown)"
setsid nohup "$PY" "${ARGS[@]}" < /dev/null > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID" 2>/dev/null || true

echo "[run-collect] pid=$PID (saved to $PID_FILE)"
echo "[run-collect] tail with: tail -F --pid=$PID $LOG_FILE"
echo "[run-collect] kill with:  kill -INT $PID"
