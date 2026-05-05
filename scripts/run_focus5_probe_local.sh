#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${1:-probe_focus5_${STAMP}}"
OUTPUT_ROOT="./Local_Output/Probe_Cache/${RUN_NAME}"
GAMES="${PROBE_GAMES:-sp80,lp85,ar25,ls20,r11l}"
SEEDS="${PROBE_SEEDS:-0,1,2,3}"
PROBE_BUDGET="${PROBE_BUDGET:-16}"
MAX_STEPS="${PROBE_MAX_STEPS:-160}"
STALL_STEPS="${PROBE_STALL_STEPS:-32}"
RESET_LIMIT="${PROBE_RESET_LIMIT:-2}"
CLICK_CANDIDATES="${PROBE_CLICK_CANDIDATES:-8}"

python -m src.collect_probe \
  --project-root "." \
  --output-root "$OUTPUT_ROOT" \
  --games "$GAMES" \
  --seeds "$SEEDS" \
  --probe-budget "$PROBE_BUDGET" \
  --max-steps "$MAX_STEPS" \
  --stall-steps "$STALL_STEPS" \
  --reset-limit "$RESET_LIMIT" \
  --click-candidates "$CLICK_CANDIDATES"

EPISODES_PATH="$OUTPUT_ROOT/collected/episodes.jsonl.gz"
EVAL_PATH="$OUTPUT_ROOT/probe_eval.json"

python -m src.eval_probe \
  --input "$EPISODES_PATH" \
  --output "$EVAL_PATH" \
  --label probe \
  --print-overall

echo
echo "Probe collect + eval complete:"
echo "  episodes: $EPISODES_PATH"
echo "  eval:     $EVAL_PATH"
echo
echo "To compare against the teammate's collector on the same data:"
echo "  python -m src.eval_probe --input <baseline_episodes.jsonl.gz> --output <baseline_eval.json> --label baseline"
