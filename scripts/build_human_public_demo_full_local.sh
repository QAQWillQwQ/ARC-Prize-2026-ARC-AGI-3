#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  echo "Missing .venv in $PROJECT_ROOT"
  exit 1
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

INPUT_PATH="${HUMAN_REPLAY_INPUT:-/Users/wangyiding/Downloads/arc_agi_3_public_demo_human_testing.zip}"
OUTPUT_ROOT="${HUMAN_OUTPUT_ROOT:-./Local_Output/Human_Cache}"
OUTPUT_PATH="${HUMAN_OUTPUT_PATH:-${OUTPUT_ROOT}/arc_agi_3_public_demo_human_testing.gz}"
GAMES="${HUMAN_GAMES:-}"
MIN_LEVELS="${HUMAN_MIN_LEVELS:-0}"
TOP_K="${HUMAN_TOPK_PER_GAME:-}"
SOLVED_ONLY="${HUMAN_SOLVED_ONLY:-0}"
PROGRESS_EVERY="${HUMAN_PROGRESS_EVERY:-5}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Starting human replay conversion"
echo "  input  = $INPUT_PATH"
echo "  output = $OUTPUT_PATH"
echo "  games  = ${GAMES:-ALL}"
echo "  min_levels = $MIN_LEVELS"
echo "  top_k_per_game = ${TOP_K:-none}"
echo "  solved_only = $SOLVED_ONLY"
echo

CMD=(
  python -m src.import_human_replays
  --project-root "."
  --input "$INPUT_PATH"
  --output "$OUTPUT_PATH"
  --min-levels "$MIN_LEVELS"
  --progress-every "$PROGRESS_EVERY"
)

if [ -n "$GAMES" ]; then
  CMD+=(--games "$GAMES")
fi

if [ -n "$TOP_K" ]; then
  CMD+=(--top-k-per-game "$TOP_K")
fi

if [ "$SOLVED_ONLY" = "1" ]; then
  CMD+=(--solved-only)
fi

"${CMD[@]}"

echo
echo "Created training data:"
ls -lh "$OUTPUT_PATH"
if [ -f "${OUTPUT_PATH}.summary.json" ]; then
  ls -lh "${OUTPUT_PATH}.summary.json"
fi
