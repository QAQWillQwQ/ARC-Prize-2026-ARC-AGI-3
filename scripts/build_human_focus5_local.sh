#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  echo "Missing .venv in $PROJECT_ROOT"
  exit 1
fi

source .venv/bin/activate

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${1:-human_focus5_${STAMP}}"
INPUT_PATH="${HUMAN_REPLAY_INPUT:-/Users/wangyiding/Downloads/arc_agi_3_public_demo_human_testing.zip}"
OUTPUT_ROOT="./Local_Output/Human_Cache/${RUN_NAME}"
OUTPUT_PATH="${OUTPUT_ROOT}/collected/episodes.jsonl.gz"
GAMES="${HUMAN_GAMES:-sp80,lp85,ar25,ls20,r11l}"
MIN_LEVELS="${HUMAN_MIN_LEVELS:-1}"
TOP_K="${HUMAN_TOPK_PER_GAME:-10}"

mkdir -p "$OUTPUT_ROOT"

python -m src.import_human_replays \
  --project-root "." \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --games "$GAMES" \
  --min-levels "$MIN_LEVELS" \
  --top-k-per-game "$TOP_K"
