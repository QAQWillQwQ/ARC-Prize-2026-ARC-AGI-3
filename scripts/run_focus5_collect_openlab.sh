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
RUN_NAME="${1:-focus5_visual_collect_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./Local_Output/Collection_Cache/${RUN_NAME}}"
GAMES="${FOCUS5_GAMES:-sp80,lp85,ar25,ls20,r11l}"
SEEDS="${FOCUS5_SEEDS:-0,1,2,3,4,5,6,7}"
WORKERS="${FOCUS5_WORKERS:-16}"
PROFILE="${FOCUS5_PROFILE:-a100}"

python -m src.collect_staged \
  --project-root "." \
  --output-root "$OUTPUT_ROOT" \
  --hardware-profile "$PROFILE" \
  --games "$GAMES" \
  --seeds "$SEEDS" \
  --workers "$WORKERS"
