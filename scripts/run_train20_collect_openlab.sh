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
RUN_NAME="${1:-train20_visual_collect_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./Local_Output/Collection_Cache/${RUN_NAME}}"
GAMES="${TRAIN20_GAMES:-ar25,cd82,cn04,ft09,g50t,ka59,lf52,lp85,ls20,m0r0,r11l,s5i5,sc25,sk48,sp80,su15,tn36,tr87,tu93,vc33}"
HOLDOUT_GAMES="${HOLDOUT_GAMES:-bp35,dc22,re86,sb26,wa30}"
SEEDS="${TRAIN20_SEEDS:-0,1,2,3,4,5,6,7}"
WORKERS="${TRAIN20_WORKERS:-16}"
PROFILE="${TRAIN20_PROFILE:-a100}"

echo "Collecting train games: $GAMES"
echo "Reserved holdout games, not collected by this script: $HOLDOUT_GAMES"

python -m src.collect_staged \
  --project-root "." \
  --output-root "$OUTPUT_ROOT" \
  --hardware-profile "$PROFILE" \
  --games "$GAMES" \
  --seeds "$SEEDS" \
  --workers "$WORKERS"
