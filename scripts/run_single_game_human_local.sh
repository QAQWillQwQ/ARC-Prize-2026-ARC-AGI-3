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
GAME_ID="${SINGLE_GAME_ID:-ar25}"
RUN_NAME="${1:-single_game_${GAME_ID}_${STAMP}}"
SOURCE_GZ="${SOURCE_HUMAN_GZ:-$PROJECT_ROOT/Local_Output/Human_Cache/arc_agi_3_public_demo_human_testing.gz}"
OUTPUT_ROOT="$PROJECT_ROOT/Local_Output/Single_Game_Experiments/${RUN_NAME}"
FILTERED_GZ="$OUTPUT_ROOT/${GAME_ID}.episodes.jsonl.gz"
EPISODE_VAL_FRACTION="${EPISODE_VAL_FRACTION:-0.2}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-m3_cpu}"
MAX_STEPS="${MAX_STEPS:-192}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-50}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-5}"
DATA_WORKERS="${DATA_WORKERS:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
ONLINE_VAL_GAMES="${ONLINE_VAL_GAMES:-1}"

mkdir -p "$OUTPUT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Game id: $GAME_ID"
echo "Source human gzip: $SOURCE_GZ"
echo "Output root: $OUTPUT_ROOT"
echo "Filtered gzip: $FILTERED_GZ"
echo "Hardware profile: $HARDWARE_PROFILE"
echo "Episode val fraction: $EPISODE_VAL_FRACTION"

python -m src.filter_episodes \
  --input "$SOURCE_GZ" \
  --output "$FILTERED_GZ" \
  --games "$GAME_ID" | tee "$OUTPUT_ROOT/filter_stdout.log"

TRAIN_CMD=(
  python -m src.train
  --project-root "$PROJECT_ROOT"
  --data "$FILTERED_GZ"
  --games "$GAME_ID"
  --split-mode episode
  --episode-val-fraction "$EPISODE_VAL_FRACTION"
  --output-dir "$OUTPUT_ROOT"
  --hardware-profile "$HARDWARE_PROFILE"
  --max-steps "$MAX_STEPS"
  --online-val-games "$ONLINE_VAL_GAMES"
  --checkpoint-every-steps "$CHECKPOINT_EVERY_STEPS"
  --data-workers "$DATA_WORKERS"
  --log-every-batches "$LOG_EVERY_BATCHES"
)

if [ -n "$RESUME_CHECKPOINT" ]; then
  TRAIN_CMD+=(--resume "$RESUME_CHECKPOINT")
fi

PYTHONUNBUFFERED=1 "${TRAIN_CMD[@]}" | tee "$OUTPUT_ROOT/train_stdout.log"

python -m src.evaluate \
  --project-root "$PROJECT_ROOT" \
  --checkpoint "$OUTPUT_ROOT/checkpoints/best.pth" \
  --output "$OUTPUT_ROOT/${GAME_ID}_best_public_eval.json" \
  --games "$GAME_ID" | tee "$OUTPUT_ROOT/eval_best_stdout.log"

python -m src.evaluate \
  --project-root "$PROJECT_ROOT" \
  --checkpoint "$OUTPUT_ROOT/checkpoints/last.pth" \
  --output "$OUTPUT_ROOT/${GAME_ID}_last_public_eval.json" \
  --games "$GAME_ID" | tee "$OUTPUT_ROOT/eval_last_stdout.log"

echo "Done."
echo "Metrics: $OUTPUT_ROOT/metrics.csv"
echo "Best eval: $OUTPUT_ROOT/${GAME_ID}_best_public_eval.json"
echo "Last eval: $OUTPUT_ROOT/${GAME_ID}_last_public_eval.json"
