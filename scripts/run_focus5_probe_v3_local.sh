#!/bin/bash
# Probe-first v3 runner: probe phase unchanged, exploit phase adds anti-loop
# logic, local follow-up around promising clicks, and a global_change-triggered
# re-probe budget.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${1:-probe_focus5_v3_${STAMP}}"
OUTPUT_ROOT="./Local_Output/Probe_Cache/${RUN_NAME}"

GAMES="${PROBE_GAMES:-sp80,lp85,ar25,ls20,r11l}"
SEEDS="${PROBE_SEEDS:-0,1,2,3}"
PROBE_BUDGET="${PROBE_BUDGET:-16}"
MAX_STEPS="${PROBE_MAX_STEPS:-160}"
STALL_STEPS="${PROBE_STALL_STEPS:-32}"
RESET_LIMIT="${PROBE_RESET_LIMIT:-2}"
CLICK_CANDIDATES="${PROBE_CLICK_CANDIDATES:-8}"

# v3 anti-loop / exploit knobs (override via env).
STATE_ACTION_PENALTY="${PROBE_STATE_ACTION_PENALTY:-0.35}"
CYCLE_2_PENALTY="${PROBE_CYCLE_2_PENALTY:-0.7}"
CYCLE_3_PENALTY="${PROBE_CYCLE_3_PENALTY:-0.55}"
STALE_SIGNATURE_PENALTY="${PROBE_STALE_SIGNATURE_PENALTY:-0.3}"
STALE_SIGNATURE_WINDOW="${PROBE_STALE_SIGNATURE_WINDOW:-6}"
STAGNATION_WINDOW="${PROBE_STAGNATION_WINDOW:-12}"
STAGNATION_PROGRESSLESS_RATIO="${PROBE_STAGNATION_PROGRESSLESS_RATIO:-0.85}"
GLOBAL_CHANGE_REPROBE_BUDGET="${PROBE_GLOBAL_CHANGE_REPROBE_BUDGET:-4}"
LOCAL_FOLLOWUP_RADIUS="${PROBE_LOCAL_FOLLOWUP_RADIUS:-6}"
LOCAL_FOLLOWUP_WINDOW="${PROBE_LOCAL_FOLLOWUP_WINDOW:-6}"
LOCAL_FOLLOWUP_BONUS="${PROBE_LOCAL_FOLLOWUP_BONUS:-1.2}"

python -m src.collect_probe \
  --project-root "." \
  --output-root "$OUTPUT_ROOT" \
  --games "$GAMES" \
  --seeds "$SEEDS" \
  --probe-budget "$PROBE_BUDGET" \
  --max-steps "$MAX_STEPS" \
  --stall-steps "$STALL_STEPS" \
  --reset-limit "$RESET_LIMIT" \
  --click-candidates "$CLICK_CANDIDATES" \
  --state-action-penalty "$STATE_ACTION_PENALTY" \
  --cycle-2-penalty "$CYCLE_2_PENALTY" \
  --cycle-3-penalty "$CYCLE_3_PENALTY" \
  --stale-signature-penalty "$STALE_SIGNATURE_PENALTY" \
  --stale-signature-window "$STALE_SIGNATURE_WINDOW" \
  --stagnation-window "$STAGNATION_WINDOW" \
  --stagnation-progressless-ratio "$STAGNATION_PROGRESSLESS_RATIO" \
  --global-change-reprobe-budget "$GLOBAL_CHANGE_REPROBE_BUDGET" \
  --local-followup-radius "$LOCAL_FOLLOWUP_RADIUS" \
  --local-followup-window "$LOCAL_FOLLOWUP_WINDOW" \
  --local-followup-bonus "$LOCAL_FOLLOWUP_BONUS"

EPISODES_PATH="$OUTPUT_ROOT/collected/episodes.jsonl.gz"
EVAL_PATH="$OUTPUT_ROOT/probe_eval.json"

python -m src.eval_probe \
  --input "$EPISODES_PATH" \
  --output "$EVAL_PATH" \
  --label probe_v3 \
  --print-overall

echo
echo "Probe v3 collect + eval complete:"
echo "  episodes: $EPISODES_PATH"
echo "  eval:     $EVAL_PATH"
echo
echo "Per-game loop_metrics is in the eval JSON; key fields:"
echo "  loop_metrics.totals.{cycle_2_count, cycle_3_count, repeated_state_action_count}"
echo "  loop_metrics.local_followup_success_rate"
echo "  loop_metrics.totals.{stagnation_escapes, reprobe_windows_opened, reprobe_steps_used}"
