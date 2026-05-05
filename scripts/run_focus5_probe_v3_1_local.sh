#!/bin/bash
# Probe-first v3.1 runner: keeps state-action repeat penalty, cycle logic,
# local follow-up, and global-change-aware logic, but softens stagnation
# escape and re-probe. Exposes the new knobs as PROBE_* env vars.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${1:-probe_focus5_v3_1_${STAMP}}"
OUTPUT_ROOT="./Local_Output/Probe_Cache/${RUN_NAME}"

GAMES="${PROBE_GAMES:-sp80,lp85,ar25,ls20,r11l}"
SEEDS="${PROBE_SEEDS:-0,1,2,3}"
PROBE_BUDGET="${PROBE_BUDGET:-16}"
MAX_STEPS="${PROBE_MAX_STEPS:-160}"
STALL_STEPS="${PROBE_STALL_STEPS:-32}"
RESET_LIMIT="${PROBE_RESET_LIMIT:-2}"
CLICK_CANDIDATES="${PROBE_CLICK_CANDIDATES:-8}"

# Anti-loop knobs (kept from v3).
STATE_ACTION_PENALTY="${PROBE_STATE_ACTION_PENALTY:-0.35}"
CYCLE_2_PENALTY="${PROBE_CYCLE_2_PENALTY:-0.7}"
CYCLE_3_PENALTY="${PROBE_CYCLE_3_PENALTY:-0.55}"
STALE_SIGNATURE_PENALTY="${PROBE_STALE_SIGNATURE_PENALTY:-0.3}"
STALE_SIGNATURE_WINDOW="${PROBE_STALE_SIGNATURE_WINDOW:-6}"

# v3.1 softer stagnation defaults.
STAGNATION_WINDOW="${PROBE_STAGNATION_WINDOW:-16}"
STAGNATION_PROGRESSLESS_RATIO="${PROBE_STAGNATION_PROGRESSLESS_RATIO:-0.90}"
STAGNATION_ESCAPE_COOLDOWN="${PROBE_STAGNATION_ESCAPE_COOLDOWN:-8}"
ESCAPE_PRIORITY_BLEND="${PROBE_ESCAPE_PRIORITY_BLEND:-0.5}"
ESCAPE_SKIP_DEAD="${PROBE_ESCAPE_SKIP_DEAD:-1}"

# v3.1 reprobe gates and softer bonuses.
GLOBAL_CHANGE_REPROBE_BUDGET="${PROBE_GLOBAL_CHANGE_REPROBE_BUDGET:-3}"
REPROBE_EPISODE_CAP="${PROBE_REPROBE_EPISODE_CAP:-8}"
REPROBE_COOLDOWN_STEPS="${PROBE_REPROBE_COOLDOWN_STEPS:-6}"
REPROBE_SKIP_IF_RECENT_PROGRESS="${PROBE_REPROBE_SKIP_IF_RECENT_PROGRESS:-1}"
RECENT_PROGRESS_WINDOW="${PROBE_RECENT_PROGRESS_WINDOW:-8}"
REPROBE_DEAD_ACTION_BONUS="${PROBE_REPROBE_DEAD_ACTION_BONUS:-0.25}"
REPROBE_LOW_TRIAL_BONUS="${PROBE_REPROBE_LOW_TRIAL_BONUS:-0.1}"

# Local follow-up (kept from v3).
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
  --stagnation-escape-cooldown "$STAGNATION_ESCAPE_COOLDOWN" \
  --escape-priority-blend "$ESCAPE_PRIORITY_BLEND" \
  --escape-skip-dead "$ESCAPE_SKIP_DEAD" \
  --global-change-reprobe-budget "$GLOBAL_CHANGE_REPROBE_BUDGET" \
  --reprobe-episode-cap "$REPROBE_EPISODE_CAP" \
  --reprobe-cooldown-steps "$REPROBE_COOLDOWN_STEPS" \
  --reprobe-skip-if-recent-progress "$REPROBE_SKIP_IF_RECENT_PROGRESS" \
  --recent-progress-window "$RECENT_PROGRESS_WINDOW" \
  --reprobe-dead-action-bonus "$REPROBE_DEAD_ACTION_BONUS" \
  --reprobe-low-trial-bonus "$REPROBE_LOW_TRIAL_BONUS" \
  --local-followup-radius "$LOCAL_FOLLOWUP_RADIUS" \
  --local-followup-window "$LOCAL_FOLLOWUP_WINDOW" \
  --local-followup-bonus "$LOCAL_FOLLOWUP_BONUS"

EPISODES_PATH="$OUTPUT_ROOT/collected/episodes.jsonl.gz"
EVAL_PATH="$OUTPUT_ROOT/probe_eval.json"

python -m src.eval_probe \
  --input "$EPISODES_PATH" \
  --output "$EVAL_PATH" \
  --label probe_v3_1 \
  --print-overall

echo
echo "Probe v3.1 collect + eval complete:"
echo "  episodes: $EPISODES_PATH"
echo "  eval:     $EVAL_PATH"
echo
echo "Key v3.1 diagnostic fields in eval JSON loop_metrics:"
echo "  totals.{escape_steps, escape_dead_steps, reprobe_steps_used, reprobe_dead_steps}"
echo "  escape_dead_action_rate, reprobe_dead_action_rate"
echo "  totals.{escape_blocked_by_cooldown, reprobe_blocked_by_cap, reprobe_blocked_by_recent_progress}"
