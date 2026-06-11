#!/bin/bash
# Run the FORGEHybrid V2 notebook agent backend on OpenLab vulcan.
# This script is intentionally a quality pass: one full trajectory per
# train game first, with detailed step traces and complete gzipped episodes.

#SBATCH --job-name=arc_forgev2_collect
#SBATCH --partition=openlab.p
#SBATCH --nodelist=vulcan
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/yidingw6/projects/ARC-Prize-2026-ARC-AGI-3/slurm_logs/%x_%j.out
#SBATCH --error=/home/yidingw6/projects/ARC-Prize-2026-ARC-AGI-3/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/home/yidingw6/projects/ARC-Prize-2026-ARC-AGI-3"
cd "$PROJECT_ROOT"
source .venv/bin/activate

export MPLCONFIGDIR=/tmp/mpl_yidingw6_arc
export PYTHONPYCACHEPREFIX=/tmp/arc_pycache_yidingw6
mkdir -p "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TRAIN_GAMES="${TRAIN_GAMES:-bp35,cd82,cn04,dc22,ft09,g50t,ka59,lf52,m0r0,re86,s5i5,sb26,sc25,sk48,su15,tn36,tr87,tu93,vc33,wa30}"
HELDOUT_GAMES="${HELDOUT_GAMES:-ar25,lp85,ls20,r11l,sp80}"
SEEDS="${SEEDS:-0}"
EPISODES_PER_GAME="${EPISODES_PER_GAME:-1}"
WORKERS="${WORKERS:-16}"
MAX_STEPS="${MAX_STEPS:-240}"
RESET_LIMIT="${RESET_LIMIT:-3}"
STEP_LOG_EVERY="${STEP_LOG_EVERY:-5}"
EPISODE_TIMEOUT="${EPISODE_TIMEOUT:-3600}"

RUN_ROOT="$PROJECT_ROOT/Local_Output/OpenLab_ForgeV2Collect_train20_heldout5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > "$PROJECT_ROOT/LATEST_OPENLAB_FORGE_COLLECT_DIR.txt"

{
  echo "[config] user=yidingw6"
  echo "[config] node=vulcan"
  echo "[config] partition=openlab.p"
  echo "[config] cpus=16"
  echo "[config] mem=48G"
  echo "[config] run_root=$RUN_ROOT"
  echo "[config] train_games=$TRAIN_GAMES"
  echo "[config] heldout_games=$HELDOUT_GAMES"
  echo "[config] seeds=$SEEDS"
  echo "[config] episodes_per_game=$EPISODES_PER_GAME"
  echo "[config] workers=$WORKERS"
  echo "[config] max_steps=$MAX_STEPS"
  echo "[config] reset_limit=$RESET_LIMIT"
  echo "[config] step_log_every=$STEP_LOG_EVERY"
  echo "[config] episode_timeout=$EPISODE_TIMEOUT"
  echo "[config] git_commit=$(git rev-parse HEAD)"
  echo "[config] python=$(python --version)"
  echo "[config] started_at=$(date)"
} | tee "$RUN_ROOT/run_header.log"

python -u -m src.collect_forge_openlab \
  --project-root "$PROJECT_ROOT" \
  --output-root "$RUN_ROOT" \
  --games "$TRAIN_GAMES" \
  --heldout-games "$HELDOUT_GAMES" \
  --seeds "$SEEDS" \
  --episodes-per-game "$EPISODES_PER_GAME" \
  --workers "$WORKERS" \
  --max-steps "$MAX_STEPS" \
  --reset-limit "$RESET_LIMIT" \
  --step-log-every "$STEP_LOG_EVERY" \
  --episode-timeout "$EPISODE_TIMEOUT" \
  --notebook "$PROJECT_ROOT/ColabNotebook/Submissions/0.31 Yiding_FORGEHybridV2 0525.0217.ipynb" \
  2>&1 | tee "$RUN_ROOT/collect.log"

gzip -t "$RUN_ROOT/all_episodes.jsonl.gz"

python -u -m src.build_ranker_dataset \
  --episodes "$RUN_ROOT/all_episodes.jsonl.gz" \
  --output "$RUN_ROOT/ranker_examples.jsonl.gz" \
  --metadata-output "$RUN_ROOT/ranker_metadata.json" \
  --coord-budget 32 \
  --max-steps 240 \
  --min-positive-utility 0.05 \
  --progress-every 1000 \
  2>&1 | tee "$RUN_ROOT/build_ranker_dataset.log"

{
  echo "[done] finished_at=$(date)"
  echo "[done] run_root=$RUN_ROOT"
  echo "[done] aggregate=$RUN_ROOT/all_episodes.jsonl.gz"
  echo "[done] ranker_examples=$RUN_ROOT/ranker_examples.jsonl.gz"
  echo "[done] metadata=$RUN_ROOT/ranker_metadata.json"
} | tee -a "$RUN_ROOT/run_header.log"
