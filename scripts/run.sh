#!/usr/bin/env bash
#
# scripts/run.sh — Step dispatcher for ARC AGI 3 worldmodel pipeline.
#
# Linux-only (Openlab, WSL2). Uses Python 3.12 + venv (no conda required).
#
# Usage:
#   bash scripts/run.sh <command> [options]
#
# Commands:
#   setup [--cpu|--cuda]      Create .venv at project root and install deps
#   check                     Verify imports + show env status
#   collect-quick [opts]      Step 1: small-subset collect (default games: sp80,ar25,lp85,r11l,cn04,ls20)
#   inspect [opts]            Step 2: aggregate stats over a .gz dataset
#   collect-full [opts]       Step 3: full 25-game collect
#   pretrain [opts]           Step 4: encoder pretraining
#   train [opts]              Step 6: joint worldmodel training
#   eval [opts]               Step 7: Phase 3 sanity eval
#   status                    Show running pipeline processes
#   logs [pattern]            Tail latest log file matching pattern
#   help [command]            Show help for a specific command
#
# Run any command with --help to see its options:
#   bash scripts/run.sh collect-full --help

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PY="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
LOG_DIR="${PROJECT_ROOT}/Local_Output/Logs"
COLLECT_DIR="${PROJECT_ROOT}/Local_Output/Collection_Cache"
TRAIN_DIR="${PROJECT_ROOT}/Training_Output"

mkdir -p "$LOG_DIR" "$COLLECT_DIR" "$TRAIN_DIR"

# ---------- helpers ----------

log() { printf '[run.sh] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

require_venv() {
  [ -x "$PY" ] || die ".venv not found. Run: bash scripts/run.sh setup"
}

find_py312() {
  for cand in python3.12 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
      ver="$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
      if [ "$ver" = "3.12" ]; then
        echo "$cand"
        return 0
      fi
    fi
  done
  return 1
}

ts() { date +%Y%m%d_%H%M%S; }

# Background-launch a python module, log into per-output-name subdir, save pid + run metadata.
# Args: spawn <command-tag> <output-name> -- <python-args>
#   command-tag: short identifier (collect_full, train, eval, ...)
#   output-name: user-facing run label (mirrors Collection_Cache/Training_Output dir name)
spawn() {
  require_venv
  local command="$1"; shift
  local output_name="$1"; shift
  local stamp="$(ts)"
  local subdir="${LOG_DIR}/${output_name}"
  mkdir -p "$subdir"

  local logpath="${subdir}/${command}_${stamp}.log"
  local pidpath="${subdir}/${command}.pid"
  local metapath="${subdir}/${command}_${stamp}.run.json"
  local label="${command}_${output_name}"   # for status/log lookup messages

  # Save run metadata so status/audit can reconstruct what was launched
  printf '{"command":"%s","output_name":"%s","start":"%s","cmd":%s,"log":"%s","cwd":"%s"}\n' \
    "$command" "$output_name" "$(date -Is)" \
    "$(printf '%s\n' "$@" | "$PY" -c 'import json,sys; print(json.dumps([line.rstrip() for line in sys.stdin]))')" \
    "$logpath" "$PROJECT_ROOT" > "$metapath" 2>/dev/null || true

  log "log: $logpath"
  # Full detach:
  #   setsid:    new session, no controlling tty (so SIGHUP / parent shell exit don't kill it)
  #   < /dev/null: no inherited stdin (parent shell doesn't get blocked on FD)
  #   > log 2>&1: stdout + stderr to file (no inherited tty FDs)
  #   &:         background
  #   disown:    drop from bash job table so run.sh exits cleanly
  setsid nohup "$PY" "$@" < /dev/null > "$logpath" 2>&1 &
  local pid=$!
  echo "$pid" > "$pidpath"
  disown "$pid" 2>/dev/null || true
  log "pid: $pid (saved to $pidpath)"
  log "tail with: tail -f $logpath"
  log "or: bash scripts/run.sh logs ${label}"
}

# Ensure a directory exists, log on creation
ensure_dir() {
  if [ ! -d "$1" ]; then
    log "creating dir: $1"
    mkdir -p "$1"
  fi
}

# ---------- command: setup ----------

cmd_setup() {
  local mode="auto"
  while [ $# -gt 0 ]; do
    case "$1" in
      --cpu) mode="cpu";;
      --cuda) mode="cuda";;
      --help|-h)
        cat <<EOF
setup — Create .venv and install dependencies.

Usage: bash scripts/run.sh setup [--cpu|--cuda]

  --cpu    Install CPU-only PyTorch (smaller, no GPU needed)
  --cuda   Install CUDA 12.1 PyTorch
  (default: auto-detect via nvidia-smi)
EOF
        return 0;;
      *) die "unknown setup option: $1";;
    esac
    shift
  done

  if [ "$mode" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      mode="cuda"
      log "auto-detected GPU; installing CUDA torch"
    else
      mode="cpu"
      log "no GPU detected; installing CPU torch"
    fi
  fi

  local py312
  py312="$(find_py312)" || die "Python 3.12 not found in PATH. Install it first."
  log "using $py312 ($($py312 --version))"

  if [ ! -d "$VENV_DIR" ]; then
    log "creating venv at $VENV_DIR"
    "$py312" -m venv "$VENV_DIR"
  else
    log "reusing existing venv at $VENV_DIR"
  fi

  log "upgrading pip + wheel"
  "$PIP" install --upgrade pip wheel >/dev/null

  log "installing torch ($mode)"
  if [ "$mode" = "cpu" ]; then
    "$PIP" install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
  else
    "$PIP" install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
  fi

  if [ -d "${PROJECT_ROOT}/arc_agi_3_wheels" ]; then
    log "installing bundled wheels from arc_agi_3_wheels/"
    "$PIP" install "${PROJECT_ROOT}/arc_agi_3_wheels"/*.whl
  else
    log "WARN: arc_agi_3_wheels/ not found — skipping bundled wheel install"
  fi

  log "installing remaining deps"
  "$PIP" install \
    numpy \
    Pillow \
    matplotlib \
    tqdm \
    requests

  log "setup complete. Run: bash scripts/run.sh check"
}

# ---------- command: check ----------

cmd_check() {
  require_venv
  "$PY" - <<'PY'
import sys, importlib, traceback

print(f"python: {sys.version.split()[0]}")
print(f"executable: {sys.executable}")

mods = ["torch", "arc_agi", "arcengine", "numpy", "PIL", "matplotlib"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "(no __version__)")
        print(f"  {m}: ok ({ver})")
    except Exception as e:
        print(f"  {m}: FAIL — {type(e).__name__}: {e}")

try:
    import torch
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
        print(f"  memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except Exception:
    traceback.print_exc()
PY

  log "checking project paths"
  for d in src environment_files arc_agi_3_wheels; do
    if [ -d "${PROJECT_ROOT}/$d" ]; then
      printf '  %s: ok\n' "$d"
    else
      printf '  %s: MISSING\n' "$d"
    fi
  done

  log "checking disk space"
  df -h "$PROJECT_ROOT" | head -2
  log "checking memory"
  free -h | head -2 2>/dev/null || vm_stat | head -5
}

# ---------- command: collect-quick ----------

cmd_collect_quick() {
  local games="sp80,ar25,lp85,r11l,cn04,ls20"
  local workers=8
  local output_name="wm_quick_v3"
  local profile="rtx4070super"
  while [ $# -gt 0 ]; do
    case "$1" in
      --games) games="$2"; shift 2;;
      --workers) workers="$2"; shift 2;;
      --output-name) output_name="$2"; shift 2;;
      --profile) profile="$2"; shift 2;;
      --help|-h)
        cat <<EOF
collect-quick — Step 1: small-subset collect.

Usage: bash scripts/run.sh collect-quick [options]

  --games G1,G2,...      Games to collect (default: sp80,ar25,lp85,r11l,cn04,ls20)
  --workers N            ProcessPool size (default: 8)
  --output-name NAME     Subdir under Collection_Cache (default: wm_quick_v3)
  --profile NAME         Hardware profile (default: rtx4070super)

Output: Local_Output/Collection_Cache/<output-name>/collected/episodes.jsonl.gz
EOF
        return 0;;
      *) die "unknown collect-quick option: $1";;
    esac
  done

  ensure_dir "${COLLECT_DIR}/${output_name}/collected"
  log "games=$games workers=$workers output_name=$output_name profile=$profile"
  spawn "collect_quick" "${output_name}" -m src.collect_staged \
    --project-root "$PROJECT_ROOT" \
    --output-root "${COLLECT_DIR}/${output_name}" \
    --hardware-profile "$profile" \
    --workers "$workers" \
    --games "$games"
}

# ---------- command: collect-full ----------

cmd_collect_full() {
  local workers=64
  local output_name="wm_full_v2_openlab"
  local profile="rtx4070super"
  while [ $# -gt 0 ]; do
    case "$1" in
      --workers) workers="$2"; shift 2;;
      --output-name) output_name="$2"; shift 2;;
      --profile) profile="$2"; shift 2;;
      --help|-h)
        cat <<EOF
collect-full — Step 3: full 25-game collect.

Usage: bash scripts/run.sh collect-full [options]

  --workers N            ProcessPool size (default: 64 — tune for your CPU/RAM)
  --output-name NAME     Subdir under Collection_Cache (default: wm_full_v2_openlab)
  --profile NAME         Hardware profile (default: rtx4070super)

Output: Local_Output/Collection_Cache/<output-name>/collected/episodes.jsonl.gz
Estimated wall: ~90-180 min at workers=64 on a CPU-rich box.
EOF
        return 0;;
      *) die "unknown collect-full option: $1";;
    esac
  done

  ensure_dir "${COLLECT_DIR}/${output_name}/collected"
  log "workers=$workers output_name=$output_name profile=$profile"
  spawn "collect_full" "${output_name}" -m src.collect_staged \
    --project-root "$PROJECT_ROOT" \
    --output-root "${COLLECT_DIR}/${output_name}" \
    --hardware-profile "$profile" \
    --workers "$workers"
}

# ---------- command: inspect ----------

cmd_inspect() {
  local data=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --data) data="$2"; shift 2;;
      --help|-h)
        cat <<EOF
inspect — Step 2: aggregate stats over a .gz dataset.

Usage: bash scripts/run.sh inspect --data <path/to/episodes.jsonl.gz>

Prints: total episodes, positive %, per-game counts/max-level/dead-click rate,
and pass-criteria check.
EOF
        return 0;;
      *) die "unknown inspect option: $1";;
    esac
  done
  [ -n "$data" ] || die "inspect requires --data <path>"
  [ -f "$data" ] || die "data file not found: $data"

  "$PY" - <<PY
from pathlib import Path
from collections import Counter, defaultdict
from statistics import median
import sys
sys.path.insert(0, "${PROJECT_ROOT}")
from src.common import iter_jsonl_gz

p = Path("$data")
per_game_count = Counter()
per_game_pos = Counter()
per_game_max_lvl = defaultdict(int)
per_game_dead_actions = defaultdict(list)
per_game_a6_count = defaultdict(int)
per_game_a6_zero = defaultdict(int)
total_eps = total_pos = total_lvl_ge2 = 0
final_state_counter = Counter()

for ep in iter_jsonl_gz(p):
    g = ep.get("game_id"); score = float(ep.get("score", 0.0) or 0.0)
    lvl = int(ep.get("levels_completed", 0) or 0)
    actions = int(ep.get("actions_taken", 0) or 0)
    final = str(ep.get("final_state"))
    total_eps += 1
    final_state_counter[final] += 1
    per_game_count[g] += 1
    if score > 0 or lvl > 0:
        per_game_pos[g] += 1; total_pos += 1
    if lvl >= 2: total_lvl_ge2 += 1
    per_game_max_lvl[g] = max(per_game_max_lvl[g], lvl)
    if score == 0 and lvl == 0:
        per_game_dead_actions[g].append(actions)
    for tr in ep.get("transitions", []):
        if int(tr.get("action_id", 0) or 0) == 6:
            per_game_a6_count[g] += 1
            if int(tr.get("delta_pixels", 0) or 0) == 0:
                per_game_a6_zero[g] += 1

print(f"=== {p.name} ===")
print(f"total_episodes: {total_eps}")
print(f"positive: {total_pos} ({100*total_pos/max(total_eps,1):.1f}%)")
print(f"levels_ge2: {total_lvl_ge2}")
print(f"final_states: {dict(final_state_counter)}")
print()
print(f"{'game':<10} {'eps':>4} {'pos':>4} {'max_lvl':>7} {'dead_med':>8} {'a6_zero%':>8}")
print("-" * 60)
for g in sorted(per_game_count.keys()):
    eps = per_game_count[g]; pos = per_game_pos[g]
    max_lvl = per_game_max_lvl[g]
    dead_med = int(median(per_game_dead_actions[g])) if per_game_dead_actions[g] else 0
    a6_total = per_game_a6_count[g]; a6_zero = per_game_a6_zero[g]
    a6_pct = (100*a6_zero/a6_total) if a6_total else 0.0
    print(f"{g:<10} {eps:>4} {pos:>4} {max_lvl:>7} {dead_med:>8} {a6_pct:>7.1f}%")
PY
}

# ---------- command: pretrain ----------

cmd_pretrain() {
  local data=""
  local output_name="pretrain_v3"
  local profile="rtx4070super"
  local epochs=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --data) data="$2"; shift 2;;
      --output-name) output_name="$2"; shift 2;;
      --profile) profile="$2"; shift 2;;
      --epochs) epochs="$2"; shift 2;;
      --help|-h)
        cat <<EOF
pretrain — Step 4: encoder pretraining.

Usage: bash scripts/run.sh pretrain --data <gz-path> [options]

  --data PATH            (required) Path to episodes.jsonl.gz (comma-sep for multiple)
  --output-name NAME     Subdir under Training_Output (default: pretrain_v3)
  --profile NAME         Hardware profile (default: rtx4070super)
  --epochs N             Override profile default
EOF
        return 0;;
      *) die "unknown pretrain option: $1";;
    esac
  done
  [ -n "$data" ] || die "pretrain requires --data"

  local args=(
    -m src.pretrain_encoder
    --data "$data"
    --output "${TRAIN_DIR}/${output_name}/encoder.pth"
    --hardware-profile "$profile"
    --log-every-batches 50
  )
  [ -n "$epochs" ] && args+=(--epochs "$epochs")

  ensure_dir "${TRAIN_DIR}/${output_name}"
  log "data=$data output_name=$output_name profile=$profile epochs=${epochs:-default}"
  spawn "pretrain" "${output_name}" "${args[@]}"
}

# ---------- command: train ----------

cmd_train() {
  local data=""
  local output_name="worldmodel_v2"
  local encoder=""
  local profile="rtx4070super"
  local epochs=""
  local extra=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --data) data="$2"; shift 2;;
      --output-name) output_name="$2"; shift 2;;
      --encoder|--pretrained-encoder) encoder="$2"; shift 2;;
      --profile) profile="$2"; shift 2;;
      --epochs) epochs="$2"; shift 2;;
      --) shift; extra+=("$@"); break;;
      --help|-h)
        cat <<EOF
train — Step 6: joint worldmodel training.

Usage: bash scripts/run.sh train --data <gz-path> [options]

  --data PATH            (required) Path to episodes.jsonl.gz
  --output-name NAME     Subdir under Training_Output (default: worldmodel_v2)
  --encoder PATH         Pretrained encoder.pth (optional but recommended)
  --profile NAME         Hardware profile (default: rtx4070super)
  --epochs N             Override profile default
  -- <extra args>        Anything after -- is passed verbatim to train_worldmodel.py
                         (e.g. --w-inverse-action 1.0 --freeze-encoder)
EOF
        return 0;;
      *) die "unknown train option: $1";;
    esac
  done
  [ -n "$data" ] || die "train requires --data"

  local args=(
    -m src.train_worldmodel
    --project-root "$PROJECT_ROOT"
    --data "$data"
    --output-dir "${TRAIN_DIR}/${output_name}"
    --hardware-profile "$profile"
    --num-workers 1
  )
  [ -n "$encoder" ] && args+=(--pretrained-encoder "$encoder")
  [ -n "$epochs" ] && args+=(--epochs "$epochs")
  [ ${#extra[@]} -gt 0 ] && args+=("${extra[@]}")

  ensure_dir "${TRAIN_DIR}/${output_name}/checkpoints"
  log "data=$data output_name=$output_name encoder=${encoder:-none} profile=$profile epochs=${epochs:-default}"
  spawn "train" "${output_name}" "${args[@]}"
}

# ---------- command: eval ----------

cmd_eval() {
  local checkpoint=""
  local output_name=""
  local games=""
  local k=""; local h=""; local maxsteps=""
  local extra=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --checkpoint) checkpoint="$2"; shift 2;;
      --output-name) output_name="$2"; shift 2;;
      --games) games="$2"; shift 2;;
      --K|--rollouts-per-candidate) k="$2"; shift 2;;
      --H|--rollout-horizon) h="$2"; shift 2;;
      --max-steps) maxsteps="$2"; shift 2;;
      --) shift; extra+=("$@"); break;;
      --help|-h)
        cat <<EOF
eval — Step 7: Phase 3 sanity eval.

Usage: bash scripts/run.sh eval --checkpoint <path> [options]

  --checkpoint PATH        (required) Path to worldmodel best.pth
  --output-name NAME       Eval JSON name (default: derived from checkpoint dir)
  --games G1,G2,...        Games to eval (default: all 25)
  --K N                    Rollouts per candidate (default: 1)
  --H N                    Rollout horizon (default: 5)
  --max-steps N            Max env actions per game (default: 192)
  -- <extra args>          Anything after -- is passed verbatim to evaluate_worldmodel.py
EOF
        return 0;;
      *) die "unknown eval option: $1";;
    esac
  done
  [ -n "$checkpoint" ] || die "eval requires --checkpoint"
  [ -f "$checkpoint" ] || die "checkpoint not found: $checkpoint"

  if [ -z "$output_name" ]; then
    output_name="eval_$(basename $(dirname $(dirname "$checkpoint")))"
    [ -n "$games" ] && output_name="${output_name}_$(echo "$games" | tr ',' '_')"
    [ -n "$k" ] && [ -n "$h" ] && output_name="${output_name}_K${k}H${h}"
  fi
  local output_path="$(dirname $(dirname "$checkpoint"))/${output_name}.json"

  local args=(
    -m src.evaluate_worldmodel
    --project-root "$PROJECT_ROOT"
    --checkpoint "$checkpoint"
    --output "$output_path"
  )
  [ -n "$games" ] && args+=(--games "$games")
  [ -n "$k" ] && args+=(--rollouts-per-candidate "$k")
  [ -n "$h" ] && args+=(--rollout-horizon "$h")
  [ -n "$maxsteps" ] && args+=(--max-steps "$maxsteps")
  [ ${#extra[@]} -gt 0 ] && args+=("${extra[@]}")

  ensure_dir "$(dirname "$output_path")"
  log "checkpoint=$checkpoint games=${games:-all} K=${k:-1} H=${h:-5} -> $output_path"
  spawn "eval" "${output_name}" "${args[@]}"
}

# ---------- command: status ----------

cmd_status() {
  log "running pipeline processes:"
  ps -ef | grep -E "(src\.collect_staged|src\.train_worldmodel|src\.pretrain_encoder|src\.evaluate_worldmodel)" | grep -v grep | head -20 || echo "(none running)"
  echo
  log "memory:"
  free -h 2>/dev/null | head -2 || vm_stat | head -5
  echo
  log "saved pid files (recursive):"
  while IFS= read -r pf; do
    [ -f "$pf" ] || continue
    local pid="$(cat "$pf")"
    local rel="${pf#$LOG_DIR/}"
    local label="${rel%.pid}"      # e.g. wm_full_v2_openlab/collect_full
    if kill -0 "$pid" 2>/dev/null; then
      printf '  %-50s pid=%s ALIVE\n' "$label" "$pid"
    else
      printf '  %-50s pid=%s dead\n' "$label" "$pid"
    fi
  done < <(find "$LOG_DIR" -type f -name "*.pid" 2>/dev/null | sort)
  echo
  log "recent logs (last 5, across all subdirs):"
  find "$LOG_DIR" -type f -name "*.log" -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr | head -5 | awk '{print "  " substr($0, index($0,$2))}'
}

# ---------- command: logs ----------

cmd_logs() {
  local pattern="${1:-}"
  local target
  if [ -n "$pattern" ]; then
    # search across all subdirs; match pattern against relative path
    target="$(find "$LOG_DIR" -type f -name "*.log" 2>/dev/null \
      | grep "$pattern" \
      | xargs -I{} stat -c "%Y {}" {} 2>/dev/null \
      | sort -nr | head -1 | awk '{print $2}')"
  else
    target="$(find "$LOG_DIR" -type f -name "*.log" -printf "%T@ %p\n" 2>/dev/null \
      | sort -nr | head -1 | awk '{print $2}')"
  fi
  [ -n "$target" ] || die "no matching log found (searched $LOG_DIR)"
  log "tailing: $target"
  tail -f "$target"
}

# ---------- command: help ----------

cmd_help() {
  if [ $# -eq 0 ]; then
    sed -n '3,25p' "${BASH_SOURCE[0]}"
  else
    "cmd_${1//-/_}" --help
  fi
}

# ---------- dispatch ----------

case "${1:-help}" in
  setup) shift; cmd_setup "$@";;
  check) shift; cmd_check "$@";;
  collect-quick) shift; cmd_collect_quick "$@";;
  collect-full) shift; cmd_collect_full "$@";;
  inspect) shift; cmd_inspect "$@";;
  pretrain) shift; cmd_pretrain "$@";;
  train) shift; cmd_train "$@";;
  eval) shift; cmd_eval "$@";;
  status) shift; cmd_status "$@";;
  logs) shift; cmd_logs "$@";;
  help|--help|-h) shift; cmd_help "$@";;
  *) die "unknown command: $1. Run: bash scripts/run.sh help";;
esac
