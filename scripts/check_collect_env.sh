#!/usr/bin/env bash
#
# scripts/check_collect_env.sh — Verify the env is ready to run the collect.
#
# Checks (in order):
#   1. Python 3.12 is on PATH
#   2. Required packages: torch, numpy, arc_agi, arcengine
#   3. Project structure: scripts/, src/, kaggle_notebook/
#   4. GT replays present in environment_files/<game>/replays/
#   5. Per-game priors present at Local_Output/per_game_priors.json
#   6. The collect script can be invoked (--help works)
#   7. A single-game smoke run completes
#
# Usage:
#   bash scripts/check_collect_env.sh           # quick checks (no env runs)
#   bash scripts/check_collect_env.sh --smoke   # also run a 30s collect smoke
#
# Exit code 0 = all clear, 1+ = problems found (count = number of issues).

set -u  # don't set -e — we want to count failures

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PY:-python}"
ISSUES=0
SMOKE=0

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1;;
    --help|-h) head -25 "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0;;
  esac
done

ok()    { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail()  { printf '  \033[31m✗\033[0m %s\n' "$1"; ISSUES=$((ISSUES+1)); }
warn()  { printf '  \033[33m!\033[0m %s\n' "$1"; }
section(){ printf '\n\033[1m%s\033[0m\n' "$1"; }

# ---------- 1. Python ----------
section "1. Python interpreter"
if ! command -v "$PY" >/dev/null 2>&1; then
  fail "no python found (PY=$PY). Activate your venv or set PY=/path/to/python."
else
  PYV="$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)"
  PYPATH="$("$PY" -c 'import sys; print(sys.executable)' 2>/dev/null)"
  if [[ "$PYV" =~ ^3\.12\. ]]; then
    ok "Python $PYV ($PYPATH)"
  else
    warn "Python $PYV at $PYPATH (expected 3.12.x — bundled wheels are cp312)"
  fi
fi

# ---------- 2. Packages ----------
section "2. Required packages"
"$PY" - <<'PY' 2>&1 | sed 's/^/  /'
import sys
import importlib
checks = [
    ("torch",     "torch",      "CPU OK; CUDA optional for collect"),
    ("numpy",     "numpy",      ""),
    ("arc_agi",   "arc_agi",    "Arcade env loader"),
    ("arcengine", "arcengine",  "GameAction enum + state machine"),
]
fails = 0
for label, mod, note in checks:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "(no __version__)")
        suffix = f" — {note}" if note else ""
        print(f"\033[32m✓\033[0m {label:10s} {ver}{suffix}")
    except Exception as e:
        print(f"\033[31m✗\033[0m {label:10s} import failed: {type(e).__name__}: {e}")
        fails += 1
try:
    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        print(f"\033[32m✓\033[0m torch.cuda  available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\033[33m!\033[0m torch.cuda  not available (collect doesn't need it; ok)")
except Exception:
    pass
sys.exit(fails)
PY
PY_FAILS=$?
if [ "$PY_FAILS" -gt 0 ]; then ISSUES=$((ISSUES+PY_FAILS)); fi

# ---------- 3. Project structure ----------
section "3. Project structure"
for d in scripts src kaggle_notebook arc_agi_3_wheels environment_files; do
  if [ -d "$d" ]; then ok "$d/"; else fail "$d/ missing"; fi
done
for f in scripts/collect_gt_warmstart.py scripts/run_openlab_collect.sh kaggle_notebook/my_agent.py src/agent.py src/common.py; do
  if [ -f "$f" ]; then ok "$f"; else fail "$f missing"; fi
done

# ---------- 4. GT replays ----------
section "4. GT replays (environment_files/*/replays/)"
N_GAMES_WITH_REPLAYS=0
N_GAMES=0
for d in environment_files/*/; do
  game=$(basename "$d")
  N_GAMES=$((N_GAMES+1))
  rep_dir="$d/replays"
  if [ -d "$rep_dir" ] && [ "$(ls "$rep_dir"/*.json 2>/dev/null | wc -l)" -gt 0 ]; then
    N_GAMES_WITH_REPLAYS=$((N_GAMES_WITH_REPLAYS+1))
  fi
done
if [ "$N_GAMES_WITH_REPLAYS" -eq "$N_GAMES" ] && [ "$N_GAMES" -ge 25 ]; then
  TOTAL_SIZE=$(du -ch environment_files/*/replays/*.json 2>/dev/null | tail -1 | cut -f1)
  ok "$N_GAMES_WITH_REPLAYS/$N_GAMES games have replay files (total: $TOTAL_SIZE)"
elif [ "$N_GAMES_WITH_REPLAYS" -eq 0 ]; then
  fail "0/$N_GAMES games have replays — Buckets 1, 3, 4 will be DEGENERATE!"
  fail "  → SCP environment_files/*/replays/ from local box (~586 MB)"
else
  fail "$N_GAMES_WITH_REPLAYS/$N_GAMES games have replays — partial!"
fi

# ---------- 5. Per-game priors ----------
section "5. Per-game priors (Local_Output/per_game_priors.json)"
PRIORS="Local_Output/per_game_priors.json"
if [ -f "$PRIORS" ]; then
  SIZE=$(stat -c %s "$PRIORS" 2>/dev/null || stat -f %z "$PRIORS" 2>/dev/null)
  N_KEYS=$("$PY" -c "import json; print(len(json.load(open('$PRIORS'))))" 2>/dev/null || echo "?")
  if [ "$N_KEYS" -ge 25 ]; then
    ok "$PRIORS ($SIZE bytes, $N_KEYS games)"
  else
    warn "$PRIORS exists but only $N_KEYS keys — expected 25"
  fi
else
  fail "$PRIORS missing — Phase B will use default 'no_prior' (degraded)"
  fail "  → SCP from local: scp ${PRIORS} <openlab>:~/path/Local_Output/"
fi

# ---------- 6. Collect script invocable ----------
section "6. collect_gt_warmstart.py --help"
if "$PY" scripts/collect_gt_warmstart.py --help >/dev/null 2>&1; then
  ok "imports + argparse OK"
else
  fail "scripts/collect_gt_warmstart.py --help errored. Run manually for details:"
  fail "  $PY scripts/collect_gt_warmstart.py --help"
fi

# ---------- 7. Single-game smoke (optional) ----------
if [ "$SMOKE" -eq 1 ]; then
  section "7. Single-game smoke run (~30 sec)"
  TMP_OUT=$(mktemp -d)
  if "$PY" scripts/collect_gt_warmstart.py \
        --output-root "$TMP_OUT" \
        --games r11l \
        --deviation-points 0,100 \
        --seeds-per-deviation 1 \
        --strategies random_full \
        --perturb-rates "" \
        --workers 1 \
        --max-steps 500 >"$TMP_OUT/log" 2>&1; then
    N_EPS=$(zcat "$TMP_OUT/collected/episodes.jsonl.gz" 2>/dev/null | wc -l)
    if [ "$N_EPS" -ge 1 ]; then
      ok "smoke run produced $N_EPS episode(s)"
      # Inspect priors loading
      if grep -q "prior=(no_prior)" "$TMP_OUT/log" 2>/dev/null; then
        warn "prior=(no_prior) seen in smoke log — Phase B is using defaults (degraded). Set ARC_PRIORS_PATH or place priors at Local_Output/per_game_priors.json"
      else
        ok "priors loaded successfully"
      fi
    else
      fail "smoke run completed but produced no episodes"
    fi
  else
    fail "smoke run errored. Tail of log:"
    tail -20 "$TMP_OUT/log" | sed 's/^/    /'
  fi
  rm -rf "$TMP_OUT" 2>/dev/null
fi

# ---------- summary ----------
section "Summary"
if [ "$ISSUES" -eq 0 ]; then
  printf '  \033[32mAll checks passed.\033[0m Ready to launch:\n'
  printf '    bash scripts/run_openlab_collect.sh --workers 96 --seeds 16 --perturb-seeds 16\n'
  exit 0
else
  printf '  \033[31m%d issue(s) found.\033[0m Fix before launching the collect.\n' "$ISSUES"
  exit "$ISSUES"
fi
