# Setup, Commands & Reference

This is the operational manual: how to install the venv on Windows or
macOS, how to run each stage of the pipeline, what files come out, and
what to do when something breaks. For *what the pipeline is and why it
is built this way*, read [README.md](README.md) first.

**Contents**

1. [Install — Windows (PowerShell)](#1-install--windows-powershell)
2. [Install — macOS (zsh / bash)](#2-install--macos-zsh--bash)
3. [Verify the install](#3-verify-the-install)
4. [Running the pipeline](#4-running-the-pipeline)
5. [Training](#5-training)
6. [Output layout](#6-output-layout)
7. [Triage labels](#7-triage-labels)
8. [Files in the repo](#8-files-in-the-repo)
9. [Openlab (Linux) collect](#9-openlab-linux-collect)
10. [Troubleshooting](#10-troubleshooting)
11. [Entry-point dependency table](#11-entry-point-dependency-table)

If you only need to fix `ModuleNotFoundError: No module named 'torch'`,
jump to [Quick fix: torch missing](#quick-fix-torch-missing).

---

## 1. Install — Windows (PowerShell)

### 1.1 Pick a Python

Use **Python 3.12 or 3.13 (64-bit)**. Both have prebuilt wheels for
`torch`, `numpy`, `pillow`, and `matplotlib` on PyPI.

```powershell
python --version          # expect 3.12.x or 3.13.x
py -3.13 --version        # explicit launcher if needed
```

### 1.2 Create and activate a virtualenv

From the repo root:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
```

If activation is blocked by execution policy, run once per user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 1.3 Install `arc_agi` and `arcengine` from the bundled wheels

```powershell
pip install ".\arc_agi_3_wheels\arc_agi-0.9.8-py3-none-any.whl"
pip install ".\arc_agi_3_wheels\arcengine-0.9.3-py3-none-any.whl"
```

> **Do not bulk-install `arc_agi_3_wheels/` on Windows.** Most files in
> that folder are `manylinux` / `cp312-cp312` Linux wheels (numpy, pillow,
> matplotlib, contourpy, kiwisolver, fonttools, markupsafe, pydantic_core,
> charset_normalizer). Only the two pure-Python wheels above are
> Windows-safe; everything else comes from PyPI via the requirements
> files below. The bulk install is for the Linux / Openlab path only.

### 1.4 Install PyTorch

CPU-only (simplest):

```powershell
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

GPU (CUDA 12.1 — adjust `cuXXX` for your driver):

```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

### 1.5 Install the rest of the deps

Probe-first only (`collect_probe`, `collect_probe_staged`, `eval_probe`,
`probe_triage`):

```powershell
pip install -r requirements_probe.txt
```

Full pipeline (adds `train`, `evaluate`, `competition`, `inspect_collect`,
`import_human_replays`):

```powershell
pip install -r requirements_full.txt
```

Both files leave `arc_agi`, `arcengine`, and `torch` out on purpose —
those were installed by the dedicated steps above.

---

## 2. Install — macOS (zsh / bash)

The macOS path mirrors Windows with two differences: only the two
pure-Python wheels can be installed by hand from `arc_agi_3_wheels/`
(everything else comes from PyPI), and shell line-continuation uses `\`.

### 2.1 Pick a Python

```bash
brew install python@3.13
python3.13 --version
```

### 2.2 Create and activate a virtualenv

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
```

### 2.3 Install `arc_agi` and `arcengine` from the bundled wheels

```bash
pip install ./arc_agi_3_wheels/arc_agi-0.9.8-py3-none-any.whl
pip install ./arc_agi_3_wheels/arcengine-0.9.3-py3-none-any.whl
```

> **Do not run `pip install arc_agi_3_wheels/*.whl` on macOS.** The other
> wheels in that folder are Linux-targeted and will fail.

### 2.4 Install PyTorch

```bash
pip install torch
```

The default PyPI wheel works on Apple Silicon (uses MPS automatically on
M1/M2/M3). There is no separate CUDA build for macOS.

### 2.5 Install the rest of the deps

```bash
pip install -r requirements_probe.txt   # probe-first only
# or
pip install -r requirements_full.txt    # full pipeline
```

---

## 3. Verify the install

One-liner that imports everything the probe-first path needs:

```bash
python -c "import torch, arc_agi, arcengine; from src import collect_probe, eval_probe, probe_agent, probe_memory, effect_signatures; print('OK', torch.__version__)"
```

Expected: `OK <torch version>`.

If you installed the full profile, also check training imports:

```bash
python -c "from src import train, agent, model; print('train imports OK')"
```

---

## 4. Running the pipeline

All commands assume the venv is active and the working directory is the
repo root. Outputs land under `Local_Output/Probe_Cache/<run_name>/`.

### 4.1 Single-pass focus-5 (smoke test)

Runs `collect_probe` on `sp80,lp85,ar25,ls20,r11l` then `eval_probe`.
Wall-clock ~100 s on a laptop CPU.

```powershell
.\scripts\run_focus5_probe_v3_1_local.ps1                  # Windows
```
```bash
bash scripts/run_focus5_probe_v3_1_local.sh                # macOS / Linux
```

Override the game list or seeds via env vars:

```bash
PROBE_GAMES="sp80,lp85" PROBE_SEEDS="0,1" \
  bash scripts/run_focus5_probe_v3_1_local.sh
```

### 4.2 Staged focus-5 (with priors and triage)

Runs five stages per game (`probe_seed`, `exploit_safe`, `followup_focus`,
`rescue_reprobe`, `harvest_best`), recomputing per-game priors before
each stage. Each stage gets its own knob overrides (e.g. `probe_seed`
disables anti-loop to build a clean baseline; `rescue_reprobe` raises the
reprobe budget for stuck games).

```powershell
.\scripts\run_staged_focus5_local.ps1                      # Windows
```
```bash
python -m src.collect_probe_staged \
  --project-root "." \
  --output-root "./Local_Output/Probe_Cache/probe_focus5_staged_$(date +%Y%m%d_%H%M%S)" \
  --games "sp80,lp85,ar25,ls20,r11l" \
  --episodes-per-game 32 \
  --budget-allocator adaptive
```

### 4.3 25-game broad first pass

~250 episodes total, ~6.5 min on CPU. Produces the priors that the
harvest pass reads.

```powershell
.\scripts\run_staged_all_local.ps1                         # Windows
```

### 4.4 Harvest pass on triaged games

Reuses the priors from the broad run and re-collects only the games
labeled `promising`, `signal_but_stuck`, `click_promising`, or
`movement_promising`. Per-stage budgets lean toward `harvest_best`,
`rescue_reprobe`, and `followup_focus`.

```powershell
.\scripts\run_harvest_promising_local.ps1 <broad_run_name>
```

### 4.5 Re-run triage on an existing collection

Cheap: reads `episodes.jsonl.gz` and rewrites the `triage/` folder.
Useful after editing `probe_triage` thresholds.

```powershell
.\scripts\run_triage_local.ps1 <run_name>                  # Windows
```
```bash
python -m src.probe_triage \
  --input    "./Local_Output/Probe_Cache/<run_name>/collected/episodes.jsonl.gz" \
  --priors   "./Local_Output/Probe_Cache/<run_name>/priors" \
  --output-dir "./Local_Output/Probe_Cache/<run_name>/triage"
```

### 4.6 Re-run eval on an existing collection

```bash
python -m src.eval_probe \
  --input  "./Local_Output/Probe_Cache/<run_name>/collected/episodes.jsonl.gz" \
  --output "./Local_Output/Probe_Cache/<run_name>/probe_eval.json" \
  --label  probe_v3_1 \
  --print-overall
```

Useful fields in the eval JSON:

- `loop_metrics.totals.{cycle_2_count, cycle_3_count, repeated_state_action_count}`
- `loop_metrics.local_followup_success_rate`
- `loop_metrics.totals.{stagnation_escapes, reprobe_windows_opened, reprobe_steps_used}`
- `loop_metrics.totals.{escape_blocked_by_cooldown, reprobe_blocked_by_cap, reprobe_blocked_by_recent_progress}`

---

## 5. Training

`src.train` accepts a comma-separated `--data` argument so multiple
collected `.gz` files merge into a single checkpoint without an
intermediate format:

```bash
python -m src.train \
  --project-root "." \
  --data './path/to/broad.gz,./path/to/harvest.gz,./path/to/focus5.gz' \
  --output-dir './Local_Output/Training/team_focus_train_v1'
```

Training writes:

```
Local_Output/Training/<tag>/
├── metrics.csv
├── checkpoints/best.pth
├── checkpoints/last.pth
├── checkpoints/interrupt.pth         only when training is stopped manually
└── summary.json
```

Hardware profiles and recommended hyperparameters live in
[README.md "Hardware"](README.md#hardware).

The Colab notebook (`ColabNotebook/train_arc_agi3_colab.ipynb`) reads
cached trajectories from `ARC Prize 2026_AGI_3/Collection_Cache/<collect_tag>/`
and writes outputs to `ARC Prize 2026_AGI_3/Training_Output/<timestamp>/`.
To reuse a locally-collected run in Colab, copy the run folder into
`Collection_Cache/`, then set `RUN_COLLECTION = False` and
`COLLECT_TAG = '<run_name>'` in the notebook.

---

## 6. Output layout

A complete pipeline run produces:

```
Local_Output/Probe_Cache/<run_name>/
├── collected/episodes.jsonl.gz       one line per episode, gzipped JSON
├── probe_eval.json                   per-game + overall metrics
├── priors/<game_id>.json             per-game persistent state
└── triage/
    ├── triage_summary.json           multi-label per-game classification
    ├── triage_summary.csv            same, flat CSV for spreadsheets
    └── per_game_stage_summary.json   rates split by (game, stage)
```

Each line in `episodes.jsonl.gz` is one episode dict with:

- `game_id`, `seed`, `score`, `level_scores`, `max_score`
- `transitions[]` — each carries `phase` (`probe` / `exploit` / `escape` /
  `followup` / `reprobe`), `effect_signature`, and a richer `effect`
  payload
- `memory_summary` — final per-action stats (`trials`, signature counts,
  `role` ∈ {`progress`, `navigation`, `interaction`, `global`, `dead`,
  `uncertain`, `unknown`})
- `signature_counts`, `action_roles`
- `stage`, `episode_index_in_game`, `episode_index_global` (staged
  collector only)

`priors/<game_id>.json` carries dead actions, dead coords, promising
clicks, color affordances, stage stats, best seed, and the running color
accumulator. Subsequent stages and runs warm-start from this file.

---

## 7. Triage labels

Assigned per game from observed counts only. A game can carry several
labels at once.

| Label | Condition |
|---|---|
| `promising`           | `best_score > 0` or `progress_rate >= 0.2` |
| `signal_but_stuck`    | plenty of `LOCAL_TOGGLE` / `GLOBAL_CHANGE`, no progress |
| `low_signal`          | mostly `NO_CHANGE` / `SMALL_CHANGE` |
| `dead_or_noisy`       | very high dead-action rate, no progress |
| `click_promising`     | `ACTION6` effective rate `>= 0.4` |
| `movement_promising`  | `MOTION_LIKE` share of effective signatures `>= 0.35` |

The harvest pass reads `triage_summary.json` and `--include-labels` to
pick which games to revisit.

---

## 8. Files in the repo

```
src/
├── collect_probe.py          single-pass probe-first collector
├── collect_probe_staged.py   five-stage collector with per-game priors
├── probe_agent.py            two-phase (probe / exploit) agent
├── probe_memory.py           per-episode action + coord effect memory
├── effect_signatures.py      transition classifier
├── color_features.py         color observation primitives
├── eval_probe.py             per-game + overall metrics
├── probe_triage.py           multi-label per-game classification
├── collect_probe.py          single-pass collector entry point
├── train.py                  policy model training
├── evaluate.py               public-game evaluation
├── competition.py            competition-mode runner
├── inspect_collect.py        GIF / HTML inspection of collected episodes
└── import_human_replays.py   convert human replay traces into episodes

scripts/
├── run_focus5_probe_v3_1_local.{ps1,sh}   single-pass focus-5
├── run_focus5_probe_local.sh              older single-pass focus-5 (no v3 knobs)
├── run_focus5_probe_v3_local.sh           v3 single-pass focus-5
├── run_staged_focus5_local.ps1            staged focus-5 + triage
├── run_staged_all_local.ps1               25-game broad pass
├── run_harvest_promising_local.ps1        harvest pass on triaged games
└── run_triage_local.ps1                   re-run triage on existing run

ColabNotebook/
└── train_arc_agi3_colab.ipynb             Colab training entry point

arc_agi_3_wheels/                          bundled dataset-package wheels
ARC-AGI-3-Agents/                          official reference assets
docs/gifs/                                 example trajectories shown in README
```

---

## 9. Openlab (Linux) collect

For longer CPU-heavy collection runs on UCI ICS Openlab, use Slurm
instead of an interactive shell — long-running non-Slurm processes can
be reniced or suspended on shared nodes.

Copy the project to Openlab:

```bash
rsync -av --delete --exclude '.git' \
  '/Users/wangyiding/ARC Prize 2026 - ARC-AGI-3/' \
  yidingw6@openlab.ics.uci.edu:~/arc_agi3/
```

Set up the venv from the bundled Linux wheels — Linux is the one platform
where the bulk install of `arc_agi_3_wheels/` is correct:

```bash
cd ~/arc_agi3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install arc_agi_3_wheels/*.whl
```

Single-node parallel collect:

```bash
python -m src.collect \
  --project-root "." \
  --output-root "./Local_Output/Collection_Cache/openlab_search_v1" \
  --hardware-profile a100 \
  --seeds 0,1,2,3 \
  --max-steps 96 \
  --workers 16
```

`--workers` parallelizes collection across CPU processes on one node.
Increase only as far as the node's memory and process limits allow.

---

## 10. Troubleshooting

### Quick fix: torch missing

If `arc_agi` is already installed and the only failure is
`ModuleNotFoundError: No module named 'torch'`:

```powershell
# Windows, CPU-only
pip install --index-url https://download.pytorch.org/whl/cpu torch

# Windows, GPU (CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```
```bash
# macOS
pip install torch
```

### `ModuleNotFoundError: No module named 'arc_agi'` / `'arcengine'`

The wheels were installed in a different venv. Reactivate the right one
and reinstall:

```powershell
.\.venv313\Scripts\Activate.ps1
pip install ".\arc_agi_3_wheels\arc_agi-0.9.8-py3-none-any.whl"
pip install ".\arc_agi_3_wheels\arcengine-0.9.3-py3-none-any.whl"
python -c "import arc_agi, arcengine; print(arc_agi.__file__); print(arcengine.__file__)"
```

The printed paths should be inside your venv's `site-packages`.

### `ERROR: ... is not a supported wheel on this platform`

A Linux wheel from `arc_agi_3_wheels/` was installed on Windows or macOS
(filenames ending `manylinux*.whl` or `cp312-cp312-*.whl`). Install only
`arc_agi-0.9.8-py3-none-any.whl` and `arcengine-0.9.3-py3-none-any.whl`
from that folder; the rest comes from PyPI via the requirements files.

### Wrong Python version

If `python --version` is 3.10 or older, `pip install torch` may fail or
install an old build. Recreate the venv with `py -3.13` (Windows) or
`python3.13` (macOS). If it is 3.14+, drop back to 3.13 — wheels for the
newest line may not exist yet.

### `Activate.ps1 cannot be loaded because running scripts is disabled`

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Then reopen PowerShell.

### `python` runs the Microsoft Store stub

After activation, `where.exe python` should point inside
`.venv313\Scripts\`. If it points to `WindowsApps`, re-run `Activate.ps1`
from the repo root, or call `.\.venv313\Scripts\python.exe` directly.

### `pip install torch` is extremely slow

The default PyPI `torch` wheel is large (CUDA bundles especially). On
Windows, use the CPU-only index from step 1.4 if you do not need GPU.

---

## 11. Entry-point dependency table

| Entry point | Direct third-party imports | Transitive (via `src/common.py`) |
|---|---|---|
| `src.collect_probe`         | `arc_agi`, `arcengine` | `torch` |
| `src.collect_probe_staged`  | `arc_agi`, `arcengine` | `torch` |
| `src.eval_probe`            | (none)                 | `torch` |
| `src.probe_triage`          | (none)                 | `torch` |
| `src.probe_agent`           | `arcengine`            | `torch` |
| `src.collect`               | `arc_agi`, `arcengine` | `torch` |
| `src.collect_staged`        | `arcengine`            | `torch` |
| `src.train`                 | `arc_agi`, `torch`     | `torch` |
| `src.evaluate`              | `arc_agi`              | `torch` |
| `src.competition`           | `arc_agi`              | `torch` |
| `src.inspect_collect`       | `Pillow`               | `torch` |
| `src.import_human_replays`  | (none)                 | `torch` |

The right column means: even if the script does not `import torch`
itself, it imports `src.common`, which imports `torch` at module load.
That is why **`torch` is required for every entry point in this repo.**
