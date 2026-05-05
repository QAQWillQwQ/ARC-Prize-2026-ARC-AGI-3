# Pipeline Overview

This document walks through the full ARC-AGI-3 probe-first pipeline from raw
environment files to a trained checkpoint and an evaluation report. It is the
companion to the top-level `README.md`, which is mostly about installation
and running commands. This document is about *what each stage does* and how
the artifacts connect.

---

## 1. Environment files

ARC-AGI-3 ships 25 public games as Python environment definitions in
`environment_files/<game_id>/<env_hash>/`. Each game folder contains:

- `<game_id>.py` — the environment implementation
- `metadata.json` — baseline action list, grid size, scoring rules

The `arc_agi.Arcade` constructor takes the parent directory and discovers
all games at startup. The pipeline expects `environment_files/` to live at the
project root so `--project-root .` resolves correctly.

The `arcengine` package supplies the `GameAction` and `GameState` types and is
shared by all collectors and evaluators. Both `arc_agi` and `arcengine` are
pure-Python wheels and ship in `wheels/`.

## 2. Game selection

Game subsets are not encoded in the source — they are passed in by command
line and configs:

- **Focus-5**: `sp80,lp85,ar25,ls20,r11l`. Three positive-core games where
  level 1 is reachable, plus two control games that surface stalling and
  death loops. Used for fast method iteration.
- **Broad-25**: all 25 public games via `--all-public-games`. Used to build
  the per-game prior set.
- **Harvest subset**: derived from the broad-25 triage output via
  `--restrict-from-triage triage_summary.json --include-labels ...`.

The harvest subset is not a static list. It depends on what the broad pass
observed.

## 3. Probing (collect/probe.py)

The probe-first collector runs `ProbeFirstAgent` from `agents/probe_agent.py`.
For every episode:

1. **Probe phase** (~16 steps). Every non-coordinate action is tried at least
   once. A diverse set of `ACTION6` click candidates is sampled from a coarse
   grid and from any salient pixels detected in recent frames.
2. **Exploit phase** (remaining budget). Action priorities, click region
   scoring, and dead-action pruning all read from the probe memory built in
   phase 1. Anti-loop logic (repeat penalty, 2-cycle and 3-cycle detection,
   stagnation escape with cooldown, local follow-up window, global-change
   reprobe budget) emits counters into `loop_metrics.totals` so regressions
   show up in the eval JSON instead of the GIF.

Each episode is appended to `outputs/probe_cache/<run>/collected/episodes.jsonl.gz`
with the schema described in §6.

## 4. Staged collection (collect/probe_staged.py)

The staged collector runs five stages per game:

1. `probe_seed` — anti-loop disabled, builds a clean baseline of action
   effects.
2. `exploit_safe` — uses the seed memory; safe priorities only.
3. `followup_focus` — biases toward action sequences that produced
   `LOCAL_TOGGLE` or `GLOBAL_CHANGE` in earlier stages.
4. `rescue_reprobe` — for stuck games, raises the reprobe budget and triggers
   re-exploration after sustained no-progress windows.
5. `harvest_best` — uses the highest-scoring seed and prior to push for
   maximum progress.

Per-game priors are recomputed from the running episode pool before each
stage and persisted to `outputs/probe_cache/<run>/priors/<game_id>.json`. The
adaptive budget allocator can shift episode budget across stages based on
which stage has been most productive on a given game.

## 5. Episode artifacts

Each line in `episodes.jsonl.gz` is one episode dict:

```json
{
  "game_id": "sp80",
  "seed": 0,
  "score": 1,
  "level_scores": [1, 0],
  "max_score": 5,
  "transitions": [
    {
      "action_id": 5,
      "phase": "probe",                     // probe / exploit / escape / followup / reprobe
      "effect_signature": "local_toggle",   // see §7
      "effect": { ... }                     // richer payload
    }
  ],
  "memory_summary": {
    "1": { "trials": 4, "no_change": 2, "small_change": 1, "progress": 1, "role": "progress" }
  },
  "signature_counts": { "no_change": 12, "progress": 1, ... },
  "action_roles": { "1": "progress", "2": "navigation", "5": "dead", ... },
  "stage": "exploit_safe",                  // staged collector only
  "episode_index_in_game": 3,
  "episode_index_global": 47
}
```

`memory_summary` is the per-action effect log built up during the episode.
`role` is one of `progress`, `navigation`, `interaction`, `global`, `dead`,
`uncertain`, `unknown`.

## 6. Effect signatures (utils/effect_signatures.py)

Every transition is classified into one of:

| Signature       | Meaning |
|---              |---|
| `no_change`     | frame delta below `SMALL_DELTA_LIMIT` |
| `small_change`  | small localized delta, no shape change |
| `motion_like`   | one connected component shifted by a small offset |
| `local_toggle`  | small bounded change with shape difference |
| `global_change` | most of the frame changed |
| `progress`      | score increased |
| `game_over`     | terminal `GAME_OVER` |
| `win`           | terminal `WIN` |

The classifier is cheap (frame delta + connected components), stateless, and
can be replaced by a learned one without touching the rest of the pipeline.

## 7. Color features (utils/color_features.py)

Color is exposed as observation features only — color histograms, dominant
colors, change-shape labels, scene-context keys. There are no per-game
"color X means Y" rules anywhere in the pipeline. Color affordances are
learned per-game and accumulated in
`priors/<game_id>.json["color_accumulator"]`. Refined color contexts replace
coarse lift estimates after enough trials (≥3).

## 8. Triage (eval/triage.py)

Per-game multi-label classification driven by counts on the collected
episodes:

| Label                 | Condition |
|---                    |---|
| `promising`           | `best_score > 0` or `progress_rate >= 0.2` |
| `signal_but_stuck`    | plenty of `LOCAL_TOGGLE` / `GLOBAL_CHANGE`, no progress |
| `low_signal`          | mostly `NO_CHANGE` / `SMALL_CHANGE` |
| `dead_or_noisy`       | very high dead-action rate, no progress |
| `click_promising`     | `ACTION6` effective rate `>= 0.4` |
| `movement_promising`  | `MOTION_LIKE` share of effective signatures `>= 0.35` |

Labels are derived purely from observed counts — there is no static "good
games" list. A game can carry several labels at once. The harvest pass
consumes `--include-labels` to decide what to revisit.

## 9. Probe evaluation (eval/probe.py)

`probe_eval.json` reports per-game and overall metrics:

- `levels_after` distribution
- `signature_counts`
- `progress_rate`, `effective_action_rate`, `click_effective_rate`
- `loop_metrics.totals.{cycle_2_count, cycle_3_count, repeated_state_action_count}`
- `loop_metrics.local_followup_success_rate`
- `loop_metrics.totals.{stagnation_escapes, reprobe_windows_opened, reprobe_steps_used}`
- `loop_metrics.totals.{escape_blocked_by_cooldown, reprobe_blocked_by_cap, reprobe_blocked_by_recent_progress}`

These are the numbers a regression check should watch.

## 10. Training (train/train.py)

`src.train.train` accepts a comma-separated `--data` argument so multiple
collected `.gz` files merge into a single checkpoint without an intermediate
format. The model is the object-centric policy in `agents/model.py`. It
predicts:

- next action distribution (8 actions including `ACTION6` click)
- click position (heatmap over a coarse grid)
- value estimate
- a small transition target for the next latent state (auxiliary loss)

Outputs:

```
outputs/training/<run>/
├── metrics.csv
├── checkpoints/best.pth
├── checkpoints/last.pth
├── checkpoints/interrupt.pth   # only when training is stopped manually
└── summary.json
```

## 11. Evaluation (eval/policy.py)

`src.eval.policy` runs the trained checkpoint on the public games using
`PolicyGuidedAgent` from `agents/policy_agent.py`. It writes a per-game JSON
with levels completed, actions taken, and the final state for each game.

For real competition submissions, `src.eval.competition` runs in
`OperationMode.COMPETITION` (one scorecard per run, each environment created
once, scorecard closed at the end).

## 12. Inspection (eval/inspect_run.py)

`src.eval.inspect_run` reads an `episodes.jsonl.gz` and produces GIFs and
HTML summaries useful for debugging individual episodes — what the agent saw,
what action it took, and what effect signature came out. It is not part of
the automated pipeline; it is a debugging tool.

---

## End-to-end run sequence

```
.\scripts\run_probe.ps1                       # smoke (~100s)
.\scripts\run_staged_collection.ps1           # focus-5 staged (~10 min)
.\scripts\run_broad_collection.ps1            # 25-game broad (~6.5 min)
.\scripts\run_harvest.ps1 <broad_run_name>    # harvest (~6.6 min)
.\scripts\train_probe_model.ps1 <train_name>  # training (GPU recommended)
.\scripts\evaluate_checkpoint.ps1 outputs\training\<train_name>\checkpoints\best.pth
```
