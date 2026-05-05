# Experiment Notes (Internal, Preliminary)

This file is the internal experiment log for the probe-first pipeline. All
numbers below are **preliminary internal experiments** — they are not
competition results and should not be quoted as headline performance.
Aggregates may change as the collector and the trainer evolve.

The point of this file is to keep the *reasoning* behind freezes, knob
settings, and label thresholds visible to the next contributor. Specific
numbers will rot; the engineering rationale should not.

---

## Frozen baseline (probe_first_healthy_v1)

Frozen as the working baseline for the probe-first staged collector + triage
+ harvest + train pipeline.

**Frozen configs** (defaults in `agents/probe_agent.py` / `collect/probe_staged.py`):

| Knob                              | Value |
|---                                |---:|
| `state_action_penalty`            | 0.35 |
| `state_action_penalty_cap`        | 1.4  |
| `stale_signature_penalty`         | 0.30 |
| `stale_signature_window`          | 6    |
| `local_followup_radius`           | 6    |
| `local_followup_window`           | 6    |
| `local_followup_bonus`            | 1.20 |
| `stagnation_window`               | 16   |
| `stagnation_progressless_ratio`   | 0.90 |
| `global_change_reprobe_budget`    | 3    |
| `color_action_priority_weight`    | 0.30 |
| `color_click_priority_weight`     | 0.40 |

**Focus-5 staged smoke** (90 episodes, 16 ep/game):

| metric                              | value |
|---                                  |---:|
| mean_best_rhae                      | 0.56 |
| mean_dead_action_rate               | 0.03 |
| mean_repeat_ratio                   | 0.85 |
| mean_action6_effectiveness          | 0.57 |
| loop_share_of_actions               | 0.10 |
| max_levels_any_game                 | 1    |

These are the numbers any regression should stay close to on the focus-5
set. The pipeline is producing useful interaction signal (action6 effective
roughly 57% of the time, dead action rate ~3%) and reaching level 1 on the
positive-core games. Repeat ratio is the open risk — see below.

---

## Open risk: repeat_ratio is high (~0.85)

The collector revisits state-action-coord keys often. Useful signal still
gets through (level 1 reached, click effectiveness high), but trajectories
have a lot of repetition.

### Tuning experiment that did *not* work

Lifted `state_action_penalty` 0.35 → 0.55 and `stale_signature_penalty`
0.30 → 0.50 (and separately the penalty cap 1.4 → 2.5). Result on focus-5,
identical seed and allocator:

| metric                              | baseline | tuned_v1 | tuned_v2 |
|---                                  |---:|---:|---:|
| mean_best_rhae                      | 0.556 | 0.556 | 0.556 |
| mean_repeat_ratio                   | 0.854 | 0.856 | 0.856 |
| loop_share_of_actions               | 0.102 | 0.098 | 0.098 |
| reprobe_dead_action_rate            | 0.077 | 0.069 | 0.069 |

**Conclusion: the additive penalty knobs are not the binding constraint** on
repeat behavior. The penalties *do* fire (loop share and reprobe dead-action
rate move in the right direction), but the agent then re-emits the same
coords because the candidate set is small. The repeat metric is upstream of
the priority comparison — the lever lives in candidate generation.

### What to try next (separate experiments)

- `click_candidates_per_step` (currently 8): widen the candidate pool so the
  agent has alternatives once a coord is penalized.
- `local_followup_radius` / `local_followup_window`: localised followup may
  be pinning the agent to recently-clicked neighbourhoods.
- A coord-level visit-count cool-down inside the click ranker rather than the
  additive priority penalty.

The CLI knobs `--state-action-penalty`, `--state-action-penalty-cap`,
`--stale-signature-penalty` are exposed on `collect.probe_staged` with
`None` defaults so future experiments are inspectable from the CLI without
editing source.

---

## Triage thresholds

Set conservatively so the harvest pass spends budget on games with *some*
observable signal, not just on games with the highest raw score:

- `promising`: `best_score > 0` OR `progress_rate >= 0.20`
- `signal_but_stuck`: combined `LOCAL_TOGGLE` + `GLOBAL_CHANGE` share `>= 0.15`
  AND `progress_rate < 0.05`
- `low_signal`: `NO_CHANGE` + `SMALL_CHANGE` share `>= 0.80`
- `dead_or_noisy`: dead-action rate `>= 0.40` AND no progress
- `click_promising`: `ACTION6` effective rate `>= 0.40`
- `movement_promising`: `MOTION_LIKE` share of effective signatures `>= 0.35`

A game can carry several labels at once (e.g. `signal_but_stuck` +
`click_promising`). The harvest pass reads `--include-labels` and pulls the
union, weighted toward `harvest_best`, `rescue_reprobe`, and `followup_focus`
stages.

---

## Color learning logic

The flow inside `agents/probe_agent.py::observe_color` is:

- **NEW context** → record the observed signature lift.
- **OLD context, same signature** → reinforce.
- **OLD context, different signature** → split: keep both contexts and let
  the priority comparison pick.
- After ≥3 trials, refined entries replace coarse-context lift estimates so
  later episodes see sharper priors.

Color affordances are persisted to
`priors/<game_id>.json["color_accumulator"]`. Keys must be `str(int(color))`;
otherwise cross-stage save/load fails on `sort_keys`.

This is observation-only learning — there are no per-game "color X means Y"
rules anywhere in the source. Color affordances must be learnable from
observed outcomes.

---

## Engineering principles

These have been internalized in code review, recording them so the next
contributor inherits them:

1. **Additive, observable changes.** New behaviors should emit counters into
   `loop_metrics.totals` so a regression shows up in the eval JSON instead
   of in the GIF.
2. **Preserve what works.** When a knob does not move the metric, leave the
   default in place rather than changing it speculatively.
3. **No architectural rewrites unless the metric demands it.** A flat repeat
   ratio is not a license to rewrite the click ranker.
4. **Color is observation, not instruction.** Color features feed the policy
   and the priors. They never become hard-coded rules.
