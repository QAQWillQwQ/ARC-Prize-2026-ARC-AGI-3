"""Staged large-scale collector for the probe-first branch.

Five stages are run per game:

    probe_seed       broad probing with no anti-loop, builds initial priors
    exploit_safe     v3.1 defaults — measured exploit
    followup_focus   biases search around recently promising clicks
    rescue_reprobe   only on stuck games, raises reprobe budget
    harvest_best     replays best-seeded configurations with relaxed penalties

Per-game prioritization is recomputed before every stage from a persistent
per-game prior file (`{output-root}/priors/{game_id}.json`). Priors are
distilled from each completed episode and merged with the running aggregate
so subsequent stages can warm-start from known-dead actions/coords and known
promising clicks.

Output is the same `episodes.jsonl.gz` format as `src.collect_probe`, with two
extra fields per episode:

    stage                    — which stage produced this episode
    episode_index_in_game    — 0-indexed counter within the game
    episode_index_global     — 0-indexed counter across the whole run

The teammate collector is NOT touched. This module only depends on
`src.common`, `src.effect_signatures`, `src.probe_memory`, `src.probe_agent`.

Example:

    python -m src.collect_probe_staged \\
        --project-root "." \\
        --output-root "./Local_Output/Probe_Cache/probe_focus5_staged_v1" \\
        --games "sp80,lp85,ar25,ls20,r11l" \\
        --episodes-per-game 32

The default per-stage budget (4 / 12 / 8 / 4 / 4) sums to 32; pass
`--budget-<stage>` to override individual stage counts.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from arc_agi import Arcade, OperationMode

from .color_features import (
    color_profile_summary,
    merge_change_summary,
    record_dead_click_color,
)
from .common import (
    append_jsonl_gz,
    episode_level_actions,
    load_metadata_map,
    rhae_score,
    save_json,
    seed_everything,
    utc_timestamp,
)
from .effect_signatures import (
    GLOBAL_CHANGE,
    LOCAL_TOGGLE,
    MOTION_LIKE,
    NO_CHANGE,
    PROGRESS,
    SMALL_CHANGE,
)
from .probe_agent import ProbeAgentConfig, ProbeFirstAgent

DEFAULT_FOCUS5 = "sp80,lp85,ar25,ls20,r11l"

STAGE_ORDER = ("probe_seed", "exploit_safe", "followup_focus", "rescue_reprobe", "harvest_best")

DEFAULT_STAGE_BUDGETS: Dict[str, int] = {
    "probe_seed": 4,
    "exploit_safe": 12,
    "followup_focus": 8,
    "rescue_reprobe": 4,
    "harvest_best": 4,
}

# Stage-specific overrides applied on top of the user's CLI base config.
# Anything not present here inherits from the base.
STAGE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "probe_seed": {
        # Pure probing: no escape, no reprobe, no warm-start.
        "probe_budget": 24,
        "stagnation_progressless_ratio": 1.1,
        "global_change_reprobe_budget": 0,
        "stale_signature_penalty": 0.0,
        "state_action_penalty": 0.0,
        "_warm_start": False,
    },
    "exploit_safe": {
        "probe_budget": 12,
        "_warm_start": True,
    },
    "followup_focus": {
        "probe_budget": 8,
        "local_followup_bonus": 1.6,
        "local_followup_window": 10,
        "local_followup_radius": 8,
        "_warm_start": True,
    },
    "rescue_reprobe": {
        "probe_budget": 10,
        "global_change_reprobe_budget": 6,
        "reprobe_episode_cap": 14,
        "reprobe_cooldown_steps": 4,
        "reprobe_dead_action_bonus": 0.4,
        "reprobe_low_trial_bonus": 0.2,
        "reprobe_skip_if_recent_progress": False,
        "_warm_start": True,
    },
    "harvest_best": {
        "probe_budget": 6,
        "stale_signature_penalty": 0.15,
        "state_action_penalty": 0.25,
        "stagnation_progressless_ratio": 0.95,
        "_warm_start": True,
    },
}


# ----------------------------------------------------------------------- priors


def _empty_prior(game_id: str) -> Dict[str, Any]:
    return {
        "game_id": game_id,
        "episode_count": 0,
        "best_score": 0.0,
        "best_levels": 0,
        "progress_episodes": 0,
        "progress_rate": 0.0,
        "local_followup_attempts": 0,
        "local_followup_successes": 0,
        "local_followup_success_rate": 0.0,
        "dead_actions": [],
        "dead_coords": [],
        "promising_clicks": [],
        "stages_completed": {},
        "best_seed": None,
        # Aggregated stats used for per-stage observability + triage-aware budgets.
        "stage_stats": {},
        "rolling_metrics": {
            "total_actions": 0,
            "dead_actions": 0,
            "repeat_actions": 0,
            "action6_attempts": 0,
            "action6_effective": 0,
            "signature_counts": {},
        },
        # Per-(action_id, color) -> sigs counts. Carried across stages so the
        # agent's warm-start picks up learned color affordances.
        "color_action": [],
        # Refined (action_id, color, context_key) entries — context-conditioned
        # hypotheses produced by observe_color when outcomes split.
        "color_context": [],
        # Color accumulator -> color_profile_summary at save time.
        "color_accumulator": {},
        "color_profile": {},
        "triage_labels": [],
    }


def _load_prior(priors_dir: Path, game_id: str) -> Dict[str, Any]:
    path = priors_dir / ("%s.json" % game_id)
    if not path.exists():
        return _empty_prior(game_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Backfill new fields when reading older prior files.
    template = _empty_prior(game_id)
    for key, value in template.items():
        payload.setdefault(key, value)
    return payload


def _save_prior(priors_dir: Path, game_id: str, prior: Dict[str, Any]) -> None:
    priors_dir.mkdir(parents=True, exist_ok=True)
    (priors_dir / ("%s.json" % game_id)).write_text(
        json.dumps(prior, indent=2, sort_keys=True), encoding="utf-8"
    )


def _distill_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    actions_summary = (episode.get("memory_summary") or {}).get("actions") or {}
    dead_actions: List[int] = []
    for action_id_str, action in actions_summary.items():
        try:
            if action.get("role") == "dead":
                dead_actions.append(int(action_id_str))
        except (TypeError, AttributeError):
            continue

    promising: List[Dict[str, Any]] = []
    dead_coords: set = set()
    for transition in episode.get("transitions", []) or []:
        if int(transition.get("action_id", 0)) != 6:
            continue
        ad = transition.get("action_data") or {}
        x, y = int(ad.get("x", -1)), int(ad.get("y", -1))
        if x < 0 or y < 0:
            continue
        sig = str(transition.get("effect_signature", ""))
        if sig in (PROGRESS, GLOBAL_CHANGE, LOCAL_TOGGLE):
            promising.append({"x": x, "y": y, "signature": sig})
        elif sig == NO_CHANGE:
            dead_coords.add((x, y))
    color_action_summary = (episode.get("memory_summary") or {}).get("color_action") or []
    color_context_summary = (episode.get("memory_summary") or {}).get("color_context") or []
    return {
        "dead_actions": sorted(dead_actions),
        "dead_coords": sorted(list(dead_coords)),
        "promising_clicks": promising[-32:],
        "color_action": list(color_action_summary),
        "color_context": list(color_context_summary),
    }


def _classify_prior(prior: Dict[str, Any]) -> List[str]:
    """Lightweight triage labels derived from the prior alone.

    Identical philosophy to src.probe_triage but uses the running rolling
    metrics so we can label after every episode rather than at the end.
    """
    rolling = prior.get("rolling_metrics") or {}
    total = max(1, int(rolling.get("total_actions", 0) or 0))
    sig_counts = rolling.get("signature_counts") or {}
    total_sigs = max(1, sum(int(v) for v in sig_counts.values()))
    a6_eff = (
        int(rolling.get("action6_effective", 0))
        / max(1, int(rolling.get("action6_attempts", 1)))
    ) if int(rolling.get("action6_attempts", 0)) > 0 else 0.0
    dead_rate = int(rolling.get("dead_actions", 0)) / total
    repeat = int(rolling.get("repeat_actions", 0)) / total
    progress_rate = float(prior.get("progress_rate", 0.0))
    best_score = float(prior.get("best_score", 0.0))
    fu_rate = float(prior.get("local_followup_success_rate", 0.0))

    toggle_share = int(sig_counts.get(LOCAL_TOGGLE, 0)) / float(total_sigs)
    glob_share = int(sig_counts.get(GLOBAL_CHANGE, 0)) / float(total_sigs)
    motion_share = int(sig_counts.get(MOTION_LIKE, 0)) / float(total_sigs)
    effective = sum(int(sig_counts.get(s, 0)) for s in (PROGRESS, GLOBAL_CHANGE, LOCAL_TOGGLE, MOTION_LIKE))
    motion_eff = int(sig_counts.get(MOTION_LIKE, 0)) / float(max(1, effective))

    labels: List[str] = []
    if best_score > 0.0 or progress_rate >= 0.2:
        labels.append("promising")
    if (toggle_share + glob_share) >= 0.05 and progress_rate < 0.1 and best_score == 0.0:
        labels.append("signal_but_stuck")
    if dead_rate >= 0.55 and progress_rate == 0.0 and best_score == 0.0:
        labels.append("dead_or_noisy")
    if (toggle_share + glob_share) < 0.02 and progress_rate == 0.0 and "promising" not in labels:
        labels.append("low_signal")
    if a6_eff >= 0.4 and int(rolling.get("action6_attempts", 0)) >= 4:
        labels.append("click_promising")
    if motion_eff >= 0.35:
        labels.append("movement_promising")
    if repeat >= 0.85 and "promising" not in labels:
        labels.append("repeat_heavy")
    if fu_rate >= 0.5 and int(prior.get("local_followup_attempts", 0)) >= 4:
        labels.append("followup_strong")
    if not labels:
        labels.append("uncertain")
    return labels


def _ensure_stage_stats(prior: Dict[str, Any], stage: str) -> Dict[str, Any]:
    stats = prior.setdefault("stage_stats", {})
    bucket = stats.get(stage)
    if bucket is None:
        bucket = {
            "episodes": 0,
            "best_score": 0.0,
            "best_levels": 0,
            "progress_episodes": 0,
            "total_actions": 0,
            "dead_actions": 0,
            "repeat_actions": 0,
            "action6_attempts": 0,
            "action6_effective": 0,
            "signature_counts": {},
            "loop_local_followup_attempts": 0,
            "loop_local_followup_successes": 0,
            "loop_escape_steps": 0,
            "loop_escape_dead_steps": 0,
            "loop_reprobe_steps_used": 0,
            "loop_reprobe_dead_steps": 0,
        }
        stats[stage] = bucket
    return bucket


def _merge_color_action_into_prior(prior: Dict[str, Any], distilled: Dict[str, Any]) -> None:
    """Merge per-(action, color) signature counts from the distilled episode."""
    existing: Dict[tuple, Dict[str, Any]] = {}
    for entry in prior.get("color_action") or []:
        try:
            key = (int(entry["action_id"]), int(entry["color"]))
        except (TypeError, KeyError, ValueError):
            continue
        existing[key] = entry
    for entry in distilled.get("color_action") or []:
        try:
            action_id = int(entry["action_id"])
            color = int(entry["color"])
            counts = entry.get("counts") or {}
            trials = int(entry.get("trials", 0))
        except (TypeError, KeyError, ValueError):
            continue
        cur = existing.get((action_id, color))
        if cur is None:
            cur = {
                "action_id": action_id,
                "color": color,
                "counts": {sig: int(c) for sig, c in counts.items()},
                "trials": trials,
            }
            existing[(action_id, color)] = cur
        else:
            for sig, count in counts.items():
                try:
                    count_int = int(count)
                except (TypeError, ValueError):
                    continue
                cur["counts"][sig] = int(cur["counts"].get(sig, 0)) + count_int
            cur["trials"] = int(cur["trials"]) + trials
    prior["color_action"] = list(existing.values())


def _merge_color_context_into_prior(prior: Dict[str, Any], distilled: Dict[str, Any]) -> None:
    """Merge refined (action, color, context_key) signature counts."""
    existing: Dict[tuple, Dict[str, Any]] = {}
    for entry in prior.get("color_context") or []:
        try:
            ctx = entry.get("context_key") or []
            if len(ctx) != 2:
                continue
            key = (int(entry["action_id"]), int(entry["color"]),
                   (int(ctx[0]), int(ctx[1])))
        except (TypeError, KeyError, ValueError):
            continue
        existing[key] = entry
    for entry in distilled.get("color_context") or []:
        try:
            action_id = int(entry["action_id"])
            color = int(entry["color"])
            ctx = entry.get("context_key") or []
            if len(ctx) != 2:
                continue
            ctx_tuple = (int(ctx[0]), int(ctx[1]))
            counts = entry.get("counts") or {}
            trials = int(entry.get("trials", 0))
            splits = int(entry.get("splits", 0))
        except (TypeError, KeyError, ValueError):
            continue
        full_key = (action_id, color, ctx_tuple)
        cur = existing.get(full_key)
        if cur is None:
            cur = {
                "action_id": action_id,
                "color": color,
                "context_key": [ctx_tuple[0], ctx_tuple[1]],
                "counts": {sig: int(c) for sig, c in counts.items()},
                "trials": trials,
                "splits": splits,
            }
            existing[full_key] = cur
        else:
            for sig, count in counts.items():
                try:
                    count_int = int(count)
                except (TypeError, ValueError):
                    continue
                cur["counts"][sig] = int(cur["counts"].get(sig, 0)) + count_int
            cur["trials"] = int(cur["trials"]) + trials
            cur["splits"] = int(cur["splits"]) + splits
    prior["color_context"] = list(existing.values())


def _accumulate_rolling(prior: Dict[str, Any], stage_bucket: Dict[str, Any], episode: Dict[str, Any]) -> None:
    rolling = prior.setdefault("rolling_metrics", {
        "total_actions": 0, "dead_actions": 0, "repeat_actions": 0,
        "action6_attempts": 0, "action6_effective": 0, "signature_counts": {},
    })
    sig_counts = rolling.setdefault("signature_counts", {})
    stage_sig_counts = stage_bucket.setdefault("signature_counts", {})

    transitions = list(episode.get("transitions", []))
    seen_state_action: set = set()
    for transition in transitions:
        sig = str(transition.get("effect_signature") or "")
        action_id = int(transition.get("action_id", 0))
        rolling["total_actions"] = int(rolling["total_actions"]) + 1
        stage_bucket["total_actions"] = int(stage_bucket["total_actions"]) + 1
        sig_counts[sig] = int(sig_counts.get(sig, 0)) + 1
        stage_sig_counts[sig] = int(stage_sig_counts.get(sig, 0)) + 1
        if sig == NO_CHANGE:
            rolling["dead_actions"] = int(rolling["dead_actions"]) + 1
            stage_bucket["dead_actions"] = int(stage_bucket["dead_actions"]) + 1
        if action_id == 6:
            rolling["action6_attempts"] = int(rolling["action6_attempts"]) + 1
            stage_bucket["action6_attempts"] = int(stage_bucket["action6_attempts"]) + 1
            if sig not in (NO_CHANGE, "game_over"):
                rolling["action6_effective"] = int(rolling["action6_effective"]) + 1
                stage_bucket["action6_effective"] = int(stage_bucket["action6_effective"]) + 1
        ad = transition.get("action_data") or {}
        key = (
            str(transition.get("state_before", "")),
            int(transition.get("levels_before", 0)),
            action_id,
            int(ad.get("x", -1)),
            int(ad.get("y", -1)),
        )
        if key in seen_state_action:
            rolling["repeat_actions"] = int(rolling["repeat_actions"]) + 1
            stage_bucket["repeat_actions"] = int(stage_bucket["repeat_actions"]) + 1
        seen_state_action.add(key)


def _accumulate_color(prior: Dict[str, Any], episode: Dict[str, Any]) -> None:
    accumulator = prior.setdefault("color_accumulator", {})
    for transition in episode.get("transitions", []) or []:
        color_summary = transition.get("color_summary") or {}
        if not color_summary:
            continue
        sig = str(transition.get("effect_signature") or "")
        accumulator = merge_change_summary(
            accumulator,
            {
                "change_label": color_summary.get("change_label"),
                "colors_changed": color_summary.get("colors_changed") or [],
            },
            signature=sig,
        )
        if int(transition.get("action_id", 0)) == 6 and sig == NO_CHANGE:
            click_color = color_summary.get("click_color")
            if click_color is not None:
                record_dead_click_color(accumulator, int(click_color))
    prior["color_accumulator"] = accumulator
    prior["color_profile"] = color_profile_summary(accumulator)


def _merge_prior(prior: Dict[str, Any], episode: Dict[str, Any], stage: str, seed: int) -> Dict[str, Any]:
    distilled = _distill_episode(episode)
    stage_bucket = _ensure_stage_stats(prior, stage)

    prior["episode_count"] = int(prior.get("episode_count", 0)) + 1
    stage_bucket["episodes"] = int(stage_bucket["episodes"]) + 1
    score = float(episode.get("score", 0.0))
    levels = int(episode.get("levels_completed", 0))
    if score > float(prior.get("best_score", 0.0)):
        prior["best_score"] = score
        prior["best_seed"] = int(seed)
        prior["best_stage"] = stage
    if score > float(stage_bucket.get("best_score", 0.0)):
        stage_bucket["best_score"] = score
    prior["best_levels"] = max(int(prior.get("best_levels", 0)), levels)
    stage_bucket["best_levels"] = max(int(stage_bucket.get("best_levels", 0)), levels)
    if levels > 0:
        stage_bucket["progress_episodes"] = int(stage_bucket["progress_episodes"]) + 1

    dead_set = set(int(a) for a in prior.get("dead_actions", []) or [])
    dead_set.update(distilled["dead_actions"])
    prior["dead_actions"] = sorted(dead_set)

    dead_coords_set = set(tuple(c) for c in prior.get("dead_coords", []) or [])
    dead_coords_set.update(tuple(c) for c in distilled["dead_coords"])
    if len(dead_coords_set) > 256:
        dead_coords_set = set(list(dead_coords_set)[-256:])
    prior["dead_coords"] = [list(c) for c in sorted(dead_coords_set)]

    promising = list(prior.get("promising_clicks", []) or [])
    promising.extend(distilled["promising_clicks"])
    prior["promising_clicks"] = promising[-64:]

    _merge_color_action_into_prior(prior, distilled)
    _merge_color_context_into_prior(prior, distilled)

    loop = episode.get("loop_metrics") or {}
    fa = int(prior.get("local_followup_attempts", 0)) + int(loop.get("local_followup_attempts", 0) or 0)
    fs = int(prior.get("local_followup_successes", 0)) + int(loop.get("local_followup_successes", 0) or 0)
    prior["local_followup_attempts"] = fa
    prior["local_followup_successes"] = fs
    prior["local_followup_success_rate"] = round(fs / fa, 3) if fa > 0 else 0.0

    stage_bucket["loop_local_followup_attempts"] += int(loop.get("local_followup_attempts", 0) or 0)
    stage_bucket["loop_local_followup_successes"] += int(loop.get("local_followup_successes", 0) or 0)
    stage_bucket["loop_escape_steps"] += int(loop.get("escape_steps", 0) or 0)
    stage_bucket["loop_escape_dead_steps"] += int(loop.get("escape_dead_steps", 0) or 0)
    stage_bucket["loop_reprobe_steps_used"] += int(loop.get("reprobe_steps_used", 0) or 0)
    stage_bucket["loop_reprobe_dead_steps"] += int(loop.get("reprobe_dead_steps", 0) or 0)

    progress_eps = int(prior.get("progress_episodes", 0)) + (1 if levels > 0 else 0)
    prior["progress_episodes"] = progress_eps
    prior["progress_rate"] = round(progress_eps / max(1, prior["episode_count"]), 3)

    stages_completed = dict(prior.get("stages_completed") or {})
    stages_completed[stage] = int(stages_completed.get(stage, 0)) + 1
    prior["stages_completed"] = stages_completed

    _accumulate_rolling(prior, stage_bucket, episode)
    _accumulate_color(prior, episode)
    prior["triage_labels"] = _classify_prior(prior)

    return prior


# ----------------------------------------------------------- scheduling


def _stage_priority(prior: Dict[str, Any], stage: str) -> float:
    fu = float(prior.get("local_followup_success_rate", 0.0))
    progress = float(prior.get("progress_rate", 0.0))
    best_score = float(prior.get("best_score", 0.0))
    eps = int(prior.get("episode_count", 0))
    labels = set(prior.get("triage_labels") or [])
    if stage == "probe_seed":
        return 1.0 / (1.0 + eps)
    if stage == "exploit_safe":
        bonus = 0.15 if "click_promising" in labels or "movement_promising" in labels else 0.0
        return 0.5 * progress + 0.3 * (1.0 if best_score > 0 else 0.0) + 0.2 * fu + bonus
    if stage == "followup_focus":
        bonus = 0.2 if "click_promising" in labels or "followup_strong" in labels else 0.0
        return 0.6 * fu + 0.3 * progress + 0.1 + bonus
    if stage == "rescue_reprobe":
        bonus = 0.2 if "signal_but_stuck" in labels else 0.0
        return 0.7 * (1.0 - progress) + 0.2 * (1.0 - fu) + 0.1 + bonus
    if stage == "harvest_best":
        return 0.7 * min(best_score / 50.0, 1.0) + 0.3 * progress
    return 0.5


def _should_skip_stage(stage: str, prior: Dict[str, Any]) -> bool:
    labels = set(prior.get("triage_labels") or [])
    if stage == "rescue_reprobe":
        if "dead_or_noisy" in labels:
            return True
        return float(prior.get("progress_rate", 0.0)) >= 0.75
    if stage == "harvest_best":
        return float(prior.get("best_score", 0.0)) <= 0.0
    if stage == "followup_focus":
        # Skip follow-up on games with no interaction signal at all.
        return "low_signal" in labels and float(prior.get("best_score", 0.0)) == 0.0
    return False


def _adaptive_episode_budget(stage: str, base_count: int, prior: Dict[str, Any]) -> int:
    """Per-game budget adjustment for a stage.

    Returns an integer episode count for this game in this stage. Conservative
    by default — multiplies the base count by a factor in [0.5, 2.0] driven by
    triage labels. Used only when --budget-allocator=adaptive.
    """
    labels = set(prior.get("triage_labels") or [])
    factor = 1.0
    if stage == "probe_seed":
        # Probe budgets are roughly equal across games; only knock down clearly
        # dead games slightly so budget can flow to others later.
        if "dead_or_noisy" in labels:
            factor = 0.75
    elif stage == "exploit_safe":
        if "promising" in labels:
            factor = 1.5
        elif "dead_or_noisy" in labels:
            factor = 0.5
        elif "low_signal" in labels:
            factor = 0.75
    elif stage == "followup_focus":
        if "click_promising" in labels or "followup_strong" in labels:
            factor = 1.75
        elif "low_signal" in labels or "dead_or_noisy" in labels:
            factor = 0.5
    elif stage == "rescue_reprobe":
        if "signal_but_stuck" in labels:
            factor = 1.75
        elif "promising" in labels:
            factor = 0.5
        elif "dead_or_noisy" in labels:
            factor = 0.0
    elif stage == "harvest_best":
        if "promising" in labels and float(prior.get("best_score", 0.0)) >= 5.0:
            factor = 1.75
    return max(0, int(round(base_count * factor)))


def _build_stage_config(base: ProbeAgentConfig, stage: str) -> ProbeAgentConfig:
    overrides = {k: v for k, v in STAGE_OVERRIDES.get(stage, {}).items() if not k.startswith("_")}
    return replace(base, **overrides)


def _stage_warm_start(stage: str) -> bool:
    return bool(STAGE_OVERRIDES.get(stage, {}).get("_warm_start", False))


# --------------------------------------------------------------- entry point


@dataclass
class _Counters:
    started: int = 0
    finished: int = 0
    failed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staged probe-first ARC-AGI-3 collector.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--games", type=str, default=None)
    parser.add_argument("--all-public-games", action="store_true",
                        help="Include every short game ID found under environment_files/.")
    parser.add_argument("--include-labels", type=str, default=None,
                        help="Comma-separated triage labels. Only games whose existing "
                             "prior carries any of these labels are kept.")
    parser.add_argument("--exclude-labels", type=str, default=None,
                        help="Comma-separated triage labels. Games whose existing prior "
                             "carries any of these labels are skipped.")
    parser.add_argument("--restrict-from-triage", type=str, default=None,
                        help="Path to a triage_summary.json. When set, the include/exclude "
                             "label filters use that file's labels instead of the local priors.")
    parser.add_argument("--budget-allocator", type=str, default="adaptive",
                        choices=("fixed", "adaptive"),
                        help="fixed = use --budget-<stage> for every game. "
                             "adaptive = scale per-game per-stage by triage labels.")
    parser.add_argument("--episodes-per-game", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--stall-steps", type=int, default=32)
    parser.add_argument("--reset-limit", type=int, default=2)
    parser.add_argument("--click-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    # Loop / repeat tuning knobs. Default None preserves ProbeAgentConfig defaults
    # (state_action_penalty=0.35, stale_signature_penalty=0.3). These flow into
    # the base config so exploit_safe / followup_focus / rescue_reprobe stages
    # inherit them; probe_seed and harvest_best keep their own STAGE_OVERRIDES.
    parser.add_argument("--state-action-penalty", type=float, default=None,
                        dest="state_action_penalty")
    parser.add_argument("--state-action-penalty-cap", type=float, default=None,
                        dest="state_action_penalty_cap")
    parser.add_argument("--stale-signature-penalty", type=float, default=None,
                        dest="stale_signature_penalty")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the schedule and exit; do not run any episodes.")
    # Per-stage budget overrides (counts of episodes per game per stage).
    for stage, count in DEFAULT_STAGE_BUDGETS.items():
        parser.add_argument("--budget-%s" % stage.replace("_", "-"),
                            type=int, default=count, dest="budget_" + stage)
    return parser.parse_args()


def _resolve_games(
    args: argparse.Namespace,
    metadata_map: Dict[str, Dict[str, Any]],
    priors_dir: Path,
) -> List[str]:
    if args.all_public_games:
        candidates = sorted(metadata_map.keys())
    else:
        games_arg = args.games or DEFAULT_FOCUS5
        candidates = sorted({g.strip() for g in games_arg.split(",") if g.strip()})
    if not candidates:
        raise SystemExit("Need at least one game (set --games or --all-public-games).")

    include = _parse_labels(args.include_labels)
    exclude = _parse_labels(args.exclude_labels)
    if not include and not exclude:
        return [g for g in candidates if g in metadata_map]

    label_source: Dict[str, List[str]] = {}
    if args.restrict_from_triage:
        triage_path = Path(args.restrict_from_triage).expanduser().resolve()
        if not triage_path.exists():
            raise SystemExit("triage file not found: %s" % triage_path)
        triage = json.loads(triage_path.read_text(encoding="utf-8"))
        for entry in triage.get("games") or []:
            label_source[str(entry["game_id"])] = list(entry.get("labels") or [])
    else:
        for game_id in candidates:
            prior = _load_prior(priors_dir, game_id)
            label_source[game_id] = list(prior.get("triage_labels") or [])

    out: List[str] = []
    for game_id in candidates:
        if game_id not in metadata_map:
            continue
        labels = set(label_source.get(game_id, []) or [])
        if include and not (labels & include):
            continue
        if exclude and (labels & exclude):
            continue
        out.append(game_id)
    if not out:
        raise SystemExit(
            "Label filter excluded every game. Check --include-labels / --exclude-labels."
        )
    return out


def _parse_labels(raw: Optional[str]) -> set:
    if not raw:
        return set()
    return {token.strip() for token in raw.split(",") if token.strip()}


def _resolve_stage_budgets(args: argparse.Namespace) -> Dict[str, int]:
    budgets = {stage: int(getattr(args, "budget_" + stage)) for stage in STAGE_ORDER}
    requested_total = sum(budgets.values())
    target = int(args.episodes_per_game)
    if requested_total != target:
        # Scale each stage by target / requested_total; preserve probe_seed at minimum 1.
        scale = target / float(max(1, requested_total))
        scaled = {stage: max(0, int(round(count * scale))) for stage, count in budgets.items()}
        diff = target - sum(scaled.values())
        # Adjust on the largest non-zero stage.
        if diff != 0:
            largest = max(scaled, key=scaled.get)
            scaled[largest] = max(0, scaled[largest] + diff)
        budgets = scaled
    return budgets


def _per_game_stage_rates(prior: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for stage, bucket in (prior.get("stage_stats") or {}).items():
        total = max(1, int(bucket.get("total_actions", 0)))
        attempts6 = max(1, int(bucket.get("action6_attempts", 0)))
        sig_counts = bucket.get("signature_counts") or {}
        total_sigs = max(1, sum(int(v) for v in sig_counts.values()))
        a6_eff = (
            int(bucket.get("action6_effective", 0)) / float(attempts6)
            if int(bucket.get("action6_attempts", 0)) > 0 else 0.0
        )
        fa = int(bucket.get("loop_local_followup_attempts", 0))
        fs = int(bucket.get("loop_local_followup_successes", 0))
        out[stage] = {
            "episodes": int(bucket.get("episodes", 0)),
            "best_score": round(float(bucket.get("best_score", 0.0)), 3),
            "best_levels": int(bucket.get("best_levels", 0)),
            "progress_rate": round(
                int(bucket.get("progress_episodes", 0)) / max(1, int(bucket.get("episodes", 1))), 3),
            "dead_action_rate": round(int(bucket.get("dead_actions", 0)) / float(total), 3),
            "repeat_ratio": round(int(bucket.get("repeat_actions", 0)) / float(total), 3),
            "action6_effectiveness": round(a6_eff, 3),
            "no_change_share": round(int(sig_counts.get(NO_CHANGE, 0)) / float(total_sigs), 3),
            "local_toggle_share": round(int(sig_counts.get(LOCAL_TOGGLE, 0)) / float(total_sigs), 3),
            "global_change_share": round(int(sig_counts.get(GLOBAL_CHANGE, 0)) / float(total_sigs), 3),
            "motion_share": round(int(sig_counts.get(MOTION_LIKE, 0)) / float(total_sigs), 3),
            "local_followup_success_rate": round(fs / max(1, fa), 3) if fa > 0 else 0.0,
            "escape_dead_rate": (
                round(int(bucket.get("loop_escape_dead_steps", 0))
                      / max(1, int(bucket.get("loop_escape_steps", 0))), 3)
                if int(bucket.get("loop_escape_steps", 0)) > 0 else 0.0
            ),
            "reprobe_dead_rate": (
                round(int(bucket.get("loop_reprobe_dead_steps", 0))
                      / max(1, int(bucket.get("loop_reprobe_steps_used", 0))), 3)
                if int(bucket.get("loop_reprobe_steps_used", 0)) > 0 else 0.0
            ),
        }
    return out


def _write_per_game_stage_summary(output_root: Path, games: Sequence[str], priors_dir: Path) -> None:
    summary: Dict[str, Any] = {}
    for game_id in games:
        prior = _load_prior(priors_dir, game_id)
        summary[game_id] = {
            "triage_labels": list(prior.get("triage_labels") or []),
            "best_score": float(prior.get("best_score", 0.0)),
            "best_levels": int(prior.get("best_levels", 0)),
            "best_seed": prior.get("best_seed"),
            "best_stage": prior.get("best_stage"),
            "stages_completed": prior.get("stages_completed") or {},
            "stages": _per_game_stage_rates(prior),
            "color_profile": prior.get("color_profile") or {},
        }
    save_json(output_root / "per_game_stage_summary.json", summary)


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / "collected"
    episodes_path = run_dir / "episodes.jsonl.gz"
    priors_dir = output_root / "priors"
    metadata_map = load_metadata_map(project_root / "environment_files")

    games = _resolve_games(args, metadata_map, priors_dir)

    stage_budgets = _resolve_stage_budgets(args)
    config_payload = {
        "agent": "ProbeFirstAgent",
        "agent_version": "probe_v3_1_staged",
        "stage_budgets": stage_budgets,
        "episodes_per_game": int(args.episodes_per_game),
        "max_steps": int(args.max_steps),
        "stall_steps": int(args.stall_steps),
        "reset_limit": int(args.reset_limit),
        "click_candidates": int(args.click_candidates),
        "seed_base": int(args.seed_base),
        "games": games,
        "all_public_games": bool(args.all_public_games),
        "include_labels": sorted(_parse_labels(args.include_labels)),
        "exclude_labels": sorted(_parse_labels(args.exclude_labels)),
        "budget_allocator": str(args.budget_allocator),
        "stage_overrides": STAGE_OVERRIDES,
        "created_at_utc": utc_timestamp(),
    }

    base_config_kwargs: Dict[str, Any] = {
        "max_steps": int(args.max_steps),
        "stall_steps": int(args.stall_steps),
        "reset_limit": int(args.reset_limit),
        "click_candidates_per_step": int(args.click_candidates),
        "seed": 0,
    }
    if args.state_action_penalty is not None:
        base_config_kwargs["state_action_penalty"] = float(args.state_action_penalty)
    if args.state_action_penalty_cap is not None:
        base_config_kwargs["state_action_penalty_cap"] = float(args.state_action_penalty_cap)
    if args.stale_signature_penalty is not None:
        base_config_kwargs["stale_signature_penalty"] = float(args.stale_signature_penalty)
    base_config = ProbeAgentConfig(**base_config_kwargs)
    config_payload["base_config_overrides"] = {
        k: v for k, v in base_config_kwargs.items()
        if k in ("state_action_penalty", "state_action_penalty_cap", "stale_signature_penalty")
    }
    save_json(output_root / "collect_probe_staged_config.json", config_payload)

    if args.dry_run:
        use_adaptive = str(args.budget_allocator) == "adaptive"
        print("Allocator: %s" % args.budget_allocator)
        print("Stage budgets per game (base): %s" % stage_budgets)
        print("Total episodes planned (base, fixed): %d" % (
            len(games) * sum(stage_budgets.values())
        ))
        for stage in STAGE_ORDER:
            print("  %-15s budget=%d" % (stage, stage_budgets[stage]))
        adaptive_total = 0
        for game_id in games:
            prior = _load_prior(priors_dir, game_id)
            labels = ",".join(prior.get("triage_labels") or []) or "-"
            for stage in STAGE_ORDER:
                priority = _stage_priority(prior, stage)
                skip = _should_skip_stage(stage, prior)
                base_count = int(stage_budgets[stage])
                effective = (
                    0 if skip else (
                        _adaptive_episode_budget(stage, base_count, prior)
                        if use_adaptive else base_count
                    )
                )
                adaptive_total += effective
                print("    %-5s %-15s priority=%.3f skip=%s base=%d -> %d  labels=%s" % (
                    game_id, stage, priority, skip, base_count, effective, labels
                ))
        if use_adaptive:
            print("Total episodes planned (adaptive): %d" % adaptive_total)
        return

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(output_root / "recordings"),
    )

    counters = _Counters()
    aggregate: Dict[str, Dict[str, Any]] = {
        game_id: {"episodes": 0, "best_score": 0.0, "max_levels": 0, "stages": {}}
        for game_id in games
    }

    start = time.perf_counter()
    global_index = 0
    in_game_index: Dict[str, int] = {game_id: 0 for game_id in games}

    use_adaptive = str(args.budget_allocator) == "adaptive"

    for stage in STAGE_ORDER:
        per_stage_count = int(stage_budgets[stage])
        if per_stage_count <= 0:
            continue

        priors_snapshot = {g: _load_prior(priors_dir, g) for g in games}
        ordered_games = sorted(
            games,
            key=lambda g: _stage_priority(priors_snapshot[g], stage),
            reverse=True,
        )

        stage_config_template = _build_stage_config(base_config, stage)
        warm_start_enabled = _stage_warm_start(stage)

        for game_id in ordered_games:
            if game_id not in metadata_map:
                print("[staged] skipping unknown game %s" % game_id, flush=True)
                continue
            prior = priors_snapshot[game_id]
            if _should_skip_stage(stage, prior):
                labels = ",".join(prior.get("triage_labels") or []) or "-"
                print("[staged] %-5s stage=%-15s skipped (gate, labels=%s)" % (
                    game_id, stage, labels), flush=True)
                continue
            game_episode_count = (
                _adaptive_episode_budget(stage, per_stage_count, prior)
                if use_adaptive else per_stage_count
            )
            if game_episode_count <= 0:
                print("[staged] %-5s stage=%-15s skipped (adaptive budget=0)" % (
                    game_id, stage), flush=True)
                continue
            baseline_actions = list(metadata_map[game_id].get("baseline_actions", []) or [])

            for ep_idx in range(game_episode_count):
                seed = int(args.seed_base) + global_index
                episode_id = "%s_%s_%02d_seed%d" % (game_id, stage, ep_idx, seed)
                elapsed = time.perf_counter() - start
                print("[staged] %s game=%s stage=%s ep=%d seed=%d (global=%d, elapsed=%.1fs)" % (
                    episode_id, game_id, stage, ep_idx, seed, global_index, elapsed
                ), flush=True)

                stage_config = replace(stage_config_template, seed=seed)
                agent = ProbeFirstAgent(stage_config)
                env = arc.make(game_id, seed=seed)
                if env is None:
                    print("[staged] failed to create env for %s" % game_id, flush=True)
                    counters.failed += 1
                    continue

                effective_prior = prior if warm_start_enabled else None
                episode = agent.play_env(
                    env=env,
                    game_id=game_id,
                    baseline_actions=baseline_actions,
                    prior=effective_prior,
                )

                rhae = rhae_score(
                    baseline_actions=baseline_actions,
                    completed_level_actions=episode_level_actions(episode.get("transitions", [])),
                )
                episode["score"] = float(rhae["score"])
                episode["level_scores"] = list(rhae["level_scores"])
                episode["max_score"] = float(rhae["max_score"])
                episode["episode_id"] = episode_id
                episode["seed"] = int(seed)
                episode["agent"] = "ProbeFirstAgent"
                episode["stage"] = stage
                episode["episode_index_in_game"] = int(in_game_index[game_id])
                episode["episode_index_global"] = int(global_index)
                append_jsonl_gz(episodes_path, episode)

                # Refresh + persist prior so subsequent episodes (and stages) see it.
                prior = _merge_prior(prior, episode, stage=stage, seed=seed)
                _save_prior(priors_dir, game_id, prior)
                priors_snapshot[game_id] = prior

                bucket = aggregate[game_id]
                bucket["episodes"] += 1
                bucket["best_score"] = max(bucket["best_score"], float(episode["score"]))
                bucket["max_levels"] = max(bucket["max_levels"], int(episode["levels_completed"]))
                bucket["stages"][stage] = int(bucket["stages"].get(stage, 0)) + 1

                in_game_index[game_id] += 1
                global_index += 1
                counters.finished += 1

                loop_metrics = episode.get("loop_metrics", {}) or {}
                labels = ",".join(prior.get("triage_labels") or []) or "-"
                print("[staged]   -> state=%s levels=%d actions=%d score=%.2f esc=%d/%d rp=%d/%d fu=%d/%d labels=%s" % (
                    episode["final_state"],
                    int(episode["levels_completed"]),
                    int(episode["actions_taken"]),
                    float(episode["score"]),
                    int(loop_metrics.get("escape_dead_steps", 0)),
                    int(loop_metrics.get("escape_steps", 0)),
                    int(loop_metrics.get("reprobe_dead_steps", 0)),
                    int(loop_metrics.get("reprobe_steps_used", 0)),
                    int(loop_metrics.get("local_followup_successes", 0)),
                    int(loop_metrics.get("local_followup_attempts", 0)),
                    labels,
                ), flush=True)

        # End of stage: persist a per-game stage summary so external triage /
        # dashboards can poll between stages without re-reading episodes.jsonl.gz.
        _write_per_game_stage_summary(output_root, games, priors_dir)

    elapsed_total = time.perf_counter() - start
    label_counter: Counter = Counter()
    for game_id in games:
        prior = _load_prior(priors_dir, game_id)
        for label in prior.get("triage_labels") or []:
            label_counter[label] += 1
    summary = {
        "agent": "ProbeFirstAgent",
        "agent_version": "probe_v3_1_staged",
        "episodes_path": str(episodes_path),
        "priors_dir": str(priors_dir),
        "completed_episodes": counters.finished,
        "failed_episodes": counters.failed,
        "total_episodes_planned": len(games) * sum(stage_budgets.values()),
        "elapsed_seconds": elapsed_total,
        "per_game": aggregate,
        "stage_budgets": stage_budgets,
        "budget_allocator": str(args.budget_allocator),
        "triage_label_counts": dict(label_counter),
    }
    save_json(output_root / "collect_probe_staged_summary.json", summary)
    _write_per_game_stage_summary(output_root, games, priors_dir)
    print("[staged] done %d episodes in %.1fs -> %s" % (
        counters.finished, elapsed_total, episodes_path
    ))
    print("[staged] triage_label_counts=%s" % dict(label_counter))


if __name__ == "__main__":
    main()
