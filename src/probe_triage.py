"""Per-game triage for the probe-first staged collector.

Reads `episodes.jsonl.gz` (and optionally `priors/`) produced by
`src.collect_probe_staged`, computes a multi-label classification for every
game, and writes machine- and human-readable summaries:

    triage_summary.json          one entry per game with metrics + labels
    triage_summary.csv           same, flat CSV for spreadsheets
    per_game_stage_summary.json  rates broken down per (game, stage)

Triage labels (multi-label — a game can carry several at once):

    promising           best_score > 0 OR progress_rate >= 0.2
    signal_but_stuck    plenty of LOCAL_TOGGLE / GLOBAL_CHANGE but no progress
    low_signal          mostly NO_CHANGE / SMALL_CHANGE
    dead_or_noisy       very high dead_action_rate, no progress
    click_promising     ACTION6 effective rate >= 0.4
    movement_promising  MOTION_LIKE share of effective signatures >= 0.35

The classifier uses observed counts only — it does not look at game IDs and
does not embed any per-game rule.

Run:

    python -m src.probe_triage \\
        --input ./Local_Output/Probe_Cache/probe_focus5_staged_v1/collected/episodes.jsonl.gz \\
        --priors ./Local_Output/Probe_Cache/probe_focus5_staged_v1/priors \\
        --output-dir ./Local_Output/Probe_Cache/probe_focus5_staged_v1/triage
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .color_features import (
    ALL_CHANGE_LABELS,
    color_profile_summary,
    merge_change_summary,
    record_dead_click_color,
)
from .common import iter_jsonl_gz, save_json
from .effect_signatures import (
    GAME_OVER,
    GLOBAL_CHANGE,
    LOCAL_TOGGLE,
    MOTION_LIKE,
    NO_CHANGE,
    PROGRESS,
    SMALL_CHANGE,
)


CLICK_ACTION_ID = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-game triage for the probe-first collector.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a collected episodes.jsonl.gz file.")
    parser.add_argument("--priors", type=str, default=None,
                        help="Optional priors directory written by collect_probe_staged.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write triage_summary.json/.csv and per-stage summaries.")
    parser.add_argument("--min-episodes", type=int, default=2,
                        help="Skip games with fewer than this many episodes.")
    return parser.parse_args()


def _empty_bucket() -> Dict[str, Any]:
    return {
        "episodes": 0,
        "best_score": 0.0,
        "best_levels": 0,
        "progress_episodes": 0,
        "total_actions": 0,
        "dead_actions": 0,
        "repeat_actions": 0,
        "action6_attempts": 0,
        "action6_effective": 0,
        "signature_counts": Counter(),
        "loop_metrics_totals": Counter(),
        "loop_metric_episodes": 0,
        "stages": defaultdict(_empty_stage_bucket),
        "color_accumulator": {},
        # Color-learning observability: how many transitions saw a NEW color
        # (first time the agent encountered it), a NEW (color, context) pair,
        # a coarse-vs-current disagreement, or an actual refined split.
        "color_learning": {
            "observations": 0,
            "new_color": 0,
            "new_context": 0,
            "disagreements": 0,
            "splits": 0,
        },
    }


def _empty_stage_bucket() -> Dict[str, Any]:
    return {
        "episodes": 0,
        "best_score": 0.0,
        "best_levels": 0,
        "progress_episodes": 0,
        "total_actions": 0,
        "dead_actions": 0,
        "repeat_actions": 0,
        "action6_attempts": 0,
        "action6_effective": 0,
        "signature_counts": Counter(),
        "loop_metrics_totals": Counter(),
        "loop_metric_episodes": 0,
    }


_LOOP_KEYS = (
    "repeated_state_action_count",
    "cycle_2_count",
    "cycle_3_count",
    "stagnation_escapes",
    "local_followup_attempts",
    "local_followup_successes",
    "reprobe_windows_opened",
    "reprobe_steps_used",
    "escape_steps",
    "escape_dead_steps",
    "reprobe_dead_steps",
)


def _accumulate_episode(
    game_bucket: Dict[str, Any],
    stage_bucket: Dict[str, Any],
    episode: Dict[str, Any],
) -> None:
    for bucket in (game_bucket, stage_bucket):
        bucket["episodes"] += 1
        score = float(episode.get("score", 0.0))
        levels = int(episode.get("levels_completed", 0))
        bucket["best_score"] = max(float(bucket["best_score"]), score)
        bucket["best_levels"] = max(int(bucket["best_levels"]), levels)
        if levels > 0:
            bucket["progress_episodes"] += 1

    transitions = list(episode.get("transitions", []))
    seen_state_action: set = set()
    for transition in transitions:
        sig = str(transition.get("effect_signature") or "")
        action_id = int(transition.get("action_id", 0))
        ad = transition.get("action_data") or {}
        for bucket in (game_bucket, stage_bucket):
            bucket["total_actions"] += 1
            bucket["signature_counts"][sig] += 1
            if sig == NO_CHANGE:
                bucket["dead_actions"] += 1
            if action_id == CLICK_ACTION_ID:
                bucket["action6_attempts"] += 1
                if sig not in (NO_CHANGE, GAME_OVER):
                    bucket["action6_effective"] += 1
        key = (
            str(transition.get("state_before", "")),
            int(transition.get("levels_before", 0)),
            action_id,
            int(ad.get("x", -1)),
            int(ad.get("y", -1)),
        )
        if key in seen_state_action:
            for bucket in (game_bucket, stage_bucket):
                bucket["repeat_actions"] += 1
        seen_state_action.add(key)

        # Color accumulator (per-game only; we don't split colors per-stage to
        # keep summaries readable).
        color_summary = transition.get("color_summary") or {}
        if color_summary:
            change_label = color_summary.get("change_label")
            colors_changed = color_summary.get("colors_changed") or []
            game_bucket["color_accumulator"] = merge_change_summary(
                game_bucket["color_accumulator"],
                {
                    "change_label": change_label,
                    "colors_changed": colors_changed,
                },
                signature=sig,
            )
            if action_id == CLICK_ACTION_ID and sig == NO_CHANGE:
                click_color = color_summary.get("click_color")
                if click_color is not None:
                    record_dead_click_color(game_bucket["color_accumulator"], int(click_color))
            obs = color_summary.get("observation") or {}
            if obs:
                game_bucket["color_learning"]["observations"] += 1
                if obs.get("is_new_color"):
                    game_bucket["color_learning"]["new_color"] += 1
                if obs.get("is_new_context"):
                    game_bucket["color_learning"]["new_context"] += 1
                if obs.get("disagreement"):
                    game_bucket["color_learning"]["disagreements"] += 1
                if obs.get("split"):
                    game_bucket["color_learning"]["splits"] += 1

    loop_metrics = episode.get("loop_metrics") or {}
    if loop_metrics:
        for bucket in (game_bucket, stage_bucket):
            bucket["loop_metric_episodes"] += 1
            for key in _LOOP_KEYS:
                bucket["loop_metrics_totals"][key] += int(loop_metrics.get(key, 0) or 0)


def _rates_from_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    total = max(1, int(bucket["total_actions"]))
    attempts6 = max(1, int(bucket["action6_attempts"]))
    sig_counts = bucket["signature_counts"]
    total_sigs = max(1, sum(sig_counts.values()))
    effective = sum(sig_counts.get(s, 0) for s in (PROGRESS, GLOBAL_CHANGE, LOCAL_TOGGLE, MOTION_LIKE))
    motion_share_of_effective = (
        sig_counts.get(MOTION_LIKE, 0) / float(max(1, effective))
    )
    loop_eps = max(1, int(bucket["loop_metric_episodes"]))
    loop_totals = bucket["loop_metrics_totals"]
    fa = int(loop_totals.get("local_followup_attempts", 0))
    fs = int(loop_totals.get("local_followup_successes", 0))
    color_learning = bucket.get("color_learning") or {}
    obs_count = max(1, int(color_learning.get("observations", 0)))
    return {
        "episodes": int(bucket["episodes"]),
        "best_score": round(float(bucket["best_score"]), 3),
        "best_levels": int(bucket["best_levels"]),
        "progress_rate": round(int(bucket["progress_episodes"]) / max(1, int(bucket["episodes"])), 3),
        "total_actions": int(bucket["total_actions"]),
        "dead_action_rate": round(int(bucket["dead_actions"]) / total, 3),
        "repeat_ratio": round(int(bucket["repeat_actions"]) / total, 3),
        "action6_attempts": int(bucket["action6_attempts"]),
        "action6_effectiveness": round(int(bucket["action6_effective"]) / attempts6, 3)
            if int(bucket["action6_attempts"]) > 0 else 0.0,
        "no_change_share": round(sig_counts.get(NO_CHANGE, 0) / float(total_sigs), 3),
        "small_change_share": round(sig_counts.get(SMALL_CHANGE, 0) / float(total_sigs), 3),
        "local_toggle_share": round(sig_counts.get(LOCAL_TOGGLE, 0) / float(total_sigs), 3),
        "global_change_share": round(sig_counts.get(GLOBAL_CHANGE, 0) / float(total_sigs), 3),
        "motion_share": round(sig_counts.get(MOTION_LIKE, 0) / float(total_sigs), 3),
        "motion_share_of_effective": round(motion_share_of_effective, 3),
        "progress_share": round(sig_counts.get(PROGRESS, 0) / float(total_sigs), 3),
        "local_followup_attempts": fa,
        "local_followup_successes": fs,
        "local_followup_success_rate": round(fs / max(1, fa), 3) if fa > 0 else 0.0,
        "escape_dead_rate": (
            round(int(loop_totals.get("escape_dead_steps", 0))
                  / max(1, int(loop_totals.get("escape_steps", 0))), 3)
            if int(loop_totals.get("escape_steps", 0)) > 0 else 0.0
        ),
        "reprobe_dead_rate": (
            round(int(loop_totals.get("reprobe_dead_steps", 0))
                  / max(1, int(loop_totals.get("reprobe_steps_used", 0))), 3)
            if int(loop_totals.get("reprobe_steps_used", 0)) > 0 else 0.0
        ),
        "color_observations": int(color_learning.get("observations", 0)),
        "color_new_color_rate": round(int(color_learning.get("new_color", 0)) / obs_count, 3),
        "color_new_context_rate": round(int(color_learning.get("new_context", 0)) / obs_count, 3),
        "color_disagreement_rate": round(int(color_learning.get("disagreements", 0)) / obs_count, 3),
        "color_split_rate": round(int(color_learning.get("splits", 0)) / obs_count, 3),
    }


def classify(rates: Dict[str, Any]) -> List[str]:
    """Multi-label triage. Counts only — no per-game rules."""
    labels: List[str] = []
    progress_rate = float(rates["progress_rate"])
    best_score = float(rates["best_score"])
    a6_eff = float(rates["action6_effectiveness"])
    motion_eff = float(rates["motion_share_of_effective"])
    dead_rate = float(rates["dead_action_rate"])
    repeat = float(rates["repeat_ratio"])
    toggle = float(rates["local_toggle_share"])
    glob = float(rates["global_change_share"])
    fu = float(rates["local_followup_success_rate"])

    if best_score > 0.0 or progress_rate >= 0.2:
        labels.append("promising")
    if (toggle + glob) >= 0.05 and progress_rate < 0.1 and best_score == 0.0:
        labels.append("signal_but_stuck")
    if dead_rate >= 0.55 and progress_rate == 0.0 and best_score == 0.0:
        labels.append("dead_or_noisy")
    if (toggle + glob) < 0.02 and progress_rate == 0.0 and "promising" not in labels:
        labels.append("low_signal")
    if a6_eff >= 0.4 and rates["action6_attempts"] >= 4:
        labels.append("click_promising")
    if motion_eff >= 0.35:
        labels.append("movement_promising")
    if repeat >= 0.85 and "promising" not in labels:
        labels.append("repeat_heavy")
    if fu >= 0.5 and rates["local_followup_attempts"] >= 4:
        labels.append("followup_strong")
    if not labels:
        labels.append("uncertain")
    return labels


def _read_priors(priors_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if priors_dir is None or not priors_dir.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(priors_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        game_id = str(payload.get("game_id") or path.stem)
        out[game_id] = payload
    return out


def evaluate(
    episodes_path: Path,
    priors_dir: Optional[Path],
    min_episodes: int,
) -> Dict[str, Any]:
    games: Dict[str, Dict[str, Any]] = defaultdict(_empty_bucket)
    for episode in iter_jsonl_gz(episodes_path):
        game_id = str(episode.get("game_id", "unknown"))
        bucket = games[game_id]
        stage = str(episode.get("stage") or "unknown")
        stage_bucket = bucket["stages"][stage]
        _accumulate_episode(bucket, stage_bucket, episode)

    priors = _read_priors(priors_dir)

    triage_entries: List[Dict[str, Any]] = []
    per_game_stage_summary: Dict[str, Dict[str, Any]] = {}
    for game_id, bucket in sorted(games.items()):
        if int(bucket["episodes"]) < int(min_episodes):
            continue
        rates = _rates_from_bucket(bucket)
        labels = classify(rates)
        color_profile = color_profile_summary(bucket["color_accumulator"])
        prior = priors.get(game_id)
        prior_summary = None
        if prior:
            prior_summary = {
                "best_score": float(prior.get("best_score", 0.0)),
                "best_levels": int(prior.get("best_levels", 0)),
                "best_seed": prior.get("best_seed"),
                "best_stage": prior.get("best_stage"),
                "stages_completed": prior.get("stages_completed") or {},
                "dead_actions": prior.get("dead_actions") or [],
                "n_dead_coords": len(prior.get("dead_coords") or []),
                "n_promising_clicks": len(prior.get("promising_clicks") or []),
            }

        per_stage = {}
        for stage_id, stage_bucket in sorted(bucket["stages"].items()):
            per_stage[stage_id] = _rates_from_bucket(stage_bucket)

        triage_entries.append({
            "game_id": game_id,
            "labels": labels,
            "metrics": rates,
            "color_profile": color_profile,
            "prior_summary": prior_summary,
            "stages": per_stage,
        })
        per_game_stage_summary[game_id] = per_stage

    triage_entries.sort(key=lambda e: (
        # Sort by "promising"-ness: best_score then progress_rate.
        -float(e["metrics"]["best_score"]),
        -float(e["metrics"]["progress_rate"]),
        e["game_id"],
    ))

    label_counter: Counter = Counter()
    for entry in triage_entries:
        for label in entry["labels"]:
            label_counter[label] += 1
    overall = {
        "total_games": len(triage_entries),
        "label_counts": dict(label_counter),
    }

    return {
        "input": str(episodes_path),
        "overall": overall,
        "games": triage_entries,
        "per_game_stage_summary": per_game_stage_summary,
    }


def write_csv(triage: Dict[str, Any], path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for entry in triage["games"]:
        metrics = entry["metrics"]
        labels = ",".join(entry["labels"])
        rows.append({
            "game_id": entry["game_id"],
            "labels": labels,
            "episodes": metrics["episodes"],
            "best_score": metrics["best_score"],
            "best_levels": metrics["best_levels"],
            "progress_rate": metrics["progress_rate"],
            "dead_action_rate": metrics["dead_action_rate"],
            "repeat_ratio": metrics["repeat_ratio"],
            "action6_effectiveness": metrics["action6_effectiveness"],
            "motion_share_of_effective": metrics["motion_share_of_effective"],
            "local_toggle_share": metrics["local_toggle_share"],
            "global_change_share": metrics["global_change_share"],
            "local_followup_success_rate": metrics["local_followup_success_rate"],
            "color_new_color_rate": metrics.get("color_new_color_rate", 0.0),
            "color_new_context_rate": metrics.get("color_new_context_rate", 0.0),
            "color_disagreement_rate": metrics.get("color_disagreement_rate", 0.0),
            "color_split_rate": metrics.get("color_split_rate", 0.0),
        })
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    episodes_path = Path(args.input).resolve()
    priors_dir = Path(args.priors).resolve() if args.priors else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    triage = evaluate(episodes_path, priors_dir, int(args.min_episodes))
    save_json(output_dir / "triage_summary.json", triage)
    save_json(output_dir / "per_game_stage_summary.json", triage["per_game_stage_summary"])
    write_csv(triage, output_dir / "triage_summary.csv")

    print("Wrote triage outputs to %s" % output_dir)
    print(json.dumps(triage["overall"], indent=2))


if __name__ == "__main__":
    main()
