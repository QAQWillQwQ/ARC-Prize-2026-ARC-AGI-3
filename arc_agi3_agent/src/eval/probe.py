"""Evaluation / comparison metrics for the probe-first collector.

Reads an episodes.jsonl.gz produced either by `src.collect_probe` or by the
teammate's `src.collect` / `src.collect_staged`, and writes a per-game summary
JSON. The same metrics are derived for both formats — episodes that don't
carry an `effect_signature` field (i.e. the teammate's collector) get a
heuristic fallback signature so the comparison is apples-to-apples.

Recommended usage:

    python -m src.eval_probe \\
        --input ./Local_Output/Probe_Cache/probe_focus5_v1/collected/episodes.jsonl.gz \\
        --output ./Local_Output/Probe_Cache/probe_focus5_v1/probe_eval.json \\
        --label probe

    python -m src.eval_probe \\
        --input ./Local_Output/Collection_Cache/openlab_search_v1/collected/episodes.jsonl.gz \\
        --output ./Local_Output/Collection_Cache/openlab_search_v1/baseline_eval.json \\
        --label baseline
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..utils.common import iter_jsonl_gz, save_json
from ..utils.effect_signatures import (
    GAME_OVER,
    GLOBAL_CHANGE,
    LARGE_DELTA_LIMIT,
    LOCAL_TOGGLE,
    NO_CHANGE,
    PROGRESS,
    SMALL_CHANGE,
    SMALL_DELTA_LIMIT,
    WIN,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate collected episodes (probe or baseline).")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to an episodes.jsonl.gz file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Where to write the summary JSON.")
    parser.add_argument("--label", type=str, default="probe",
                        help="Label tag for this run (e.g. 'probe' or 'baseline').")
    parser.add_argument("--print-overall", action="store_true",
                        help="Print the overall block to stdout.")
    return parser.parse_args()


def _fallback_signature(transition: Dict[str, Any]) -> str:
    """Used when transitions lack `effect_signature` (teammate format)."""
    delta = int(transition.get("delta_pixels", 0))
    progress = int(transition.get("levels_after", 0)) - int(transition.get("levels_before", 0))
    state_after = transition.get("state_after", "")
    if state_after == "WIN":
        return WIN
    if state_after == "GAME_OVER":
        return GAME_OVER
    if progress > 0:
        return PROGRESS
    if delta == 0:
        return NO_CHANGE
    if delta >= LARGE_DELTA_LIMIT:
        return GLOBAL_CHANGE
    if delta <= SMALL_DELTA_LIMIT:
        return LOCAL_TOGGLE
    return SMALL_CHANGE


def _signature_for(transition: Dict[str, Any]) -> str:
    sig = transition.get("effect_signature")
    if isinstance(sig, str) and sig:
        return sig
    return _fallback_signature(transition)


def _steps_to_first_progress(transitions: Sequence[Dict[str, Any]]) -> int:
    for transition in transitions:
        if int(transition.get("levels_after", 0)) > int(transition.get("levels_before", 0)):
            return int(transition.get("step_index", -1)) + 1
    return -1


def _is_action6_effective(transition: Dict[str, Any]) -> bool:
    if int(transition.get("action_id", 0)) != 6:
        return False
    return _signature_for(transition) not in (NO_CHANGE, GAME_OVER)


def _top_counts(counter: Counter, n: int = 5) -> List[List[Any]]:
    items = sorted(counter.items(), key=lambda item: item[1], reverse=True)[:n]
    return [[name, int(count)] for name, count in items if count > 0]


_LOOP_METRIC_KEYS = (
    "repeated_state_action_count",
    "cycle_2_count",
    "cycle_3_count",
    "stagnation_escapes",
    "local_followup_attempts",
    "local_followup_successes",
    "reprobe_windows_opened",
    "reprobe_steps_used",
    # ---- v3.1 diagnostics ----
    "escape_steps",
    "escape_dead_steps",
    "reprobe_dead_steps",
    "escape_blocked_by_cooldown",
    "reprobe_blocked_by_cap",
    "reprobe_blocked_by_cooldown",
    "reprobe_blocked_by_recent_progress",
)


def evaluate(path: Path, label: str) -> Dict[str, Any]:
    per_game: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "episodes": 0,
        "max_levels": 0,
        "best_rhae": 0.0,
        "rhae_values": [],
        "first_progress_steps": [],
        "total_actions": 0,
        "dead_actions": 0,
        "repeat_actions": 0,
        "action6_attempts": 0,
        "action6_effective": 0,
        "signature_counts": Counter(),
        "probe_signature_counts": Counter(),
        "phase_action_counts": Counter(),
        "probe_episodes_with_findings": 0,
        "probe_unique_signatures_per_episode": [],
        "loop_metrics_totals": {key: 0 for key in _LOOP_METRIC_KEYS},
        "loop_metric_episodes": 0,
    })

    for episode in iter_jsonl_gz(path):
        game_id = str(episode.get("game_id", "unknown"))
        bucket = per_game[game_id]
        bucket["episodes"] += 1
        bucket["max_levels"] = max(bucket["max_levels"], int(episode.get("levels_completed", 0)))
        score = float(episode.get("score", 0.0))
        bucket["rhae_values"].append(score)
        bucket["best_rhae"] = max(bucket["best_rhae"], score)

        transitions = list(episode.get("transitions", []))
        bucket["total_actions"] += len(transitions)
        sfp = _steps_to_first_progress(transitions)
        if sfp > 0:
            bucket["first_progress_steps"].append(sfp)

        seen_state_action: set = set()
        for transition in transitions:
            sig = _signature_for(transition)
            bucket["signature_counts"][sig] += 1
            phase = transition.get("phase", "exploit")
            bucket["phase_action_counts"][phase] += 1
            if phase == "probe":
                bucket["probe_signature_counts"][sig] += 1
            if sig == NO_CHANGE:
                bucket["dead_actions"] += 1

            ad = transition.get("action_data") or {}
            key = (
                str(transition.get("state_before", "")),
                int(transition.get("levels_before", 0)),
                int(transition.get("action_id", 0)),
                int(ad.get("x", -1)),
                int(ad.get("y", -1)),
            )
            if key in seen_state_action:
                bucket["repeat_actions"] += 1
            seen_state_action.add(key)

            if int(transition.get("action_id", 0)) == 6:
                bucket["action6_attempts"] += 1
                if _is_action6_effective(transition):
                    bucket["action6_effective"] += 1

        probe_signature_counts = episode.get("probe_signature_counts") or {}
        unique_probe = sum(1 for v in probe_signature_counts.values() if int(v) > 0)
        bucket["probe_unique_signatures_per_episode"].append(unique_probe)
        if unique_probe > 0:
            bucket["probe_episodes_with_findings"] += 1

        loop_metrics = episode.get("loop_metrics") or {}
        if loop_metrics:
            bucket["loop_metric_episodes"] += 1
            for key in _LOOP_METRIC_KEYS:
                bucket["loop_metrics_totals"][key] += int(loop_metrics.get(key, 0) or 0)

    summaries: List[Dict[str, Any]] = []
    for game_id, bucket in per_game.items():
        total = max(1, bucket["total_actions"])
        attempts6 = max(1, bucket["action6_attempts"])
        sfp_values = bucket["first_progress_steps"]
        rhae_values = bucket["rhae_values"]
        probe_uniques = bucket["probe_unique_signatures_per_episode"]
        loop_eps = max(1, bucket["loop_metric_episodes"])
        loop_totals = bucket["loop_metrics_totals"]
        loop_summary = {
            "episodes_with_loop_metrics": int(bucket["loop_metric_episodes"]),
            "totals": {key: int(loop_totals[key]) for key in _LOOP_METRIC_KEYS},
            "per_episode_means": {
                key: round(loop_totals[key] / loop_eps, 3) for key in _LOOP_METRIC_KEYS
            },
            "local_followup_success_rate": (
                round(loop_totals["local_followup_successes"] / loop_totals["local_followup_attempts"], 3)
                if loop_totals["local_followup_attempts"] > 0 else 0.0
            ),
            "escape_dead_action_rate": (
                round(loop_totals["escape_dead_steps"] / loop_totals["escape_steps"], 3)
                if loop_totals["escape_steps"] > 0 else 0.0
            ),
            "reprobe_dead_action_rate": (
                round(loop_totals["reprobe_dead_steps"] / loop_totals["reprobe_steps_used"], 3)
                if loop_totals["reprobe_steps_used"] > 0 else 0.0
            ),
            "loop_share_of_actions": (
                round(
                    (loop_totals["cycle_2_count"] + loop_totals["cycle_3_count"]
                     + loop_totals["repeated_state_action_count"])
                    / max(1, bucket["total_actions"]),
                    3,
                )
            ),
        }
        summaries.append({
            "label": label,
            "game_id": game_id,
            "episodes": bucket["episodes"],
            "max_levels_completed": bucket["max_levels"],
            "best_rhae": round(bucket["best_rhae"], 3),
            "mean_rhae": round(sum(rhae_values) / max(1, len(rhae_values)), 3) if rhae_values else 0.0,
            "mean_steps_to_first_progress": (
                round(sum(sfp_values) / max(1, len(sfp_values)), 2) if sfp_values else None
            ),
            "episodes_with_progress": len(sfp_values),
            "total_actions": bucket["total_actions"],
            "dead_action_rate": round(bucket["dead_actions"] / total, 3),
            "repeat_ratio": round(bucket["repeat_actions"] / total, 3),
            "action6_attempts": bucket["action6_attempts"],
            "action6_effectiveness": (
                round(bucket["action6_effective"] / attempts6, 3) if bucket["action6_attempts"] > 0 else 0.0
            ),
            "top_signatures": _top_counts(bucket["signature_counts"], n=5),
            "probe_signatures": _top_counts(bucket["probe_signature_counts"], n=5),
            "probe_episodes_with_findings": bucket["probe_episodes_with_findings"],
            "avg_unique_probe_signatures": (
                round(sum(probe_uniques) / max(1, len(probe_uniques)), 2) if probe_uniques else 0.0
            ),
            "phase_action_counts": dict(bucket["phase_action_counts"]),
            "loop_metrics": loop_summary,
        })

    summaries.sort(key=lambda item: item["game_id"])
    overall = _aggregate_overall(summaries, label)
    return {"label": label, "input": str(path), "per_game": summaries, "overall": overall}


def _aggregate_overall(summaries: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    if not summaries:
        return {"label": label, "games": 0}
    keys_avg = ["best_rhae", "mean_rhae", "dead_action_rate", "repeat_ratio", "action6_effectiveness"]
    overall: Dict[str, Any] = {"label": label, "games": len(summaries)}
    for key in keys_avg:
        values = [float(s[key]) for s in summaries if s.get(key) is not None]
        overall["mean_" + key] = round(sum(values) / max(1, len(values)), 3) if values else 0.0
    overall["total_episodes"] = sum(int(s["episodes"]) for s in summaries)
    overall["total_actions"] = sum(int(s["total_actions"]) for s in summaries)
    overall["max_levels_any_game"] = max((int(s["max_levels_completed"]) for s in summaries), default=0)
    overall["best_rhae_any_game"] = round(max((float(s["best_rhae"]) for s in summaries), default=0.0), 3)

    loop_totals = {key: 0 for key in _LOOP_METRIC_KEYS}
    loop_eps = 0
    for summary in summaries:
        loop_block = summary.get("loop_metrics") or {}
        loop_eps += int(loop_block.get("episodes_with_loop_metrics", 0) or 0)
        for key, value in (loop_block.get("totals") or {}).items():
            if key in loop_totals:
                loop_totals[key] += int(value or 0)
    overall["loop_metric_episodes"] = int(loop_eps)
    overall["loop_metric_totals"] = dict(loop_totals)
    overall["mean_local_followup_success_rate"] = (
        round(loop_totals["local_followup_successes"] / loop_totals["local_followup_attempts"], 3)
        if loop_totals["local_followup_attempts"] > 0 else 0.0
    )
    overall["mean_escape_dead_action_rate"] = (
        round(loop_totals["escape_dead_steps"] / loop_totals["escape_steps"], 3)
        if loop_totals["escape_steps"] > 0 else 0.0
    )
    overall["mean_reprobe_dead_action_rate"] = (
        round(loop_totals["reprobe_dead_steps"] / loop_totals["reprobe_steps_used"], 3)
        if loop_totals["reprobe_steps_used"] > 0 else 0.0
    )
    overall["loop_share_of_actions"] = (
        round(
            (loop_totals["cycle_2_count"] + loop_totals["cycle_3_count"]
             + loop_totals["repeated_state_action_count"])
            / max(1, overall["total_actions"]),
            3,
        )
    )
    return overall


def main() -> None:
    args = parse_args()
    summary = evaluate(Path(args.input).resolve(), label=args.label)
    save_json(Path(args.output).resolve(), summary)
    print("Wrote evaluation summary to %s" % args.output)
    if args.print_overall:
        print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
