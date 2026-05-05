"""Entry point for the probe-first experimental collector.

Writes episodes in the same gzip-JSONL format as src.collect (so downstream
tooling can read both), but each transition is annotated with `phase`,
`effect_signature`, and a richer `effect` payload, and each episode carries
a `memory_summary`, `signature_counts`, and `action_roles` block.

Run on the focus-5 set:

    python -m src.collect_probe \\
        --project-root "." \\
        --output-root "./Local_Output/Probe_Cache/probe_focus5_v1" \\
        --games "sp80,lp85,ar25,ls20,r11l" \\
        --seeds "0,1,2,3" \\
        --probe-budget 16

Then evaluate with src.eval_probe.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from arc_agi import Arcade, OperationMode

from .common import (
    append_jsonl_gz,
    episode_level_actions,
    load_metadata_map,
    rhae_score,
    save_json,
    seed_everything,
    utc_timestamp,
)
from .probe_agent import ProbeAgentConfig, ProbeFirstAgent

DEFAULT_FOCUS5 = "sp80,lp85,ar25,ls20,r11l"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe-first ARC-AGI-3 trajectory collector.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--games", type=str, default=None,
                        help="Comma-separated short game IDs. Defaults to focus-5.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3")
    parser.add_argument("--probe-budget", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--stall-steps", type=int, default=32)
    parser.add_argument("--reset-limit", type=int, default=2)
    parser.add_argument("--click-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    # v3 anti-loop / exploit knobs
    parser.add_argument("--state-action-penalty", type=float, default=0.35)
    parser.add_argument("--cycle-2-penalty", type=float, default=0.7)
    parser.add_argument("--cycle-3-penalty", type=float, default=0.55)
    parser.add_argument("--stale-signature-penalty", type=float, default=0.3)
    parser.add_argument("--stale-signature-window", type=int, default=6)
    # v3.1 stagnation / reprobe softening knobs (defaults match probe_v3_1).
    parser.add_argument("--stagnation-window", type=int, default=16)
    parser.add_argument("--stagnation-progressless-ratio", type=float, default=0.90)
    parser.add_argument("--stagnation-escape-cooldown", type=int, default=8)
    parser.add_argument("--escape-priority-blend", type=float, default=0.5)
    parser.add_argument("--escape-skip-dead", type=int, default=1,
                        help="1 to keep escape from picking known-dead actions/coords, 0 to allow.")
    parser.add_argument("--global-change-reprobe-budget", type=int, default=3)
    parser.add_argument("--reprobe-episode-cap", type=int, default=8)
    parser.add_argument("--reprobe-cooldown-steps", type=int, default=6)
    parser.add_argument("--reprobe-skip-if-recent-progress", type=int, default=1)
    parser.add_argument("--recent-progress-window", type=int, default=8)
    parser.add_argument("--reprobe-dead-action-bonus", type=float, default=0.25)
    parser.add_argument("--reprobe-low-trial-bonus", type=float, default=0.1)
    parser.add_argument("--local-followup-radius", type=int, default=6)
    parser.add_argument("--local-followup-window", type=int, default=6)
    parser.add_argument("--local-followup-bonus", type=float, default=1.2)
    return parser.parse_args()


def build_arcade(project_root: Path, recordings_root: Path) -> Arcade:
    return Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(recordings_root),
    )


def _attach_score(episode: Dict[str, Any]) -> Dict[str, Any]:
    info = rhae_score(
        baseline_actions=episode.get("baseline_actions", []) or [],
        completed_level_actions=episode_level_actions(episode.get("transitions", [])),
    )
    episode["score"] = float(info["score"])
    episode["level_scores"] = list(info["level_scores"])
    episode["max_score"] = float(info["max_score"])
    return episode


def _format_findings(probe_signature_counts: Dict[str, int]) -> Dict[str, int]:
    return {sig: int(count) for sig, count in probe_signature_counts.items() if count > 0}


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / "collected"
    episodes_path = run_dir / "episodes.jsonl.gz"
    metadata_map = load_metadata_map(project_root / "environment_files")

    games_arg = args.games or DEFAULT_FOCUS5
    games = sorted({g.strip() for g in games_arg.split(",") if g.strip()})
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not games or not seeds:
        raise SystemExit("Need at least one game and one seed.")

    config_payload = {
        "agent": "ProbeFirstAgent",
        "agent_version": "probe_v3_1",
        "probe_budget": int(args.probe_budget),
        "max_steps": int(args.max_steps),
        "stall_steps": int(args.stall_steps),
        "reset_limit": int(args.reset_limit),
        "click_candidates": int(args.click_candidates),
        "state_action_penalty": float(args.state_action_penalty),
        "cycle_2_penalty": float(args.cycle_2_penalty),
        "cycle_3_penalty": float(args.cycle_3_penalty),
        "stale_signature_penalty": float(args.stale_signature_penalty),
        "stale_signature_window": int(args.stale_signature_window),
        "stagnation_window": int(args.stagnation_window),
        "stagnation_progressless_ratio": float(args.stagnation_progressless_ratio),
        "stagnation_escape_cooldown": int(args.stagnation_escape_cooldown),
        "escape_priority_blend": float(args.escape_priority_blend),
        "escape_skip_dead": bool(int(args.escape_skip_dead)),
        "global_change_reprobe_budget": int(args.global_change_reprobe_budget),
        "reprobe_episode_cap": int(args.reprobe_episode_cap),
        "reprobe_cooldown_steps": int(args.reprobe_cooldown_steps),
        "reprobe_skip_if_recent_progress": bool(int(args.reprobe_skip_if_recent_progress)),
        "recent_progress_window": int(args.recent_progress_window),
        "reprobe_dead_action_bonus": float(args.reprobe_dead_action_bonus),
        "reprobe_low_trial_bonus": float(args.reprobe_low_trial_bonus),
        "local_followup_radius": int(args.local_followup_radius),
        "local_followup_window": int(args.local_followup_window),
        "local_followup_bonus": float(args.local_followup_bonus),
        "seeds": seeds,
        "games": games,
        "created_at_utc": utc_timestamp(),
    }
    save_json(output_root / "collect_probe_config.json", config_payload)

    arc = build_arcade(project_root=project_root, recordings_root=output_root / "recordings")

    total = len(games) * len(seeds)
    completed = 0
    start = time.perf_counter()
    aggregate: Dict[str, Dict[str, Any]] = {game_id: {"episodes": 0, "best_score": 0.0, "max_levels": 0}
                                              for game_id in games}

    for game_id in games:
        if game_id not in metadata_map:
            print("[probe] skipping unknown game %s" % game_id, flush=True)
            continue
        baseline_actions = list(metadata_map[game_id].get("baseline_actions", []) or [])
        for seed in seeds:
            elapsed = time.perf_counter() - start
            print("[probe] %s seed=%d  (%d/%d, elapsed=%.1fs)" % (
                game_id, seed, completed + 1, total, elapsed
            ), flush=True)

            agent = ProbeFirstAgent(ProbeAgentConfig(
                probe_budget=int(args.probe_budget),
                max_steps=int(args.max_steps),
                stall_steps=int(args.stall_steps),
                reset_limit=int(args.reset_limit),
                click_candidates_per_step=int(args.click_candidates),
                seed=int(seed),
                state_action_penalty=float(args.state_action_penalty),
                cycle_2_penalty=float(args.cycle_2_penalty),
                cycle_3_penalty=float(args.cycle_3_penalty),
                stale_signature_penalty=float(args.stale_signature_penalty),
                stale_signature_window=int(args.stale_signature_window),
                stagnation_window=int(args.stagnation_window),
                stagnation_progressless_ratio=float(args.stagnation_progressless_ratio),
                stagnation_escape_cooldown=int(args.stagnation_escape_cooldown),
                escape_priority_blend=float(args.escape_priority_blend),
                escape_skip_dead=bool(int(args.escape_skip_dead)),
                global_change_reprobe_budget=int(args.global_change_reprobe_budget),
                reprobe_episode_cap=int(args.reprobe_episode_cap),
                reprobe_cooldown_steps=int(args.reprobe_cooldown_steps),
                reprobe_skip_if_recent_progress=bool(int(args.reprobe_skip_if_recent_progress)),
                recent_progress_window=int(args.recent_progress_window),
                reprobe_dead_action_bonus=float(args.reprobe_dead_action_bonus),
                reprobe_low_trial_bonus=float(args.reprobe_low_trial_bonus),
                local_followup_radius=int(args.local_followup_radius),
                local_followup_window=int(args.local_followup_window),
                local_followup_bonus=float(args.local_followup_bonus),
            ))
            env = arc.make(game_id, seed=int(seed))
            if env is None:
                print("[probe] failed to create env for %s" % game_id, flush=True)
                continue

            episode = agent.play_env(env=env, game_id=game_id, baseline_actions=baseline_actions)
            episode = _attach_score(episode)
            episode["episode_id"] = "%s_seed%d_probe" % (game_id, int(seed))
            episode["seed"] = int(seed)
            episode["agent"] = "ProbeFirstAgent"
            append_jsonl_gz(episodes_path, episode)

            bucket = aggregate.setdefault(game_id, {"episodes": 0, "best_score": 0.0, "max_levels": 0})
            bucket["episodes"] += 1
            bucket["best_score"] = max(bucket["best_score"], float(episode["score"]))
            bucket["max_levels"] = max(bucket["max_levels"], int(episode["levels_completed"]))

            completed += 1
            loop_metrics = episode.get("loop_metrics", {}) or {}
            print("[probe] saved %s state=%s levels=%d actions=%d score=%.2f probe=%s loops=%s" % (
                episode["episode_id"], episode["final_state"],
                int(episode["levels_completed"]), int(episode["actions_taken"]),
                float(episode["score"]),
                _format_findings(episode["probe_signature_counts"]),
                {
                    "rep_sa": int(loop_metrics.get("repeated_state_action_count", 0)),
                    "c2": int(loop_metrics.get("cycle_2_count", 0)),
                    "c3": int(loop_metrics.get("cycle_3_count", 0)),
                    "esc": "%d(d=%d/%d,blk=%d)" % (
                        int(loop_metrics.get("stagnation_escapes", 0)),
                        int(loop_metrics.get("escape_dead_steps", 0)),
                        int(loop_metrics.get("escape_steps", 0)),
                        int(loop_metrics.get("escape_blocked_by_cooldown", 0)),
                    ),
                    "rp": "%d steps (d=%d, blk_cap=%d, blk_prog=%d)" % (
                        int(loop_metrics.get("reprobe_steps_used", 0)),
                        int(loop_metrics.get("reprobe_dead_steps", 0)),
                        int(loop_metrics.get("reprobe_blocked_by_cap", 0)),
                        int(loop_metrics.get("reprobe_blocked_by_recent_progress", 0)),
                    ),
                    "fu": "%d/%d" % (
                        int(loop_metrics.get("local_followup_successes", 0)),
                        int(loop_metrics.get("local_followup_attempts", 0)),
                    ),
                },
            ), flush=True)

    elapsed_total = time.perf_counter() - start
    summary = {
        "agent": "ProbeFirstAgent",
        "episodes_path": str(episodes_path),
        "completed_episodes": completed,
        "total_episodes_planned": total,
        "elapsed_seconds": elapsed_total,
        "per_game": aggregate,
    }
    save_json(output_root / "collect_probe_summary.json", summary)
    print("[probe] done %d/%d episodes in %.1fs -> %s" % (completed, total, elapsed_total, episodes_path))


if __name__ == "__main__":
    main()
