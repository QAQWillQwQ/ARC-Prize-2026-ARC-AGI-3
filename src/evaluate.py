from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from arc_agi import Arcade, OperationMode

from .agent import PolicyGuidedAgent
from .common import (
    episode_level_actions,
    load_metadata_map,
    rhae_score,
    safe_mean,
    save_json,
    split_games,
)
from .source_planner import SourceSearchPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on public ARC-AGI-3 games.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--games", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", None])
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--stall-steps", type=int, default=24)
    parser.add_argument("--reset-limit", type=int, default=4)
    parser.add_argument("--source-planner", action="store_true")
    parser.add_argument("--planner-timeout", type=float, default=45.0)
    parser.add_argument("--planner-max-states", type=int, default=120000)
    parser.add_argument("--planner-depth", type=int, default=36)
    parser.add_argument("--planner-candidate-budget", type=int, default=24)
    parser.add_argument("--planner-branch-factor", type=int, default=18)
    return parser.parse_args()


def select_games(metadata_map: Dict[str, Dict[str, Any]], requested: Optional[str], split: Optional[str]) -> List[str]:
    if requested:
        return sorted(set(game.strip() for game in requested.split(",") if game.strip()))
    all_games = sorted(metadata_map.keys())
    if split is None:
        return all_games
    train_games, val_games = split_games(all_games)
    return train_games if split == "train" else val_games


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output).resolve()
    metadata_map = load_metadata_map(project_root / "environment_files")
    selected_games = select_games(metadata_map, args.games, args.split)
    print(
        "[eval] selected_games=%s"
        % ",".join(selected_games),
        flush=True,
    )

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(output_path.parent / "eval_recordings"),
    )

    source_planner = None
    if args.source_planner:
        source_planner = SourceSearchPlanner(
            environments_dir=project_root / "environment_files",
            search_timeout=float(args.planner_timeout),
            max_states=int(args.planner_max_states),
            max_depth=int(args.planner_depth),
            candidate_budget=int(args.planner_candidate_budget),
            branch_factor=int(args.planner_branch_factor),
        )

    agent = PolicyGuidedAgent(
        checkpoint_path=args.checkpoint,
        max_steps=args.max_steps,
        stall_steps=args.stall_steps,
        reset_limit=args.reset_limit,
        source_planner=source_planner,
    )

    per_game: List[Dict[str, Any]] = []
    for game_id in selected_games:
        print("[eval] loading game=%s" % game_id, flush=True)
        env = arc.make(game_id)
        if env is None:
            raise RuntimeError("Unable to create environment for %s" % game_id)
        baseline_actions = metadata_map[game_id].get("baseline_actions", [])
        result = agent.play_env(env=env, game_id=game_id, baseline_actions=baseline_actions)
        score_info = rhae_score(
            baseline_actions=baseline_actions,
            completed_level_actions=episode_level_actions(result["transitions"]),
        )
        per_game.append(
            {
                "game_id": game_id,
                "score": float(score_info["score"]),
                "levels_completed": int(result["levels_completed"]),
                "actions_taken": int(result["actions_taken"]),
                "final_state": result["final_state"],
            }
        )
        print(
            "[eval] game=%s score=%.6f levels_completed=%d actions_taken=%d final_state=%s"
            % (
                game_id,
                float(score_info["score"]),
                int(result["levels_completed"]),
                int(result["actions_taken"]),
                result["final_state"],
            ),
            flush=True,
        )

    summary = {
        "checkpoint": args.checkpoint,
        "source_planner": {
            "enabled": bool(args.source_planner),
            "timeout": float(args.planner_timeout),
            "max_states": int(args.planner_max_states),
            "depth": int(args.planner_depth),
            "candidate_budget": int(args.planner_candidate_budget),
            "branch_factor": int(args.planner_branch_factor),
        },
        "num_games": len(per_game),
        "mean_score": safe_mean([item["score"] for item in per_game]),
        "mean_levels_completed": safe_mean([float(item["levels_completed"]) for item in per_game]),
        "games": per_game,
    }
    save_json(output_path, summary)
    print(
        "[eval] mean_score=%.6f mean_levels_completed=%.6f num_games=%d"
        % (
            float(summary["mean_score"]),
            float(summary["mean_levels_completed"]),
            int(summary["num_games"]),
        ),
        flush=True,
    )
    print("Saved evaluation to %s" % output_path)


if __name__ == "__main__":
    main()
