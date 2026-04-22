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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on public ARC-AGI-3 games.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--games", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", None])
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--stall-steps", type=int, default=24)
    parser.add_argument("--reset-limit", type=int, default=4)
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

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(output_path.parent / "eval_recordings"),
    )

    agent = PolicyGuidedAgent(
        checkpoint_path=args.checkpoint,
        max_steps=args.max_steps,
        stall_steps=args.stall_steps,
        reset_limit=args.reset_limit,
    )

    per_game: List[Dict[str, Any]] = []
    for game_id in selected_games:
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

    summary = {
        "checkpoint": args.checkpoint,
        "num_games": len(per_game),
        "mean_score": safe_mean([item["score"] for item in per_game]),
        "mean_levels_completed": safe_mean([float(item["levels_completed"]) for item in per_game]),
        "games": per_game,
    }
    save_json(output_path, summary)
    print("Saved evaluation to %s" % output_path)


if __name__ == "__main__":
    main()
