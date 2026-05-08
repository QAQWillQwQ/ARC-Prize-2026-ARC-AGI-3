"""Phase 3 of worldmodel_v1: evaluate a trained world-model checkpoint via planning rollouts.

Mirrors `src/evaluate.py` but loads a `WorldModelBundle` checkpoint and uses
`WorldModelAgent` for the rollout. Same scoring, same per-game output schema —
so eval JSONs from `evaluate.py` and `evaluate_worldmodel.py` are directly
comparable.

Usage
-----
    python -m src.evaluate_worldmodel \
        --project-root . \
        --checkpoint ./Training_Output/worldmodel_v1/checkpoints/best.pth \
        --output ./Training_Output/worldmodel_v1/eval_public.json \
        --hardware-profile rtx4070super
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from arc_agi import Arcade, OperationMode

from .agent_worldmodel import WorldModelAgent
from .common import (
    episode_level_actions,
    load_metadata_map,
    rhae_score,
    safe_mean,
    save_json,
    split_games,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a worldmodel_v1 checkpoint on public ARC-AGI-3 games.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a worldmodel best.pth.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--games", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", None])
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--stall-steps", type=int, default=24)
    parser.add_argument("--reset-limit", type=int, default=4)
    parser.add_argument("--coord-budget", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=5, help="H — planner rollout depth.")
    parser.add_argument("--rollouts-per-candidate", type=int, default=1, help="K_inner — random rollouts per candidate.")
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def select_games(metadata_map: Dict[str, Dict[str, Any]], requested: Optional[str], split: Optional[str]) -> List[str]:
    if requested:
        return sorted({game.strip() for game in requested.split(",") if game.strip()})
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
    print("[wm-eval] selected_games=%s" % ",".join(selected_games), flush=True)

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(output_path.parent / "eval_worldmodel_recordings"),
    )

    agent = WorldModelAgent(
        checkpoint_path=args.checkpoint,
        max_steps=int(args.max_steps),
        stall_steps=int(args.stall_steps),
        reset_limit=int(args.reset_limit),
        coord_budget=int(args.coord_budget),
        rollout_horizon=int(args.rollout_horizon),
        rollouts_per_candidate=int(args.rollouts_per_candidate),
        gamma=float(args.gamma),
        random_seed=int(args.seed),
    )
    print(
        "[wm-eval] planner H=%d K_inner=%d coord_budget=%d gamma=%.3f"
        % (args.rollout_horizon, args.rollouts_per_candidate, args.coord_budget, args.gamma),
        flush=True,
    )

    per_game: List[Dict[str, Any]] = []
    for game_id in selected_games:
        print("[wm-eval] loading game=%s" % game_id, flush=True)
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
            "[wm-eval] game=%s score=%.6f levels_completed=%d actions_taken=%d final_state=%s"
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
        "agent": "worldmodel",
        "checkpoint": args.checkpoint,
        "num_games": len(per_game),
        "mean_score": safe_mean([item["score"] for item in per_game]),
        "mean_levels_completed": safe_mean([float(item["levels_completed"]) for item in per_game]),
        "planner": {
            "rollout_horizon": int(args.rollout_horizon),
            "rollouts_per_candidate": int(args.rollouts_per_candidate),
            "coord_budget": int(args.coord_budget),
            "gamma": float(args.gamma),
        },
        "games": per_game,
    }
    save_json(output_path, summary)
    print(
        "[wm-eval] mean_score=%.6f mean_levels_completed=%.6f num_games=%d"
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
