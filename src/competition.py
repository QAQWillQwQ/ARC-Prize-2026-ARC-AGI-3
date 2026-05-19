from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from arc_agi import Arcade, OperationMode

from .agent import PolicyGuidedAgent
from .common import save_json
from .source_planner import SourceSearchPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal ARC-AGI-3 agent in competition mode.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./competition_output")
    parser.add_argument("--games", type=str, default=None)
    parser.add_argument("--operation-mode", type=str, default="competition", choices=["competition", "offline"])
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


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    operation_mode = OperationMode.COMPETITION if args.operation_mode == "competition" else OperationMode.OFFLINE
    arc = Arcade(
        operation_mode=operation_mode,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(output_dir / "recordings"),
    )
    requested_games = None
    if args.games:
        requested_games = sorted(set(game.strip() for game in args.games.split(",") if game.strip()))

    environments = arc.get_environments()
    if requested_games is not None:
        environments = [env for env in environments if env.game_id.split("-", 1)[0] in requested_games]

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

    card_id = arc.open_scorecard(tags=["arcagi3", "minimal-hybrid-agent"])
    per_game: List[Dict[str, Any]] = []
    for env_info in environments:
        game_id = env_info.game_id
        env = arc.make(game_id, scorecard_id=card_id)
        if env is None:
            continue
        result = agent.play_env(env=env, game_id=game_id.split("-", 1)[0], baseline_actions=env_info.baseline_actions or [])
        per_game.append(
            {
                "game_id": game_id,
                "levels_completed": result["levels_completed"],
                "actions_taken": result["actions_taken"],
                "final_state": result["final_state"],
            }
        )

    scorecard = arc.close_scorecard(card_id)
    payload = {
        "card_id": card_id,
        "operation_mode": args.operation_mode,
        "games": per_game,
        "source_planner": {
            "enabled": bool(args.source_planner),
            "timeout": float(args.planner_timeout),
            "max_states": int(args.planner_max_states),
            "depth": int(args.planner_depth),
            "candidate_budget": int(args.planner_candidate_budget),
            "branch_factor": int(args.planner_branch_factor),
        },
        "scorecard": scorecard.model_dump() if scorecard is not None else None,
    }
    save_json(output_dir / "competition_run.json", payload)
    print("Competition run finished. Scorecard saved to %s" % (output_dir / "competition_run.json"))


if __name__ == "__main__":
    main()
