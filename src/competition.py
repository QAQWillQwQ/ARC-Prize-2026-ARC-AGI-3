from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from arc_agi import Arcade, OperationMode

from .agent import PolicyGuidedAgent
from .common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ARC-AGI-3 agent in competition mode.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to ObjectCentricPolicy .pth checkpoint.")
    parser.add_argument("--output-dir", type=str, default="./competition_output")
    parser.add_argument("--games", type=str, default=None,
                        help="Comma-separated short game ids (e.g. sp80,ar25). Default: all 25.")
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--stall-steps", type=int, default=24)
    parser.add_argument("--reset-limit", type=int, default=4)
    parser.add_argument("--scorecard-tags", type=str, default="arcagi3",
                        help="Comma-separated tags for the Kaggle scorecard.")
    return parser.parse_args()


def _build_agent(args: argparse.Namespace) -> PolicyGuidedAgent:
    return PolicyGuidedAgent(
        checkpoint_path=args.checkpoint,
        max_steps=args.max_steps,
        stall_steps=args.stall_steps,
        reset_limit=args.reset_limit,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[competition] checkpoint={args.checkpoint}", flush=True)
    print(f"[competition] max_steps={args.max_steps} stall={args.stall_steps} reset_limit={args.reset_limit}", flush=True)

    arc = Arcade(
        operation_mode=OperationMode.COMPETITION,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(output_dir / "recordings"),
    )
    requested_games = None
    if args.games:
        requested_games = sorted(set(g.strip() for g in args.games.split(",") if g.strip()))

    environments = arc.get_environments()
    if requested_games is not None:
        environments = [e for e in environments if e.game_id.split("-", 1)[0] in requested_games]
    print(f"[competition] running {len(environments)} environments", flush=True)

    tags = [t.strip() for t in args.scorecard_tags.split(",") if t.strip()] or ["arcagi3"]
    card_id = arc.open_scorecard(tags=tags)
    per_game: List[Dict[str, Any]] = []
    overall_t0 = time.time()
    for env_info in environments:
        game_id = env_info.game_id
        short_id = game_id.split("-", 1)[0]
        env = arc.make(game_id, scorecard_id=card_id)
        if env is None:
            continue
        agent = _build_agent(args)
        t0 = time.time()
        try:
            result = agent.play_env(
                env=env,
                game_id=short_id,
                baseline_actions=env_info.baseline_actions or [],
            )
        except Exception as exc:  # one game shouldn't kill the run
            print(f"[competition] {short_id} crashed: {exc}", flush=True)
            per_game.append({"game_id": game_id, "error": str(exc)})
            continue
        wall = time.time() - t0
        print(
            "[competition] %s levels=%d actions=%d state=%s wall=%.1fs"
            % (short_id, result.get("levels_completed", -1), result.get("actions_taken", -1),
               result.get("final_state", "?"), wall),
            flush=True,
        )
        per_game.append({
            "game_id": game_id,
            "levels_completed": result.get("levels_completed"),
            "actions_taken": result.get("actions_taken"),
            "final_state": result.get("final_state"),
            "wall_seconds": wall,
        })
    overall_wall = time.time() - overall_t0
    print(f"[competition] all games done in {overall_wall:.1f}s", flush=True)

    scorecard = arc.close_scorecard(card_id)
    payload = {
        "card_id": card_id,
        "agent_type": "policy",
        "games": per_game,
        "wall_seconds": overall_wall,
        "scorecard": scorecard.model_dump() if scorecard is not None else None,
    }
    save_json(output_dir / "competition_run.json", payload)
    print(f"[competition] scorecard saved to {output_dir / 'competition_run.json'}", flush=True)


if __name__ == "__main__":
    main()
