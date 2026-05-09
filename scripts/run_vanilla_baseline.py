"""Vanilla PolicyGuidedAgent baseline on sp80, ar25, lp85.

Same seed (42) and same per-game prior as the α-stack smoke, but no
worldmodel, no TTT, no goal/effect bonuses. Gives us the floor that the
full α-stack must beat.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arc_agi import Arcade, OperationMode  # noqa: E402

from src.agent import PolicyGuidedAgent  # noqa: E402
from src.replay_loader import build_game_prior  # noqa: E402


def main() -> int:
    out_path = ROOT / "Training_Output" / "worldmodel_v3" / "eval" / "vanilla_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    games = ["sp80", "ar25", "lp85"]
    results = []

    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    print(f"[base] PolicyGuidedAgent (no worldmodel, no TTT, no bonuses)", flush=True)
    print(f"[base] games={games} seed=42", flush=True)

    for game_id in games:
        print(f"[base] === {game_id} ===", flush=True)
        try:
            prior = build_game_prior(ROOT, game_id)
        except Exception as exc:
            print(f"[base] {game_id} prior load failed: {exc}", flush=True)
            traceback.print_exc()
            prior = {}

        agent = PolicyGuidedAgent(
            checkpoint_path=None,
            max_steps=192,
            stall_steps=24,
            reset_limit=4,
            coord_budget=20,
            game_prior=prior,
            random_seed=42,
        )
        try:
            env = arc.make(game_id, seed=42)
        except Exception as exc:
            print(f"[base] {game_id} arc.make failed: {exc}", flush=True)
            traceback.print_exc()
            results.append({"game_id": game_id, "error": f"make: {exc}"})
            continue

        t0 = time.time()
        try:
            r = agent.play_env(env, game_id, baseline_actions=[])
            wall = time.time() - t0
            levels = r.get("levels_completed", "?")
            actions = r.get("actions_taken", "?")
            state = r.get("final_state", "?")
            resets = r.get("resets_used", "?")
            print(
                f"[base] {game_id} levels={levels} actions={actions} "
                f"state={state} resets={resets} wall={wall:.1f}s",
                flush=True,
            )
            slim = {k: v for k, v in r.items() if k != "transitions"}
            slim["wall_seconds"] = wall
            results.append(slim)
        except Exception as exc:
            wall = time.time() - t0
            print(f"[base] {game_id} play_env failed after {wall:.1f}s: {exc}", flush=True)
            traceback.print_exc()
            results.append({"game_id": game_id, "error": f"play_env: {exc}", "wall_seconds": wall})
        finally:
            try:
                env.close()
            except Exception:
                pass

    out_path.write_text(json.dumps(results, indent=2))
    print(f"[base] wrote {out_path}", flush=True)
    print("[base] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
