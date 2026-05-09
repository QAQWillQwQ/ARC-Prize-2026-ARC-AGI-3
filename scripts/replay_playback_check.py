"""Verify whether GT replay actions, played verbatim against the local env,
reach the same levels as the replay. Tests determinism + version-match.

Usage:
    python scripts/replay_playback_check.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

from src.replay_loader import (  # noqa: E402
    _normalize_action_id,
    load_replay,
    replay_path,
)


def play_replay_actions(arc: Arcade, game_id: str, seeds_to_try=(42, 0, 1, 7, 13)) -> dict:
    """For each seed, feed the replay's actions into env.step() and report
    the maximum level reached vs the replay's expected level at that step.

    Returns a dict per seed with: max_level_local, max_level_replay,
    n_actions_used, final_state.
    """
    p = replay_path(ROOT, game_id)
    if p is None:
        return {"error": f"no replay file for {game_id}"}
    records = load_replay(p)
    replay_max_level = max(
        r.get("data", {}).get("levels_completed", 0) for r in records
    )

    # Local env version vs replay version
    base = ROOT / "environment_files" / game_id
    local_versions = [d.name for d in base.iterdir() if d.is_dir() and d.name != "replays"]
    local_ver = local_versions[0] if local_versions else "?"
    replay_gid = records[0].get("data", {}).get("game_id", "?")
    replay_ver = replay_gid.split("-", 1)[-1] if "-" in replay_gid else "?"
    version_match = (local_ver == replay_ver)

    seed_results = {}
    for seed in seeds_to_try:
        try:
            env = arc.make(game_id, seed=seed)
        except Exception as exc:
            seed_results[seed] = {"error": f"make: {exc}"}
            continue
        max_level_local = 0
        n_actions_used = 0
        n_resets_used = 0
        final_state = "?"
        try:
            obs = env.observation_space
            for i, rec in enumerate(records):
                ai = rec.get("data", {}).get("action_input", {}) or {}
                raw_aid = ai.get("id")
                aid = _normalize_action_id(raw_aid)
                if aid is None:
                    # action_id_raw == 0 marks RESET / initial state. If the
                    # env is currently GAME_OVER, the replay survived by
                    # resetting here — we must do the same to stay in sync.
                    state_name = (
                        obs.state.name if hasattr(obs.state, "name") else str(obs.state)
                    )
                    if state_name == "GAME_OVER":
                        try:
                            obs = env.reset()
                            n_resets_used += 1
                        except Exception:
                            break
                        if obs is None:
                            break
                    continue
                ad = ai.get("data") or {}
                if int(aid) == 6:
                    x = int(ad.get("x", 0))
                    y = int(ad.get("y", 0))
                    action_data = {"x": x, "y": y}
                else:
                    action_data = {}
                game_action = GameAction.from_id(int(aid))
                try:
                    next_obs = env.step(game_action, data=action_data)
                except Exception:
                    break
                if next_obs is None:
                    break
                obs = next_obs
                n_actions_used += 1
                lc = int(getattr(obs, "levels_completed", 0))
                if lc > max_level_local:
                    max_level_local = lc
                state_name = (
                    obs.state.name if hasattr(obs.state, "name") else str(obs.state)
                )
                final_state = state_name
                if state_name == "WIN":
                    break
        finally:
            try:
                env.close()
            except Exception:
                pass
        seed_results[seed] = {
            "max_level_local": max_level_local,
            "n_actions_used": n_actions_used,
            "n_resets_used": n_resets_used,
            "final_state": final_state,
        }

    return {
        "game_id": game_id,
        "local_version": local_ver,
        "replay_version": replay_ver,
        "version_match": version_match,
        "replay_max_level": replay_max_level,
        "replay_action_count": sum(
            1 for r in records
            if _normalize_action_id(r.get("data", {}).get("action_input", {}).get("id")) is not None
        ),
        "by_seed": seed_results,
    }


def main() -> int:
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    out = []
    for g in ["sp80", "ar25", "lp85"]:
        print(f"=== {g} ===", flush=True)
        result = play_replay_actions(arc, g)
        out.append(result)
        print(json.dumps(result, indent=2), flush=True)
        print(flush=True)

    out_path = ROOT / "Training_Output" / "worldmodel_v3" / "eval" / "replay_playback.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
