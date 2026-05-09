"""Local validation harness for kaggle_notebook/my_agent.py.

Runs the same MyAgent the Kaggle submission uses, but against a LOCAL
arc_agi.Arcade in OFFLINE mode (no gateway, no Kaggle environment). This
lets us iterate on the agent + see real game outcomes before %%writefile-ing
the code into the submission notebook.

Differences from the Kaggle execution path:
  - sys.path is set so MyAgent's `from agents.agent import Agent` resolves
    to the vendored ARC-AGI-3-Agents/ directory.
  - ARC_REPLAY_BASE_DIR points to environment_files/ so MyAgent finds the
    same replays the dataset would expose at /kaggle/working/replays/.
  - The Agent's HTTP-gateway machinery is bypassed: we drive the per-step
    is_done/choose_action loop ourselves and feed actions to env.step().

Usage:
    python scripts/test_my_agent_local.py
    python scripts/test_my_agent_local.py --games sp80,ar25,lp85
    python scripts/test_my_agent_local.py --games cn04,cd82 --max-steps 192
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Replays live alongside game code in the project.
os.environ.setdefault("ARC_REPLAY_BASE_DIR", str(ROOT / "environment_files"))


# Minimal `agents` package shim. The vendored ARC-AGI-3-Agents/agents/__init__.py
# eagerly imports llm/langgraph templates whose deps aren't installed locally.
# We mirror the Kaggle rerun's approach: build a namespace package pointing at
# the framework's agents/ directory but skip the heavy __init__.py — only
# `agents.agent` is needed by MyAgent.
import importlib.util
import types

_AGENTS_DIR = ROOT / "ARC-AGI-3-Agents" / "agents"
_agents_pkg = types.ModuleType("agents")
_agents_pkg.__path__ = [str(_AGENTS_DIR)]
sys.modules["agents"] = _agents_pkg

_agent_spec = importlib.util.spec_from_file_location("agents.agent", _AGENTS_DIR / "agent.py")
_agent_mod = importlib.util.module_from_spec(_agent_spec)
sys.modules["agents.agent"] = _agent_mod
_agent_spec.loader.exec_module(_agent_mod)

# Now safe to import the heavy stuff.
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import FrameData, GameAction, GameState  # noqa: E402

sys.path.insert(0, str(ROOT / "kaggle_notebook"))
import my_agent as _my_agent_mod  # noqa: E402
from my_agent import MyAgent  # noqa: E402

# Import per_game_priors path so we can pre-check
PRIORS_JSON = ROOT / "Local_Output" / "per_game_priors.json"


class _MockArcEnv:
    """Minimal arc_env stand-in. The framework calls arc_env.step(action,
    data=..., reasoning=...) and expects a FrameData-like response. Here we
    just delegate to the real env.step() with the right signature."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.observation_space = env.observation_space

    def step(self, action: GameAction, data: Optional[Dict[str, Any]] = None,
             reasoning: Optional[Any] = None) -> Any:
        # arc_agi.LocalEnvironmentWrapper.step accepts (action, data=...)
        # The framework also passes reasoning but the local env ignores it.
        return self.env.step(action, data=data or {})

    def reset(self) -> Any:
        return self.env.reset()


def _make_agent(game_id: str, env: Any, *, hide_id: bool = False) -> MyAgent:
    """Construct MyAgent with the framework's required ctor args.

    `hide_id=True` masks the game_id so the agent cannot find a per-game
    prior — forces it through the transfer-learning resolution path. Used
    to simulate hidden games without leaving the local 25-game set.
    """
    arc_env = _MockArcEnv(env)
    visible_id = f"hidden-{abs(hash(game_id)) % 100000}" if hide_id else game_id
    agent = MyAgent(
        card_id="local-test",
        game_id=visible_id,
        agent_name="myagent",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=arc_env,
        tags=[],
    )
    return agent


def _convert_obs_to_frame(raw_obs: Any) -> FrameData:
    """Build a FrameData from a LocalEnvironmentWrapper observation.

    The agent's choose_action is called with `latest_frame: FrameData`.
    LocalEnvironmentWrapper's observation_space is already FrameData-shaped.
    """
    # In arc_agi's local wrapper, observation_space IS a FrameData-compatible
    # object — but it's the raw wrapper format (FrameDataRaw). The framework's
    # _convert_raw_frame_data does the conversion to FrameData. We mirror that.
    if isinstance(raw_obs, FrameData):
        return raw_obs
    # Build a FrameData from the raw fields. Same shape as the framework's
    # _convert_raw_frame_data (see ARC-AGI-3-Agents/agents/agent.py).
    frame = raw_obs.frame
    if frame and hasattr(frame[0], "tolist"):
        frame = [arr.tolist() for arr in frame]
    return FrameData(
        game_id=raw_obs.game_id,
        frame=frame,
        state=raw_obs.state,
        levels_completed=raw_obs.levels_completed,
        win_levels=getattr(raw_obs, "win_levels", 0),
        guid=getattr(raw_obs, "guid", "") or "",
        full_reset=getattr(raw_obs, "full_reset", False),
        available_actions=list(getattr(raw_obs, "available_actions", [])),
    )


def _flatten_frame_local(frame: Any) -> Optional[List[List[int]]]:
    if not isinstance(frame, list) or not frame:
        return None
    if isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
        return frame[-1]
    return frame


def _frame_delta_local(prev: Any, cur: Any) -> int:
    if prev is None or cur is None:
        return 0
    n = 0
    for y in range(min(len(prev), len(cur))):
        pr = prev[y]
        cu = cur[y]
        for x in range(min(len(pr), len(cu))):
            if int(pr[x]) != int(cu[x]):
                n += 1
    return n


def run_game(arc: Arcade, game_id: str, *, max_steps: int = 192,
             hide_id: bool = False, record_transitions: bool = False) -> Dict[str, Any]:
    env = arc.make(game_id, seed=42)
    raw = env.observation_space
    if raw is None:
        return {"game_id": game_id, "error": "no observation_space"}
    frame = _convert_obs_to_frame(raw)
    agent = _make_agent(game_id, env, hide_id=hide_id)
    agent.frames = [frame]

    actions_taken = 0
    max_levels = int(frame.levels_completed)
    final_state = "?"
    t0 = time.time()
    error: Optional[str] = None
    transitions: List[Dict[str, Any]] = []  # for episodes.jsonl.gz output

    try:
        while actions_taken < max_steps:
            try:
                latest_frame = agent.frames[-1]
                if agent.is_done(agent.frames, latest_frame):
                    break
                action = agent.choose_action(agent.frames, latest_frame)
            except Exception as exc:
                error = f"agent.choose_action: {exc}"
                traceback.print_exc()
                break

            try:
                action_data = action.action_data.model_dump() if hasattr(action.action_data, "model_dump") else (action.action_data or {})
            except Exception:
                action_data = {}
            # Snapshot pre-step state for the transition row.
            prev_frame_flat = _flatten_frame_local(latest_frame.frame) if record_transitions else None
            prev_state_name = (
                latest_frame.state.name if hasattr(latest_frame.state, "name") else str(latest_frame.state)
            )
            prev_levels = int(latest_frame.levels_completed)
            try:
                next_raw = env.step(action, data=action_data)
            except Exception as exc:
                error = f"env.step: {exc}"
                traceback.print_exc()
                break
            if next_raw is None:
                error = "env returned None"
                break
            next_frame = _convert_obs_to_frame(next_raw)
            agent.append_frame(next_frame)
            actions_taken += 1
            agent.action_counter = actions_taken
            lvl = int(next_frame.levels_completed)
            if lvl > max_levels:
                max_levels = lvl
            final_state = next_frame.state.name if hasattr(next_frame.state, "name") else str(next_frame.state)

            if record_transitions:
                next_flat = _flatten_frame_local(next_frame.frame)
                # If env returns None/empty frame (typically after GAME_OVER
                # with no remaining resets), stop stepping. The agent's loop
                # would otherwise produce a stream of (None, action, None)
                # transitions that break downstream visualization.
                if next_flat is None and prev_frame_flat is None:
                    break
                clean_data = {k: int(v) for k, v in (action_data or {}).items() if k in ("x", "y")}
                try:
                    aid_int = int(action.value if hasattr(action, "value") else 0)
                except Exception:
                    aid_int = 0
                transitions.append({
                    "frame": prev_frame_flat,
                    "available_actions": list(getattr(latest_frame, "available_actions", [])),
                    "action_id": aid_int,
                    "action_data": clean_data if aid_int == 6 else {},
                    "next_frame": next_flat if next_flat is not None else prev_frame_flat,
                    "levels_before": prev_levels,
                    "levels_after": lvl,
                    "state_before": prev_state_name,
                    "state_after": final_state,
                    "delta_pixels": _frame_delta_local(prev_frame_flat, next_flat),
                    "novelty": 0.0,
                    "step_index": actions_taken - 1,
                })
            if final_state == "WIN":
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    wall = time.time() - t0
    out: Dict[str, Any] = {
        "game_id": game_id,
        "max_levels": max_levels,
        "actions_taken": actions_taken,
        "final_state": final_state,
        "wall_seconds": round(wall, 2),
        "error": error,
    }
    if record_transitions:
        out["episode"] = {
            "game_id": game_id,
            "score": float(max_levels),  # rough; inspect_collect just shows it
            "levels_completed": max_levels,
            "actions_taken": actions_taken,
            "final_state": final_state,
            "transitions": transitions,
            "source": "v3_agent_local",
            "episode_id": f"v3_agent_{game_id}_seed42",
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=str, default="sp80,ar25,lp85,cn04,cd82,sb26,ft09,r11l")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--output", type=str, default=str(ROOT / "Local_Output" / "Logs" / "test_my_agent_local.json"))
    parser.add_argument(
        "--no-replays",
        action="store_true",
        help="Point ARC_REPLAY_BASE_DIR at a non-existent path so warmstart "
             "skips entirely. Forces Phase B and lets us validate the "
             "strategy bank + priors path on PUBLIC games.",
    )
    parser.add_argument(
        "--simulate-hidden",
        action="store_true",
        help="Mask the agent's game_id so it cannot find a per-game prior. "
             "Forces transfer-learning resolution. Combine with --no-replays "
             "for the full hidden-game simulation.",
    )
    parser.add_argument(
        "--save-episodes-gz",
        type=str,
        default=None,
        help="Path to write episodes.jsonl.gz. Each game's run becomes one "
             "episode. Use with src/inspect_collect.py to generate per-episode "
             "GIF + summary + trace artifacts.",
    )
    args = parser.parse_args()

    if args.no_replays:
        os.environ["ARC_REPLAY_BASE_DIR"] = str(ROOT / "Local_Output" / "no_replays_for_local_test")
    # Always make priors discoverable for the local harness.
    os.environ.setdefault("ARC_PRIORS_PATH", str(PRIORS_JSON))
    # Override module-level REPLAY_BASE_DIR (set at import time before env edits).
    _my_agent_mod.REPLAY_BASE_DIR = Path(os.environ["ARC_REPLAY_BASE_DIR"])

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    print(f"[local-test] games={games} max_steps={args.max_steps}")
    print(f"[local-test] ARC_REPLAY_BASE_DIR={os.environ['ARC_REPLAY_BASE_DIR']}")
    print(f"[local-test] ARC_PRIORS_PATH={os.environ['ARC_PRIORS_PATH']}")
    print(f"[local-test] per_game_priors exists: {PRIORS_JSON.exists()}")
    print(f"[local-test] no_replays mode: {args.no_replays}")
    print(f"[local-test] simulate_hidden mode: {args.simulate_hidden}")
    print()

    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    results = []
    record = bool(args.save_episodes_gz)
    episodes_gz_path = Path(args.save_episodes_gz) if record else None
    if record and episodes_gz_path is not None:
        episodes_gz_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any prior file at this path.
        if episodes_gz_path.exists():
            episodes_gz_path.unlink()

    for game_id in games:
        print(f"[local-test] === {game_id} ===")
        r = run_game(arc, game_id, max_steps=args.max_steps,
                     hide_id=args.simulate_hidden, record_transitions=record)
        results.append(r)
        if record and r.get("episode") is not None:
            from src.common import append_jsonl_gz  # noqa: E402  local import after sys.path setup
            append_jsonl_gz(episodes_gz_path, r["episode"])
        err = r.get("error")
        marker = "ERR" if err else "OK"
        print(f"[local-test] {marker} {game_id}: levels={r.get('max_levels','?')} actions={r.get('actions_taken','?')} state={r.get('final_state','?')} wall={r.get('wall_seconds',0)}s{f' err={err}' if err else ''}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print()
    print(f"[local-test] wrote {out}")
    print()
    print(f'{"game":<5} | {"levels":>6} | {"actions":>7} | {"state":>14}')
    print("-" * 45)
    for r in results:
        print(f'{r["game_id"]:<5} | {r.get("max_levels", "?"):>6} | {r.get("actions_taken", "?"):>7} | {r.get("final_state", "?"):>14}')
    levels = [int(r.get("max_levels", 0)) for r in results if r.get("error") is None]
    print()
    print(f"mean levels: {sum(levels)/max(len(levels),1):.2f}  (n={len(levels)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
