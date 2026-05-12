"""Detailed wm_quick_v2-style inspection for top-performing games.

For each game, runs the agent and records per-step:
  - action_id, x, y (if click)
  - frame_delta to previous frame (pixel-diff count)
  - levels_completed delta (when level cleared)
  - active strategy (warmstart / fallback / specific strategy name)
  - state (NOT_FINISHED / WIN / GAME_OVER)

Then prints per-game:
  - levels reached + the actions that cleared each level
  - per-level action counts (so we see "level 1 took N actions")
  - dead-action rate (actions with frame_delta == 0)
  - strategy breakdown (warmstart vs each Phase B strategy)
  - timeline of level-clearing events

Usage:
    python scripts/inspect_top_performers.py
    python scripts/inspect_top_performers.py --games lp85,r11l,tu93
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
import types
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ARC_REPLAY_BASE_DIR", str(ROOT / "environment_files"))
os.environ.setdefault("ARC_PRIORS_PATH", str(ROOT / "Local_Output" / "per_game_priors.json"))

# Mock minimal agents package (skips framework's heavy __init__.py).
_AGENTS_DIR = ROOT / "ARC-AGI-3-Agents" / "agents"
_agents_pkg = types.ModuleType("agents")
_agents_pkg.__path__ = [str(_AGENTS_DIR)]
sys.modules["agents"] = _agents_pkg
_agent_spec = importlib.util.spec_from_file_location("agents.agent", _AGENTS_DIR / "agent.py")
_agent_mod = importlib.util.module_from_spec(_agent_spec)
sys.modules["agents.agent"] = _agent_mod
_agent_spec.loader.exec_module(_agent_mod)

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import FrameData, GameAction, GameState  # noqa: E402

sys.path.insert(0, str(ROOT / "kaggle_notebook" / "agents"))
from my_agent import MyAgent  # noqa: E402


def _frame_delta(prev: Any, cur: Any) -> int:
    if prev is None or cur is None:
        return 0
    try:
        n = 0
        for y in range(min(len(prev), len(cur))):
            pr = prev[y]
            cu = cur[y]
            for x in range(min(len(pr), len(cu))):
                if int(pr[x]) != int(cu[x]):
                    n += 1
        return n
    except Exception:
        return 0


def _convert_obs_to_frame(raw_obs: Any) -> FrameData:
    if isinstance(raw_obs, FrameData):
        return raw_obs
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


def _flatten_frame(frame: Any) -> Optional[List[List[int]]]:
    if not isinstance(frame, list) or not frame:
        return None
    if isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
        return frame[-1]
    return frame


class _MockArcEnv:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.observation_space = env.observation_space

    def step(self, action: GameAction, data: Optional[Dict[str, Any]] = None,
             reasoning: Optional[Any] = None) -> Any:
        return self.env.step(action, data=data or {})


def inspect_game(arc: Arcade, game_id: str, max_steps: int = 192) -> Dict[str, Any]:
    env = arc.make(game_id, seed=42)
    raw = env.observation_space
    if raw is None:
        return {"game_id": game_id, "error": "no observation_space"}
    frame = _convert_obs_to_frame(raw)
    arc_env = _MockArcEnv(env)
    agent = MyAgent(
        card_id="local-inspect",
        game_id=game_id,
        agent_name="myagent",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=arc_env,
        tags=[],
    )
    agent.frames = [frame]

    timeline: List[Dict[str, Any]] = []
    prev_flat = _flatten_frame(frame.frame)
    prev_levels = int(frame.levels_completed)
    max_levels = prev_levels
    level_clearing_events: List[Dict[str, Any]] = []
    actions_per_level: Counter = Counter()
    dead_actions = 0
    n_actions = 0
    strategy_counts: Counter = Counter()
    final_state = "?"
    error: Optional[str] = None

    try:
        while n_actions < max_steps:
            latest_frame = agent.frames[-1]
            try:
                if agent.is_done(agent.frames, latest_frame):
                    break
                action = agent.choose_action(agent.frames, latest_frame)
            except Exception as exc:
                error = f"choose_action: {exc}"
                traceback.print_exc()
                break

            # Identify strategy from action.reasoning
            reasoning = getattr(action, 'reasoning', None) or {}
            if isinstance(reasoning, dict):
                phase = reasoning.get('phase', '?')
                strategy = reasoning.get('strategy', reasoning.get('marker', phase))
            else:
                phase = 'warmstart' if 'warmstart' in str(reasoning) else 'fallback'
                strategy = phase
            strategy_counts[strategy] += 1

            try:
                action_data = action.action_data.model_dump() if hasattr(action.action_data, 'model_dump') else (action.action_data or {})
            except Exception:
                action_data = {}
            try:
                next_raw = env.step(action, data=action_data)
            except Exception as exc:
                error = f"env.step: {exc}"
                break
            if next_raw is None:
                break
            next_frame = _convert_obs_to_frame(next_raw)
            agent.append_frame(next_frame)
            n_actions += 1
            agent.action_counter = n_actions

            cur_flat = _flatten_frame(next_frame.frame)
            delta = _frame_delta(prev_flat, cur_flat)
            if delta == 0:
                dead_actions += 1
            cur_levels = int(next_frame.levels_completed)
            if cur_levels > max_levels:
                max_levels = cur_levels
                level_clearing_events.append({
                    "step": n_actions,
                    "from_level": prev_levels,
                    "to_level": cur_levels,
                    "action_id": int(action.value if hasattr(action, 'value') else 0),
                    "x": int((action_data or {}).get('x', 0)) if action_data else 0,
                    "y": int((action_data or {}).get('y', 0)) if action_data else 0,
                    "strategy": strategy,
                    "phase": phase,
                })
            actions_per_level[prev_levels] += 1
            prev_flat = cur_flat
            prev_levels = cur_levels

            try:
                aid = int(action.value if hasattr(action, 'value') else 0)
            except Exception:
                aid = 0
            timeline.append({
                "step": n_actions,
                "action_id": aid,
                "x": int((action_data or {}).get('x', 0)),
                "y": int((action_data or {}).get('y', 0)),
                "frame_delta": delta,
                "level": cur_levels,
                "strategy": strategy,
                "phase": phase,
            })

            final_state = next_frame.state.name if hasattr(next_frame.state, 'name') else str(next_frame.state)
            if final_state == "WIN":
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    return {
        "game_id": game_id,
        "max_levels": max_levels,
        "actions_taken": n_actions,
        "final_state": final_state,
        "dead_actions": dead_actions,
        "dead_action_rate": round(dead_actions / max(n_actions, 1), 3),
        "level_clearing_events": level_clearing_events,
        "actions_per_level": dict(actions_per_level),
        "strategy_counts": dict(strategy_counts),
        "error": error,
        "timeline": timeline,  # detailed per-step
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=str, default="lp85,r11l,tu93,ar25,sc25,tr87")
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--output", type=str, default=str(ROOT / "Local_Output" / "Logs" / "inspect_top_performers.json"))
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    results = {}
    for g in games:
        print(f"=== {g} ===", flush=True)
        r = inspect_game(arc, g, max_steps=args.max_steps)
        results[g] = r
        print(f"  max_levels={r['max_levels']}  actions={r['actions_taken']}  state={r['final_state']}  "
              f"dead_rate={r['dead_action_rate']}", flush=True)
        print(f"  strategies: {r['strategy_counts']}", flush=True)
        if r.get('level_clearing_events'):
            print(f"  level transitions:", flush=True)
            for ev in r['level_clearing_events']:
                coord = f"({ev['x']},{ev['y']})" if ev['action_id'] == 6 else ""
                print(f"    step {ev['step']}: lvl {ev['from_level']}→{ev['to_level']} via aid={ev['action_id']}{coord} (phase={ev['phase']})", flush=True)
        print(f"  actions_per_level: {r['actions_per_level']}", flush=True)
        print(flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
