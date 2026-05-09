"""Collect deep training trajectories for all 25 public games.

For each (game, deviation, seed) and (game, perturb_rate, seed) tuple, this
script produces one episode in episodes.jsonl.gz. All buckets share output;
the `source` and tuning fields tag each episode for downstream filtering.

Buckets covered (per .claude/doc/bc_v2_data_collection_plan.md):

  1. gt_verbatim                — full GT, no Phase B. Deterministic per game,
                                  so 1 task per game (seeds collapsed).
  3. gt_warmstart_partial       — GT to deviation_step, then PolicyGuidedAgent
                                  heuristic for the remainder.
  4. gt_warmstart_perturbed     — GT verbatim but with perturb_rate chance of
                                  each action being replaced by random.
  5. heuristic_only             — no GT, full Phase B from level 0.

Bucket 2 (full GT then explore beyond) is skipped — public-game GT already
reaches WIN with the v4 tactical-RESET fix; there's no "beyond WIN" to
explore.

Diversity per task comes from:
  - Deviation point (5 options: 0%, 25%, 50%, 75%, 100% of GT length)
  - Agent seed (default 4 per deviation) — PolicyGuidedAgent is stochastic
    per seed (random click pattern, random direction roll, random reset
    timing) so each seed produces a distinct trajectory after deviation.
  - Perturbation rate for Bucket 4 (default {0.05, 0.10, 0.15})

Default counts: 25 games × (5 dev × 4 seeds + 3 perturb × 4 seeds) ≈ 800
tasks. Tune via flags. Each task is ~5–30 s on a CPU worker.

Usage on Openlab (~32+ cores recommended):

    python scripts/collect_gt_warmstart.py \\
        --project-root . \\
        --output-root Local_Output/Collection_Cache/gt_warmstart_v1 \\
        --workers 32 \\
        --deviation-points 0,25,50,75,100 \\
        --seeds-per-deviation 4 \\
        --perturb-rates 0.05,0.10,0.15 \\
        --perturb-seeds 4

Output: <output-root>/collected/episodes.jsonl.gz with episodes tagged by
`source`, `deviation_pct`, `perturb_rate`, and `seed`.
"""
from __future__ import annotations

import argparse
import json
import random as pyrandom
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

# NOTE: src.agent transitively imports torch (~500MB per worker process).
# When every Phase B task has a forced strategy (the typical case), the
# PolicyGuidedAgent fallback below is unreachable. We lazy-import inside the
# else branch so workers that never hit it don't load torch at all.
# Memory savings: ~500MB × n_workers.
from src.common import (  # noqa: E402
    append_jsonl_gz,
    final_subframe,
    frame_delta,
    load_metadata_map,
    save_json,
    seed_everything,
    utc_timestamp,
)


# ============================================================================
# Per-worker cached Arcade + env instances (the real memory fix).
# ============================================================================
# Pre-v: each task called `arc.make(game_id)` → fresh env + fresh game class.
# Across 12,425 tasks that's 12,425 leaked envs (the user's 50 GB blowup).
#
# Now: ONE Arcade per worker, plus a per-worker dict of {short_id: env} that
# caches up to 25 envs. Each task fetches the cached env and calls env.reset()
# instead of re-creating. The 25 envs become a fixed working set per worker.
# Memory growth across tasks: ~zero.
_WORKER_ARC: Optional[Arcade] = None
_WORKER_ENVS: Dict[str, Any] = {}


def _init_worker(project_root_str: str, recordings_dir_str: str, priors_path_str: Optional[str]) -> None:
    """ProcessPoolExecutor initializer. Creates one Arcade per worker."""
    global _WORKER_ARC, _WORKER_ENVS
    if priors_path_str:
        import os
        os.environ.setdefault("ARC_PRIORS_PATH", priors_path_str)
    _WORKER_ARC = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(Path(project_root_str) / "environment_files"),
        recordings_dir=str(recordings_dir_str),
    )
    _WORKER_ENVS = {}


def _get_or_make_env(arc: Arcade, full_id: str) -> Any:
    """Return a cached env for `full_id`, creating + caching it on first use.

    For repeat calls within the same worker, returns the SAME env instance
    (after env.reset() — see collect_one). Across all worker tasks, a worker
    holds at most 25 envs (one per game), no matter how many tasks run.
    """
    global _WORKER_ENVS
    env = _WORKER_ENVS.get(full_id)
    if env is None:
        env = arc.make(full_id)
        if env is not None:
            _WORKER_ENVS[full_id] = env
    return env


# Phase-B strategies sourced from kaggle_notebook/my_agent.py:
ALL_STRATEGIES: Tuple[str, ...] = (
    "random_full",
    "keyboard_only",
    "click_grid",
    "directional_sustained",
    "edge_sweep",
    "color_targeted",
    "action_id_sweep",
)


def _stub_agents_module() -> None:
    """Provide a minimal `agents.agent.Agent` stub so MyAgent can import.

    MyAgent imports `from agents.agent import Agent` (Kaggle framework base
    class). For collect we don't need the real Kaggle framework's helpers
    (Recorder, GUI, etc.) — just an Agent base with the few attributes
    MyAgent reads from `self`. Idempotent across worker processes.
    """
    if "agents.agent" in sys.modules:
        return
    import types
    am = types.ModuleType("agents")
    aam = types.ModuleType("agents.agent")

    class _Agent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.game_id = kwargs.get("game_id", "")
            self.frames: List[Any] = []
            self.guid = None
            self.is_playback = False
            self.action_counter = 0

    aam.Agent = _Agent
    am.agent = aam
    sys.modules["agents"] = am
    sys.modules["agents.agent"] = aam


# ---------------------- replay loader (port of MyAgent's v4 logic) ----------------------


def _normalize_action_id(raw: Any) -> int:
    if raw is None:
        return 0
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw if 1 <= raw <= 7 else 0
    if isinstance(raw, str):
        s = raw.strip().upper()
        if s == "RESET":
            return 0
        if s.startswith("ACTION") and s[6:].isdigit():
            n = int(s[6:])
            return n if 1 <= n <= 7 else 0
    return 0


def _load_replay_records(short_id: str, project_root: Path) -> List[Dict[str, Any]]:
    rd = project_root / "environment_files" / short_id / "replays"
    if not rd.is_dir():
        return []
    files = sorted(rd.glob("*.json"))
    if not files:
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(files[0], "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ai = (rec.get("data") or {}).get("action_input") or {}
                aid = _normalize_action_id(ai.get("id"))
                if aid == 0:
                    out.append({"type": "reset"})
                    continue
                ad = ai.get("data") or {}
                try:
                    x = int(ad.get("x", 0)) if aid == 6 else 0
                    y = int(ad.get("y", 0)) if aid == 6 else 0
                except (TypeError, ValueError):
                    x, y = 0, 0
                out.append({"type": "action", "id": aid, "x": x, "y": y})
    except Exception:
        return []
    return out


# ---------------------- shared step helper ----------------------


def _step_env(env: Any, action: GameAction) -> Any:
    """env.step with action_data extraction. Returns next obs or None."""
    try:
        if action.is_complex():
            data = (
                action.action_data.model_dump()
                if hasattr(action.action_data, "model_dump")
                else (action.action_data or {})
            )
        else:
            data = {}
    except Exception:
        data = {}
    try:
        return env.step(action, data=data)
    except Exception:
        return None


def _record_transition(
    transitions: List[Dict[str, Any]],
    prev_obs: Any,
    next_obs: Any,
    action_id: int,
    action_data: Dict[str, int],
    step_index: int,
    phase: str,
) -> None:
    prev_frame = final_subframe(prev_obs.frame)
    next_frame = final_subframe(next_obs.frame)
    transitions.append({
        "frame": [list(row) for row in prev_frame],
        "available_actions": list(getattr(prev_obs, "available_actions", []) or []),
        "action_id": int(action_id),
        "action_data": dict(action_data) if action_data else {},
        "next_frame": [list(row) for row in next_frame],
        "levels_before": int(getattr(prev_obs, "levels_completed", 0) or 0),
        "levels_after": int(getattr(next_obs, "levels_completed", 0) or 0),
        "state_before": prev_obs.state.name if hasattr(prev_obs.state, "name") else str(prev_obs.state),
        "state_after": next_obs.state.name if hasattr(next_obs.state, "name") else str(next_obs.state),
        "delta_pixels": frame_delta(prev_frame, next_frame),
        "novelty": 0.0,
        "step_index": int(step_index),
        "ttt_phase": phase,
    })


# ---------------------- GT warmstart playback (v4 tactical-RESET preserving) ----------------------


def play_gt_warmstart(
    env: Any,
    replay: Sequence[Dict[str, Any]],
    raw_obs: Any,
    deviation_step: int,
    max_steps: int,
    perturb_rate: float = 0.0,
    rng: Optional[pyrandom.Random] = None,
) -> Tuple[Any, int, int, List[Dict[str, Any]]]:
    """Play GT verbatim with v4 tactical-RESET preservation.

    - ALWAYS issue RESET when next entry is a reset marker (preserves human's
      tactical resets in painter games like cd82).
    - When env enters NOT_PLAYED/GAME_OVER but next entry is an action,
      advance replay_idx past the next reset marker forward to re-sync at
      attempt boundaries.
    - With perturb_rate > 0 (Bucket 4), each non-reset action has perturb_rate
      chance of being replaced by a random valid action — produces "near-GT"
      data with controlled noise.

    Returns (raw_obs_after, n_steps_used, n_resets_used, transitions).
    """
    if rng is None:
        rng = pyrandom.Random(0)
    transitions: List[Dict[str, Any]] = []
    n_steps = 0
    n_resets = 0
    replay_idx = 0
    cap = min(deviation_step, max_steps)
    while replay_idx < len(replay) and n_steps < cap:
        entry = replay[replay_idx]
        state_name = (
            raw_obs.state.name if hasattr(raw_obs.state, "name") else str(raw_obs.state)
        )
        if state_name == GameState.WIN.name:
            break
        if entry.get("type") == "reset":
            replay_idx += 1
            try:
                raw_obs = env.reset()
            except Exception:
                break
            if raw_obs is None:
                break
            n_resets += 1
            continue
        if state_name in (GameState.NOT_PLAYED.name, GameState.GAME_OVER.name):
            next_reset = None
            for i in range(replay_idx, len(replay)):
                if replay[i].get("type") == "reset":
                    next_reset = i
                    break
            if next_reset is None:
                replay_idx = len(replay)
                break
            replay_idx = next_reset + 1
            try:
                raw_obs = env.reset()
            except Exception:
                break
            if raw_obs is None:
                break
            n_resets += 1
            continue
        # Healthy state, action entry — play (possibly perturbed).
        replay_idx += 1
        if perturb_rate > 0.0 and rng.random() < perturb_rate:
            # Replace with random valid action.
            available = list(getattr(raw_obs, "available_actions", []) or [])
            if available:
                aid = int(rng.choice(available))
            else:
                aid = int(entry["id"])
            x = rng.randint(0, 63) if aid == 6 else 0
            y = rng.randint(0, 63) if aid == 6 else 0
        else:
            aid = int(entry["id"])
            x = int(entry.get("x", 0))
            y = int(entry.get("y", 0))
        try:
            action = GameAction.from_id(aid)
        except Exception:
            continue
        if action.is_complex():
            try:
                action.set_data({"x": int(x), "y": int(y)})
            except Exception:
                pass
        prev_obs = raw_obs
        next_obs = _step_env(env, action)
        if next_obs is None:
            break
        action_data_dict = {"x": int(x), "y": int(y)} if action.is_complex() else {}
        _record_transition(
            transitions, prev_obs, next_obs, aid, action_data_dict, n_steps, "warmstart",
        )
        raw_obs = next_obs
        n_steps += 1
    return raw_obs, n_steps, n_resets, transitions


# ---------------------- one collect task ----------------------


def _phase_b_with_myagent(
    env: Any,
    raw_obs: Any,
    short_id: str,
    full_id: str,
    forced_strategy: str,
    remaining_budget: int,
    seed: int,
    starting_step_index: int,
) -> Tuple[Any, List[Dict[str, Any]], int]:
    """Run Phase B using MyAgent with a single forced strategy.

    Disables strategy rotation/exploration, locks `_strategy_idx` to the
    chosen strategy, and runs `choose_action` until the budget exhausts or
    env reaches WIN.

    Returns (raw_obs_after, transitions, n_resets).
    """
    _stub_agents_module()
    # Tell MyAgent NOT to load replay (so Phase A doesn't fire on top of ours).
    # MUST also pin ARC_PRIORS_PATH explicitly — otherwise MyAgent derives the
    # priors lookup paths from REPLAY_BASE_DIR.parent and won't find them once
    # we override REPLAY_BASE_DIR to a fake location.
    import os
    os.environ["ARC_REPLAY_BASE_DIR"] = "/__nonexistent_replay_base_dir__"
    priors_path = ROOT / "Local_Output" / "per_game_priors.json"
    if priors_path.is_file() and not os.environ.get("ARC_PRIORS_PATH"):
        os.environ["ARC_PRIORS_PATH"] = str(priors_path)
    import importlib
    if "my_agent" in sys.modules:
        importlib.reload(sys.modules["my_agent"])
    sys.path.insert(0, str(ROOT / "kaggle_notebook"))
    import my_agent  # noqa: E402

    if forced_strategy not in my_agent.MyAgent.STRATEGIES:
        forced_strategy = "random_full"
    strategy_idx = my_agent.MyAgent.STRATEGIES.index(forced_strategy)
    agent = my_agent.MyAgent(game_id=full_id)
    # Lock the agent to the forced strategy. _advance_strategy resets the
    # freeze streak (otherwise the freeze-detection branch in choose_action
    # would log "freeze detected" every step on long stalls).
    agent._initial_strategy_idx = strategy_idx
    agent._strategy_idx = strategy_idx

    def _no_advance(*args: Any, **kwargs: Any) -> None:
        # Set streak deeply negative so choose_action's freeze branch
        # (`if streak >= 8 ...`) never re-fires for the rest of the episode.
        # Each `_record_last_action_effect` call only increments by 1 on
        # zero-delta steps, so after thousands of steps it still won't reach 8.
        agent._frozen_streak = -100000

    agent._advance_strategy = _no_advance  # type: ignore[assignment]
    agent._frozen_streak = -100000
    # Replay is empty so Phase A is skipped.
    agent._replay = []
    agent._replay_idx = 0

    transitions: List[Dict[str, Any]] = []
    n_resets = 0
    step = 0
    cur_obs = raw_obs
    while step < remaining_budget:
        try:
            agent.append_frame(cur_obs)
        except Exception:
            pass
        state_name = cur_obs.state.name if hasattr(cur_obs.state, "name") else str(cur_obs.state)
        if state_name == GameState.WIN.name:
            break
        if state_name in (GameState.NOT_PLAYED.name, GameState.GAME_OVER.name):
            try:
                cur_obs = env.reset()
            except Exception:
                break
            n_resets += 1
            if cur_obs is None:
                break
            continue
        try:
            action = agent.choose_action(agent.frames, cur_obs)
        except Exception:
            break
        prev_obs = cur_obs
        try:
            aid = int(action.value if hasattr(action, "value") else 0)
        except Exception:
            aid = 0
        if aid == 0:
            # RESET — handled by env.reset path next iteration.
            try:
                cur_obs = env.reset()
            except Exception:
                break
            n_resets += 1
            if cur_obs is None:
                break
            continue
        next_obs = _step_env(env, action)
        if next_obs is None:
            break
        try:
            d = action.action_data.model_dump() if hasattr(action.action_data, "model_dump") else (action.action_data or {})
        except Exception:
            d = {}
        action_data_dict = (
            {"x": int(d.get("x", 0)), "y": int(d.get("y", 0))}
            if action.is_complex()
            else {}
        )
        _record_transition(
            transitions, prev_obs, next_obs, aid, action_data_dict,
            starting_step_index + step, "explore",
        )
        cur_obs = next_obs
        step += 1
    return cur_obs, transitions, n_resets


def collect_one(
    full_id: str,
    short_id: str,
    deviation_pct: float,
    seed: int,
    max_steps: int,
    perturb_rate: float,
    strategy: Optional[str],
    project_root: Path,
    recordings_dir: Path,
) -> Dict[str, Any]:
    """Run one episode and return the payload."""
    seed_everything(seed)
    rng = pyrandom.Random(seed)
    # Reuse the per-worker Arcade (set by _init_worker). Falls back to a fresh
    # Arcade if running outside ProcessPoolExecutor (e.g., direct invocation).
    arc = _WORKER_ARC
    if arc is None:
        arc = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(project_root / "environment_files"),
            recordings_dir=str(recordings_dir),
        )
    # Fetch or create the cached env for this game. The cached env is reused
    # across all tasks for the same game_id within this worker — only 25 envs
    # max per worker no matter how many tasks. Memory growth across tasks: ~0.
    env = _get_or_make_env(arc, full_id)
    if env is None:
        return {"game_id": full_id, "error": "env_make_returned_None"}
    # Reset to a clean initial state. For a freshly-made env this is a no-op;
    # for a cached env it restarts the game so the next task starts cleanly.
    try:
        raw_obs = env.reset()
    except Exception:
        raw_obs = env.observation_space
    if raw_obs is None:
        raw_obs = env.observation_space
    if raw_obs is None:
        return {"game_id": full_id, "error": "no_observation_space"}

    replay = _load_replay_records(short_id, project_root)

    # Decide bucket + parameters.
    if perturb_rate > 0.0:
        source_tag = "gt_warmstart_perturbed"
        deviation_step = len(replay)
    elif deviation_pct >= 1.0:
        source_tag = "gt_verbatim"
        deviation_step = len(replay)
    elif deviation_pct <= 0.0:
        source_tag = "heuristic_only"
        deviation_step = 0
    else:
        source_tag = "gt_warmstart_partial"
        deviation_step = int(deviation_pct * len(replay)) if replay else 0

    transitions: List[Dict[str, Any]] = []
    n_resets = 0

    # Phase A: GT warmstart (possibly perturbed).
    if deviation_step > 0 and replay:
        raw_obs, gt_n, gt_resets, gt_trans = play_gt_warmstart(
            env=env,
            replay=replay,
            raw_obs=raw_obs,
            deviation_step=deviation_step,
            max_steps=max_steps,
            perturb_rate=perturb_rate,
            rng=rng,
        )
        transitions.extend(gt_trans)
        n_resets += gt_resets

    # Phase B (only for buckets that explore).
    state_name = (
        raw_obs.state.name if hasattr(raw_obs.state, "name") else str(raw_obs.state)
    )
    remaining = max_steps - len(transitions)
    do_phase_b = (
        remaining > 0
        and state_name != GameState.WIN.name
        and source_tag in ("heuristic_only", "gt_warmstart_partial")
    )
    if do_phase_b:
        if strategy:
            try:
                raw_obs, pb_trans, pb_resets = _phase_b_with_myagent(
                    env=env,
                    raw_obs=raw_obs,
                    short_id=short_id,
                    full_id=full_id,
                    forced_strategy=strategy,
                    remaining_budget=remaining,
                    seed=seed,
                    starting_step_index=len(transitions),
                )
                transitions.extend(pb_trans)
                n_resets += pb_resets
            except Exception:
                traceback.print_exc()
        else:
            # Default Phase B: PolicyGuidedAgent (no strategy override).
            # Lazy import — keeps torch out of workers that never reach this path.
            from src.agent import PolicyGuidedAgent  # noqa: E402
            agent = PolicyGuidedAgent(
                checkpoint_path=None,
                max_steps=remaining,
                stall_steps=24,
                reset_limit=4,
                random_seed=seed,
            )
            try:
                result = agent.play_env(env=env, game_id=short_id, baseline_actions=[])
                for t in result.get("transitions", []):
                    t["ttt_phase"] = "explore"
                    t["step_index"] = len(transitions)
                    transitions.append(t)
                n_resets += int(result.get("resets_used", 0) or 0)
                raw_obs = env.observation_space
            except Exception:
                traceback.print_exc()

    final_state = (
        raw_obs.state.name if hasattr(raw_obs.state, "name") else str(raw_obs.state)
    )
    levels = int(getattr(raw_obs, "levels_completed", 0) or 0)
    return {
        "game_id": full_id,
        "short_id": short_id,
        "source": source_tag,
        "deviation_pct": float(deviation_pct),
        "deviation_step": int(deviation_step),
        "perturb_rate": float(perturb_rate),
        "seed": int(seed),
        "strategy": strategy,
        "score": float(levels),
        "levels_completed": levels,
        "actions_taken": len(transitions),
        "final_state": final_state,
        "resets_used": n_resets,
        "transitions": transitions,
    }


# ---------------------- task generation ----------------------


def generate_tasks(
    games: List[Tuple[str, str]],  # (full_id, short_id)
    deviations: List[float],
    seeds_per_deviation: int,
    perturb_rates: List[float],
    perturb_seeds: int,
    strategies: List[str],
) -> List[Tuple[str, str, float, int, float, Optional[str]]]:
    """Build (full_id, short_id, deviation_pct, seed, perturb_rate, strategy) tasks.

    Strategy dimension applies ONLY to deviations < 1.0 (where Phase B fires).
    For dev=100% (gt_verbatim) and Bucket 4 (perturbation), strategy is None.
    Dup-seed elision: dev=100% has only 1 task per game (env deterministic).
    """
    tasks: List[Tuple[str, str, float, int, float, Optional[str]]] = []
    for full_id, short_id in games:
        for dev in deviations:
            if dev >= 1.0:
                # gt_verbatim — single deterministic episode per game.
                tasks.append((full_id, short_id, 1.0, 0, 0.0, None))
                continue
            for seed in range(seeds_per_deviation):
                for strat in strategies:
                    tasks.append((full_id, short_id, float(dev), seed, 0.0, strat))
        # Bucket 4: perturbation, no strategy needed (Phase B doesn't run).
        for rate in perturb_rates:
            for seed in range(perturb_seeds):
                tasks.append((full_id, short_id, 1.0, seed + 10000, float(rate), None))
    return tasks


# ---------------------- driver ----------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--games", type=str, default=None,
                   help="Comma-separated short ids. Default: all 25.")
    p.add_argument("--deviation-points", type=str, default="0,25,50,75,100",
                   help="Comma-separated deviation pcts (0-100). Default: 0,25,50,75,100")
    p.add_argument("--seeds-per-deviation", type=int, default=4)
    p.add_argument("--perturb-rates", type=str, default="0.05,0.10,0.15",
                   help="Comma-separated perturbation rates for Bucket 4. Set to '' to disable.")
    p.add_argument("--perturb-seeds", type=int, default=4)
    p.add_argument("--strategies", type=str,
                   default="random_full,keyboard_only,click_grid,directional_sustained,edge_sweep,color_targeted,action_id_sweep",
                   help="Comma-separated Phase B strategies (from MyAgent.STRATEGIES). "
                        "Forced per task during Phase B. Default: all 7.")
    p.add_argument("--max-steps", type=int, default=1500,
                   help="Per-task action budget. 1500 covers the longest GT replay (wa30=1565). "
                        "Higher costs memory linearly per worker. Default: 1500.")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    out_dir = output_root / "collected"
    out_dir.mkdir(parents=True, exist_ok=True)
    gz_path = out_dir / "episodes.jsonl.gz"
    recordings_dir = output_root / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)

    metas = load_metadata_map(project_root / "environment_files")
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(recordings_dir),
    )
    full_ids: Dict[str, str] = {}
    for e in arc.get_environments():
        sid = e.game_id.split("-", 1)[0]
        full_ids.setdefault(sid, e.game_id)

    if args.games:
        wanted = sorted(set(g.strip() for g in args.games.split(",") if g.strip()))
    else:
        wanted = sorted(full_ids.keys())

    deviations = [float(p.strip()) / 100.0 for p in args.deviation_points.split(",") if p.strip()]
    perturb_rates = [float(p.strip()) for p in args.perturb_rates.split(",") if p.strip()] if args.perturb_rates else []
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()] if args.strategies else []
    if not strategies:
        strategies = list(ALL_STRATEGIES)

    games_pairs = [(full_ids[s], s) for s in wanted if s in full_ids]
    tasks = generate_tasks(
        games_pairs,
        deviations,
        int(args.seeds_per_deviation),
        perturb_rates,
        int(args.perturb_seeds),
        strategies,
    )
    total = len(tasks)
    print(
        f"[collect-gt] games={len(wanted)} dev={deviations} seeds={args.seeds_per_deviation} "
        f"strategies={strategies} perturb_rates={perturb_rates} perturb_seeds={args.perturb_seeds} "
        f"workers={args.workers} max_steps={args.max_steps} total_tasks={total}",
        flush=True,
    )
    save_json(output_root / "run_config.json", {
        "started_at": utc_timestamp(),
        "tasks_total": total,
        "deviations": deviations,
        "seeds_per_deviation": args.seeds_per_deviation,
        "strategies": strategies,
        "perturb_rates": perturb_rates,
        "perturb_seeds": args.perturb_seeds,
        "max_steps": args.max_steps,
        "games": wanted,
    })

    completed = 0
    n_errors = 0
    t0 = time.time()
    last_log_t = t0
    bucket_counts: Dict[str, int] = {}

    def _fmt_secs(secs: float) -> str:
        secs = int(max(0, secs))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}h{m:02d}m{s:02d}s"
        return f"{m:d}m{s:02d}s"

    print(
        f"[collect-gt] start: total_tasks={total} workers={args.workers} "
        f"max_steps={args.max_steps}",
        flush=True,
    )
    print(
        f"[collect-gt] expected per-task wall: ~5-30s; estimated finish: "
        f"~{_fmt_secs(total * 8.0 / max(1, int(args.workers)))} on {args.workers} workers",
        flush=True,
    )

    # Per-worker init: create one Arcade per worker (reused across tasks),
    # pin ARC_PRIORS_PATH so MyAgent loads priors. Big memory win vs creating
    # an Arcade per task.
    priors_for_init = (
        str(project_root / "Local_Output" / "per_game_priors.json")
        if (project_root / "Local_Output" / "per_game_priors.json").is_file()
        else None
    )
    pool_kwargs: Dict[str, Any] = {
        "max_workers": max(1, int(args.workers)),
        "initializer": _init_worker,
        "initargs": (str(project_root), str(recordings_dir), priors_for_init),
    }
    # max_tasks_per_child=50 (Python 3.11+): worker recycled every 50 tasks
    # to bound any leaked state in arc_agi internals. 50 is gentle (vs 5 which
    # caused freezes) — it gives workers enough lifetime to amortize startup
    # but bounds the memory footprint.
    try:
        import inspect as _insp
        if "max_tasks_per_child" in _insp.signature(ProcessPoolExecutor).parameters:
            pool_kwargs["max_tasks_per_child"] = 50
    except Exception:
        pass

    with ProcessPoolExecutor(**pool_kwargs) as pool:
        futures = {
            pool.submit(
                collect_one,
                full_id, short_id, dev, seed, args.max_steps, perturb, strat,
                project_root, recordings_dir,
            ): (full_id, short_id, dev, seed, perturb, strat)
            for (full_id, short_id, dev, seed, perturb, strat) in tasks
        }
        for fut in as_completed(futures):
            full_id, short_id, dev, seed, perturb, strat = futures[fut]
            try:
                payload = fut.result()
            except Exception as exc:
                payload = {"game_id": full_id, "error": str(exc)}
            append_jsonl_gz(gz_path, payload)
            completed += 1
            tag = payload.get("source", "?")
            bucket_counts[tag] = bucket_counts.get(tag, 0) + 1
            wall = time.time() - t0
            err = payload.get("error")
            if err:
                n_errors += 1
            lvl = payload.get("levels_completed", "?")
            n_act = payload.get("actions_taken", "?")
            # Log every 50 completed OR at end OR every 30 wall-seconds OR on error.
            now = time.time()
            should_log = (
                completed % 50 == 0
                or completed == total
                or (now - last_log_t) >= 30.0
                or err
            )
            if should_log:
                last_log_t = now
                rate = completed / max(0.001, wall)
                eta_secs = (total - completed) / max(0.001, rate)
                pct = 100.0 * completed / max(1, total)
                if err:
                    print(
                        f"[collect-gt] {completed}/{total} ({pct:.1f}%) "
                        f"elapsed={_fmt_secs(wall)} eta={_fmt_secs(eta_secs)} "
                        f"rate={rate:.2f}/s errs={n_errors} :: {short_id} ERR={err}",
                        flush=True,
                    )
                else:
                    print(
                        f"[collect-gt] {completed}/{total} ({pct:.1f}%) "
                        f"elapsed={_fmt_secs(wall)} eta={_fmt_secs(eta_secs)} "
                        f"rate={rate:.2f}/s errs={n_errors} buckets={dict(bucket_counts)} "
                        f":: {short_id} dev={int(dev*100)}% seed={seed} "
                        f"perturb={perturb:.2f} strat={strat} levels={lvl} "
                        f"actions={n_act} tag={tag}",
                        flush=True,
                    )

    save_json(output_root / "summary.json", {
        "tasks_total": total,
        "tasks_completed": completed,
        "wall_seconds": time.time() - t0,
        "output_gz": str(gz_path),
        "bucket_counts": bucket_counts,
    })
    print(f"[collect-gt] done. completed={completed}/{total} buckets={bucket_counts} wall={time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
