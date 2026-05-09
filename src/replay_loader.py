"""Load and process ground-truth replay JSONL files into agent priors.

Each replay file under `environment_files/<game_id>/replays/*.json` is a JSONL
log of a complete winning playthrough. Records have shape:

    {
      "timestamp": "...",
      "data": {
        "game_id": str,
        "frame": [[[ row x 64 ints, ... ] x 64 ]],  # outer list of 1
        "state": "NOT_FINISHED" | "WIN" | ...,
        "levels_completed": int,
        "win_levels": int,
        "action_input": {"id": int|str, "data": {x?, y?, game_id}, "reasoning": None},
        "guid": str,
        "full_reset": bool,
        "available_actions": [int, ...]
      }
    }

We extract three artifacts from each replay:

  1. `action_scores`      — empirical action distribution → search prior weights
  2. `coord_hint_points`  — every (x, y) used by ACTION6 → click prior locations
  3. `state_graph_seed`   — observed (frame_hash, action) → next_frame_hash edges
                            for pre-populating PolicyGuidedAgent.state_graph

Use `build_game_prior(project_root, game_id)` to assemble all three at once.
The result is a dict that can be passed directly to
`PolicyGuidedAgent(game_prior=...)`.

Hidden-set caveat: these priors are GAME-SPECIFIC. They help the search agent
on the public games we have replays for, but do NOT directly benefit hidden
test games. Their value for the hidden set is indirect — by improving collect
quality on the public games, we get richer training data for downstream BC /
TTT components which DO have a chance at transfer.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import frame_hash

# Some replays (e.g., cn04's session) record action ids as strings like
# "ACTION1" instead of integers. Normalize both formats to int ids.
_ACTION_NAME_TO_ID: Dict[str, int] = {
    "ACTION1": 1,
    "ACTION2": 2,
    "ACTION3": 3,
    "ACTION4": 4,
    "ACTION5": 5,
    "ACTION6": 6,
    "ACTION7": 7,
    "RESET": 0,  # 0 = filter out
}


def _normalize_action_id(raw: Any) -> Optional[int]:
    """Convert action_input.id from raw record (int or 'ACTIONn'/'RESET') to int.

    Returns None for RESET (action_id 0) and unknown values; otherwise the int
    1..7 that callers can compare against `ACTION_IDS`.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):  # bool is subclass of int — guard explicitly
        return None
    if isinstance(raw, int):
        return raw if 1 <= raw <= 7 else None
    if isinstance(raw, str):
        result = _ACTION_NAME_TO_ID.get(raw.strip().upper())
        if result is None or result <= 0:
            return None
        return result
    return None


def _decode_frame(frame_field: Any) -> List[List[int]]:
    """The replay's `frame` field is `[[[ row x 64 ints ] x 64 ]]` — outer
    list of 1 wrapping the 64x64 grid. Decode to a clean `List[List[int]]`.
    Returns a 64x64 zero grid on any malformed input rather than raising.
    """
    if not isinstance(frame_field, list) or not frame_field:
        return [[0] * 64 for _ in range(64)]
    inner = frame_field[0]
    # The replay format has one extra wrapping list; check for it.
    if isinstance(inner, list) and inner and isinstance(inner[0], list):
        return [[int(c) for c in row] for row in inner]
    # Fallback: maybe frame_field IS already the 64-row grid.
    if isinstance(frame_field[0], list) and frame_field[0] and not isinstance(frame_field[0][0], list):
        return [[int(c) for c in row] for row in frame_field]
    return [[0] * 64 for _ in range(64)]


def replay_path(project_root: Path, game_id: str) -> Optional[Path]:
    """Locate the replay JSONL for a given game_id under
    environment_files/<game_id>/replays/. Returns the first matching .json
    (the GT distribution shipped one replay per game) or None if none found.
    """
    base = Path(project_root) / "environment_files" / game_id / "replays"
    if not base.is_dir():
        return None
    files = sorted(base.glob("*.json"))
    return files[0] if files else None


def load_replay(path: Path) -> List[Dict[str, Any]]:
    """Decode a JSONL replay file into a list of records.

    Skips malformed lines silently — replays are external data and we don't
    want a single bad line to kill priors construction.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_action_distribution(records: List[Dict[str, Any]]) -> Dict[int, float]:
    """Empirical action distribution from the replay's non-RESET actions.

    Returns: dict mapping action_id (int in 1..7) → probability in [0, 1].
    Sums to 1.0 (or empty dict if no recognizable actions).
    """
    counts: Counter = Counter()
    for r in records:
        ai = r.get("data", {}).get("action_input")
        if not isinstance(ai, dict):
            continue
        aid = _normalize_action_id(ai.get("id"))
        if aid is None:
            continue
        counts[aid] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {aid: float(count) / float(total) for aid, count in counts.items()}


def extract_coord_hints(
    records: List[Dict[str, Any]],
    dedup: bool = True,
) -> List[Tuple[int, int]]:
    """All `(x, y)` coords used by ACTION6 in the replay. Order-preserving.

    These are by-construction successful click sites — every coord here was
    chosen by a human/agent who completed the game. Use as `coord_hint_points`
    in `PolicyGuidedAgent.game_prior`.

    `dedup=True` (default) keeps one entry per unique (x, y); set False to
    keep duplicates (useful if you want to weight by frequency downstream).
    """
    seen: set = set() if dedup else set()
    hints: List[Tuple[int, int]] = []
    for r in records:
        ai = r.get("data", {}).get("action_input")
        if not isinstance(ai, dict):
            continue
        if _normalize_action_id(ai.get("id")) != 6:
            continue
        ad = ai.get("data") or {}
        x_raw = ad.get("x")
        y_raw = ad.get("y")
        if x_raw is None or y_raw is None:
            continue
        try:
            x, y = int(x_raw), int(y_raw)
        except (TypeError, ValueError):
            continue
        if not (0 <= x < 64 and 0 <= y < 64):
            continue
        key = (x, y)
        if dedup:
            if key in seen:
                continue
            seen.add(key)
        hints.append(key)
    return hints


def extract_state_graph_seed(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[Tuple[int, int, int], str]]:
    """Walk the replay and emit `(frame_hash, action) -> next_frame_hash`
    edges, matching the schema used by `PolicyGuidedAgent.state_graph`.

    For ACTION6 the action key is `(6, x, y)`; for simple actions it is
    `(action_id, 0, 0)`. Returns a dict with the same shape as the agent's
    runtime graph so it can be merged in directly.
    """
    graph: Dict[str, Dict[Tuple[int, int, int], str]] = {}
    pending_action: Optional[Tuple[int, int, int]] = None
    pending_frame_hash: Optional[str] = None

    for r in records:
        d = r.get("data", {})
        cur_frame = _decode_frame(d.get("frame"))
        cur_hash = frame_hash(cur_frame)

        if pending_action is not None and pending_frame_hash is not None:
            graph.setdefault(pending_frame_hash, {})[pending_action] = cur_hash

        ai = d.get("action_input")
        next_action: Optional[Tuple[int, int, int]] = None
        if isinstance(ai, dict):
            aid = _normalize_action_id(ai.get("id"))
            if aid is not None:
                if aid == 6:
                    ad = ai.get("data") or {}
                    try:
                        x, y = int(ad.get("x", 0)), int(ad.get("y", 0))
                    except (TypeError, ValueError):
                        x, y = 0, 0
                    if 0 <= x < 64 and 0 <= y < 64:
                        next_action = (aid, x, y)
                else:
                    next_action = (aid, 0, 0)

        pending_action = next_action
        pending_frame_hash = cur_hash

    return graph


def build_game_prior(
    project_root: Path,
    game_id: str,
    include_state_graph_seed: bool = True,
) -> Optional[Dict[str, Any]]:
    """Assemble the full `game_prior` dict for `PolicyGuidedAgent` from a
    game's replay file. Returns None if no replay is found for the game.

    Caller can pass the result directly to `PolicyGuidedAgent(game_prior=...)`.
    The agent already consumes `action_scores` and `coord_hint_points`; the
    `state_graph_seed` key is read by a small `__init__` extension (see
    agent.py 2026-05-08 changelog) to pre-populate the runtime state graph.
    """
    path = replay_path(project_root, game_id)
    if path is None:
        return None
    records = load_replay(path)
    if not records:
        return None

    prior: Dict[str, Any] = {
        "action_scores": extract_action_distribution(records),
        "coord_hint_points": extract_coord_hints(records, dedup=True),
    }
    if include_state_graph_seed:
        prior["state_graph_seed"] = extract_state_graph_seed(records)
    return prior


def replay_summary(project_root: Path, game_id: str) -> Dict[str, Any]:
    """Quick stats for a game's replay — useful for debugging and reporting."""
    path = replay_path(project_root, game_id)
    if path is None:
        return {"game_id": game_id, "found": False}
    records = load_replay(path)
    max_lvl = 0
    win_steps = 0
    a6_unique = set()
    aid_counter: Counter = Counter()
    for r in records:
        d = r.get("data", {})
        max_lvl = max(max_lvl, int(d.get("levels_completed") or 0))
        if str(d.get("state")) == "WIN":
            win_steps += 1
        ai = d.get("action_input")
        if isinstance(ai, dict):
            aid = _normalize_action_id(ai.get("id"))
            if aid is not None:
                aid_counter[aid] += 1
                if aid == 6:
                    ad = ai.get("data") or {}
                    try:
                        a6_unique.add((int(ad.get("x", -1)), int(ad.get("y", -1))))
                    except (TypeError, ValueError):
                        pass
    return {
        "game_id": game_id,
        "found": True,
        "path": str(path),
        "n_records": len(records),
        "max_levels_completed": max_lvl,
        "win_steps": win_steps,
        "unique_action6_coords": len(a6_unique),
        "action_id_counts": dict(aid_counter),
    }
