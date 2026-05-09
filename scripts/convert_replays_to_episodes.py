#!/usr/bin/env python3
"""Convert ground-truth replays into the project's `.gz` episode format.

Reads each replay JSONL under `environment_files/<game_id>/replays/*.json`,
slices the trajectory at RESET boundaries (each attempt becomes one episode),
and writes the result to:

    Local_Output/Collection_Cache/replays_v2/collected/episodes.jsonl.gz

**Per-attempt slicing (v2):** unlike v1 which emitted one episode per cleared
level, v2 preserves the full multi-attempt exploration arc. Each attempt is a
continuous trajectory between human RESET events (action_id_raw=0 while
state=GAME_OVER). Failed attempts (final_state=GAME_OVER) AND winning attempts
(final_state=WIN or NOT_FINISHED-but-progressed) are both emitted, with an
`attempt_index` field for downstream "different strategies per attempt"
training.

This rewrite addresses a v1 bug: the previous filter dropped every record
with action_id=None, silently throwing away the GAME_OVER → RESET → retry
pattern. Result: the worldmodel never saw the human-exploration signal that
the replays were specifically chosen to capture. Verified on ar25's v1
output: 0 GAME_OVER transitions out of 577, despite the recording containing
deaths and resets.

Output schema matches the existing collector's episode format so downstream
training (`src/train_worldmodel.py`, `src/train.py`) can consume it directly.
Each episode additionally carries `attempt_index` (0, 1, 2, ...) and
`source = "gt_replay_v2"`.

Usage:
    python -m scripts.convert_replays_to_episodes
    python scripts/convert_replays_to_episodes.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

# Ensure src/ is importable when this script is run from project root via either
#   `python -m scripts.convert_replays_to_episodes`
# or
#   `python scripts/convert_replays_to_episodes.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import frame_delta  # noqa: E402
from src.replay_loader import (  # noqa: E402
    _decode_frame,
    _normalize_action_id,
    load_replay,
    replay_path,
)


OUTPUT_DIR = PROJECT_ROOT / "Local_Output" / "Collection_Cache" / "replays_v2" / "collected"
OUTPUT_FILE = OUTPUT_DIR / "episodes.jsonl.gz"


def _emit_attempt(
    game_id: str,
    attempt_index: int,
    transitions: List[Dict[str, Any]],
    final_state: str,
    levels_completed: int,
    baseline_actions: List[int],
) -> Dict[str, Any]:
    """Build one episode for a single human attempt (continuous trajectory
    between RESETs). Either a winning attempt or a failed (GAME_OVER) one.

    Score is rhae-style on the levels actually cleared this attempt; a failed
    attempt (levels_completed == 0) gets score=0. The agent benefits more from
    seeing the failure trajectory than from any score signal — score is only
    used downstream for replay weighting.

    Metadata-gap fallback: 7 games (ar25, cd82, lp85, sb26, sc25, tu93, vc33)
    record `levels_completed=null` throughout despite ending in WIN. Without
    intervention, those winning trajectories arrive at the trainer with
    `levels=0` and pick up the dead-episode `weight_mult=0.4`, effectively
    discarding ~5K winning transitions. When `final_state == "WIN"` and
    computed levels_completed is 0, fall back to `len(baseline_actions)` (the
    full level count for that game) so the trainer up-weights it correctly.
    """
    actions_taken = len(transitions)
    if str(final_state) == "WIN" and int(levels_completed) == 0:
        levels_completed = max(1, len(baseline_actions))
    if levels_completed > 0 and len(baseline_actions) >= levels_completed:
        baseline_sum = sum(int(b) for b in baseline_actions[:levels_completed])
        score = min(
            ((float(baseline_sum) / float(max(1, actions_taken))) ** 2) * 100.0,
            115.0,
        )
    else:
        score = 0.0
    return {
        "game_id": game_id,
        "score": float(score),
        "levels_completed": int(levels_completed),
        "actions_taken": int(actions_taken),
        "final_state": str(final_state),
        "transitions": transitions,
        "source": "gt_replay_v2",
        "attempt_index": int(attempt_index),
    }


def replay_to_episodes(
    records: List[Dict[str, Any]],
    game_id: str,
    baseline_actions: List[int],
) -> Iterator[Dict[str, Any]]:
    """Slice a single replay into per-attempt episodes (v2).

    An "attempt" is a continuous trajectory between human RESET events. Each
    is emitted as one episode. Failed attempts (final_state=GAME_OVER,
    typically levels_completed=0) sit alongside winning ones — both are
    informative for "human exploration" training. attempt_index increments
    across resets within a game.

    A RESET is detected when a record's action_input.id is null/0 AND the
    PRECEDING transition produced state=GAME_OVER. The pre-reset record's
    own transitions stay in the previous attempt; the new attempt begins
    on the post-reset frame.

    Records where action_input.id is null/0 outside of GAME_OVER recovery
    (e.g., the bootstrap record at index 0) are not emitted as transitions
    but their frame seeds the first attempt's history.
    """
    if len(records) < 2:
        return

    attempt_transitions: List[Dict[str, Any]] = []
    attempt_index = 0
    step_index_within_attempt = 0
    attempt_initial_levels = 0  # levels_completed at the start of this attempt

    def _attempt_levels_cleared() -> int:
        if not attempt_transitions:
            return 0
        last = attempt_transitions[-1]
        return max(0, int(last.get("levels_after", 0)) - attempt_initial_levels)

    # Iterate transitions (s_t, a, s_{t+1}) where a is the action_input from
    # records[i+1] — the action that PRODUCED frame_{t+1}. action_input on
    # records[i+1] being None marks either bootstrap (i==0) or a RESET event.
    for i in range(len(records) - 1):
        d_t = records[i].get("data", {})
        d_tp1 = records[i + 1].get("data", {})

        ai = d_tp1.get("action_input") or {}
        action_id = _normalize_action_id(ai.get("id"))

        if action_id is None:
            # RESET (or bootstrap). If the previous transition ended in
            # GAME_OVER, close the current attempt and start a new one
            # anchored on records[i+1].frame (the post-reset frame).
            state_at_t = str(d_t.get("state") or "")
            if state_at_t == "GAME_OVER" and attempt_transitions:
                yield _emit_attempt(
                    game_id=game_id,
                    attempt_index=attempt_index,
                    transitions=attempt_transitions,
                    final_state="GAME_OVER",
                    levels_completed=_attempt_levels_cleared(),
                    baseline_actions=baseline_actions,
                )
                attempt_index += 1
                attempt_transitions = []
                step_index_within_attempt = 0
                attempt_initial_levels = int(d_tp1.get("levels_completed") or 0)
            continue

        if not (1 <= action_id <= 7):
            continue

        frame_t = _decode_frame(d_t.get("frame"))
        frame_tp1 = _decode_frame(d_tp1.get("frame"))

        action_data_raw = ai.get("data") or {}
        x = int(action_data_raw.get("x", 0)) if action_id == 6 else 0
        y = int(action_data_raw.get("y", 0)) if action_id == 6 else 0

        levels_before = int(d_t.get("levels_completed") or 0)
        levels_after = int(d_tp1.get("levels_completed") or 0)
        state_before = str(d_t.get("state") or "")
        state_after = str(d_tp1.get("state") or "")
        available_actions = list(d_t.get("available_actions", []))

        # On the first usable transition of an attempt, anchor the initial
        # level count so per-attempt scoring counts only progress THIS attempt.
        if not attempt_transitions:
            attempt_initial_levels = levels_before

        transition = {
            "frame": frame_t,
            "available_actions": available_actions,
            "action_id": int(action_id),
            "action_data": {"x": x, "y": y} if action_id == 6 else {},
            "next_frame": frame_tp1,
            "levels_before": levels_before,
            "levels_after": levels_after,
            "state_before": state_before,
            "state_after": state_after,
            "delta_pixels": frame_delta(frame_t, frame_tp1),
            "novelty": 0.0,
            "step_index": step_index_within_attempt,
        }
        attempt_transitions.append(transition)
        step_index_within_attempt += 1

    if attempt_transitions:
        last_state = str(attempt_transitions[-1].get("state_after") or "NOT_FINISHED")
        yield _emit_attempt(
            game_id=game_id,
            attempt_index=attempt_index,
            transitions=attempt_transitions,
            final_state=last_state,
            levels_completed=_attempt_levels_cleared(),
            baseline_actions=baseline_actions,
        )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_transitions = 0
    games_with_replays = 0
    games_without_replays: List[str] = []
    per_game_eps: Dict[str, int] = {}
    per_game_transitions: Dict[str, int] = {}

    env_root = PROJECT_ROOT / "environment_files"
    if not env_root.is_dir():
        print(f"environment_files/ not found at {env_root}", file=sys.stderr)
        return 1

    with gzip.open(OUTPUT_FILE, "wt", encoding="utf-8") as gz:
        for game_dir in sorted(env_root.iterdir()):
            if not game_dir.is_dir():
                continue
            game_id = game_dir.name

            metadata_files = sorted(game_dir.glob("*/metadata.json"))
            if not metadata_files:
                games_without_replays.append(f"{game_id} (no metadata.json)")
                continue
            try:
                with open(metadata_files[0]) as fh:
                    meta = json.load(fh)
            except (json.JSONDecodeError, OSError):
                games_without_replays.append(f"{game_id} (metadata unreadable)")
                continue
            baseline_actions = [int(b) for b in meta.get("baseline_actions", []) if isinstance(b, (int, float))]

            path = replay_path(PROJECT_ROOT, game_id)
            if path is None:
                games_without_replays.append(f"{game_id} (no replay file)")
                continue

            records = load_replay(path)
            if not records:
                games_without_replays.append(f"{game_id} (empty replay)")
                continue
            games_with_replays += 1

            game_eps = 0
            game_trans = 0
            for episode in replay_to_episodes(records, game_id, baseline_actions):
                gz.write(json.dumps(episode))
                gz.write("\n")
                game_eps += 1
                game_trans += len(episode["transitions"])

            per_game_eps[game_id] = game_eps
            per_game_transitions[game_id] = game_trans
            total_episodes += game_eps
            total_transitions += game_trans

    print(f"Wrote output: {OUTPUT_FILE}")
    print(f"  total episodes:    {total_episodes}")
    print(f"  total transitions: {total_transitions}")
    print(f"  games with replays: {games_with_replays}/{games_with_replays + len(games_without_replays)}")
    if games_without_replays:
        print(f"  games skipped: {games_without_replays}")
    print()
    print(f"  {'game':<6} {'episodes':>8} {'transitions':>12} {'avg_actions/lvl':>16}")
    print("  " + "-" * 50)
    for game_id in sorted(per_game_eps.keys()):
        eps = per_game_eps[game_id]
        trans = per_game_transitions[game_id]
        avg = trans / eps if eps else 0.0
        print(f"  {game_id:<6} {eps:>8} {trans:>12} {avg:>16.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
