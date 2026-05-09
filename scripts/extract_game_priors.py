"""Extract per-game behavioral priors from replays_v2.gz.

For each public game, distills the human's playstyle into a structured prior
the agent can use at runtime to pick game-appropriate strategies. Unlike
`replay_loader.build_game_prior` (which only emits action_scores +
coord_hint_points), this captures sequence/timing patterns:

- opening_actions: first ~10 (action_id, x, y) across winning attempts
- action_type_ratio: fraction of actions that are clicks (ACTION6)
- click_heatmap: 8x8 binned heatmap of click coords (downsampled from 64x64)
- click_hot_spots: top-K (x, y) coords with highest click frequency
- death_signatures: last 3 actions before each GAME_OVER (what to avoid)
- post_reset_actions: first 3 actions after each RESET (what humans retry)
- per_level_action_dist: per-level distribution of action_ids
- archetype: heuristic label — keyboard_dominant / click_dominant / mixed
- repeat_kept_actions: list of actions humans repeat consecutively (signals
  sustained-direction games like ls20)

Output: Local_Output/per_game_priors.json (one entry per public game).

Usage:
    python scripts/extract_game_priors.py
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


REPLAYS_GZ = ROOT / "Local_Output" / "Collection_Cache" / "replays_v2" / "collected" / "episodes.jsonl.gz"
OUT_PATH = ROOT / "Local_Output" / "per_game_priors.json"

GRID = 64
HEATMAP_BINS = 8        # 8x8 spatial bins (each bin = 8x8 pixels)
TOP_HOT_SPOTS = 8
OPENING_LEN = 10
REPEAT_THRESHOLD = 3    # consecutive same-action runs of length >= this count


def _bin(x: int) -> int:
    return min(HEATMAP_BINS - 1, max(0, x * HEATMAP_BINS // GRID))


def _archetype(action_type_ratio: float, n_distinct_keys: int) -> str:
    if action_type_ratio >= 0.7:
        return "click_dominant"
    if action_type_ratio <= 0.15:
        return "keyboard_dominant"
    return "mixed"


def _consecutive_runs(action_ids: List[int]) -> Counter:
    """Count consecutive runs of length >= REPEAT_THRESHOLD per action_id."""
    runs: Counter = Counter()
    if not action_ids:
        return runs
    cur = action_ids[0]
    cur_len = 1
    for aid in action_ids[1:]:
        if aid == cur:
            cur_len += 1
        else:
            if cur_len >= REPEAT_THRESHOLD:
                runs[cur] += 1
            cur = aid
            cur_len = 1
    if cur_len >= REPEAT_THRESHOLD:
        runs[cur] += 1
    return runs


def extract_per_game() -> Dict[str, Dict[str, Any]]:
    """Walk replays_v2 episodes, group by game, extract per-game priors."""
    by_game: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with gzip.open(REPLAYS_GZ, "rt") as fh:
        for line in fh:
            ep = json.loads(line)
            by_game[ep["game_id"]].append(ep)

    out: Dict[str, Dict[str, Any]] = {}
    for game, episodes in by_game.items():
        # Sort by attempt_index so opening = first attempt, death/recovery uses
        # the actual chronology.
        episodes = sorted(episodes, key=lambda e: int(e.get("attempt_index", 0)))
        winning = [e for e in episodes if e.get("final_state") == "WIN"]
        failed = [e for e in episodes if e.get("final_state") == "GAME_OVER"]

        opening_actions: List[Dict[str, int]] = []
        action_id_counter: Counter = Counter()
        click_count = 0
        non_click_count = 0
        click_heatmap = [[0] * HEATMAP_BINS for _ in range(HEATMAP_BINS)]
        click_coord_count: Counter = Counter()
        death_signatures: List[List[Dict[str, int]]] = []
        post_reset_actions: List[List[Dict[str, int]]] = []
        per_level_actions: Dict[int, Counter] = defaultdict(Counter)
        all_action_ids: List[int] = []

        # Opening actions: from the first winning attempt (or first attempt
        # overall if none are flagged WIN).
        opening_source = winning[0] if winning else (episodes[0] if episodes else None)
        if opening_source is not None:
            for tr in opening_source.get("transitions", [])[:OPENING_LEN]:
                ad = tr.get("action_data") or {}
                opening_actions.append({
                    "action_id": int(tr["action_id"]),
                    "x": int(ad.get("x", 0)) if int(tr["action_id"]) == 6 else 0,
                    "y": int(ad.get("y", 0)) if int(tr["action_id"]) == 6 else 0,
                })

        # Aggregate across ALL episodes (winning + failed).
        for ep in episodes:
            transitions = ep.get("transitions", [])
            if not transitions:
                continue
            # post_reset_actions: episodes with attempt_index > 0 are by
            # definition post-reset; record their first 3 actions.
            if int(ep.get("attempt_index", 0)) > 0:
                first_n = transitions[:3]
                post_reset_actions.append([
                    {
                        "action_id": int(t["action_id"]),
                        "x": int((t.get("action_data") or {}).get("x", 0)),
                        "y": int((t.get("action_data") or {}).get("y", 0)),
                    }
                    for t in first_n
                ])
            # death_signatures: failed episodes contributing last 3 actions.
            if ep.get("final_state") == "GAME_OVER" and len(transitions) >= 1:
                last_n = transitions[-3:]
                death_signatures.append([
                    {
                        "action_id": int(t["action_id"]),
                        "x": int((t.get("action_data") or {}).get("x", 0)),
                        "y": int((t.get("action_data") or {}).get("y", 0)),
                    }
                    for t in last_n
                ])

            for tr in transitions:
                aid = int(tr["action_id"])
                action_id_counter[aid] += 1
                all_action_ids.append(aid)
                if aid == 6:
                    click_count += 1
                    ad = tr.get("action_data") or {}
                    x = int(ad.get("x", 0))
                    y = int(ad.get("y", 0))
                    click_heatmap[_bin(y)][_bin(x)] += 1
                    click_coord_count[(x, y)] += 1
                else:
                    non_click_count += 1

                lvl = int(tr.get("levels_before", 0))
                per_level_actions[lvl][aid] += 1

        total_actions = max(1, click_count + non_click_count)
        click_ratio = click_count / total_actions
        n_distinct_keys = sum(1 for aid in action_id_counter if aid != 6)
        arche = _archetype(click_ratio, n_distinct_keys)

        # Top click hotspots (raw 64x64 coords, before binning).
        click_hot_spots = [
            {"x": int(x), "y": int(y), "count": int(c)}
            for (x, y), c in click_coord_count.most_common(TOP_HOT_SPOTS)
        ]

        repeat_runs = _consecutive_runs(all_action_ids)

        # First-frame color histogram + available_actions for transfer-learning
        # on hidden games. Use the FIRST winning attempt's first transition; if
        # none, use the first episode's first transition.
        first_frame: List[List[int]] = []
        available_actions_seen: List[int] = []
        first_source = winning[0] if winning else (episodes[0] if episodes else None)
        if first_source is not None:
            transitions = first_source.get("transitions", [])
            if transitions:
                first_frame = transitions[0].get("frame") or []
                available_actions_seen = list(transitions[0].get("available_actions", []))

        # 16-dim normalized color histogram of the first frame.
        color_hist = [0.0] * 16
        if first_frame:
            for row in first_frame:
                for c in row:
                    ci = int(c)
                    if 0 <= ci < 16:
                        color_hist[ci] += 1.0
            total = sum(color_hist)
            if total > 0:
                color_hist = [round(v / total, 6) for v in color_hist]

        # Union of available_actions across all transitions in all episodes.
        union_available: set = set()
        for ep in episodes:
            for tr in ep.get("transitions", []):
                for a in tr.get("available_actions", []):
                    try:
                        union_available.add(int(a))
                    except Exception:
                        continue

        out[game] = {
            "n_attempts": len(episodes),
            "n_winning_attempts": len(winning),
            "n_failed_attempts": len(failed),
            "total_actions_recorded": int(total_actions),
            "opening_actions": opening_actions,
            "action_id_counts": dict(action_id_counter),
            "click_ratio": round(click_ratio, 3),
            "click_heatmap_8x8": click_heatmap,
            "click_hot_spots": click_hot_spots,
            "death_signatures": death_signatures,
            "post_reset_actions": post_reset_actions,
            "per_level_action_dist": {
                str(lvl): dict(c) for lvl, c in sorted(per_level_actions.items())
            },
            "archetype": arche,
            "repeat_kept_actions": dict(repeat_runs),
            # Transfer-learning fields:
            "first_frame_color_hist": color_hist,            # 16-dim normalized
            "available_actions_union": sorted(union_available),
        }
    return out


def _summarize(prior: Dict[str, Any]) -> str:
    bits = [
        f"arche={prior['archetype']}",
        f"click={prior['click_ratio']}",
        f"attempts={prior['n_attempts']} (W{prior['n_winning_attempts']}/F{prior['n_failed_attempts']})",
        f"total={prior['total_actions_recorded']}",
    ]
    if prior["repeat_kept_actions"]:
        bits.append(f"repeats={prior['repeat_kept_actions']}")
    if prior["click_hot_spots"]:
        top = prior["click_hot_spots"][0]
        bits.append(f"top_click=({top['x']},{top['y']})x{top['count']}")
    return " ".join(bits)


def main() -> int:
    if not REPLAYS_GZ.exists():
        print(f"Missing {REPLAYS_GZ}. Run scripts/convert_replays_to_episodes.py first.", file=sys.stderr)
        return 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    priors = extract_per_game()
    OUT_PATH.write_text(json.dumps(priors, indent=2))
    print(f"wrote {OUT_PATH}  ({len(priors)} games, {OUT_PATH.stat().st_size / 1024:.1f} KB)")
    print()
    for game in sorted(priors):
        print(f"  {game}: {_summarize(priors[game])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
