"""Validate the hypothesis: "humans click on rare-colored regions first."

For each public game with a GT replay:
  1. Take the first frame the human saw.
  2. Compute rare-color centroids (color count 1-200, skip color 0).
  3. Find the first N=5 ACTION6 clicks the human made.
  4. For each click, measure distance to the NEAREST rare-color centroid.
  5. Also measure: is the clicked pixel ITSELF a rare color?

Reports:
  - Per-game: % of first-N clicks within 8 pixels of a rare centroid
  - Per-game: % of first-N clicks whose pixel value is a rare color
  - Aggregate: hit-rate across all click-bearing games

If hit-rate is high (≥ 70%), the saliency hypothesis is confirmed and we
should ship rare-color centroids as click_grid candidates. If low, we need
a different visual feature (edges, symmetry, etc).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.replay_loader import _decode_frame, _normalize_action_id, load_replay, replay_path  # noqa: E402


def _rare_color_centroids(frame: List[List[int]],
                          min_count: int = 1, max_count: int = 200) -> List[Tuple[int, int, int, int]]:
    """Returns list of (color, centroid_x, centroid_y, count) tuples for rare colors.

    Rare = color appears in [min_count, max_count] pixels, excluding color 0.
    """
    color_counts: Dict[int, int] = {}
    color_pixels: Dict[int, List[Tuple[int, int]]] = {}
    for y, row in enumerate(frame):
        for x, c in enumerate(row):
            ci = int(c)
            color_counts[ci] = color_counts.get(ci, 0) + 1
            color_pixels.setdefault(ci, []).append((x, y))

    out: List[Tuple[int, int, int, int]] = []
    for c, cnt in color_counts.items():
        if c == 0:
            continue
        if min_count <= cnt <= max_count:
            pts = color_pixels[c]
            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            out.append((c, cx, cy, cnt))
    return sorted(out, key=lambda t: t[3])  # rarest first


def _expanded_saliency_points(frame: List[List[int]]) -> List[Tuple[int, int]]:
    """Mirrors `kaggle_notebook/agents/my_agent.py::_extract_saliency`: rare-color
    centroid + 4 bbox corners + 4 directional offsets per rare color.
    Returns deduplicated list, capped at 30."""
    color_counts: Dict[int, int] = {}
    color_pixels: Dict[int, List[Tuple[int, int]]] = {}
    for y, row in enumerate(frame):
        for x, c in enumerate(row):
            ci = int(c)
            color_counts[ci] = color_counts.get(ci, 0) + 1
            color_pixels.setdefault(ci, []).append((x, y))

    rare_colors = sorted(
        (c for c, cnt in color_counts.items() if c != 0 and 1 <= cnt <= 200),
        key=lambda c: color_counts[c],
    )
    salient: List[Tuple[int, int]] = []
    for color in rare_colors[:6]:
        pts = color_pixels[color]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if not xs:
            continue
        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))
        salient.append((cx, cy))
        if len(pts) >= 3:
            x_lo, x_hi = min(xs), max(xs)
            y_lo, y_hi = min(ys), max(ys)
            salient.extend([(x_lo, y_lo), (x_lo, y_hi), (x_hi, y_lo), (x_hi, y_hi)])
        if len(pts) >= 5:
            for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4)):
                salient.append((max(0, min(63, cx + dx)), max(0, min(63, cy + dy))))
    seen: set = set()
    out: List[Tuple[int, int]] = []
    for p in salient:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:30]


def _first_n_action6_clicks(records: List[Dict[str, Any]], n: int = 5) -> List[Tuple[int, int, int]]:
    """Returns list of (frame_index, x, y) for the first n ACTION6 clicks.

    frame_index = which record index the click action was recorded at — useful
    if we want to use the corresponding frame_at_click_time later.
    """
    clicks: List[Tuple[int, int, int]] = []
    for i, rec in enumerate(records):
        ai = (rec.get("data") or {}).get("action_input") or {}
        aid = _normalize_action_id(ai.get("id"))
        if aid != 6:
            continue
        ad = ai.get("data") or {}
        try:
            x = int(ad.get("x", 0))
            y = int(ad.get("y", 0))
        except Exception:
            continue
        clicks.append((i, x, y))
        if len(clicks) >= n:
            break
    return clicks


def _min_dist_to_centroid(x: int, y: int, centroids: List[Tuple[int, int, int, int]]) -> int:
    if not centroids:
        return 9999
    return min(abs(x - cx) + abs(y - cy) for _, cx, cy, _ in centroids)


def main() -> int:
    games_with_clicks: List[str] = []
    games_no_clicks: List[str] = []

    per_game: Dict[str, Dict[str, Any]] = {}

    env_root = ROOT / "environment_files"
    for game_dir in sorted(env_root.iterdir()):
        if not game_dir.is_dir() or game_dir.name == "replays":
            continue
        game = game_dir.name
        path = replay_path(ROOT, game)
        if path is None:
            continue
        records = load_replay(path)
        if len(records) < 2:
            continue

        # First frame: records[0].data.frame
        first_frame = _decode_frame(records[0].get("data", {}).get("frame"))
        rare_centroids = _rare_color_centroids(first_frame)
        # Set of rare colors for the second test (is the clicked pixel a rare color?)
        rare_color_set = {c for c, _, _, _ in rare_centroids}

        clicks = _first_n_action6_clicks(records, n=5)
        if not clicks:
            games_no_clicks.append(game)
            per_game[game] = {
                "n_rare_colors": len(rare_centroids),
                "n_clicks_examined": 0,
                "skipped": "no ACTION6 in replay",
            }
            continue

        games_with_clicks.append(game)

        # Expanded saliency pool (mirrors my_agent.py::_extract_saliency):
        # centroid + bbox corners + offsets per rare color.
        expanded_pool = _expanded_saliency_points(first_frame)
        # For each click, distance to nearest rare-color centroid + whether
        # the clicked pixel itself is a rare color.
        click_dists_centroid: List[int] = []
        click_dists_expanded: List[int] = []
        click_on_rare: List[bool] = []
        for fi, x, y in clicks:
            d_centroid = _min_dist_to_centroid(x, y, rare_centroids)
            d_expanded = (
                min(abs(x - px) + abs(y - py) for px, py in expanded_pool)
                if expanded_pool else 9999
            )
            click_dists_centroid.append(d_centroid)
            click_dists_expanded.append(d_expanded)
            try:
                clicked_color = int(first_frame[y][x])
                click_on_rare.append(clicked_color in rare_color_set)
            except Exception:
                click_on_rare.append(False)

        n_within_8_centroid = sum(1 for d in click_dists_centroid if d <= 8)
        n_within_16_centroid = sum(1 for d in click_dists_centroid if d <= 16)
        n_within_8_expanded = sum(1 for d in click_dists_expanded if d <= 8)
        n_within_16_expanded = sum(1 for d in click_dists_expanded if d <= 16)
        n_on_rare = sum(1 for v in click_on_rare if v)
        n = len(clicks)
        per_game[game] = {
            "n_rare_colors": len(rare_centroids),
            "n_expanded_pool": len(expanded_pool),
            "n_clicks_examined": n,
            "click_dists_to_nearest_centroid": click_dists_centroid,
            "click_dists_to_nearest_expanded": click_dists_expanded,
            "click_on_rare_color_pixel": click_on_rare,
            "pct_within_8_centroid": round(100 * n_within_8_centroid / n, 1),
            "pct_within_16_centroid": round(100 * n_within_16_centroid / n, 1),
            "pct_within_8_expanded": round(100 * n_within_8_expanded / n, 1),
            "pct_within_16_expanded": round(100 * n_within_16_expanded / n, 1),
            "pct_on_rare_color_pixel": round(100 * n_on_rare / n, 1),
        }

    # Aggregate
    eligible = [g for g in games_with_clicks if per_game[g]["n_rare_colors"] > 0]
    total_clicks = sum(per_game[g]["n_clicks_examined"] for g in eligible)
    def _agg(key):
        return sum(int(per_game[g][key] * per_game[g]["n_clicks_examined"] / 100) for g in eligible)
    total_within_8c = _agg("pct_within_8_centroid")
    total_within_16c = _agg("pct_within_16_centroid")
    total_within_8e = _agg("pct_within_8_expanded")
    total_within_16e = _agg("pct_within_16_expanded")
    total_on_rare = _agg("pct_on_rare_color_pixel")

    print(f"games with clicks: {len(games_with_clicks)} / 25")
    print(f"games skipped (no ACTION6 in replay): {len(games_no_clicks)} ({games_no_clicks})")
    print(f"games with at least one rare-color centroid: {len(eligible)}")
    print()
    print(f'{"game":<5} | {"#rare":>5} | {"#pool":>5} | {"clicks":>6} | {"c<=8":>5} | {"c<=16":>6} | {"e<=8":>5} | {"e<=16":>6} | {"onR":>5}')
    print('-' * 90)
    for g in sorted(games_with_clicks):
        p = per_game[g]
        if not p.get("n_clicks_examined"):
            continue
        print(f'{g:<5} | {p["n_rare_colors"]:>5} | {p.get("n_expanded_pool", 0):>5} | {p["n_clicks_examined"]:>6} | '
              f'{p["pct_within_8_centroid"]:>4}% | {p["pct_within_16_centroid"]:>5}% | '
              f'{p["pct_within_8_expanded"]:>4}% | {p["pct_within_16_expanded"]:>5}% | '
              f'{p["pct_on_rare_color_pixel"]:>4}%')
    print()
    print('=== AGGREGATE ===')
    if total_clicks > 0:
        print(f'Total clicks across {len(eligible)} games: {total_clicks}')
        print(f'  centroid only — within 8:   {100*total_within_8c/total_clicks:.1f}%')
        print(f'  centroid only — within 16:  {100*total_within_16c/total_clicks:.1f}%')
        print(f'  EXPANDED pool — within 8:   {100*total_within_8e/total_clicks:.1f}%  ← new')
        print(f'  EXPANDED pool — within 16:  {100*total_within_16e/total_clicks:.1f}%  ← new')
        print(f'  on a rare-color pixel:      {100*total_on_rare/total_clicks:.1f}%')

    out_path = ROOT / "Local_Output" / "saliency_hypothesis_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(per_game, indent=2))
    print()
    print(f'wrote {out_path}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
