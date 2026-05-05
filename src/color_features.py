"""Color-aware feature primitives for the probe-first path.

Color is treated as part of the observation, not as game-specific instruction.
No function in this module contains a per-game rule, a color->meaning lookup,
or hard-coded "color X means Y" semantics. Every primitive is a statistic of
the current frame (or pair of frames) that the agent / triage can learn from.

Design constraints:
  * All outputs are JSON-serializable (ints / floats / lists / dicts).
  * No numpy / no torch — we already iterate 64x64 elsewhere; staying in
    plain Python keeps the collector portable and inspectable.
  * Background pixels (color 0) are still counted in summaries, but the
    "non-background" variants are exposed for callers that care.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import GRID_SIZE

BACKGROUND_COLOR = 0

# Categorical label for the "shape" of a color change between two frames.
# These are statistical descriptors, not game-specific semantics.
CHANGE_NONE = "no_change"
CHANGE_MOTION_LIKE = "motion_like_color_preserving"      # color set is unchanged
CHANGE_LOCALIZED_REPLACE = "localized_color_replacement"  # one or two colors swapped, small bbox
CHANGE_BROAD_MULTI = "broad_multi_color_change"           # many colors changed across a wide bbox

ALL_CHANGE_LABELS: Tuple[str, ...] = (
    CHANGE_NONE,
    CHANGE_MOTION_LIKE,
    CHANGE_LOCALIZED_REPLACE,
    CHANGE_BROAD_MULTI,
)


# --------------------------------------------------------------------- frame


def color_histogram(frame: Sequence[Sequence[int]]) -> Dict[int, int]:
    """Counts of each color value in the frame. Includes background."""
    counts: Dict[int, int] = {}
    for row in frame:
        for value in row:
            color = int(value)
            counts[color] = counts.get(color, 0) + 1
    return counts


def top_k_colors(
    frame: Sequence[Sequence[int]],
    k: int = 4,
    include_background: bool = False,
) -> List[Tuple[int, float]]:
    """Top-K color codes with their fractional share of the frame.

    Background (color 0) is excluded by default since most ARC frames are
    dominated by it and it carries little interaction signal.
    """
    counts = color_histogram(frame)
    if not include_background:
        counts.pop(BACKGROUND_COLOR, None)
    total = float(sum(counts.values())) or 1.0
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [(int(color), round(count / total, 4)) for color, count in ranked[: max(1, int(k))]]


def color_at(frame: Sequence[Sequence[int]], point: Tuple[int, int]) -> int:
    x, y = int(point[0]), int(point[1])
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
        return BACKGROUND_COLOR
    return int(frame[y][x])


# Context-key resolution. Used for context-conditioned color learning so the
# agent can detect "same color, different neighborhood -> different outcome"
# and split a learned hypothesis without ever consulting a per-game rule.
NON_BG_BUCKETS = 4  # discretize non_bg_share to {0, 0.25, 0.5, 0.75, 1.0}


def _bucket_share(share: float, buckets: int = NON_BG_BUCKETS) -> int:
    if share <= 0.0:
        return 0
    if share >= 1.0:
        return buckets
    return max(0, min(buckets, int(round(float(share) * buckets))))


def context_key_from_descriptor(descriptor: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    """Cheap stable context key for click regions.

    Returns (secondary_color, non_bg_bucket). secondary_color = -1 when none
    is present. The bucket discretizes density into integer steps so that
    semantically similar regions share keys.
    """
    if not descriptor:
        return (-1, 0)
    secondary = descriptor.get("secondary_color")
    secondary_int = int(secondary) if secondary is not None else -1
    bucket = _bucket_share(float(descriptor.get("non_bg_share", 0.0)))
    return (secondary_int, bucket)


def scene_context_key(frame: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """Same key shape, applied to the whole frame for non-coord actions.

    The "secondary" slot here is the second-most-common non-bg color, the
    bucket is the global non-bg density. This lets non-coord and click
    contexts live in the same memory namespace.
    """
    counts = color_histogram(frame)
    counts.pop(BACKGROUND_COLOR, None)
    total_cells = float(GRID_SIZE * GRID_SIZE)
    non_bg_share = float(sum(counts.values())) / total_cells
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    secondary = int(ranked[1][0]) if len(ranked) >= 2 else -1
    return (secondary, _bucket_share(non_bg_share))


def local_color_descriptor(
    frame: Sequence[Sequence[int]],
    point: Tuple[int, int],
    radius: int = 2,
) -> Dict[str, Any]:
    """Compact summary of the color neighbourhood around a point.

    Returns:
        center_color    color directly under the click
        dominant_color  most common non-background color in the (2r+1)x(2r+1)
                        window, or background if everything is background
        secondary_color second-most common non-background color, if any
        non_bg_share    fraction of the window that is non-background
        unique_colors   number of distinct colors in the window (incl. bg)
    """
    cx, cy = int(point[0]), int(point[1])
    radius = max(0, int(radius))
    counts: Dict[int, int] = {}
    cells = 0
    for dy in range(-radius, radius + 1):
        ny = cy + dy
        if not (0 <= ny < GRID_SIZE):
            continue
        for dx in range(-radius, radius + 1):
            nx = cx + dx
            if not (0 <= nx < GRID_SIZE):
                continue
            color = int(frame[ny][nx])
            counts[color] = counts.get(color, 0) + 1
            cells += 1
    cells = max(1, cells)
    non_bg = {c: n for c, n in counts.items() if c != BACKGROUND_COLOR}
    ranked_non_bg = sorted(non_bg.items(), key=lambda item: item[1], reverse=True)
    dominant = int(ranked_non_bg[0][0]) if ranked_non_bg else BACKGROUND_COLOR
    secondary = int(ranked_non_bg[1][0]) if len(ranked_non_bg) > 1 else None
    non_bg_share = round(sum(non_bg.values()) / float(cells), 3)
    return {
        "center_color": color_at(frame, (cx, cy)),
        "dominant_color": dominant,
        "secondary_color": secondary,
        "non_bg_share": non_bg_share,
        "unique_colors": len(counts),
        "radius": radius,
    }


# ---------------------------------------------------------------- transition


def transition_color_change(
    prev_frame: Sequence[Sequence[int]],
    next_frame: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Per-color change statistics between two frames.

    Output schema:
        delta_pixels:     int
        before_dominant:  list[(color, share)] top non-bg colors
        after_dominant:   list[(color, share)]
        per_color_delta:  dict color -> {"added": int, "removed": int}
        colors_changed:   sorted list of color codes that gained or lost cells
        change_label:     one of ALL_CHANGE_LABELS
        bbox_of_change:   [x0, y0, x1, y1] | None
    """
    rows = min(len(prev_frame), len(next_frame), GRID_SIZE)
    per_color: Dict[int, Dict[str, int]] = {}
    delta_points: List[Tuple[int, int]] = []
    for y in range(rows):
        prev_row = prev_frame[y]
        next_row = next_frame[y]
        cols = min(len(prev_row), len(next_row), GRID_SIZE)
        for x in range(cols):
            old = int(prev_row[x])
            new = int(next_row[x])
            if old == new:
                continue
            delta_points.append((x, y))
            old_record = per_color.setdefault(old, {"added": 0, "removed": 0})
            new_record = per_color.setdefault(new, {"added": 0, "removed": 0})
            old_record["removed"] += 1
            new_record["added"] += 1

    delta_pixels = len(delta_points)
    bbox: Optional[Tuple[int, int, int, int]] = None
    if delta_points:
        xs = [p[0] for p in delta_points]
        ys = [p[1] for p in delta_points]
        bbox = (min(xs), min(ys), max(xs), max(ys))

    before_dominant = top_k_colors(prev_frame, k=4)
    after_dominant = top_k_colors(next_frame, k=4)

    colors_changed = sorted(
        c for c, rec in per_color.items()
        if rec["added"] > 0 or rec["removed"] > 0
    )

    if delta_pixels == 0:
        label = CHANGE_NONE
    else:
        # Was the color set preserved? If yes -> motion-like.
        before_set = {c for c, _ in before_dominant} | {BACKGROUND_COLOR}
        after_set = {c for c, _ in after_dominant} | {BACKGROUND_COLOR}
        bbox_area = 0
        if bbox is not None:
            bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        if before_set == after_set and delta_pixels <= 32 and bbox_area <= 64:
            label = CHANGE_MOTION_LIKE
        elif delta_pixels <= 12 and len([c for c in colors_changed if c != BACKGROUND_COLOR]) <= 2:
            label = CHANGE_LOCALIZED_REPLACE
        elif delta_pixels >= 96 or len([c for c in colors_changed if c != BACKGROUND_COLOR]) >= 4:
            label = CHANGE_BROAD_MULTI
        else:
            # Fall through: small-to-medium change, color set roughly preserved.
            label = CHANGE_LOCALIZED_REPLACE

    return {
        "delta_pixels": int(delta_pixels),
        "before_dominant": [[int(c), float(s)] for c, s in before_dominant],
        "after_dominant": [[int(c), float(s)] for c, s in after_dominant],
        "per_color_delta": {int(c): {"added": int(v["added"]), "removed": int(v["removed"])}
                            for c, v in per_color.items()},
        "colors_changed": [int(c) for c in colors_changed],
        "change_label": label,
        "bbox_of_change": list(bbox) if bbox is not None else None,
    }


# ---------------------------------------------------------------- aggregator


def merge_change_summary(
    accumulator: Dict[str, Any],
    change: Dict[str, Any],
    signature: str,
) -> Dict[str, Any]:
    """Fold a single transition_color_change into a per-game accumulator.

    The accumulator keeps:
        change_label_counts:   {label: count}
        signature_by_color:    {color: {signature: count}}
        progress_colors:       counts of changed colors that coincided with PROGRESS
        dead_click_colors:     counts of colors at center_color of NO_CHANGE clicks
                               (filled separately by the caller)
    """
    if not accumulator:
        accumulator = {
            "change_label_counts": {label: 0 for label in ALL_CHANGE_LABELS},
            "signature_by_color": {},
            "progress_colors": {},
            "dead_click_colors": {},
        }
    label = change.get("change_label", CHANGE_NONE)
    accumulator["change_label_counts"][label] = accumulator["change_label_counts"].get(label, 0) + 1
    for color in change.get("colors_changed", []) or []:
        # Use str keys for color so the accumulator round-trips through JSON
        # (which only supports string keys) without producing a mixed
        # int/str dict on the next save.
        ckey = str(int(color))
        per_sig = accumulator["signature_by_color"].setdefault(ckey, {})
        per_sig[signature] = per_sig.get(signature, 0) + 1
        if signature == "progress":
            accumulator["progress_colors"][ckey] = (
                accumulator["progress_colors"].get(ckey, 0) + 1
            )
    return accumulator


def record_dead_click_color(accumulator: Dict[str, Any], color: int) -> None:
    if not accumulator:
        return
    counts = accumulator.setdefault("dead_click_colors", {})
    ckey = str(int(color))
    counts[ckey] = counts.get(ckey, 0) + 1


def color_profile_summary(accumulator: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a per-game color accumulator into a compact summary."""
    if not accumulator:
        return {
            "change_label_counts": {label: 0 for label in ALL_CHANGE_LABELS},
            "top_progress_colors": [],
            "top_local_toggle_colors": [],
            "top_global_change_colors": [],
            "top_dead_click_colors": [],
            "color_change_concentration": 0.0,
        }
    label_counts = dict(accumulator.get("change_label_counts", {}))
    sig_by_color = accumulator.get("signature_by_color", {}) or {}

    def top_for(signature: str, n: int = 5) -> List[List[float]]:
        entries = []
        for color, sigs in sig_by_color.items():
            count = int(sigs.get(signature, 0) or 0)
            if count > 0:
                entries.append((int(color), count))
        entries.sort(key=lambda item: item[1], reverse=True)
        return [[c, n] for c, n in entries[:n]]

    dead_clicks = accumulator.get("dead_click_colors", {}) or {}
    dead_click_top = sorted(
        ((int(c), int(n)) for c, n in dead_clicks.items() if n > 0),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    total_changes = sum(label_counts.values()) or 1
    concentration = round(
        max(label_counts.values()) / float(total_changes) if label_counts else 0.0,
        3,
    )

    return {
        "change_label_counts": {k: int(v) for k, v in label_counts.items()},
        "top_progress_colors": top_for("progress"),
        "top_local_toggle_colors": top_for("local_toggle"),
        "top_global_change_colors": top_for("global_change"),
        "top_dead_click_colors": [[c, n] for c, n in dead_click_top],
        "color_change_concentration": concentration,
    }
