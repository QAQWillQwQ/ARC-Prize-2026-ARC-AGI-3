"""Heuristic transition classifier for the probe-first agent.

A frame transition is mapped to one semantic label drawn from a small fixed
vocabulary. The labels are intended to give the agent a coarse handle on what
an action *did* in the world, rather than just whether the score changed.

Labels:
    no_change       frame is byte-identical to the previous frame
    small_change    a few pixels changed but no clear local toggle / motion
    motion_like     a connected component shifted while the rest stayed still
    local_toggle    a small localized pixel flip (likely an interaction)
    global_change   large-scale grid delta (level reset, mechanic shift, etc.)
    progress        levels_completed advanced
    game_over       state_after == GAME_OVER
    win             state_after == WIN

This is intentionally cheap: the heuristics use only utilities that already
exist in src.common (frame delta + connected components), and the classifier
is stateless apart from the inputs. v1 is good enough; a learned classifier
can replace this later without touching the rest of the path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import GRID_SIZE, changed_points, connected_components

# Labels are plain strings so episodes serialize cleanly to JSON.
NO_CHANGE = "no_change"
SMALL_CHANGE = "small_change"
MOTION_LIKE = "motion_like"
LOCAL_TOGGLE = "local_toggle"
GLOBAL_CHANGE = "global_change"
PROGRESS = "progress"
GAME_OVER = "game_over"
WIN = "win"

ALL_SIGNATURES: Tuple[str, ...] = (
    NO_CHANGE,
    SMALL_CHANGE,
    MOTION_LIKE,
    LOCAL_TOGGLE,
    GLOBAL_CHANGE,
    PROGRESS,
    GAME_OVER,
    WIN,
)

# A "local" change is fewer than this many flipped pixels.
SMALL_DELTA_LIMIT = 6
# A "global" change crosses this many flipped pixels (out of 4096).
LARGE_DELTA_LIMIT = 96
# Match tolerance when trying to identify the avatar via component matching.
MOTION_AREA_TOLERANCE = 0.25


@dataclass
class EffectRecord:
    signature: str
    delta_pixels: int
    progress_gain: int
    bbox_of_change: Optional[Tuple[int, int, int, int]]
    inferred_motion: Optional[Tuple[int, int]]
    component_count_before: int
    component_count_after: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "delta_pixels": int(self.delta_pixels),
            "progress_gain": int(self.progress_gain),
            "bbox_of_change": list(self.bbox_of_change) if self.bbox_of_change else None,
            "inferred_motion": list(self.inferred_motion) if self.inferred_motion else None,
            "component_count_before": int(self.component_count_before),
            "component_count_after": int(self.component_count_after),
        }


def _bbox_of_points(points: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not points:
        return None
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _try_infer_motion(
    components_before: List[Dict[str, Any]],
    components_after: List[Dict[str, Any]],
) -> Optional[Tuple[int, int]]:
    """Greedy match top components by color + similar area, return the largest centroid shift."""
    if not components_before or not components_after:
        return None
    if abs(len(components_before) - len(components_after)) > 1:
        return None
    used = [False] * min(len(components_after), 3)
    matched: List[Tuple[int, int, int, int, int]] = []
    for comp_b in components_before[:3]:
        best_idx = -1
        best_score = float("inf")
        for idx, comp_a in enumerate(components_after[:3]):
            if used[idx]:
                continue
            if comp_b["color"] != comp_a["color"]:
                continue
            area_b = max(1, int(comp_b["area"]))
            area_diff = abs(area_b - int(comp_a["area"])) / float(area_b)
            if area_diff > MOTION_AREA_TOLERANCE:
                continue
            cx_b, cy_b = comp_b["center"]
            cx_a, cy_a = comp_a["center"]
            distance = abs(cx_b - cx_a) + abs(cy_b - cy_a)
            score = distance + area_diff * 8.0
            if score < best_score:
                best_score = score
                best_idx = idx
        if best_idx == -1:
            continue
        used[best_idx] = True
        cx_b, cy_b = comp_b["center"]
        cx_a, cy_a = components_after[best_idx]["center"]
        matched.append((cx_b, cy_b, cx_a, cy_a, int(comp_b["area"])))
    if not matched:
        return None
    matched.sort(key=lambda m: abs(m[0] - m[2]) + abs(m[1] - m[3]), reverse=True)
    cx_b, cy_b, cx_a, cy_a, _area = matched[0]
    dx = cx_a - cx_b
    dy = cy_a - cy_b
    if dx == 0 and dy == 0:
        return None
    return (int(dx), int(dy))


def classify_transition(
    prev_frame: Sequence[Sequence[int]],
    next_frame: Sequence[Sequence[int]],
    levels_before: int,
    levels_after: int,
    state_after: str,
) -> EffectRecord:
    progress_gain = max(0, int(levels_after) - int(levels_before))
    delta_pts = changed_points(prev_frame, next_frame)
    delta = len(delta_pts)
    bbox = _bbox_of_points(delta_pts)
    components_before = connected_components(prev_frame)
    components_after = connected_components(next_frame)
    motion = _try_infer_motion(components_before, components_after)

    if state_after == "WIN":
        signature = WIN
    elif state_after == "GAME_OVER":
        signature = GAME_OVER
    elif progress_gain > 0:
        signature = PROGRESS
    elif delta == 0:
        signature = NO_CHANGE
    elif delta >= LARGE_DELTA_LIMIT or abs(len(components_before) - len(components_after)) >= 3:
        signature = GLOBAL_CHANGE
    elif delta <= SMALL_DELTA_LIMIT:
        signature = MOTION_LIKE if motion is not None else LOCAL_TOGGLE
    else:
        signature = MOTION_LIKE if motion is not None else SMALL_CHANGE

    return EffectRecord(
        signature=signature,
        delta_pixels=int(delta),
        progress_gain=int(progress_gain),
        bbox_of_change=bbox,
        inferred_motion=motion if signature == MOTION_LIKE else None,
        component_count_before=len(components_before),
        component_count_after=len(components_after),
    )


def summarize_signatures(records: Sequence[EffectRecord]) -> Dict[str, int]:
    counts = {sig: 0 for sig in ALL_SIGNATURES}
    for record in records:
        counts[record.signature] = counts.get(record.signature, 0) + 1
    return counts


def is_effective(signature: str) -> bool:
    return signature not in (NO_CHANGE, GAME_OVER)
