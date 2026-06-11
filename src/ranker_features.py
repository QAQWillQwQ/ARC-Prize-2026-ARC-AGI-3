from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import (
    ACTION_IDS,
    GRID_SIZE,
    NUM_COLORS,
    CandidateAction,
    point_visual_features,
    salient_point_features,
    visual_saliency_summary,
)


FEATURE_NAMES: List[str] = [
    "bias",
    "is_action_1",
    "is_action_2",
    "is_action_3",
    "is_action_4",
    "is_action_5",
    "is_action_6",
    "is_action_7",
    "available_action_count",
    "has_click_xy",
    "x_norm",
    "y_norm",
    "center_distance_norm",
    "edge_distance_norm",
    "level_norm",
    "step_norm",
    "frame_dominant_frequency",
    "frame_num_colors_norm",
    "frame_color_entropy",
    "frame_rare_color_count_norm",
    "color_frequency",
    "color_rarity",
    "non_dominant",
    "local_contrast",
    "neighbor_color_diversity",
    "component_area_ratio",
    "component_rarity",
    "component_fill",
    "component_elongation",
    "component_edge_cell",
    "shape_signal",
    "changed_here",
    "changed_nearby_ratio",
    "salience_score",
    "candidate_source_saliency",
    "candidate_source_changed",
    "candidate_source_rare_color",
    "candidate_source_grid",
    "candidate_source_actual",
    "source_verified",
]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _blank_point_features() -> Dict[str, Any]:
    return {
        "color_frequency": 0.0,
        "color_rarity": 0.0,
        "non_dominant": 0,
        "local_contrast": 0.0,
        "neighbor_color_diversity": 0.0,
        "component_area_ratio": 0.0,
        "component_rarity": 0.0,
        "component_fill": 0.0,
        "component_elongation": 0.0,
        "component_edge_cell": 0,
        "shape_signal": 0.0,
        "changed_here": 0,
        "changed_nearby_ratio": 0.0,
        "salience_score": 0.0,
        "source": "",
    }


def action_matches(
    candidate: CandidateAction,
    action_id: int,
    action_data: Optional[Dict[str, Any]],
    click_tolerance: int = 1,
) -> bool:
    if int(candidate.action_id) != int(action_id):
        return False
    if int(action_id) != 6:
        return True
    candidate_data = dict(candidate.action_data or {})
    observed_data = dict(action_data or {})
    if "x" not in candidate_data or "y" not in candidate_data:
        return False
    if "x" not in observed_data or "y" not in observed_data:
        return False
    return (
        abs(int(candidate_data["x"]) - int(observed_data["x"])) <= int(click_tolerance)
        and abs(int(candidate_data["y"]) - int(observed_data["y"])) <= int(click_tolerance)
    )


def transition_utility(transition: Dict[str, Any]) -> float:
    progress = max(0, int(transition.get("levels_after", 0)) - int(transition.get("levels_before", 0)))
    delta = max(0, int(transition.get("delta_pixels", 0)))
    novelty = float(transition.get("novelty", 0.0))
    state_after = str(transition.get("state_after", ""))
    utility = float(progress) * 8.0
    utility += min(float(delta) / 128.0, 1.5)
    utility += min(max(novelty, 0.0), 1.0) * 0.25
    if delta == 0 and progress == 0:
        utility -= 0.35
    if state_after == "GAME_OVER":
        utility -= 2.0
    if state_after == "WIN":
        utility += 10.0
    return float(utility)


def build_candidate_actions(
    frame: Sequence[Sequence[int]],
    prev_frame: Optional[Sequence[Sequence[int]]],
    available_actions: Sequence[int],
    actual_action_id: Optional[int] = None,
    actual_action_data: Optional[Dict[str, Any]] = None,
    coord_budget: int = 32,
) -> List[CandidateAction]:
    available = sorted(set(int(action_id) for action_id in available_actions if int(action_id) in ACTION_IDS))
    candidates: List[CandidateAction] = []
    seen: set[Tuple[int, int, int]] = set()

    for action_id in available:
        if action_id == 6:
            features = salient_point_features(frame=frame, prev_frame=prev_frame, budget=coord_budget)
            if actual_action_id == 6 and actual_action_data:
                x = int(actual_action_data.get("x", 0))
                y = int(actual_action_data.get("y", 0))
                actual_feature = point_visual_features(frame=frame, point=(x, y), prev_frame=prev_frame)
                actual_feature["source"] = "actual"
                features.insert(0, actual_feature)
            for feature in features:
                x = int(feature["x"])
                y = int(feature["y"])
                key = (action_id, x, y)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    CandidateAction(
                        action_id=6,
                        action_data={"x": x, "y": y},
                        score=float(feature.get("salience_score", 0.0)),
                        source=str(feature.get("source", "saliency")),
                        metadata={"point_visual": dict(feature)},
                    )
                )
        else:
            key = (action_id, -1, -1)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                CandidateAction(
                    action_id=action_id,
                    action_data={},
                    score=0.0,
                    source="available_action",
                    metadata={},
                )
            )
    return candidates


def candidate_feature_vector(
    candidate: CandidateAction,
    frame: Sequence[Sequence[int]],
    prev_frame: Optional[Sequence[Sequence[int]]],
    available_actions: Sequence[int],
    levels_before: int,
    step_index: int,
    max_steps: int,
    source_verified: bool = False,
) -> List[float]:
    action_id = int(candidate.action_id)
    action_data = dict(candidate.action_data or {})
    frame_summary = visual_saliency_summary(frame)
    point_features = _blank_point_features()
    has_xy = action_id == 6 and "x" in action_data and "y" in action_data
    x = int(action_data.get("x", GRID_SIZE // 2)) if has_xy else GRID_SIZE // 2
    y = int(action_data.get("y", GRID_SIZE // 2)) if has_xy else GRID_SIZE // 2
    if has_xy:
        metadata = dict(candidate.metadata or {})
        if isinstance(metadata.get("point_visual"), dict):
            point_features.update(metadata["point_visual"])
        else:
            point_features = point_visual_features(frame=frame, point=(x, y), prev_frame=prev_frame)

    source = str(candidate.source or point_features.get("source", ""))
    x_norm = float(x) / float(GRID_SIZE - 1)
    y_norm = float(y) / float(GRID_SIZE - 1)
    center_distance = (abs(x - ((GRID_SIZE - 1) / 2.0)) + abs(y - ((GRID_SIZE - 1) / 2.0))) / float(GRID_SIZE)
    edge_distance = min(x, y, GRID_SIZE - 1 - x, GRID_SIZE - 1 - y) / float((GRID_SIZE - 1) / 2.0)

    values: Dict[str, float] = {
        "bias": 1.0,
        "available_action_count": float(len(set(int(a) for a in available_actions))) / float(len(ACTION_IDS)),
        "has_click_xy": float(has_xy),
        "x_norm": x_norm,
        "y_norm": y_norm,
        "center_distance_norm": _clamp01(center_distance),
        "edge_distance_norm": _clamp01(edge_distance),
        "level_norm": min(float(levels_before) / 10.0, 1.0),
        "step_norm": min(float(step_index) / float(max(1, max_steps)), 1.0),
        "frame_dominant_frequency": float(frame_summary.get("dominant_frequency", 0.0)),
        "frame_num_colors_norm": float(frame_summary.get("num_colors", 0.0)) / float(NUM_COLORS),
        "frame_color_entropy": float(frame_summary.get("color_entropy", 0.0)),
        "frame_rare_color_count_norm": float(frame_summary.get("rare_color_count", 0.0)) / float(NUM_COLORS),
        "color_frequency": float(point_features.get("color_frequency", 0.0)),
        "color_rarity": float(point_features.get("color_rarity", 0.0)),
        "non_dominant": float(point_features.get("non_dominant", 0.0)),
        "local_contrast": float(point_features.get("local_contrast", 0.0)),
        "neighbor_color_diversity": float(point_features.get("neighbor_color_diversity", 0.0)),
        "component_area_ratio": float(point_features.get("component_area_ratio", 0.0)),
        "component_rarity": float(point_features.get("component_rarity", 0.0)),
        "component_fill": float(point_features.get("component_fill", 0.0)),
        "component_elongation": float(point_features.get("component_elongation", 0.0)),
        "component_edge_cell": float(point_features.get("component_edge_cell", 0.0)),
        "shape_signal": float(point_features.get("shape_signal", 0.0)),
        "changed_here": float(point_features.get("changed_here", 0.0)),
        "changed_nearby_ratio": float(point_features.get("changed_nearby_ratio", 0.0)),
        "salience_score": float(point_features.get("salience_score", 0.0)),
        "candidate_source_saliency": float(source in {"component", "saliency", "component_event"}),
        "candidate_source_changed": float(source in {"changed", "event_delta"}),
        "candidate_source_rare_color": float(source == "rare_color"),
        "candidate_source_grid": float(source in {"grid", "coarse_grid", "center"}),
        "candidate_source_actual": float(source == "actual"),
        "source_verified": float(bool(source_verified)),
    }
    for action in ACTION_IDS:
        values["is_action_%d" % action] = float(action_id == action)
    return [float(values[name]) for name in FEATURE_NAMES]


def source_phase_label(transition: Dict[str, Any]) -> str:
    source = str(transition.get("action_source", ""))
    metadata = dict(transition.get("action_metadata") or {})
    if source == "source_planner":
        return str(metadata.get("planner_method", "source_planner"))
    if source == "forge_bfs_replay":
        return "source_bfs_replay"
    if source == "forge_cnn_fallback":
        return "cnn_fallback"
    if source in {"policy", "policy+coord"}:
        return "checkpoint_policy"
    if source in {"coord", "heuristic"}:
        return "heuristic_fallback"
    return source or "unknown"
