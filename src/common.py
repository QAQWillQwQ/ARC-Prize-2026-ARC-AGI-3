from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

GRID_SIZE = 64
NUM_COLORS = 16
ACTION_IDS = [1, 2, 3, 4, 5, 6, 7]
ACTION_TO_INDEX = {action_id: idx for idx, action_id in enumerate(ACTION_IDS)}
INDEX_TO_ACTION = {idx: action_id for action_id, idx in ACTION_TO_INDEX.items()}


@dataclass
class CandidateAction:
    action_id: int
    action_data: Optional[Dict[str, int]] = None
    score: float = 0.0
    source: str = "heuristic"
    metadata: Optional[Dict[str, Any]] = None


PROFILES: Dict[str, Dict[str, Any]] = {
    "a100": {
        "batch_size": 192,
        "grad_accum": 1,
        "model_dim": 384,
        "num_slots": 8,
        "depth": 6,
        "num_heads": 8,
        "history": 4,
        "collect_episodes_per_game": 24,
        "beam_width": 6,
        "branch_factor": 10,
        "coord_budget": 20,
        "epochs": 16,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "online_val_every": 2,
    },
    "h100": {
        "batch_size": 256,
        "grad_accum": 1,
        "model_dim": 448,
        "num_slots": 10,
        "depth": 8,
        "num_heads": 8,
        "history": 4,
        "collect_episodes_per_game": 32,
        "beam_width": 8,
        "branch_factor": 12,
        "coord_budget": 24,
        "epochs": 18,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "online_val_every": 2,
    },
    "rtx3070ti": {
        "batch_size": 16,
        "grad_accum": 8,
        "model_dim": 256,
        "num_slots": 6,
        "depth": 4,
        "num_heads": 8,
        "history": 4,
        "collect_episodes_per_game": 10,
        "beam_width": 4,
        "branch_factor": 6,
        "coord_budget": 12,
        "epochs": 12,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "online_val_every": 3,
    },
    "cpu_debug": {
        "batch_size": 4,
        "grad_accum": 1,
        "model_dim": 128,
        "num_slots": 4,
        "depth": 2,
        "num_heads": 4,
        "history": 2,
        "collect_episodes_per_game": 2,
        "beam_width": 2,
        "branch_factor": 4,
        "coord_budget": 8,
        "epochs": 2,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "online_val_every": 1000,
    },
    "m3_cpu": {
        "batch_size": 8,
        "grad_accum": 2,
        "model_dim": 192,
        "num_slots": 4,
        "depth": 3,
        "num_heads": 4,
        "history": 4,
        "collect_episodes_per_game": 6,
        "beam_width": 3,
        "branch_factor": 5,
        "coord_budget": 10,
        "epochs": 8,
        "lr": 4e-4,
        "weight_decay": 0.02,
        "online_val_every": 4,
    },
}


def get_profile(name: str) -> Dict[str, Any]:
    if name not in PROFILES:
        raise KeyError("Unknown hardware profile: %s" % name)
    return dict(PROFILES[name])


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_metrics_row(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_jsonl_gz(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def append_jsonl_gz(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")


def iter_jsonl_gz(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return iter(())

    def _generator() -> Iterator[Dict[str, Any]]:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    return _generator()


def load_metadata_map(environments_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for metadata_path in environments_dir.rglob("metadata.json"):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        game_id = str(payload["game_id"]).split("-", 1)[0]
        out[game_id] = payload
    return out


def split_games(game_ids: Sequence[str], holdout_fraction: float = 0.2) -> Tuple[List[str], List[str]]:
    unique_games = sorted(set(game_ids))
    if not unique_games:
        return [], []
    if len(unique_games) == 1:
        return list(unique_games), list(unique_games)

    ranked = []
    for game_id in unique_games:
        digest = hashlib.sha1(("arcagi3-split-" + game_id).encode("utf-8")).hexdigest()
        ranked.append((digest, game_id))
    ranked.sort()
    holdout = max(1, int(round(len(unique_games) * holdout_fraction)))
    val_games = sorted(game_id for _, game_id in ranked[:holdout])
    train_games = sorted(game_id for _, game_id in ranked[holdout:])
    return train_games, val_games


def action_mask(available_actions: Sequence[int]) -> List[int]:
    available = set(int(action) for action in available_actions)
    return [1 if action_id in available else 0 for action_id in ACTION_IDS]


def final_subframe(frame_stack: Sequence[Sequence[Sequence[int]]]) -> List[List[int]]:
    if not frame_stack:
        return [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    frame = frame_stack[-1]
    return [[int(value) for value in row] for row in frame]


def informative_subframe(
    frame_stack: Sequence[Sequence[Sequence[int]]],
    reference_frame: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[List[List[int]], int, int]:
    if not frame_stack:
        blank = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        return blank, 0, 0
    if reference_frame is None:
        return final_subframe(frame_stack), len(frame_stack) - 1, 0

    best_frame = final_subframe(frame_stack)
    best_index = len(frame_stack) - 1
    best_delta = frame_delta(reference_frame, best_frame)
    for index, raw_frame in enumerate(frame_stack):
        frame = [[int(value) for value in row] for row in raw_frame]
        delta = frame_delta(reference_frame, frame)
        if delta > best_delta:
            best_frame = frame
            best_index = index
            best_delta = delta
    return best_frame, best_index, best_delta


def frame_hash(frame: Sequence[Sequence[int]]) -> str:
    flat = bytes(int(value) & 0x0F for row in frame for value in row)
    return hashlib.sha1(flat).hexdigest()


def frame_delta(prev_frame: Sequence[Sequence[int]], next_frame: Sequence[Sequence[int]]) -> int:
    delta = 0
    for y in range(min(len(prev_frame), len(next_frame))):
        row_a = prev_frame[y]
        row_b = next_frame[y]
        for x in range(min(len(row_a), len(row_b))):
            if int(row_a[x]) != int(row_b[x]):
                delta += 1
    return delta


def non_background_density(frame: Sequence[Sequence[int]]) -> float:
    if torch.is_tensor(frame):
        if frame.numel() == 0:
            return 0.0
        return float((frame != 0).to(dtype=torch.float32).mean().item())
    total = GRID_SIZE * GRID_SIZE
    occupied = 0
    for row in frame:
        for value in row:
            if int(value) != 0:
                occupied += 1
    return occupied / float(total)


def _grid_points(step: int) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for y in range(step // 2, GRID_SIZE, step):
        for x in range(step // 2, GRID_SIZE, step):
            coords.append((x, y))
    return coords


def connected_components(frame: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    visited = [[False for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    components: List[Dict[str, Any]] = []
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            color = int(frame[y][x])
            if color == 0 or visited[y][x]:
                continue
            stack = [(x, y)]
            visited[y][x] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in offsets:
                    nx = cx + dx
                    ny = cy + dy
                    if nx < 0 or ny < 0 or nx >= GRID_SIZE or ny >= GRID_SIZE:
                        continue
                    if visited[ny][nx]:
                        continue
                    if int(frame[ny][nx]) != color:
                        continue
                    visited[ny][nx] = True
                    stack.append((nx, ny))
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            components.append(
                {
                    "color": color,
                    "area": len(cells),
                    "center": (int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))),
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "cells": cells,
                }
            )
    components.sort(key=lambda item: item["area"], reverse=True)
    return components


def changed_points(prev_frame: Sequence[Sequence[int]], next_frame: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if int(prev_frame[y][x]) != int(next_frame[y][x]):
                points.append((x, y))
    return points


def color_counts(frame: Sequence[Sequence[int]]) -> Dict[int, int]:
    counts = {color: 0 for color in range(NUM_COLORS)}
    for y in range(min(GRID_SIZE, len(frame))):
        row = frame[y]
        for x in range(min(GRID_SIZE, len(row))):
            color = int(row[x])
            if color < 0 or color >= NUM_COLORS:
                color = 0
            counts[color] += 1
    return counts


def ranked_colors(frame: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    counts = color_counts(frame)
    total = max(1, sum(counts.values()))
    ranked = [
        {
            "color": int(color),
            "count": int(count),
            "frequency": float(count) / float(total),
        }
        for color, count in counts.items()
        if count > 0
    ]
    ranked.sort(key=lambda item: (int(item["count"]), int(item["color"])), reverse=True)
    return ranked


def visual_saliency_summary(frame: Sequence[Sequence[int]]) -> Dict[str, Any]:
    counts = color_counts(frame)
    total = max(1, sum(counts.values()))
    ranked = ranked_colors(frame)
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = float(count) / float(total)
        entropy -= probability * math.log(probability + 1e-12)
    entropy /= math.log(float(NUM_COLORS))

    dominant_color = int(ranked[0]["color"]) if ranked else 0
    dominant_count = int(ranked[0]["count"]) if ranked else 0
    rare_threshold = max(1, int(total * 0.025))
    rare_colors = [
        int(color)
        for color, count in counts.items()
        if count > 0 and count <= rare_threshold
    ]
    return {
        "dominant_color": dominant_color,
        "dominant_frequency": float(dominant_count) / float(total),
        "num_colors": len(ranked),
        "color_entropy": entropy,
        "rare_color_count": len(rare_colors),
        "rare_colors": rare_colors,
        "top_colors": ranked[:6],
    }


def _component_lookup(components: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], int]:
    lookup: Dict[Tuple[int, int], int] = {}
    for index, component in enumerate(components):
        for point in component.get("cells", []):
            lookup[(int(point[0]), int(point[1]))] = index
    return lookup


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round_float(value: float) -> float:
    return round(float(value), 6)


def point_visual_features(
    frame: Sequence[Sequence[int]],
    point: Tuple[int, int],
    prev_frame: Optional[Sequence[Sequence[int]]] = None,
    components: Optional[Sequence[Dict[str, Any]]] = None,
    component_lookup: Optional[Dict[Tuple[int, int], int]] = None,
    counts: Optional[Dict[int, int]] = None,
    changed_lookup: Optional[set[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    x = max(0, min(GRID_SIZE - 1, int(point[0])))
    y = max(0, min(GRID_SIZE - 1, int(point[1])))
    color = int(frame[y][x])
    if color < 0 or color >= NUM_COLORS:
        color = 0

    counts = dict(counts or color_counts(frame))
    total = max(1, sum(counts.values()))
    color_count = int(counts.get(color, 0))
    color_frequency = float(color_count) / float(total)
    color_rarity = 1.0 - math.sqrt(max(0.0, min(1.0, color_frequency)))

    ranked = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    dominant_color = int(ranked[0][0]) if ranked else 0
    dominant_frequency = float(ranked[0][1]) / float(total) if ranked else 0.0
    non_dominant = int(color != dominant_color)

    neighbor_total = 0
    different_neighbors = 0
    neighbor_colors = set()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= GRID_SIZE or ny >= GRID_SIZE:
                continue
            neighbor_total += 1
            neighbor_color = int(frame[ny][nx])
            if neighbor_color < 0 or neighbor_color >= NUM_COLORS:
                neighbor_color = 0
            neighbor_colors.add(neighbor_color)
            different_neighbors += int(neighbor_color != color)
    local_contrast = float(different_neighbors) / float(max(1, neighbor_total))
    neighbor_color_diversity = float(len(neighbor_colors)) / float(max(1, NUM_COLORS))

    components = list(components or connected_components(frame))
    component_lookup = dict(component_lookup or _component_lookup(components))
    component_index = component_lookup.get((x, y), -1)
    component_area = 0
    component_center = (x, y)
    component_bbox = (x, y, x, y)
    component_width = 1
    component_height = 1
    bbox_area = 1
    edge_cell = 0
    component_fill = 1.0
    if component_index >= 0 and component_index < len(components):
        component = components[component_index]
        component_area = int(component.get("area", 0))
        component_center = tuple(component.get("center", (x, y)))  # type: ignore[assignment]
        component_bbox = tuple(component.get("bbox", (x, y, x, y)))  # type: ignore[assignment]
        x0, y0, x1, y1 = (int(value) for value in component_bbox)
        component_width = max(1, x1 - x0 + 1)
        component_height = max(1, y1 - y0 + 1)
        bbox_area = max(1, component_width * component_height)
        component_fill = float(component_area) / float(bbox_area)
        edge_cell = int(x in (x0, x1) or y in (y0, y1))
    component_area_ratio = float(component_area) / float(total)
    component_rarity = 1.0 - math.sqrt(max(0.0, min(1.0, component_area_ratio)))
    elongation = abs(component_width - component_height) / float(max(component_width, component_height, 1))
    shape_signal = _clamp01((edge_cell * 0.45) + (1.0 - component_fill) * 0.35 + elongation * 0.2)

    if changed_lookup is None and prev_frame is not None:
        changed_lookup = set(changed_points(prev_frame, frame))
    changed_lookup = changed_lookup or set()
    changed_here = int((x, y) in changed_lookup)
    changed_nearby = 0
    for ny in range(max(0, y - 2), min(GRID_SIZE, y + 3)):
        for nx in range(max(0, x - 2), min(GRID_SIZE, x + 3)):
            changed_nearby += int((nx, ny) in changed_lookup)
    changed_nearby_ratio = float(changed_nearby) / 25.0

    salience = (
        color_rarity * 0.38
        + local_contrast * 0.22
        + component_rarity * 0.20
        + shape_signal * 0.10
        + changed_nearby_ratio * 0.16
        + neighbor_color_diversity * 0.08
        + non_dominant * 0.06
    )
    if color == dominant_color and dominant_frequency >= 0.35 and component_area_ratio >= 0.12:
        salience -= 0.18
    if component_area == 0:
        salience *= 0.7
    salience = _clamp01(salience)

    return {
        "x": x,
        "y": y,
        "color": color,
        "color_count": color_count,
        "color_frequency": _round_float(color_frequency),
        "color_rarity": _round_float(color_rarity),
        "dominant_color": dominant_color,
        "dominant_frequency": _round_float(dominant_frequency),
        "non_dominant": non_dominant,
        "local_contrast": _round_float(local_contrast),
        "neighbor_color_diversity": _round_float(neighbor_color_diversity),
        "component_index": int(component_index),
        "component_area": int(component_area),
        "component_area_ratio": _round_float(component_area_ratio),
        "component_rarity": _round_float(component_rarity),
        "component_center": [int(component_center[0]), int(component_center[1])],
        "component_bbox": [int(value) for value in component_bbox],
        "component_width": int(component_width),
        "component_height": int(component_height),
        "component_fill": _round_float(component_fill),
        "component_elongation": _round_float(elongation),
        "component_edge_cell": edge_cell,
        "shape_signal": _round_float(shape_signal),
        "changed_here": changed_here,
        "changed_nearby_ratio": _round_float(changed_nearby_ratio),
        "salience_score": _round_float(salience),
    }


def salient_point_features(
    frame: Sequence[Sequence[int]],
    prev_frame: Optional[Sequence[Sequence[int]]] = None,
    budget: int = 20,
) -> List[Dict[str, Any]]:
    budget = max(1, int(budget))
    counts = color_counts(frame)
    total = max(1, sum(counts.values()))
    components = connected_components(frame)
    lookup = _component_lookup(components)
    changed_lookup = set(changed_points(prev_frame, frame)) if prev_frame is not None else set()
    by_point: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def add(point: Tuple[int, int], source: str, source_bonus: float = 0.0) -> None:
        x = max(0, min(GRID_SIZE - 1, int(point[0])))
        y = max(0, min(GRID_SIZE - 1, int(point[1])))
        feature = point_visual_features(
            frame=frame,
            point=(x, y),
            prev_frame=prev_frame,
            components=components,
            component_lookup=lookup,
            counts=counts,
            changed_lookup=changed_lookup,
        )
        adjusted = _clamp01(float(feature["salience_score"]) + float(source_bonus))
        feature["salience_score"] = _round_float(adjusted)
        feature["source"] = source
        existing = by_point.get((x, y))
        if existing is None or float(feature["salience_score"]) > float(existing["salience_score"]):
            by_point[(x, y)] = feature

    ranked_component_points: List[Tuple[float, Dict[str, Any]]] = []
    for component in components:
        area = int(component.get("area", 0))
        color = int(component.get("color", 0))
        color_frequency = float(counts.get(color, 0)) / float(total)
        area_ratio = float(area) / float(total)
        component_signal = (
            (1.0 - math.sqrt(max(0.0, min(1.0, color_frequency)))) * 0.45
            + (1.0 - math.sqrt(max(0.0, min(1.0, area_ratio)))) * 0.35
        )
        if area_ratio >= 0.25:
            component_signal -= 0.25
        ranked_component_points.append((component_signal, component))
    ranked_component_points.sort(key=lambda item: item[0], reverse=True)

    component_limit = max(budget * 2, 12)
    for signal, component in ranked_component_points[:component_limit]:
        center = tuple(component["center"])
        x0, y0, x1, y1 = (int(value) for value in component["bbox"])
        points = [
            center,
            (x0, y0),
            (x1, y0),
            (x0, y1),
            (x1, y1),
            ((x0 + x1) // 2, y0),
            ((x0 + x1) // 2, y1),
            (x0, (y0 + y1) // 2),
            (x1, (y0 + y1) // 2),
        ]
        bonus = max(0.0, min(0.08, signal * 0.05))
        for point in points:
            add(point, source="component", source_bonus=bonus)

    if changed_lookup:
        delta_points = sorted(changed_lookup)
        stride = max(1, len(delta_points) // max(1, budget))
        for point in delta_points[::stride]:
            add(point, source="changed", source_bonus=0.10)

    rare_color_threshold = max(1, int(total * 0.025))
    rare_colors = {
        int(color)
        for color, count in counts.items()
        if count > 0 and count <= rare_color_threshold
    }
    for component in components:
        if int(component.get("color", 0)) not in rare_colors:
            continue
        add(tuple(component["center"]), source="rare_color", source_bonus=0.08)

    for point in _grid_points(step=16):
        add(point, source="grid", source_bonus=-0.12)
    for point in _grid_points(step=32):
        add(point, source="grid", source_bonus=-0.10)
    add((GRID_SIZE // 2, GRID_SIZE // 2), source="grid", source_bonus=-0.08)

    ranked = sorted(
        by_point.values(),
        key=lambda item: (
            float(item["salience_score"]),
            float(item["color_rarity"]),
            float(item["local_contrast"]),
            -float(item["component_area_ratio"]),
        ),
        reverse=True,
    )
    return ranked[:budget]


def salient_points(
    frame: Sequence[Sequence[int]],
    prev_frame: Optional[Sequence[Sequence[int]]] = None,
    budget: int = 20,
) -> List[Tuple[int, int]]:
    return [
        (int(feature["x"]), int(feature["y"]))
        for feature in salient_point_features(frame=frame, prev_frame=prev_frame, budget=budget)
    ]


def one_hot_frames(history: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
    encoded = torch.zeros((len(history) * NUM_COLORS, GRID_SIZE, GRID_SIZE), dtype=torch.float32)
    for frame_idx, frame in enumerate(history):
        for y in range(GRID_SIZE):
            row = frame[y]
            for x in range(GRID_SIZE):
                color = int(row[x])
                if color < 0 or color >= NUM_COLORS:
                    color = 0
                encoded[frame_idx * NUM_COLORS + color, y, x] = 1.0
    return encoded


def pad_history(frames: Sequence[Sequence[Sequence[int]]], history: int) -> List[List[List[int]]]:
    if not frames:
        blank = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        return [blank for _ in range(history)]
    stacked = [
        [[int(value) for value in row] for row in frame]
        for frame in frames[-history:]
    ]
    while len(stacked) < history:
        stacked.insert(0, [row[:] for row in stacked[0]])
    return stacked


def scalar_features(
    available_actions: Sequence[int],
    last_action_id: Optional[int],
    levels_completed: int,
    steps_since_progress: int,
    step_index: int,
    frame: Sequence[Sequence[int]],
    max_steps: int,
) -> torch.Tensor:
    features: List[float] = []
    features.extend(float(value) for value in action_mask(available_actions))
    last_action = [0.0 for _ in ACTION_IDS]
    if last_action_id in ACTION_TO_INDEX:
        last_action[ACTION_TO_INDEX[last_action_id]] = 1.0
    features.extend(last_action)
    features.extend(
        [
            min(levels_completed / 10.0, 1.0),
            min(steps_since_progress / max(max_steps, 1), 1.0),
            min(step_index / max(max_steps, 1), 1.0),
            non_background_density(frame),
        ]
    )
    return torch.tensor(features, dtype=torch.float32)


def rhae_score(baseline_actions: Sequence[int], completed_level_actions: Sequence[int]) -> Dict[str, Any]:
    weighted_score = 0.0
    total_weights = 0.0
    unlocked_weights = 0.0
    level_scores: List[float] = []

    for idx, baseline in enumerate(baseline_actions, start=1):
        total_weights += idx
        if idx - 1 < len(completed_level_actions):
            actions = max(1, int(completed_level_actions[idx - 1]))
            score = ((float(baseline) / float(actions)) ** 2) * 100.0
            score = min(score, 115.0)
            unlocked_weights += idx
        else:
            score = 0.0
        level_scores.append(score)
        weighted_score += score * idx

    if total_weights == 0:
        return {
            "score": 0.0,
            "level_scores": level_scores,
            "max_score": 0.0,
        }

    score = weighted_score / total_weights
    max_score = (unlocked_weights / total_weights) * 100.0
    score = min(score, max_score)
    return {
        "score": score,
        "level_scores": level_scores,
        "max_score": max_score,
    }


def episode_level_actions(transitions: Sequence[Dict[str, Any]]) -> List[int]:
    completed_level_actions: List[int] = []
    running = 0
    previous_levels = 0
    for transition in transitions:
        running += 1
        levels_after = int(transition.get("levels_after", previous_levels))
        if levels_after > previous_levels:
            increments = levels_after - previous_levels
            completed_level_actions.append(running)
            for _ in range(max(0, increments - 1)):
                completed_level_actions.append(1)
            running = 0
            previous_levels = levels_after
    return completed_level_actions


def novelty_bonus(signature: str, seen_signatures: Sequence[str]) -> float:
    if signature not in seen_signatures:
        return 1.0
    count = 0
    for existing in seen_signatures:
        if existing == signature:
            count += 1
    return 1.0 / float(1 + count)


def compute_discounted_returns(rewards: Sequence[float], gamma: float = 0.97) -> List[float]:
    out = [0.0 for _ in rewards]
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        out[idx] = running
    return out


def transition_reward(transition: Dict[str, Any]) -> float:
    progress = int(transition.get("levels_after", 0)) - int(transition.get("levels_before", 0))
    delta = float(transition.get("delta_pixels", 0))
    novelty = float(transition.get("novelty", 0.0))
    won = 1.0 if transition.get("state_after") == "WIN" else 0.0
    metadata = dict(transition.get("action_metadata") or transition.get("candidate_metadata") or {})
    visual_salience = float(metadata.get("visual_salience", 0.0))
    if "point_visual" in metadata and isinstance(metadata["point_visual"], dict):
        visual_salience = max(visual_salience, float(metadata["point_visual"].get("salience_score", 0.0)))
    visual_term = min(max(visual_salience, 0.0), 1.0) * (0.06 if delta > 0 or progress > 0 else -0.02)
    reward = (3.0 * progress) + min(delta / 128.0, 1.0) * 0.1 + novelty * 0.25 + won * 2.0 + visual_term - 0.01
    return reward


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def topk_indices(values: Sequence[float], k: int) -> List[int]:
    ranked = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    return ranked[:k]


def tensor_to_list(tensor: torch.Tensor) -> List[float]:
    return [float(value) for value in tensor.detach().cpu().view(-1)]


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def merge_config(profile_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    config = get_profile(profile_name)
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config
