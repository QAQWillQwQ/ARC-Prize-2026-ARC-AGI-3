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


def salient_points(
    frame: Sequence[Sequence[int]],
    prev_frame: Optional[Sequence[Sequence[int]]] = None,
    budget: int = 20,
) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int]] = []
    seen = set()

    def add(point: Tuple[int, int]) -> None:
        x, y = point
        x = max(0, min(GRID_SIZE - 1, int(x)))
        y = max(0, min(GRID_SIZE - 1, int(y)))
        key = (x, y)
        if key not in seen:
            seen.add(key)
            candidates.append(key)

    for component in connected_components(frame)[: max(1, budget // 2)]:
        add(component["center"])
        x0, y0, x1, y1 = component["bbox"]
        add((x0, y0))
        add((x1, y0))
        add((x0, y1))
        add((x1, y1))

    if prev_frame is not None:
        delta_points = changed_points(prev_frame, frame)
        if delta_points:
            stride = max(1, len(delta_points) // max(1, budget // 3))
            for point in delta_points[::stride]:
                add(point)

    for point in _grid_points(step=16):
        add(point)
    for point in _grid_points(step=32):
        add(point)
    add((GRID_SIZE // 2, GRID_SIZE // 2))

    return candidates[:budget]


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
    reward = (3.0 * progress) + min(delta / 128.0, 1.0) * 0.1 + novelty * 0.25 + won * 2.0 - 0.01
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
