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
    "rtx4070super": {
        "batch_size": 96,
        "grad_accum": 2,
        "model_dim": 384,
        "num_slots": 8,
        "depth": 6,
        "num_heads": 8,
        "history": 4,
        "collect_episodes_per_game": 16,
        "beam_width": 6,
        "branch_factor": 10,
        "coord_budget": 20,
        "epochs": 16,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "online_val_every": 2,
        "decoder_dim": 64,
        "slot_iters": 1,
        "collect_workers": 16,
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


_FINAL_SUBFRAME_NON_CANONICAL_WARNED = False
_SAFE_COLOR_TYPES_SEEN: set = set()
_SAFE_COLOR_WARNS_REMAINING = 5


def _is_iterable(obj: Any) -> bool:
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes))


def _safe_color(value: Any) -> int:
    """Coerce ``value`` to an int in ``[0, 15]``. Returns 0 on any conversion failure.

    Logs each new failure type up to 5 times per process so we can see what
    arcengine is actually putting in frame cells (e.g., 'function', 'NoneType',
    custom wrapper classes). After 5 distinct failure types, further failures
    are silent — we just substitute 0.

    This is the single coercion point used by every frame-iterating helper
    (``frame_hash``, ``frame_delta``, ``non_background_density``,
    ``connected_components``, ``changed_points``, ``one_hot_frames``,
    ``pad_history``) so that NO downstream consumer crashes on bad cells.
    """
    global _SAFE_COLOR_WARNS_REMAINING
    try:
        return int(value) & 0x0F
    except (TypeError, ValueError):
        type_name = type(value).__name__
        if type_name not in _SAFE_COLOR_TYPES_SEEN and _SAFE_COLOR_WARNS_REMAINING > 0:
            _SAFE_COLOR_TYPES_SEEN.add(type_name)
            _SAFE_COLOR_WARNS_REMAINING -= 1
            print(
                "[_safe_color] non-int value of type %r in frame; substituting 0. "
                "(repr: %s)" % (type_name, repr(value)[:80]),
                flush=True,
            )
        return 0


def final_subframe(frame_stack: Sequence) -> List[List[int]]:
    """Normalize an arcengine FrameData.frame to a clean 64x64 ``List[List[int]]``.

    Defensive: if the input is not the canonical 3-D ``[[[...]]]`` shape the
    annotation promises, this function falls back to a best-effort interpretation
    rather than crashing. Specifically it handles:

    - empty ``frame_stack``  -> 64x64 of zeros
    - canonical 3-D stack    -> last 2-D frame, padded / truncated to 64x64
    - 2-D frame at top level (degenerate but observed) -> use as the frame
    - jagged or short rows   -> pad with zeros
    - non-iterable rows      -> replace with a zero row
    - out-of-range color ints -> clamp to ``[0, 15]`` via ``& 0x0F``

    Emits a one-time stderr warning per process the first time a non-canonical
    shape is encountered, so we can root-cause without spamming logs. Downstream
    callers (``frame_hash``, ``frame_delta``, ``connected_components``) can then
    assume a clean 64x64 list-of-lists.
    """
    global _FINAL_SUBFRAME_NON_CANONICAL_WARNED

    blank_row = [0] * GRID_SIZE

    def blank_grid() -> List[List[int]]:
        return [list(blank_row) for _ in range(GRID_SIZE)]

    def warn_once(reason: str) -> None:
        global _FINAL_SUBFRAME_NON_CANONICAL_WARNED
        if not _FINAL_SUBFRAME_NON_CANONICAL_WARNED:
            _FINAL_SUBFRAME_NON_CANONICAL_WARNED = True
            print(
                "[final_subframe] non-canonical frame shape (%s); padding to %dx%d. "
                "Subsequent occurrences are silenced." % (reason, GRID_SIZE, GRID_SIZE),
                flush=True,
            )

    if not frame_stack:
        return blank_grid()
    try:
        candidate = frame_stack[-1]
    except (TypeError, IndexError):
        warn_once("frame_stack[-1] failed")
        return blank_grid()
    if not _is_iterable(candidate):
        warn_once("frame_stack[-1] is %s, not iterable" % type(candidate).__name__)
        return blank_grid()

    rows = list(candidate)
    if rows and not _is_iterable(rows[0]):
        warn_once("frame_stack[-1][0] is %s, treating frame_stack as 2-D" % type(rows[0]).__name__)
        rows = list(frame_stack)
    if not rows:
        return blank_grid()

    out: List[List[int]] = []
    needed_padding = False
    for row in rows[:GRID_SIZE]:
        if not _is_iterable(row):
            out.append(list(blank_row))
            needed_padding = True
            continue
        ints: List[int] = []
        for v in row:
            try:
                ints.append(int(v) & 0x0F)
            except (TypeError, ValueError):
                ints.append(0)
                needed_padding = True
            if len(ints) >= GRID_SIZE:
                break
        if len(ints) < GRID_SIZE:
            ints.extend([0] * (GRID_SIZE - len(ints)))
            needed_padding = True
        out.append(ints)
    if len(out) < GRID_SIZE:
        for _ in range(GRID_SIZE - len(out)):
            out.append(list(blank_row))
        needed_padding = True
    if needed_padding:
        warn_once("rows or row-lengths needed padding")
    return out


def frame_hash(frame: Sequence[Sequence[int]]) -> str:
    """Hash a frame to a stable hex digest. Robust to non-int cells via _safe_color."""
    values: List[int] = []
    for row in frame:
        if not _is_iterable(row):
            continue
        for value in row:
            values.append(_safe_color(value))
    return hashlib.sha1(bytes(values)).hexdigest()


def frame_delta(prev_frame: Sequence[Sequence[int]], next_frame: Sequence[Sequence[int]]) -> int:
    """Count cells whose color changed. Robust to non-int cells via _safe_color."""
    delta = 0
    for y in range(min(len(prev_frame), len(next_frame))):
        row_a = prev_frame[y]
        row_b = next_frame[y]
        if not _is_iterable(row_a) or not _is_iterable(row_b):
            continue
        for x in range(min(len(row_a), len(row_b))):
            if _safe_color(row_a[x]) != _safe_color(row_b[x]):
                delta += 1
    return delta


def non_background_density(frame: Sequence[Sequence[int]]) -> float:
    """Fraction of cells that are non-background (color != 0). Robust to non-int cells."""
    if torch.is_tensor(frame):
        if frame.numel() == 0:
            return 0.0
        return float((frame != 0).to(dtype=torch.float32).mean().item())
    total = GRID_SIZE * GRID_SIZE
    occupied = 0
    for row in frame:
        if not _is_iterable(row):
            continue
        for value in row:
            if _safe_color(value) != 0:
                occupied += 1
    return occupied / float(total)


def _grid_points(step: int) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for y in range(step // 2, GRID_SIZE, step):
        for x in range(step // 2, GRID_SIZE, step):
            coords.append((x, y))
    return coords


def _frame_cell(frame: Sequence[Sequence[int]], y: int, x: int) -> int:
    """Read frame[y][x] coerced to a safe color int. Robust to short/missing rows."""
    try:
        row = frame[y]
    except (TypeError, IndexError):
        return 0
    if not _is_iterable(row):
        return 0
    try:
        return _safe_color(row[x])
    except (TypeError, IndexError):
        return 0


def connected_components(frame: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    """Flood-fill connected color regions. Robust to non-int cells via _frame_cell."""
    visited = [[False for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    components: List[Dict[str, Any]] = []
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            color = _frame_cell(frame, y, x)
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
                    if _frame_cell(frame, ny, nx) != color:
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
    """Coordinates where prev and next differ. Robust to non-int cells via _frame_cell."""
    points: List[Tuple[int, int]] = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if _frame_cell(prev_frame, y, x) != _frame_cell(next_frame, y, x):
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
    """One-hot encode a frame history. Robust to non-int cells (those become color 0)."""
    encoded = torch.zeros((len(history) * NUM_COLORS, GRID_SIZE, GRID_SIZE), dtype=torch.float32)
    for frame_idx, frame in enumerate(history):
        for y in range(GRID_SIZE):
            try:
                row = frame[y]
            except (TypeError, IndexError):
                continue
            if not _is_iterable(row):
                continue
            for x in range(GRID_SIZE):
                try:
                    raw = row[x]
                except (TypeError, IndexError):
                    continue
                color = _safe_color(raw)
                encoded[frame_idx * NUM_COLORS + color, y, x] = 1.0
    return encoded


def pad_history(frames: Sequence[Sequence[Sequence[int]]], history: int) -> List[List[List[int]]]:
    """Pad a frame history to ``history`` frames. Robust to non-int cells."""
    if not frames:
        blank = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        return [blank for _ in range(history)]
    stacked: List[List[List[int]]] = []
    for frame in frames[-history:]:
        rows: List[List[int]] = []
        for row in frame:
            if _is_iterable(row):
                rows.append([_safe_color(v) for v in row])
            else:
                rows.append([0 for _ in range(GRID_SIZE)])
        stacked.append(rows)
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


# Option D aux-loss helpers (BC v1).
ARCHETYPE_LABELS: Dict[str, int] = {
    "click_dominant": 0,
    "keyboard_dominant": 1,
    "mixed": 2,
}
ARCHETYPE_NAMES: List[str] = ["click_dominant", "keyboard_dominant", "mixed"]


def load_per_game_archetypes(path: Optional[Path] = None) -> Dict[str, int]:
    """Map short_id → archetype-int from per_game_priors.json. Empty dict if missing."""
    candidate = Path(path) if path is not None else Path("Local_Output") / "per_game_priors.json"
    if not candidate.exists():
        return {}
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, int] = {}
    for short_id, prior in data.items():
        archetype = str((prior or {}).get("archetype", "mixed"))
        out[str(short_id)] = ARCHETYPE_LABELS.get(archetype, ARCHETYPE_LABELS["mixed"])
    return out


def compute_saliency_mask(
    frame: Sequence[Sequence[int]],
    neighborhood: int = 1,
) -> torch.Tensor:
    """64x64 binary saliency target — same algorithm as kaggle_notebook/my_agent.py:_extract_saliency.

    Salient points: rare-color centroids + bbox corners + 4-direction offsets,
    plus 9 default fallback points (center, corners, edge midpoints). Each point
    becomes a (2*neighborhood+1) square of 1s in the mask. Default neighborhood=1
    yields ~6-10% positive pixels — a reasonable BCE target density.
    """
    rows = len(frame) if frame else 0
    cols = len(frame[0]) if rows else 0
    mask = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.float32)

    def _stamp(points: Sequence[Tuple[int, int]]) -> None:
        for x, y in points:
            for dy in range(-neighborhood, neighborhood + 1):
                for dx in range(-neighborhood, neighborhood + 1):
                    xx, yy = int(x) + dx, int(y) + dy
                    if 0 <= xx < GRID_SIZE and 0 <= yy < GRID_SIZE:
                        mask[yy, xx] = 1.0

    if rows < 1 or cols < 1:
        _stamp([(32, 32), (5, 5), (5, 58), (58, 5), (58, 58)])
        return mask

    color_counts: Dict[int, int] = {}
    color_pixels: Dict[int, List[Tuple[int, int]]] = {}
    for y in range(min(rows, GRID_SIZE)):
        row = frame[y]
        if not _is_iterable(row):
            continue
        cols_y = min(cols, GRID_SIZE, len(row))
        for x in range(cols_y):
            ci = _safe_color(row[x])
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
                ox = max(0, min(GRID_SIZE - 1, cx + dx))
                oy = max(0, min(GRID_SIZE - 1, cy + dy))
                salient.append((ox, oy))

    salient.extend([
        (32, 32), (5, 5), (5, 58), (58, 5), (58, 58),
        (32, 5), (32, 58), (5, 32), (58, 32),
    ])
    _stamp(salient)
    return mask


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
