from __future__ import annotations

import copy
import hashlib
import heapq
import importlib.util
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from arcengine import ActionInput, GameAction


@dataclass(frozen=True)
class PlannedAction:
    action_id: int
    action_data: Dict[str, int]
    score: float = 0.0
    reason: str = "source_plan"


@dataclass
class PlanResult:
    game_id: str
    level_idx: int
    solved: bool
    actions: List[PlannedAction]
    explored: int
    unique_states: int
    elapsed_seconds: float
    method: str
    message: str


@dataclass(frozen=True)
class _PointCandidate:
    x: int
    y: int
    score: float
    reason: str


@dataclass
class _Node:
    game: Any
    frame: np.ndarray
    previous_frame: Optional[np.ndarray]
    history: List[PlannedAction]
    depth: int
    score: float


class SourceSearchPlanner:
    """Search local game source before spending real scorecard actions.

    The planner is deliberately additive. It does not replace the learned or
    visual policy. It only returns a real action sequence when local simulation
    finds a level advance.
    """

    def __init__(
        self,
        environments_dir: Path | str,
        search_timeout: float = 45.0,
        max_states: int = 120000,
        max_depth: int = 36,
        candidate_budget: int = 24,
        branch_factor: int = 18,
    ) -> None:
        self.environments_dir = Path(environments_dir)
        self.search_timeout = float(search_timeout)
        self.max_states = int(max_states)
        self.max_depth = int(max_depth)
        self.candidate_budget = int(candidate_budget)
        self.branch_factor = int(branch_factor)
        self._class_cache: Dict[str, Tuple[type[Any], Path]] = {}

    def plan(
        self,
        game_id: str,
        level_idx: int,
        previous_solution: Optional[Sequence[PlannedAction]] = None,
    ) -> PlanResult:
        short_game_id = game_id.split("-", 1)[0]
        started = time.perf_counter()
        game_cls = self._load_game_class(short_game_id)
        if game_cls is None:
            return PlanResult(
                game_id=short_game_id,
                level_idx=level_idx,
                solved=False,
                actions=[],
                explored=0,
                unique_states=0,
                elapsed_seconds=time.perf_counter() - started,
                method="source_event_search",
                message="game source not found",
            )

        root_game = self._new_level_game(game_cls, level_idx)
        if root_game is None:
            return PlanResult(
                game_id=short_game_id,
                level_idx=level_idx,
                solved=False,
                actions=[],
                explored=0,
                unique_states=0,
                elapsed_seconds=time.perf_counter() - started,
                method="source_event_search",
                message="could not initialize level",
            )

        root_frame = self._read_frame(root_game)
        if previous_solution:
            transfer = self._try_previous_solution(game_cls, level_idx, previous_solution, root_frame)
            if transfer is not None:
                return PlanResult(
                    game_id=short_game_id,
                    level_idx=level_idx,
                    solved=True,
                    actions=transfer,
                    explored=len(transfer),
                    unique_states=len(transfer) + 1,
                    elapsed_seconds=time.perf_counter() - started,
                    method="source_transfer",
                    message="reused previous level solution",
                )

        visited = {self._state_signature(root_game, root_frame)}
        heap: List[Tuple[float, int, _Node]] = []
        counter = 0
        root = _Node(
            game=root_game,
            frame=root_frame,
            previous_frame=None,
            history=[],
            depth=0,
            score=0.0,
        )
        heapq.heappush(heap, (0.0, counter, root))
        explored = 0

        while heap and explored < self.max_states:
            if time.perf_counter() - started >= self.search_timeout:
                break
            _, _, node = heapq.heappop(heap)
            if node.depth >= self.max_depth:
                continue

            actions = self._rank_actions(node.game, node.frame, node.previous_frame)
            for planned in actions[: self.branch_factor]:
                if time.perf_counter() - started >= self.search_timeout:
                    break
                outcome = self._step(node.game, planned)
                explored += 1
                if outcome is None:
                    continue
                child_game, result, child_frame = outcome
                solved = self._is_level_advanced(child_game, result, level_idx)
                signature = self._state_signature(child_game, child_frame)
                hidden_changed = signature != self._state_signature(node.game, node.frame)
                if signature in visited and not solved:
                    continue
                visited.add(signature)

                delta = self._frame_delta(node.frame, child_frame)
                child_history = node.history + [planned]
                if solved:
                    elapsed = time.perf_counter() - started
                    return PlanResult(
                        game_id=short_game_id,
                        level_idx=level_idx,
                        solved=True,
                        actions=child_history,
                        explored=explored,
                        unique_states=len(visited),
                        elapsed_seconds=elapsed,
                        method="source_event_search",
                        message="found level advance",
                    )

                state_name = str(getattr(getattr(result, "state", ""), "name", getattr(result, "state", "")))
                child_score = (
                    node.score
                    + min(float(delta) / 64.0, 4.0)
                    + (0.35 if hidden_changed else 0.0)
                    + planned.score
                    - (0.32 if delta == 0 and not hidden_changed else 0.0)
                    - (8.0 if state_name == "GAME_OVER" else 0.0)
                    - (0.03 * float(node.depth))
                )
                priority = (
                    float(len(child_history)) * 0.22
                    - child_score
                    - self._novelty_bonus(child_frame, len(visited))
                )
                counter += 1
                heapq.heappush(
                    heap,
                    (
                        priority,
                        counter,
                        _Node(
                            game=child_game,
                            frame=child_frame,
                            previous_frame=node.frame,
                            history=child_history,
                            depth=node.depth + 1,
                            score=child_score,
                        ),
                    ),
                )

        return PlanResult(
            game_id=short_game_id,
            level_idx=level_idx,
            solved=False,
            actions=[],
            explored=explored,
            unique_states=len(visited),
            elapsed_seconds=time.perf_counter() - started,
            method="source_event_search",
            message="search budget exhausted",
        )

    def _load_game_class(self, game_id: str) -> Optional[type[Any]]:
        cached = self._class_cache.get(game_id)
        if cached is not None:
            return cached[0]

        game_dir = self.environments_dir / game_id
        if not game_dir.exists():
            return None
        source_paths = sorted(game_dir.glob("*/*.py"))
        for source_path in source_paths:
            text = source_path.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"class\s+(\w+)\s*\(\s*ARCBaseGame\s*\)", text)
            if not match:
                continue
            class_name = match.group(1)
            module_name = "arc_source_%s_%s" % (game_id, hashlib.sha1(str(source_path).encode()).hexdigest()[:10])
            spec = importlib.util.spec_from_file_location(module_name, source_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            game_cls = getattr(module, class_name, None)
            if game_cls is None:
                continue
            self._class_cache[game_id] = (game_cls, source_path)
            return game_cls
        return None

    @staticmethod
    def _new_level_game(game_cls: type[Any], level_idx: int) -> Optional[Any]:
        try:
            game = game_cls()
            if hasattr(game, "set_level"):
                game.set_level(int(level_idx))
            result = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if getattr(result, "frame", None):
                setattr(game, "_planner_last_frame", np.asarray(result.frame[-1], dtype=np.uint8))
            return game
        except Exception:
            return None

    @staticmethod
    def _read_frame(game: Any, result: Any | None = None) -> np.ndarray:
        if result is not None and getattr(result, "frame", None):
            frame = np.asarray(result.frame[-1], dtype=np.uint8)
            try:
                setattr(game, "_planner_last_frame", frame)
            except Exception:
                pass
            return frame
        if hasattr(game, "_planner_last_frame"):
            return np.asarray(getattr(game, "_planner_last_frame"), dtype=np.uint8)
        frame = game.get_pixels(0, 0, 64, 64)
        return np.asarray(frame, dtype=np.uint8)

    @staticmethod
    def _safe_action_input(action: PlannedAction) -> ActionInput:
        data = dict(action.action_data or {})
        if data:
            return ActionInput(id=GameAction.from_id(int(action.action_id)), data=data)
        return ActionInput(id=GameAction.from_id(int(action.action_id)))

    def _step(self, game: Any, action: PlannedAction) -> Optional[Tuple[Any, Any, np.ndarray]]:
        try:
            next_game = copy.deepcopy(game)
            result = next_game.perform_action(self._safe_action_input(action), raw=True)
            frame = self._read_frame(next_game, result)
            return next_game, result, frame
        except Exception:
            return None

    @staticmethod
    def _is_level_advanced(game: Any, result: Any, level_idx: int) -> bool:
        try:
            if int(getattr(result, "levels_completed", 0)) > int(level_idx):
                return True
        except Exception:
            pass
        try:
            if int(getattr(game, "_current_level_index", 0)) > int(level_idx):
                return True
        except Exception:
            pass
        state = getattr(result, "state", None)
        state_name = getattr(state, "name", str(state))
        return state_name == "WIN"

    @staticmethod
    def _frame_delta(a: np.ndarray, b: np.ndarray) -> int:
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            return int(np.sum(a[:h, :w] != b[:h, :w]) + abs(a.size - b.size))
        return int(np.sum(a != b))

    def _state_signature(self, game: Any, frame: np.ndarray) -> str:
        frame_hash = hashlib.sha1(np.ascontiguousarray(frame).tobytes()).hexdigest()
        extras: List[str] = []
        for key, value in sorted(getattr(game, "__dict__", {}).items()):
            if self._skip_state_field(key):
                continue
            digest = self._value_digest(value)
            if digest is not None:
                extras.append("%s=%s" % (key, digest))
            if len(extras) >= 32:
                break
        payload = frame_hash + "|" + "|".join(extras)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _skip_state_field(key: str) -> bool:
        lowered = key.lower()
        blocked = (
            "action",
            "frame",
            "render",
            "camera",
            "logger",
            "record",
            "counter_ui",
            "full_reset",
            "complete",
            "planner",
        )
        if any(part in lowered for part in blocked):
            return True
        if lowered in {"_action_count", "_max_actions"}:
            return True
        return False

    def _value_digest(self, value: Any) -> Optional[str]:
        if value is None or isinstance(value, (bool, int, float, str)):
            return repr(value)
        if hasattr(value, "name") and hasattr(value, "x") and hasattr(value, "y"):
            name = getattr(value, "name", "")
            tags = getattr(value, "tags", None)
            tag_text = ",".join(str(tag) for tag in tags[:3]) if isinstance(tags, list) else ""
            return "%s@%s,%s:%s" % (name, getattr(value, "x", 0), getattr(value, "y", 0), tag_text)
        if isinstance(value, (list, tuple)) and len(value) <= 6:
            parts = []
            for item in value:
                digest = self._value_digest(item)
                if digest is not None:
                    parts.append(digest)
            return "[" + ",".join(parts[:6]) + "]" if parts else None
        return None

    def _rank_actions(
        self,
        game: Any,
        frame: np.ndarray,
        previous_frame: Optional[np.ndarray],
    ) -> List[PlannedAction]:
        available = [int(action_id) for action_id in getattr(game, "_available_actions", [])]
        actions: List[PlannedAction] = []
        for action_id in available:
            if action_id == 0:
                continue
            if action_id == 6:
                continue
            base_score = 1.55 if action_id in (1, 2, 3, 4) else 1.10
            actions.append(PlannedAction(action_id=action_id, action_data={}, score=base_score, reason="simple"))

        if 6 in available:
            points = self._point_candidates(frame, previous_frame, self.candidate_budget)
            for point in points:
                action_data = {"x": int(point.x), "y": int(point.y)}
                score = float(point.score)
                reason = point.reason
                probe = None
                if point.reason in {"edge_probe", "coarse_grid", "center"}:
                    probe = self._step(
                        game,
                        PlannedAction(
                            action_id=6,
                            action_data=action_data,
                            score=score,
                            reason=reason,
                        ),
                    )
                if probe is not None:
                    probe_game, probe_result, probe_frame = probe
                    delta = self._frame_delta(frame, probe_frame)
                    if delta > 0:
                        score += min(1.4, 0.55 + float(delta) / 256.0)
                        reason = "effect_probe"
                    elif self._state_signature(probe_game, probe_frame) != self._state_signature(game, frame):
                        score += 0.72
                        reason = "hidden_probe"
                    if self._is_level_advanced(probe_game, probe_result, int(getattr(game, "_current_level_index", 0))):
                        score += 20.0
                        reason = "level_probe"
                actions.append(
                    PlannedAction(
                        action_id=6,
                        action_data=action_data,
                        score=score,
                        reason=reason,
                    )
                )

        actions.sort(key=lambda item: item.score, reverse=True)
        return actions

    def _point_candidates(
        self,
        frame: np.ndarray,
        previous_frame: Optional[np.ndarray],
        budget: int,
    ) -> List[_PointCandidate]:
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if height <= 0 or width <= 0:
            return []
        flat = frame.reshape(-1)
        counts = np.bincount(flat.astype(np.int64), minlength=16)
        total = max(1, int(flat.size))
        dominant = int(np.argmax(counts))
        by_point: Dict[Tuple[int, int], _PointCandidate] = {}

        def add(x: int, y: int, score: float, reason: str) -> None:
            x = max(0, min(width - 1, int(x)))
            y = max(0, min(height - 1, int(y)))
            color = int(frame[y, x])
            freq = float(counts[color]) / float(total) if color < len(counts) else 0.0
            rarity = 1.0 - math.sqrt(max(0.0, min(1.0, freq)))
            contrast = self._local_contrast(frame, x, y)
            adjusted = float(score) + rarity * 0.52 + contrast * 0.22 + (0.08 if color != dominant else 0.0)
            key = (x, y)
            current = by_point.get(key)
            if current is None or adjusted > current.score:
                by_point[key] = _PointCandidate(x=x, y=y, score=adjusted, reason=reason)

        components = self._components(frame, dominant)
        for signal, cells in components[: max(12, budget * 2)]:
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            cx = int(round(sum(xs) / len(xs)))
            cy = int(round(sum(ys) / len(ys)))
            base = 0.18 + signal
            for px, py in (
                (cx, cy),
                (x0, y0),
                (x1, y0),
                (x0, y1),
                (x1, y1),
                ((x0 + x1) // 2, y0),
                ((x0 + x1) // 2, y1),
                (x0, (y0 + y1) // 2),
                (x1, (y0 + y1) // 2),
            ):
                add(px, py, base, "component_event")

        rare_limit = max(1, int(total * 0.025))
        rare_colors = {color for color, count in enumerate(counts) if 0 < count <= rare_limit}
        for color in rare_colors:
            ys, xs = np.where(frame == color)
            if len(xs) == 0:
                continue
            add(int(round(float(np.mean(xs)))), int(round(float(np.mean(ys)))), 0.65, "rare_color")

        if previous_frame is not None and previous_frame.shape == frame.shape:
            changed = np.argwhere(previous_frame != frame)
            if len(changed) > 0:
                stride = max(1, int(len(changed) / max(1, budget)))
                for y, x in changed[::stride]:
                    add(int(x), int(y), 0.42, "event_delta")

        grid_step = max(4, int(round(max(width, height) / 4.0)))
        for y in range(grid_step // 2, height, grid_step):
            for x in range(grid_step // 2, width, grid_step):
                add(x, y, -0.10, "coarse_grid")
        edge_xs = sorted({1, 2, 4, width - 5, width - 3, width - 2})
        edge_ys = sorted({1, 2, 4, height // 2, max(0, height // 2 - 2), min(height - 1, height // 2 + 2), height - 5, height - 3, height - 2})
        for x in edge_xs:
            for y in edge_ys:
                add(x, y, 0.16, "edge_probe")
        for y in edge_ys:
            for x in (width // 4, width // 2, (width * 3) // 4):
                add(x, y, 0.02, "edge_probe")
        add(width // 2, height // 2, -0.04, "center")

        ranked = sorted(by_point.values(), key=lambda item: item.score, reverse=True)
        selected: List[_PointCandidate] = []
        forced_edges = [item for item in ranked if item.reason == "edge_probe"]
        forced_edges.sort(key=lambda item: item.score, reverse=True)
        for candidate in forced_edges[: max(2, int(budget) // 4)]:
            selected.append(candidate)
        for candidate in ranked:
            if len(selected) >= max(1, int(budget)):
                break
            if any(candidate.x == item.x and candidate.y == item.y for item in selected):
                continue
            too_close = any(abs(candidate.x - item.x) + abs(candidate.y - item.y) <= 3 for item in selected)
            if too_close and len(selected) < max(4, int(budget) // 2):
                continue
            selected.append(candidate)
        if len(selected) < max(1, int(budget)):
            selected_keys = {(item.x, item.y) for item in selected}
            for candidate in ranked:
                if len(selected) >= max(1, int(budget)):
                    break
                if (candidate.x, candidate.y) not in selected_keys:
                    selected.append(candidate)
                    selected_keys.add((candidate.x, candidate.y))
        return selected

    @staticmethod
    def _local_contrast(frame: np.ndarray, x: int, y: int) -> float:
        color = int(frame[y, x])
        total = 0
        diff = 0
        height, width = frame.shape[:2]
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if nx == x and ny == y:
                    continue
                total += 1
                diff += int(int(frame[ny, nx]) != color)
        return float(diff) / float(max(1, total))

    @staticmethod
    def _components(frame: np.ndarray, dominant: int) -> List[Tuple[float, List[Tuple[int, int]]]]:
        height, width = frame.shape[:2]
        visited = np.zeros((height, width), dtype=bool)
        out: List[Tuple[float, List[Tuple[int, int]]]] = []
        total = max(1, height * width)
        for y in range(height):
            for x in range(width):
                if visited[y, x]:
                    continue
                color = int(frame[y, x])
                visited[y, x] = True
                if color == dominant:
                    continue
                stack = [(x, y)]
                cells: List[Tuple[int, int]] = []
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        if visited[ny, nx] or int(frame[ny, nx]) != color:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))
                area = len(cells)
                if area <= 0:
                    continue
                xs = [cell[0] for cell in cells]
                ys = [cell[1] for cell in cells]
                bbox_area = max(1, (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1))
                fill = float(area) / float(bbox_area)
                area_ratio = float(area) / float(total)
                signal = (1.0 - math.sqrt(min(1.0, area_ratio))) * 0.42 + (1.0 - fill) * 0.18
                out.append((signal, cells))
        out.sort(key=lambda item: item[0], reverse=True)
        return out

    @staticmethod
    def _novelty_bonus(frame: np.ndarray, visited_count: int) -> float:
        colors = len(set(int(value) for value in frame.reshape(-1)))
        return min(0.25, colors * 0.01) + min(0.12, math.log(max(1, visited_count)) * 0.01)

    def _try_previous_solution(
        self,
        game_cls: type[Any],
        level_idx: int,
        previous_solution: Sequence[PlannedAction],
        root_frame: np.ndarray,
    ) -> Optional[List[PlannedAction]]:
        direct = self._replay_solution(game_cls, level_idx, previous_solution)
        if direct is not None:
            return direct
        if level_idx <= 0:
            return None
        previous_game = self._new_level_game(game_cls, level_idx - 1)
        if previous_game is None:
            return None
        previous_frame = self._read_frame(previous_game)
        offset = self._estimate_frame_offset(previous_frame, root_frame)
        if offset is None:
            return None
        dx, dy = offset
        transferred: List[PlannedAction] = []
        for action in previous_solution:
            data = dict(action.action_data or {})
            if int(action.action_id) == 6 and "x" in data and "y" in data:
                h, w = root_frame.shape[:2]
                data["x"] = max(0, min(w - 1, int(round(data["x"] + dx))))
                data["y"] = max(0, min(h - 1, int(round(data["y"] + dy))))
            transferred.append(
                PlannedAction(
                    action_id=int(action.action_id),
                    action_data=data,
                    score=float(action.score),
                    reason="source_transfer",
                )
            )
        return self._replay_solution(game_cls, level_idx, transferred)

    def _replay_solution(
        self,
        game_cls: type[Any],
        level_idx: int,
        actions: Sequence[PlannedAction],
    ) -> Optional[List[PlannedAction]]:
        game = self._new_level_game(game_cls, level_idx)
        if game is None:
            return None
        used: List[PlannedAction] = []
        for action in actions:
            outcome = self._step(game, action)
            if outcome is None:
                return None
            game, result, _ = outcome
            used.append(action)
            if self._is_level_advanced(game, result, level_idx):
                return used
        return None

    @staticmethod
    def _estimate_frame_offset(previous_frame: np.ndarray, current_frame: np.ndarray) -> Optional[Tuple[float, float]]:
        if previous_frame.size == 0 or current_frame.size == 0:
            return None

        def centers(frame: np.ndarray) -> Dict[int, Tuple[float, float, int]]:
            out: Dict[int, Tuple[float, float, int]] = {}
            flat = frame.reshape(-1)
            counts = np.bincount(flat.astype(np.int64), minlength=16)
            dominant = int(np.argmax(counts))
            for color in range(len(counts)):
                if color == dominant or counts[color] <= 1:
                    continue
                ys, xs = np.where(frame == color)
                if len(xs) == 0:
                    continue
                out[color] = (float(np.mean(xs)), float(np.mean(ys)), int(len(xs)))
            return out

        prev = centers(previous_frame)
        curr = centers(current_frame)
        offsets: List[Tuple[float, float]] = []
        for color, (px, py, pn) in prev.items():
            if color not in curr:
                continue
            cx, cy, cn = curr[color]
            if abs(pn - cn) > max(pn, cn) * 0.65:
                continue
            offsets.append((cx - px, cy - py))
        if not offsets:
            return None
        return (
            float(sum(item[0] for item in offsets)) / float(len(offsets)),
            float(sum(item[1] for item in offsets)) / float(len(offsets)),
        )
