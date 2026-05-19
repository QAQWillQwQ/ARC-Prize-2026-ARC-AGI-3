from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch
from arcengine import GameAction, GameState

from .common import (
    ACTION_IDS,
    ACTION_TO_INDEX,
    CandidateAction,
    changed_points,
    final_subframe,
    frame_delta,
    frame_hash,
    informative_subframe,
    novelty_bonus,
    one_hot_frames,
    pad_history,
    point_visual_features,
    salient_point_features,
    scalar_features,
    visual_saliency_summary,
)
from .model import build_model, load_checkpoint
from .source_planner import PlannedAction, SourceSearchPlanner

OPPOSITE_ACTION_IDS: Dict[int, int] = {1: 2, 2: 1, 3: 4, 4: 3}
MOVE_ACTION_VECTORS: Dict[int, Tuple[int, int]] = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


@dataclass
class PolicyOutput:
    action_scores: List[float]
    x_scores: List[float]
    y_scores: List[float]
    value: float


class PolicyBundle:
    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        payload = load_checkpoint(checkpoint_path, device=device)
        config = dict(payload["config"])
        self.model = build_model(config)
        self.model.load_state_dict(payload["model_state"])
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.history = int(config["history"])
        self.max_steps = int(config.get("max_steps", 192))

    @torch.no_grad()
    def infer(
        self,
        history_frames: Sequence[Sequence[Sequence[int]]],
        available_actions: Sequence[int],
        last_action_id: Optional[int],
        levels_completed: int,
        steps_since_progress: int,
        step_index: int,
    ) -> PolicyOutput:
        frames = pad_history(history_frames, self.history)
        obs = one_hot_frames(frames).unsqueeze(0).to(self.device)
        scalar = scalar_features(
            available_actions=available_actions,
            last_action_id=last_action_id,
            levels_completed=levels_completed,
            steps_since_progress=steps_since_progress,
            step_index=step_index,
            frame=frames[-1],
            max_steps=self.max_steps,
        ).unsqueeze(0).to(self.device)
        out = self.model(obs, scalar)
        return PolicyOutput(
            action_scores=out["action_logits"].softmax(dim=-1).squeeze(0).cpu().tolist(),
            x_scores=out["x_logits"].softmax(dim=-1).squeeze(0).cpu().tolist(),
            y_scores=out["y_logits"].softmax(dim=-1).squeeze(0).cpu().tolist(),
            value=float(out["value"].item()),
        )


class PolicyGuidedAgent:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        history: int = 4,
        max_steps: int = 192,
        stall_steps: int = 24,
        reset_limit: int = 4,
        coord_budget: int = 20,
        random_seed: Optional[int] = None,
        game_prior: Optional[Dict[str, Any]] = None,
        source_planner: Optional[SourceSearchPlanner] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.policy: Optional[PolicyBundle] = None
        self.history_size = history
        if checkpoint_path:
            self.policy = PolicyBundle(checkpoint_path=checkpoint_path, device=self.device)
            self.history_size = self.policy.history
            self.max_steps = self.policy.max_steps
        else:
            self.max_steps = max_steps
        self.stall_steps = stall_steps
        self.reset_limit = reset_limit
        self.coord_budget = coord_budget
        self.history_frames: Deque[List[List[int]]] = deque(maxlen=self.history_size)
        self.last_action_id: Optional[int] = None
        self.steps_since_progress = 0
        self.current_levels_completed = 0
        self.zero_delta_streak = 0
        self.low_delta_streak = 0
        self.seen_signatures: List[str] = []
        self.recent_signatures: Deque[str] = deque(maxlen=16)
        self.action_history: Deque[int] = deque(maxlen=12)
        self.coord_history: Deque[Tuple[int, int]] = deque(maxlen=12)
        self.motion_history: Deque[Tuple[float, float]] = deque(maxlen=12)
        self.delta_history: Deque[int] = deque(maxlen=12)
        self.motion_centroid: Optional[Tuple[float, float]] = None
        self.action_effect: Dict[int, float] = {action_id: 0.0 for action_id in ACTION_IDS}
        self.action_success: Dict[int, float] = {action_id: 0.0 for action_id in ACTION_IDS}
        self.tried_points: Dict[Tuple[int, int], int] = {}
        self.rng = random.Random(random_seed)
        self.game_prior = dict(game_prior or {})
        self.source_planner = source_planner
        self.source_plan_queue: Deque[CandidateAction] = deque()
        self.source_plan_level: Optional[int] = None
        self.source_attempted_levels: set[Tuple[str, int]] = set()
        self.source_previous_solutions: Dict[int, List[PlannedAction]] = {}
        raw_action_scores = dict(self.game_prior.get("action_scores") or {})
        self.action_prior_scores: Dict[int, float] = {
            int(action_id): float(score)
            for action_id, score in raw_action_scores.items()
            if int(action_id) in ACTION_IDS
        }
        self.risky_actions = {int(action_id) for action_id in self.game_prior.get("risky_actions", []) if int(action_id) in ACTION_IDS}
        self.coord_hint_points: List[Tuple[int, int]] = []
        for point in self.game_prior.get("coord_hint_points", []):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            self.coord_hint_points.append((int(point[0]), int(point[1])))
        self.coord_hint_set = set(self.coord_hint_points)

    def reset_source_plans(self) -> None:
        self.source_plan_queue.clear()
        self.source_plan_level = None
        self.source_attempted_levels = set()
        self.source_previous_solutions = {}

    def reset_memory(self, initial_frame: Sequence[Sequence[int]]) -> None:
        self.history_frames.clear()
        frame = [[int(value) for value in row] for row in initial_frame]
        for _ in range(self.history_size):
            self.history_frames.append([row[:] for row in frame])
        self.last_action_id = None
        self.steps_since_progress = 0
        self.current_levels_completed = 0
        self.zero_delta_streak = 0
        self.low_delta_streak = 0
        self.seen_signatures = [frame_hash(frame)]
        self.recent_signatures = deque([self.seen_signatures[0]], maxlen=16)
        self.action_history = deque(maxlen=12)
        self.coord_history = deque(maxlen=12)
        self.motion_history = deque(maxlen=12)
        self.delta_history = deque(maxlen=12)
        self.motion_centroid = None
        self.action_effect = {action_id: 0.0 for action_id in ACTION_IDS}
        self.action_success = {action_id: 0.0 for action_id in ACTION_IDS}
        self.tried_points = {}

    def _repeated_template_penalty(self, action_id: int) -> float:
        history = list(self.action_history)
        penalty = 0.0
        for span, weight in ((2, 0.35), (3, 0.7), (4, 1.05)):
            if len(history) < (2 * span) - 1:
                continue
            previous = history[-((2 * span) - 1) : -(span - 1)]
            current = history[-(span - 1) :] + [action_id]
            if current == previous:
                penalty += weight
        return penalty

    def _has_recent_template_loop(self, span: int) -> bool:
        history = list(self.action_history)
        if len(history) < span * 2:
            return False
        return history[-span:] == history[-(span * 2) : -span]

    def _recent_repeat_penalty(self, action_id: int) -> float:
        if not self.action_history:
            return 0.0
        history = list(self.action_history)
        penalty = 0.0

        if self.last_action_id is not None and OPPOSITE_ACTION_IDS.get(self.last_action_id) == action_id:
            penalty += 0.6 + min(self.steps_since_progress * 0.02, 0.4)

        same_tail = 0
        for existing in reversed(history):
            if existing == action_id:
                same_tail += 1
            else:
                break
        if same_tail >= 2:
            penalty += min(0.18 * float(same_tail - 1), 0.8)
            if action_id == 6:
                penalty += min(0.12 * float(same_tail - 1), 0.9)

        if len(history) >= 3 and history[-3] == history[-1] and action_id == history[-2]:
            penalty += 0.9
        if len(history) >= 5 and history[-5] == history[-3] == history[-1] and action_id == history[-2] == history[-4]:
            penalty += 1.1
        penalty += self._repeated_template_penalty(action_id)

        if self.zero_delta_streak >= 2 and history[-1] == action_id:
            penalty += min(0.22 * float(self.zero_delta_streak), 1.4)
        if self.low_delta_streak >= 4 and action_id == 6:
            penalty += min(0.18 * float(self.low_delta_streak - 3), 1.0)
        if self.current_levels_completed > 0 and self.steps_since_progress >= max(4, self.stall_steps // 4):
            if history[-1] == action_id:
                penalty += 0.3
            if action_id == 6 and self.zero_delta_streak >= 1:
                penalty += 0.45

        if self.steps_since_progress >= max(6, self.stall_steps // 3):
            penalty *= 1.35
        return penalty

    def _coord_repeat_penalty(self, point: Tuple[int, int]) -> float:
        penalty = 0.08 * float(self.tried_points.get(point, 0))
        if not self.coord_history:
            return penalty

        repeated = 0
        close_recent = 0
        for previous in self.coord_history:
            distance = abs(previous[0] - point[0]) + abs(previous[1] - point[1])
            if distance == 0:
                repeated += 1
            if distance <= 2:
                close_recent += 1
        penalty += min(0.12 * float(repeated), 0.6)
        if close_recent >= 3:
            penalty += min(0.06 * float(close_recent - 2), 0.4)
        if self.zero_delta_streak >= 2:
            penalty += min(0.08 * float(self.zero_delta_streak), 0.5)
        return penalty

    def _coord_distance_bonus(self, point: Tuple[int, int]) -> float:
        if not self.coord_history:
            return 0.0
        last_x, last_y = self.coord_history[-1]
        distance = abs(last_x - point[0]) + abs(last_y - point[1])
        return min(distance / 48.0, 0.35)

    @staticmethod
    def _centroid(points: Sequence[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
        if not points:
            return None
        return (
            float(sum(point[0] for point in points)) / float(len(points)),
            float(sum(point[1] for point in points)) / float(len(points)),
        )

    def _update_motion_memory(
        self,
        action_id: int,
        previous_frame: Sequence[Sequence[int]],
        next_frame: Sequence[Sequence[int]],
    ) -> None:
        if action_id not in MOVE_ACTION_VECTORS:
            return
        points = changed_points(previous_frame, next_frame)
        if not points or len(points) > 768:
            return
        centroid = self._centroid(points)
        if centroid is None:
            return
        self.motion_centroid = centroid
        self.motion_history.append(centroid)

    def _movement_goal_signal(
        self,
        action_id: int,
        frame: Sequence[Sequence[int]],
        prev_frame: Optional[Sequence[Sequence[int]]],
    ) -> Tuple[float, Dict[str, Any]]:
        if action_id not in MOVE_ACTION_VECTORS or self.motion_centroid is None:
            return 0.0, {}

        cx, cy = self.motion_centroid
        target_features = salient_point_features(
            frame=frame,
            prev_frame=prev_frame,
            budget=max(8, min(16, self.coord_budget)),
        )
        best_target: Optional[Dict[str, Any]] = None
        best_target_score = -1e9
        for feature in target_features:
            x = float(feature["x"])
            y = float(feature["y"])
            distance = abs(x - cx) + abs(y - cy)
            if distance < 4.0:
                continue
            point = (int(x), int(y))
            recent_penalty = 0.0
            for previous in self.coord_history:
                if abs(previous[0] - point[0]) + abs(previous[1] - point[1]) <= 2:
                    recent_penalty += 0.08
            salience = float(feature.get("salience_score", 0.0))
            target_score = salience + min(distance / 64.0, 0.35) - recent_penalty
            if target_score > best_target_score:
                best_target_score = target_score
                best_target = feature

        if best_target is None:
            return 0.0, {}

        goal_x = float(best_target["x"])
        goal_y = float(best_target["y"])
        dx = goal_x - cx
        dy = goal_y - cy
        norm = max(abs(dx) + abs(dy), 1.0)
        vx, vy = MOVE_ACTION_VECTORS[action_id]
        alignment = ((float(vx) * dx) + (float(vy) * dy)) / norm
        salience = float(best_target.get("salience_score", 0.0))
        if alignment <= 0.0:
            bonus = max(-0.18, alignment * 0.16)
        else:
            bonus = min(0.65, alignment * (0.22 + salience * 0.58))
            if self.steps_since_progress >= max(4, self.stall_steps // 4):
                bonus *= 1.2
        metadata = {
            "movement_goal": {
                "x": int(goal_x),
                "y": int(goal_y),
                "salience_score": float(best_target.get("salience_score", 0.0)),
                "color": int(best_target.get("color", 0)),
                "source": str(best_target.get("source", "")),
            },
            "motion_centroid": [round(cx, 3), round(cy, 3)],
            "movement_alignment": round(float(alignment), 6),
            "movement_goal_bonus": round(float(bonus), 6),
        }
        return float(bonus), metadata

    def _policy_output(
        self,
        available_actions: Sequence[int],
        levels_completed: int,
        step_index: int,
    ) -> Optional[PolicyOutput]:
        if self.policy is None:
            return None
        return self.policy.infer(
            history_frames=list(self.history_frames),
            available_actions=available_actions,
            last_action_id=self.last_action_id,
            levels_completed=levels_completed,
            steps_since_progress=self.steps_since_progress,
            step_index=step_index,
        )

    def rank_candidates(
        self,
        frame: Sequence[Sequence[int]],
        prev_frame: Optional[Sequence[Sequence[int]]],
        available_actions: Sequence[int],
        levels_completed: int,
        step_index: int,
    ) -> List[CandidateAction]:
        output = self._policy_output(available_actions, levels_completed, step_index)
        candidates: List[CandidateAction] = []
        available_set = set(int(action) for action in available_actions)
        movement_available = any(action_id in available_set for action_id in MOVE_ACTION_VECTORS)

        for action_id in ACTION_IDS:
            if action_id not in available_set:
                continue
            if action_id == 6:
                coord_features: List[Dict[str, Any]] = []
                seen_points = set()
                for point in self.coord_hint_points:
                    if point not in seen_points:
                        seen_points.add(point)
                        feature = point_visual_features(frame=frame, point=point, prev_frame=prev_frame)
                        feature["source"] = "probe_hint"
                        feature["salience_score"] = min(1.0, float(feature["salience_score"]) + 0.06)
                        coord_features.append(feature)
                for feature in salient_point_features(frame, prev_frame=prev_frame, budget=self.coord_budget):
                    point = (int(feature["x"]), int(feature["y"]))
                    if point not in seen_points:
                        seen_points.add(point)
                        coord_features.append(feature)
                if output is None:
                    hinted = [feature for feature in coord_features if (int(feature["x"]), int(feature["y"])) in self.coord_hint_set]
                    fallback = [feature for feature in coord_features if (int(feature["x"]), int(feature["y"])) not in self.coord_hint_set]
                    fallback.sort(
                        key=lambda feature: (
                            float(feature.get("salience_score", 0.0)),
                            float(feature.get("color_rarity", 0.0)),
                            self.rng.random() * 0.001,
                        ),
                        reverse=True,
                    )
                    coord_features = hinted + fallback
                for feature in coord_features:
                    x = int(feature["x"])
                    y = int(feature["y"])
                    visual_salience = float(feature.get("salience_score", 0.0))
                    score = 0.0
                    if output is not None:
                        score += output.action_scores[ACTION_TO_INDEX[action_id]]
                        score += output.x_scores[x] + output.y_scores[y]
                    score += self.action_prior_scores.get(action_id, 0.0) * 0.25
                    score += self.action_effect[action_id] * 0.2
                    score += self.action_success[action_id] * 0.4
                    visual_weight = 0.85 if output is None else 0.45
                    if movement_available:
                        visual_weight = 0.35 if output is None else 0.24
                        if self.action_prior_scores.get(action_id, 0.0) > 0.0 or self.action_effect[action_id] > 0.05:
                            visual_weight += 0.22
                        if self.motion_centroid is None and len(self.action_history) < 4:
                            visual_weight *= 0.55
                    score += visual_salience * visual_weight
                    score += float(feature.get("color_rarity", 0.0)) * 0.12
                    score += float(feature.get("local_contrast", 0.0)) * 0.08
                    score += float(feature.get("changed_nearby_ratio", 0.0)) * 0.16
                    score += 0.1 / float(1 + self.tried_points.get((x, y), 0))
                    score += 0.02 if int(feature.get("non_dominant", 0)) else 0.0
                    if (x, y) in self.coord_hint_set:
                        score += 0.45
                    if action_id in self.risky_actions:
                        score -= 0.5
                    score += self._coord_distance_bonus((x, y))
                    score -= self._recent_repeat_penalty(action_id)
                    score -= self._coord_repeat_penalty((x, y))
                    if output is None:
                        score += self.rng.uniform(0.0, 0.05)
                    candidates.append(
                        CandidateAction(
                            action_id=action_id,
                            action_data={"x": int(x), "y": int(y)},
                            score=score,
                            source="policy+coord" if output is not None else "coord",
                            metadata={
                                "visual_salience": visual_salience,
                                "point_visual": dict(feature),
                                "selection_mode": "visual_saliency_coord",
                                "visual_weight": round(float(visual_weight), 6),
                            },
                        )
                    )
            else:
                score = 0.0
                metadata: Dict[str, Any] = {}
                if output is not None:
                    score += output.action_scores[ACTION_TO_INDEX[action_id]]
                score += self.action_prior_scores.get(action_id, 0.0) * 0.25
                score += self.action_effect[action_id] * 0.2
                score += self.action_success[action_id] * 0.4
                if action_id in MOVE_ACTION_VECTORS and self.motion_centroid is None and len(self.action_history) < 4:
                    calibration_bonus = 0.22 - (0.03 * len(self.action_history))
                    score += calibration_bonus
                    metadata["movement_calibration_bonus"] = round(float(calibration_bonus), 6)
                movement_bonus, movement_metadata = self._movement_goal_signal(
                    action_id=action_id,
                    frame=frame,
                    prev_frame=prev_frame,
                )
                score += movement_bonus
                metadata.update(movement_metadata)
                if action_id in self.risky_actions:
                    score -= 0.5
                if self.last_action_id == action_id and self.steps_since_progress > 2:
                    score -= 0.1
                score -= self._recent_repeat_penalty(action_id)
                if output is None:
                    score += self.rng.uniform(0.0, 0.05)
                candidates.append(
                    CandidateAction(
                        action_id=action_id,
                        action_data=None,
                        score=score,
                        source="policy" if output is not None else "heuristic",
                        metadata=metadata or None,
                    )
                )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def choose_action(
        self,
        frame: Sequence[Sequence[int]],
        prev_frame: Optional[Sequence[Sequence[int]]],
        available_actions: Sequence[int],
        levels_completed: int,
        step_index: int,
    ) -> CandidateAction:
        candidates = self.rank_candidates(
            frame=frame,
            prev_frame=prev_frame,
            available_actions=available_actions,
            levels_completed=levels_completed,
            step_index=step_index,
        )
        if not candidates:
            fallback = int(available_actions[0]) if available_actions else 1
            return CandidateAction(action_id=fallback, action_data=None, score=0.0, source="fallback")
        return candidates[0]

    def update_memory(
        self,
        action: CandidateAction,
        previous_frame: Sequence[Sequence[int]],
        next_frame: Sequence[Sequence[int]],
        levels_before: int,
        levels_after: int,
        observed_delta: Optional[int] = None,
    ) -> float:
        delta = frame_delta(previous_frame, next_frame)
        if observed_delta is not None:
            delta = max(delta, int(observed_delta))
        signature = frame_hash(next_frame)
        novelty = novelty_bonus(signature, self.seen_signatures)
        repeated_recent = int(signature in self.recent_signatures)
        reverse = int(self.last_action_id is not None and OPPOSITE_ACTION_IDS.get(self.last_action_id) == action.action_id)
        template_repeat = self._repeated_template_penalty(action.action_id)
        two_cycle = int(
            len(self.action_history) >= 3
            and self.action_history[-3] == self.action_history[-1]
            and action.action_id == self.action_history[-2]
        )
        progress_gain = max(0, int(levels_after) - int(levels_before))
        stagnant = progress_gain == 0
        next_zero_delta_streak = self.zero_delta_streak + 1 if stagnant and delta == 0 else 0
        next_low_delta_streak = self.low_delta_streak + 1 if stagnant and delta <= 2 else 0
        self.seen_signatures.append(signature)
        self.recent_signatures.append(signature)
        delta_term = min(delta / 256.0, 1.0)
        utility = (
            (float(progress_gain) * 1.2)
            + (novelty * 0.18)
            + (delta_term * 0.08)
            - (0.45 if repeated_recent else 0.0)
            - (0.55 if reverse and progress_gain == 0 else 0.0)
            - (0.7 if two_cycle and progress_gain == 0 else 0.0)
            - (0.12 if delta == 0 else 0.0)
            - (0.32 if template_repeat > 0.0 and stagnant else 0.0)
        )
        if self.steps_since_progress >= max(6, self.stall_steps // 3) and progress_gain == 0:
            utility -= 0.08
        if stagnant and delta == 0:
            utility -= min(0.22 * float(next_zero_delta_streak), 1.2)
        if stagnant and delta <= 2:
            utility -= min(0.08 * float(next_low_delta_streak), 0.8)
        if int(levels_before) > 0 and stagnant:
            utility -= min(0.05 * float(max(self.steps_since_progress, 0)), 0.8)
            if delta <= 2:
                utility -= 0.22
        if action.action_id == 6 and stagnant and delta == 0:
            utility -= 1.1
        if action.action_id == 6 and action.action_data is not None:
            point = (int(action.action_data["x"]), int(action.action_data["y"]))
            utility -= self._coord_repeat_penalty(point) * 0.35
            metadata = dict(action.metadata or {})
            visual_salience = float(metadata.get("visual_salience", 0.0))
            if delta > 0 or progress_gain > 0:
                utility += min(max(visual_salience, 0.0), 1.0) * 0.12
            elif visual_salience > 0.5:
                utility -= min(visual_salience * 0.08, 0.08)
            self.coord_history.append(point)
        self._update_motion_memory(action.action_id, previous_frame, next_frame)
        self.action_effect[action.action_id] = (self.action_effect[action.action_id] * 0.7) + utility * 0.3
        if levels_after > levels_before:
            self.action_success[action.action_id] = (self.action_success[action.action_id] * 0.5) + 0.5
            self.steps_since_progress = 0
        else:
            if repeated_recent or reverse or two_cycle or template_repeat > 0.0 or (action.action_id == 6 and delta == 0):
                self.action_success[action.action_id] *= 0.9
            else:
                self.action_success[action.action_id] *= 0.97
            self.steps_since_progress += 1
        self.current_levels_completed = int(levels_after)
        self.zero_delta_streak = next_zero_delta_streak
        self.low_delta_streak = next_low_delta_streak
        if action.action_id == 6 and action.action_data is not None:
            point = (int(action.action_data["x"]), int(action.action_data["y"]))
            self.tried_points[point] = self.tried_points.get(point, 0) + 1
        self.action_history.append(action.action_id)
        self.delta_history.append(delta)
        self.last_action_id = action.action_id
        self.history_frames.append([[int(value) for value in row] for row in next_frame])
        return novelty

    def should_abort_stalled_run(self) -> bool:
        if self.current_levels_completed <= 0:
            return False
        post_progress_patience = max(8, self.stall_steps // 2)
        if self.zero_delta_streak >= max(4, post_progress_patience // 2):
            return True
        if self.low_delta_streak >= post_progress_patience:
            return True
        if self._has_recent_template_loop(4) and self.steps_since_progress >= max(6, post_progress_patience // 2):
            return True
        if self._has_recent_template_loop(3) and self.low_delta_streak >= max(4, post_progress_patience // 3):
            return True
        if len(self.action_history) >= 8 and all(action_id == 6 for action_id in list(self.action_history)[-8:]) and self.low_delta_streak >= 4:
            return True
        if self.steps_since_progress >= post_progress_patience and len(self.delta_history) >= 6:
            recent = list(self.delta_history)[-6:]
            if sum(1 for delta in recent if delta == 0) >= 4:
                return True
        return False

    def _source_plan_candidate(self, game_id: str, levels_completed: int) -> Optional[CandidateAction]:
        if self.source_planner is None:
            return None

        level_idx = int(levels_completed)
        if self.source_plan_level != level_idx:
            self.source_plan_queue.clear()
            self.source_plan_level = level_idx

        if self.source_plan_queue:
            return self.source_plan_queue.popleft()

        key = (game_id.split("-", 1)[0], level_idx)
        if key in self.source_attempted_levels:
            return None
        self.source_attempted_levels.add(key)

        previous_solution = self.source_previous_solutions.get(level_idx - 1)
        result = self.source_planner.plan(
            game_id=game_id,
            level_idx=level_idx,
            previous_solution=previous_solution,
        )
        if not result.solved or not result.actions:
            return None

        self.source_previous_solutions[level_idx] = list(result.actions)
        queue: Deque[CandidateAction] = deque()
        for index, planned in enumerate(result.actions):
            metadata = {
                "planner_method": result.method,
                "planner_message": result.message,
                "planner_level": int(result.level_idx),
                "planner_step": int(index),
                "planner_explored": int(result.explored),
                "planner_unique_states": int(result.unique_states),
                "planner_elapsed_seconds": round(float(result.elapsed_seconds), 6),
                "planned_reason": planned.reason,
            }
            queue.append(
                CandidateAction(
                    action_id=int(planned.action_id),
                    action_data=dict(planned.action_data or {}),
                    score=float(planned.score),
                    source="source_planner",
                    metadata=metadata,
                )
            )
        self.source_plan_queue = queue
        return self.source_plan_queue.popleft() if self.source_plan_queue else None

    def play_env(
        self,
        env: Any,
        game_id: str,
        baseline_actions: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        raw_obs = env.observation_space
        if raw_obs is None:
            raise RuntimeError("Environment returned no observation at start of play_env")
        frame = final_subframe(raw_obs.frame)
        self.reset_memory(frame)
        self.reset_source_plans()
        resets_used = 0
        transitions: List[Dict[str, Any]] = []

        for step_index in range(self.max_steps):
            state_name = raw_obs.state.name if hasattr(raw_obs.state, "name") else str(raw_obs.state)
            if state_name == GameState.WIN.name:
                break
            if state_name == GameState.GAME_OVER.name:
                if resets_used >= self.reset_limit:
                    break
                raw_obs = env.reset()
                resets_used += 1
                if raw_obs is None:
                    break
                frame = final_subframe(raw_obs.frame)
                self.reset_memory(frame)
                self.source_plan_queue.clear()
                self.source_plan_level = None
                continue

            prev_frame = self.history_frames[-2] if len(self.history_frames) > 1 else None
            action = self._source_plan_candidate(
                game_id=game_id,
                levels_completed=int(raw_obs.levels_completed),
            )
            if action is None:
                action = self.choose_action(
                    frame=self.history_frames[-1],
                    prev_frame=prev_frame,
                    available_actions=getattr(raw_obs, "available_actions", []),
                    levels_completed=int(raw_obs.levels_completed),
                    step_index=step_index,
                )
            game_action = GameAction.from_id(action.action_id)
            next_obs = env.step(game_action, data=action.action_data or {})
            if next_obs is None:
                break

            next_frame = final_subframe(next_obs.frame)
            event_frame, event_frame_index, event_delta_pixels = informative_subframe(
                next_obs.frame,
                reference_frame=frame,
            )
            stable_delta_pixels = frame_delta(frame, next_frame)
            delta_pixels = max(stable_delta_pixels, event_delta_pixels)
            novelty = self.update_memory(
                action=action,
                previous_frame=frame,
                next_frame=next_frame,
                levels_before=int(raw_obs.levels_completed),
                levels_after=int(next_obs.levels_completed),
                observed_delta=delta_pixels,
            )
            action_metadata = dict(action.metadata or {})
            if event_delta_pixels > stable_delta_pixels:
                action_metadata["event_frame_index"] = int(event_frame_index)
                action_metadata["event_delta_pixels"] = int(event_delta_pixels)
            transitions.append(
                {
                    "frame": [row[:] for row in frame],
                    "available_actions": list(getattr(raw_obs, "available_actions", [])),
                    "action_id": action.action_id,
                    "action_data": dict(action.action_data or {}),
                    "action_score": float(action.score),
                    "action_source": action.source,
                    "action_metadata": action_metadata,
                    "frame_visual_summary": visual_saliency_summary(frame),
                    "next_frame_visual_summary": visual_saliency_summary(next_frame),
                    "next_frame": [row[:] for row in next_frame],
                    "event_frame": [row[:] for row in event_frame] if event_delta_pixels > stable_delta_pixels else None,
                    "event_frame_index": int(event_frame_index),
                    "stable_delta_pixels": stable_delta_pixels,
                    "event_delta_pixels": event_delta_pixels,
                    "levels_before": int(raw_obs.levels_completed),
                    "levels_after": int(next_obs.levels_completed),
                    "state_before": state_name,
                    "state_after": next_obs.state.name if hasattr(next_obs.state, "name") else str(next_obs.state),
                    "delta_pixels": delta_pixels,
                    "novelty": novelty,
                    "step_index": step_index,
                }
            )
            frame = next_frame
            raw_obs = next_obs

            if self.should_abort_stalled_run():
                break

            if self.steps_since_progress >= self.stall_steps:
                if resets_used >= self.reset_limit:
                    break
                raw_obs = env.reset()
                resets_used += 1
                if raw_obs is None:
                    break
                frame = final_subframe(raw_obs.frame)
                self.reset_memory(frame)
                self.source_plan_queue.clear()
                self.source_plan_level = None

        final_state = raw_obs.state.name if hasattr(raw_obs.state, "name") else str(raw_obs.state)
        return {
            "game_id": game_id,
            "final_state": final_state,
            "levels_completed": int(raw_obs.levels_completed),
            "resets_used": resets_used,
            "actions_taken": len(transitions),
            "baseline_actions": list(baseline_actions or []),
            "transitions": transitions,
        }


def checkpoint_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
