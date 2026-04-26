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
    final_subframe,
    frame_delta,
    frame_hash,
    novelty_bonus,
    one_hot_frames,
    pad_history,
    salient_points,
    scalar_features,
)
from .model import build_model, load_checkpoint

OPPOSITE_ACTION_IDS: Dict[int, int] = {1: 2, 2: 1, 3: 4, 4: 3}


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
        self.seen_signatures: List[str] = []
        self.recent_signatures: Deque[str] = deque(maxlen=16)
        self.action_history: Deque[int] = deque(maxlen=12)
        self.coord_history: Deque[Tuple[int, int]] = deque(maxlen=12)
        self.action_effect: Dict[int, float] = {action_id: 0.0 for action_id in ACTION_IDS}
        self.action_success: Dict[int, float] = {action_id: 0.0 for action_id in ACTION_IDS}
        self.tried_points: Dict[Tuple[int, int], int] = {}
        self.rng = random.Random(random_seed)
        self.game_prior = dict(game_prior or {})
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

    def reset_memory(self, initial_frame: Sequence[Sequence[int]]) -> None:
        self.history_frames.clear()
        frame = [[int(value) for value in row] for row in initial_frame]
        for _ in range(self.history_size):
            self.history_frames.append([row[:] for row in frame])
        self.last_action_id = None
        self.steps_since_progress = 0
        self.seen_signatures = [frame_hash(frame)]
        self.recent_signatures = deque([self.seen_signatures[0]], maxlen=16)
        self.action_history = deque(maxlen=12)
        self.coord_history = deque(maxlen=12)
        self.action_effect = {action_id: 0.0 for action_id in ACTION_IDS}
        self.action_success = {action_id: 0.0 for action_id in ACTION_IDS}
        self.tried_points = {}

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
            penalty += min(0.15 * float(same_tail - 1), 0.45)

        if len(history) >= 3 and history[-3] == history[-1] and action_id == history[-2]:
            penalty += 0.9
        if len(history) >= 5 and history[-5] == history[-3] == history[-1] and action_id == history[-2] == history[-4]:
            penalty += 1.1

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
        return penalty

    def _coord_distance_bonus(self, point: Tuple[int, int]) -> float:
        if not self.coord_history:
            return 0.0
        last_x, last_y = self.coord_history[-1]
        distance = abs(last_x - point[0]) + abs(last_y - point[1])
        return min(distance / 48.0, 0.35)

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

        for action_id in ACTION_IDS:
            if action_id not in available_set:
                continue
            if action_id == 6:
                coord_points: List[Tuple[int, int]] = []
                seen_points = set()
                for point in self.coord_hint_points:
                    if point not in seen_points:
                        seen_points.add(point)
                        coord_points.append(point)
                for point in salient_points(frame, prev_frame=prev_frame, budget=self.coord_budget):
                    if point not in seen_points:
                        seen_points.add(point)
                        coord_points.append(point)
                if output is None:
                    hinted = [point for point in coord_points if point in self.coord_hint_set]
                    fallback = [point for point in coord_points if point not in self.coord_hint_set]
                    self.rng.shuffle(fallback)
                    coord_points = hinted + fallback
                for x, y in coord_points:
                    score = 0.0
                    if output is not None:
                        score += output.action_scores[ACTION_TO_INDEX[action_id]]
                        score += output.x_scores[x] + output.y_scores[y]
                    score += self.action_prior_scores.get(action_id, 0.0) * 0.25
                    score += self.action_effect[action_id] * 0.2
                    score += self.action_success[action_id] * 0.4
                    score += 0.1 / float(1 + self.tried_points.get((x, y), 0))
                    score += 0.02 if int(frame[y][x]) != 0 else 0.0
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
                        )
                    )
            else:
                score = 0.0
                if output is not None:
                    score += output.action_scores[ACTION_TO_INDEX[action_id]]
                score += self.action_prior_scores.get(action_id, 0.0) * 0.25
                score += self.action_effect[action_id] * 0.2
                score += self.action_success[action_id] * 0.4
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
    ) -> float:
        delta = frame_delta(previous_frame, next_frame)
        signature = frame_hash(next_frame)
        novelty = novelty_bonus(signature, self.seen_signatures)
        repeated_recent = int(signature in self.recent_signatures)
        reverse = int(self.last_action_id is not None and OPPOSITE_ACTION_IDS.get(self.last_action_id) == action.action_id)
        two_cycle = int(
            len(self.action_history) >= 3
            and self.action_history[-3] == self.action_history[-1]
            and action.action_id == self.action_history[-2]
        )
        progress_gain = max(0, int(levels_after) - int(levels_before))
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
        )
        if self.steps_since_progress >= max(6, self.stall_steps // 3) and progress_gain == 0:
            utility -= 0.08
        if action.action_id == 6 and action.action_data is not None:
            point = (int(action.action_data["x"]), int(action.action_data["y"]))
            utility -= self._coord_repeat_penalty(point) * 0.35
            self.coord_history.append(point)
        self.action_effect[action.action_id] = (self.action_effect[action.action_id] * 0.7) + utility * 0.3
        if levels_after > levels_before:
            self.action_success[action.action_id] = (self.action_success[action.action_id] * 0.5) + 0.5
            self.steps_since_progress = 0
        else:
            self.action_success[action.action_id] *= 0.94 if repeated_recent or reverse or two_cycle else 0.98
            self.steps_since_progress += 1
        if action.action_id == 6 and action.action_data is not None:
            point = (int(action.action_data["x"]), int(action.action_data["y"]))
            self.tried_points[point] = self.tried_points.get(point, 0) + 1
        self.action_history.append(action.action_id)
        self.last_action_id = action.action_id
        self.history_frames.append([[int(value) for value in row] for row in next_frame])
        return novelty

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
                continue

            prev_frame = self.history_frames[-2] if len(self.history_frames) > 1 else None
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
            novelty = self.update_memory(
                action=action,
                previous_frame=self.history_frames[-2] if len(self.history_frames) > 1 else frame,
                next_frame=next_frame,
                levels_before=int(raw_obs.levels_completed),
                levels_after=int(next_obs.levels_completed),
            )
            transitions.append(
                {
                    "frame": [row[:] for row in frame],
                    "available_actions": list(getattr(raw_obs, "available_actions", [])),
                    "action_id": action.action_id,
                    "action_data": dict(action.action_data or {}),
                    "next_frame": [row[:] for row in next_frame],
                    "levels_before": int(raw_obs.levels_completed),
                    "levels_after": int(next_obs.levels_completed),
                    "state_before": state_name,
                    "state_after": next_obs.state.name if hasattr(next_obs.state, "name") else str(next_obs.state),
                    "delta_pixels": frame_delta(frame, next_frame),
                    "novelty": novelty,
                    "step_index": step_index,
                }
            )
            frame = next_frame
            raw_obs = next_obs

            if self.steps_since_progress >= self.stall_steps:
                if resets_used >= self.reset_limit:
                    break
                raw_obs = env.reset()
                resets_used += 1
                if raw_obs is None:
                    break
                frame = final_subframe(raw_obs.frame)
                self.reset_memory(frame)

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
