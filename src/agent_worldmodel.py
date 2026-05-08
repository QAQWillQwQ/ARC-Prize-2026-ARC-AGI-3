"""Phase 3 of worldmodel_v1: test-time planning agent.

`WorldModelAgent` shares the `play_env(env, game_id, baseline_actions)` interface
with `PolicyGuidedAgent`, but `choose_action` performs latent-space rollouts
in the trained forward dynamics model and ranks candidate first-actions by
discounted cumulative value.

Per real env step:
    state_t  = encoder(history_frames)
    For each candidate first-action c:
        next_state = forward_model(state_t, c)
        cum_value  = value(next_state)                     # gamma^0
        for h in 1 .. H-1:
            random_a   = sample uniform from available_actions
            next_state = forward_model(next_state, random_a)
            cum_value += gamma^h * value(next_state)
    Commit: argmax_c cum_value.

Real env actions still cost score; planner rollouts are free (internal).

This is intentionally a "MCTS-lite" — no tree, no UCT — because we don't
have access to the env during planning (only the learned dynamics).
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch
from arcengine import GameAction, GameState

from .common import (
    ACTION_IDS,
    ACTION_TO_INDEX,
    CandidateAction,
    final_subframe,
    frame_delta,
    salient_points,
)
from .world_model import (
    WorldModelBundle,
    encode_history_obs,
    load_world_model,
)


@dataclass
class PlannerScore:
    candidate: CandidateAction
    cum_value: float
    rollout_actions: List[int]


class WorldModelAgent:
    """Test-time planning agent for ARC AGI 3.

    Loads a `WorldModelBundle` (Phase 2 checkpoint) and uses its forward dynamics
    + value head to do K-trial × H-step latent rollouts before committing each
    real env action.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        max_steps: int = 192,
        stall_steps: int = 24,
        reset_limit: int = 4,
        coord_budget: int = 20,
        rollout_horizon: int = 5,
        rollouts_per_candidate: int = 1,
        gamma: float = 0.97,
        random_seed: Optional[int] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.bundle: WorldModelBundle = load_world_model(checkpoint_path, device=self.device)
        self.bundle.eval()
        for module in (
            self.bundle.encoder,
            self.bundle.forward_model,
            self.bundle.inverse_model,
            self.bundle.value_head,
            self.bundle.decoder,
        ):
            for p in module.parameters():
                p.requires_grad = False

        self.history_size = int(self.bundle.config["history"])
        self.num_slots = int(self.bundle.config["num_slots"])
        self.model_dim = int(self.bundle.config["model_dim"])

        self.max_steps = max_steps
        self.stall_steps = stall_steps
        self.reset_limit = reset_limit
        self.coord_budget = coord_budget
        self.rollout_horizon = max(1, rollout_horizon)
        self.rollouts_per_candidate = max(1, rollouts_per_candidate)
        self.gamma = float(gamma)

        self.history_frames: Deque[List[List[int]]] = deque(maxlen=self.history_size)
        self.last_action_id: Optional[int] = None
        self.steps_since_progress = 0
        self.current_levels_completed = 0
        self.tried_points: Dict[Tuple[int, int], int] = {}
        self.coord_history: Deque[Tuple[int, int]] = deque(maxlen=12)
        self.action_history: Deque[int] = deque(maxlen=12)
        self.delta_history: Deque[int] = deque(maxlen=12)
        self.zero_delta_streak = 0

        self.rng = random.Random(random_seed)

    def reset_memory(self, initial_frame: Sequence[Sequence[int]]) -> None:
        self.history_frames.clear()
        frame = [[int(value) for value in row] for row in initial_frame]
        for _ in range(self.history_size):
            self.history_frames.append([row[:] for row in frame])
        self.last_action_id = None
        self.steps_since_progress = 0
        self.current_levels_completed = 0
        self.tried_points = {}
        self.coord_history = deque(maxlen=12)
        self.action_history = deque(maxlen=12)
        self.delta_history = deque(maxlen=12)
        self.zero_delta_streak = 0

    @torch.no_grad()
    def _encode_current(self) -> torch.Tensor:
        """Encode the current `self.history_frames` deque to slots (1, num_slots, model_dim)."""
        obs = encode_history_obs(list(self.history_frames)).unsqueeze(0).to(self.device)
        slots = self.bundle.encoder(obs)
        return slots

    @torch.no_grad()
    def _step_dynamics(
        self,
        slots: torch.Tensor,
        action_id: int,
        x: int,
        y: int,
    ) -> torch.Tensor:
        """Apply forward dynamics for one action. slots is (1, K, D)."""
        action_idx = torch.tensor(
            [ACTION_TO_INDEX.get(int(action_id), 0)], dtype=torch.long, device=self.device
        )
        x_t = torch.tensor([int(x)], dtype=torch.long, device=self.device)
        y_t = torch.tensor([int(y)], dtype=torch.long, device=self.device)
        has_coord = torch.tensor([1.0 if int(action_id) == 6 else 0.0], device=self.device)
        return self.bundle.forward_model(slots, action_idx, x_t, y_t, has_coord)

    def _build_first_action_candidates(
        self,
        frame: Sequence[Sequence[int]],
        prev_frame: Optional[Sequence[Sequence[int]]],
        available_actions: Sequence[int],
    ) -> List[CandidateAction]:
        """Enumerate first-action candidates for the planner.

        - Each available simple action contributes one candidate.
        - ACTION6 contributes up to `coord_budget` salient-point candidates.
        """
        candidates: List[CandidateAction] = []
        available_set = {int(a) for a in available_actions}

        coord_points = salient_points(frame, prev_frame=prev_frame, budget=self.coord_budget)

        for action_id in ACTION_IDS:
            if action_id not in available_set:
                continue
            if action_id == 6:
                for x, y in coord_points:
                    candidates.append(
                        CandidateAction(
                            action_id=6,
                            action_data={"x": int(x), "y": int(y)},
                            score=0.0,
                            source="planner-coord",
                        )
                    )
            else:
                candidates.append(
                    CandidateAction(
                        action_id=action_id,
                        action_data=None,
                        score=0.0,
                        source="planner-simple",
                    )
                )
        return candidates

    def _sample_rollout_action(
        self,
        available_actions: Sequence[int],
        coord_points: Sequence[Tuple[int, int]],
    ) -> Tuple[int, int, int]:
        """Sample a rollout action uniformly from available actions; for ACTION6,
        sample uniformly from the salient-point list (or center if empty)."""
        action_id = int(self.rng.choice(list(available_actions))) if len(available_actions) else 1
        if action_id == 6:
            if coord_points:
                x, y = self.rng.choice(coord_points)
            else:
                x, y = 32, 32
        else:
            x, y = 0, 0
        return action_id, int(x), int(y)

    @torch.no_grad()
    def rank_candidates(
        self,
        frame: Sequence[Sequence[int]],
        prev_frame: Optional[Sequence[Sequence[int]]],
        available_actions: Sequence[int],
    ) -> List[PlannerScore]:
        """Run K rollouts per candidate, return scored list sorted descending."""
        candidates = self._build_first_action_candidates(frame, prev_frame, available_actions)
        if not candidates:
            return []

        slots_t = self._encode_current()
        coord_points = salient_points(frame, prev_frame=prev_frame, budget=self.coord_budget)

        scored: List[PlannerScore] = []
        for candidate in candidates:
            x = int((candidate.action_data or {}).get("x", 0))
            y = int((candidate.action_data or {}).get("y", 0))

            cum_total = 0.0
            rollout_actions_log: List[int] = []
            for _ in range(self.rollouts_per_candidate):
                state = self._step_dynamics(slots_t, candidate.action_id, x, y)
                cum_value = float(self.bundle.value_head(state).item())  # gamma^0 = 1

                for step in range(1, self.rollout_horizon):
                    a_id, ax, ay = self._sample_rollout_action(available_actions, coord_points)
                    state = self._step_dynamics(state, a_id, ax, ay)
                    cum_value += (self.gamma ** step) * float(self.bundle.value_head(state).item())
                    rollout_actions_log.append(a_id)

                cum_total += cum_value
            cum_total /= float(self.rollouts_per_candidate)

            scored.append(
                PlannerScore(
                    candidate=CandidateAction(
                        action_id=candidate.action_id,
                        action_data=dict(candidate.action_data) if candidate.action_data else None,
                        score=cum_total,
                        source=candidate.source,
                    ),
                    cum_value=cum_total,
                    rollout_actions=rollout_actions_log,
                )
            )

        scored.sort(key=lambda item: item.cum_value, reverse=True)
        return scored

    def choose_action(
        self,
        frame: Sequence[Sequence[int]],
        prev_frame: Optional[Sequence[Sequence[int]]],
        available_actions: Sequence[int],
        levels_completed: int,
        step_index: int,
    ) -> CandidateAction:
        del levels_completed, step_index  # unused; kept for interface parity
        ranked = self.rank_candidates(frame, prev_frame, available_actions)
        if not ranked:
            fallback_id = int(available_actions[0]) if available_actions else 1
            return CandidateAction(action_id=fallback_id, action_data=None, score=0.0, source="fallback")
        winner = ranked[0].candidate
        winner.score = float(ranked[0].cum_value)
        return winner

    def update_memory(
        self,
        action: CandidateAction,
        previous_frame: Sequence[Sequence[int]],
        next_frame: Sequence[Sequence[int]],
        levels_before: int,
        levels_after: int,
    ) -> float:
        delta = frame_delta(previous_frame, next_frame)
        progress_gain = max(0, int(levels_after) - int(levels_before))
        if progress_gain > 0:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
        self.current_levels_completed = int(levels_after)
        if int(action.action_id) == 6 and action.action_data is not None:
            point = (int(action.action_data["x"]), int(action.action_data["y"]))
            self.tried_points[point] = self.tried_points.get(point, 0) + 1
            self.coord_history.append(point)
        self.action_history.append(int(action.action_id))
        self.delta_history.append(int(delta))
        if delta == 0:
            self.zero_delta_streak += 1
        else:
            self.zero_delta_streak = 0
        self.last_action_id = int(action.action_id)
        self.history_frames.append([[int(value) for value in row] for row in next_frame])
        return 0.0  # planner doesn't use frame-hash novelty (use value head signal instead)

    def should_abort_stalled_run(self) -> bool:
        if self.current_levels_completed <= 0:
            return False
        post_progress_patience = max(8, self.stall_steps // 2)
        if self.zero_delta_streak >= post_progress_patience:
            return True
        if self.steps_since_progress >= post_progress_patience and len(self.delta_history) >= 6:
            recent = list(self.delta_history)[-6:]
            if sum(1 for delta in recent if delta == 0) >= 4:
                return True
        return False

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
            self.update_memory(
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
                    "novelty": 0.0,
                    "step_index": step_index,
                    "planner_score": float(action.score),
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
