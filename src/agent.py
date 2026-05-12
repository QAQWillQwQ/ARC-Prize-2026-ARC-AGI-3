from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
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
        # strict=False so old checkpoints (pre-aux-heads) still load. The new
        # archetype/saliency/recon heads aren't read at inference anyway —
        # PolicyBundle.infer only consumes action/x/y/value logits.
        self.model.load_state_dict(payload["model_state"], strict=False)
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
        max_episode_seconds: float = 600.0,
        # Phase α.5.2: ActionEffectDictionary lookup bonus. Built from GT
        # replay transitions; provides per-candidate (similarity * (mean_delta
        # /256 + 2*level_prob)) signal so the search prefers actions whose
        # historical effect on similar frames was large and led to level
        # progress. effect_dict_game_id, when set, restricts lookups to
        # known-public games only.
        effect_dict: Optional[Any] = None,
        effect_bonus_weight: float = 0.3,
        effect_dict_game_id: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.policy: Optional[PolicyBundle] = None
        self.history_size = history
        if checkpoint_path:
            self.policy = PolicyBundle(checkpoint_path=checkpoint_path, device=self.device)
            self.history_size = self.policy.history
        # CLI/constructor max_steps ALWAYS wins over checkpoint config — caller
        # often wants to evaluate with a different budget (e.g. 2000) than the
        # 192 baked into the saved profile. Same for stall_steps / reset_limit.
        self.max_steps = max_steps
        self.stall_steps = stall_steps
        self.reset_limit = reset_limit
        self.coord_budget = coord_budget
        # Task #19: per-episode wall-clock timeout. Defends against the
        # rescue_hard worker freeze observed on Openlab 2026-05-08, where 10
        # workers stuck for 25+ minutes producing no output. Likely cause
        # is an env.step() deadlock in arcengine on a specific game state
        # OR a heuristic interaction we haven't reproduced. Either way, a
        # hard wall budget per episode prevents an entire collect from being
        # held up by one bad seed.
        self.max_episode_seconds = float(max_episode_seconds)
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
        self.delta_history: Deque[int] = deque(maxlen=12)
        self.action_effect: Dict[int, float] = {action_id: 0.0 for action_id in ACTION_IDS}
        self.action_success: Dict[int, float] = {action_id: 0.0 for action_id in ACTION_IDS}
        self.tried_points: Dict[Tuple[int, int], int] = {}
        self.dead_click_run = 0
        self.dead_click_total = 0
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
        # Track 2: directed state-transition graph for novelty-based search.
        # Maps frame_hash -> {(action_id, x, y): next_frame_hash}. Built up
        # online during play. Used by _state_graph_novelty_bonus to bias toward
        # unexplored (state, action) pairs. Per arxiv:2512.24156, the training-
        # free 3rd-place ARC AGI 3 preview agent uses this pattern + BFS to
        # untested actions to beat frontier LLMs.
        # Task #20: optionally pre-populate the graph from a replay-derived
        # seed in game_prior. When present, the search agent starts the
        # episode already knowing the (state, action) -> next_state edges
        # observed in a known winning playthrough — it avoids reproducing
        # them and instead biases toward the still-novel actions.
        seed_graph_raw = self.game_prior.get("state_graph_seed") or {}
        self.state_graph: Dict[str, Dict[Tuple[int, int, int], str]] = {}
        if isinstance(seed_graph_raw, dict):
            for src_hash, edges in seed_graph_raw.items():
                if not isinstance(src_hash, str) or not isinstance(edges, dict):
                    continue
                normalized: Dict[Tuple[int, int, int], str] = {}
                for k, v in edges.items():
                    if not isinstance(v, str):
                        continue
                    if isinstance(k, tuple) and len(k) == 3:
                        normalized[(int(k[0]), int(k[1]), int(k[2]))] = v
                    elif isinstance(k, list) and len(k) == 3:
                        normalized[(int(k[0]), int(k[1]), int(k[2]))] = v
                if normalized:
                    self.state_graph[src_hash] = normalized
        self._current_frame_hash: Optional[str] = None
        # Track 1.6: set per-step in rank_candidates from available_actions.
        # Defaults to False so unit tests that call _recent_repeat_penalty
        # directly (without ranking first) get the click-game penalty path.
        self._is_keyboard_only: bool = False
        # Phase α.5.2: retrieval-dictionary bonus state.
        self.effect_dict: Optional[Any] = effect_dict
        self.effect_bonus_weight: float = float(effect_bonus_weight)
        self.effect_dict_game_id: Optional[str] = effect_dict_game_id

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
        self.delta_history = deque(maxlen=12)
        self.action_effect = {action_id: 0.0 for action_id in ACTION_IDS}
        self.action_success = {action_id: 0.0 for action_id in ACTION_IDS}
        self.tried_points = {}
        self.dead_click_run = 0
        self.dead_click_total = 0
        # Track 2: re-anchor the current-frame hash on reset, but do NOT clear
        # state_graph — graph edges persist across env.reset() within an
        # episode (same physical game, useful to keep what we've learned).
        self._current_frame_hash = frame_hash(frame)

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

        # Opposite-action ping-pong (left↔right, up↔down): always counted.
        # Even during warmup or on keyboard games, ABAB patterns waste the
        # action budget without exploring new state.
        if self.last_action_id is not None and OPPOSITE_ACTION_IDS.get(self.last_action_id) == action_id:
            penalty += 0.6 + min(self.steps_since_progress * 0.02, 0.4)

        # Track 1.5: exploration warmup. Skip same-tail / template penalties
        # entirely for the first 6 actions of an episode. Keyboard games may
        # need 2-3 consecutive same-direction presses before any visible delta
        # occurs — Track 1's productivity gate (`zero_delta_streak == 0`) only
        # fires AFTER the first productive step. Without warmup, the agent
        # never gets to that first productive step because each repeated
        # press accumulates penalty before the env's next-frame finally
        # registers movement. wm_full_v1_openlab and wm_full_v2_openlab both
        # showed 0/4 on keyboard-only games (g50t/ls20/tr87/wa30); this
        # warmup unlocks the early exploration window.
        warmup_active = len(history) < 6
        if warmup_active:
            return penalty

        # Track 1.6: keyboard-only softening. When ACTION6 is unavailable
        # this episode, the agent has fewer escape routes from a wrong
        # direction than in click-enabled games, and the same-tail / template
        # penalties become disproportionately punishing. Apply a 0.5x scaling
        # to those terms (opposite-action and zero_delta_streak penalties
        # remain at full strength so real loops still get caught).
        kb_softening = 0.5 if getattr(self, "_is_keyboard_only", False) else 1.0

        same_tail = 0
        for existing in reversed(history):
            if existing == action_id:
                same_tail += 1
            else:
                break
        if same_tail >= 2:
            # Gate the same-tail penalty on whether the action is currently
            # productive. Keyboard-only games (ls20, g50t, tr87, wa30) and the
            # keyboard portion of keyboard_click games require repeated same-
            # direction pressing to navigate the grid; treating that as a stuck
            # loop is the bug that produced 0/4 keyboard-game positives in
            # wm_full_v1_openlab. zero_delta_streak == 0 means the most recent
            # transition produced visible change, so the agent is making progress.
            last_was_productive = self.zero_delta_streak == 0
            if not last_was_productive:
                penalty += min(0.18 * float(same_tail - 1), 0.8) * kb_softening
                if action_id == 6:
                    penalty += min(0.12 * float(same_tail - 1), 0.9)
            elif action_id == 6:
                # ACTION6 still gets a reduced penalty even when productive —
                # repeated clicks at the same coord rarely stay productive,
                # while repeated movement keys often do.
                penalty += min(0.06 * float(same_tail - 1), 0.4)

        # Track 1 cont'd: gate the 3-window, 5-window, and template-repeat
        # penalties on productivity too. With history=[3,3,3,3,3] (keyboard
        # navigation), all three windows match trivially and add ~3.05 penalty,
        # which previously overwhelmed the heuristic stack's positive bonuses
        # (max ~0.6 from action_effect + action_success). We keep these terms
        # active when the action is NOT producing change (real ping-pong loops
        # still get caught), and apply softened versions when productive.
        last_was_productive = self.zero_delta_streak == 0
        template_pen = 0.0
        if len(history) >= 3 and history[-3] == history[-1] and action_id == history[-2]:
            template_pen += 0.9
        if len(history) >= 5 and history[-5] == history[-3] == history[-1] and action_id == history[-2] == history[-4]:
            template_pen += 1.1
        template_pen += self._repeated_template_penalty(action_id)
        if last_was_productive and history and history[-1] == action_id:
            # Pure same-action repetition while producing change. For keyboard
            # actions, this is legitimate navigation — kill the template penalty.
            # For ACTION6, keep ~30% (clicks at same coord rarely stay productive).
            template_pen *= 0.0 if action_id != 6 else 0.3
        # Track 1.6: also halve the template penalty for keyboard-only games
        template_pen *= kb_softening
        penalty += template_pen

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

    def _dead_click_penalty(self) -> float:
        # Grace period right after a level transition: don't escalate.
        if self.steps_since_progress < 4:
            return 0.0
        penalty = 0.0
        if self.dead_click_run >= 2:
            penalty += min(0.4 * float(self.dead_click_run), 2.0)
        if self.dead_click_total >= 6 and self.steps_since_progress >= 8:
            penalty += 0.6
        return penalty

    def _state_graph_novelty_bonus(self, action_id: int, x: int, y: int) -> float:
        """Track 2: novelty bonus from the directed state-transition graph.

        From the current frame_hash, give a strong bonus if `(action_id, x, y)`
        has never been tried from this state (direct novelty). If all candidates
        from this state have been tried, BFS up to depth 4 through the graph to
        find a node that still has unexplored actions; the candidate whose first
        edge lies on that BFS path receives a smaller, distance-discounted bonus.

        Inspired by the training-free 3rd-place ARC AGI 3 preview agent
        (arxiv:2512.24156), which uses exactly this pattern and beats frontier
        LLMs without any neural network.
        """
        cur_hash = self._current_frame_hash
        if cur_hash is None:
            return 0.0
        edge_key: Tuple[int, int, int] = (
            (int(action_id), int(x), int(y)) if int(action_id) == 6 else (int(action_id), 0, 0)
        )
        edges_here = self.state_graph.get(cur_hash, {})
        if edge_key not in edges_here:
            # Direct novelty: this (state, action) pair never tried.
            return 0.30
        next_hash = edges_here[edge_key]
        if next_hash == cur_hash:
            # Self-loop: action is a confirmed no-op from this state. No bonus.
            return 0.0
        # BFS from next_hash to find a state with unexplored actions.
        visited = {cur_hash, next_hash}
        queue: List[Tuple[str, int]] = [(next_hash, 1)]
        while queue:
            node_hash, depth = queue.pop(0)
            if depth > 4:
                break
            node_edges = self.state_graph.get(node_hash, {})
            # Heuristic: a state with < 4 outgoing edges almost certainly has
            # unexplored actions remaining (max action_id=7 + many ACTION6 coords).
            if len(node_edges) < 4:
                return 0.18 / float(depth)
            for child_hash in node_edges.values():
                if child_hash not in visited:
                    visited.add(child_hash)
                    queue.append((child_hash, depth + 1))
        return 0.0

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
        # Track 2: anchor the current-state hash for novelty lookups in this step.
        self._current_frame_hash = frame_hash(frame)
        # Track 1.6: detect keyboard-only games (ACTION6 unavailable). Read by
        # _recent_repeat_penalty to apply 0.5x softening on same-tail / template
        # terms so the agent has more leeway with directional pressing.
        self._is_keyboard_only = 6 not in available_set

        # Phase α.5.2: precompute the normalized 256-dim feature key for the
        # current frame once per ranking call. The retrieval-dictionary lookup
        # then reuses it across all candidates (~22 per step), avoiding the
        # cost of re-hashing the frame for every (action_id, x, y).
        effect_query: Optional[np.ndarray] = None
        if self.effect_dict is not None and self.effect_bonus_weight != 0.0:
            from .action_effect_dict import compute_feature_key
            try:
                feat = compute_feature_key(frame)
                feat_norm = float(np.linalg.norm(feat))
                if feat_norm >= 1e-8:
                    effect_query = (feat / feat_norm).astype(np.float32)
            except Exception:
                effect_query = None

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
                    score += self._state_graph_novelty_bonus(action_id, x, y)
                    score -= self._recent_repeat_penalty(action_id)
                    score -= self._coord_repeat_penalty((x, y))
                    score -= self._dead_click_penalty()
                    score += self._effect_dict_bonus(
                        frame, int(action_id), int(x), int(y), effect_query
                    )
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
                score += self._state_graph_novelty_bonus(action_id, 0, 0)
                score -= self._recent_repeat_penalty(action_id)
                score += self._effect_dict_bonus(
                    frame, int(action_id), 0, 0, effect_query
                )
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

    def _effect_dict_bonus(
        self,
        frame: Sequence[Sequence[int]],
        action_id: int,
        x: int,
        y: int,
        precomputed_query: Optional[np.ndarray],
    ) -> float:
        """Phase α.5.2: retrieval-dictionary bonus for one (action, x, y).

        Returns 0.0 when no dictionary is loaded, the bonus weight is zero, or
        the lookup found no matches. Otherwise returns

            effect_bonus_weight * similarity * (mean_delta/256.0 + 2.0 * level_prob)

        which combines visual-frame similarity, the historical pixel-change
        magnitude this action produced on similar frames, and the historical
        probability that the action led to a level-progress event.
        """
        if self.effect_dict is None or self.effect_bonus_weight == 0.0:
            return 0.0
        try:
            result = self.effect_dict.lookup(
                frame=frame,
                action_id=action_id,
                x=x,
                y=y,
                k=5,
                game_id=self.effect_dict_game_id,
                precomputed_query=precomputed_query,
            )
        except Exception:
            return 0.0
        if result.get("n_matches", 0) <= 0:
            return 0.0
        similarity = float(result.get("similarity", 0.0))
        mean_delta = float(result.get("mean_delta", 0.0))
        level_prob = float(result.get("level_prob", 0.0))
        if similarity <= 0.0:
            return 0.0
        return self.effect_bonus_weight * similarity * (mean_delta / 256.0 + 2.0 * level_prob)

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
            self.coord_history.append(point)
        self.action_effect[action.action_id] = (self.action_effect[action.action_id] * 0.7) + utility * 0.3
        if levels_after > levels_before:
            self.action_success[action.action_id] = (self.action_success[action.action_id] * 0.5) + 0.5
            self.steps_since_progress = 0
            # New level: fresh dead-click budget.
            self.dead_click_run = 0
            self.dead_click_total = 0
        else:
            if repeated_recent or reverse or two_cycle or template_repeat > 0.0 or (action.action_id == 6 and delta == 0):
                self.action_success[action.action_id] *= 0.9
            else:
                self.action_success[action.action_id] *= 0.97
            self.steps_since_progress += 1
        if action.action_id == 6:
            if delta == 0:
                self.dead_click_run += 1
                self.dead_click_total += 1
            else:
                self.dead_click_run = 0
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
        # Track 2: record the (state, action) -> next_state edge in the graph.
        new_hash = frame_hash(next_frame)
        if self._current_frame_hash is not None:
            if action.action_id == 6 and action.action_data is not None:
                edge_key = (
                    int(action.action_id),
                    int(action.action_data.get("x", 0)),
                    int(action.action_data.get("y", 0)),
                )
            else:
                edge_key = (int(action.action_id), 0, 0)
            self.state_graph.setdefault(self._current_frame_hash, {})[edge_key] = new_hash
        self._current_frame_hash = new_hash
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
        # Task #19: wall-clock budget for the whole episode. Catches cases
        # where env.step() deadlocks or our heuristic stack accidentally
        # loops forever — the play_env breaks out of the step loop and the
        # collector treats the partial trajectory as a NOT_FINISHED episode.
        episode_start_time = time.time()

        for step_index in range(self.max_steps):
            if (time.time() - episode_start_time) >= self.max_episode_seconds:
                # Wall-clock timeout. Log once per occurrence so collect_staged
                # captures the seed/game in the run log for follow-up.
                print(
                    "[agent] %s episode timed out after %.1fs (max=%.1fs) at step %d — aborting"
                    % (game_id, time.time() - episode_start_time, self.max_episode_seconds, step_index),
                    flush=True,
                )
                break
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


def checkpoint_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
