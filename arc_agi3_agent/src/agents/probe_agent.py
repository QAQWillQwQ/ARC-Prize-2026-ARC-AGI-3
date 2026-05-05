"""Probe-first experimental agent.

Two phases per episode:

  probe   — for the first `probe_budget` steps (counted across env.reset()
            within the same play_env call), the agent deliberately tests every
            available non-coordinate action at least twice and samples a
            diverse set of click candidates, recording the effect signature
            of every transition.
  exploit — after probing, action and click priorities are derived from
            the probe memory: dead actions are pruned (unless a global change
            re-opens them), motion/toggle/progress signatures are rewarded,
            and clicks that landed in already-dead regions are de-prioritised.

v3 adds anti-loop logic in the exploit phase:
  - per (abstract_state, action_key) repeat penalty
  - explicit 2-cycle and 3-cycle pattern detection
  - "stale signature" penalty for actions that keep returning small_change
    or local_toggle without progress
  - stagnation detector that triggers a one-step escape biased toward
    least-tried action / coordinate
  - local follow-up search around recently promising ACTION6 clicks
  - global_change-triggered re-probe budget that temporarily un-deads
    actions that had been pruned, treating the change as an affordance shift

The agent shares the existing arc_agi environment usage but does not import
or modify the teammate's PolicyGuidedAgent.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from arcengine import GameAction, GameState

from ..utils.color_features import (
    BACKGROUND_COLOR,
    color_at,
    context_key_from_descriptor,
    local_color_descriptor,
    scene_context_key,
    top_k_colors,
    transition_color_change,
)
from ..utils.common import (
    GRID_SIZE,
    CandidateAction,
    changed_points,
    connected_components,
    final_subframe,
    frame_hash,
)
from ..utils.effect_signatures import (
    ALL_SIGNATURES,
    GAME_OVER,
    GLOBAL_CHANGE,
    LOCAL_TOGGLE,
    NO_CHANGE,
    PROGRESS,
    SMALL_CHANGE,
    classify_transition,
)
from .probe_memory import ProbeMemory

CLICK_ACTION_ID = 6
NON_COORD_ACTIONS = (1, 2, 3, 4, 5, 7)
OPPOSITE_ACTIONS: Dict[int, int] = {1: 2, 2: 1, 3: 4, 4: 3}

# Signatures we consider a "small useful effect" — repeating these without
# progress is the loop pattern we want to break.
SMALL_PROGRESSLESS_SIGS = (NO_CHANGE, SMALL_CHANGE, LOCAL_TOGGLE)
# Signatures that mark a click as "promising" — worth a local follow-up.
PROMISING_CLICK_SIGS = (LOCAL_TOGGLE, GLOBAL_CHANGE, PROGRESS)


@dataclass
class ProbeAgentConfig:
    probe_budget: int = 16
    probe_min_trials_per_action: int = 2
    probe_click_share: float = 0.4
    max_steps: int = 160
    stall_steps: int = 32
    reset_limit: int = 2
    click_candidates_per_step: int = 8
    seed: int = 0
    # ---- v3 anti-loop / exploit knobs ----
    state_action_penalty: float = 0.35
    state_action_penalty_cap: float = 1.4
    cycle_2_penalty: float = 0.7
    cycle_3_penalty: float = 0.55
    stale_signature_penalty: float = 0.3
    stale_signature_window: int = 6
    # ---- v3.1 stagnation / reprobe softening ----
    # Stagnation detector: bigger window, higher progressless ratio so we trigger less often.
    stagnation_window: int = 16
    stagnation_progressless_ratio: float = 0.90
    # Cooldown (steps) between consecutive escapes; under it the detector still fires
    # but escape_pending is suppressed and the block is counted in loop_metrics.
    stagnation_escape_cooldown: int = 8
    # In escape mode, how much of the learned priority to retain. The remaining
    # weight goes to "least visited (state, action)". Higher = gentler escape.
    escape_priority_blend: float = 0.5
    # Whether escape mode still skips actions/coords memory has tagged dead.
    # Letting escape un-dead them tends to inflate dead_action_rate.
    escape_skip_dead: bool = True
    # Reprobe: smaller per-window budget, hard cap per episode, cooldown after a
    # window closes, and gate when the agent has been making progress recently.
    global_change_reprobe_budget: int = 3
    reprobe_episode_cap: int = 8
    reprobe_cooldown_steps: int = 6
    reprobe_skip_if_recent_progress: bool = True
    recent_progress_window: int = 8
    # Bonuses on dead/low-trial actions while a reprobe window is open.
    # v3 used 0.5 / 0.2 — we soften these so dead actions don't dominate again.
    reprobe_dead_action_bonus: float = 0.25
    reprobe_low_trial_bonus: float = 0.1
    local_followup_radius: int = 6
    local_followup_window: int = 6
    local_followup_bonus: float = 1.2
    # ---- color-aware lifts (learnable, not rule-based) ----
    # Multiplier on the (action_id, scene_color) priority lift in exploit.
    # 0.0 disables color conditioning entirely; 0.3 is a soft default that
    # never dominates established action_priority.
    color_action_priority_weight: float = 0.3
    # Click-side: extra lift weight on (action6, click_dominant_color).
    color_click_priority_weight: float = 0.4
    # Local descriptor radius for click region color metadata.
    click_color_radius: int = 2


@dataclass
class _PromisingClick:
    point: Tuple[int, int]
    step_index: int
    signature: str


@dataclass
class _LoopMetrics:
    repeated_state_action_count: int = 0
    cycle_2_count: int = 0
    cycle_3_count: int = 0
    stagnation_escapes: int = 0
    local_followup_attempts: int = 0
    local_followup_successes: int = 0
    reprobe_windows_opened: int = 0
    reprobe_steps_used: int = 0
    # ---- v3.1 diagnostics ----
    # Number of steps where escape was actually taken (post-cooldown gate).
    escape_steps: int = 0
    # Of those escape steps, how many produced no_change/game_over.
    escape_dead_steps: int = 0
    # Of reprobe-step transitions, how many produced no_change/game_over.
    reprobe_dead_steps: int = 0
    # Counters for blocked attempts — useful to see whether the gates fire.
    escape_blocked_by_cooldown: int = 0
    reprobe_blocked_by_cap: int = 0
    reprobe_blocked_by_cooldown: int = 0
    reprobe_blocked_by_recent_progress: int = 0

    def to_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "repeated_state_action_count": int(self.repeated_state_action_count),
            "cycle_2_count": int(self.cycle_2_count),
            "cycle_3_count": int(self.cycle_3_count),
            "stagnation_escapes": int(self.stagnation_escapes),
            "local_followup_attempts": int(self.local_followup_attempts),
            "local_followup_successes": int(self.local_followup_successes),
            "local_followup_success_rate": (
                round(self.local_followup_successes / self.local_followup_attempts, 3)
                if self.local_followup_attempts > 0 else 0.0
            ),
            "reprobe_windows_opened": int(self.reprobe_windows_opened),
            "reprobe_steps_used": int(self.reprobe_steps_used),
            "escape_steps": int(self.escape_steps),
            "escape_dead_steps": int(self.escape_dead_steps),
            "escape_dead_action_rate": (
                round(self.escape_dead_steps / self.escape_steps, 3)
                if self.escape_steps > 0 else 0.0
            ),
            "reprobe_dead_steps": int(self.reprobe_dead_steps),
            "reprobe_dead_action_rate": (
                round(self.reprobe_dead_steps / self.reprobe_steps_used, 3)
                if self.reprobe_steps_used > 0 else 0.0
            ),
            "escape_blocked_by_cooldown": int(self.escape_blocked_by_cooldown),
            "reprobe_blocked_by_cap": int(self.reprobe_blocked_by_cap),
            "reprobe_blocked_by_cooldown": int(self.reprobe_blocked_by_cooldown),
            "reprobe_blocked_by_recent_progress": int(self.reprobe_blocked_by_recent_progress),
        }
        return out


def _bounded(point: Tuple[int, int]) -> Tuple[int, int]:
    return (
        max(0, min(GRID_SIZE - 1, int(point[0]))),
        max(0, min(GRID_SIZE - 1, int(point[1]))),
    )


def _action_key(action_id: int, action_data: Optional[Dict[str, int]]) -> Tuple[int, ...]:
    """Exact key for state-action visit tracking. Click distinguishes by coord."""
    if int(action_id) == CLICK_ACTION_ID and action_data is not None:
        return (CLICK_ACTION_ID, int(action_data.get("x", -1)), int(action_data.get("y", -1)))
    return (int(action_id),)


def _action_sig(action_id: int, action_data: Optional[Dict[str, int]]) -> Tuple[int, ...]:
    """Coarse action signature for cycle detection. Clicks bucket to 8x8."""
    if int(action_id) == CLICK_ACTION_ID and action_data is not None:
        x_bucket = int(action_data.get("x", 0)) // 8
        y_bucket = int(action_data.get("y", 0)) // 8
        return (CLICK_ACTION_ID, x_bucket, y_bucket)
    return (int(action_id),)


def _has_2_cycle(seq: Sequence[Tuple[int, ...]]) -> bool:
    if len(seq) < 4:
        return False
    a, b, c, d = seq[-4], seq[-3], seq[-2], seq[-1]
    return a != b and a == c and b == d


def _has_3_cycle(seq: Sequence[Tuple[int, ...]]) -> bool:
    if len(seq) < 6:
        return False
    s = list(seq)[-6:]
    if s[0] == s[1] or s[1] == s[2] or s[0] == s[2]:
        return False
    return s[0] == s[3] and s[1] == s[4] and s[2] == s[5]


def _color_boundary_points(
    frame: Sequence[Sequence[int]],
    components: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Tuple[int, int]]:
    """Sample points where two distinct non-background colors are adjacent."""
    out: List[Tuple[int, int]] = []
    seen: set = set()
    if not components:
        return out
    candidates: List[Tuple[int, int]] = []
    for component in components[: max(1, limit)]:
        x0, y0, x1, y1 = component["bbox"]
        for x, y in (
            (x0, y0), ((x0 + x1) // 2, y0), (x1, y0),
            (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2),
            (x0, y1), ((x0 + x1) // 2, y1), (x1, y1),
        ):
            candidates.append((int(x), int(y)))
    for cx, cy in candidates:
        cx, cy = _bounded((cx, cy))
        if (cx, cy) in seen:
            continue
        center_color = int(frame[cy][cx])
        if center_color == 0:
            continue
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            neighbour_color = int(frame[ny][nx])
            if neighbour_color != 0 and neighbour_color != center_color:
                seen.add((cx, cy))
                out.append((cx, cy))
                break
        if len(out) >= limit:
            break
    return out


def click_candidates(
    frame: Sequence[Sequence[int]],
    prev_frame: Optional[Sequence[Sequence[int]]],
    budget: int,
) -> List[Tuple[int, int]]:
    """Bounded, diversity-aware click candidate generator."""
    candidates: List[Tuple[int, int]] = []
    seen: set = set()

    def add(point: Tuple[int, int]) -> None:
        bounded = _bounded(point)
        if bounded not in seen:
            seen.add(bounded)
            candidates.append(bounded)

    components = connected_components(frame)
    top_k = max(2, min(len(components), budget))

    for component in components[:top_k]:
        add(component["center"])

    for component in components[: max(1, top_k // 2 + 1)]:
        x0, y0, x1, y1 = component["bbox"]
        for corner in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            add(corner)

    for point in _color_boundary_points(frame, components, limit=max(2, budget // 2)):
        add(point)

    if prev_frame is not None:
        delta = changed_points(prev_frame, frame)
        if delta:
            stride = max(1, len(delta) // max(1, budget // 2))
            for point in delta[::stride][: budget]:
                add(point)

    for offset in (16, 32, 48):
        add((offset, GRID_SIZE // 2))
        add((GRID_SIZE // 2, offset))
    add((GRID_SIZE // 2, GRID_SIZE // 2))

    return candidates[: max(budget * 3, 12)]


def _scene_color_key(frame: Sequence[Sequence[int]]) -> int:
    """Top-1 non-background color of the frame, or BACKGROUND if all empty."""
    top = top_k_colors(frame, k=1)
    if not top:
        return BACKGROUND_COLOR
    return int(top[0][0])


def _local_followup_points(
    center: Tuple[int, int],
    radius: int,
    used: set,
) -> List[Tuple[int, int]]:
    """Generate a small ring of candidate points around `center`."""
    cx, cy = int(center[0]), int(center[1])
    radius = max(1, int(radius))
    step = max(1, radius // 2)
    out: List[Tuple[int, int]] = []
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            if dx == 0 and dy == 0:
                continue
            point = _bounded((cx + dx, cy + dy))
            if point in used:
                continue
            used.add(point)
            out.append(point)
    return out


class ProbeFirstAgent:
    def __init__(self, config: Optional[ProbeAgentConfig] = None) -> None:
        self.config = config or ProbeAgentConfig()
        self.rng = random.Random(self.config.seed)
        self.memory = ProbeMemory()
        self.action_history: List[int] = []
        self.coord_history: List[Tuple[int, int]] = []
        self.last_action_id: Optional[int] = None
        self.steps_since_progress = 0
        self.current_levels = 0
        self.phase = "probe"
        self.reset_count = 0
        self.probe_steps_done = 0
        # ---- v3 state ----
        self.action_sig_history: Deque[Tuple[int, ...]] = deque(maxlen=12)
        self.signature_history: Deque[str] = deque(maxlen=max(12, self.config.stagnation_window))
        self.state_action_visits: Dict[Tuple[str, Tuple[int, ...]], int] = {}
        self.state_action_signatures: Dict[Tuple[str, Tuple[int, ...]], List[str]] = {}
        self.recent_promising_clicks: List[_PromisingClick] = []
        self.reprobe_budget_remaining: int = 0
        self.escape_pending: bool = False
        self.loop_metrics = _LoopMetrics()
        # ---- v3.1 gates ----
        self.last_escape_step: int = -10**9
        self.last_reprobe_close_step: int = -10**9
        self.last_progress_step: int = -10**9
        self.reprobe_steps_used_episode: int = 0
        # Bookkeeping flags for the in-flight transition.
        self._last_choice_was_followup = False
        self._step_was_reprobe = False

    # ------------------------------------------------------------------ resets

    def reset_episode(self, initial_frame: Sequence[Sequence[int]]) -> None:
        """Hard reset: called once at the start of play_env."""
        self.memory = ProbeMemory()
        self._soft_reset()
        self.probe_steps_done = 0
        self.phase = "probe"
        self.state_action_visits = {}
        self.state_action_signatures = {}
        self.recent_promising_clicks = []
        self.reprobe_budget_remaining = 0
        self.reprobe_steps_used_episode = 0
        self.last_escape_step = -10**9
        self.last_reprobe_close_step = -10**9
        self.last_progress_step = -10**9
        self.loop_metrics = _LoopMetrics()
        self.action_sig_history.clear()
        self.signature_history.clear()
        self.escape_pending = False
        self._step_was_reprobe = False

    def _soft_reset(self) -> None:
        """Light reset for env.reset() within an episode — keeps memory."""
        self.action_history = []
        self.coord_history = []
        self.last_action_id = None
        self.steps_since_progress = 0
        self.current_levels = 0
        re_probe = max(2, self.config.probe_budget // 4)
        self.probe_steps_done = max(0, self.probe_steps_done - re_probe)
        # Cycle history is per-trajectory; flush it on env.reset.
        self.action_sig_history.clear()
        self.signature_history.clear()
        self.recent_promising_clicks = []
        self.escape_pending = False

    # ---------------------------------------------------------- v3 helpers

    def _is_stagnating(self) -> bool:
        window = int(self.config.stagnation_window)
        if len(self.signature_history) < window:
            return False
        recent = list(self.signature_history)[-window:]
        if any(sig == PROGRESS for sig in recent):
            return False
        progressless = sum(1 for sig in recent if sig in SMALL_PROGRESSLESS_SIGS)
        return progressless / float(window) >= self.config.stagnation_progressless_ratio

    def _state_action_repeat_penalty(self, state_hash: str, action_key: Tuple[int, ...]) -> float:
        visits = self.state_action_visits.get((state_hash, action_key), 0)
        if visits <= 0:
            return 0.0
        raw = self.config.state_action_penalty * float(visits)
        return min(raw, self.config.state_action_penalty_cap)

    def _stale_signature_penalty(self, state_hash: str, action_key: Tuple[int, ...]) -> float:
        history = self.state_action_signatures.get((state_hash, action_key))
        if not history:
            return 0.0
        window = list(history)[-self.config.stale_signature_window:]
        if not window:
            return 0.0
        small = sum(1 for sig in window if sig in SMALL_PROGRESSLESS_SIGS)
        if small == len(window) and small >= 2:
            return self.config.stale_signature_penalty * float(small)
        return 0.0

    def _cycle_penalty_for(self, action_sig: Tuple[int, ...]) -> float:
        if len(self.action_sig_history) < 3:
            return 0.0
        # Probe what the trailing 3 / 5 entries plus this candidate would look like.
        seq2 = list(self.action_sig_history)[-3:] + [action_sig]
        seq3 = list(self.action_sig_history)[-5:] + [action_sig]
        penalty = 0.0
        if _has_2_cycle(seq2):
            penalty += self.config.cycle_2_penalty
        if _has_3_cycle(seq3):
            penalty += self.config.cycle_3_penalty
        return penalty

    def _active_promising_click(self, step_index: int) -> Optional[_PromisingClick]:
        window = int(self.config.local_followup_window)
        # Prune stale entries.
        self.recent_promising_clicks = [
            pc for pc in self.recent_promising_clicks
            if step_index - pc.step_index <= window
        ]
        if not self.recent_promising_clicks:
            return None
        return self.recent_promising_clicks[-1]

    def _record_promising_click(self, point: Tuple[int, int], signature: str, step_index: int) -> None:
        self.recent_promising_clicks.append(_PromisingClick(point=point, step_index=step_index, signature=signature))
        # Keep the list bounded.
        if len(self.recent_promising_clicks) > 6:
            self.recent_promising_clicks = self.recent_promising_clicks[-6:]

    def _open_reprobe_window(self, step_index: int) -> None:
        budget = int(self.config.global_change_reprobe_budget)
        if budget <= 0:
            return
        if self.reprobe_steps_used_episode >= int(self.config.reprobe_episode_cap):
            self.loop_metrics.reprobe_blocked_by_cap += 1
            return
        if step_index - self.last_reprobe_close_step < int(self.config.reprobe_cooldown_steps):
            self.loop_metrics.reprobe_blocked_by_cooldown += 1
            return
        if self.config.reprobe_skip_if_recent_progress:
            if step_index - self.last_progress_step < int(self.config.recent_progress_window):
                self.loop_metrics.reprobe_blocked_by_recent_progress += 1
                return
        if self.reprobe_budget_remaining < budget:
            self.loop_metrics.reprobe_windows_opened += 1
        self.reprobe_budget_remaining = max(self.reprobe_budget_remaining, budget)

    # ---------------------------------------------------------------- selection

    def _pick_unexplored_click(
        self,
        frame: Sequence[Sequence[int]],
        prev_frame: Optional[Sequence[Sequence[int]]],
    ) -> Optional[Tuple[int, int]]:
        candidates = click_candidates(frame, prev_frame, budget=self.config.click_candidates_per_step)
        best: Optional[Tuple[int, int]] = None
        best_score = float("-inf")
        for point in candidates:
            coord_record = self.memory.coords.get(point)
            if coord_record is not None and coord_record.is_dead:
                continue
            tried = coord_record.trials if coord_record else 0
            score = -float(tried)
            x, y = point
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and int(frame[y][x]) != 0:
                score += 1.0
            score += self.rng.uniform(0.0, 0.05)
            if score > best_score:
                best_score = score
                best = point
        return best

    def _select_probe_action(
        self,
        frame: Sequence[Sequence[int]],
        prev_frame: Optional[Sequence[Sequence[int]]],
        available_actions: Sequence[int],
    ) -> CandidateAction:
        available_set = {int(a) for a in available_actions}
        non_coord = [a for a in NON_COORD_ACTIONS if a in available_set]
        click_available = CLICK_ACTION_ID in available_set

        least_tested: Optional[int] = None
        least_trials = self.config.probe_min_trials_per_action
        for action_id in non_coord:
            trials = self.memory.trial_count(action_id)
            if trials < least_trials:
                least_tested = action_id
                least_trials = trials
        if least_tested is not None:
            return CandidateAction(
                action_id=int(least_tested),
                action_data=None,
                source="probe_action",
                score=0.0,
            )

        non_coord_trials = sum(self.memory.trial_count(a) for a in non_coord)
        click_trials = self.memory.trial_count(CLICK_ACTION_ID) if click_available else 0
        target_action_steps = max(
            int(self.config.probe_budget * (1.0 - self.config.probe_click_share)),
            len(non_coord) * self.config.probe_min_trials_per_action,
        )

        if click_available and (non_coord_trials >= target_action_steps or not non_coord):
            point = self._pick_unexplored_click(frame, prev_frame)
            if point is not None:
                return CandidateAction(
                    action_id=CLICK_ACTION_ID,
                    action_data={"x": int(point[0]), "y": int(point[1])},
                    source="probe_click",
                    score=0.0,
                )

        if non_coord:
            ranked = sorted(non_coord, key=lambda a: self.memory.trial_count(a))
            return CandidateAction(
                action_id=int(ranked[0]),
                action_data=None,
                source="probe_action_extra",
                score=0.0,
            )

        if click_available:
            point = self._pick_unexplored_click(frame, prev_frame) or (GRID_SIZE // 2, GRID_SIZE // 2)
            return CandidateAction(
                action_id=CLICK_ACTION_ID,
                action_data={"x": int(point[0]), "y": int(point[1])},
                source="probe_click_fallback",
                score=0.0,
            )

        if available_actions:
            return CandidateAction(action_id=int(available_actions[0]), action_data=None, source="probe_fallback", score=0.0)
        return CandidateAction(action_id=1, action_data=None, source="probe_fallback", score=0.0)

    def _select_exploit_action(
        self,
        frame: Sequence[Sequence[int]],
        prev_frame: Optional[Sequence[Sequence[int]]],
        available_actions: Sequence[int],
        step_index: int,
        state_hash: str,
        scene_color: int,
        scene_ctx: Tuple[int, int],
    ) -> CandidateAction:
        available_set = {int(a) for a in available_actions}
        reprobe_active = self.reprobe_budget_remaining > 0
        global_change_recent = reprobe_active or self.memory.has_recent_global_change(since_step=step_index - 4)
        stagnating = self._is_stagnating()
        if stagnating:
            cooldown = int(self.config.stagnation_escape_cooldown)
            if step_index - self.last_escape_step < cooldown:
                self.loop_metrics.escape_blocked_by_cooldown += 1
            else:
                self.loop_metrics.stagnation_escapes += 1
                self.escape_pending = True
                self.last_escape_step = step_index

        promising_click = self._active_promising_click(step_index)
        # In escape mode, optionally still skip already-dead actions/coords so the
        # escape doesn't manufacture dead steps.
        escape_skip_dead = self.escape_pending and bool(self.config.escape_skip_dead)
        scored: List[Tuple[float, CandidateAction, Tuple[int, ...]]] = []

        # ------- non-coord actions -------
        for action_id in NON_COORD_ACTIONS:
            if action_id not in available_set:
                continue
            action_is_dead = self.memory.is_action_dead(action_id)
            if action_is_dead and not global_change_recent:
                continue
            # In escape mode, optionally still skip dead actions even if reprobe
            # is active. v3 didn't and that drove dead_action_rate up.
            if escape_skip_dead and action_is_dead:
                continue

            priority = self.memory.action_priority(action_id)
            action_key = (int(action_id),)
            action_sig = (int(action_id),)

            # Anti-loop adjustments.
            priority -= self._state_action_repeat_penalty(state_hash, action_key)
            priority -= self._stale_signature_penalty(state_hash, action_key)
            priority -= self._cycle_penalty_for(action_sig)

            if self.last_action_id == action_id:
                priority -= 0.2
            opp = OPPOSITE_ACTIONS.get(self.last_action_id) if self.last_action_id is not None else None
            if opp is not None and opp == action_id:
                priority -= 0.4
            if len(self.action_history) >= 4:
                tail = list(self.action_history)[-4:]
                if tail == [tail[0], tail[1], tail[0], tail[1]] and action_id in (tail[0], tail[1]):
                    priority -= 0.6

            # Reprobe bonus: dead/low-trial actions get a small lift while the
            # window is open so they get re-tested. v3.1 softens these.
            if reprobe_active:
                if action_is_dead:
                    priority += float(self.config.reprobe_dead_action_bonus)
                elif self.memory.trial_count(action_id) <= 2:
                    priority += float(self.config.reprobe_low_trial_bonus)

            # Color-aware learned lift. Pure statistical: zero until evidence
            # accumulates for this (action_id, scene_color, scene_ctx) triple.
            # Context-conditioned lookup means same color in different scenes
            # can carry different beliefs without us writing per-scene rules.
            if self.config.color_action_priority_weight > 0.0:
                priority += float(self.config.color_action_priority_weight) * (
                    self.memory.color_action_priority(action_id, scene_color, scene_ctx)
                )

            jitter_range = 0.4 if self.escape_pending else 0.05
            jitter = self.rng.uniform(0.0, jitter_range)

            # Escape mode: blend learned priority with -visits instead of nuking it.
            if self.escape_pending:
                visits = self.state_action_visits.get((state_hash, action_key), 0)
                blend = float(self.config.escape_priority_blend)
                priority = blend * priority - (1.0 - blend) * float(visits)

            scored.append((
                priority + jitter,
                CandidateAction(
                    action_id=int(action_id),
                    action_data=None,
                    source="exploit_action" if not self.escape_pending else "exploit_escape_action",
                    score=float(priority),
                ),
                action_sig,
            ))

        # ------- click candidates -------
        click_baseline = self.memory.action_priority(CLICK_ACTION_ID) if CLICK_ACTION_ID in available_set else 0.0
        click_dead = self.memory.is_action_dead(CLICK_ACTION_ID) and not global_change_recent

        if CLICK_ACTION_ID in available_set and not click_dead:
            base_points = click_candidates(frame, prev_frame, budget=self.config.click_candidates_per_step)
            seen_points = set(base_points)

            # Local follow-up: extra points near a recently promising click.
            followup_points: List[Tuple[int, int]] = []
            if promising_click is not None:
                followup_points = _local_followup_points(
                    promising_click.point,
                    radius=self.config.local_followup_radius,
                    used=seen_points,
                )

            all_points: List[Tuple[Tuple[int, int], bool]] = (
                [(p, False) for p in base_points] + [(p, True) for p in followup_points]
            )

            for point, is_followup in all_points:
                coord_record = self.memory.coords.get(point)
                coord_is_dead = coord_record is not None and coord_record.is_dead
                if coord_is_dead and not global_change_recent:
                    continue
                # In escape mode, optionally still skip dead coords (v3 didn't,
                # which inflated dead_action_rate when escape and reprobe overlapped).
                if escape_skip_dead and coord_is_dead:
                    continue
                # Avoid exact dead-click repeats even during reprobe — only
                # un-dead if we haven't already replayed it this trajectory.
                if (
                    coord_is_dead
                    and reprobe_active
                    and self.coord_history
                    and self.coord_history[-1] == point
                ):
                    continue

                score = click_baseline + self.memory.coord_priority(point, frame)
                tried = coord_record.trials if coord_record else 0
                score -= 0.3 * float(tried)
                if self.coord_history and self.coord_history[-1] == point:
                    score -= 0.6

                # State-action repeat penalty.
                action_key = (CLICK_ACTION_ID, int(point[0]), int(point[1]))
                score -= self._state_action_repeat_penalty(state_hash, action_key)
                score -= self._stale_signature_penalty(state_hash, action_key)

                # Cycle penalty using bucketed signature.
                action_sig = (CLICK_ACTION_ID, int(point[0]) // 8, int(point[1]) // 8)
                score -= self._cycle_penalty_for(action_sig)

                if is_followup:
                    score += self.config.local_followup_bonus
                    # If the seed click was a global_change, extra weight — affordance shifted nearby.
                    if promising_click is not None and promising_click.signature == GLOBAL_CHANGE:
                        score += 0.4

                # Reprobe lift: bias toward less-tried coords while a window is open,
                # but softer than v3 to avoid amplifying NO_CHANGE clicks.
                if reprobe_active and tried <= 1:
                    score += float(self.config.reprobe_low_trial_bonus)

                # Color-aware click lift: condition on the local dominant color
                # AND the local descriptor's context key (secondary color +
                # density bucket). Same-color/different-context regions carry
                # independent beliefs that the agent learns from interaction.
                if self.config.color_click_priority_weight > 0.0:
                    descriptor = local_color_descriptor(
                        frame, point, radius=int(self.config.click_color_radius)
                    )
                    click_color = int(descriptor.get("dominant_color", BACKGROUND_COLOR))
                    click_ctx = context_key_from_descriptor(descriptor)
                    score += float(self.config.color_click_priority_weight) * (
                        self.memory.color_action_priority(
                            CLICK_ACTION_ID, click_color, click_ctx,
                        )
                    )

                jitter_range = 0.4 if self.escape_pending else 0.05
                jitter = self.rng.uniform(0.0, jitter_range)

                if self.escape_pending:
                    visits = self.state_action_visits.get((state_hash, action_key), 0)
                    blend = float(self.config.escape_priority_blend)
                    score = blend * score - (1.0 - blend) * float(visits)

                scored.append((
                    score + jitter,
                    CandidateAction(
                        action_id=CLICK_ACTION_ID,
                        action_data={"x": int(point[0]), "y": int(point[1])},
                        source="exploit_click_followup" if is_followup else (
                            "exploit_escape_click" if self.escape_pending else "exploit_click"
                        ),
                        score=float(score),
                    ),
                    action_sig,
                ))

        if not scored:
            for action_id in NON_COORD_ACTIONS:
                if action_id in available_set:
                    return CandidateAction(action_id=int(action_id), action_data=None, source="exploit_fallback", score=0.0)
            if CLICK_ACTION_ID in available_set:
                point = self._pick_unexplored_click(frame, prev_frame) or (GRID_SIZE // 2, GRID_SIZE // 2)
                return CandidateAction(
                    action_id=CLICK_ACTION_ID,
                    action_data={"x": int(point[0]), "y": int(point[1])},
                    source="exploit_fallback_click",
                    score=0.0,
                )
            if available_actions:
                return CandidateAction(action_id=int(available_actions[0]), action_data=None, source="exploit_fallback", score=0.0)
            return CandidateAction(action_id=1, action_data=None, source="exploit_fallback", score=0.0)

        scored.sort(key=lambda item: item[0], reverse=True)
        # Track whether the chosen click is a follow-up — used for metrics.
        chosen = scored[0][1]
        self._last_choice_was_followup = chosen.source == "exploit_click_followup"
        return chosen

    # ----------------------------------------------------------------- driver

    def play_env(
        self,
        env: Any,
        game_id: str,
        baseline_actions: Optional[Sequence[int]] = None,
        prior: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_obs = env.observation_space
        if raw_obs is None:
            raise RuntimeError("Environment returned no initial observation")
        frame = final_subframe(raw_obs.frame)
        self.reset_episode(frame)
        if prior:
            self.memory.warm_start_from_prior(prior)
        prev_frame = frame
        transitions: List[Dict[str, Any]] = []

        for step_index in range(self.config.max_steps):
            state_name = raw_obs.state.name if hasattr(raw_obs.state, "name") else str(raw_obs.state)
            if state_name == GameState.WIN.name:
                break
            if state_name == GameState.GAME_OVER.name:
                if self.reset_count >= self.config.reset_limit:
                    break
                raw_obs = env.reset()
                if raw_obs is None:
                    break
                self.reset_count += 1
                frame = final_subframe(raw_obs.frame)
                prev_frame = frame
                self._soft_reset()
                continue

            available_actions = list(getattr(raw_obs, "available_actions", []))

            self.phase = "probe" if self.probe_steps_done < self.config.probe_budget else "exploit"
            state_hash_before = frame_hash(frame)
            scene_color = _scene_color_key(frame)
            scene_ctx = scene_context_key(frame)

            self._last_choice_was_followup = False
            self._step_was_reprobe = bool(self.reprobe_budget_remaining > 0 and self.phase == "exploit")
            if self.phase == "probe":
                choice = self._select_probe_action(frame, prev_frame, available_actions)
            else:
                choice = self._select_exploit_action(
                    frame, prev_frame, available_actions, step_index,
                    state_hash_before, scene_color, scene_ctx,
                )

            action_id = int(choice.action_id)
            action_data = dict(choice.action_data or {})

            next_obs = env.step(GameAction.from_id(action_id), data=action_data)
            if next_obs is None:
                break
            next_frame = final_subframe(next_obs.frame)
            next_state_name = next_obs.state.name if hasattr(next_obs.state, "name") else str(next_obs.state)
            levels_before = int(raw_obs.levels_completed)
            levels_after = int(next_obs.levels_completed)

            record = classify_transition(
                prev_frame=frame,
                next_frame=next_frame,
                levels_before=levels_before,
                levels_after=levels_after,
                state_after=next_state_name,
            )
            self.memory.record(
                action_id=action_id,
                action_data=action_data if action_id == CLICK_ACTION_ID else None,
                signature=record.signature,
                frame_before=frame,
                step_index=step_index,
            )

            # Color-aware bookkeeping. For clicks we use the local dominant
            # color + local context key; for non-coord actions we use the
            # frame's top non-bg color + scene-level context key. observe_color
            # decides if this triple is NEW or OLD, reinforces or splits the
            # learned hypothesis accordingly, and reports back what happened.
            click_color_key: Optional[int] = None
            click_descriptor: Optional[Dict[str, Any]] = None
            color_for_obs: int
            ctx_for_obs: Tuple[int, int]
            if action_id == CLICK_ACTION_ID:
                point_xy = (int(action_data.get("x", 0)), int(action_data.get("y", 0)))
                click_descriptor = local_color_descriptor(
                    frame, point_xy, radius=int(self.config.click_color_radius)
                )
                click_color_key = int(click_descriptor.get("dominant_color", BACKGROUND_COLOR))
                color_for_obs = click_color_key
                ctx_for_obs = context_key_from_descriptor(click_descriptor)
            else:
                color_for_obs = int(scene_color)
                ctx_for_obs = scene_ctx
            color_observation = self.memory.observe_color(
                action_id, color_for_obs, ctx_for_obs, record.signature,
            )

            color_change = transition_color_change(frame, next_frame)
            transition_payload = {
                "frame": [row[:] for row in frame],
                "available_actions": available_actions,
                "action_id": action_id,
                "action_data": action_data,
                "next_frame": [row[:] for row in next_frame],
                "levels_before": levels_before,
                "levels_after": levels_after,
                "state_before": state_name,
                "state_after": next_state_name,
                "delta_pixels": int(record.delta_pixels),
                "step_index": int(step_index),
                "phase": self.phase,
                "effect_signature": record.signature,
                "effect": record.to_dict(),
                "source": choice.source,
                "color_summary": {
                    "scene_color": int(scene_color),
                    "scene_context_key": list(scene_ctx),
                    "top_colors": [
                        [int(c), float(s)] for c, s in top_k_colors(frame, k=4)
                    ],
                    "click_color": click_color_key,
                    "click_context_key": (
                        list(ctx_for_obs) if action_id == CLICK_ACTION_ID else None
                    ),
                    "click_descriptor": click_descriptor,
                    "change_label": color_change.get("change_label"),
                    "colors_changed": color_change.get("colors_changed", []),
                    "per_color_delta": color_change.get("per_color_delta", {}),
                    "observation": color_observation.to_dict(),
                },
            }
            transitions.append(transition_payload)

            # ---- v3 bookkeeping ----
            action_key = _action_key(action_id, action_data if action_id == CLICK_ACTION_ID else None)
            sa_key = (state_hash_before, action_key)
            prev_visits = self.state_action_visits.get(sa_key, 0)
            if prev_visits > 0:
                self.loop_metrics.repeated_state_action_count += 1
            self.state_action_visits[sa_key] = prev_visits + 1
            sig_history = self.state_action_signatures.setdefault(sa_key, [])
            sig_history.append(record.signature)
            if len(sig_history) > self.config.stale_signature_window * 2:
                self.state_action_signatures[sa_key] = sig_history[-self.config.stale_signature_window * 2:]

            action_sig = _action_sig(action_id, action_data if action_id == CLICK_ACTION_ID else None)
            seq2 = list(self.action_sig_history)[-3:] + [action_sig]
            seq3 = list(self.action_sig_history)[-5:] + [action_sig]
            if _has_2_cycle(seq2):
                self.loop_metrics.cycle_2_count += 1
            if _has_3_cycle(seq3):
                self.loop_metrics.cycle_3_count += 1
            self.action_sig_history.append(action_sig)
            self.signature_history.append(record.signature)

            if action_id == CLICK_ACTION_ID:
                point = (int(action_data.get("x", 0)), int(action_data.get("y", 0)))
                if record.signature in PROMISING_CLICK_SIGS:
                    self._record_promising_click(point, record.signature, step_index)
                if self._last_choice_was_followup:
                    self.loop_metrics.local_followup_attempts += 1
                    if record.signature in PROMISING_CLICK_SIGS:
                        self.loop_metrics.local_followup_successes += 1

            # ---- v3.1 escape / reprobe outcome tracking ----
            sig_was_dead = record.signature in (NO_CHANGE, GAME_OVER)
            if self.escape_pending:
                self.loop_metrics.escape_steps += 1
                if sig_was_dead:
                    self.loop_metrics.escape_dead_steps += 1
            if self._step_was_reprobe:
                if sig_was_dead:
                    self.loop_metrics.reprobe_dead_steps += 1

            if record.progress_gain > 0:
                self.last_progress_step = int(step_index)

            if record.signature == GLOBAL_CHANGE:
                self._open_reprobe_window(step_index=step_index)

            if self.reprobe_budget_remaining > 0 and self.phase == "exploit":
                self.reprobe_budget_remaining -= 1
                self.reprobe_steps_used_episode += 1
                self.loop_metrics.reprobe_steps_used += 1
                if self.reprobe_budget_remaining == 0:
                    self.last_reprobe_close_step = int(step_index)

            self.escape_pending = False
            self._step_was_reprobe = False

            self.action_history.append(action_id)
            if action_id == CLICK_ACTION_ID:
                self.coord_history.append((int(action_data.get("x", 0)), int(action_data.get("y", 0))))
            self.last_action_id = action_id
            if record.progress_gain > 0:
                self.steps_since_progress = 0
            else:
                self.steps_since_progress += 1
            self.current_levels = levels_after
            if self.phase == "probe":
                self.probe_steps_done += 1

            prev_frame = frame
            frame = next_frame
            raw_obs = next_obs

            if self.steps_since_progress >= self.config.stall_steps:
                if self.reset_count >= self.config.reset_limit:
                    break
                raw_obs = env.reset()
                if raw_obs is None:
                    break
                self.reset_count += 1
                frame = final_subframe(raw_obs.frame)
                prev_frame = frame
                self._soft_reset()

        final_state = raw_obs.state.name if hasattr(raw_obs.state, "name") else str(raw_obs.state)

        signature_counts = {sig: 0 for sig in ALL_SIGNATURES}
        probe_signature_counts = {sig: 0 for sig in ALL_SIGNATURES}
        phase_action_counts = {"probe": 0, "exploit": 0}
        for transition in transitions:
            sig = transition["effect_signature"]
            signature_counts[sig] = signature_counts.get(sig, 0) + 1
            phase = transition["phase"]
            phase_action_counts[phase] = phase_action_counts.get(phase, 0) + 1
            if phase == "probe":
                probe_signature_counts[sig] = probe_signature_counts.get(sig, 0) + 1

        action_roles = {int(action_id): stat.role for action_id, stat in self.memory.actions.items()}

        return {
            "game_id": game_id,
            "final_state": final_state,
            "levels_completed": int(raw_obs.levels_completed),
            "actions_taken": len(transitions),
            "baseline_actions": list(baseline_actions or []),
            "transitions": transitions,
            "probe_budget": int(self.config.probe_budget),
            "probe_steps_used": int(self.probe_steps_done),
            "phase_at_end": self.phase,
            "memory_summary": self.memory.to_dict(),
            "signature_counts": signature_counts,
            "probe_signature_counts": probe_signature_counts,
            "phase_action_counts": phase_action_counts,
            "action_roles": action_roles,
            "resets_used": int(self.reset_count),
            "loop_metrics": self.loop_metrics.to_dict(),
            "agent_version": "probe_v3_1",
        }
