"""Per-episode probe memory for the probe-first agent.

Records, per action, the distribution of effect signatures observed so far,
and, for ACTION6 (click), the same distribution per coordinate and per
coarse (color, region-bucket) pair. The intent is object-conditioned reasoning
without paying for connected-component labelling on every priority lookup —
the region key is computed from the pixel color + an 8x8 spatial bucket.

Memory is intentionally NOT reset on env.reset() within the same play_env call;
that is one of the explicit weaknesses we want to fix relative to the teammate
path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import GRID_SIZE
from .effect_signatures import (
    ALL_SIGNATURES,
    GAME_OVER,
    GLOBAL_CHANGE,
    LOCAL_TOGGLE,
    MOTION_LIKE,
    NO_CHANGE,
    PROGRESS,
)

# 8 buckets across each axis of the 64x64 grid.
REGION_BUCKET_SIZE = 8
_BUCKET_MAX = (GRID_SIZE // REGION_BUCKET_SIZE) - 1


def bucket_point(point: Tuple[int, int]) -> Tuple[int, int]:
    x, y = int(point[0]), int(point[1])
    return (
        max(0, min(_BUCKET_MAX, x // REGION_BUCKET_SIZE)),
        max(0, min(_BUCKET_MAX, y // REGION_BUCKET_SIZE)),
    )


def region_key_at(point: Tuple[int, int], frame: Sequence[Sequence[int]]) -> Tuple[Any, ...]:
    """Cheap region key: (color_at_point, bucket). No CC required."""
    x, y = int(point[0]), int(point[1])
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
        return ("oob", bucket_point(point))
    color = int(frame[y][x])
    bucket = bucket_point(point)
    if color == 0:
        return ("background", bucket)
    return ("color", color, bucket)


@dataclass
class ActionStat:
    action_id: int
    trials: int = 0
    counts: Dict[str, int] = field(default_factory=lambda: {sig: 0 for sig in ALL_SIGNATURES})
    no_change_streak: int = 0

    def record(self, signature: str) -> None:
        self.trials += 1
        self.counts[signature] = self.counts.get(signature, 0) + 1
        self.no_change_streak = self.no_change_streak + 1 if signature == NO_CHANGE else 0

    @property
    def role(self) -> str:
        if self.trials == 0:
            return "unknown"
        if self.counts.get(PROGRESS, 0) > 0:
            return "progress"
        if self.counts.get(MOTION_LIKE, 0) >= max(1, self.trials // 2):
            return "navigation"
        if self.counts.get(LOCAL_TOGGLE, 0) >= 1:
            return "interaction"
        if self.counts.get(GLOBAL_CHANGE, 0) >= 1:
            return "global"
        if self.counts.get(NO_CHANGE, 0) == self.trials and self.trials >= 2:
            return "dead"
        return "uncertain"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": int(self.action_id),
            "trials": int(self.trials),
            "counts": {k: int(v) for k, v in self.counts.items()},
            "role": self.role,
        }


@dataclass
class CoordRecord:
    point: Tuple[int, int]
    region_key: Tuple[Any, ...]
    counts: Dict[str, int] = field(default_factory=lambda: {sig: 0 for sig in ALL_SIGNATURES})
    trials: int = 0

    def record(self, signature: str) -> None:
        self.trials += 1
        self.counts[signature] = self.counts.get(signature, 0) + 1

    @property
    def is_dead(self) -> bool:
        if self.trials < 2:
            return False
        return self.counts.get(NO_CHANGE, 0) == self.trials


@dataclass
class RegionStat:
    key: Tuple[Any, ...]
    counts: Dict[str, int] = field(default_factory=lambda: {sig: 0 for sig in ALL_SIGNATURES})
    trials: int = 0
    sample_points: List[Tuple[int, int]] = field(default_factory=list)

    def record(self, signature: str, point: Tuple[int, int]) -> None:
        self.trials += 1
        self.counts[signature] = self.counts.get(signature, 0) + 1
        if len(self.sample_points) < 4:
            self.sample_points.append(tuple(int(v) for v in point))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": [str(part) for part in self.key],
            "trials": int(self.trials),
            "counts": {k: int(v) for k, v in self.counts.items()},
            "sample_points": [list(p) for p in self.sample_points],
        }


def _dominant_signature(counts: Dict[str, int]) -> Optional[str]:
    if not counts:
        return None
    best_sig: Optional[str] = None
    best_count = 0
    for signature, count in counts.items():
        if int(count) > best_count:
            best_count = int(count)
            best_sig = signature
    return best_sig if best_count > 0 else None


@dataclass
class ColorActionStat:
    """(action_id, color) -> signature counts.

    Color is either the click's dominant local color (action 6) or the
    frame's top non-background color (non-coord actions). The agent uses
    this purely as a learned statistical lift: counts are recorded from
    interaction outcomes only, never from any game-specific rule.
    """

    action_id: int
    color: int
    counts: Dict[str, int] = field(default_factory=lambda: {sig: 0 for sig in ALL_SIGNATURES})
    trials: int = 0

    def record(self, signature: str) -> None:
        self.trials += 1
        self.counts[signature] = self.counts.get(signature, 0) + 1

    @property
    def dominant_signature(self) -> Optional[str]:
        return _dominant_signature(self.counts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": int(self.action_id),
            "color": int(self.color),
            "trials": int(self.trials),
            "counts": {k: int(v) for k, v in self.counts.items()},
        }


@dataclass
class ColorContextStat:
    """(action_id, color, context_key) -> signature counts.

    Context key is a small tuple computed from the local color descriptor
    (or a scene equivalent for non-coord actions). It refines the broader
    (action_id, color) hypothesis whenever observed outcomes diverge across
    contexts — the "split when behavior differs" rule.
    """

    action_id: int
    color: int
    context_key: Tuple[int, int]
    counts: Dict[str, int] = field(default_factory=lambda: {sig: 0 for sig in ALL_SIGNATURES})
    trials: int = 0
    splits: int = 0  # number of times this entry's outcome disagreed with the coarse parent

    def record(self, signature: str) -> None:
        self.trials += 1
        self.counts[signature] = self.counts.get(signature, 0) + 1

    @property
    def dominant_signature(self) -> Optional[str]:
        return _dominant_signature(self.counts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": int(self.action_id),
            "color": int(self.color),
            "context_key": list(self.context_key),
            "trials": int(self.trials),
            "splits": int(self.splits),
            "counts": {k: int(v) for k, v in self.counts.items()},
        }


@dataclass
class ColorObservation:
    """Outcome of a single observe_color call. Used by the agent for
    inspectability — saved in the transition's color_summary so episodes
    record whether each step was a NEW color, a NEW context, or a
    consistency-violating OLD context that got refined."""

    is_new_color: bool          # first time seeing (action_id, color)
    is_new_context: bool        # first time seeing (action_id, color, context_key)
    coarse_dominant_before: Optional[str]
    refined_active: bool        # refined entry has enough evidence to be authoritative
    disagreement: bool          # this signature differs from the coarse dominant
    split: bool                 # this observation triggered a refinement split

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_new_color": bool(self.is_new_color),
            "is_new_context": bool(self.is_new_context),
            "coarse_dominant_before": self.coarse_dominant_before,
            "refined_active": bool(self.refined_active),
            "disagreement": bool(self.disagreement),
            "split": bool(self.split),
        }


class ProbeMemory:
    """Per-episode memory. Persists across env.reset() within one play_env call."""

    def __init__(self) -> None:
        self.actions: Dict[int, ActionStat] = {}
        self.coords: Dict[Tuple[int, int], CoordRecord] = {}
        self.regions: Dict[Tuple[Any, ...], RegionStat] = {}
        self.global_change_steps: List[int] = []
        # (action_id, color) -> ColorActionStat. color = -1 means "unknown".
        self.color_action: Dict[Tuple[int, int], ColorActionStat] = {}
        # (action_id, color, context_key) -> refined stats. Entries are created
        # lazily — there is no preset "meaning" for any color/context.
        self.color_context: Dict[Tuple[int, int, Tuple[int, int]], ColorContextStat] = {}
        # Lightweight novelty/refinement counters, surfaced via to_dict so
        # triage and episode summaries can show how the agent's color
        # hypotheses evolved during the episode.
        self.color_observation_counts = {
            "observations": 0,
            "new_color": 0,
            "new_context": 0,
            "disagreements": 0,
            "splits": 0,
        }

    def ensure_action(self, action_id: int) -> ActionStat:
        action_id = int(action_id)
        if action_id not in self.actions:
            self.actions[action_id] = ActionStat(action_id=action_id)
        return self.actions[action_id]

    def record(
        self,
        action_id: int,
        action_data: Optional[Dict[str, int]],
        signature: str,
        frame_before: Sequence[Sequence[int]],
        step_index: int,
    ) -> None:
        stat = self.ensure_action(action_id)
        stat.record(signature)
        if signature == GLOBAL_CHANGE:
            self.global_change_steps.append(int(step_index))
            # A global change means our model of dead actions may be stale.
            # Reset per-action no-change streaks so the agent will re-test them.
            for action_stat in self.actions.values():
                action_stat.no_change_streak = 0
        if int(action_id) == 6 and action_data is not None:
            point = (int(action_data.get("x", 0)), int(action_data.get("y", 0)))
            region_key = region_key_at(point, frame_before)
            coord_record = self.coords.get(point)
            if coord_record is None:
                coord_record = CoordRecord(point=point, region_key=region_key)
                self.coords[point] = coord_record
            coord_record.record(signature)
            region_stat = self.regions.get(region_key)
            if region_stat is None:
                region_stat = RegionStat(key=region_key)
                self.regions[region_key] = region_stat
            region_stat.record(signature, point)

    def is_action_dead(self, action_id: int) -> bool:
        stat = self.actions.get(int(action_id))
        if stat is None:
            return False
        return stat.role == "dead"

    def has_recent_global_change(self, since_step: int) -> bool:
        return any(step >= int(since_step) for step in self.global_change_steps)

    def action_priority(self, action_id: int) -> float:
        stat = self.actions.get(int(action_id))
        if stat is None or stat.trials == 0:
            return 0.0
        weighted = (
            stat.counts.get(PROGRESS, 0) * 4.0
            + stat.counts.get(MOTION_LIKE, 0) * 1.0
            + stat.counts.get(LOCAL_TOGGLE, 0) * 1.5
            + stat.counts.get(GLOBAL_CHANGE, 0) * 0.5
            - stat.counts.get(NO_CHANGE, 0) * 0.7
            - stat.counts.get(GAME_OVER, 0) * 5.0
        )
        return weighted / float(max(1, stat.trials))

    def coord_priority(
        self,
        point: Tuple[int, int],
        frame: Sequence[Sequence[int]],
    ) -> float:
        score = 0.0
        coord_record = self.coords.get(point)
        if coord_record is not None:
            score += coord_record.counts.get(PROGRESS, 0) * 5.0
            score += coord_record.counts.get(LOCAL_TOGGLE, 0) * 1.5
            score += coord_record.counts.get(MOTION_LIKE, 0) * 0.5
            score += coord_record.counts.get(GLOBAL_CHANGE, 0) * 0.5
            score -= coord_record.counts.get(NO_CHANGE, 0) * 1.5
            score -= coord_record.counts.get(GAME_OVER, 0) * 5.0
            if coord_record.is_dead:
                score -= 4.0
        region_key = region_key_at(point, frame)
        region_stat = self.regions.get(region_key)
        if region_stat is not None:
            trials = max(1, region_stat.trials)
            score += region_stat.counts.get(PROGRESS, 0) * 1.5 / trials
            score += region_stat.counts.get(LOCAL_TOGGLE, 0) * 0.6 / trials
            score -= region_stat.counts.get(NO_CHANGE, 0) * 0.4 / trials
        return score

    def trial_count(self, action_id: int) -> int:
        stat = self.actions.get(int(action_id))
        return 0 if stat is None else stat.trials

    # ------------------------------------------------------------- color path

    # A refined entry needs at least this many trials before it can override
    # the broader (action_id, color) belief. Below the threshold the coarse
    # entry still drives priorities — this prevents a single observation from
    # fragmenting an otherwise-consistent hypothesis.
    REFINED_MIN_TRIALS = 3

    def _coarse_lift(self, record: Optional[ColorActionStat]) -> float:
        if record is None or record.trials == 0:
            return 0.0
        weighted = (
            record.counts.get(PROGRESS, 0) * 4.0
            + record.counts.get(MOTION_LIKE, 0) * 1.0
            + record.counts.get(LOCAL_TOGGLE, 0) * 1.5
            + record.counts.get(GLOBAL_CHANGE, 0) * 0.5
            - record.counts.get(NO_CHANGE, 0) * 0.7
            - record.counts.get(GAME_OVER, 0) * 5.0
        )
        damp = min(1.0, record.trials / 4.0)
        return damp * (weighted / float(max(1, record.trials)))

    def _refined_lift(self, record: ColorContextStat) -> float:
        weighted = (
            record.counts.get(PROGRESS, 0) * 4.0
            + record.counts.get(MOTION_LIKE, 0) * 1.0
            + record.counts.get(LOCAL_TOGGLE, 0) * 1.5
            + record.counts.get(GLOBAL_CHANGE, 0) * 0.5
            - record.counts.get(NO_CHANGE, 0) * 0.7
            - record.counts.get(GAME_OVER, 0) * 5.0
        )
        damp = min(1.0, record.trials / 4.0)
        return damp * (weighted / float(max(1, record.trials)))

    def observe_color(
        self,
        action_id: int,
        color: int,
        context_key: Tuple[int, int],
        signature: str,
    ) -> ColorObservation:
        """Record one (action, color, context) -> signature outcome.

        Implements the user's "new -> learn, old -> verify, same -> reinforce,
        different -> split" loop:

        * If the (action_id, color) coarse key has never been seen, this
          observation creates the broad hypothesis (new_color = True).
        * If the coarse key exists but the (action_id, color, context_key)
          triple has never been seen, the refined entry is created
          (new_context = True). The coarse entry is still updated.
        * If the new signature disagrees with the coarse dominant, the
          refined entry's `splits` counter is incremented and `disagreement`
          is reported back. This is the "split into a more refined
          hypothesis" path.
        * The coarse entry is always updated so that broad evidence still
          accumulates; the refined entry is what the agent prefers when
          it has enough trials.

        Returns a ColorObservation that the agent can attach to the
        transition for full inspectability.
        """
        coarse_key = (int(action_id), int(color))
        coarse = self.color_action.get(coarse_key)
        is_new_color = coarse is None
        coarse_dominant_before: Optional[str] = (
            coarse.dominant_signature if coarse is not None else None
        )
        if coarse is None:
            coarse = ColorActionStat(action_id=int(action_id), color=int(color))
            self.color_action[coarse_key] = coarse

        ctx_tuple: Tuple[int, int] = (int(context_key[0]), int(context_key[1]))
        refined_key = (int(action_id), int(color), ctx_tuple)
        refined = self.color_context.get(refined_key)
        is_new_context = refined is None
        if refined is None:
            refined = ColorContextStat(
                action_id=int(action_id),
                color=int(color),
                context_key=ctx_tuple,
            )
            self.color_context[refined_key] = refined

        # Update both layers. Order matters: read disagreement against the
        # coarse dominant *before* we add this observation.
        disagreement = (
            coarse_dominant_before is not None
            and coarse_dominant_before != signature
        )
        split = bool(disagreement and not is_new_color)
        coarse.record(signature)
        refined.record(signature)
        if split:
            refined.splits += 1

        refined_active = bool(refined.trials >= self.REFINED_MIN_TRIALS)

        # Counters for inspectability.
        self.color_observation_counts["observations"] += 1
        if is_new_color:
            self.color_observation_counts["new_color"] += 1
        if is_new_context:
            self.color_observation_counts["new_context"] += 1
        if disagreement:
            self.color_observation_counts["disagreements"] += 1
        if split:
            self.color_observation_counts["splits"] += 1

        return ColorObservation(
            is_new_color=is_new_color,
            is_new_context=is_new_context,
            coarse_dominant_before=coarse_dominant_before,
            refined_active=refined_active,
            disagreement=disagreement,
            split=split,
        )

    def record_color(self, action_id: int, color: int, signature: str) -> None:
        """Backward-compatible wrapper. New callers should prefer observe_color
        with an explicit context_key — this fallback uses an empty context."""
        self.observe_color(action_id, color, (-1, 0), signature)

    def color_action_priority(
        self,
        action_id: int,
        color: int,
        context_key: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Soft learned lift / penalty for (action_id, color) pairs.

        When `context_key` is provided AND the refined entry has enough trials
        to be authoritative, the refined lift fully replaces the coarse one.
        Otherwise the coarse lift is returned (possibly mixed with whatever
        partial refined evidence exists, weighted by trial count).

        Returns 0.0 when there is no evidence yet, so this is safe to add to
        any baseline.
        """
        coarse = self.color_action.get((int(action_id), int(color)))
        coarse_lift = self._coarse_lift(coarse)
        if context_key is None:
            return coarse_lift
        ctx_tuple = (int(context_key[0]), int(context_key[1]))
        refined = self.color_context.get((int(action_id), int(color), ctx_tuple))
        if refined is None or refined.trials == 0:
            return coarse_lift
        refined_lift = self._refined_lift(refined)
        if refined.trials >= self.REFINED_MIN_TRIALS:
            return refined_lift
        # Blend in proportion to refined evidence so the lift moves smoothly
        # from "broad belief" toward "context-specific belief".
        weight = float(refined.trials) / float(self.REFINED_MIN_TRIALS)
        return (1.0 - weight) * coarse_lift + weight * refined_lift

    def color_summary(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Inspectable list of the strongest (action, color) signals so far."""
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for (action_id, color), record in self.color_action.items():
            if record.trials < 2:
                continue
            score = (
                record.counts.get(PROGRESS, 0) * 4.0
                + record.counts.get(LOCAL_TOGGLE, 0) * 1.5
                + record.counts.get(GLOBAL_CHANGE, 0) * 0.5
                - record.counts.get(NO_CHANGE, 0) * 0.7
            ) / float(record.trials)
            scored.append((score, record.to_dict()))
        scored.sort(key=lambda item: abs(item[0]), reverse=True)
        return [entry[1] for entry in scored[: max(0, int(top_k * 4))]]

    def color_context_summary(self, top_k: int = 6) -> List[Dict[str, Any]]:
        """Refined-entry summary, sorted by absolute strength."""
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for (_, _, _), record in self.color_context.items():
            if record.trials < 2:
                continue
            score = (
                record.counts.get(PROGRESS, 0) * 4.0
                + record.counts.get(LOCAL_TOGGLE, 0) * 1.5
                + record.counts.get(GLOBAL_CHANGE, 0) * 0.5
                - record.counts.get(NO_CHANGE, 0) * 0.7
            ) / float(record.trials)
            scored.append((score, record.to_dict()))
        scored.sort(key=lambda item: abs(item[0]), reverse=True)
        return [entry[1] for entry in scored[: max(0, int(top_k))]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": {int(k): v.to_dict() for k, v in self.actions.items()},
            "regions": [stat.to_dict() for stat in self.regions.values()],
            "global_change_steps": list(self.global_change_steps),
            "num_unique_clicks": len(self.coords),
            "color_action": [stat.to_dict() for stat in self.color_action.values()],
            "color_context": [stat.to_dict() for stat in self.color_context.values()],
            "color_observation_counts": dict(self.color_observation_counts),
        }

    # ---------------------------------------------------------------- warm start
    def warm_start_from_prior(self, prior: Dict[str, Any]) -> None:
        """Seed memory from a persistent per-game prior.

        The prior is treated as a soft hint, not ground truth: dead_actions /
        dead_coords get just enough NO_CHANGE records to flip them to "dead"
        in the role classifier, so the agent skips them by default but a
        global_change reprobe window can still re-test them. promising_clicks
        get a single record of their last observed signature so coord_priority
        nudges toward them on the first exploit step.
        """
        for action_id in prior.get("dead_actions", []) or []:
            stat = self.ensure_action(int(action_id))
            while stat.trials < 2:
                stat.record(NO_CHANGE)
        for entry in prior.get("dead_coords", []) or []:
            try:
                x, y = int(entry[0]), int(entry[1])
            except (TypeError, IndexError, ValueError):
                continue
            point = (x, y)
            region_key = ("warm", (x // REGION_BUCKET_SIZE, y // REGION_BUCKET_SIZE))
            coord_record = self.coords.get(point)
            if coord_record is None:
                coord_record = CoordRecord(point=point, region_key=region_key)
                self.coords[point] = coord_record
            while coord_record.trials < 2:
                coord_record.record(NO_CHANGE)
        for entry in prior.get("promising_clicks", []) or []:
            try:
                x = int(entry["x"])
                y = int(entry["y"])
                signature = str(entry["signature"])
            except (TypeError, KeyError, ValueError):
                continue
            if signature == NO_CHANGE:
                continue
            point = (x, y)
            region_key = ("warm", (x // REGION_BUCKET_SIZE, y // REGION_BUCKET_SIZE))
            coord_record = self.coords.get(point)
            if coord_record is None:
                coord_record = CoordRecord(point=point, region_key=region_key)
                self.coords[point] = coord_record
            coord_record.record(signature)

        # Color-action stats: previously observed (action_id, color) -> sigs.
        # We only carry forward entries with trials >= 2 to avoid noise.
        for entry in prior.get("color_action", []) or []:
            try:
                action_id = int(entry["action_id"])
                color = int(entry["color"])
                trials = int(entry.get("trials", 0))
                counts = entry.get("counts") or {}
            except (TypeError, KeyError, ValueError):
                continue
            if trials < 2:
                continue
            key = (action_id, color)
            stat = self.color_action.get(key)
            if stat is None:
                stat = ColorActionStat(action_id=action_id, color=color)
                self.color_action[key] = stat
            for signature, count in counts.items():
                try:
                    count_int = int(count)
                except (TypeError, ValueError):
                    continue
                if count_int <= 0:
                    continue
                stat.counts[signature] = stat.counts.get(signature, 0) + count_int
                stat.trials += count_int

        # Refined (action_id, color, context_key) entries. These let warm-start
        # carry context-specific hypotheses across stages — same shape as the
        # in-episode refinement, just initialised from disk.
        for entry in prior.get("color_context", []) or []:
            try:
                action_id = int(entry["action_id"])
                color = int(entry["color"])
                ctx_raw = entry.get("context_key")
                if not ctx_raw or len(ctx_raw) != 2:
                    continue
                context_key = (int(ctx_raw[0]), int(ctx_raw[1]))
                trials = int(entry.get("trials", 0))
                splits = int(entry.get("splits", 0))
                counts = entry.get("counts") or {}
            except (TypeError, KeyError, ValueError):
                continue
            if trials < 2:
                continue
            key = (action_id, color, context_key)
            stat = self.color_context.get(key)
            if stat is None:
                stat = ColorContextStat(
                    action_id=action_id, color=color, context_key=context_key,
                )
                self.color_context[key] = stat
            for signature, count in counts.items():
                try:
                    count_int = int(count)
                except (TypeError, ValueError):
                    continue
                if count_int <= 0:
                    continue
                stat.counts[signature] = stat.counts.get(signature, 0) + count_int
                stat.trials += count_int
            stat.splits += splits
