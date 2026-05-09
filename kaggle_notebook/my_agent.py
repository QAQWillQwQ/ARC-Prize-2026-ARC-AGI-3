# =====================================================================
# WarmstartAgent v3 — replay-warm-start + per-game priors + visual
# saliency + online action-effect learning.
#
# Phase A (warmstart): plays GT replay actions verbatim until exhausted.
#   Honors RESET markers (action_input.id == 0 while state == GAME_OVER).
#
# Phase B (post-replay / hidden-game fallback): cycles through a strategy
#   bank as the agent burns through resets. v3 enrichments over v2:
#     - PER-GAME PRIORS: when self._short_id matches a known game in
#       per_game_priors.json, the initial strategy is chosen from the
#       distilled human archetype (keyboard_dominant -> directional_sustained,
#       click_dominant -> click_grid).
#     - VISUAL SALIENCY: at first Phase-B click step, extracts salient (x,y)
#       coords from the current frame (rare-color centroids + prior
#       hot_spots) and uses those as the click_grid sequence.
#     - ONLINE ACTION-EFFECT MAP: tracks per-(action_id, 8x8 spatial bin)
#       frame_delta. Currently logged for diagnostics; future versions can
#       use it to re-rank candidates within a strategy.
#     - DIRECTIONAL HISTORY: directional_sustained seeds direction from
#       the game prior's repeat_kept_actions when known.
#
# Strategies (in cycle order):
#     0. random_full          — random non-RESET, ACTION6 random coords
#     1. keyboard_only        — ACTION1-5 only (keyboard games)
#     2. click_grid           — ACTION6 cycling salient/grid coords
#     3. directional_sustained — repeat one direction key 3-5 times
#
# Patterns from arc3-sample-submission-stochastic-goose.ipynb:
#   - MAX_ACTIONS = inf (gateway caps).
#   - _MAX_FRAMES = 10 sliding window via append_frame override.
#   - try/except around is_done + choose_action with random fallback.
#   - 12h - 5min wall-clock safety net.
#   - DEBUG print on first action to introspect gateway frame format.
#   - available_actions iterable supports both ints and GameAction enums.
# =====================================================================
from __future__ import annotations

import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


REPLAY_BASE_DIR = Path(os.environ.get('ARC_REPLAY_BASE_DIR', '/kaggle/working/replays'))
WALL_BUDGET_SECONDS = 12 * 3600 - 5 * 60  # 12h - 5min safety margin

# ---------------------- helpers ----------------------


def _normalize_action_id(raw: Any) -> int:
    if raw is None:
        return 0
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw if 1 <= raw <= 7 else 0
    if isinstance(raw, str):
        s = raw.strip().upper()
        if s == 'RESET':
            return 0
        if s.startswith('ACTION') and s[6:].isdigit():
            n = int(s[6:])
            return n if 1 <= n <= 7 else 0
    return 0


def _load_replay_actions(short_game_id: str) -> List[Dict[str, Any]]:
    """Returns a list of {'type': 'reset' | 'action', ...} entries — full replay.

    The full GT recording (all attempts including failures) is preserved. The
    Phase A loop in MyAgent.choose_action handles env state transitions: when
    env enters GAME_OVER it issues RESET and continues the replay from the
    current cursor (skipping any GT reset markers that may follow). This way
    the agent walks every attempt the human made, accumulating levels across
    the cumulative sequence.
    """
    replay_dir = REPLAY_BASE_DIR / short_game_id / 'replays'
    if not replay_dir.is_dir():
        return []
    files = sorted(replay_dir.glob('*.json'))
    if not files:
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(files[0], 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ai = (rec.get('data') or {}).get('action_input') or {}
                aid = _normalize_action_id(ai.get('id'))
                if aid == 0:
                    out.append({'type': 'reset'})
                    continue
                ad = ai.get('data') or {}
                try:
                    x = int(ad.get('x', 0)) if aid == 6 else 0
                    y = int(ad.get('y', 0)) if aid == 6 else 0
                except (TypeError, ValueError):
                    x, y = 0, 0
                out.append({'type': 'action', 'id': aid, 'x': x, 'y': y})
    except Exception:
        return []
    return out


def _archetype_from_action_set(available: List[int]) -> Optional[str]:
    """v3 transfer-learning trivial cases: archetype is determined by
    available_actions alone for ~52% of games (verified by LOO).

    Returns:
        'keyboard_dominant' if no ACTION6 in available
        'click_dominant' if available is narrow (<= 3 actions) and includes 6
        None for mixed action sets — caller should fall back to histogram.
    """
    if not available:
        return None
    if 6 not in available:
        return 'keyboard_dominant'
    # 6 is in there. If ≤ 3 actions total and 6 is one of them, this is
    # almost certainly a click_dominant game.
    if len(available) <= 3:
        return 'click_dominant'
    return None  # mixed — needs histogram matching


def _jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _cos_hist(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


def _color_hist_normalized(frame: Sequence[Sequence[int]]) -> List[float]:
    """16-dim normalized color histogram of a 64x64 frame."""
    counts = [0.0] * 16
    for row in frame:
        for c in row:
            ci = int(c)
            if 0 <= ci < 16:
                counts[ci] += 1.0
    total = sum(counts)
    if total > 0:
        return [v / total for v in counts]
    return counts


def _load_per_game_priors() -> Dict[str, Any]:
    """Try multiple paths for per_game_priors.json. Returns empty dict if none found."""
    candidates = [
        os.environ.get('ARC_PRIORS_PATH'),
        str(REPLAY_BASE_DIR.parent / 'per_game_priors.json'),
        # Project layout (local validation harness).
        str(REPLAY_BASE_DIR.parent / 'Local_Output' / 'per_game_priors.json'),
        # Kaggle layouts.
        '/kaggle/working/per_game_priors.json',
        '/kaggle/working/replays/per_game_priors.json',
        '/kaggle/input/arc-agi-3-replays-v1/per_game_priors.json',
    ]
    for path in candidates:
        if not path:
            continue
        p = Path(path)
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return {}


def _available_action_ids(latest_frame: FrameData) -> List[int]:
    """Gateway sends ints [1..6]; toolkit sends GameAction enums. Handle both."""
    raw = getattr(latest_frame, 'available_actions', None)
    if raw is None:
        return list(range(1, 8))
    out = []
    for a in raw:
        try:
            v = a.value if hasattr(a, 'value') else int(a)
            if 1 <= int(v) <= 7:
                out.append(int(v))
        except Exception:
            continue
    return out or list(range(1, 8))


def _safe_random_action(rng: random.Random, latest_frame: FrameData) -> GameAction:
    """Random non-RESET action, restricted to available_actions when known."""
    available_ids = _available_action_ids(latest_frame)
    aid = rng.choice(available_ids) if available_ids else 1
    try:
        action = GameAction.from_id(int(aid))
    except Exception:
        action = GameAction.ACTION1
    if action.is_complex():
        action.set_data({'x': rng.randint(0, 63), 'y': rng.randint(0, 63)})
        action.reasoning = {'phase': 'fallback', 'strategy': 'random_full'}
    elif action.is_simple():
        action.reasoning = {'phase': 'fallback', 'strategy': 'random_full'}
    return action


def _flatten_frame(frame: Any) -> Optional[Sequence[Sequence[int]]]:
    """The gateway/toolkit may return frame as List[List[List[int]]] (a stack)
    or List[List[int]] (single grid). Return the latest 64x64 grid."""
    try:
        if not isinstance(frame, list) or not frame:
            return None
        if isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
            return frame[-1]
        return frame
    except Exception:
        return None


def _extract_saliency(frame: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    """Find salient (x, y) coords from a 64x64 frame.

    Heuristic: rare colors are likely "interesting" (humans gravitate toward
    visually distinctive elements). For each rare color (count between 1 and
    200), emit centroid + bbox corners. Always include a default grid as
    a tail so the click sequence is non-empty.

    Returns up to 20 deduplicated (x, y) tuples in priority order.
    """
    rows = len(frame)
    cols = len(frame[0]) if rows else 0
    if rows < 1 or cols < 1:
        return [(32, 32), (5, 5), (5, 58), (58, 5), (58, 58)]

    color_counts: Dict[int, int] = {}
    color_pixels: Dict[int, List[Tuple[int, int]]] = {}
    for y, row in enumerate(frame):
        for x, c in enumerate(row):
            ci = int(c)
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
        # 1. Centroid (the geometric center)
        salient.append((cx, cy))
        # 2. Per-2026-05-08 saliency_hypothesis_validation: humans click near
        #    rare-color regions but rarely AT centroids (33.7% within
        #    manhattan 8, 70.5% within 16). Add bbox corners + offset points
        #    to cover the "near-the-blob" radius the GT actually shows.
        if len(pts) >= 3:
            x_lo, x_hi = min(xs), max(xs)
            y_lo, y_hi = min(ys), max(ys)
            # Full 4 bbox corners (was only 2 — min/min and max/max).
            salient.append((x_lo, y_lo))
            salient.append((x_lo, y_hi))
            salient.append((x_hi, y_lo))
            salient.append((x_hi, y_hi))
        if len(pts) >= 5:
            # 4-direction offsets around the centroid.
            for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4)):
                ox = max(0, min(63, cx + dx))
                oy = max(0, min(63, cy + dy))
                salient.append((ox, oy))

    salient.extend([
        (32, 32), (5, 5), (5, 58), (58, 5), (58, 58),
        (32, 5), (32, 58), (5, 32), (58, 32),
    ])

    seen: set = set()
    out: List[Tuple[int, int]] = []
    for p in salient:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:30]  # widened from 20 to fit the expanded saliency pool


def _frame_delta(prev: Sequence[Sequence[int]], cur: Sequence[Sequence[int]]) -> int:
    """Pixel-difference count between two 64x64 frames. 0 if same."""
    try:
        n = 0
        for y in range(min(len(prev), len(cur))):
            pr = prev[y]
            cu = cur[y]
            for x in range(min(len(pr), len(cu))):
                if int(pr[x]) != int(cu[x]):
                    n += 1
        return n
    except Exception:
        return 0


# 13-point click grid: center + corners + edge midpoints + inner-quadrant
# midpoints. Used as the BASE click grid when no saliency is extracted yet.
CLICK_GRID_POINTS: Tuple[Tuple[int, int], ...] = (
    (32, 32),  # center
    (5, 5), (5, 58), (58, 5), (58, 58),  # corners
    (32, 5), (32, 58), (5, 32), (58, 32),  # edge midpoints
    (16, 16), (16, 48), (48, 16), (48, 48),  # inner quadrant midpoints
)


def _build_edge_sweep_points() -> List[Tuple[int, int]]:
    """Perimeter points (clockwise from top-left) + 4x4 interior grid.

    Used by the edge_sweep Phase B strategy. ~40 points; cycles through them
    deterministically. Different from CLICK_GRID_POINTS which has 13 'safe'
    pre-known click candidates.
    """
    points: List[Tuple[int, int]] = []
    for x in range(4, 61, 8):
        points.append((x, 2))
    for y in range(4, 61, 8):
        points.append((61, y))
    for x in range(60, 3, -8):
        points.append((x, 61))
    for y in range(60, 3, -8):
        points.append((2, y))
    for x in (12, 24, 36, 48):
        for y in (12, 24, 36, 48):
            points.append((x, y))
    return points


EDGE_SWEEP_POINTS: Tuple[Tuple[int, int], ...] = tuple(_build_edge_sweep_points())


# ---------------------- main agent ----------------------


class MyAgent(Agent):
    """Replay-warm-start v3: warmstart + priors + saliency + online learning."""

    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    STRATEGIES: Tuple[str, ...] = (
        'random_full',
        'keyboard_only',
        'click_grid',
        'directional_sustained',
        'edge_sweep',
        'color_targeted',
        'action_id_sweep',
    )

    # Map archetype string (from per_game_priors) -> starting strategy_idx.
    ARCHETYPE_TO_STRATEGY_IDX: Dict[str, int] = {
        'mixed': 0,
        'click_dominant': 2,
        'keyboard_dominant': 3,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._short_id = self.game_id.split('-', 1)[0] if self.game_id else ''
        self._replay = _load_replay_actions(self._short_id)
        self._replay_idx = 0
        self._start_time = time.time()
        self._debug_logged = False
        seed_int = (int(time.time() * 1_000_000) + hash(self.game_id)) & 0xFFFFFFFF
        self._rng = random.Random(seed_int)

        # v3: per-game prior lookup. If game_id is known, use directly. Else
        # fall back to transfer-learning resolution at first choose_action
        # (we need the first frame, which arrives via the framework loop).
        self._all_priors = _load_per_game_priors()
        self._game_prior: Dict[str, Any] = self._all_priors.get(self._short_id, {})
        self._is_hidden_game = not bool(self._game_prior)
        self._transfer_resolved = False  # set True once we attempt resolution
        archetype = str(self._game_prior.get('archetype', 'mixed'))
        self._initial_strategy_idx = self.ARCHETYPE_TO_STRATEGY_IDX.get(archetype, 0)

        # Phase B (post-replay / hidden-game fallback) state.
        self._post_replay_reset_count = 0
        self._strategy_idx = self._initial_strategy_idx
        self._click_grid_points: List[Tuple[int, int]] = list(CLICK_GRID_POINTS)
        self._click_grid_idx = 0
        self._sustained_dir: Optional[int] = None
        self._sustained_remaining = 0
        self._saliency_built = False

        # v3: online action-effect tracking. Key = (action_id, x_bin, y_bin).
        # x_bin/y_bin are 0..7 (8x8 spatial bins, each = 8x8 pixels).
        self._action_effects: Dict[Tuple[int, int, int], Tuple[int, float]] = {}
        self._last_returned_action: Optional[GameAction] = None

        # v4 (improvements 1-5): online learning + freeze detection +
        # cross-attempt strategy memory + new-strategy state.
        self._frozen_streak: int = 0                          # consecutive zero-delta steps
        self._strategy_levels: Dict[str, int] = {}            # per-strategy levels cleared this game
        self._last_levels_seen: int = 0                       # for level-delta detection
        self._edge_sweep_idx: int = 0                         # cursor into EDGE_SWEEP_POINTS
        self._color_targeted_state: Dict[int, List[Tuple[int, int]]] = {}   # color -> remaining points
        self._color_targeted_color_order: List[int] = []      # iteration order across colors
        self._color_targeted_color_idx: int = 0
        self._action_id_sweep_idx: int = 0                    # which available action we're cycling on
        self._action_id_sweep_count: int = 0                  # repeats so far for current action

        prior_summary = (
            f"archetype={archetype} click_ratio={self._game_prior.get('click_ratio', '?')}"
            if self._game_prior else "no_prior"
        )
        print(
            f'[MyAgent] init game_id={self.game_id} short_id={self._short_id} '
            f'replay_entries={len(self._replay)} prior=({prior_summary}) '
            f'initial_strategy={self.STRATEGIES[self._initial_strategy_idx]}',
            flush=True,
        )

    # ---------------------- framework hooks ----------------------

    def append_frame(self, frame: FrameData) -> None:
        """Sliding window on self.frames to bound memory across long replays."""
        self.frames.append(frame)
        if len(self.frames) > self._MAX_FRAMES:
            self.frames = self.frames[-self._MAX_FRAMES:]
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, 'recorder') and not self.is_playback:
            try:
                self.recorder.record(json.loads(frame.model_dump_json()))
            except Exception:
                pass

    def _wall_elapsed(self) -> bool:
        return (time.time() - self._start_time) >= WALL_BUDGET_SECONDS

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        try:
            if latest_frame.state is GameState.WIN:
                return True
            if self._wall_elapsed():
                print(f'[MyAgent] {self._short_id} wall budget elapsed', flush=True)
                return True
            return False
        except Exception as exc:
            print(f'[MyAgent] is_done crashed: {exc}', flush=True)
            traceback.print_exc()
            return True

    # ---------------------- v3: saliency + online learning ----------------------

    def _resolve_hidden_game_prior(self, latest_frame: FrameData) -> None:
        """For hidden games (no per-game-id match): infer archetype + priors
        by combining (a) action-set rule (deterministic for ~52% of games)
        with (b) histogram cosine fallback among action-similar known games.

        The action-set rule covers:
          - no ACTION6 → keyboard_dominant
          - narrow action set including 6 (≤ 3 actions) → click_dominant
        These two cases hit ~13/25 known games with 100% accuracy (verified
        by LOO). The remaining mixed-action games need histogram matching.

        Sets self._game_prior to the aggregated/inferred prior and
        self._initial_strategy_idx to the matching archetype.
        """
        if self._transfer_resolved:
            return
        self._transfer_resolved = True
        if not self._all_priors:
            return  # no priors loaded; agent stays at v2 default

        available = _available_action_ids(latest_frame)
        # Trivial action-set rule first.
        trivial_archetype = _archetype_from_action_set(available)
        if trivial_archetype is not None:
            # Pick a representative prior matching the inferred archetype.
            same_arche = [
                p for p in self._all_priors.values()
                if p.get('archetype') == trivial_archetype
            ]
            # Aggregate hot_spots across all same-archetype known games (for
            # click_dominant case). Repeats too.
            agg: Dict[str, Any] = {
                'archetype': trivial_archetype,
                'click_hot_spots': [],
                'repeat_kept_actions': {},
            }
            if same_arche:
                hot_pool: Dict[Tuple[int, int], float] = {}
                rep_pool: Dict[str, float] = {}
                for p in same_arche:
                    for hp in p.get('click_hot_spots', []):
                        try:
                            xy = (int(hp['x']), int(hp['y']))
                            hot_pool[xy] = hot_pool.get(xy, 0.0) + float(hp.get('count', 1))
                        except Exception:
                            continue
                    for k, v in p.get('repeat_kept_actions', {}).items():
                        rep_pool[str(k)] = rep_pool.get(str(k), 0.0) + float(v)
                agg['click_hot_spots'] = [
                    {'x': x, 'y': y, 'count': int(s)}
                    for (x, y), s in sorted(hot_pool.items(), key=lambda kv: -kv[1])[:8]
                ]
                agg['repeat_kept_actions'] = rep_pool
            self._game_prior = agg
            self._initial_strategy_idx = self.ARCHETYPE_TO_STRATEGY_IDX.get(trivial_archetype, 0)
            self._strategy_idx = self._initial_strategy_idx
            print(
                f'[MyAgent] {self._short_id} HIDDEN: trivial action-set rule '
                f'→ archetype={trivial_archetype} initial_strategy={self.STRATEGIES[self._initial_strategy_idx]}',
                flush=True,
            )
            return

        # Non-trivial: histogram cosine matching among action-similar games.
        frame = _flatten_frame(getattr(latest_frame, 'frame', None))
        if frame is None:
            return
        query_hist = _color_hist_normalized(frame)
        candidates: List[Tuple[str, float, Dict[str, Any]]] = []
        for gid, prior in self._all_priors.items():
            other_actions = prior.get('available_actions_union', [])
            if _jaccard(available, other_actions) < 0.5:
                continue
            sim = _cos_hist(query_hist, prior.get('first_frame_color_hist', []))
            candidates.append((gid, sim, prior))
        candidates.sort(key=lambda x: -x[1])
        top3 = candidates[:3]
        if not top3 or top3[0][1] < 0.5:
            print(
                f'[MyAgent] {self._short_id} HIDDEN: no confident histogram match '
                f'(best_sim={top3[0][1] if top3 else 0:.2f}) → keep v2 default',
                flush=True,
            )
            return

        # Vote on archetype, weighted by sim.
        from collections import Counter
        votes: Counter = Counter()
        hot_pool2: Dict[Tuple[int, int], float] = {}
        rep_pool2: Dict[str, float] = {}
        for gid, sim, p in top3:
            votes[p.get('archetype', 'mixed')] += sim
            for hp in p.get('click_hot_spots', []):
                try:
                    xy = (int(hp['x']), int(hp['y']))
                    hot_pool2[xy] = hot_pool2.get(xy, 0.0) + float(hp.get('count', 1)) * sim
                except Exception:
                    continue
            for k, v in p.get('repeat_kept_actions', {}).items():
                rep_pool2[str(k)] = rep_pool2.get(str(k), 0.0) + float(v) * sim
        archetype = votes.most_common(1)[0][0]
        self._game_prior = {
            'archetype': archetype,
            'click_hot_spots': [
                {'x': x, 'y': y, 'count': int(s)}
                for (x, y), s in sorted(hot_pool2.items(), key=lambda kv: -kv[1])[:8]
            ],
            'repeat_kept_actions': rep_pool2,
        }
        self._initial_strategy_idx = self.ARCHETYPE_TO_STRATEGY_IDX.get(archetype, 0)
        self._strategy_idx = self._initial_strategy_idx
        print(
            f'[MyAgent] {self._short_id} HIDDEN: histogram top-3 → '
            f'{[(g, round(s, 2)) for g, s, _ in top3]} '
            f'→ archetype={archetype} initial_strategy={self.STRATEGIES[self._initial_strategy_idx]}',
            flush=True,
        )

    def _build_click_grid_from_saliency(self, latest_frame: FrameData) -> None:
        """Populate self._click_grid_points from prior hot_spots + frame saliency."""
        salient: List[Tuple[int, int]] = []
        # 1. Prior hot_spots (from per_game_priors.json) — placed first.
        for hp in self._game_prior.get('click_hot_spots', []):
            try:
                salient.append((int(hp['x']), int(hp['y'])))
            except Exception:
                continue
        # 2. Frame saliency.
        frame = _flatten_frame(getattr(latest_frame, 'frame', None))
        if frame is not None:
            try:
                salient.extend(_extract_saliency(frame))
            except Exception:
                pass
        # 3. Default grid as last-resort tail.
        salient.extend(CLICK_GRID_POINTS)

        seen: set = set()
        out: List[Tuple[int, int]] = []
        for p in salient:
            if p not in seen:
                seen.add(p)
                out.append(p)
        self._click_grid_points = out[:20]
        self._saliency_built = True
        print(
            f'[MyAgent] {self._short_id} built click_grid '
            f'(len={len(self._click_grid_points)}, prior_hot_spots={len(self._game_prior.get("click_hot_spots", []))})',
            flush=True,
        )

    def _record_last_action_effect(self, frames: List[FrameData], latest_frame: FrameData) -> None:
        """If we have a previous frame and a previously-returned action,
        compute frame_delta and update the online action-effects map.

        Also bumps the freeze streak (consecutive zero-delta steps) used by
        choose_action's freeze-detection branch.
        """
        if self._last_returned_action is None or len(frames) < 2:
            return
        prev = _flatten_frame(getattr(frames[-2], 'frame', None))
        cur = _flatten_frame(getattr(latest_frame, 'frame', None))
        if prev is None or cur is None:
            return
        delta = _frame_delta(prev, cur)
        # v4 freeze detection: track consecutive zero-delta steps.
        if delta == 0:
            self._frozen_streak += 1
        else:
            self._frozen_streak = 0
        try:
            aid = int(self._last_returned_action.value if hasattr(self._last_returned_action, 'value') else 0)
        except Exception:
            aid = 0
        if aid == 6:
            try:
                d = self._last_returned_action.action_data.model_dump() if hasattr(self._last_returned_action.action_data, 'model_dump') else (self._last_returned_action.action_data or {})
                x = int((d or {}).get('x', 0))
                y = int((d or {}).get('y', 0))
            except Exception:
                x, y = 0, 0
            key = (6, x // 8, y // 8)
        else:
            key = (aid, 0, 0)
        c, m = self._action_effects.get(key, (0, 0.0))
        n_new = c + 1
        m_new = m + (delta - m) / n_new
        self._action_effects[key] = (n_new, m_new)

    def _score_action_candidate(self, action_id: int, x: int = 0, y: int = 0) -> float:
        """Online action-effect score for a candidate. Used by click_grid and
        directional_sustained re-rankings.

        Score in [0.1, 1.0]:
          - 1.0  → never tried (explore)
          - 0.1  → tried but mean delta < 1 pixel (known dead — strongly avoid)
          - else → mean_delta normalized, capped at 1.0
        """
        if int(action_id) == 6:
            key = (6, int(x) // 8, int(y) // 8)
        else:
            key = (int(action_id), 0, 0)
        count, mean_delta = self._action_effects.get(key, (0, 0.0))
        if count == 0:
            return 1.0
        if mean_delta < 1.0:
            return 0.1
        return min(1.0, mean_delta / 10.0)

    # ---------------------- strategy implementations ----------------------

    def _strategy_random_full(self, latest_frame: FrameData) -> GameAction:
        return _safe_random_action(self._rng, latest_frame)

    def _strategy_keyboard_only(self, latest_frame: FrameData) -> GameAction:
        available = [a for a in _available_action_ids(latest_frame) if a != 6]
        if not available:
            return _safe_random_action(self._rng, latest_frame)
        aid = self._rng.choice(available)
        try:
            action = GameAction.from_id(aid)
        except Exception:
            return _safe_random_action(self._rng, latest_frame)
        action.reasoning = {'phase': 'fallback', 'strategy': 'keyboard_only'}
        return action

    def _strategy_click_grid(self, latest_frame: FrameData) -> GameAction:
        available = _available_action_ids(latest_frame)
        if 6 not in available:
            return self._strategy_keyboard_only(latest_frame)
        if not self._saliency_built:
            self._build_click_grid_from_saliency(latest_frame)
        if not self._click_grid_points:
            self._click_grid_points = list(CLICK_GRID_POINTS)

        # v4 improvement #1: re-rank click_grid_points by online action-effect
        # score, then pick from top-K via weighted random. Falls back to the
        # original sequential cursor when scoring is uniform (early in the
        # episode, before any data is gathered).
        scored = [(p, self._score_action_candidate(6, p[0], p[1])) for p in self._click_grid_points]
        top_k = sorted(scored, key=lambda kv: -kv[1])[:5]
        if top_k and any(s > 0 for _, s in top_k):
            weights = [s for _, s in top_k]
            try:
                chosen = self._rng.choices([p for p, _ in top_k], weights=weights, k=1)[0]
            except Exception:
                chosen = top_k[0][0]
            x, y = chosen
            source = 'top_k'
        else:
            x, y = self._click_grid_points[self._click_grid_idx % len(self._click_grid_points)]
            source = 'cursor'
        self._click_grid_idx += 1
        try:
            action = GameAction.ACTION6
            action.set_data({'x': int(x), 'y': int(y)})
        except Exception:
            return _safe_random_action(self._rng, latest_frame)
        action.reasoning = {
            'phase': 'fallback',
            'strategy': 'click_grid',
            'point': [int(x), int(y)],
            'idx': self._click_grid_idx - 1,
            'source': source,
        }
        return action

    def _strategy_directional_sustained(self, latest_frame: FrameData) -> GameAction:
        available = [a for a in _available_action_ids(latest_frame) if 1 <= a <= 5]
        if not available:
            return _safe_random_action(self._rng, latest_frame)
        if self._sustained_dir is None or self._sustained_remaining <= 0 or self._sustained_dir not in available:
            # v4 improvement #5: combine prior weight (repeat_kept_actions, normalized)
            # with online action-effect score. Online weight dominates after a few
            # samples are collected for the action.
            repeat_kept = self._game_prior.get('repeat_kept_actions', {})
            if isinstance(repeat_kept, dict):
                prior_max = max((float(v) for v in repeat_kept.values()), default=1.0) or 1.0
            else:
                prior_max = 1.0
                repeat_kept = {}

            weighted: List[Tuple[int, float]] = []
            for aid in available:
                prior_w = float(repeat_kept.get(str(aid), 0.0)) / prior_max
                online_w = self._score_action_candidate(aid)
                weighted.append((aid, prior_w * 0.5 + online_w * 1.0))

            if weighted and any(w > 0 for _, w in weighted):
                actions = [aid for aid, _ in weighted]
                weights = [w for _, w in weighted]
                try:
                    chosen = self._rng.choices(actions, weights=weights, k=1)[0]
                except Exception:
                    chosen = self._rng.choice(actions)
            else:
                chosen = self._rng.choice(available)
            self._sustained_dir = int(chosen)
            self._sustained_remaining = self._rng.randint(3, 5)
        aid = int(self._sustained_dir)
        self._sustained_remaining -= 1
        try:
            action = GameAction.from_id(aid)
        except Exception:
            return _safe_random_action(self._rng, latest_frame)
        action.reasoning = {
            'phase': 'fallback',
            'strategy': 'directional_sustained',
            'remaining': self._sustained_remaining,
            'aid': aid,
        }
        return action

    def _strategy_edge_sweep(self, latest_frame: FrameData) -> GameAction:
        """v4 strategy: cycle perimeter points + 4x4 interior grid for ACTION6.

        Useful for games where the goal is on the screen edge or where boundary
        clicks reveal hidden state. Different from click_grid (saliency-based);
        edge_sweep doesn't depend on visual rare-color content.
        """
        available = _available_action_ids(latest_frame)
        if 6 not in available:
            return self._strategy_keyboard_only(latest_frame)
        x, y = EDGE_SWEEP_POINTS[self._edge_sweep_idx % len(EDGE_SWEEP_POINTS)]
        self._edge_sweep_idx += 1
        try:
            action = GameAction.ACTION6
            action.set_data({'x': int(x), 'y': int(y)})
        except Exception:
            return _safe_random_action(self._rng, latest_frame)
        action.reasoning = {
            'phase': 'fallback',
            'strategy': 'edge_sweep',
            'point': [int(x), int(y)],
            'idx': self._edge_sweep_idx - 1,
        }
        return action

    def _extract_color_targeted_points(
        self, frame: Sequence[Sequence[int]]
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Per-rare-color: centroid + bbox corners. Returns dict color → points list."""
        rows = len(frame) if frame is not None and len(frame) > 0 else 0
        cols = len(frame[0]) if rows else 0
        if rows < 1 or cols < 1:
            return {}
        color_pixels: Dict[int, List[Tuple[int, int]]] = {}
        for y, row in enumerate(frame):
            for x, c in enumerate(row):
                try:
                    ci = int(c)
                except (TypeError, ValueError):
                    # Non-scalar (e.g. numpy ndarray, nested list) — try .item().
                    try:
                        ci = int(c.item()) if hasattr(c, "item") else 0
                    except Exception:
                        continue
                if ci == 0:
                    continue
                color_pixels.setdefault(ci, []).append((x, y))
        rare = {c: pts for c, pts in color_pixels.items() if 1 <= len(pts) <= 200}
        out: Dict[int, List[Tuple[int, int]]] = {}
        for color, pts in sorted(rare.items(), key=lambda kv: len(kv[1]))[:6]:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))
            per_color: List[Tuple[int, int]] = [(cx, cy)]
            if len(pts) >= 3:
                per_color.extend([
                    (min(xs), min(ys)),
                    (max(xs), min(ys)),
                    (min(xs), max(ys)),
                    (max(xs), max(ys)),
                ])
            out[color] = per_color
        return out

    def _strategy_color_targeted(self, latest_frame: FrameData) -> GameAction:
        """v4 strategy: per-rare-color exhaustive click cycle.

        Different from click_grid (which mixes all rare colors): this strategy
        clicks all points of color A first (centroid + 4 bbox corners), then
        moves to color B, etc. Lets the agent test "is this color the goal?"
        as a discrete hypothesis per color.
        """
        available = _available_action_ids(latest_frame)
        if 6 not in available:
            return self._strategy_keyboard_only(latest_frame)

        # Build per-color point queues lazily, and rebuild after a full pass.
        if not self._color_targeted_color_order:
            frame = _flatten_frame(getattr(latest_frame, 'frame', None))
            if frame is None:
                return _safe_random_action(self._rng, latest_frame)
            color_pts = self._extract_color_targeted_points(frame)
            if not color_pts:
                return _safe_random_action(self._rng, latest_frame)
            self._color_targeted_state = {c: list(pts) for c, pts in color_pts.items()}
            self._color_targeted_color_order = list(color_pts.keys())
            self._color_targeted_color_idx = 0

        # Find next non-empty queue, rotating across colors.
        attempts = 0
        while attempts < len(self._color_targeted_color_order):
            color = self._color_targeted_color_order[
                self._color_targeted_color_idx % len(self._color_targeted_color_order)
            ]
            queue = self._color_targeted_state.get(color, [])
            if queue:
                x, y = queue.pop(0)
                self._color_targeted_color_idx += 1  # rotate next time
                try:
                    action = GameAction.ACTION6
                    action.set_data({'x': int(x), 'y': int(y)})
                except Exception:
                    return _safe_random_action(self._rng, latest_frame)
                action.reasoning = {
                    'phase': 'fallback',
                    'strategy': 'color_targeted',
                    'color': int(color),
                    'point': [int(x), int(y)],
                }
                return action
            self._color_targeted_color_idx += 1
            attempts += 1

        # All queues exhausted — rebuild from current frame (state may have changed).
        self._color_targeted_color_order = []
        self._color_targeted_state = {}
        return self._strategy_color_targeted(latest_frame)

    def _strategy_action_id_sweep(self, latest_frame: FrameData) -> GameAction:
        """v4 strategy: cycle every available action_id with N repeats each.

        Forces the agent to try each action_id multiple times before moving on.
        Catches games where one specific action_id is the gate but the agent
        never tried it (e.g., ACTION7 when most games don't use it).
        """
        available = _available_action_ids(latest_frame)
        if not available:
            return _safe_random_action(self._rng, latest_frame)
        repeats_per_action = 3
        current_aid = available[self._action_id_sweep_idx % len(available)]
        self._action_id_sweep_count += 1
        if self._action_id_sweep_count >= repeats_per_action:
            self._action_id_sweep_idx += 1
            self._action_id_sweep_count = 0
        try:
            action = GameAction.from_id(int(current_aid))
        except Exception:
            return _safe_random_action(self._rng, latest_frame)
        if action.is_complex():
            # ACTION6: first repeat clicks center, others are random.
            if self._action_id_sweep_count == 1:
                action.set_data({'x': 32, 'y': 32})
            else:
                action.set_data({
                    'x': self._rng.randint(0, 63),
                    'y': self._rng.randint(0, 63),
                })
        action.reasoning = {
            'phase': 'fallback',
            'strategy': 'action_id_sweep',
            'aid': int(current_aid),
            'count': self._action_id_sweep_count,
        }
        return action

    def _strategy_action(self, latest_frame: FrameData) -> GameAction:
        strategy = self.STRATEGIES[self._strategy_idx % len(self.STRATEGIES)]
        if strategy == 'random_full':
            return self._strategy_random_full(latest_frame)
        if strategy == 'keyboard_only':
            return self._strategy_keyboard_only(latest_frame)
        if strategy == 'click_grid':
            return self._strategy_click_grid(latest_frame)
        if strategy == 'directional_sustained':
            return self._strategy_directional_sustained(latest_frame)
        if strategy == 'edge_sweep':
            return self._strategy_edge_sweep(latest_frame)
        if strategy == 'color_targeted':
            return self._strategy_color_targeted(latest_frame)
        if strategy == 'action_id_sweep':
            return self._strategy_action_id_sweep(latest_frame)
        return _safe_random_action(self._rng, latest_frame)

    def _advance_strategy(self, reason: str = 'game_over') -> None:
        """Pick the next Phase B strategy.

        v4 improvement #3 — cross-attempt memory:
          - First full cycle (attempts 1..N_STRATEGIES): rotate deterministically
            from the archetype-default index, mirroring v3 behavior.
          - After a full rotation: prefer the strategy that has cleared the most
            levels in this game so far. Falls back to random if none has shown
            progress.
        """
        self._post_replay_reset_count += 1
        n_strategies = len(self.STRATEGIES)
        if self._post_replay_reset_count <= n_strategies:
            self._strategy_idx = (self._initial_strategy_idx + self._post_replay_reset_count) % n_strategies
            mode = 'rotation'
        else:
            best = None
            best_levels = 0
            for s, lv in self._strategy_levels.items():
                if lv > best_levels:
                    best_levels = lv
                    best = s
            if best is not None and best in self.STRATEGIES:
                self._strategy_idx = self.STRATEGIES.index(best)
                mode = 'exploit'
            else:
                self._strategy_idx = self._rng.randrange(n_strategies)
                mode = 'random'
        # Reset per-strategy cursors (existing + new v4 state).
        self._click_grid_idx = 0
        self._sustained_dir = None
        self._sustained_remaining = 0
        self._edge_sweep_idx = 0
        self._color_targeted_state = {}
        self._color_targeted_color_order = []
        self._color_targeted_color_idx = 0
        self._action_id_sweep_idx = 0
        self._action_id_sweep_count = 0
        self._frozen_streak = 0
        next_strategy = self.STRATEGIES[self._strategy_idx % len(self.STRATEGIES)]
        print(
            f'[MyAgent] {self._short_id} reset #{self._post_replay_reset_count} '
            f'reason={reason} mode={mode} -> strategy={next_strategy} '
            f'strategy_levels={dict(self._strategy_levels)}',
            flush=True,
        )

    # ---------------------- main per-step decision ----------------------

    def choose_action(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        try:
            if not self._debug_logged:
                self._debug_logged = True
                print(
                    f'[MyAgent] first action for {self._short_id} '
                    f'state={latest_frame.state} levels={getattr(latest_frame, "levels_completed", "?")} '
                    f'available_actions={getattr(latest_frame, "available_actions", "?")}',
                    flush=True,
                )

            # v3: for hidden games, resolve archetype + prior at first call
            # (we need the first frame for histogram matching).
            if self._is_hidden_game and not self._transfer_resolved:
                self._resolve_hidden_game_prior(latest_frame)

            # v3: record online action-effect from the previous transition.
            self._record_last_action_effect(frames, latest_frame)

            # v4 improvement #3: track per-strategy level progress.
            current_levels = int(getattr(latest_frame, 'levels_completed', 0) or 0)
            if current_levels > self._last_levels_seen:
                delta = current_levels - self._last_levels_seen
                cur_strat = self.STRATEGIES[self._strategy_idx % len(self.STRATEGIES)]
                self._strategy_levels[cur_strat] = self._strategy_levels.get(cur_strat, 0) + delta
                self._last_levels_seen = current_levels

            # v4 improvement #4: freeze detection.
            # Trigger only mid-episode (not during PHASE A replay or initial state),
            # not on GAME_OVER (that path advances strategy via the existing branch),
            # and only if we have replay-exhausted (so we're in Phase B).
            if (self._frozen_streak >= 8
                    and self._replay_idx >= len(self._replay)
                    and latest_frame.state not in (GameState.NOT_PLAYED, GameState.GAME_OVER, GameState.WIN)):
                print(
                    f'[MyAgent] {self._short_id} freeze detected '
                    f'(streak={self._frozen_streak}) → switching strategy',
                    flush=True,
                )
                self._advance_strategy(reason='freeze')

            # Phase A: replay-warm-start (v4 fix: tactical RESET preserved).
            #
            # Critical insight: in cd82 (and likely ft09, sb26, others), the
            # GT contains RESET markers in NOT_FINISHED state — the human is
            # using RESET as a TACTICAL move (e.g., "clear paint and try
            # again on this level") rather than a death-recovery action. Our
            # pre-v4 code skipped these tactical resets when env state was
            # NOT_FINISHED, causing the env to diverge from GT after the
            # first tactical reset (record 21 for cd82).
            #
            # v4 fix: ALWAYS issue RESET when the next replay entry is a
            # reset marker, regardless of env state. The env handles RESET
            # gracefully in any state.
            #
            # Plus the prior fix: if env enters NOT_PLAYED/GAME_OVER but the
            # NEXT replay entry isn't a reset, advance the cursor past the
            # next reset marker forward so we re-sync at the next attempt.
            while self._replay_idx < len(self._replay):
                entry = self._replay[self._replay_idx]
                if entry.get('type') == 'reset':
                    # Tactical RESET preserved — issue RESET and advance.
                    self._replay_idx += 1
                    action = GameAction.RESET
                    action.reasoning = {
                        'phase': 'warmstart',
                        'marker': 'reset',
                        'replay_idx': self._replay_idx,
                    }
                    self._last_returned_action = action
                    return action
                # Non-reset entry. If env is dead but GT didn't reset here,
                # advance to next reset marker + 1 to re-sync segments.
                if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                    next_reset = None
                    for i in range(self._replay_idx, len(self._replay)):
                        if self._replay[i].get('type') == 'reset':
                            next_reset = i
                            break
                    if next_reset is None:
                        self._replay_idx = len(self._replay)
                        break
                    self._replay_idx = next_reset + 1
                    action = GameAction.RESET
                    action.reasoning = {
                        'phase': 'warmstart',
                        'marker': 'reset',
                        'state_driven': True,
                        'segment_aligned_to': self._replay_idx,
                    }
                    self._last_returned_action = action
                    return action
                # State is healthy + entry is an action — play it.
                self._replay_idx += 1
                aid = int(entry['id'])
                try:
                    action = GameAction.from_id(aid)
                except Exception:
                    continue
                if action.is_complex():
                    action.set_data({'x': int(entry['x']), 'y': int(entry['y'])})
                    action.reasoning = {'phase': 'warmstart', 'replay_idx': self._replay_idx}
                elif action.is_simple():
                    action.reasoning = {'phase': 'warmstart', 'replay_idx': self._replay_idx}
                self._last_returned_action = action
                return action

            # Phase B: replay exhausted or hidden game.
            if latest_frame.state is GameState.NOT_PLAYED:
                action = GameAction.RESET
                action.reasoning = {'phase': 'fallback', 'marker': 'reset', 'reason': 'NOT_PLAYED'}
                self._last_returned_action = action
                return action
            if latest_frame.state is GameState.GAME_OVER:
                self._advance_strategy()
                action = GameAction.RESET
                action.reasoning = {
                    'phase': 'fallback',
                    'marker': 'reset',
                    'next_strategy': self.STRATEGIES[self._strategy_idx % len(self.STRATEGIES)],
                }
                self._last_returned_action = action
                return action

            action = self._strategy_action(latest_frame)
            self._last_returned_action = action
            return action

        except Exception as exc:
            print(
                f'[MyAgent] choose_action crashed at counter={self.action_counter}: '
                f'{type(exc).__name__}: {exc}',
                flush=True,
            )
            traceback.print_exc()
            action = _safe_random_action(self._rng, latest_frame)
            self._last_returned_action = action
            return action
