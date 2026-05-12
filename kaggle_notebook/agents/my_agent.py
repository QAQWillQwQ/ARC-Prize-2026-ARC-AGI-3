# =====================================================================
# WarmstartAgent v4 — replay-warm-start + per-game priors + visual
# saliency + online action-effect learning + v4 enrichments below.
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

# v4.3 TTT: try-import the vendored BC policy + TTT primitives. Fail-soft so
# the agent still runs without torch (e.g., during pure-replay smoke tests).
try:
    import bc_policy as _bc_policy
    _BC_POLICY_OK = True
except Exception as _bc_exc:
    _bc_policy = None  # type: ignore
    _BC_POLICY_OK = False
    print(f"[MyAgent] bc_policy import failed (TTT disabled): {_bc_exc}", flush=True)


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


# ---------------------- action-effect retrieval dictionary ----------------------
# Ported from src/action_effect_dict.py (load-only, no build path).
# Provides a cosine-similarity kNN lookup over ~15k GT replay transitions:
# given the current frame and a candidate (action_id, x, y), returns the
# typical frame_delta + level_prob + similarity. v4 hidden-mode strategies
# blend this into _score_action_candidate so coord/key picks bias toward
# the kind of moves the human demonstrators made productively.

try:
    import numpy as _np  # type: ignore
    _NUMPY_OK = True
except Exception:
    _np = None  # type: ignore
    _NUMPY_OK = False


_FEATURE_GRID = 64
_FEATURE_QUAD = 4
_FEATURE_NUM_COLORS = 16
_FEATURE_DIM = _FEATURE_QUAD * _FEATURE_QUAD * _FEATURE_NUM_COLORS  # 256


def _compute_feature_key(frame: Sequence[Sequence[int]]) -> Optional["_np.ndarray"]:
    """256-d per-quadrant color-histogram feature for a 64x64 frame."""
    if not _NUMPY_OK:
        return None
    try:
        arr = _np.asarray(frame, dtype=_np.int32)
        if arr.shape != (_FEATURE_GRID, _FEATURE_GRID):
            out = _np.zeros((_FEATURE_GRID, _FEATURE_GRID), dtype=_np.int32)
            h = min(arr.shape[0], _FEATURE_GRID) if arr.ndim >= 1 else 0
            w = min(arr.shape[1], _FEATURE_GRID) if arr.ndim >= 2 else 0
            if h and w:
                out[:h, :w] = arr[:h, :w]
            arr = out
        quad = _FEATURE_GRID // _FEATURE_QUAD  # 16
        feat = _np.zeros((_FEATURE_QUAD, _FEATURE_QUAD, _FEATURE_NUM_COLORS), dtype=_np.float32)
        for qy in range(_FEATURE_QUAD):
            for qx in range(_FEATURE_QUAD):
                patch = arr[qy * quad:(qy + 1) * quad, qx * quad:(qx + 1) * quad]
                patch_flat = _np.clip(patch.ravel(), 0, _FEATURE_NUM_COLORS - 1)
                counts = _np.bincount(patch_flat, minlength=_FEATURE_NUM_COLORS).astype(_np.float32)
                total = counts.sum()
                if total > 0:
                    counts /= total
                feat[qy, qx] = counts
        return feat.reshape(_FEATURE_DIM)
    except Exception:
        return None


class ActionEffectDictionary:
    """In-memory kNN retrieval over GT-replay (frame, action) → effect."""

    def __init__(
        self,
        feature_keys: "_np.ndarray",
        action_ids: "_np.ndarray",
        xs: "_np.ndarray",
        ys: "_np.ndarray",
        frame_deltas: "_np.ndarray",
        level_progresses: "_np.ndarray",
        game_id_idx: "_np.ndarray",
        game_ids: List[str],
    ) -> None:
        norms = _np.linalg.norm(feature_keys, axis=1, keepdims=True)
        norms = _np.where(norms < 1e-8, 1.0, norms)
        self.feature_keys = (feature_keys / norms).astype(_np.float32)
        self.action_ids = action_ids.astype(_np.int8)
        self.xs = xs.astype(_np.int8)
        self.ys = ys.astype(_np.int8)
        self.frame_deltas = frame_deltas.astype(_np.int32)
        self.level_progresses = level_progresses.astype(_np.int8)
        self.game_id_idx = game_id_idx.astype(_np.int16)
        self.game_ids = list(game_ids)
        self._action_masks: Dict[int, "_np.ndarray"] = {}
        for aid in range(1, 8):
            self._action_masks[aid] = (self.action_ids == aid)

    def __len__(self) -> int:
        return int(self.feature_keys.shape[0])

    def lookup(
        self,
        action_id: int,
        x: int = 0,
        y: int = 0,
        k: int = 5,
        coord_tolerance: int = 4,
        precomputed_query: Optional["_np.ndarray"] = None,
    ) -> Dict[str, float]:
        """Returns {'mean_delta', 'level_prob', 'n_matches', 'similarity'}.

        `precomputed_query` MUST be supplied (call `_compute_feature_key` once
        per frame and pass it here for every candidate). The frame argument is
        intentionally absent — recomputing the feature per candidate would
        dominate runtime.
        """
        zero_result = {"mean_delta": 0.0, "level_prob": 0.0, "n_matches": 0, "similarity": 0.0}
        if precomputed_query is None:
            return zero_result
        mask = self._action_masks.get(int(action_id))
        if mask is None or not mask.any():
            return zero_result
        if int(action_id) == 6:
            coord_mask = mask & (
                (_np.abs(self.xs.astype(_np.int32) - int(x)) <= coord_tolerance)
                & (_np.abs(self.ys.astype(_np.int32) - int(y)) <= coord_tolerance)
            )
            if coord_mask.any():
                mask = coord_mask
        n_matches = int(mask.sum())
        if n_matches == 0:
            return zero_result
        candidate_idxs = _np.where(mask)[0]
        sims = self.feature_keys[candidate_idxs] @ precomputed_query
        if sims.size == 0:
            return zero_result
        k_actual = min(k, sims.size)
        if k_actual <= 0:
            return zero_result
        top_local = _np.argpartition(-sims, k_actual - 1)[:k_actual]
        top_global = candidate_idxs[top_local]
        return {
            "mean_delta": float(self.frame_deltas[top_global].mean()),
            "level_prob": float(self.level_progresses[top_global].mean()),
            "n_matches": n_matches,
            "similarity": float(sims[top_local].mean()),
        }

    def top_clicks_by_similarity(
        self,
        query: "_np.ndarray",
        k: int = 10,
        prefer_progress: bool = True,
    ) -> List[Tuple[int, int]]:
        """Return up to `k` distinct ACTION6 coords from the most-similar
        dict entries to `query`. If `prefer_progress`, restrict to entries
        that produced level progress when at least k such entries exist;
        otherwise fall back to highest-similarity entries.
        """
        a6_mask = self._action_masks.get(6)
        if a6_mask is None or not a6_mask.any():
            return []
        if prefer_progress:
            prog_mask = a6_mask & (self.level_progresses > 0)
            if int(prog_mask.sum()) >= k:
                a6_mask = prog_mask
        a6_idxs = _np.where(a6_mask)[0]
        sims = self.feature_keys[a6_idxs] @ query
        if sims.size == 0:
            return []
        scan_k = min(k * 4, sims.size)
        top_local = _np.argpartition(-sims, scan_k - 1)[:scan_k]
        top_global = a6_idxs[top_local]
        # Dedup while preserving similarity-rank order.
        order = _np.argsort(-sims[top_local])
        seen: set = set()
        out: List[Tuple[int, int]] = []
        for j in order:
            i = int(top_global[int(j)])
            xy = (int(self.xs[i]), int(self.ys[i]))
            if xy in seen:
                continue
            seen.add(xy)
            out.append(xy)
            if len(out) >= k:
                break
        return out

    @classmethod
    def load(cls, path: Path) -> "ActionEffectDictionary":
        z = _np.load(path, allow_pickle=True)
        return cls(
            feature_keys=z["feature_keys"],
            action_ids=z["action_ids"],
            xs=z["xs"],
            ys=z["ys"],
            frame_deltas=z["frame_deltas"],
            level_progresses=z["level_progresses"],
            game_id_idx=z["game_id_idx"],
            game_ids=list(z["game_ids"].tolist()),
        )


def _load_action_effect_dict() -> Optional[ActionEffectDictionary]:
    """Try multiple paths for action_effect_dict.npz. Returns None on any failure.

    Set `ARC_DISABLE_EFFECT_DICT=1` to short-circuit and return None — useful
    for A/B comparing v4 (heuristic only) against v4.1 (heuristic + dict).
    """
    if not _NUMPY_OK:
        return None
    if str(os.environ.get('ARC_DISABLE_EFFECT_DICT', '')).strip() in ('1', 'true', 'TRUE', 'yes'):
        print('[MyAgent] effect_dict disabled via ARC_DISABLE_EFFECT_DICT env var', flush=True)
        return None
    candidates = [
        os.environ.get('ARC_EFFECT_DICT_PATH'),
        str(REPLAY_BASE_DIR.parent / 'action_effect_dict.npz'),
        str(REPLAY_BASE_DIR.parent / 'Local_Output' / 'action_effect_dict.npz'),
        '/kaggle/working/action_effect_dict.npz',
        '/kaggle/working/replays/action_effect_dict.npz',
        '/kaggle/input/arc-agi-3-replays-v1/action_effect_dict.npz',
    ]
    for path_str in candidates:
        if not path_str:
            continue
        p = Path(path_str)
        if p.is_file():
            try:
                d = ActionEffectDictionary.load(p)
                print(f'[MyAgent] effect_dict loaded ({len(d)} entries) from {p}', flush=True)
                return d
            except Exception as exc:
                print(f'[MyAgent] effect_dict load failed at {p}: {exc}', flush=True)
                continue
    return None


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
        # v4.2 additions — dict-driven exploit strategies. Inserted AFTER the
        # original 7 so existing rotation order + cross-attempt exploit memory
        # in _advance_strategy keep working unchanged.
        'dict_top_k_pursuit',      # idx 7 — click_dominant exploit
        'keyboard_dict_sweep',     # idx 8 — keyboard_dominant exploit
    )

    # Map archetype string (from per_game_priors) -> starting strategy_idx.
    # v4.2: archetype defaults updated to the new dict-driven strategies when
    # the dict is loaded — they're more focused exploit modes than the
    # explore-heavy click_grid / directional_sustained. _resolve_archetype_strategy
    # handles the dict-vs-no-dict branch at __init__ time.
    ARCHETYPE_TO_STRATEGY_IDX: Dict[str, int] = {
        'mixed': 0,
        'click_dominant': 2,
        'keyboard_dominant': 3,
    }
    # v4.2 hidden-mode A/B at 1000 steps showed dict_top_k_pursuit as the
    # initial strategy regresses on click-dominant games (too greedy for the
    # short budget — commits to one coord for 3+ visits before rotating).
    # Reverted to click_grid / directional_sustained as initial picks; the
    # new strategies stay in the rotation via _advance_strategy and become
    # active later in the cycle when click_grid has already explored.
    ARCHETYPE_TO_STRATEGY_IDX_DICT: Dict[str, int] = {
        'mixed': 0,
        'click_dominant': 2,        # click_grid (was 7 dict_top_k_pursuit)
        'keyboard_dominant': 3,     # directional_sustained (was 8 keyboard_dict_sweep)
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
        self._effect_dict: Optional[ActionEffectDictionary] = _load_action_effect_dict()
        # Per-frame feature-vector cache. Keyed by id(latest_frame) so we
        # compute the 256-d query feature only once per choose_action call.
        self._cached_query_key: Optional[int] = None
        self._cached_query_vec: Optional[Any] = None  # numpy array or None
        self._game_prior: Dict[str, Any] = self._all_priors.get(self._short_id, {})
        self._is_hidden_game = not bool(self._game_prior)
        self._transfer_resolved = False  # set True once we attempt resolution
        archetype = str(self._game_prior.get('archetype', 'mixed'))
        # v4.2: when dict is loaded, prefer the dict-driven strategy for the
        # archetype's exploit slot. Falls back to the v3/v4 strategy when
        # the dict is missing.
        archetype_map = (
            self.ARCHETYPE_TO_STRATEGY_IDX_DICT if self._effect_dict is not None
            else self.ARCHETYPE_TO_STRATEGY_IDX
        )
        self._initial_strategy_idx = archetype_map.get(archetype, 0)
        # v4.4 random-first bias rate (see _advance_strategy). Hidden-only by
        # default. Override via env ARC_RANDOM_FIRST_BIAS=0.5 (or any 0..1).
        # 0.0 disables, 1.0 always picks random_full on reset.
        try:
            self._random_first_bias = float(os.environ.get(
                'ARC_RANDOM_FIRST_BIAS',
                '0.5' if self._is_hidden_game else '0.0',
            ))
        except (TypeError, ValueError):
            self._random_first_bias = 0.5 if self._is_hidden_game else 0.0

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

        # v4.2 #1: coord-visit memory. Persists across GAME_OVER resets so the
        # agent stops re-clicking the same dead coords cycle after cycle.
        # Cleared only on level-clear (forward progress).
        from collections import Counter, deque
        self._clicked_points: Counter = Counter()             # exact (x, y) visit count
        self._clicked_bins: Counter = Counter()               # (x//8, y//8) visit count
        # v4.2 #2: frame-hash novelty memory. Rolling window of post-action
        # frame hashes; when novelty rate < 0.3 the strategy switcher forces
        # a rotation to break the cycle.
        self._seen_frame_hashes: "deque" = deque(maxlen=200)
        self._novelty_window_actions: int = 0
        self._novelty_window_nonnovel: int = 0
        # v4.2 #4: dict_top_k_pursuit cursor state.
        self._pursuit_pool: List[Tuple[int, int]] = []        # current top-k coords from dict
        self._pursuit_cursor: int = 0                          # which coord we're committing to
        self._pursuit_max_visits: int = 3                      # rotate when this coord visited > N
        self._pursuit_actions_since_refresh: int = 0          # frames between pool rebuilds
        self._pursuit_refresh_interval: int = 20              # rebuild pool every N pursuit actions
        # v4.2 #5: keyboard_dict_sweep state — like directional_sustained but
        # picks the action with the highest dict score per available set.
        self._kbd_dict_aid: Optional[int] = None
        self._kbd_dict_remaining: int = 0
        # v4.3 TTT: load BC checkpoint (lazy — only if bc_policy is available
        # AND a checkpoint can be found). Run TTT once before first action of
        # this game (after we know whether replays are available).
        self._policy: Any = None
        self._policy_loaded: bool = False
        self._ttt_done: bool = False
        # v4.3 TTT (Gen 1 enhancement): rolling buffer of (history_frames, action,
        # next_frame) triples captured during the agent's first ~30 actions in
        # the game. Used by ttt_rollout_finetune to adapt the encoder to actual
        # game dynamics. Triggered at TTT_ROLLOUT_TRIGGER_STEP via choose_action.
        from collections import deque as _deque
        self._rollout_buffer_history: List[List[List[List[int]]]] = []
        self._rollout_buffer_actions: List[Dict[str, Any]] = []
        self._rollout_buffer_next_frames: List[List[List[int]]] = []
        self._rollout_ttt_done: bool = False
        self.TTT_ROLLOUT_TRIGGER_STEP = 30
        self.TTT_ROLLOUT_BUFFER_MAX = 50
        ttt_disabled = str(os.environ.get('ARC_DISABLE_TTT', '')).strip() in ('1', 'true', 'TRUE', 'yes')
        if _BC_POLICY_OK and not ttt_disabled:
            ckpt_path = _bc_policy.find_checkpoint()
            if ckpt_path is not None:
                self._policy = _bc_policy.PolicyHelper.load(ckpt_path)
                self._policy_loaded = self._policy is not None
            else:
                print('[MyAgent] no BC checkpoint found; TTT disabled', flush=True)
        elif ttt_disabled:
            print('[MyAgent] TTT disabled via ARC_DISABLE_TTT env var', flush=True)

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

    @staticmethod
    def _frame_hash_str(frame: Sequence[Sequence[int]]) -> str:
        """Stable hash of a 64x64 frame. Uses bytes of the flattened uint8
        array — fast (~50us) and collision-safe for our 16-color, 4096-cell
        grids. Falls back to Python hash() on any frame-shape oddity.
        """
        try:
            if _NUMPY_OK:
                arr = _np.asarray(frame, dtype=_np.uint8)
                return arr.tobytes().hex()
            # Numpy-less fallback (Kaggle always has numpy; this is just defensive).
            flat = []
            for row in frame:
                for c in row:
                    flat.append(int(c))
            return repr(tuple(flat))
        except Exception:
            try:
                return repr(frame)
            except Exception:
                return ""

    def _record_last_action_effect(self, frames: List[FrameData], latest_frame: FrameData) -> None:
        """If we have a previous frame and a previously-returned action,
        compute frame_delta and update the online action-effects map.

        Also bumps the freeze streak (consecutive zero-delta steps) used by
        choose_action's freeze-detection branch.

        v4.2 #1: also records the click coord into _clicked_points / _clicked_bins
        so future scoring can penalize re-visits across resets.
        v4.2 #2: also hashes the post-action frame and tracks novelty rate
        in a 200-step window. choose_action consults the rate to force
        strategy switches when cycling.
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
            # v4.2 #1: cross-attempt coord visit memory (only for ACTION6).
            self._clicked_points[(x, y)] += 1
            self._clicked_bins[(x // 8, y // 8)] += 1
        else:
            key = (aid, 0, 0)
        c, m = self._action_effects.get(key, (0, 0.0))
        n_new = c + 1
        m_new = m + (delta - m) / n_new
        self._action_effects[key] = (n_new, m_new)

        # v4.2 #2: post-action frame-hash novelty.
        h = self._frame_hash_str(cur)
        self._novelty_window_actions += 1
        if h and h in self._seen_frame_hashes:
            self._novelty_window_nonnovel += 1
        if h:
            self._seen_frame_hashes.append(h)
        # Periodic window reset to avoid stale rates after major progress.
        if self._novelty_window_actions >= 100:
            self._novelty_window_actions = 0
            self._novelty_window_nonnovel = 0

    def _on_level_clear(self) -> None:
        """v4.2: clear the visit/novelty memories on real progress so the
        agent doesn't keep penalizing coords that worked. Called from
        choose_action when current_levels > _last_levels_seen.
        """
        if self._clicked_points:
            self._clicked_points.clear()
        if self._clicked_bins:
            self._clicked_bins.clear()
        self._seen_frame_hashes.clear()
        self._novelty_window_actions = 0
        self._novelty_window_nonnovel = 0

    def _do_ttt_once(self, latest_frame: FrameData) -> None:
        """Stage 1 TTT — runs ONCE before first action.

        Only fires `replay_finetune` if we have a GT replay (public games).
        For hidden games, this is a no-op — wait for `_do_rollout_ttt` at
        TTT_ROLLOUT_TRIGGER_STEP after we've collected real (s, a, s') pairs.
        """
        if self._ttt_done or not self._policy_loaded or self._policy is None:
            return
        self._ttt_done = True
        try:
            if self._replay:
                t0 = time.time()
                stats_a = _bc_policy.ttt_replay_finetune(
                    self._policy, self._replay, n_steps=30, batch_size=8, lr=1e-4,
                )
                print(
                    f'[MyAgent] {self._short_id} TTT replay_finetune: '
                    f'steps={stats_a.get("steps",0)} '
                    f'first_loss={stats_a.get("first_loss",0):.3f} '
                    f'final_loss={stats_a.get("final_loss",0):.3f} '
                    f'n_actions={stats_a.get("n_actions",0)} '
                    f'wall={time.time()-t0:.2f}s',
                    flush=True,
                )
            else:
                print(
                    f'[MyAgent] {self._short_id} TTT replay skipped (no replay); '
                    f'will run rollout TTT at step {self.TTT_ROLLOUT_TRIGGER_STEP}',
                    flush=True,
                )
        except Exception as exc:
            print(f'[MyAgent] {self._short_id} TTT replay failed (continuing): {exc}', flush=True)
            traceback.print_exc()

    def _do_rollout_ttt(self, latest_frame: FrameData) -> None:
        """Stage 2 TTT — runs ONCE at TTT_ROLLOUT_TRIGGER_STEP using buffered rollout.

        Adapts the model to the actual game dynamics observed during the first
        30 actions. Uses ttt_rollout_finetune (action CE + coord CE + next-frame
        recon CE). No-op if buffer is too small or policy isn't loaded.
        """
        if self._rollout_ttt_done or not self._policy_loaded or self._policy is None:
            return
        if len(self._rollout_buffer_actions) < 8:  # need at least one batch
            return
        self._rollout_ttt_done = True
        try:
            t0 = time.time()
            stats = _bc_policy.ttt_rollout_finetune(
                self._policy,
                history_buffer=self._rollout_buffer_history,
                action_buffer=self._rollout_buffer_actions,
                next_frame_buffer=self._rollout_buffer_next_frames,
                goal_frame=None,  # Gen 2: predict goal from rollout
                n_steps=30, batch_size=8, lr=5e-5,
            )
            print(
                f'[MyAgent] {self._short_id} TTT rollout_finetune: '
                f'steps={stats.get("steps",0)} '
                f'first_loss={stats.get("first_loss",0):.3f} '
                f'final_loss={stats.get("final_loss",0):.3f} '
                f'n_pairs={stats.get("n_pairs",0)} '
                f'wall={time.time()-t0:.2f}s',
                flush=True,
            )
            # Drop the buffer to free memory; we won't re-trigger this generation.
            self._rollout_buffer_history = []
            self._rollout_buffer_actions = []
            self._rollout_buffer_next_frames = []
        except Exception as exc:
            print(f'[MyAgent] {self._short_id} TTT rollout failed (continuing): {exc}', flush=True)
            traceback.print_exc()

    def _capture_rollout_pair(self, frames: List[FrameData], latest_frame: FrameData) -> None:
        """Record (history, last_action, next_frame) into the rollout buffer.

        Called from choose_action right BEFORE picking the next action so we
        capture the result of the LAST returned action.
        """
        if (self._rollout_ttt_done
                or not self._policy_loaded
                or self._last_returned_action is None
                or len(frames) < 2):
            return
        if len(self._rollout_buffer_actions) >= self.TTT_ROLLOUT_BUFFER_MAX:
            return
        try:
            prev_history = self._collect_recent_frames(latest_frame)
            cur_grid = _flatten_frame(getattr(latest_frame, 'frame', None))
            if not prev_history or cur_grid is None:
                return
            try:
                aid = int(self._last_returned_action.value if hasattr(self._last_returned_action, 'value') else 0)
            except Exception:
                aid = 0
            if aid == 0 or aid > 7:
                return  # Skip RESET / invalid
            x, y = 0, 0
            if aid == 6:
                try:
                    d = self._last_returned_action.action_data.model_dump() if hasattr(self._last_returned_action.action_data, 'model_dump') else (self._last_returned_action.action_data or {})
                    x = int((d or {}).get('x', 0)); y = int((d or {}).get('y', 0))
                except Exception:
                    pass
            self._rollout_buffer_history.append([list(map(list, f)) for f in prev_history])
            self._rollout_buffer_actions.append({"action_id": aid, "action_data": {"x": x, "y": y} if aid == 6 else {}})
            self._rollout_buffer_next_frames.append([list(row) for row in cur_grid])
        except Exception:
            pass

    def _collect_recent_frames(self, latest_frame: FrameData) -> List[List[List[int]]]:
        """Pull the last `history` frames as raw List[List[int]] grids."""
        history = self._policy.history if self._policy is not None else 4
        out: List[List[List[int]]] = []
        for f in self.frames[-history:]:
            grid = _flatten_frame(getattr(f, 'frame', None))
            if grid is not None:
                out.append(grid)
        if not out:
            cur = _flatten_frame(getattr(latest_frame, 'frame', None))
            if cur is not None:
                out.append(cur)
        return out

    def _policy_score_current(self, latest_frame: FrameData) -> Optional[Dict[str, Any]]:
        """One BC forward pass for the current frame. Returns None when policy
        isn't loaded or scoring fails. Caches by id(latest_frame) so multiple
        candidate evaluations in one choose_action don't repeat the forward.
        """
        if not self._policy_loaded or self._policy is None:
            return None
        key = id(latest_frame)
        cached = getattr(self, '_policy_score_cache_key', None)
        if cached == key:
            return getattr(self, '_policy_score_cache_value', None)
        history = self._collect_recent_frames(latest_frame)
        cur_frame = _flatten_frame(getattr(latest_frame, 'frame', None))
        if cur_frame is None or not history:
            return None
        scores = self._policy.score_frame(
            history_frames=history,
            latest_frame=cur_frame,
            last_action_id=None,
            levels_completed=int(getattr(latest_frame, 'levels_completed', 0) or 0),
            steps_since_progress=int(self._frozen_streak),
            step_index=int(getattr(self, 'action_counter', 0) or 0),
            available_actions=_available_action_ids(latest_frame),
        )
        self._policy_score_cache_key = key
        self._policy_score_cache_value = scores
        return scores

    def _novelty_rate(self) -> float:
        """Fraction of the last window's transitions that produced a novel
        post-action frame. 1.0 = all novel; 0.0 = all repeats. Returns 1.0
        until at least 20 transitions have been recorded.
        """
        if self._novelty_window_actions < 20:
            return 1.0
        return 1.0 - (self._novelty_window_nonnovel / max(1, self._novelty_window_actions))

    def _get_current_query_vec(self, latest_frame: FrameData) -> Optional[Any]:
        """Compute (and cache) the 256-d query feature for `latest_frame`.

        Cached by id(latest_frame) so callers can hit it once per
        `choose_action`. Returns None when numpy is unavailable, the dict
        isn't loaded, or the frame can't be decoded.
        """
        if self._effect_dict is None or not _NUMPY_OK:
            return None
        key = id(latest_frame)
        if key == self._cached_query_key:
            return self._cached_query_vec
        self._cached_query_key = key
        self._cached_query_vec = None
        frame = _flatten_frame(getattr(latest_frame, 'frame', None))
        if frame is None:
            return None
        feat = _compute_feature_key(frame)
        if feat is None:
            return None
        n = float(_np.linalg.norm(feat))
        if n < 1e-8:
            return None
        self._cached_query_vec = (feat / n).astype(_np.float32)
        return self._cached_query_vec

    def _dict_lookup_score(
        self,
        action_id: int,
        x: int,
        y: int,
        latest_frame: FrameData,
    ) -> float:
        """Score in [0.0, 1.0] from action-effect dict kNN lookup.

        Blends three signals:
          - mean_delta normalized (>50 px change is "did something")
          - level_prob × 10 (any positive level rate is a strong bias)
          - cosine similarity (top-k closeness to query frame)

        Returns 0.0 when dict is missing, query can't be computed, or no
        matches were found for this (action_id, coord) query.
        """
        if self._effect_dict is None:
            return 0.0
        qv = self._get_current_query_vec(latest_frame)
        if qv is None:
            return 0.0
        try:
            res = self._effect_dict.lookup(int(action_id), int(x), int(y), k=5, precomputed_query=qv)
        except Exception:
            return 0.0
        if int(res.get('n_matches', 0)) == 0:
            return 0.0
        delta_norm = min(1.0, float(res['mean_delta']) / 50.0)
        level_bonus = min(1.0, float(res['level_prob']) * 10.0)
        sim = max(0.0, min(1.0, float(res['similarity'])))
        return 0.4 * delta_norm + 0.5 * level_bonus + 0.1 * sim

    def _score_action_candidate(
        self,
        action_id: int,
        x: int = 0,
        y: int = 0,
        latest_frame: Optional[FrameData] = None,
    ) -> float:
        """Online action-effect score for a candidate. Used by click_grid and
        directional_sustained re-rankings.

        Online score in [0.1, 1.0]:
          - 1.0  → never tried (explore)
          - 0.1  → tried but mean delta < 1 pixel (known dead — strongly avoid)
          - else → mean_delta normalized, capped at 1.0

        When `latest_frame` is provided AND the action-effect dict is loaded,
        blends a dict-based prior into the score (50/50 by default). The
        dict signal lifts both new candidates (no online data yet) and
        re-confirms productive coords the demonstrators used.
        """
        if int(action_id) == 6:
            key = (6, int(x) // 8, int(y) // 8)
        else:
            key = (int(action_id), 0, 0)
        count, mean_delta = self._action_effects.get(key, (0, 0.0))
        if count == 0:
            online_score = 1.0
        elif mean_delta < 1.0:
            online_score = 0.1
        else:
            online_score = min(1.0, mean_delta / 10.0)

        if latest_frame is None or self._effect_dict is None:
            blended = online_score
        else:
            dict_score = self._dict_lookup_score(action_id, x, y, latest_frame)
            if dict_score <= 0.0:
                # No retrieved evidence — defer fully to online signal.
                blended = online_score
            elif count == 0:
                # Never tried — dict prior dominates.
                blended = 0.3 * online_score + 0.7 * dict_score
            else:
                # Sampled — online signal dominates with dict still influencing.
                blended = 0.6 * online_score + 0.4 * dict_score

        # v4.3 TTT: blend BC policy score (post-adaptation) into the candidate
        # ranking. The TTT-adapted model is per-game specialized, so it can
        # contribute meaningful per-action and per-coord guidance even on
        # hidden games. Weight 0.3 — significant but not overwhelming.
        if latest_frame is not None and self._policy_loaded and _BC_POLICY_OK:
            scores = self._policy_score_current(latest_frame)
            if scores is not None:
                aid_idx = _bc_policy.ACTION_TO_INDEX.get(int(action_id))
                if aid_idx is not None:
                    policy_action = float(scores['action_scores'][aid_idx])
                    if int(action_id) == 6:
                        policy_coord = float(scores['x_scores'][int(x)] * scores['y_scores'][int(y)] * 64.0)
                        # x_scores * y_scores is in [0, 1/64²] so × 64 normalizes a perfect single-cell hit to ~1.
                        policy_combined = 0.5 * policy_action + 0.5 * min(1.0, policy_coord)
                    else:
                        policy_combined = policy_action
                    blended = 0.7 * blended + 0.3 * policy_combined

        # v4.2 #1: subtract a visit-count penalty for ACTION6 candidates so
        # the agent stops re-clicking coords it has already burned across
        # GAME_OVER cycles. Penalty grows from 0 (no visits) toward 0.225
        # (many visits), so unvisited coords retain their full blended score
        # while saturated coords get nudged down enough to lose on tie.
        # 0.15 * v/(1+v): 0 at v=0, 0.075 at v=1, 0.10 at v=2, 0.125 at v=3, …
        if int(action_id) == 6:
            point_visits = self._clicked_points.get((int(x), int(y)), 0)
            bin_visits = self._clicked_bins.get((int(x) // 8, int(y) // 8), 0)
            point_pen = 0.15 * point_visits / (1.0 + point_visits)
            bin_pen = 0.075 * bin_visits / (1.0 + bin_visits)
            blended = blended - (point_pen + bin_pen)
        return blended

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
        # v4.1 (action-effect dict): widen the candidate pool with two extra
        # sources when the dict is loaded — (a) EDGE_SWEEP_POINTS, (b) up to
        # 10 coords retrieved from dict entries with the most-similar frame
        # AND positive level progress (i.e., "what coords cleared a level
        # from a frame like this in the human replays?"). Then rerank the
        # whole pool with the blended online + dict score.
        if self._effect_dict is not None:
            seen = set(self._click_grid_points)
            pool = list(self._click_grid_points)
            for p in EDGE_SWEEP_POINTS:
                if p not in seen:
                    seen.add(p)
                    pool.append(p)
            qv = self._get_current_query_vec(latest_frame)
            if qv is not None:
                try:
                    demo_coords = self._effect_dict.top_clicks_by_similarity(qv, k=10)
                except Exception:
                    demo_coords = []
                for p in demo_coords:
                    if p not in seen:
                        seen.add(p)
                        pool.append(p)
        else:
            pool = self._click_grid_points
        scored = [(p, self._score_action_candidate(6, p[0], p[1], latest_frame=latest_frame)) for p in pool]
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
                online_w = self._score_action_candidate(aid, latest_frame=latest_frame)
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

    def _strategy_dict_top_k_pursuit(self, latest_frame: FrameData) -> GameAction:
        """v4.2 #4 strategy: deterministic exploit of the dict's top click matches.

        Different from click_grid (weighted-random over top-5): this commits
        to ONE coord at a time, only rotating after the visit count for that
        coord exceeds _pursuit_max_visits. For games where one specific click
        clears the level, this beats sample-from-top-K which spreads attempts.

        Falls back to click_grid when the dict isn't loaded or the query
        vector can't be computed.
        """
        available = _available_action_ids(latest_frame)
        if 6 not in available:
            return self._strategy_keyboard_only(latest_frame)
        if self._effect_dict is None:
            return self._strategy_click_grid(latest_frame)
        qv = self._get_current_query_vec(latest_frame)
        if qv is None:
            return self._strategy_click_grid(latest_frame)

        # (Re)build pool when empty, fully visited, OR every refresh_interval
        # actions. The frame-conditioned similarity matters: an early-game
        # pool stays committed to coords that fit the starting frame even
        # after the game has visibly changed. Periodic refresh keeps the
        # recommendations grounded in the current state.
        self._pursuit_actions_since_refresh += 1
        needs_refresh = (
            not self._pursuit_pool
            or self._pursuit_cursor >= len(self._pursuit_pool)
            or self._pursuit_actions_since_refresh >= self._pursuit_refresh_interval
        )
        if needs_refresh:
            try:
                self._pursuit_pool = self._effect_dict.top_clicks_by_similarity(
                    qv, k=20, prefer_progress=True
                )
            except Exception:
                self._pursuit_pool = []
            self._pursuit_cursor = 0
            self._pursuit_actions_since_refresh = 0
        if not self._pursuit_pool:
            return self._strategy_click_grid(latest_frame)

        # Advance cursor past coords already burned this run.
        scanned = 0
        while scanned < len(self._pursuit_pool):
            x, y = self._pursuit_pool[self._pursuit_cursor % len(self._pursuit_pool)]
            visits = self._clicked_points.get((int(x), int(y)), 0)
            if visits < self._pursuit_max_visits:
                break
            self._pursuit_cursor += 1
            scanned += 1
        x, y = self._pursuit_pool[self._pursuit_cursor % len(self._pursuit_pool)]
        # Don't auto-advance the cursor — we want to repeat the same coord
        # several times unless it saturates. Visit-count gating handles rotation.

        try:
            action = GameAction.ACTION6
            action.set_data({'x': int(x), 'y': int(y)})
        except Exception:
            return _safe_random_action(self._rng, latest_frame)
        action.reasoning = {
            'phase': 'fallback',
            'strategy': 'dict_top_k_pursuit',
            'point': [int(x), int(y)],
            'cursor': self._pursuit_cursor,
            'pool_size': len(self._pursuit_pool),
        }
        return action

    def _strategy_keyboard_dict_sweep(self, latest_frame: FrameData) -> GameAction:
        """v4.2 #5 strategy: dict-driven action selection for keyboard games.

        Like directional_sustained but the action choice is dict-conditional
        on the current frame instead of relying on per-game prior's
        repeat_kept_actions (which doesn't transfer to hidden games). Still
        repeats the chosen action 3-5 times before reconsidering.

        Falls back to directional_sustained when the dict isn't loaded.
        """
        available = [a for a in _available_action_ids(latest_frame) if 1 <= a <= 5]
        if not available:
            return _safe_random_action(self._rng, latest_frame)
        if self._effect_dict is None:
            return self._strategy_directional_sustained(latest_frame)
        if self._kbd_dict_remaining > 0 and self._kbd_dict_aid in available:
            aid = int(self._kbd_dict_aid)
            self._kbd_dict_remaining -= 1
        else:
            qv = self._get_current_query_vec(latest_frame)
            if qv is None:
                return self._strategy_directional_sustained(latest_frame)
            scored: List[Tuple[int, float]] = []
            for a in available:
                try:
                    res = self._effect_dict.lookup(int(a), k=5, precomputed_query=qv)
                except Exception:
                    res = {"mean_delta": 0.0, "level_prob": 0.0, "n_matches": 0, "similarity": 0.0}
                if int(res.get('n_matches', 0)) == 0:
                    score = 0.0
                else:
                    # Heavier weight on level_prob — this is keyboard-game
                    # specific where most actions visibly change the frame
                    # but only a few make level progress.
                    score = (
                        0.6 * min(1.0, float(res['level_prob']) * 10.0)
                        + 0.3 * min(1.0, float(res['mean_delta']) / 100.0)
                        + 0.1 * max(0.0, min(1.0, float(res['similarity'])))
                    )
                scored.append((a, score))
            scored.sort(key=lambda kv: -kv[1])
            if scored and scored[0][1] > 0.0:
                aid = int(scored[0][0])
            else:
                aid = self._rng.choice(available)
            self._kbd_dict_aid = aid
            self._kbd_dict_remaining = self._rng.randint(3, 5) - 1
        try:
            action = GameAction.from_id(aid)
        except Exception:
            return _safe_random_action(self._rng, latest_frame)
        action.reasoning = {
            'phase': 'fallback',
            'strategy': 'keyboard_dict_sweep',
            'aid': aid,
            'remaining': self._kbd_dict_remaining,
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
        if strategy == 'dict_top_k_pursuit':
            return self._strategy_dict_top_k_pursuit(latest_frame)
        if strategy == 'keyboard_dict_sweep':
            return self._strategy_keyboard_dict_sweep(latest_frame)
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
        # v4.4 random-first bias on hidden games. Kaggle baselines showed
        # v1 pure random=0.17 > v2/v3 v4-strategy bank=0.08-0.10, so the
        # structured strategies have been net-negative on hidden. Bias
        # toward STRATEGIES[0] (random_full) with prob `_random_first_bias`
        # (env ARC_RANDOM_FIRST_BIAS, default 0.5 on hidden, 0.0 on known).
        # Falls through to the prior rotation/exploit logic otherwise so
        # structured strategies still get tried half the time.
        if self._is_hidden_game and self._rng.random() < self._random_first_bias:
            self._strategy_idx = 0
            mode = 'random_bias'
        elif self._post_replay_reset_count <= n_strategies:
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
        # Reset per-strategy cursors (existing + new v4/v4.2 state).
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
        # v4.2: clear pursuit + kbd-dict cursors so new strategy starts fresh.
        # NOTE: _clicked_points / _clicked_bins / _seen_frame_hashes are NOT
        # cleared here — visit + novelty memories persist across attempts so
        # the agent doesn't re-burn the same coords. _on_level_clear() clears
        # those on real progress.
        self._pursuit_pool = []
        self._pursuit_cursor = 0
        self._pursuit_actions_since_refresh = 0
        self._kbd_dict_aid = None
        self._kbd_dict_remaining = 0
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

            # v4.3 TTT (Stage 1): replay-finetune at action 0 for public games.
            if not self._ttt_done and self._policy_loaded:
                self._do_ttt_once(latest_frame)
            # v4.3 TTT (Stage 2): rollout TTT — capture (s, a, s') pairs into a
            # buffer, then trigger ttt_rollout_finetune at TTT_ROLLOUT_TRIGGER_STEP.
            # This is the critical path for HIDDEN games where no replay exists.
            if self._policy_loaded and not self._rollout_ttt_done:
                self._capture_rollout_pair(frames, latest_frame)
                # Trigger condition: enough buffered pairs collected. Don't rely on
                # action_counter (the framework may not advance it before each
                # choose_action call). Buffer-size-driven is more robust.
                if len(self._rollout_buffer_actions) >= self.TTT_ROLLOUT_TRIGGER_STEP:
                    self._do_rollout_ttt(latest_frame)

            # v3: record online action-effect from the previous transition.
            self._record_last_action_effect(frames, latest_frame)

            # v4 improvement #3: track per-strategy level progress.
            current_levels = int(getattr(latest_frame, 'levels_completed', 0) or 0)
            if current_levels > self._last_levels_seen:
                delta = current_levels - self._last_levels_seen
                cur_strat = self.STRATEGIES[self._strategy_idx % len(self.STRATEGIES)]
                self._strategy_levels[cur_strat] = self._strategy_levels.get(cur_strat, 0) + delta
                self._last_levels_seen = current_levels
                # v4.2: real progress — clear coord-visit + frame-novelty
                # memories so the agent doesn't keep avoiding what just worked.
                self._on_level_clear()

            # v4 improvement #4: freeze detection.
            # Trigger only mid-episode (not during PHASE A replay or initial state),
            # not on GAME_OVER (that path advances strategy via the existing branch),
            # and only if we have replay-exhausted (so we're in Phase B).
            # v4.2 #2: also force a switch when novelty rate is low (the agent
            # is cycling through the same frame states without progress).
            in_phase_b = (
                self._replay_idx >= len(self._replay)
                and latest_frame.state not in (GameState.NOT_PLAYED, GameState.GAME_OVER, GameState.WIN)
            )
            novelty_low = self._novelty_rate() < 0.3
            if in_phase_b and (self._frozen_streak >= 8 or novelty_low):
                reason = 'freeze' if self._frozen_streak >= 8 else 'low_novelty'
                print(
                    f'[MyAgent] {self._short_id} {reason} detected '
                    f'(streak={self._frozen_streak} novelty={self._novelty_rate():.2f}) → switching strategy',
                    flush=True,
                )
                self._advance_strategy(reason=reason)

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
