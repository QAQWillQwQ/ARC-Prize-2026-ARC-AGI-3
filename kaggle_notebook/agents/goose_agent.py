# =====================================================================
# GooseAgent — online change-reward CNN policy
#
# Re-implementation of the StochasticGoose pattern from the public ARC
# AGI 3 sample submission (Tufa Labs, Smit + Cole). Same core idea:
#   - small CNN encodes 16-color one-hot 64x64 frame
#   - action head (6 logits for ACTION{1,2,3,4,5,7}) + coord head (64x64 = 4096 logits for ACTION6)
#   - intrinsic reward: "did the frame change after my action?"
#   - online BCE training every train_frequency steps on (selected_logit, reward)
#   - reset model + optimizer at every new level
#
# Differences vs the public sample:
#   - 7 game actions, not 5 (the sample dropped ACTION7).
#   - Pluggable deltas via env vars so we can run T1..T4 experiments
#     against the T0 anchor without forking the file.
#   - Same MyAgent base class our framework uses (compatibility with
#     scripts/test_my_agent_local.py and the Kaggle gateway runner).
# =====================================================================
from __future__ import annotations

import hashlib
import os
import random
import time
import traceback
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


# Action IDs that are NOT click (ACTION6 is the click). We treat these as
# the simple action head's 6 outputs, in this fixed order so action_idx
# maps to an action id deterministically.
SIMPLE_ACTION_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 7)
NUM_SIMPLE_ACTIONS = len(SIMPLE_ACTION_IDS)
GRID_SIZE = 64
NUM_COLORS = 16
NUM_COORDS = GRID_SIZE * GRID_SIZE


def _env_flag(name: str, default: bool = False) -> bool:
    return str(os.environ.get(name, '1' if default else '')).strip() in ('1', 'true', 'TRUE', 'yes')


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _available_action_ids(latest_frame: FrameData) -> List[int]:
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


def _final_subframe(frame_data: FrameData) -> np.ndarray:
    """Pull the last 64x64 grid from FrameData.frame (list of stacked grids)."""
    arr = np.asarray(frame_data.frame, dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr


class ActionModel(nn.Module):
    """Plain 4-conv CNN with action + spatial coord heads.

    Identical conv shape family to the published StochasticGoose model
    (16 -> 32 -> 64 -> 128 -> 256, stride 1 everywhere). Coord head is a
    fully-convolutional spatial decoder that outputs a 64x64 logit map.
    """

    def __init__(
        self,
        input_channels: int = NUM_COLORS,
        grid_size: int = GRID_SIZE,
        num_simple_actions: int = NUM_SIMPLE_ACTIONS,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_simple_actions = num_simple_actions

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Simple-action head
        self.action_pool = nn.MaxPool2d(4, 4)
        action_flat = 256 * (grid_size // 4) * (grid_size // 4)
        self.action_fc = nn.Linear(action_flat, 512)
        self.action_head = nn.Linear(512, num_simple_actions)

        # Coord head (64x64 spatial logits for ACTION6)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return concatenated logits: [simple_actions (6), coord_flat (4096)] = (B, 4102)."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feats = F.relu(self.conv4(x))  # (B, 256, 64, 64)

        a = self.action_pool(feats)
        a = a.view(a.size(0), -1)
        a = F.relu(self.action_fc(a))
        a = self.dropout(a)
        action_logits = self.action_head(a)  # (B, 6)

        c = F.relu(self.coord_conv1(feats))
        c = F.relu(self.coord_conv2(c))
        c = F.relu(self.coord_conv3(c))
        coord_logits = self.coord_conv4(c).view(c.size(0), -1)  # (B, 4096)

        return torch.cat([action_logits, coord_logits], dim=1)


class GooseAgent(Agent):
    """Online change-reward CNN policy. Resets per level.

    Pluggable deltas via env vars:
      ARC_GOOSE_DELTA   = 'none' | 't1_bc' | 't2_gru' | 't3_priors' | 't4_goal'
      ARC_GOOSE_LR      = AdamW lr (default 1e-4)
      ARC_GOOSE_TRAIN_EVERY = train_frequency (default 5)
      ARC_GOOSE_BUFFER  = experience buffer maxlen (default 200000)
      ARC_GOOSE_BATCH   = train batch size (default 64)
    """

    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = (int(time.time() * 1_000_000) + hash(self.game_id)) & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))
        self._rng = random.Random(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.delta = os.environ.get('ARC_GOOSE_DELTA', 'none').strip().lower()
        self.lr = _env_float('ARC_GOOSE_LR', 1e-4)
        self.train_every = int(_env_float('ARC_GOOSE_TRAIN_EVERY', 5))
        self.buffer_maxlen = int(_env_float('ARC_GOOSE_BUFFER', 200000))
        self.batch_size = int(_env_float('ARC_GOOSE_BATCH', 64))

        self.short_id = self.game_id.split('-', 1)[0] if self.game_id else ''
        self.start_time = time.time()
        self.current_level = -1
        self.action_model: Optional[ActionModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.experience_buffer: deque = deque(maxlen=self.buffer_maxlen)
        self.experience_hashes: set = set()
        self.prev_frame: Optional[np.ndarray] = None  # bool (16, 64, 64)
        self.prev_action_idx: Optional[int] = None
        self.online_step_count = 0

        # T1/T3 prior decay knobs (shared)
        self._prior_decay_steps = int(_env_float('ARC_GOOSE_PRIOR_DECAY', 500))
        self._prior_weight_init = _env_float('ARC_GOOSE_PRIOR_WEIGHT', 4.0)

        # T1 delta state (bc_v4 logits as decaying prior)
        self._bc_helper = None
        self._bc_history: deque = deque(maxlen=4)
        self._bc_last_action_id: Optional[int] = None
        self._bc_step_index: int = 0
        self._bc_steps_since_progress: int = 0
        self._bc_levels_completed: int = 0
        # ARC_T1_PERM_BC_COLORS=1 permutes the 16-color palette before
        # feeding frames to bc_v4 — a memorization-vs-generalization test.
        # If bc_v4 learned structural patterns, perm-on ~= perm-off.
        # If bc_v4 memorized specific color sprites, perm-on collapses
        # toward T0 (Pure Goose).
        self._bc_color_perm_enabled = (
            str(os.environ.get('ARC_T1_PERM_BC_COLORS', '')).strip() in ('1', 'true', 'TRUE', 'yes')
        )
        self._bc_color_perm: Optional[np.ndarray] = None  # length-16, set per level
        if self.delta == 't1_bc':
            try:
                from bc_policy import PolicyHelper, find_checkpoint
                from pathlib import Path as _Path
                ckpt_env = os.environ.get('ARC_BC_CHECKPOINT_PATH')
                ckpt_path = _Path(ckpt_env) if ckpt_env else find_checkpoint()
                if ckpt_path is None:
                    print('[GooseAgent] T1: no bc_v4 checkpoint found; falling back to T0', flush=True)
                else:
                    self._bc_helper = PolicyHelper.load(ckpt_path, device=self.device)
            except Exception as ex:
                print(f'[GooseAgent] T1 bc_v4 load failed: {ex}', flush=True)
                self._bc_helper = None
            print(
                f'[GooseAgent] T1 bc prior: helper_loaded={self._bc_helper is not None} '
                f'decay_steps={self._prior_decay_steps} init_weight={self._prior_weight_init}',
                flush=True,
            )

        # T3 delta state (lazy load to keep T0 path clean)
        self._effect_dict = None
        if self.delta == 't3_priors':
            try:
                from my_agent import _load_action_effect_dict, _extract_saliency
                self._effect_dict = _load_action_effect_dict()
                self._extract_saliency = _extract_saliency
            except Exception as ex:
                print(f'[GooseAgent] T3 prior load failed: {ex}', flush=True)
                self._extract_saliency = None
            print(
                f'[GooseAgent] T3 priors: dict_loaded={self._effect_dict is not None} '
                f'decay_steps={self._prior_decay_steps} init_weight={self._prior_weight_init}',
                flush=True,
            )

        print(
            f'[GooseAgent] init game_id={self.game_id} short_id={self.short_id} '
            f'delta={self.delta} device={self.device} lr={self.lr} '
            f'train_every={self.train_every}',
            flush=True,
        )

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if len(self.frames) > self._MAX_FRAMES:
            self.frames = self.frames[-self._MAX_FRAMES:]
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, 'recorder') and not getattr(self, 'is_playback', False):
            import json
            try:
                self.recorder.record(json.loads(frame.model_dump_json()))
            except Exception:
                pass

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        try:
            return latest_frame.state is GameState.WIN
        except Exception:
            return False

    def _frame_to_tensor(self, frame_data: FrameData) -> torch.Tensor:
        frame = _final_subframe(frame_data)
        if frame.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError(f'unexpected frame shape {frame.shape}')
        t = torch.zeros(NUM_COLORS, GRID_SIZE, GRID_SIZE, dtype=torch.float32, device=self.device)
        idx = torch.from_numpy(frame).long().clamp(0, NUM_COLORS - 1).to(self.device)
        t.scatter_(0, idx.unsqueeze(0), 1)
        return t

    def _build_model_for_new_level(self) -> None:
        self.experience_buffer.clear()
        self.experience_hashes.clear()
        self.action_model = ActionModel().to(self.device)
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=self.lr)
        self.prev_frame = None
        self.prev_action_idx = None
        self.online_step_count = 0

    def _experience_hash(self, frame_bool: np.ndarray, action_idx: int) -> str:
        return hashlib.md5(frame_bool.tobytes() + str(action_idx).encode()).hexdigest()

    def _train(self) -> None:
        if self.action_model is None or self.optimizer is None:
            return
        if len(self.experience_buffer) < self.batch_size:
            return
        idxs = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in idxs]
        states = torch.stack([torch.from_numpy(e['state']).float().to(self.device) for e in batch])
        action_idx = torch.tensor([e['action_idx'] for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e['reward'] for e in batch], dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        logits = self.action_model(states)
        selected = logits.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(selected, rewards)
        # Small entropy bonus on the full distribution to keep coords exploring.
        with torch.no_grad():
            probs = torch.sigmoid(logits)
        entropy = -(probs[:, :NUM_SIMPLE_ACTIONS].mean() * 1e-4
                    + probs[:, NUM_SIMPLE_ACTIONS:].mean() * 1e-5)
        (loss + entropy).backward()
        self.optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_priors(
        self,
        latest_frame: FrameData,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute (simple_prior_6, coord_prior_4096) tensors per the active delta.

        Each returned tensor is on self.device, or None if no prior applies.
        Both T1 and T3 priors decay linearly with self.online_step_count.
        """
        simple_prior: Optional[torch.Tensor] = None
        coord_prior: Optional[torch.Tensor] = None

        # ---- T3: salient + effect-dict prior ----
        if self.delta == 't3_priors' and self._extract_saliency is not None:
            try:
                raw = _final_subframe(latest_frame).tolist()
            except Exception:
                raw = None
            if raw is not None:
                salient = self._extract_saliency(raw)
                cp = torch.zeros(NUM_COORDS, dtype=torch.float32, device=self.device)
                for (x, y) in salient[:30]:
                    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                        cp[y * GRID_SIZE + x] += 1.0
                if self._effect_dict is not None:
                    try:
                        arr = np.asarray(raw, dtype=np.int64).reshape(-1)
                        hist = np.bincount(arr, minlength=NUM_COLORS).astype(np.float32)
                        norm = np.linalg.norm(hist) + 1e-6
                        qv = hist / norm
                        target = self._effect_dict.feature_keys.shape[1]
                        if qv.shape[0] != target:
                            if qv.shape[0] < target:
                                qv = np.pad(qv, (0, target - qv.shape[0]))
                            else:
                                qv = qv[:target]
                        for (x, y) in self._effect_dict.top_clicks_by_similarity(qv, k=10):
                            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                                cp[y * GRID_SIZE + x] += 0.5
                    except Exception as ex:
                        if self.online_step_count == 0:
                            print(f'[GooseAgent] T3 dict query failed: {ex}', flush=True)
                coord_prior = cp

        # ---- T1: bc_v4 logits as prior ----
        if self.delta == 't1_bc' and self._bc_helper is not None:
            try:
                raw_arr = _final_subframe(latest_frame)
                # Memorization test: optionally permute colors before feeding to bc_v4.
                # x_logits / y_logits are coordinate priors and unaffected by color
                # permutation in semantics — only the visual frame content shifts.
                if self._bc_color_perm is not None:
                    raw_for_bc = self._bc_color_perm[raw_arr].tolist()
                else:
                    raw_for_bc = raw_arr.tolist()
                self._bc_history.append(raw_for_bc)
                scores = self._bc_helper.score_frame(
                    history_frames=list(self._bc_history),
                    latest_frame=raw_for_bc,
                    last_action_id=self._bc_last_action_id,
                    levels_completed=self._bc_levels_completed,
                    steps_since_progress=self._bc_steps_since_progress,
                    step_index=self._bc_step_index,
                    available_actions=_available_action_ids(latest_frame),
                )
                if scores is not None:
                    a = scores['action_logits']  # (7,) for ACTION1..ACTION7
                    x = scores['x_logits']        # (64,)
                    y = scores['y_logits']        # (64,)
                    sp = torch.zeros(NUM_SIMPLE_ACTIONS, dtype=torch.float32, device=self.device)
                    for i, aid in enumerate(SIMPLE_ACTION_IDS):
                        sp[i] = float(a[aid - 1])
                    # outer-product of y and x logits → 64x64 → flat 4096
                    yy = torch.from_numpy(y.astype(np.float32)).to(self.device)
                    xx = torch.from_numpy(x.astype(np.float32)).to(self.device)
                    cp = (yy.unsqueeze(1) + xx.unsqueeze(0)).reshape(-1)
                    simple_prior = sp
                    coord_prior = cp
            except Exception as ex:
                if self.online_step_count == 0:
                    print(f'[GooseAgent] T1 bc score failed: {ex}', flush=True)
            self._bc_step_index += 1

        return simple_prior, coord_prior

    def _sample_action(
        self,
        combined_logits: torch.Tensor,
        available_ids: List[int],
        simple_prior: Optional[torch.Tensor] = None,
        coord_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[int, Optional[Tuple[int, int]], int]:
        """Return (action_idx, coords_or_none, full_idx) where full_idx indexes the combined output."""
        simple_logits = combined_logits[:NUM_SIMPLE_ACTIONS].clone()
        coord_logits = combined_logits[NUM_SIMPLE_ACTIONS:].clone()

        # Apply decaying priors to logits.
        decay = max(0.0, 1.0 - self.online_step_count / max(1, self._prior_decay_steps))
        if simple_prior is not None:
            simple_logits = simple_logits + self._prior_weight_init * decay * simple_prior
        if coord_prior is not None:
            coord_logits = coord_logits + self._prior_weight_init * decay * coord_prior

        action6_available = 6 in available_ids
        simple_available_mask = torch.tensor(
            [1.0 if aid in available_ids else 0.0 for aid in SIMPLE_ACTION_IDS],
            dtype=torch.float32,
            device=combined_logits.device,
        )
        simple_logits = torch.where(
            simple_available_mask > 0,
            simple_logits,
            torch.full_like(simple_logits, float('-inf')),
        )
        if not action6_available:
            coord_logits = torch.full_like(coord_logits, float('-inf'))

        simple_probs = torch.sigmoid(simple_logits)
        coord_probs = torch.sigmoid(coord_logits) / NUM_COORDS

        all_probs = torch.cat([simple_probs, coord_probs])
        total = all_probs.sum().item()
        if not np.isfinite(total) or total <= 0:
            # Last-resort uniform over the available actions
            uniform = torch.zeros_like(all_probs)
            for aid in available_ids:
                if aid == 6:
                    uniform[NUM_SIMPLE_ACTIONS:] = 1.0 / NUM_COORDS
                elif aid in SIMPLE_ACTION_IDS:
                    uniform[SIMPLE_ACTION_IDS.index(aid)] = 1.0
            all_probs = uniform
            total = all_probs.sum().item()
        all_probs = all_probs / total
        probs_np = all_probs.detach().cpu().numpy()
        full_idx = int(np.random.choice(len(probs_np), p=probs_np))

        if full_idx < NUM_SIMPLE_ACTIONS:
            return full_idx, None, full_idx
        coord_idx = full_idx - NUM_SIMPLE_ACTIONS
        y = coord_idx // GRID_SIZE
        x = coord_idx % GRID_SIZE
        return NUM_SIMPLE_ACTIONS, (int(y), int(x)), full_idx

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        try:
            # Reset model when level changes
            new_level = int(getattr(latest_frame, 'levels_completed', 0))
            if new_level != self.current_level:
                print(f'[GooseAgent] level change {self.current_level}->{new_level}; resetting model', flush=True)
                self._build_model_for_new_level()
                self.current_level = new_level
                # Reset T1 BC state on level change too (bc_v4 history becomes stale).
                self._bc_history.clear()
                self._bc_last_action_id = None
                self._bc_step_index = 0
                self._bc_steps_since_progress = 0
                self._bc_levels_completed = new_level
                # T1 memorization-test: fresh color permutation per level.
                if self._bc_color_perm_enabled:
                    self._bc_color_perm = np.arange(NUM_COLORS)
                    np.random.shuffle(self._bc_color_perm)
                    print(f'[GooseAgent] T1 perm: {self._bc_color_perm.tolist()}', flush=True)
                else:
                    self._bc_color_perm = None
            else:
                self._bc_steps_since_progress += 1

            # Hard reset of episode if RESET state
            state = getattr(latest_frame, 'state', None)
            if state is GameState.NOT_PLAYED or state is GameState.GAME_OVER:
                self.prev_frame = None
                self.prev_action_idx = None
                action = GameAction.RESET
                action.reasoning = {'strategy': 'goose', 'phase': 'reset'}
                return action

            current = self._frame_to_tensor(latest_frame)
            current_bool = current.detach().cpu().numpy().astype(bool)

            # Record experience from previous step: did the frame change?
            if self.prev_frame is not None and self.prev_action_idx is not None:
                exp_hash = self._experience_hash(self.prev_frame, self.prev_action_idx)
                if exp_hash not in self.experience_hashes:
                    changed = not np.array_equal(self.prev_frame, current_bool)
                    self.experience_buffer.append({
                        'state': self.prev_frame,
                        'action_idx': self.prev_action_idx,
                        'reward': 1.0 if changed else 0.0,
                    })
                    self.experience_hashes.add(exp_hash)

            available_ids = _available_action_ids(latest_frame)
            with torch.no_grad():
                logits = self.action_model(current.unsqueeze(0)).squeeze(0)

            simple_prior, coord_prior = self._build_priors(latest_frame)
            action_slot, coords, full_idx = self._sample_action(
                logits, available_ids, simple_prior, coord_prior
            )

            if action_slot < NUM_SIMPLE_ACTIONS:
                aid = SIMPLE_ACTION_IDS[action_slot]
                act = GameAction.from_id(int(aid))
                act.reasoning = {'strategy': 'goose', 'phase': 'simple', 'aid': aid}
                self._bc_last_action_id = int(aid)
            else:
                y, x = coords
                act = GameAction.ACTION6
                act.set_data({'x': int(x), 'y': int(y)})
                act.reasoning = {'strategy': 'goose', 'phase': 'click', 'xy': (int(x), int(y))}
                self._bc_last_action_id = 6

            self.prev_frame = current_bool
            self.prev_action_idx = full_idx
            self.online_step_count += 1

            if self.online_step_count % self.train_every == 0:
                self._train()

            return act

        except Exception as e:
            print(f'[GooseAgent] choose_action crashed: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
            # Fallback: a safe random simple action
            avail = _available_action_ids(latest_frame)
            simple_avail = [a for a in avail if a in SIMPLE_ACTION_IDS]
            aid = self._rng.choice(simple_avail) if simple_avail else 1
            act = GameAction.from_id(int(aid))
            act.reasoning = {'strategy': 'goose', 'phase': 'fallback_error', 'err': str(e)}
            return act
