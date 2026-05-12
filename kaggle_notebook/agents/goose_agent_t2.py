# =====================================================================
# GooseT2Agent — Goose + 1-layer GRU episode memory
#
# Forked from goose_agent.py (T0/T1/T3 implementation). Tests theory T2:
# does episode memory help an online change-reward CNN learner on hidden
# games?
#
# Architecture differences vs T0:
#   - ActionModelGRU: CNN backbone identical, but globally avg-pooled
#     conv4 features (256,) feed a GRU(input=256, hidden=128, num_layers=1).
#     GRU hidden output is concatenated with action_fc(512) features for
#     the action head (640 -> 6). Coord head is unchanged (spatial-only).
#   - GRU hidden state h_t carried across steps within a level.
#   - Reset hidden state to zeros on every level change (alongside the
#     model + optimizer reset Goose already does).
#
# Training:
#   - Buffer stores (prev_frame, prev_hidden, prev_action_idx, reward).
#     Hidden state at action-selection time is the "context" for that
#     selection — stored once, never recomputed.
#   - Train step: forward(prev_frame, prev_hidden) -> BCE on selected
#     logit. Single-step backprop through CNN + GRU + heads. No multi-step
#     BPTT (keeps training ~same speed as T0).
#   - GRU recurrent weights still learn via the gradient flowing through
#     the action head's GRU-conditioned features.
#
# Pluggable via env vars (same names as goose_agent.py where it makes sense):
#   ARC_GOOSE_LR=1e-4           AdamW lr
#   ARC_GOOSE_TRAIN_EVERY=5     train every N actions
#   ARC_GOOSE_BUFFER=200000     experience buffer maxlen
#   ARC_GOOSE_BATCH=64          train batch size
#   ARC_T2_HIDDEN=128           GRU hidden size (defaults to 128)
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


SIMPLE_ACTION_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 7)
NUM_SIMPLE_ACTIONS = len(SIMPLE_ACTION_IDS)
GRID_SIZE = 64
NUM_COLORS = 16
NUM_COORDS = GRID_SIZE * GRID_SIZE


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
    arr = np.asarray(frame_data.frame, dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr


class ActionModelGRU(nn.Module):
    """4-conv CNN + GRU memory + action/coord heads.

    Returns (combined_logits, new_hidden) — caller persists `new_hidden`
    across steps within a level.
    """

    def __init__(
        self,
        input_channels: int = NUM_COLORS,
        grid_size: int = GRID_SIZE,
        num_simple_actions: int = NUM_SIMPLE_ACTIONS,
        gru_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_simple_actions = num_simple_actions
        self.gru_hidden = gru_hidden

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # GRU operating on globally-pooled conv4 features (256 -> 128 hidden)
        self.gru = nn.GRU(input_size=256, hidden_size=gru_hidden, num_layers=1, batch_first=True)

        # Simple-action head: action_fc features (512) ++ GRU hidden (128) -> 6
        self.action_pool = nn.MaxPool2d(4, 4)
        action_flat = 256 * (grid_size // 4) * (grid_size // 4)
        self.action_fc = nn.Linear(action_flat, 512)
        self.action_head = nn.Linear(512 + gru_hidden, num_simple_actions)

        # Coord head: spatial-only, no GRU bias (4096 fan-out would blow up params)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)

        self.dropout = nn.Dropout(0.2)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 16, 64, 64). h: (1, B, gru_hidden) or None.

        Returns (combined_logits (B, 6+4096), new_hidden (1, B, gru_hidden)).
        """
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        feats = F.relu(self.conv4(x3))  # (B, 256, 64, 64)

        # Global avg pool conv4 features → GRU input
        gru_in = feats.mean(dim=[2, 3]).unsqueeze(1)  # (B, 1, 256)
        if h is None:
            h = torch.zeros(1, x.size(0), self.gru_hidden, device=x.device, dtype=gru_in.dtype)
        gru_out, new_h = self.gru(gru_in, h)  # gru_out: (B, 1, gru_hidden)
        gru_h = gru_out.squeeze(1)  # (B, gru_hidden)

        # Action head
        a = self.action_pool(feats)
        a = a.view(a.size(0), -1)
        a = F.relu(self.action_fc(a))
        a = self.dropout(a)
        a = torch.cat([a, gru_h], dim=1)  # (B, 512 + gru_hidden)
        action_logits = self.action_head(a)  # (B, 6)

        # Coord head (spatial-only, no GRU)
        c = F.relu(self.coord_conv1(feats))
        c = F.relu(self.coord_conv2(c))
        c = F.relu(self.coord_conv3(c))
        coord_logits = self.coord_conv4(c).view(c.size(0), -1)  # (B, 4096)

        combined = torch.cat([action_logits, coord_logits], dim=1)
        return combined, new_h


class GooseT2Agent(Agent):
    """T2: Goose + 1-layer GRU episode memory."""

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
        self.lr = _env_float('ARC_GOOSE_LR', 1e-4)
        self.train_every = int(_env_float('ARC_GOOSE_TRAIN_EVERY', 5))
        self.buffer_maxlen = int(_env_float('ARC_GOOSE_BUFFER', 200000))
        self.batch_size = int(_env_float('ARC_GOOSE_BATCH', 64))
        self.gru_hidden = int(_env_float('ARC_T2_HIDDEN', 128))

        self.short_id = self.game_id.split('-', 1)[0] if self.game_id else ''
        self.start_time = time.time()
        self.current_level = -1
        self.action_model: Optional[ActionModelGRU] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.experience_buffer: deque = deque(maxlen=self.buffer_maxlen)
        self.experience_hashes: set = set()
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_hidden: Optional[np.ndarray] = None  # (1, 1, gru_hidden) as numpy
        self.prev_action_idx: Optional[int] = None
        self.gru_hidden_state: Optional[torch.Tensor] = None  # live GRU state on device
        self.online_step_count = 0

        print(
            f'[GooseT2Agent] init game_id={self.game_id} short_id={self.short_id} '
            f'device={self.device} lr={self.lr} gru_hidden={self.gru_hidden}',
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
        self.action_model = ActionModelGRU(gru_hidden=self.gru_hidden).to(self.device)
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=self.lr)
        self.prev_frame = None
        self.prev_hidden = None
        self.prev_action_idx = None
        # Fresh GRU hidden state for the new level
        self.gru_hidden_state = torch.zeros(1, 1, self.gru_hidden, device=self.device)
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
        # Stack stored hidden states. Each was (1, 1, gru_hidden) — stack to (1, B, gru_hidden).
        hidden = torch.cat(
            [torch.from_numpy(e['hidden']).to(self.device, dtype=torch.float32) for e in batch],
            dim=1,
        )
        action_idx = torch.tensor([e['action_idx'] for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e['reward'] for e in batch], dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        logits, _ = self.action_model(states, hidden)
        selected = logits.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(selected, rewards)
        # Same tiny entropy bonus as T0
        with torch.no_grad():
            probs = torch.sigmoid(logits)
        entropy = -(probs[:, :NUM_SIMPLE_ACTIONS].mean() * 1e-4
                    + probs[:, NUM_SIMPLE_ACTIONS:].mean() * 1e-5)
        (loss + entropy).backward()
        self.optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sample_action(
        self,
        combined_logits: torch.Tensor,
        available_ids: List[int],
    ) -> Tuple[int, Optional[Tuple[int, int]], int]:
        simple_logits = combined_logits[:NUM_SIMPLE_ACTIONS].clone()
        coord_logits = combined_logits[NUM_SIMPLE_ACTIONS:].clone()

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
            new_level = int(getattr(latest_frame, 'levels_completed', 0))
            if new_level != self.current_level:
                print(f'[GooseT2Agent] level change {self.current_level}->{new_level}; resetting model+hidden', flush=True)
                self._build_model_for_new_level()
                self.current_level = new_level

            state = getattr(latest_frame, 'state', None)
            if state is GameState.NOT_PLAYED or state is GameState.GAME_OVER:
                self.prev_frame = None
                self.prev_hidden = None
                self.prev_action_idx = None
                # Don't reset GRU hidden on env-reset (only on level change)
                action = GameAction.RESET
                action.reasoning = {'strategy': 'goose_t2', 'phase': 'reset'}
                return action

            current = self._frame_to_tensor(latest_frame)
            current_bool = current.detach().cpu().numpy().astype(bool)

            # Record experience from previous step: did the frame change?
            if (
                self.prev_frame is not None
                and self.prev_action_idx is not None
                and self.prev_hidden is not None
            ):
                exp_hash = self._experience_hash(self.prev_frame, self.prev_action_idx)
                if exp_hash not in self.experience_hashes:
                    changed = not np.array_equal(self.prev_frame, current_bool)
                    self.experience_buffer.append({
                        'state': self.prev_frame,
                        'hidden': self.prev_hidden,  # (1, 1, gru_hidden) as numpy float32
                        'action_idx': self.prev_action_idx,
                        'reward': 1.0 if changed else 0.0,
                    })
                    self.experience_hashes.add(exp_hash)

            available_ids = _available_action_ids(latest_frame)

            # Snapshot the hidden state at the moment of action selection
            # (this h_t is what conditions the action, and what we store with
            # the experience so training can recreate the same forward path).
            h_at_action = self.gru_hidden_state.detach().clone()

            with torch.no_grad():
                logits, new_h = self.action_model(
                    current.unsqueeze(0),
                    self.gru_hidden_state,
                )
                logits = logits.squeeze(0)
            # Advance the live hidden state.
            self.gru_hidden_state = new_h.detach()

            action_slot, coords, full_idx = self._sample_action(logits, available_ids)

            if action_slot < NUM_SIMPLE_ACTIONS:
                aid = SIMPLE_ACTION_IDS[action_slot]
                act = GameAction.from_id(int(aid))
                act.reasoning = {'strategy': 'goose_t2', 'phase': 'simple', 'aid': aid}
            else:
                y, x = coords
                act = GameAction.ACTION6
                act.set_data({'x': int(x), 'y': int(y)})
                act.reasoning = {'strategy': 'goose_t2', 'phase': 'click', 'xy': (int(x), int(y))}

            self.prev_frame = current_bool
            self.prev_hidden = h_at_action.cpu().numpy()
            self.prev_action_idx = full_idx
            self.online_step_count += 1

            if self.online_step_count % self.train_every == 0:
                self._train()

            return act

        except Exception as e:
            print(f'[GooseT2Agent] choose_action crashed: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
            avail = _available_action_ids(latest_frame)
            simple_avail = [a for a in avail if a in SIMPLE_ACTION_IDS]
            aid = self._rng.choice(simple_avail) if simple_avail else 1
            act = GameAction.from_id(int(aid))
            act.reasoning = {'strategy': 'goose_t2', 'phase': 'fallback_error', 'err': str(e)}
            return act
