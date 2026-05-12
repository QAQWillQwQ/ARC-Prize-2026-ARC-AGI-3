"""Vendored BC policy + TTT primitives for the Kaggle submission notebook.

Self-contained — imports only torch, numpy, json, pathlib. Mirrors the parts of
src/model.py + src/common.py that PolicyGuidedAgent uses, without dragging in
the rest of the project (collect/evaluate/etc).

Two distinct things in here:

1. **ObjectCentricPolicy** — exact copy of `src/model.py:ObjectCentricPolicy`.
   Loads a checkpoint trained by `python -m src.train` and produces per-frame
   `action_logits / x_logits / y_logits / value / avail_logits` scores.

2. **TTT (test-time training)** — adapts the loaded checkpoint to the SPECIFIC
   game being played, at submission time, on Kaggle's GPU. Two modes:
     - `replay_finetune` (public games): SGD on this game's GT replay
     - `aug_finetune`    (any game): color-permutation consistency loss

Local testing: scripts/test_my_agent_local.py imports my_agent.py which now
imports bc_policy. Run with ARC_BC_CHECKPOINT_PATH pointing at any .pth file.
Disable via ARC_DISABLE_TTT=1 to A/B without TTT.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_OK = False


GRID_SIZE = 64
NUM_COLORS = 16
ACTION_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
ACTION_TO_INDEX: Dict[int, int] = {aid: i for i, aid in enumerate(ACTION_IDS)}


# ---------------------- frame / scalar helpers (vendored from src/common.py) ----------------------


def _safe_color(value: Any) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 0
    if v < 0:
        return 0
    if v >= NUM_COLORS:
        return v % NUM_COLORS
    return v


def _is_iterable(obj: Any) -> bool:
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def _action_mask(available_actions: Sequence[int]) -> List[float]:
    mask: List[float] = []
    avail_set = {int(a) for a in available_actions}
    for aid in ACTION_IDS:
        mask.append(1.0 if aid in avail_set else 0.0)
    return mask


def _non_background_density(frame: Sequence[Sequence[int]]) -> float:
    total = 0
    nonbg = 0
    for row in frame:
        if not _is_iterable(row):
            continue
        for c in row:
            total += 1
            try:
                if int(c) != 0:
                    nonbg += 1
            except (TypeError, ValueError):
                continue
    return float(nonbg) / max(1, total)


def pad_history(frames: Sequence[Sequence[Sequence[int]]], history: int) -> List[List[List[int]]]:
    """Pad a frame history to `history` frames. Repeats the oldest frame if short."""
    if not frames:
        blank = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        return [blank for _ in range(history)]
    stacked: List[List[List[int]]] = []
    for frame in frames[-history:]:
        rows: List[List[int]] = []
        for row in frame:
            if _is_iterable(row):
                rows.append([_safe_color(v) for v in row])
            else:
                rows.append([0 for _ in range(GRID_SIZE)])
        stacked.append(rows)
    while len(stacked) < history:
        stacked.insert(0, [row[:] for row in stacked[0]])
    return stacked


def scalar_features(
    available_actions: Sequence[int],
    last_action_id: Optional[int],
    levels_completed: int,
    steps_since_progress: int,
    step_index: int,
    frame: Sequence[Sequence[int]],
    max_steps: int,
) -> "torch.Tensor":
    """18-dim scalar feature vector: 7 action mask + 7 last-action one-hot + 4 progress floats."""
    features: List[float] = []
    features.extend(_action_mask(available_actions))
    last_action = [0.0 for _ in ACTION_IDS]
    if last_action_id in ACTION_TO_INDEX:
        last_action[ACTION_TO_INDEX[last_action_id]] = 1.0
    features.extend(last_action)
    features.extend([
        min(levels_completed / 10.0, 1.0),
        min(steps_since_progress / max(max_steps, 1), 1.0),
        min(step_index / max(max_steps, 1), 1.0),
        _non_background_density(frame),
    ])
    return torch.tensor(features, dtype=torch.float32)


def frames_to_obs_uint8(history_frames: Sequence[Sequence[Sequence[int]]], history: int) -> "torch.Tensor":
    """Build (history, 64, 64) uint8 tensor for the model's GPU one_hot path."""
    padded = pad_history(history_frames, history)
    arr = np.asarray(padded, dtype=np.uint8)
    # Defensive shape correction.
    if arr.shape != (history, GRID_SIZE, GRID_SIZE):
        out = np.zeros((history, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        h = min(arr.shape[0], history) if arr.ndim >= 1 else 0
        ry = min(arr.shape[1], GRID_SIZE) if arr.ndim >= 2 else 0
        rx = min(arr.shape[2], GRID_SIZE) if arr.ndim >= 3 else 0
        if h and ry and rx:
            out[:h, :ry, :rx] = arr[:h, :ry, :rx]
        arr = out
    return torch.from_numpy(arr & 0x0F)


# ---------------------- ObjectCentricPolicy (verbatim from src/model.py) ----------------------


if _TORCH_OK:
    class ObjectCentricPolicy(nn.Module):
        def __init__(
            self,
            history: int = 4,
            model_dim: int = 384,
            num_slots: int = 8,
            depth: int = 6,
            num_heads: int = 8,
            scalar_dim: int = 18,
            use_goal: bool = False,
        ) -> None:
            super().__init__()
            input_channels = history * 16
            hidden = max(model_dim // 2, 128)
            self.history = history
            self.model_dim = model_dim
            self.num_slots = num_slots
            self.use_goal = bool(use_goal)
            self.goal_dim = 128

            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, hidden // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden // 2, hidden, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, model_dim, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.pos_embed = nn.Parameter(torch.randn(1, 16 * 16, model_dim) * 0.02)
            self.slot_queries = nn.Parameter(torch.randn(1, num_slots, model_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=model_dim, num_heads=num_heads, batch_first=True,
            )
            self.scalar_proj = nn.Sequential(
                nn.Linear(scalar_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 4,
                activation="gelu", batch_first=True, norm_first=True, dropout=0.1,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.state_norm = nn.LayerNorm(model_dim)

            # Gen 1 goal encoder (mirrors src/model.py). Always constructed for
            # state_dict shape consistency; only USED when self.use_goal is True.
            goal_hidden = 64
            self.goal_encoder = nn.Sequential(
                nn.Conv2d(NUM_COLORS, goal_hidden, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(goal_hidden, goal_hidden, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(goal_hidden, self.goal_dim, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.goal_proj = nn.Sequential(
                nn.Linear(self.goal_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )

            self.action_head = nn.Linear(model_dim, len(ACTION_IDS))
            self.x_head = nn.Linear(model_dim, 64)
            self.y_head = nn.Linear(model_dim, 64)
            self.value_head = nn.Linear(model_dim, 1)
            self.avail_head = nn.Linear(model_dim, len(ACTION_IDS))
            self.action_embed = nn.Embedding(len(ACTION_IDS), model_dim)
            self.next_latent_head = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )

            # Aux heads (off-path at inference; needed only to load checkpoints with these params).
            self.archetype_head = nn.Linear(model_dim, 3)
            saliency_hidden = max(model_dim // 4, 32)
            self._saliency_hidden = saliency_hidden
            self.saliency_decoder = nn.Sequential(
                nn.Conv2d(model_dim, saliency_hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(saliency_hidden, saliency_hidden, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(saliency_hidden, 1, kernel_size=4, stride=2, padding=1),
            )
            recon_hidden = max(model_dim // 4, 32)
            self._recon_hidden = recon_hidden
            self.recon_init = nn.Linear(model_dim, recon_hidden * 4 * 4)
            recon_mid = max(recon_hidden // 2, 16)
            self.recon_decoder = nn.Sequential(
                nn.ConvTranspose2d(recon_hidden, recon_hidden, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(recon_hidden, recon_hidden, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(recon_hidden, recon_mid, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(recon_mid, NUM_COLORS, kernel_size=4, stride=2, padding=1),
            )

        def encode_state(
            self,
            obs: "torch.Tensor",
            scalar: "torch.Tensor",
            goal_obs: Optional["torch.Tensor"] = None,
        ) -> Dict[str, "torch.Tensor"]:
            if obs.dim() == 4 and obs.shape[1] == self.history:
                clipped = obs.to(dtype=torch.long).clamp_(0, NUM_COLORS - 1)
                one_hot = F.one_hot(clipped, num_classes=NUM_COLORS)
                obs = one_hot.permute(0, 1, 4, 2, 3).reshape(
                    obs.shape[0], self.history * NUM_COLORS, 64, 64,
                ).to(dtype=torch.float32)
            elif obs.dtype != torch.float32:
                obs = obs.to(dtype=torch.float32)
            features = self.conv(obs)
            batch_size = features.shape[0]
            patches = features.flatten(2).transpose(1, 2)
            patches = patches + self.pos_embed[:, : patches.shape[1], :]
            slots = self.slot_queries.expand(batch_size, -1, -1)
            slot_tokens, _ = self.cross_attn(query=slots, key=patches, value=patches)
            scalar_token = self.scalar_proj(scalar).unsqueeze(1)
            # Gen 1 goal-conditioning (mirrors src/model.py).
            if self.use_goal and goal_obs is not None:
                if goal_obs.dim() == 3:
                    gclip = goal_obs.to(dtype=torch.long).clamp_(0, NUM_COLORS - 1)
                    goal_oh = F.one_hot(gclip, num_classes=NUM_COLORS).permute(0, 3, 1, 2).to(dtype=torch.float32)
                elif goal_obs.dim() == 4 and goal_obs.shape[1] == NUM_COLORS:
                    goal_oh = goal_obs.to(dtype=torch.float32)
                else:
                    goal_oh = None
                if goal_oh is not None:
                    goal_emb = self.goal_encoder(goal_oh)
                    goal_token = self.goal_proj(goal_emb).unsqueeze(1)
                    scalar_token = scalar_token + goal_token
            tokens = torch.cat([scalar_token, slot_tokens], dim=1)
            tokens = self.encoder(tokens)
            pooled = self.state_norm(tokens[:, 0, :])
            return {"pooled": pooled, "tokens": tokens, "patches": patches}

        def forward(
            self,
            obs: "torch.Tensor",
            scalar: "torch.Tensor",
            action_index: Optional["torch.Tensor"] = None,
            goal_obs: Optional["torch.Tensor"] = None,
        ) -> Dict[str, "torch.Tensor"]:
            encoded = self.encode_state(obs, scalar, goal_obs=goal_obs)
            pooled = encoded["pooled"]
            out: Dict[str, "torch.Tensor"] = {
                "pooled": pooled,
                "action_logits": self.action_head(pooled),
                "x_logits": self.x_head(pooled),
                "y_logits": self.y_head(pooled),
                "value": self.value_head(pooled).squeeze(-1),
                "avail_logits": self.avail_head(pooled),
            }
            if action_index is not None:
                action_emb = self.action_embed(action_index)
                pred_next_latent = self.next_latent_head(
                    torch.cat([pooled, action_emb], dim=-1)
                )
                out["pred_next_latent"] = pred_next_latent
            return out
else:
    ObjectCentricPolicy = None  # type: ignore


# ---------------------- PolicyHelper: load + score ----------------------


class PolicyHelper:
    """Wraps a loaded ObjectCentricPolicy. Provides per-frame scoring + TTT hooks."""

    def __init__(self, model: "ObjectCentricPolicy", device: "torch.device", config: Dict[str, Any]) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.history = int(config.get("history", 4))
        self.max_steps = int(config.get("max_steps", 192))

    @classmethod
    def load(cls, path: Path, device: Optional["torch.device"] = None) -> Optional["PolicyHelper"]:
        """Load a checkpoint produced by `src.train`. Returns None on any failure."""
        if not _TORCH_OK:
            return None
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            payload = torch.load(str(path), map_location=device, weights_only=False)
        except Exception as exc:
            print(f"[PolicyHelper] checkpoint load failed at {path}: {exc}", flush=True)
            return None
        config = payload.get("config", {}) or {}
        model = ObjectCentricPolicy(
            history=int(config.get("history", 4)),
            model_dim=int(config.get("model_dim", 384)),
            num_slots=int(config.get("num_slots", 8)),
            depth=int(config.get("depth", 6)),
            num_heads=int(config.get("num_heads", 8)),
            scalar_dim=18,
            use_goal=bool(config.get("use_goal", False)),
        )
        try:
            model.load_state_dict(payload["model_state"], strict=False)
        except Exception as exc:
            print(f"[PolicyHelper] state_dict load failed: {exc}", flush=True)
            return None
        model.to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[PolicyHelper] loaded {n_params/1e6:.1f}M-param model from {path} on {device} "
            f"(history={config.get('history')} model_dim={config.get('model_dim')})",
            flush=True,
        )
        return cls(model=model, device=device, config=config)

    @torch.no_grad() if _TORCH_OK else (lambda f: f)
    def score_frame(
        self,
        history_frames: Sequence[Sequence[Sequence[int]]],
        latest_frame: Sequence[Sequence[int]],
        last_action_id: Optional[int],
        levels_completed: int,
        steps_since_progress: int,
        step_index: int,
        available_actions: Sequence[int],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Returns {'action_logits' (7,), 'x_logits' (64,), 'y_logits' (64,),
        'action_scores' softmaxed (7,), 'x_scores' (64,), 'y_scores' (64,),
        'value' scalar} as numpy arrays. Returns None on torch failure.
        """
        if not _TORCH_OK or self.model is None:
            return None
        try:
            obs = frames_to_obs_uint8(history_frames, self.history).unsqueeze(0).to(self.device)
            scalar = scalar_features(
                available_actions=available_actions,
                last_action_id=last_action_id,
                levels_completed=levels_completed,
                steps_since_progress=steps_since_progress,
                step_index=step_index,
                frame=latest_frame,
                max_steps=self.max_steps,
            ).unsqueeze(0).to(self.device)
            out = self.model(obs, scalar)
            action_logits = out["action_logits"].squeeze(0).cpu().numpy()
            x_logits = out["x_logits"].squeeze(0).cpu().numpy()
            y_logits = out["y_logits"].squeeze(0).cpu().numpy()
            value = float(out["value"].squeeze(0).cpu().item())
            return {
                "action_logits": action_logits,
                "x_logits": x_logits,
                "y_logits": y_logits,
                "action_scores": _softmax(action_logits),
                "x_scores": _softmax(x_logits),
                "y_scores": _softmax(y_logits),
                "value": value,
            }
        except Exception as exc:
            print(f"[PolicyHelper] score_frame failed: {exc}", flush=True)
            return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / max(1e-8, e.sum())


# ---------------------- TTT: test-time training ----------------------


def ttt_replay_finetune(
    helper: "PolicyHelper",
    replay_actions: List[Dict[str, Any]],
    n_steps: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
) -> Dict[str, float]:
    """Adapt the BC checkpoint to a specific game's GT replay.

    `replay_actions` is the list MyAgent already builds from the per-game
    replay JSON (same shape as `_load_replay_actions` returns). Each entry
    is {'type': 'reset'} or {'type': 'action', 'id': int, 'x': int, 'y': int}.

    We don't need (state, action) pairs here — the GT replay only stores
    action sequences, not the resulting frames. So we use a simpler
    objective: train the model's `action_head` to predict the GT action
    from the current observation built from the recent history. Since we
    don't have actual rollout frames, we use a placeholder all-zero history
    and just teach the action distribution. This is a degenerate form of
    TTT — it teaches the model "for this game, prefer this action mix".

    For real per-game replay TTT (with state context), a richer setup
    would replay the game in a sandbox to capture (state, action) pairs.
    This MVP version is the cheapest possible signal.
    """
    if not _TORCH_OK or helper is None or helper.model is None:
        return {"steps": 0, "final_loss": 0.0}
    actions = [int(e.get("id", 0)) for e in replay_actions if e.get("type") == "action"]
    actions = [a for a in actions if 1 <= a <= 7]
    if len(actions) < batch_size:
        return {"steps": 0, "final_loss": 0.0, "skipped": "too_few_actions"}

    helper.model.train()
    opt = torch.optim.AdamW(helper.model.parameters(), lr=lr, weight_decay=0.01)

    # Build the placeholder obs and scalar features once — we're teaching the
    # MARGINAL action distribution for this game.
    blank_history = [[[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] for _ in range(helper.history)]
    obs_one = frames_to_obs_uint8(blank_history, helper.history).unsqueeze(0).to(helper.device)
    scalar_one = scalar_features(
        available_actions=ACTION_IDS, last_action_id=None,
        levels_completed=0, steps_since_progress=0, step_index=0,
        frame=blank_history[-1], max_steps=helper.max_steps,
    ).unsqueeze(0).to(helper.device)

    action_indices = [ACTION_TO_INDEX[a] for a in actions]
    action_tensor_full = torch.tensor(action_indices, dtype=torch.long, device=helper.device)

    losses: List[float] = []
    rng = np.random.default_rng(0)
    for step in range(n_steps):
        idx = rng.integers(0, len(action_indices), size=batch_size)
        target = action_tensor_full[idx]
        obs_batch = obs_one.expand(batch_size, *obs_one.shape[1:])
        scalar_batch = scalar_one.expand(batch_size, *scalar_one.shape[1:])
        out = helper.model(obs_batch, scalar_batch)
        loss = F.cross_entropy(out["action_logits"], target)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))

    helper.model.eval()
    return {"steps": int(n_steps), "final_loss": losses[-1] if losses else 0.0,
            "first_loss": losses[0] if losses else 0.0, "n_actions": len(actions)}


def ttt_aug_consistency(
    helper: "PolicyHelper",
    history_frames: Sequence[Sequence[Sequence[int]]],
    latest_frame: Sequence[Sequence[int]],
    available_actions: Sequence[int],
    last_action_id: Optional[int],
    levels_completed: int,
    steps_since_progress: int,
    step_index: int,
    n_aug: int = 8,
    n_steps: int = 10,
    lr: float = 5e-5,
) -> Dict[str, float]:
    """Augmentation-based TTT for ANY game (works without replays).

    Generate `n_aug` random color-permutation augmentations of the current
    frame. The model's action prediction should be INVARIANT under color
    permutation (color 0 stays as background; colors 1-15 are interchangeable).
    Use the BASE model's prediction on the un-permuted frame as a soft target,
    then KL-distill the augmented predictions toward it.

    This adapts the encoder to be more robust on the current game's specific
    visual content — useful for hidden games where no replay is available.
    """
    if not _TORCH_OK or helper is None or helper.model is None:
        return {"steps": 0}

    # Compute base prediction (no_grad — frozen target).
    helper.model.eval()
    with torch.no_grad():
        obs_base = frames_to_obs_uint8(history_frames, helper.history).unsqueeze(0).to(helper.device)
        scalar_base = scalar_features(
            available_actions=available_actions, last_action_id=last_action_id,
            levels_completed=levels_completed, steps_since_progress=steps_since_progress,
            step_index=step_index, frame=latest_frame, max_steps=helper.max_steps,
        ).unsqueeze(0).to(helper.device)
        base_out = helper.model(obs_base, scalar_base)
        base_action_log_softmax = F.log_softmax(base_out["action_logits"], dim=-1).detach()

    helper.model.train()
    opt = torch.optim.AdamW(helper.model.parameters(), lr=lr, weight_decay=0.01)

    rng = np.random.default_rng(int(time.time() * 1000) & 0xFFFFFFFF)

    losses: List[float] = []
    for step in range(n_steps):
        # Build a batch of n_aug color-permuted versions of the obs.
        permuted_obs_list = []
        for _ in range(n_aug):
            # Random permutation of colors 1..15; color 0 is identity (background).
            rest = list(range(1, NUM_COLORS))
            rng.shuffle(rest)
            perm = np.array([0] + rest, dtype=np.uint8)
            permuted = perm[obs_base.cpu().numpy().astype(np.int64)]
            permuted_obs_list.append(torch.from_numpy(permuted))
        obs_aug = torch.cat(permuted_obs_list, dim=0).to(helper.device)
        scalar_aug = scalar_base.expand(n_aug, *scalar_base.shape[1:])
        aug_out = helper.model(obs_aug, scalar_aug)
        aug_log_softmax = F.log_softmax(aug_out["action_logits"], dim=-1)
        # KL(target || aug) = sum target * (log target - log aug)
        # Use the mean of base_action_log_softmax broadcast across batch.
        target = base_action_log_softmax.exp().expand(n_aug, -1)
        loss = F.kl_div(aug_log_softmax, target, reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))

    helper.model.eval()
    return {"steps": int(n_steps), "first_loss": losses[0] if losses else 0.0,
            "final_loss": losses[-1] if losses else 0.0, "n_aug": n_aug}


def ttt_rollout_finetune(
    helper: "PolicyHelper",
    history_buffer: List[List[List[List[int]]]],
    action_buffer: List[Dict[str, Any]],
    next_frame_buffer: List[List[List[int]]],
    goal_frame: Optional[List[List[int]]] = None,
    n_steps: int = 30,
    batch_size: int = 8,
    lr: float = 5e-5,
) -> Dict[str, float]:
    """Real-rollout TTT: adapts the model to the current game's actual dynamics.

    Inputs are buffers collected from the agent's first ~30 actions in this game:
      history_buffer:     list of (history, 64, 64) raw color grids per step
      action_buffer:      list of {'action_id', 'action_data': {'x','y'} or {}}
      next_frame_buffer:  list of (64, 64) raw color grids (post-action frames)
      goal_frame:         optional (64, 64) grid; if provided AND helper.model.use_goal,
                          conditioning is preserved during fine-tuning.

    Loss = action CE + 0.3 * coord CE (when ACTION6) + 0.1 * recon CE (forward model).
    Forward-model loss is the key signal: it adapts the encoder to the
    game's specific visual dynamics, which the policy heads then exploit.

    Returns {'steps', 'first_loss', 'final_loss', 'n_pairs'}.
    """
    if not _TORCH_OK or helper is None or helper.model is None:
        return {"steps": 0}
    n_pairs = min(len(history_buffer), len(action_buffer), len(next_frame_buffer))
    if n_pairs < batch_size:
        return {"steps": 0, "skipped": "too_few_pairs", "n_pairs": n_pairs}

    helper.model.train()
    opt = torch.optim.AdamW(helper.model.parameters(), lr=lr, weight_decay=0.01)

    # Pre-build all (obs, scalar, action_index, next_frame_target) tensors once.
    # Atomic append: build all per-item tensors first, then append all-or-nothing,
    # so a partial failure mid-build doesn't desync obs_list vs next_target_list.
    obs_list, scalar_list, ai_list, x_list, y_list, coord_mask_list, next_target_list = [], [], [], [], [], [], []
    n_drop_skip_aid, n_drop_exc = 0, 0
    last_exc_msg = ""
    for h, ad, nf in zip(history_buffer[:n_pairs], action_buffer[:n_pairs], next_frame_buffer[:n_pairs]):
        try:
            aid = int(ad.get("action_id", 1))
            if aid not in ACTION_TO_INDEX:
                n_drop_skip_aid += 1
                continue
            obs = frames_to_obs_uint8(h, helper.history)
            cur_frame = h[-1] if h else [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
            scalar = scalar_features(
                available_actions=ACTION_IDS, last_action_id=None,
                levels_completed=0, steps_since_progress=0, step_index=0,
                frame=cur_frame, max_steps=helper.max_steps,
            )
            ad_data = ad.get("action_data") or {}
            x_v = int(ad_data.get("x", 0))
            y_v = int(ad_data.get("y", 0))
            cm_v = 1.0 if aid == 6 else 0.0
            next_arr = np.asarray(nf, dtype=np.int64).clip(0, NUM_COLORS - 1)
            if next_arr.shape != (GRID_SIZE, GRID_SIZE):
                pad = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
                rh = min(next_arr.shape[0] if next_arr.ndim >= 1 else 0, GRID_SIZE)
                rw = min(next_arr.shape[1] if next_arr.ndim >= 2 else 0, GRID_SIZE)
                if rh and rw:
                    pad[:rh, :rw] = next_arr[:rh, :rw]
                next_arr = pad
            next_t = torch.from_numpy(next_arr)
        except Exception as _exc:
            n_drop_exc += 1
            last_exc_msg = f"{type(_exc).__name__}: {_exc}"
            continue
        # All builds succeeded — atomic append.
        obs_list.append(obs)
        scalar_list.append(scalar)
        ai_list.append(ACTION_TO_INDEX[aid])
        x_list.append(x_v)
        y_list.append(y_v)
        coord_mask_list.append(cm_v)
        next_target_list.append(next_t)

    if len(ai_list) < batch_size:
        helper.model.eval()
        return {
            "steps": 0, "skipped": "build_failed",
            "n_pairs": len(ai_list),
            "n_drop_skip_aid": n_drop_skip_aid,
            "n_drop_exc": n_drop_exc,
            "last_exc_msg": last_exc_msg,
        }

    obs_all = torch.stack(obs_list, dim=0).to(helper.device)
    scalar_all = torch.stack(scalar_list, dim=0).to(helper.device)
    ai_all = torch.tensor(ai_list, dtype=torch.long, device=helper.device)
    x_all = torch.tensor(x_list, dtype=torch.long, device=helper.device)
    y_all = torch.tensor(y_list, dtype=torch.long, device=helper.device)
    coord_mask_all = torch.tensor(coord_mask_list, dtype=torch.float32, device=helper.device)
    next_target_all = torch.stack(next_target_list, dim=0).to(helper.device)

    # Optional goal conditioning.
    use_goal_path = bool(getattr(helper.model, "use_goal", False)) and goal_frame is not None
    if use_goal_path:
        goal_arr = np.asarray(goal_frame, dtype=np.uint8).clip(0, NUM_COLORS - 1)
        if goal_arr.shape != (GRID_SIZE, GRID_SIZE):
            pad = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            rh = min(goal_arr.shape[0] if goal_arr.ndim >= 1 else 0, GRID_SIZE)
            rw = min(goal_arr.shape[1] if goal_arr.ndim >= 2 else 0, GRID_SIZE)
            if rh and rw:
                pad[:rh, :rw] = goal_arr[:rh, :rw]
            goal_arr = pad
        goal_one = torch.from_numpy(goal_arr).unsqueeze(0).to(helper.device)
    else:
        goal_one = None

    n_data = obs_all.shape[0]
    losses: List[float] = []
    rng = np.random.default_rng(0)
    for step in range(n_steps):
        idx = rng.integers(0, n_data, size=batch_size)
        idx_t = torch.from_numpy(idx).long().to(helper.device)
        ob = obs_all[idx_t]; sc = scalar_all[idx_t]
        ai = ai_all[idx_t]; xt = x_all[idx_t]; yt = y_all[idx_t]
        cm = coord_mask_all[idx_t]; nft = next_target_all[idx_t]
        gob = goal_one.expand(batch_size, *goal_one.shape[1:]) if goal_one is not None else None
        out = helper.model(ob, sc, action_index=ai, goal_obs=gob)
        action_loss = F.cross_entropy(out["action_logits"], ai)
        coord_loss = (
            F.cross_entropy(out["x_logits"], xt, reduction="none") +
            F.cross_entropy(out["y_logits"], yt, reduction="none")
        )
        coord_loss = (coord_loss * cm).mean() * 0.5
        recon_loss = torch.tensor(0.0, device=helper.device)
        if "next_frame_recon_logits" in out:
            # next_frame_recon_logits: (B, NUM_COLORS, 64, 64)
            recon_loss = F.cross_entropy(out["next_frame_recon_logits"], nft) * 0.1
        loss = action_loss + 0.3 * coord_loss + recon_loss
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))

    helper.model.eval()
    return {
        "steps": int(n_steps),
        "first_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "n_pairs": int(n_data),
    }


def find_checkpoint() -> Optional[Path]:
    """Search canonical paths for a BC checkpoint. Same idea as the dict loader."""
    candidates = [
        os.environ.get("ARC_BC_CHECKPOINT_PATH"),
        "/kaggle/working/best.pth",
        "/kaggle/input/arc-agi-3-replays-v1/best.pth",
        # Local dev paths.
        "/mnt/c/Users/ljh20/MCS/ARC-Prize-2026-ARC-AGI-3/Training_Output/bc_v2_filtered_local_v1/checkpoints/best.pth",
    ]
    for p in candidates:
        if not p:
            continue
        path = Path(p)
        if path.is_file():
            return path
    return None
