"""World-model components for the worldmodel_v1 branch.

Phase 1: object-centric encoder + spatial-broadcast reconstruction decoder.
Phase 2: forward / inverse dynamics + value head + WorldModelBundle.

The encoder mirrors `ObjectCentricPolicy.encode_state` up through the slot
cross-attention, so pretrained weights load 1:1 into the corresponding
submodules of the policy.

Phase 3 (planner) will live in `agent_worldmodel.py`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .common import ACTION_IDS, GRID_SIZE, NUM_COLORS


class ObjectCentricEncoder(nn.Module):
    """Conv stack + slot cross-attention. Returns slot tokens.

    Architecture is intentionally identical to `ObjectCentricPolicy.encode_state`
    up through `slot_tokens` (no scalar branch, no transformer encoder, no heads),
    so a pretrained `state_dict` here loads directly into the policy's
    `conv`, `pos_embed`, `slot_queries`, and `cross_attn` submodules.

    Input:  obs    (B, history * NUM_COLORS, 64, 64) — one-hot history stack
    Output: slots  (B, num_slots, model_dim)
    """

    def __init__(
        self,
        history: int = 4,
        num_colors: int = NUM_COLORS,
        model_dim: int = 384,
        num_slots: int = 8,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        input_channels = history * num_colors
        hidden = max(model_dim // 2, 128)
        self.history = history
        self.num_colors = num_colors
        self.model_dim = model_dim
        self.num_slots = num_slots

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
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.conv(obs)
        batch_size = features.shape[0]
        patches = features.flatten(2).transpose(1, 2)
        patches = patches + self.pos_embed[:, : patches.shape[1], :]
        slots = self.slot_queries.expand(batch_size, -1, -1)
        slot_tokens, _ = self.cross_attn(query=slots, key=patches, value=patches)
        return slot_tokens

    def config(self) -> Dict[str, Any]:
        return {
            "history": self.history,
            "num_colors": self.num_colors,
            "model_dim": self.model_dim,
            "num_slots": self.num_slots,
        }


class SpatialBroadcastDecoder(nn.Module):
    """Slot-Attention-style decoder: each slot is upsampled to a (H, W) grid via
    transposed convolutions, predicting per-pixel color logits + an alpha logit.

    Per-slot outputs are composited with softmax-over-slots on the alpha to
    produce the final reconstruction.

    Input:  slots   (B, num_slots, model_dim)
    Output: logits  (B, NUM_COLORS, 64, 64)
    """

    def __init__(
        self,
        model_dim: int = 384,
        num_colors: int = NUM_COLORS,
        decoder_dim: int = 64,
        spatial_size: int = GRID_SIZE,
        start_size: int = 4,
    ) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.start_size = start_size
        self.num_colors = num_colors
        self.decoder_dim = decoder_dim

        # Project each slot to a small spatial grid.
        self.broadcast_proj = nn.Linear(model_dim, decoder_dim * start_size * start_size)

        # Upsample 4x4 -> 8 -> 16 -> 32 -> 64 with transposed convolutions.
        upsampling_layers = []
        size = start_size
        while size < spatial_size:
            upsampling_layers.extend(
                [
                    nn.ConvTranspose2d(decoder_dim, decoder_dim, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                ]
            )
            size *= 2
        if size != spatial_size:
            raise ValueError("spatial_size must be start_size * 2^N (got %d, %d)" % (start_size, spatial_size))

        # Final 3x3 conv to color logits + 1 alpha channel.
        upsampling_layers.append(nn.Conv2d(decoder_dim, num_colors + 1, kernel_size=3, padding=1))
        self.deconv = nn.Sequential(*upsampling_layers)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        batch_size, num_slots, model_dim = slots.shape
        flat_slots = slots.reshape(batch_size * num_slots, model_dim)
        broadcast = self.broadcast_proj(flat_slots)
        broadcast = broadcast.reshape(batch_size * num_slots, self.decoder_dim, self.start_size, self.start_size)
        deconv = self.deconv(broadcast)
        # deconv: (B*K, num_colors+1, H, W)
        deconv = deconv.reshape(batch_size, num_slots, self.num_colors + 1, self.spatial_size, self.spatial_size)
        color_logits_per_slot = deconv[:, :, : self.num_colors, :, :]
        alpha_logits = deconv[:, :, self.num_colors, :, :]
        # Softmax across slots so each pixel is "owned" by one slot.
        alpha = F.softmax(alpha_logits, dim=1).unsqueeze(2)
        # Weighted sum across slots -> (B, num_colors, H, W)
        recon_logits = (alpha * color_logits_per_slot).sum(dim=1)
        return recon_logits

    def slot_alpha_maps(self, slots: torch.Tensor) -> torch.Tensor:
        """Diagnostic helper: returns per-slot alpha masks (B, num_slots, H, W)."""
        batch_size, num_slots, model_dim = slots.shape
        flat_slots = slots.reshape(batch_size * num_slots, model_dim)
        broadcast = self.broadcast_proj(flat_slots)
        broadcast = broadcast.reshape(batch_size * num_slots, self.decoder_dim, self.start_size, self.start_size)
        deconv = self.deconv(broadcast)
        deconv = deconv.reshape(batch_size, num_slots, self.num_colors + 1, self.spatial_size, self.spatial_size)
        alpha_logits = deconv[:, :, self.num_colors, :, :]
        return F.softmax(alpha_logits, dim=1)

    def config(self) -> Dict[str, Any]:
        return {
            "model_dim": self.broadcast_proj.in_features,
            "num_colors": self.num_colors,
            "decoder_dim": self.decoder_dim,
            "spatial_size": self.spatial_size,
            "start_size": self.start_size,
        }


@dataclass
class PretrainBundle:
    """Container for an encoder + decoder pair plus their shared config."""

    encoder: ObjectCentricEncoder
    decoder: SpatialBroadcastDecoder
    config: Dict[str, Any]


def build_pretrain_bundle(config: Dict[str, Any]) -> PretrainBundle:
    """Construct an encoder + decoder pair from a runtime config dict.

    Honours profile keys: history, model_dim, num_slots, num_heads, decoder_dim.
    Defaults are sensible for `rtx4070super`.
    """
    encoder = ObjectCentricEncoder(
        history=int(config["history"]),
        num_colors=NUM_COLORS,
        model_dim=int(config["model_dim"]),
        num_slots=int(config["num_slots"]),
        num_heads=int(config["num_heads"]),
    )
    decoder = SpatialBroadcastDecoder(
        model_dim=int(config["model_dim"]),
        num_colors=NUM_COLORS,
        decoder_dim=int(config.get("decoder_dim", 64)),
        spatial_size=GRID_SIZE,
        start_size=4,
    )
    return PretrainBundle(
        encoder=encoder,
        decoder=decoder,
        config={
            "history": int(config["history"]),
            "model_dim": int(config["model_dim"]),
            "num_slots": int(config["num_slots"]),
            "num_heads": int(config["num_heads"]),
            "decoder_dim": int(config.get("decoder_dim", 64)),
        },
    )


def reconstruction_loss(
    recon_logits: torch.Tensor,
    target_frame: torch.Tensor,
) -> torch.Tensor:
    """Per-pixel cross-entropy. recon_logits (B, num_colors, H, W); target (B, H, W) long."""
    return F.cross_entropy(recon_logits, target_frame.long())


def encode_history_obs(history_frames: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
    """Convert a Python history-of-frames list to the encoder's input tensor.

    history_frames[i] is a 64x64 list of color indices. Returns a tensor of
    shape (history * NUM_COLORS, 64, 64) — same encoding as `one_hot_frames`
    in `common.py`, but verified compatible with the world-model encoder shape.
    """
    history = len(history_frames)
    encoded = torch.zeros((history * NUM_COLORS, GRID_SIZE, GRID_SIZE), dtype=torch.float32)
    for frame_idx, frame in enumerate(history_frames):
        for y in range(GRID_SIZE):
            row = frame[y]
            for x in range(GRID_SIZE):
                color = int(row[x])
                if color < 0 or color >= NUM_COLORS:
                    color = 0
                encoded[frame_idx * NUM_COLORS + color, y, x] = 1.0
    return encoded


def save_pretrained_encoder(
    path: str,
    encoder: ObjectCentricEncoder,
    config: Dict[str, Any],
    epoch: int,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save just the encoder weights + minimal config for Phase 2 loading."""
    payload: Dict[str, Any] = {
        "encoder_state": encoder.state_dict(),
        "encoder_config": encoder.config(),
        "training_config": dict(config),
        "epoch": int(epoch),
    }
    if metrics is not None:
        payload["metrics"] = dict(metrics)
    torch.save(payload, path)


def load_pretrained_encoder(path: str, device: torch.device) -> ObjectCentricEncoder:
    """Reverse of save_pretrained_encoder. Returns a ready-to-use encoder on `device`."""
    payload = torch.load(path, map_location=device, weights_only=False)
    encoder_config = payload["encoder_config"]
    encoder = ObjectCentricEncoder(
        history=int(encoder_config["history"]),
        num_colors=int(encoder_config["num_colors"]),
        model_dim=int(encoder_config["model_dim"]),
        num_slots=int(encoder_config["num_slots"]),
    )
    encoder.load_state_dict(payload["encoder_state"])
    encoder.to(device)
    return encoder


# ---------------------------------------------------------------------------
# Phase 2 components
# ---------------------------------------------------------------------------


class ActionEmbedding(nn.Module):
    """Embed (action_id, x, y) into a single token of shape (B, model_dim).

    For non-ACTION6 actions, x/y contributions are zeroed via the `has_coord`
    mask so simple actions and clicks don't share coord tokens.
    """

    def __init__(self, model_dim: int, num_actions: int = 7, grid_size: int = GRID_SIZE) -> None:
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, model_dim)
        # Half each axis -> concatenated to model_dim total for the coord part.
        coord_half = model_dim // 2
        self.x_emb = nn.Embedding(grid_size, coord_half)
        self.y_emb = nn.Embedding(grid_size, model_dim - coord_half)

    def forward(
        self,
        action_idx: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        has_coord: torch.Tensor,
    ) -> torch.Tensor:
        action_token = self.action_emb(action_idx)  # (B, model_dim)
        coord_token = torch.cat([self.x_emb(x), self.y_emb(y)], dim=-1)  # (B, model_dim)
        coord_token = coord_token * has_coord.to(coord_token.dtype).unsqueeze(-1)
        return action_token + coord_token


class ForwardDynamicsModel(nn.Module):
    """Predict slots_{t+1} given slots_t and an action token.

    Uses a small Transformer encoder with the action token appended to the
    slot sequence. The output is the slots-portion of the sequence — the
    action token's output is dropped.

    Input:  slots      (B, K, model_dim)
            action_idx (B,)            in [0, len(ACTION_IDS) - 1]
            x, y       (B,)            in [0, GRID_SIZE - 1]
            has_coord  (B,) bool       True only for ACTION6
    Output: slots_pred (B, K, model_dim)
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        depth: int = 2,
        num_actions: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.depth = depth
        self.action_proj = ActionEmbedding(model_dim=model_dim, num_actions=num_actions)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        slots: torch.Tensor,
        action_idx: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        has_coord: torch.Tensor,
    ) -> torch.Tensor:
        action_token = self.action_proj(action_idx, x, y, has_coord).unsqueeze(1)
        seq = torch.cat([slots, action_token], dim=1)
        out = self.transformer(seq)
        slots_out = out[:, : slots.shape[1]]
        return self.out_norm(slots_out)


class InverseDynamicsModel(nn.Module):
    """Predict (action_id, x, y) given slots_t and slots_{t+1}.

    A learnable [CLS] token is prepended to the concatenated slot streams,
    and the model outputs action / x / y logits on the [CLS] output.

    Input:  slots_t    (B, K, model_dim)
            slots_tp1  (B, K, model_dim)
    Output: dict with action_logits (B, num_actions), x_logits (B, GRID_SIZE),
            y_logits (B, GRID_SIZE).
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        depth: int = 2,
        num_actions: int = 7,
        grid_size: int = GRID_SIZE,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_actions = num_actions
        self.grid_size = grid_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        # Two learned segment embeddings to distinguish slots_t vs slots_tp1
        self.segment_emb = nn.Parameter(torch.randn(2, 1, model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.cls_norm = nn.LayerNorm(model_dim)
        self.action_head = nn.Linear(model_dim, num_actions)
        self.x_head = nn.Linear(model_dim, grid_size)
        self.y_head = nn.Linear(model_dim, grid_size)

    def forward(self, slots_t: torch.Tensor, slots_tp1: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = slots_t.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        slots_t_seg = slots_t + self.segment_emb[0]
        slots_tp1_seg = slots_tp1 + self.segment_emb[1]
        seq = torch.cat([cls, slots_t_seg, slots_tp1_seg], dim=1)
        out = self.transformer(seq)
        cls_out = self.cls_norm(out[:, 0])
        return {
            "action_logits": self.action_head(cls_out),
            "x_logits": self.x_head(cls_out),
            "y_logits": self.y_head(cls_out),
        }


class ValueHead(nn.Module):
    """Mean-pool slots, project to a scalar value estimate."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, 1)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        pooled = self.norm(slots.mean(dim=1))
        return self.head(pooled).squeeze(-1)


@dataclass
class WorldModelBundle:
    """All Phase 2 modules + their shared config."""

    encoder: ObjectCentricEncoder
    decoder: SpatialBroadcastDecoder
    forward_model: ForwardDynamicsModel
    inverse_model: InverseDynamicsModel
    value_head: ValueHead
    config: Dict[str, Any]

    def parameters(self):
        for module in (
            self.encoder,
            self.decoder,
            self.forward_model,
            self.inverse_model,
            self.value_head,
        ):
            yield from module.parameters()

    def to(self, device: torch.device) -> "WorldModelBundle":
        self.encoder.to(device)
        self.decoder.to(device)
        self.forward_model.to(device)
        self.inverse_model.to(device)
        self.value_head.to(device)
        return self

    def train(self, mode: bool = True) -> "WorldModelBundle":
        self.encoder.train(mode)
        self.decoder.train(mode)
        self.forward_model.train(mode)
        self.inverse_model.train(mode)
        self.value_head.train(mode)
        return self

    def eval(self) -> "WorldModelBundle":
        return self.train(False)


def build_world_model(config: Dict[str, Any]) -> WorldModelBundle:
    """Construct the full Phase 2 world model from a runtime config dict.

    Honours profile keys: history, model_dim, num_slots, num_heads, decoder_dim.
    Forward / inverse depths default to 2; tuneable via config keys
    `forward_depth`, `inverse_depth`. The encoder is FRESH — call
    `load_pretrained_encoder` separately and copy its state dict if you have
    a Phase 1 checkpoint.
    """
    encoder = ObjectCentricEncoder(
        history=int(config["history"]),
        num_colors=NUM_COLORS,
        model_dim=int(config["model_dim"]),
        num_slots=int(config["num_slots"]),
        num_heads=int(config["num_heads"]),
    )
    decoder = SpatialBroadcastDecoder(
        model_dim=int(config["model_dim"]),
        num_colors=NUM_COLORS,
        decoder_dim=int(config.get("decoder_dim", 64)),
        spatial_size=GRID_SIZE,
        start_size=4,
    )
    forward_model = ForwardDynamicsModel(
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        depth=int(config.get("forward_depth", 2)),
        num_actions=len(ACTION_IDS),
    )
    inverse_model = InverseDynamicsModel(
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        depth=int(config.get("inverse_depth", 2)),
        num_actions=len(ACTION_IDS),
        grid_size=GRID_SIZE,
    )
    value_head = ValueHead(model_dim=int(config["model_dim"]))

    bundle_config = {
        "history": int(config["history"]),
        "model_dim": int(config["model_dim"]),
        "num_slots": int(config["num_slots"]),
        "num_heads": int(config["num_heads"]),
        "decoder_dim": int(config.get("decoder_dim", 64)),
        "forward_depth": int(config.get("forward_depth", 2)),
        "inverse_depth": int(config.get("inverse_depth", 2)),
        "num_actions": len(ACTION_IDS),
        "grid_size": GRID_SIZE,
    }
    return WorldModelBundle(
        encoder=encoder,
        decoder=decoder,
        forward_model=forward_model,
        inverse_model=inverse_model,
        value_head=value_head,
        config=bundle_config,
    )


def forward_dynamics_loss(
    slots_pred: torch.Tensor,
    slots_target: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared error between predicted and (detached) target slots.

    Targets must already be detached by the caller — we don't detach here so
    the call site is explicit about the stop-gradient.
    """
    return F.mse_loss(slots_pred, slots_target)


def inverse_dynamics_loss(
    inv_out: Dict[str, torch.Tensor],
    action_idx: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    coord_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Cross-entropy on action; coord CE gated by ACTION6 mask.

    Returns a dict with `action`, `x`, `y` losses (each scalar) so the trainer
    can apply per-term weights.
    """
    action_loss = F.cross_entropy(inv_out["action_logits"], action_idx)
    x_loss_per = F.cross_entropy(inv_out["x_logits"], x, reduction="none")
    y_loss_per = F.cross_entropy(inv_out["y_logits"], y, reduction="none")
    coord_denom = coord_mask.sum().clamp_min(1.0)
    x_loss = (x_loss_per * coord_mask).sum() / coord_denom
    y_loss = (y_loss_per * coord_mask).sum() / coord_denom
    return {"action": action_loss, "x": x_loss, "y": y_loss}


def save_world_model(
    path: str,
    bundle: WorldModelBundle,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_score: float = float("-inf"),
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save all Phase 2 modules + config + optimizer state."""
    payload: Dict[str, Any] = {
        "encoder_state": bundle.encoder.state_dict(),
        "decoder_state": bundle.decoder.state_dict(),
        "forward_model_state": bundle.forward_model.state_dict(),
        "inverse_model_state": bundle.inverse_model.state_dict(),
        "value_head_state": bundle.value_head.state_dict(),
        "bundle_config": dict(bundle.config),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_score": float(best_score),
    }
    if metrics is not None:
        payload["metrics"] = dict(metrics)
    torch.save(payload, path)


def load_world_model(path: str, device: torch.device) -> WorldModelBundle:
    """Reverse of save_world_model. Returns a populated WorldModelBundle on `device`."""
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["bundle_config"]
    bundle = build_world_model(
        {
            "history": int(config["history"]),
            "model_dim": int(config["model_dim"]),
            "num_slots": int(config["num_slots"]),
            "num_heads": int(config["num_heads"]),
            "decoder_dim": int(config["decoder_dim"]),
            "forward_depth": int(config.get("forward_depth", 2)),
            "inverse_depth": int(config.get("inverse_depth", 2)),
        }
    )
    bundle.encoder.load_state_dict(payload["encoder_state"])
    bundle.decoder.load_state_dict(payload["decoder_state"])
    bundle.forward_model.load_state_dict(payload["forward_model_state"])
    bundle.inverse_model.load_state_dict(payload["inverse_model_state"])
    bundle.value_head.load_state_dict(payload["value_head_state"])
    bundle.to(device)
    return bundle


def load_pretrained_encoder_into_bundle(
    bundle: WorldModelBundle,
    pretrained_path: str,
    device: torch.device,
) -> None:
    """Copy weights from a Phase 1 encoder checkpoint into `bundle.encoder`.

    Validates that encoder configs are compatible (same history/model_dim/num_slots).
    """
    payload = torch.load(pretrained_path, map_location=device, weights_only=False)
    src_config = payload["encoder_config"]
    own = bundle.encoder.config()
    for key in ("history", "model_dim", "num_slots"):
        if int(src_config[key]) != int(own[key]):
            raise ValueError(
                "Pretrained encoder config mismatch on '%s': pretrained=%s, bundle=%s"
                % (key, src_config[key], own[key])
            )
    bundle.encoder.load_state_dict(payload["encoder_state"])
    bundle.encoder.to(device)
