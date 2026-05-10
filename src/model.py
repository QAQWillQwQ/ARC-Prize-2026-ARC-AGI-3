from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .common import ACTION_IDS, NUM_COLORS


class ObjectCentricPolicy(nn.Module):
    def __init__(
        self,
        history: int = 4,
        model_dim: int = 384,
        num_slots: int = 8,
        depth: int = 6,
        num_heads: int = 8,
        scalar_dim: int = 18,
    ) -> None:
        super().__init__()
        input_channels = history * 16
        hidden = max(model_dim // 2, 128)
        self.history = history
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
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.state_norm = nn.LayerNorm(model_dim)

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

        # Option D aux heads. Off-path at inference (PolicyGuidedAgent doesn't read these),
        # so wiring them in is purely a training-time signal.

        # Archetype: pooled state → 3-way classifier (click_dominant / keyboard_dominant / mixed).
        self.archetype_head = nn.Linear(model_dim, 3)

        # Saliency: patches (B, 256, model_dim) → reshape (B, model_dim, 16, 16)
        # → ConvT-stride2 ×2 → (B, 1, 64, 64) logit map.
        saliency_hidden = max(model_dim // 4, 32)
        self._saliency_hidden = saliency_hidden
        self.saliency_decoder = nn.Sequential(
            nn.Conv2d(model_dim, saliency_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(saliency_hidden, saliency_hidden, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(saliency_hidden, 1, kernel_size=4, stride=2, padding=1),
        )

        # Next-frame recon: pred_next_latent (B, model_dim) → spatial broadcast
        # → ConvT-stride2 ×4 → (B, NUM_COLORS, 64, 64) per-pixel color logits.
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

    def encode_state(self, obs: torch.Tensor, scalar: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Accept either raw color indices or pre-encoded one-hot frames so the
        # model is backward-compatible with PolicyGuidedAgent (which calls
        # `one_hot_frames` on CPU before model forward) AND with the optimized
        # training path (which ships uint8 frames and lets the GPU one_hot
        # them — eliminates the dominant CPU cost in the DataLoader workers).
        #
        # Detection rule: obs with channel-dim == self.history*16 is already
        # one-hot encoded float32; obs with channel-dim == self.history is raw.
        # Both shapes are (B, C, 64, 64).
        if obs.dim() == 4 and obs.shape[1] == self.history:
            # Raw color indices path: one_hot on GPU, much faster than CPU.
            clipped = obs.to(dtype=torch.long).clamp_(0, NUM_COLORS - 1)
            # (B, history, 64, 64) -> (B, history, 64, 64, NUM_COLORS) -> (B, history*NUM_COLORS, 64, 64)
            one_hot = F.one_hot(clipped, num_classes=NUM_COLORS)
            obs = one_hot.permute(0, 1, 4, 2, 3).reshape(
                obs.shape[0], self.history * NUM_COLORS, 64, 64
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
        tokens = torch.cat([scalar_token, slot_tokens], dim=1)
        tokens = self.encoder(tokens)
        pooled = self.state_norm(tokens[:, 0, :])
        return {
            "pooled": pooled,
            "tokens": tokens,
            "patches": patches,
        }

    def forward(
        self,
        obs: torch.Tensor,
        scalar: torch.Tensor,
        action_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_state(obs, scalar)
        pooled = encoded["pooled"]
        patches = encoded["patches"]
        out: Dict[str, torch.Tensor] = {
            "pooled": pooled,
            "action_logits": self.action_head(pooled),
            "x_logits": self.x_head(pooled),
            "y_logits": self.y_head(pooled),
            "value": self.value_head(pooled).squeeze(-1),
            "avail_logits": self.avail_head(pooled),
            "archetype_logits": self.archetype_head(pooled),
        }

        # Saliency map from spatial patches (15×15→ but our grid is 16×16=256 tokens).
        batch_size = patches.shape[0]
        patch_grid = patches.transpose(1, 2).reshape(batch_size, self.model_dim, 16, 16)
        out["saliency_logits"] = self.saliency_decoder(patch_grid).squeeze(1)

        if action_index is not None:
            action_emb = self.action_embed(action_index)
            pred_next_latent = self.next_latent_head(
                torch.cat([pooled, action_emb], dim=-1)
            )
            out["pred_next_latent"] = pred_next_latent
            recon_seed = self.recon_init(pred_next_latent).reshape(
                batch_size, self._recon_hidden, 4, 4
            )
            out["next_frame_recon_logits"] = self.recon_decoder(recon_seed)

        return out


def build_model(config: Dict[str, Any], scalar_dim: int = 18) -> ObjectCentricPolicy:
    return ObjectCentricPolicy(
        history=int(config["history"]),
        model_dim=int(config["model_dim"]),
        num_slots=int(config["num_slots"]),
        depth=int(config["depth"]),
        num_heads=int(config["num_heads"]),
        scalar_dim=scalar_dim,
    )


def save_checkpoint(
    path: str,
    model: ObjectCentricPolicy,
    optimizer: Optional[torch.optim.Optimizer],
    config: Dict[str, Any],
    epoch: int,
    best_score: float,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "epoch": epoch,
        "best_score": best_score,
    }
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)
