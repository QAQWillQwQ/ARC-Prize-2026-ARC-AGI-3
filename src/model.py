from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .common import ACTION_IDS


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

    def encode_state(self, obs: torch.Tensor, scalar: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        out: Dict[str, torch.Tensor] = {
            "pooled": pooled,
            "action_logits": self.action_head(pooled),
            "x_logits": self.x_head(pooled),
            "y_logits": self.y_head(pooled),
            "value": self.value_head(pooled).squeeze(-1),
            "avail_logits": self.avail_head(pooled),
        }
        if action_index is not None:
            action_emb = self.action_embed(action_index)
            out["pred_next_latent"] = self.next_latent_head(
                torch.cat([pooled, action_emb], dim=-1)
            )
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
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "epoch": epoch,
        "best_score": best_score,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)

