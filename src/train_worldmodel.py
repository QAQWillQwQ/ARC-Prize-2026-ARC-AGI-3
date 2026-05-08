"""Phase 2 of worldmodel_v1: joint training of dynamics + inverse + value + recon-keepalive.

Loads a pretrained encoder from Phase 1 (`pretrain_encoder.py` output), builds
the full WorldModelBundle, and trains all five modules jointly on collected
`.gz` episodes. The encoder is unfrozen by default — the reconstruction-keepalive
loss prevents catastrophic forgetting of slot structure during fine-tuning.

Usage
-----
    python -m src.train_worldmodel \
        --project-root . \
        --data './Local_Output/Collection_Cache/openlab_collect_best_v3/collected/episodes.jsonl.gz' \
        --output-dir ./Training_Output/worldmodel_v1 \
        --pretrained-encoder ./Training_Output/pretrain_v1/encoder.pth \
        --hardware-profile rtx4070super

Eval is **forward-dynamics MSE on a held-out 20% of episodes** — no env rollouts
in Phase 2. The full env eval is Phase 3 (planner integration) and Phase 4.

This script intentionally does NOT touch `src/train.py` or any policy training.
The two coexist as separate entry points so we can compare baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import (
    ACTION_IDS,
    ACTION_TO_INDEX,
    NUM_COLORS,
    append_metrics_row,
    compute_discounted_returns,
    ensure_dir,
    iter_jsonl_gz,
    merge_config,
    safe_mean,
    save_json,
    seed_everything,
    transition_reward,
)
from .world_model import (
    WorldModelBundle,
    build_world_model,
    forward_dynamics_loss,
    inverse_dynamics_loss,
    load_pretrained_encoder_into_bundle,
    reconstruction_loss,
    save_world_model,
)


class WorldModelDataset(Dataset):
    """Per-transition samples for joint world-model training.

    Each item:
        obs_t           (history * NUM_COLORS, 64, 64) float32 one-hot
        obs_tp1         same shape, history shifted by one
        target_frame    (64, 64) long — the *current* frame, decoder target
        action_idx      ()      long, in [0, len(ACTION_IDS)-1]
        x, y            ()      long, in [0, 63]; coords for ACTION6 only (else 0)
        coord_mask      ()      float, 1.0 iff ACTION_IDS[action_idx] == 6
        has_coord       ()      float, identical to coord_mask, kept distinct semantically
        return_target   ()      float, discounted return at this transition
        weight          ()      float, episode-level weight (mirrors EpisodeTransitionDataset rule)
    """

    def __init__(
        self,
        episodes: Sequence[Dict[str, Any]],
        history: int,
    ) -> None:
        self.history = history
        self.episodes: List[Dict[str, Any]] = []
        self.sample_index: List[Tuple[int, int]] = []
        for episode in episodes:
            self._add_episode(episode)

    def _add_episode(self, episode: Dict[str, Any]) -> bool:
        transitions = list(episode.get("transitions", []))
        if not transitions:
            return False

        rewards = [transition_reward(t) for t in transitions]
        returns = compute_discounted_returns(rewards)

        frame_list: List[Sequence[Sequence[int]]] = [transitions[0]["frame"]]
        for transition in transitions:
            frame_list.append(transition["next_frame"])
        frames = torch.tensor(frame_list, dtype=torch.uint8)

        # Episode-level weight (same scheme as train.py — see Track 1B in changelog).
        final_state = str(episode.get("final_state", "NOT_FINISHED"))
        levels_completed = int(episode.get("levels_completed", 0))
        episode_score = float(episode.get("score", 0.0))
        if final_state == "WIN":
            weight_mult = 4.0
        elif levels_completed >= 2:
            weight_mult = 2.5
        elif levels_completed >= 1 or episode_score > 0.0:
            weight_mult = 1.5
        else:
            weight_mult = 0.4

        episode_index = len(self.episodes)
        self.episodes.append(
            {
                "transitions": transitions,
                "frames": frames,
                "returns": returns,
                "score": episode_score,
                "weight_mult": weight_mult,
            }
        )
        self.sample_index.extend((episode_index, idx) for idx in range(len(transitions)))
        return True

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_index, transition_index = self.sample_index[index]
        episode = self.episodes[episode_index]
        transitions = episode["transitions"]
        transition = transitions[transition_index]
        frames = episode["frames"]

        obs_t = self._encode_history(self._history_at(frames, transition_index))
        obs_tp1 = self._encode_history(self._history_at(frames, transition_index + 1))
        target_frame = frames[transition_index].to(dtype=torch.long).clamp_(0, NUM_COLORS - 1)

        action_id = int(transition["action_id"])
        action_idx = ACTION_TO_INDEX[action_id]
        action_data = transition.get("action_data") or {}
        x = int(action_data.get("x", 0))
        y = int(action_data.get("y", 0))
        coord_mask = 1.0 if action_id == 6 else 0.0

        progress = int(transition["levels_after"]) - int(transition["levels_before"])
        weight = float(episode["weight_mult"]) * (
            1.0 + (float(episode["score"]) / 100.0) + max(0, progress) * 2.0
        )

        return {
            "obs_t": obs_t,
            "obs_tp1": obs_tp1,
            "target_frame": target_frame,
            "action_idx": torch.tensor(action_idx, dtype=torch.long),
            "x": torch.tensor(x, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.long),
            "coord_mask": torch.tensor(coord_mask, dtype=torch.float32),
            "has_coord": torch.tensor(coord_mask, dtype=torch.float32),
            "return_target": torch.tensor(float(episode["returns"][transition_index]), dtype=torch.float32),
            "weight": torch.tensor(float(weight), dtype=torch.float32),
        }

    def _history_at(self, frames: torch.Tensor, end_index: int) -> torch.Tensor:
        end_index = min(end_index, frames.shape[0] - 1)
        start_index = max(0, end_index - self.history + 1)
        chunk = frames[start_index : end_index + 1]
        if chunk.shape[0] < self.history:
            pad = chunk[0:1].repeat(self.history - chunk.shape[0], 1, 1)
            chunk = torch.cat([pad, chunk], dim=0)
        return chunk

    @staticmethod
    def _encode_history(history_frames: torch.Tensor) -> torch.Tensor:
        clipped = history_frames.to(dtype=torch.long).clamp_(0, NUM_COLORS - 1)
        encoded = F.one_hot(clipped, num_classes=NUM_COLORS).permute(0, 3, 1, 2)
        return encoded.reshape(-1, encoded.shape[-2], encoded.shape[-1]).to(dtype=torch.float32)


def collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint world-model training (Phase 2 of worldmodel_v1).")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--data", type=str, required=True, help="Comma-separated paths to .gz episode files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Where checkpoints + metrics go.")
    parser.add_argument("--pretrained-encoder", type=str, default=None, help="Phase 1 encoder.pth (optional but recommended).")
    parser.add_argument("--hardware-profile", type=str, default="rtx4070super")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights during training.")
    # Loss weights (override defaults from plan)
    parser.add_argument("--w-forward", type=float, default=1.0)
    parser.add_argument("--w-inverse-action", type=float, default=0.5)
    parser.add_argument("--w-inverse-xy", type=float, default=0.25)
    parser.add_argument("--w-value", type=float, default=0.5)
    parser.add_argument("--w-recon", type=float, default=0.1)
    return parser.parse_args()


def parse_data_paths(raw: str) -> List[Path]:
    paths = [Path(p.strip()).resolve() for p in raw.split(",") if p.strip()]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError("Data path not found: %s" % path)
    return paths


def episode_split_key(episode: Dict[str, Any]) -> str:
    """Same scheme as `train.py: episode_split_key` — deterministic per-episode hash."""
    for key in ("episode_id", "id", "uuid"):
        value = episode.get(key)
        if value is not None:
            return str(value)
    payload = "%s|%s|%s|%s" % (
        episode.get("game_id", "unk"),
        episode.get("seed", 0),
        episode.get("timestamp", ""),
        episode.get("score", 0.0),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def split_episodes(
    episodes: Sequence[Dict[str, Any]],
    val_fraction: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Hash-based deterministic 80/20 split."""
    ranked: List[Tuple[str, Dict[str, Any]]] = []
    for episode in episodes:
        key = episode_split_key(episode)
        digest = hashlib.sha1(("worldmodel-split-" + key).encode("utf-8")).hexdigest()
        ranked.append((digest, episode))
    ranked.sort(key=lambda item: item[0])
    holdout = max(1, int(round(len(ranked) * val_fraction)))
    val = [episode for _, episode in ranked[:holdout]]
    train = [episode for _, episode in ranked[holdout:]]
    return train, val


def iter_episodes_from_paths(
    paths: Sequence[Path],
    max_episodes: Optional[int],
) -> Iterable[Dict[str, Any]]:
    loaded = 0
    for path in paths:
        for episode in iter_jsonl_gz(path):
            yield episode
            loaded += 1
            if max_episodes is not None and loaded >= max_episodes:
                return


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def masked_average(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def run_train_step(
    bundle: WorldModelBundle,
    batch: Dict[str, torch.Tensor],
    weights_cfg: Dict[str, float],
    autocast_enabled: bool,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Single forward + loss computation. Returns un-backwarded loss tensors."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        slots_t = bundle.encoder(batch["obs_t"])
        with torch.no_grad():
            slots_tp1_target = bundle.encoder(batch["obs_tp1"]).detach()

        slots_pred = bundle.forward_model(
            slots_t,
            batch["action_idx"],
            batch["x"],
            batch["y"],
            batch["has_coord"],
        )
        inv_out = bundle.inverse_model(slots_t, slots_tp1_target)
        value_pred = bundle.value_head(slots_t)
        recon_logits = bundle.decoder(slots_t)

        # Forward MSE per-sample so we can apply per-sample weights.
        fwd_per = ((slots_pred - slots_tp1_target) ** 2).mean(dim=(1, 2))
        forward_loss_w = masked_average(fwd_per, batch["weight"])
        forward_loss_unw = fwd_per.mean()

        inv_losses = inverse_dynamics_loss(
            inv_out,
            batch["action_idx"],
            batch["x"],
            batch["y"],
            batch["coord_mask"],
        )

        value_per = F.smooth_l1_loss(value_pred, batch["return_target"], reduction="none")
        value_loss = masked_average(value_per, batch["weight"])

        recon_per = F.cross_entropy(recon_logits, batch["target_frame"], reduction="none").mean(dim=(1, 2))
        recon_loss = masked_average(recon_per, batch["weight"])

        total = (
            weights_cfg["forward"] * forward_loss_w
            + weights_cfg["inv_action"] * inv_losses["action"]
            + weights_cfg["inv_xy"] * (inv_losses["x"] + inv_losses["y"])
            + weights_cfg["value"] * value_loss
            + weights_cfg["recon"] * recon_loss
        )

    return {
        "total": total,
        "forward": forward_loss_w,
        "forward_unw": forward_loss_unw,
        "inv_action": inv_losses["action"],
        "inv_x": inv_losses["x"],
        "inv_y": inv_losses["y"],
        "value": value_loss,
        "recon": recon_loss,
        "slots_pred": slots_pred,
        "slots_target": slots_tp1_target,
        "inv_out": inv_out,
    }


def run_train_epoch(
    bundle: WorldModelBundle,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weights_cfg: Dict[str, float],
    device: torch.device,
    epoch: int,
    log_every_batches: int,
    max_grad_norm: float,
) -> Dict[str, float]:
    bundle.train()
    autocast_enabled = device.type == "cuda"
    sums = {key: 0.0 for key in ("total", "forward", "forward_unw", "inv_action", "inv_x", "inv_y", "value", "recon")}
    n_samples = 0
    correct_action = 0
    started_at = time.time()

    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        out = run_train_step(bundle, batch, weights_cfg, autocast_enabled, device)
        loss = out["total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(list(bundle.parameters()), max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            batch_size = int(batch["action_idx"].shape[0])
            n_samples += batch_size
            for key in sums:
                sums[key] += float(out[key].item()) * batch_size
            preds = out["inv_out"]["action_logits"].argmax(dim=-1)
            correct_action += int((preds == batch["action_idx"]).sum().item())

        if log_every_batches > 0 and (step + 1) % log_every_batches == 0:
            wall = time.time() - started_at
            print(
                "[wm-train] epoch=%d step=%d/%d loss=%.4f fwd=%.4f inv_a=%.4f val=%.4f rec=%.4f wall=%.1fs"
                % (
                    epoch,
                    step + 1,
                    len(loader),
                    sums["total"] / max(1, n_samples),
                    sums["forward_unw"] / max(1, n_samples),
                    sums["inv_action"] / max(1, n_samples),
                    sums["value"] / max(1, n_samples),
                    sums["recon"] / max(1, n_samples),
                    wall,
                ),
                flush=True,
            )

    metrics = {key: value / max(1, n_samples) for key, value in sums.items()}
    metrics["inv_action_acc"] = correct_action / max(1, n_samples)
    return metrics


@torch.no_grad()
def run_val_epoch(
    bundle: WorldModelBundle,
    loader: DataLoader,
    weights_cfg: Dict[str, float],
    device: torch.device,
) -> Dict[str, float]:
    bundle.eval()
    autocast_enabled = device.type == "cuda"
    sums = {key: 0.0 for key in ("total", "forward", "forward_unw", "inv_action", "inv_x", "inv_y", "value", "recon")}
    n_samples = 0
    correct_action = 0

    for batch in loader:
        batch = move_batch(batch, device)
        out = run_train_step(bundle, batch, weights_cfg, autocast_enabled, device)
        batch_size = int(batch["action_idx"].shape[0])
        n_samples += batch_size
        for key in sums:
            sums[key] += float(out[key].item()) * batch_size
        preds = out["inv_out"]["action_logits"].argmax(dim=-1)
        correct_action += int((preds == batch["action_idx"]).sum().item())

    metrics = {key: value / max(1, n_samples) for key, value in sums.items()}
    metrics["inv_action_acc"] = correct_action / max(1, n_samples)
    return metrics


def freeze_module(module: nn.Module) -> int:
    n = 0
    for param in module.parameters():
        param.requires_grad = False
        n += param.numel()
    return n


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")

    overrides: Dict[str, Any] = {}
    for key in ("epochs", "batch_size", "lr", "weight_decay"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    config = merge_config(args.hardware_profile, overrides)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[wm-train] device=%s profile=%s" % (device.type, args.hardware_profile), flush=True)

    weights_cfg = {
        "forward": float(args.w_forward),
        "inv_action": float(args.w_inverse_action),
        "inv_xy": float(args.w_inverse_xy),
        "value": float(args.w_value),
        "recon": float(args.w_recon),
    }
    print("[wm-train] loss_weights=%s" % weights_cfg, flush=True)

    data_paths = parse_data_paths(args.data)
    print("[wm-train] loading episodes from %d file(s)" % len(data_paths), flush=True)
    episodes = list(iter_episodes_from_paths(data_paths, args.max_episodes))
    if not episodes:
        raise RuntimeError("No episodes loaded — check --data paths.")
    train_episodes, val_episodes = split_episodes(episodes, args.val_fraction)
    print(
        "[wm-train] episodes total=%d train=%d val=%d (val_fraction=%.2f)"
        % (len(episodes), len(train_episodes), len(val_episodes), args.val_fraction),
        flush=True,
    )

    train_dataset = WorldModelDataset(train_episodes, history=int(config["history"]))
    val_dataset = WorldModelDataset(val_episodes, history=int(config["history"]))
    print("[wm-train] transitions train=%d val=%d" % (len(train_dataset), len(val_dataset)), flush=True)

    if len(train_dataset) == 0:
        raise RuntimeError("Empty train set — increase data or lower --val-fraction.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            collate_fn=collate,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        if len(val_dataset) > 0
        else None
    )

    bundle = build_world_model(config).to(device)
    if args.pretrained_encoder:
        load_pretrained_encoder_into_bundle(bundle, args.pretrained_encoder, device)
        print("[wm-train] loaded pretrained encoder from %s" % args.pretrained_encoder, flush=True)

    if args.freeze_encoder:
        n_frozen = freeze_module(bundle.encoder)
        print("[wm-train] encoder FROZEN (%d params)" % n_frozen, flush=True)

    def n(m): return sum(p.numel() for p in m.parameters()) / 1e6
    print(
        "[wm-train] params: enc=%.2fM dec=%.2fM fwd=%.2fM inv=%.2fM val=%.2fM total=%.2fM"
        % (
            n(bundle.encoder),
            n(bundle.decoder),
            n(bundle.forward_model),
            n(bundle.inverse_model),
            n(bundle.value_head),
            sum(n(m) for m in [bundle.encoder, bundle.decoder, bundle.forward_model, bundle.inverse_model, bundle.value_head]),
        ),
        flush=True,
    )

    trainable_params = [p for p in bundle.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"

    epochs = int(config["epochs"])
    best_val = float("inf")
    best_epoch = -1
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_metrics = run_train_epoch(
            bundle=bundle,
            loader=train_loader,
            optimizer=optimizer,
            weights_cfg=weights_cfg,
            device=device,
            epoch=epoch,
            log_every_batches=int(args.log_every_batches),
            max_grad_norm=float(args.max_grad_norm),
        )
        val_metrics = (
            run_val_epoch(bundle=bundle, loader=val_loader, weights_cfg=weights_cfg, device=device)
            if val_loader is not None
            else {key: float("nan") for key in train_metrics}
        )
        epoch_wall = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "wall_seconds": round(epoch_wall, 2),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_metrics_row(metrics_path, row)
        history.append(row)
        print(
            "[wm-train] epoch=%d done train_total=%.4f train_fwd=%.4f val_fwd=%.4f val_inv_acc=%.3f val_recon=%.4f wall=%.1fs"
            % (
                epoch,
                train_metrics["total"],
                train_metrics["forward_unw"],
                val_metrics["forward_unw"],
                val_metrics["inv_action_acc"],
                val_metrics["recon"],
                epoch_wall,
            ),
            flush=True,
        )

        # Save last every epoch.
        save_world_model(
            path=str(checkpoints_dir / "last.pth"),
            bundle=bundle,
            optimizer=optimizer,
            epoch=epoch,
            best_score=-best_val,
            metrics=val_metrics,
        )
        # Save best by val forward MSE (lower is better).
        val_score = float(val_metrics.get("forward_unw", float("inf")))
        if val_score < best_val and val_loader is not None:
            best_val = val_score
            best_epoch = epoch
            save_world_model(
                path=str(checkpoints_dir / "best.pth"),
                bundle=bundle,
                optimizer=optimizer,
                epoch=epoch,
                best_score=-best_val,
                metrics=val_metrics,
            )
            print("[wm-train] saved best -> %s (val_forward_mse=%.4f)" % (checkpoints_dir / "best.pth", best_val), flush=True)

    save_json(
        summary_path,
        {
            "profile": args.hardware_profile,
            "config": config,
            "loss_weights": weights_cfg,
            "data_paths": [str(path) for path in data_paths],
            "episodes_total": len(episodes),
            "episodes_train": len(train_episodes),
            "episodes_val": len(val_episodes),
            "transitions_train": len(train_dataset),
            "transitions_val": len(val_dataset),
            "best_val_forward_mse": best_val,
            "best_epoch": best_epoch,
            "freeze_encoder": bool(args.freeze_encoder),
            "pretrained_encoder": args.pretrained_encoder,
            "history": history,
        },
    )
    print("[wm-train] summary -> %s" % summary_path, flush=True)


if __name__ == "__main__":
    main()
