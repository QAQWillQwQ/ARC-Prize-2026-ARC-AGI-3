"""Phase 1 of worldmodel_v1: pretrain the object-centric encoder via reconstruction.

Reads collected episode `.gz` files, decodes each transition's frame as the
reconstruction target, trains an `ObjectCentricEncoder` + `SpatialBroadcastDecoder`
pair with per-pixel cross-entropy. Saves only the encoder for Phase 2 to reuse.

Usage
-----
    python -m src.pretrain_encoder \
        --project-root . \
        --data './Local_Output/Collection_Cache/openlab_collect_best_v3/collected/episodes.jsonl.gz' \
        --output ./Training_Output/pretrain_v1/encoder.pth \
        --hardware-profile rtx4070super \
        --epochs 1

The script intentionally does NOT touch `src/model.py` or any policy training.
Phase 2 will load the saved encoder into the policy's matching submodules.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import (
    GRID_SIZE,
    NUM_COLORS,
    append_metrics_row,
    ensure_dir,
    iter_jsonl_gz,
    merge_config,
    safe_mean,
    save_json,
    seed_everything,
)
from .world_model import (
    ObjectCentricEncoder,
    SpatialBroadcastDecoder,
    build_pretrain_bundle,
    reconstruction_loss,
    save_pretrained_encoder,
)


class FrameReconstructionDataset(Dataset):
    """Yields (history_obs, target_frame_long) pairs sampled from collected episodes.

    The target is the most recent frame in the history window — the encoder
    sees the same history that downstream training will see, but the decoder
    only has to reconstruct the *current* (last) frame of that window.
    """

    def __init__(
        self,
        episode_paths: Sequence[Path],
        history: int,
        max_episodes: Optional[int] = None,
        max_frames_per_episode: Optional[int] = None,
    ) -> None:
        self.history = history
        self.frame_index: List[Tuple[int, int]] = []
        self.episode_frames: List[torch.Tensor] = []
        self.episode_count = 0
        loaded_episodes = 0
        for path in episode_paths:
            for episode in iter_jsonl_gz(path):
                if max_episodes is not None and loaded_episodes >= max_episodes:
                    break
                if not self._add_episode(episode, max_frames_per_episode):
                    continue
                loaded_episodes += 1
            if max_episodes is not None and loaded_episodes >= max_episodes:
                break
        self.episode_count = len(self.episode_frames)

    def _add_episode(self, episode: Dict[str, Any], max_frames: Optional[int]) -> bool:
        transitions = list(episode.get("transitions", []))
        if not transitions:
            return False
        frame_list: List[Sequence[Sequence[int]]] = [transitions[0]["frame"]]
        for transition in transitions:
            frame_list.append(transition["next_frame"])
        if max_frames is not None and len(frame_list) > max_frames:
            frame_list = frame_list[:max_frames]
        frames = torch.tensor(frame_list, dtype=torch.uint8)
        episode_idx = len(self.episode_frames)
        self.episode_frames.append(frames)
        for frame_idx in range(frames.shape[0]):
            self.frame_index.append((episode_idx, frame_idx))
        return True

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_idx, frame_idx = self.frame_index[index]
        frames = self.episode_frames[episode_idx]
        history_frames = self._history_at(frames, frame_idx)
        target_frame = frames[frame_idx].to(dtype=torch.long).clamp_(0, NUM_COLORS - 1)
        clipped = history_frames.to(dtype=torch.long).clamp_(0, NUM_COLORS - 1)
        encoded = F.one_hot(clipped, num_classes=NUM_COLORS).permute(0, 3, 1, 2)
        obs = encoded.reshape(-1, encoded.shape[-2], encoded.shape[-1]).to(dtype=torch.float32)
        return {"obs": obs, "target": target_frame}

    def _history_at(self, frames: torch.Tensor, end_index: int) -> torch.Tensor:
        start_index = max(0, end_index - self.history + 1)
        chunk = frames[start_index : end_index + 1]
        if chunk.shape[0] < self.history:
            pad = chunk[0:1].repeat(self.history - chunk.shape[0], 1, 1)
            chunk = torch.cat([pad, chunk], dim=0)
        return chunk


def collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def parse_data_paths(raw: str) -> List[Path]:
    paths = [Path(p.strip()).resolve() for p in raw.split(",") if p.strip()]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError("Data path not found: %s" % path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the worldmodel_v1 encoder via frame reconstruction.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--data", type=str, required=True, help="Comma-separated paths to .gz episode files.")
    parser.add_argument("--output", type=str, required=True, help="Where to save encoder.pth.")
    parser.add_argument("--hardware-profile", type=str, default="rtx4070super")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=None, help="Cap episodes loaded — useful for smoke tests.")
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    return parser.parse_args()


def run_epoch(
    encoder: ObjectCentricEncoder,
    decoder: SpatialBroadcastDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every_batches: int,
) -> Dict[str, float]:
    encoder.train()
    decoder.train()
    autocast_enabled = device.type == "cuda"
    total_loss_weighted = 0.0
    total_samples = 0
    total_pixels = 0
    total_correct = 0
    seen_batches = 0
    started_at = time.time()

    for step, batch in enumerate(loader):
        obs = batch["obs"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            slots = encoder(obs)
            recon_logits = decoder(slots)
            loss = reconstruction_loss(recon_logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = recon_logits.argmax(dim=1)
            batch_size = int(target.shape[0])
            total_correct += int((preds == target).sum().item())
            total_pixels += int(target.numel())
            total_loss_weighted += float(loss.item()) * batch_size
            total_samples += batch_size
            seen_batches += 1

        if log_every_batches > 0 and (step + 1) % log_every_batches == 0:
            wall = time.time() - started_at
            print(
                "[pretrain] epoch=%d step=%d/%d loss=%.4f pixel_acc=%.3f elapsed=%.1fs"
                % (
                    epoch,
                    step + 1,
                    len(loader),
                    float(loss.item()),
                    total_correct / max(1, total_pixels),
                    wall,
                ),
                flush=True,
            )

    avg_loss = total_loss_weighted / max(1, total_samples)
    pixel_acc = total_correct / max(1, total_pixels)
    return {
        "loss": float(avg_loss),
        "pixel_acc": float(pixel_acc),
        "batches": float(seen_batches),
        "samples": float(total_samples),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output).resolve()
    ensure_dir(output_path.parent)

    overrides: Dict[str, Any] = {}
    for key in ("epochs", "batch_size", "lr", "weight_decay"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    config = merge_config(args.hardware_profile, overrides)

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[pretrain] device=%s profile=%s" % (device.type, args.hardware_profile), flush=True)
    print("[pretrain] config=%s" % {k: v for k, v in config.items() if k in {"batch_size", "model_dim", "num_slots", "history", "epochs", "lr", "decoder_dim"}}, flush=True)

    data_paths = parse_data_paths(args.data)
    print("[pretrain] loading episodes from %d file(s)" % len(data_paths), flush=True)
    dataset = FrameReconstructionDataset(
        episode_paths=data_paths,
        history=int(config["history"]),
        max_episodes=args.max_episodes,
    )
    print(
        "[pretrain] loaded episodes=%d frames=%d" % (dataset.episode_count, len(dataset)),
        flush=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("No frames loaded — check --data paths.")

    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print("[pretrain] batches_per_epoch=%d" % len(loader), flush=True)

    bundle = build_pretrain_bundle(config)
    encoder = bundle.encoder.to(device)
    decoder = bundle.decoder.to(device)

    encoder_param_count = sum(p.numel() for p in encoder.parameters())
    decoder_param_count = sum(p.numel() for p in decoder.parameters())
    print(
        "[pretrain] encoder_params=%.2fM decoder_params=%.2fM"
        % (encoder_param_count / 1e6, decoder_param_count / 1e6),
        flush=True,
    )

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    metrics_path = output_path.parent / "pretrain_metrics.csv"
    summary_path = output_path.parent / "pretrain_summary.json"

    epochs = int(config["epochs"])
    best_loss = float("inf")
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        metrics = run_epoch(
            encoder=encoder,
            decoder=decoder,
            loader=loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_every_batches=int(args.log_every_batches),
        )
        epoch_wall = time.time() - epoch_start
        row = {"epoch": epoch, "wall_seconds": round(epoch_wall, 2), **metrics}
        append_metrics_row(metrics_path, row)
        history.append(row)
        print(
            "[pretrain] epoch=%d done loss=%.4f pixel_acc=%.3f wall=%.1fs"
            % (epoch, metrics["loss"], metrics["pixel_acc"], epoch_wall),
            flush=True,
        )
        if epoch % max(1, int(args.save_every_epochs)) == 0 or epoch == epochs:
            save_pretrained_encoder(
                path=str(output_path),
                encoder=encoder,
                config=config,
                epoch=epoch,
                metrics=metrics,
            )
            print("[pretrain] saved encoder -> %s" % output_path, flush=True)
        if metrics["loss"] < best_loss:
            best_loss = float(metrics["loss"])

    save_json(
        summary_path,
        {
            "profile": args.hardware_profile,
            "config": config,
            "data_paths": [str(path) for path in data_paths],
            "episodes_loaded": dataset.episode_count,
            "frames_loaded": len(dataset),
            "encoder_params": encoder_param_count,
            "decoder_params": decoder_param_count,
            "best_loss": best_loss,
            "history": history,
        },
    )
    print("[pretrain] summary -> %s" % summary_path, flush=True)


if __name__ == "__main__":
    main()
