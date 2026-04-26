from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from arc_agi import Arcade, OperationMode
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .agent import PolicyGuidedAgent
from .common import (
    ACTION_IDS,
    ACTION_TO_INDEX,
    action_mask,
    append_metrics_row,
    compute_discounted_returns,
    episode_level_actions,
    iter_jsonl_gz,
    load_metadata_map,
    merge_config,
    one_hot_frames,
    pad_history,
    rhae_score,
    safe_mean,
    save_json,
    scalar_features,
    seed_everything,
    split_games,
    transition_reward,
)
from .model import build_model, load_checkpoint, save_checkpoint


class EpisodeTransitionDataset(Dataset):
    def __init__(self, episodes: Sequence[Dict[str, Any]], history: int, max_steps: int) -> None:
        self.samples: List[Dict[str, Any]] = []
        self.history = history
        self.max_steps = max_steps

        for episode in episodes:
            transitions = list(episode.get("transitions", []))
            if not transitions:
                continue
            rewards = [transition_reward(transition) for transition in transitions]
            returns = compute_discounted_returns(rewards)

            last_action_id: Optional[int] = None
            steps_since_progress = 0
            frames_so_far: List[List[List[int]]] = []
            for idx, transition in enumerate(transitions):
                frame = transition["frame"]
                next_frame = transition["next_frame"]
                history_frames = pad_history(frames_so_far + [frame], history=self.history)
                next_history_frames = pad_history(frames_so_far + [frame, next_frame], history=self.history)
                progress = int(transition["levels_after"]) - int(transition["levels_before"])
                next_transition = transitions[idx + 1] if idx + 1 < len(transitions) else transition
                next_steps_since_progress = 0 if progress > 0 else steps_since_progress + 1
                sample = {
                    "obs": one_hot_frames(history_frames),
                    "next_obs": one_hot_frames(next_history_frames),
                    "scalar": scalar_features(
                        available_actions=transition["available_actions"],
                        last_action_id=last_action_id,
                        levels_completed=int(transition["levels_before"]),
                        steps_since_progress=steps_since_progress,
                        step_index=idx,
                        frame=history_frames[-1],
                        max_steps=self.max_steps,
                    ),
                    "next_scalar": scalar_features(
                        available_actions=next_transition["available_actions"],
                        last_action_id=int(transition["action_id"]),
                        levels_completed=int(transition["levels_after"]),
                        steps_since_progress=next_steps_since_progress,
                        step_index=idx + 1,
                        frame=next_history_frames[-1],
                        max_steps=self.max_steps,
                    ),
                    "available_mask": torch.tensor(action_mask(transition["available_actions"]), dtype=torch.float32),
                    "action_index": torch.tensor(ACTION_TO_INDEX[int(transition["action_id"])], dtype=torch.long),
                    "x": torch.tensor(int((transition.get("action_data") or {}).get("x", 0)), dtype=torch.long),
                    "y": torch.tensor(int((transition.get("action_data") or {}).get("y", 0)), dtype=torch.long),
                    "coord_mask": torch.tensor(1.0 if int(transition["action_id"]) == 6 else 0.0, dtype=torch.float32),
                    "return_target": torch.tensor(float(returns[idx]), dtype=torch.float32),
                    "weight": torch.tensor(
                        1.0
                        + (float(episode.get("score", 0.0)) / 100.0)
                        + max(0, progress) * 2.0,
                        dtype=torch.float32,
                    ),
                }
                self.samples.append(sample)
                last_action_id = int(transition["action_id"])
                frames_so_far.append(frame)
                if progress > 0:
                    steps_since_progress = 0
                else:
                    steps_since_progress += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]


def collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal ARC-AGI-3 object-centric policy.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to one collected episodes.jsonl.gz file, or multiple comma separated paths.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--hardware-profile", type=str, default="a100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--history", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--num-slots", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--stall-steps", type=int, default=24)
    parser.add_argument("--reset-limit", type=int, default=4)
    parser.add_argument("--online-val-every", type=int, default=None)
    parser.add_argument("--online-val-games", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def parse_data_paths(raw: str) -> List[Path]:
    paths = [Path(part.strip()).expanduser().resolve() for part in raw.split(",") if part.strip()]
    if not paths:
        raise RuntimeError("No valid --data paths were provided.")
    return paths


def load_episodes(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl_gz(path))


def load_episodes_from_paths(paths: Sequence[Path | str]) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("Collected data not found: %s" % path)
        episodes.extend(load_episodes(path))
    return episodes


def split_episodes(episodes: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
    game_ids = [str(episode["game_id"]) for episode in episodes]
    train_games, val_games = split_games(game_ids)
    train_episodes = [episode for episode in episodes if episode["game_id"] in train_games]
    val_episodes = [episode for episode in episodes if episode["game_id"] in val_games]
    return train_episodes, val_episodes, train_games, val_games


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def masked_average(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def save_training_checkpoint(
    checkpoints_dir: Path,
    stem: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    config: Dict[str, Any],
    epoch: int,
    best_score: float,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Path:
    primary_path = checkpoints_dir / ("%s.pth" % stem)
    legacy_path = checkpoints_dir / ("%s.pt" % stem)
    save_checkpoint(
        path=str(primary_path),
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=epoch,
        best_score=best_score,
        extra_state=extra_state,
    )
    shutil.copyfile(primary_path, legacy_path)
    return primary_path


def restore_training_state(
    resume_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    payload = load_checkpoint(str(resume_path), device=device)
    model.load_state_dict(payload["model_state"])
    optimizer_state = payload.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    last_epoch = int(payload.get("epoch", 0))
    epoch_complete = bool(payload.get("epoch_complete", True))
    best_score = float(payload.get("best_score", float("-inf")))
    best_epoch = int(payload.get("best_epoch", last_epoch if epoch_complete else max(last_epoch - 1, -1)))
    start_epoch = last_epoch + 1 if epoch_complete else max(1, last_epoch)
    return {
        "start_epoch": start_epoch,
        "best_metric": best_score,
        "best_epoch": best_epoch,
        "epoch_complete": epoch_complete,
        "payload": payload,
    }


def evaluate_public_score(
    checkpoint_path: str,
    project_root: Path,
    metadata_map: Dict[str, Dict[str, Any]],
    selected_games: Sequence[str],
    max_steps: int,
    stall_steps: int,
    reset_limit: int,
) -> float:
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(project_root / "tmp_eval_recordings"),
    )
    agent = PolicyGuidedAgent(
        checkpoint_path=checkpoint_path,
        max_steps=max_steps,
        stall_steps=stall_steps,
        reset_limit=reset_limit,
    )
    scores: List[float] = []
    for game_id in selected_games:
        env = arc.make(game_id)
        if env is None:
            continue
        baseline_actions = metadata_map[game_id].get("baseline_actions", [])
        result = agent.play_env(env=env, game_id=game_id, baseline_actions=baseline_actions)
        score_info = rhae_score(
            baseline_actions=baseline_actions,
            completed_level_actions=episode_level_actions(result["transitions"]),
        )
        scores.append(float(score_info["score"]))
    return safe_mean(scores)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_accum: int,
    max_grad_norm: float = 0.0,
    epoch: int = 0,
    checkpoint_every_steps: int = 0,
    checkpoint_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_action_correct = 0.0
    total_action_count = 0.0
    total_coord_correct = 0.0
    total_coord_count = 0.0
    optimizer_steps = 0

    autocast_enabled = device.type == "cuda"
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(
                obs=batch["obs"],
                scalar=batch["scalar"],
                action_index=batch["action_index"],
            )
            with torch.no_grad():
                next_latent = model.encode_state(batch["next_obs"], batch["next_scalar"])["pooled"]

            action_loss_per = F.cross_entropy(
                out["action_logits"],
                batch["action_index"],
                reduction="none",
            )
            coord_loss_x = F.cross_entropy(out["x_logits"], batch["x"], reduction="none")
            coord_loss_y = F.cross_entropy(out["y_logits"], batch["y"], reduction="none")
            coord_loss_per = (coord_loss_x + coord_loss_y) * batch["coord_mask"]
            value_loss_per = F.smooth_l1_loss(out["value"], batch["return_target"], reduction="none")
            avail_loss_per = F.binary_cross_entropy_with_logits(
                out["avail_logits"],
                batch["available_mask"],
                reduction="none",
            ).mean(dim=-1)
            latent_loss_per = F.mse_loss(
                out["pred_next_latent"],
                next_latent.detach(),
                reduction="none",
            ).mean(dim=-1)

            weights = batch["weight"]
            loss = (
                masked_average(action_loss_per, weights)
                + 0.5 * masked_average(coord_loss_per, weights)
                + 0.25 * masked_average(value_loss_per, weights)
                + 0.2 * masked_average(avail_loss_per, weights)
                + 0.1 * masked_average(latent_loss_per, weights)
            )

        if optimizer is not None:
            (loss / float(max(1, grad_accum))).backward()
            if (step + 1) % max(1, grad_accum) == 0 or (step + 1) == len(loader):
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if checkpoint_callback is not None and checkpoint_every_steps > 0:
                    if optimizer_steps % checkpoint_every_steps == 0:
                        checkpoint_callback(
                            {
                                "epoch": epoch,
                                "epoch_complete": False,
                                "batch_in_epoch": step + 1,
                                "optimizer_steps_in_epoch": optimizer_steps,
                                "last_loss": float(loss.item()),
                            }
                        )

        total_loss += float(loss.item()) * batch["obs"].shape[0]
        action_pred = out["action_logits"].argmax(dim=-1)
        total_action_correct += float((action_pred == batch["action_index"]).sum().item())
        total_action_count += float(batch["action_index"].numel())

        coord_mask = batch["coord_mask"] > 0
        if coord_mask.any():
            x_correct = out["x_logits"].argmax(dim=-1)[coord_mask] == batch["x"][coord_mask]
            y_correct = out["y_logits"].argmax(dim=-1)[coord_mask] == batch["y"][coord_mask]
            total_coord_correct += float((x_correct & y_correct).sum().item())
            total_coord_count += float(coord_mask.sum().item())

    denom = max(len(loader.dataset), 1)
    return {
        "loss": total_loss / denom,
        "action_acc": total_action_correct / max(total_action_count, 1.0),
        "coord_acc": total_coord_correct / max(total_coord_count, 1.0),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "history": args.history,
        "model_dim": args.model_dim,
        "num_slots": args.num_slots,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "online_val_every": args.online_val_every,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "max_grad_norm": args.max_grad_norm,
        "max_steps": args.max_steps,
        "stall_steps": args.stall_steps,
        "reset_limit": args.reset_limit,
    }
    config = merge_config(args.hardware_profile, overrides)
    config["hardware_profile"] = args.hardware_profile
    config["seed"] = args.seed
    data_paths = parse_data_paths(args.data)
    config["data"] = [str(path) for path in data_paths]
    save_json(output_dir / "train_config.json", config)

    episodes = load_episodes_from_paths(data_paths)
    train_episodes, val_episodes, train_games, val_games = split_episodes(episodes)
    if not train_episodes:
        raise RuntimeError("No training episodes were found after the game-level split.")

    train_dataset = EpisodeTransitionDataset(
        episodes=train_episodes,
        history=int(config["history"]),
        max_steps=int(config["max_steps"]),
    )
    val_dataset = EpisodeTransitionDataset(
        episodes=val_episodes if val_episodes else train_episodes[:1],
        history=int(config["history"]),
        max_steps=int(config["max_steps"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    metrics_path = output_dir / "metrics.csv"
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metadata_map = load_metadata_map(project_root / "environment_files")

    best_metric = float("-inf")
    best_epoch = -1
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError("Resume checkpoint not found: %s" % resume_path)
        resume_state = restore_training_state(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_epoch = int(resume_state["start_epoch"])
        best_metric = float(resume_state["best_metric"])
        best_epoch = int(resume_state["best_epoch"])
        print(
            "Resumed from %s at epoch %d (%s checkpoint)"
            % (
                resume_path,
                start_epoch,
                "complete" if resume_state["epoch_complete"] else "partial",
            )
        )

    current_epoch = 0

    def checkpoint_callback(extra_state: Dict[str, Any]) -> None:
        save_training_checkpoint(
            checkpoints_dir=checkpoints_dir,
            stem="last",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=int(extra_state.get("epoch", current_epoch)),
            best_score=best_metric if best_metric != float("-inf") else 0.0,
            extra_state={
                "best_epoch": best_epoch,
                **extra_state,
            },
        )

    try:
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            current_epoch = epoch
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_accum=int(config["grad_accum"]),
                max_grad_norm=float(config.get("max_grad_norm", 0.0)),
                epoch=epoch,
                checkpoint_every_steps=int(config.get("checkpoint_every_steps", 0)),
                checkpoint_callback=checkpoint_callback,
            )
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                grad_accum=1,
            )

            last_checkpoint = save_training_checkpoint(
                checkpoints_dir=checkpoints_dir,
                stem="last",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                best_score=best_metric if best_metric != float("-inf") else 0.0,
                extra_state={
                    "epoch_complete": True,
                    "best_epoch": best_epoch,
                },
            )

            public_val_score = None
            if val_games and epoch % int(config["online_val_every"]) == 0:
                selected_games = list(val_games)[: max(1, int(args.online_val_games))]
                public_val_score = evaluate_public_score(
                    checkpoint_path=str(last_checkpoint),
                    project_root=project_root,
                    metadata_map=metadata_map,
                    selected_games=selected_games,
                    max_steps=int(config["max_steps"]),
                    stall_steps=int(config["stall_steps"]),
                    reset_limit=int(config["reset_limit"]),
                )

            row = {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_action_acc": round(train_metrics["action_acc"], 6),
                "train_coord_acc": round(train_metrics["coord_acc"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_action_acc": round(val_metrics["action_acc"], 6),
                "val_coord_acc": round(val_metrics["coord_acc"], 6),
                "public_val_score": None if public_val_score is None else round(public_val_score, 6),
            }
            append_metrics_row(metrics_path, row)

            selection_metric = public_val_score if public_val_score is not None else val_metrics["action_acc"]
            if selection_metric > best_metric:
                best_metric = float(selection_metric)
                best_epoch = epoch
                save_training_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    stem="best",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    best_score=best_metric,
                    extra_state={
                        "epoch_complete": True,
                        "best_epoch": best_epoch,
                        "selection_metric": best_metric,
                    },
                )
    except KeyboardInterrupt:
        interrupt_checkpoint = save_training_checkpoint(
            checkpoints_dir=checkpoints_dir,
            stem="interrupt",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=current_epoch,
            best_score=best_metric if best_metric != float("-inf") else 0.0,
            extra_state={
                "epoch_complete": False,
                "best_epoch": best_epoch,
                "status": "interrupted",
            },
        )
        save_training_checkpoint(
            checkpoints_dir=checkpoints_dir,
            stem="last",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=current_epoch,
            best_score=best_metric if best_metric != float("-inf") else 0.0,
            extra_state={
                "epoch_complete": False,
                "best_epoch": best_epoch,
                "status": "interrupted",
            },
        )
        summary = {
            "status": "interrupted",
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "resume_checkpoint": str(interrupt_checkpoint),
            "data_paths": [str(path) for path in data_paths],
            "train_games": train_games,
            "val_games": val_games,
            "num_train_episodes": len(train_episodes),
            "num_val_episodes": len(val_episodes),
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
        }
        save_json(output_dir / "summary.json", summary)
        print("Training interrupted. Checkpoint saved to %s" % interrupt_checkpoint)
        return

    summary = {
        "status": "finished",
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_checkpoint": str(checkpoints_dir / "best.pth"),
        "data_paths": [str(path) for path in data_paths],
        "train_games": train_games,
        "val_games": val_games,
        "num_train_episodes": len(train_episodes),
        "num_val_episodes": len(val_episodes),
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
    }
    save_json(output_dir / "summary.json", summary)
    print("Training finished. Best epoch: %d" % best_epoch)


if __name__ == "__main__":
    main()
