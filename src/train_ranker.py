from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import append_metrics_row, ensure_dir, save_json, seed_everything, split_games
from .ranker_features import FEATURE_NAMES


@dataclass
class RankerExample:
    game_id: str
    features: torch.Tensor
    selected_index: int
    selected_target: float


class CandidateRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class RankerDataset(Dataset):
    def __init__(self, examples: Sequence[RankerExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RankerExample:
        return self.examples[index]


def collate(batch: Sequence[RankerExample]) -> List[RankerExample]:
    return list(batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a guarded candidate ranker from source/search trajectory logs."
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-mode", type=str, default="game", choices=["game", "episode"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--min-target", type=float, default=0.05)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    return parser.parse_args()


def iter_jsonl_gz(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def example_key(payload: Dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(payload.get("episode_id", "")),
            str(payload.get("game_id", "")),
            str(payload.get("seed", "")),
            str(payload.get("step_index", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_examples(path: Path, min_target: float, max_candidates: int) -> List[RankerExample]:
    examples: List[RankerExample] = []
    for payload in iter_jsonl_gz(path):
        selected_target = float(payload.get("selected_target", 0.0))
        if selected_target < min_target:
            continue
        candidates = list(payload.get("candidates", []))[: max(2, int(max_candidates))]
        selected_index = int(payload.get("selected_index", -1))
        if selected_index < 0 or selected_index >= len(candidates):
            continue
        features = torch.tensor([candidate["features"] for candidate in candidates], dtype=torch.float32)
        if features.ndim != 2 or features.shape[0] < 2:
            continue
        examples.append(
            RankerExample(
                game_id=str(payload.get("game_id", "")),
                features=features,
                selected_index=selected_index,
                selected_target=selected_target,
            )
        )
    return examples


def split_examples(examples: Sequence[RankerExample], split_mode: str) -> Tuple[List[RankerExample], List[RankerExample], List[str], List[str]]:
    if split_mode == "game":
        train_games, val_games = split_games([example.game_id for example in examples])
        train_set = set(train_games)
        val_set = set(val_games)
        train = [example for example in examples if example.game_id in train_set]
        val = [example for example in examples if example.game_id in val_set]
        return train, val, train_games, val_games

    keyed = []
    for index, example in enumerate(examples):
        digest = hashlib.sha1(("ranker-episode-split-%d" % index).encode("utf-8")).hexdigest()
        keyed.append((digest, example))
    keyed.sort(key=lambda item: item[0])
    holdout = max(1, int(round(len(keyed) * 0.2)))
    val = [example for _, example in keyed[:holdout]]
    train = [example for _, example in keyed[holdout:]]
    games = sorted(set(example.game_id for example in examples))
    return train, val, games, games


def feature_stats(examples: Sequence[RankerExample]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise RuntimeError("No ranker examples are available.")
    stacked = torch.cat([example.features for example in examples], dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0).clamp_min(1e-4)
    return mean, std


def normalize_example(example: RankerExample, mean: torch.Tensor, std: torch.Tensor) -> RankerExample:
    return RankerExample(
        game_id=example.game_id,
        features=(example.features - mean) / std,
        selected_index=example.selected_index,
        selected_target=example.selected_target,
    )


def train_epoch(
    model: CandidateRanker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    confidence_threshold: float,
    margin_threshold: float,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_groups = 0
    top1 = 0.0
    mrr = 0.0
    confident = 0.0
    confident_correct = 0.0

    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        batch_loss = torch.tensor(0.0, device=device)
        batch_groups = 0
        for example in batch:
            features = example.features.to(device)
            logits = model(features).unsqueeze(0)
            target = torch.tensor([int(example.selected_index)], dtype=torch.long, device=device)
            weight = min(4.0, 1.0 + max(0.0, float(example.selected_target)) * 0.25)
            loss = F.cross_entropy(logits, target) * float(weight)
            batch_loss = batch_loss + loss
            batch_groups += 1

            with torch.no_grad():
                scores = logits.squeeze(0)
                order = torch.argsort(scores, descending=True)
                rank_tensor = (order == int(example.selected_index)).nonzero(as_tuple=False)
                rank = int(rank_tensor[0].item()) + 1 if rank_tensor.numel() else len(order)
                top1 += float(rank == 1)
                mrr += 1.0 / float(rank)
                probs = F.softmax(scores, dim=0)
                sorted_probs = torch.sort(probs, descending=True).values
                confidence = float(sorted_probs[0].item())
                margin = float((sorted_probs[0] - sorted_probs[1]).item()) if len(sorted_probs) > 1 else confidence
                is_confident = confidence >= float(confidence_threshold) and margin >= float(margin_threshold)
                confident += float(is_confident)
                confident_correct += float(is_confident and rank == 1)

        if batch_groups == 0:
            continue
        batch_loss = batch_loss / float(batch_groups)
        if optimizer is not None:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(batch_loss.item()) * batch_groups
        total_groups += batch_groups

    denom = max(1, total_groups)
    return {
        "loss": total_loss / float(denom),
        "top1": top1 / float(denom),
        "mrr": mrr / float(denom),
        "confident_rate": confident / float(denom),
        "confident_top1": confident_correct / float(max(confident, 1.0)),
    }


def save_ranker_checkpoint(
    path: Path,
    model: CandidateRanker,
    config: Dict[str, Any],
    mean: torch.Tensor,
    std: torch.Tensor,
    epoch: int,
    metric: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "feature_names": FEATURE_NAMES,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "epoch": int(epoch),
            "metric": float(metric),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)

    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_path = output_dir / "metrics.csv"

    raw_examples = load_examples(
        path=data_path,
        min_target=float(args.min_target),
        max_candidates=int(args.max_candidates),
    )
    if not raw_examples:
        raise RuntimeError("No ranker examples loaded from %s" % data_path)
    train_raw, val_raw, train_games, val_games = split_examples(raw_examples, split_mode=str(args.split_mode))
    if not train_raw:
        raise RuntimeError("No training examples after split.")
    if not val_raw:
        val_raw = train_raw[: max(1, min(128, len(train_raw)))]

    mean, std = feature_stats(train_raw)
    train_examples = [normalize_example(example, mean, std) for example in train_raw]
    val_examples = [normalize_example(example, mean, std) for example in val_raw]

    config = {
        "data": str(data_path),
        "split_mode": str(args.split_mode),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "min_target": float(args.min_target),
        "max_candidates": int(args.max_candidates),
        "confidence_threshold": float(args.confidence_threshold),
        "margin_threshold": float(args.margin_threshold),
        "train_games": train_games,
        "val_games": val_games,
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
    }
    save_json(output_dir / "train_ranker_config.json", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateRanker(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader = DataLoader(
        RankerDataset(train_examples),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        RankerDataset(val_examples),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate,
    )

    print(
        "Ranker training setup: device=%s train_examples=%d val_examples=%d train_games=%s val_games=%s"
        % (device, len(train_examples), len(val_examples), ",".join(train_games), ",".join(val_games)),
        flush=True,
    )

    best_metric = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            confidence_threshold=float(args.confidence_threshold),
            margin_threshold=float(args.margin_threshold),
        )
        with torch.no_grad():
            val_metrics = train_epoch(
                model,
                val_loader,
                None,
                device,
                confidence_threshold=float(args.confidence_threshold),
                margin_threshold=float(args.margin_threshold),
            )
        metric = float(val_metrics["mrr"])
        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_top1": round(train_metrics["top1"], 6),
            "train_mrr": round(train_metrics["mrr"], 6),
            "train_confident_rate": round(train_metrics["confident_rate"], 6),
            "train_confident_top1": round(train_metrics["confident_top1"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_top1": round(val_metrics["top1"], 6),
            "val_mrr": round(val_metrics["mrr"], 6),
            "val_confident_rate": round(val_metrics["confident_rate"], 6),
            "val_confident_top1": round(val_metrics["confident_top1"], 6),
        }
        append_metrics_row(metrics_path, row)
        print(
            "[ranker] epoch=%d train_top1=%.4f train_mrr=%.4f val_top1=%.4f val_mrr=%.4f val_confident_rate=%.4f val_confident_top1=%.4f"
            % (
                epoch,
                train_metrics["top1"],
                train_metrics["mrr"],
                val_metrics["top1"],
                val_metrics["mrr"],
                val_metrics["confident_rate"],
                val_metrics["confident_top1"],
            ),
            flush=True,
        )
        save_ranker_checkpoint(
            checkpoints_dir / "last_ranker.pth",
            model=model,
            config=config,
            mean=mean,
            std=std,
            epoch=epoch,
            metric=metric,
        )
        if metric > best_metric:
            best_metric = metric
            save_ranker_checkpoint(
                checkpoints_dir / "best_ranker.pth",
                model=model,
                config=config,
                mean=mean,
                std=std,
                epoch=epoch,
                metric=metric,
            )
            print("[ranker] best epoch=%d val_mrr=%.6f" % (epoch, best_metric), flush=True)

    save_json(
        output_dir / "summary.json",
        {
            "best_metric": best_metric,
            "best_checkpoint": str(checkpoints_dir / "best_ranker.pth"),
            "last_checkpoint": str(checkpoints_dir / "last_ranker.pth"),
            "metrics": str(metrics_path),
            "config": config,
        },
    )


if __name__ == "__main__":
    main()
