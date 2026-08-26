"""Fine-tune CLIP as a binary image/reason reward model."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
)
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

MAX_SEED = 2**32 - 1


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_value(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"value must be between 0 and {MAX_SEED}")
    return parsed


def single_label(value: Any, field: str, index: int) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], str)
        and value[0].strip()
    ):
        return value[0].strip()
    raise ValueError(f"record {index} must contain exactly one non-empty {field} label")


def load_pairs(annotations: Path, seed: int) -> list[dict[str, Any]]:
    try:
        records = json.loads(annotations.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"annotation file does not exist: {annotations}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"annotation file is not valid JSON: {annotations}") from exc

    if not isinstance(records, list) or not records:
        raise ValueError("annotation JSON must be a non-empty list")

    cleaned: list[tuple[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} is not an object")
        file_name = record.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError(f"record {index} has no usable file_name")
        reason = single_label(record.get("reason"), "reason", index)
        cleaned.append((file_name.strip(), reason))

    unique_reasons = sorted({reason for _, reason in cleaned})
    if len(unique_reasons) < 2:
        raise ValueError(
            "at least two distinct reasons are required for negative pairs"
        )

    generator = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    for file_name, reason in cleaned:
        pairs.append({"file_name": file_name, "reason": reason, "label": 1})
        alternatives = [
            candidate for candidate in unique_reasons if candidate != reason
        ]
        pairs.append(
            {
                "file_name": file_name,
                "reason": generator.choice(alternatives),
                "label": 0,
            }
        )
    generator.shuffle(pairs)
    return pairs


class SimilarityDataset(Dataset):
    def __init__(
        self,
        pairs: list[dict[str, Any]],
        image_root: Path,
        tokenizer: CLIPTokenizerFast,
        image_transform: transforms.Compose,
        max_length: int,
    ) -> None:
        self.pairs = pairs
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.pairs[index]
        image_path = self.image_root / str(item["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"dataset image does not exist: {image_path}")
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            pixels = self.image_transform(image)
        tokens = self.tokenizer(
            str(item["reason"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixels,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.float32),
        }


def similarity_logits(model: CLIPModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    image_features = model.get_image_features(pixel_values=batch["pixel_values"])
    text_features = model.get_text_features(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    return (image_features * text_features).sum(dim=1) * model.logit_scale.exp()


def evaluate(
    model: CLIPModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = {
                key: value.to(device)
                for key, value in raw_batch.items()
                if key != "labels"
            }
            labels.append(raw_batch["labels"].to(device))
            logits.append(similarity_logits(model, batch))
    if not labels:
        raise ValueError("validation loader produced no batches")

    all_labels = torch.cat(labels).int()
    all_logits = torch.cat(logits)
    probabilities = torch.sigmoid(all_logits)
    return {
        "accuracy": float(
            BinaryAccuracy(threshold=threshold).to(device)(probabilities, all_labels)
        ),
        "f1": float(
            BinaryF1Score(threshold=threshold).to(device)(probabilities, all_labels)
        ),
        "recall": float(
            BinaryRecall(threshold=threshold).to(device)(probabilities, all_labels)
        ),
        "auroc": float(BinaryAUROC().to(device)(probabilities, all_labels)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=seed_value, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    args = parser.parse_args()

    if not args.image_root.is_dir():
        parser.error(f"--image-root is not a directory: {args.image_root}")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
    ):
        parser.error(
            "learning rate must be finite and positive; weight decay must be "
            "finite and nonnegative"
        )
    if (
        not math.isfinite(args.validation_fraction)
        or not 0 < args.validation_fraction < 1
    ):
        parser.error("--validation-fraction must be finite and between 0 and 1")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
        parser.error("--threshold must be finite and between 0 and 1")
    if args.early_stopping_patience <= 0 or args.scheduler_patience < 0:
        parser.error("patience values are invalid")
    if not math.isfinite(args.gradient_clip) or args.gradient_clip <= 0:
        parser.error("--gradient-clip must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.annotations, args.seed)
    labels = [int(item["label"]) for item in pairs]
    train_pairs, validation_pairs = train_test_split(
        pairs,
        test_size=args.validation_fraction,
        random_state=args.seed,
        stratify=labels,
    )

    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    tokenizer = CLIPTokenizerFast.from_pretrained(args.model)
    crop_size = processor.image_processor.crop_size
    target_size = crop_size["height"] if isinstance(crop_size, dict) else crop_size
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                target_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(
                target_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )
    train_loader = DataLoader(
        SimilarityDataset(
            train_pairs,
            args.image_root,
            tokenizer,
            train_transform,
            args.max_length,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    validation_loader = DataLoader(
        SimilarityDataset(
            validation_pairs,
            args.image_root,
            tokenizer,
            validation_transform,
            args.max_length,
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    if len(train_loader) == 0:
        raise ValueError("training split is smaller than one full batch")

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.2,
        patience=args.scheduler_patience,
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    history: list[dict[str, float | int]] = []
    best_auroc = -1.0
    stale_epochs = 0
    checkpoint = args.output_dir / "best_clip_reward_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for raw_batch in tqdm(train_loader, desc=f"CLIP epoch {epoch}", leave=False):
            batch = {
                key: value.to(device)
                for key, value in raw_batch.items()
                if key != "labels"
            }
            labels_tensor = raw_batch["labels"].to(device)
            optimizer.zero_grad()
            logits = similarity_logits(model, batch)
            loss = loss_function(logits, labels_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total_loss += float(loss.item())

        metrics = evaluate(model, validation_loader, device, args.threshold)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / len(train_loader)
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(metrics)
        scheduler.step(metrics["auroc"])
        print(json.dumps(metrics, sort_keys=True))

        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                break

    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
