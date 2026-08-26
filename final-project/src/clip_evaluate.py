"""Evaluate pretrained or fine-tuned CLIP on driving action/reason labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available")
    return torch.device(requested)


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


def predict(
    image: Image.Image,
    candidates: list[str],
    ground_truth: str,
    model: CLIPModel,
    processor: CLIPProcessor,
) -> tuple[str, float]:
    texts = list(candidates)
    if ground_truth not in texts:
        texts.append(ground_truth)
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image[0]
    prediction = candidates[int(torch.argmax(logits[: len(candidates)]).item())]
    ground_truth_index = texts.index(ground_truth)
    ground_truth_score = float(torch.sigmoid(logits[ground_truth_index]).item())
    return prediction, ground_truth_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.annotations.is_file():
        parser.error(f"--annotations is not a file: {args.annotations}")
    if not args.image_root.is_dir():
        parser.error(f"--image-root is not a directory: {args.image_root}")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"--checkpoint is not a file: {args.checkpoint}")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        records = json.loads(args.annotations.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid annotation JSON: {args.annotations}") from exc
    if not isinstance(records, list) or not records:
        raise ValueError("annotation JSON must be a non-empty list")
    if args.limit:
        records = records[: args.limit]

    parsed: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} is not an object")
        file_name = record.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"record {index} has no file_name")
        action = single_label(record.get("action"), "action", index)
        reason = single_label(record.get("reason"), "reason", index)
        parsed.append(
            {
                "file_name": file_name,
                "action": action,
                "reason": reason,
                "action_reason": f"{action}, because {reason}",
            }
        )

    action_candidates = sorted({record["action"] for record in parsed})
    reason_candidates = sorted({record["reason"] for record in parsed})
    pair_candidates = sorted({record["action_reason"] for record in parsed})
    if not action_candidates or not reason_candidates or not pair_candidates:
        raise ValueError("candidate label sets cannot be empty")

    device = choose_device(args.device)
    model = CLIPModel.from_pretrained(args.model)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
    model = model.to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    totals = {
        "actions": {"matches": 0, "score_sum": 0.0},
        "reasons": {"matches": 0, "score_sum": 0.0},
        "action_reasons": {"matches": 0, "score_sum": 0.0},
    }
    for record in tqdm(parsed, desc="CLIP evaluation"):
        image_path = args.image_root / record["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"evaluation image does not exist: {image_path}")
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            checks = (
                (
                    "actions",
                    action_candidates,
                    record["action"],
                ),
                (
                    "reasons",
                    reason_candidates,
                    record["reason"],
                ),
                (
                    "action_reasons",
                    pair_candidates,
                    record["action_reason"],
                ),
            )
            for key, candidates, ground_truth in checks:
                prediction, score = predict(
                    image, candidates, ground_truth, model, processor
                )
                totals[key]["matches"] += int(prediction == ground_truth)
                totals[key]["score_sum"] += score

    count = len(parsed)
    result = {
        key: {
            "match_fraction": value["matches"] / count,
            "mean_ground_truth_sigmoid_similarity": value["score_sum"] / count,
        }
        for key, value in totals.items()
    }
    result["examples"] = count
    result["model"] = args.model
    result["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
