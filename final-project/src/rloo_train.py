"""RLOO fine-tuning for scene-description generation with a CLIP reward."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizerFast,
    Idefics3ForConditionalGeneration,
)

MAX_SEED = 2**32 - 1


class RLOODataset(Dataset):
    """Minimal JSON dataset containing at least a ``file_name`` per record."""

    def __init__(self, path: Path) -> None:
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"annotation file does not exist: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"annotation file is not valid JSON: {path}") from exc

        if not isinstance(records, list) or not records:
            raise ValueError("annotation JSON must be a non-empty list")
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not record.get("file_name"):
                raise ValueError(f"record {index} must contain a non-empty file_name")
        self.records: list[dict[str, Any]] = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def clip_reward(
    image: Image.Image,
    text: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    tokenizer: CLIPTokenizerFast,
) -> float:
    """Return the submitted sigmoid-scaled CLIP similarity reward."""

    text_inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    text_ids = text_inputs["input_ids"].to(model.device)
    text_mask = text_inputs["attention_mask"].to(model.device)
    pixels = processor(images=image, return_tensors="pt")["pixel_values"].to(
        model.device
    )

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=pixels)
        text_features = model.get_text_features(
            input_ids=text_ids, attention_mask=text_mask
        )
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        similarity = (image_features * text_features).sum(dim=-1)
        similarity = similarity * model.logit_scale.exp()
    return torch.sigmoid(similarity).item()


def response_text(decoded: str) -> str:
    """Strip the chat template prefix from a decoded completion."""

    return decoded.rsplit("Assistant:", maxsplit=1)[-1].strip()


def generate_completions(
    processor: AutoProcessor,
    model: Idefics3ForConditionalGeneration,
    image: Image.Image,
    samples: int,
    max_new_tokens: int,
) -> list[str]:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=1.5,
            max_new_tokens=max_new_tokens,
            num_return_sequences=samples,
            return_dict_in_generate=True,
        )
    return processor.batch_decode(output.sequences, skip_special_tokens=True)


def sequence_log_probability(
    model: Idefics3ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    completion: str,
) -> torch.Tensor:
    """Compute the completion log probability used by the original RLOO code."""

    encoded = tokenizer(
        completion,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    output = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )
    logits = output.logits[..., :-1, :].contiguous()
    labels = encoded["input_ids"][..., 1:].contiguous()
    token_log_probs = F.log_softmax(logits, dim=-1)
    chosen = torch.gather(token_log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    mask = encoded["attention_mask"][..., 1:].to(chosen.dtype)
    return (chosen * mask).sum(dim=1)


def rloo_step(
    model: Idefics3ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    clip_tokenizer: CLIPTokenizerFast,
    image: Image.Image,
    optimizer: torch.optim.Optimizer,
    samples: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    if samples < 2:
        raise ValueError("RLOO requires at least two samples")

    model.train()
    completions = generate_completions(processor, model, image, samples, max_new_tokens)
    rewards = torch.tensor(
        [
            clip_reward(
                image,
                response_text(completion),
                clip_model,
                clip_processor,
                clip_tokenizer,
            )
            for completion in completions
        ],
        dtype=torch.float32,
        device=model.device,
    )
    log_probs = torch.stack(
        [
            sequence_log_probability(model, tokenizer, completion).squeeze(0)
            for completion in completions
        ]
    )

    baselines = (rewards.sum() - rewards) / (rewards.numel() - 1)
    advantages = rewards - baselines
    loss = -(advantages * log_probs).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    best_index = int(torch.argmax(rewards).item())
    return {
        "loss": float(loss.item()),
        "rewards": rewards.detach().cpu().tolist(),
        "completions": [response_text(item) for item in completions],
        "best_completion": response_text(completions[best_index]),
    }


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    records.clear()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_value(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"value must be between 0 and {MAX_SEED}")
    return parsed


def model_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--clip-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Base model ID used for the processor and tokenizer.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional local VLM checkpoint used instead of --model.",
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=seed_value, default=42)
    parser.add_argument("--flash-attention", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not args.image_root.is_dir():
        parser.error(f"--image-root is not a directory: {args.image_root}")
    if not args.clip_checkpoint.is_file():
        parser.error(f"--clip-checkpoint is not a file: {args.clip_checkpoint}")
    if args.checkpoint is not None and not args.checkpoint.is_dir():
        parser.error(f"--checkpoint is not a model directory: {args.checkpoint}")
    return args


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if args.flash_attention and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("--flash-attention requires a CUDA GPU with BF16 support")
    set_seed(args.seed)
    dataset = RLOODataset(args.annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        epoch_dir = args.output_dir / f"epoch-{epoch}"
        if epoch_dir.exists() and any(epoch_dir.iterdir()):
            raise FileExistsError(
                f"refusing to append to an existing RLOO run: {epoch_dir}"
            )

    model_source = str(args.checkpoint) if args.checkpoint else args.model
    dtype = model_dtype(device)
    attention = "flash_attention_2" if args.flash_attention else "eager"
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_source,
        torch_dtype=dtype,
        _attn_implementation=attention,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)

    clip_model = CLIPModel.from_pretrained(args.clip_model)
    state = torch.load(
        args.clip_checkpoint,
        map_location=device,
        weights_only=True,
    )
    clip_model.load_state_dict(state)
    clip_model = clip_model.to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_tokenizer = CLIPTokenizerFast.from_pretrained(args.clip_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(1, args.epochs + 1):
        epoch_dir = args.output_dir / f"epoch-{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        log_path = epoch_dir / "training.jsonl"
        buffered: list[dict[str, Any]] = []

        for step, entry in enumerate(
            tqdm(dataset, desc=f"RLOO epoch {epoch}"), start=1
        ):
            image_path = args.image_root / str(entry["file_name"])
            if not image_path.is_file():
                raise FileNotFoundError(f"training image does not exist: {image_path}")
            with Image.open(image_path) as raw_image:
                image = raw_image.convert("RGB")
                record = rloo_step(
                    model,
                    tokenizer,
                    processor,
                    clip_model,
                    clip_processor,
                    clip_tokenizer,
                    image,
                    optimizer,
                    args.samples,
                    args.max_new_tokens,
                )
            record["step"] = step
            buffered.append(record)
            if step % args.save_every == 0:
                model.save_pretrained(epoch_dir)
                tokenizer.save_pretrained(epoch_dir)
                append_jsonl(log_path, buffered)

        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        append_jsonl(log_path, buffered)


if __name__ == "__main__":
    main()
