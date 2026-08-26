"""Stage-two supervised fine-tuning for action prediction from image captions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from PIL import Image
from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available")
    return torch.device(requested)


def model_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


def extract_action(text: Any) -> str:
    if not isinstance(text, str) or "Action:" not in text:
        raise ValueError("each example text must contain an 'Action:' field")
    action = text.split("Action:", maxsplit=1)[1].split(".", maxsplit=1)[0].strip()
    if not action:
        raise ValueError("example contains an empty action")
    return action


class CaptionActionCollator:
    """Generate a caption, then construct the supervised action batch."""

    def __init__(
        self,
        model: Idefics3ForConditionalGeneration,
        processor: AutoProcessor,
        image_root: Path,
        device: torch.device,
        max_new_tokens: int,
    ) -> None:
        self.model = model
        self.processor = processor
        self.image_root = image_root
        self.device = device
        self.max_new_tokens = max_new_tokens

        special_tokens = processor.tokenizer.additional_special_tokens
        if "<image>" not in special_tokens:
            raise ValueError("processor tokenizer does not define an <image> token")
        image_index = special_tokens.index("<image>")
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            image_index
        ]

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[Image.Image] = []
        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                for example in examples:
                    relative_image = example.get("image")
                    if not isinstance(relative_image, str) or not relative_image:
                        raise ValueError(
                            "each example must contain a non-empty image path"
                        )
                    image_path = self.image_root / relative_image
                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"dataset image does not exist: {image_path}"
                        )
                    with Image.open(image_path) as raw_image:
                        image = raw_image.convert("RGB")
                    images.append(image)

                    caption_messages = [
                        {
                            "role": "user",
                            "content": [{"type": "image"}],
                        }
                    ]
                    caption_prompt = self.processor.apply_chat_template(
                        caption_messages, add_generation_prompt=False
                    )
                    inputs = self.processor(
                        text=caption_prompt,
                        images=image,
                        return_tensors="pt",
                    ).to(self.device)
                    generated = self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens
                    )
                    caption = self.processor.batch_decode(
                        generated, skip_special_tokens=True
                    )[0].strip()

                    prompt = (
                        "Based on the caption and image, what is the next action: "
                        "<forward, stop, left, right>?"
                    )
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Caption: {caption}\n{prompt}",
                                },
                                {"type": "image"},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": extract_action(example.get("text")),
                                }
                            ],
                        },
                    ]
                    texts.append(
                        self.processor.apply_chat_template(
                            messages, add_generation_prompt=False
                        ).strip()
                    )
        finally:
            self.model.train(was_training)

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--validation-dataset", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Base model ID used for the processor.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional stage-one checkpoint used instead of --model.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--flash-attention", action="store_true")
    args = parser.parse_args()

    for name in ("train_dataset", "validation_dataset", "image_root"):
        path = getattr(args, name)
        if not path.exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {path}")
    if args.checkpoint is not None and not args.checkpoint.is_dir():
        parser.error(f"--checkpoint is not a model directory: {args.checkpoint}")
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        parser.error("--epochs must be finite and positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if args.flash_attention and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("--flash-attention requires a CUDA GPU with BF16 support")
    source = str(args.checkpoint) if args.checkpoint else args.model
    dtype = model_dtype(device)
    use_bf16 = device.type == "cuda" and dtype == torch.bfloat16
    use_fp16 = device.type == "cuda" and not use_bf16
    attention = "flash_attention_2" if args.flash_attention else "eager"

    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        source,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        _attn_implementation=attention,
        use_cache=False,
    ).to(device)

    train_dataset = load_from_disk(str(args.train_dataset))
    validation_dataset = load_from_disk(str(args.validation_dataset))
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("training and validation datasets must both be non-empty")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optim="adamw_hf",
        logging_strategy="steps",
        logging_steps=1,
        logging_dir=str(args.output_dir / "logs"),
        save_strategy="epoch",
        save_total_limit=20,
        use_cpu=device.type == "cpu",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        lr_scheduler_type="constant",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CaptionActionCollator(
            model,
            processor,
            args.image_root,
            device,
            args.max_new_tokens,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
