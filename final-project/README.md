# Label-Efficient VLM Fine-Tuning for Autonomous Driving

This team project adapts REINFORCE Leave-One-Out (RLOO) to fine-tune a
vision-language model for interpretable driving decisions. A domain-tuned CLIP
model scores image/description alignment, RLOO improves description generation,
and a second supervised stage maps the image and generated caption to a driving
action.

The portfolio surface does not imply sole authorship. Personal contact details
and the identity-bearing team report remain on the private `legacy` branch.

## Implementation

The canonical source is in [`src/`](src/):

- `clip_reward_train.py` builds positive and negative image/reason pairs and
  fine-tunes CLIP with binary cross-entropy.
- `clip_evaluate.py` evaluates pretrained or fine-tuned CLIP against the action,
  reason, and combined-label vocabularies.
- `rloo_train.py` samples multiple SmolVLM descriptions, uses the mean reward
  of the other samples as each RLOO baseline, and updates the policy without a
  learned critic.
- `sft_action.py` generates a caption and constructs the multimodal
  conversation used for supervised action tuning.

These are cleaned derivatives of the grading-time implementation. They
preserve the two-stage architecture, CLIP reward, complete-sequence bandit
formulation, and inherited loss approximations while replacing
machine-specific paths with explicit arguments, validating inputs, and making
output locations predictable. Result-affecting model-lineage details are
disclosed below.

## Data and dependencies

Use Python 3.10 or 3.11. The project needs PyTorch, Transformers, Datasets,
TorchMetrics, Pillow, and a CUDA-capable GPU for practical training. CUDA
training uses BF16 when the GPU supports it. Otherwise SFT uses FP16 automatic
mixed precision with FP32 model weights, while RLOO falls back to FP32:

```bash
cd final-project
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The BDD-OIA images and annotations, fine-tuned weights, and generated
checkpoints are not redistributed. Prepare:

- an annotation JSON list whose `file_name` is a relative image-path string and
  whose `action` and `reason` are each either a string or a one-element string
  list;
- the corresponding image directory;
- a Hugging Face dataset for each SFT split, where `image` is a relative
  image-path string and `text` contains a non-empty `Action:` field;
- the CLIP reward checkpoint before RLOO training.

## Run

Fine-tune and evaluate the CLIP reward model:

```bash
python3 src/clip_reward_train.py \
  --annotations data/annotations.json \
  --image-root data/images \
  --output-dir outputs/clip

python3 src/clip_evaluate.py \
  --annotations data/annotations.json \
  --image-root data/images \
  --checkpoint outputs/clip/best_clip_reward_model.pth \
  --output outputs/clip/evaluation.json
```

The grading-time RLOO script continued from a private local stage-one
checkpoint. Supply that checkpoint when reproducing its model lineage. The
following commands run the intended chained RLOO-to-SFT pipeline:

```bash
python3 src/rloo_train.py \
  --annotations data/train.json \
  --image-root data/images \
  --clip-checkpoint outputs/clip/best_clip_reward_model.pth \
  --checkpoint /path/to/private/stage-1-vlm-checkpoint \
  --output-dir outputs/rloo

python3 src/sft_action.py \
  --checkpoint outputs/rloo/epoch-1 \
  --train-dataset data/sft-train \
  --validation-dataset data/sft-validation \
  --image-root data/images \
  --output-dir outputs/sft
```

The active grading-time SFT source actually loaded the base SmolVLM; its local
checkpoint line was commented out. Omit `--checkpoint` from the SFT command to
mirror that exact lineage. The chained command above instead expresses the
project's intended two-stage flow. RLOO refuses to append to a non-empty
epoch directory, preventing logs from different runs from being mixed.

Set `CUDA_VISIBLE_DEVICES` externally when selecting a GPU. Flash Attention is
optional and must be requested explicitly with `--flash-attention`; the flag
requires a CUDA GPU with BF16 support.

## Results and provenance

[`results/`](results/) records compact measurements transcribed from the final
report. They are historical and were not regenerated from the cleaned source.
The report identifies limited model capacity, reward collapse, and action-format
collapse as important limitations; the retained numbers should not be read as a
production autonomous-driving claim.

[`submission/README.md`](submission/README.md) records the exact private
artifact manifest and hashes. The submitted archive, report, proposal, poster,
feedback, presentation, and toy-study documents are intentionally excluded from
the portfolio tree because they contain identities, collaborator contact
details, course-owned material, or redundant packaging metadata. They remain
recoverable byte-for-byte on the private `legacy` branch.

## Known implementation limits

- The grading-time RLOO approximation recomputed sequence log probabilities
  from decoded text without reconstructing the image-conditioned prompt. The
  canonical implementation preserves that result-affecting choice.
- The action-stage collator uses the model being trained to generate its own
  caption before constructing the supervised action example.
- SFT labels the entire non-padding, non-image conversation, not only the
  assistant action span. Prompt and generated-caption tokens therefore
  contribute to the training and validation losses, matching the grading-time
  implementation.
- CLIP creates a positive and negative pair for each image and then splits
  pairs, so the validation split is not image-identity-held-out. Its inherited
  horizontal-flip augmentation can also invert directional semantics without
  relabeling.
- Although the report formulates the state as a sequence of frames, the
  submitted implementation and canonical scripts consume one image per record.
- Full equivalence and performance validation require the private datasets,
  reward checkpoint, VLM checkpoints, and a suitable CUDA runtime.
