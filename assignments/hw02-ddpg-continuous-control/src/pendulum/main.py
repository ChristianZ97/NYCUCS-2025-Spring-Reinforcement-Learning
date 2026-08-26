"""Deterministic command-line entry point for the Pendulum DDPG portfolio."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

try:
    from .train import TrainingConfig, train
    from .utils import get_device, make_env, set_seed
except ImportError:  # Allow `python main.py` from this directory.
    from train import TrainingConfig, train
    from utils import get_device, make_env, set_seed

MAX_SEED = 2**32 - 1


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def seed_value(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"value must be between 0 and {MAX_SEED}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description="Train the retained DDPG architecture on Pendulum-v1."
    )
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--seed", type=seed_value, default=1_745_986_213)
    parser.add_argument("--episodes", type=positive_int, default=defaults.num_episodes)
    parser.add_argument("--batch-size", type=positive_int, default=defaults.batch_size)
    parser.add_argument("--warm-up-steps", type=int, default=defaults.warm_up_steps)
    parser.add_argument(
        "--updates-per-episode",
        type=positive_int,
        default=defaults.updates_per_episode,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit destination for TensorBoard data and optional checkpoints.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save actor.pth and critic.pth under OUTPUT_DIR/checkpoints.",
    )
    return parser


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.warm_up_steps < 0:
        parser.error("--warm-up-steps must be non-negative")
    if args.save_model and args.output_dir is None:
        parser.error("--save-model requires --output-dir")

    set_seed(args.seed)
    device = get_device(args.device)
    writer = None
    env = None
    config = TrainingConfig(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        warm_up_steps=args.warm_up_steps,
        updates_per_episode=args.updates_per_episode,
    )
    try:
        env = make_env(
            args.env_id,
            seed=args.seed,
            render_mode="human" if args.render else None,
        )
        if args.output_dir is not None:
            writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))
        agent, result = train(env, config, device=device, writer=writer)
        if args.save_model:
            checkpoint_dir = args.output_dir / "checkpoints"
            agent.save_model(
                checkpoint_dir / "actor.pth", checkpoint_dir / "critic.pth"
            )
    finally:
        if env is not None:
            env.close()
        if writer is not None:
            writer.close()

    status = "solved" if result.solved else "not solved within the episode budget"
    final_ewma = result.ewma_rewards[-1] if result.ewma_rewards else float("nan")
    print(
        f"{status}; episodes={result.episodes_completed} "
        f"training_steps={result.training_steps} updates={result.updates} "
        f"final_ewma={final_ewma:.2f}"
    )
    return 0 if result.solved else 1


def main() -> int:
    parser = build_parser()
    return run(parser.parse_args(), parser)


if __name__ == "__main__":
    raise SystemExit(main())
