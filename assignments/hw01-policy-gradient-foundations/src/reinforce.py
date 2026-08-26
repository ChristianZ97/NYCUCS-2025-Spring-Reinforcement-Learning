#!/usr/bin/env python3
"""Vanilla REINFORCE for the historical CartPole-v0 experiment.

The submitted implementation uses returns directly in the policy loss and
keeps an auxiliary value head whose loss also trains the shared representation.
That behavior is intentionally preserved.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_DIR / "results" / "checkpoints" / "CartPole_0.008.pth"
MAX_SEED = 2**32 - 1


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def seed_value(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"value must be between 0 and {MAX_SEED}")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def finite_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, render: bool) -> Any:
    if render:
        try:
            return gym.make(env_id, render_mode="human")
        except TypeError:
            pass
    return gym.make(env_id)


def reset_env(env: Any, seed: Optional[int] = None) -> np.ndarray:
    try:
        result = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()
    state = result[0] if isinstance(result, tuple) else result
    return np.asarray(state, dtype=np.float32)


def step_env(env: Any, action: int) -> Tuple[np.ndarray, float, bool]:
    result = env.step(action)
    if len(result) == 5:
        state, reward, terminated, truncated, _ = result
        done = bool(terminated or truncated)
    elif len(result) == 4:
        state, reward, done, _ = result
        done = bool(done)
    else:
        raise RuntimeError(
            f"unexpected environment step result of length {len(result)}"
        )
    return np.asarray(state, dtype=np.float32), float(reward), done


def space_dimensions(env: Any) -> Tuple[int, int]:
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("this implementation requires a discrete action space")
    shape = getattr(env.observation_space, "shape", None)
    if not shape or len(shape) != 1:
        raise ValueError(
            "this implementation requires a one-dimensional observation space"
        )
    return int(shape[0]), int(env.action_space.n)


class Policy(nn.Module):
    """Submitted one-hidden-layer actor with an auxiliary value head."""

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        hidden_size = 128
        self.observation_layer = nn.Linear(observation_dim, hidden_size)
        self.action_layer = nn.Linear(hidden_size, action_dim)
        self.value_layer = nn.Linear(hidden_size, 1)

        with torch.no_grad():
            for layer in (
                self.observation_layer,
                self.action_layer,
                self.value_layer,
            ):
                scale = 1.0 / np.sqrt(layer.in_features)
                layer.weight.copy_(torch.randn_like(layer.weight) * scale)

        self.saved_actions: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.rewards: List[float] = []

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.tanh(self.observation_layer(state))
        return self.action_layer(features), self.value_layer(features)

    def act(self, state: np.ndarray, device: torch.device, record: bool) -> int:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        logits, state_value = self(state_tensor)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        if record:
            self.saved_actions.append((distribution.log_prob(action), state_value))
        return int(action.item())

    def loss(self, gamma: float) -> torch.Tensor:
        if not self.saved_actions:
            raise RuntimeError("cannot calculate a loss for an empty trajectory")

        discounted_returns: List[float] = []
        running_return = 0.0
        for reward in reversed(self.rewards):
            running_return = reward + gamma * running_return
            discounted_returns.append(running_return)
        discounted_returns.reverse()

        device = self.saved_actions[0][0].device
        returns = torch.as_tensor(
            discounted_returns, dtype=torch.float32, device=device
        )
        log_probs = torch.stack([item[0] for item in self.saved_actions])
        values = torch.stack([item[1].reshape(()) for item in self.saved_actions])

        policy_loss = -(log_probs * returns).sum()
        value_loss = F.mse_loss(values, returns, reduction="sum")
        return policy_loss + value_loss

    def clear_trajectory(self) -> None:
        self.saved_actions.clear()
        self.rewards.clear()


def save_checkpoint(model: Policy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: Policy, path: Path, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint does not contain a state dictionary: {path}")
    model.load_state_dict(state, strict=True)


def default_training_checkpoint(learning_rate: float) -> Path:
    return PROJECT_DIR / "runs" / f"CartPole_{learning_rate:g}.pth"


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = choose_device(args.device)
    env = make_env(args.env, render=False)
    writer = None

    try:
        observation_dim, action_dim = space_dimensions(env)
        model = Policy(observation_dim, action_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,
            gamma=0.9,
        )

        if args.log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(args.log_dir))

        ewma_reward = 0.0
        reached_threshold = False
        for episode in range(1, args.max_episodes + 1):
            scheduler.step()
            state = reset_env(env, args.seed if episode == 1 else None)
            episode_reward = 0.0
            model.train()

            for length in range(1, args.max_steps + 1):
                action = model.act(state, device, record=True)
                state, reward, done = step_env(env, action)
                model.rewards.append(reward)
                episode_reward += reward
                if done:
                    break

            loss = model.loss(args.gamma)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.clear_trajectory()

            ewma_reward = 0.05 * episode_reward + 0.95 * ewma_reward
            print(
                f"episode={episode} length={length} reward={episode_reward:.3f} "
                f"ewma_reward={ewma_reward:.3f}"
            )
            if writer is not None:
                writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
                writer.add_scalar("Train/Episode_Length", length, episode)
                writer.add_scalar("Train/EWMA_Reward", ewma_reward, episode)
                writer.add_scalar("Train/Loss", float(loss.item()), episode)

            if ewma_reward > args.stop_ewma:
                reached_threshold = True
                break

        checkpoint_out = (
            args.checkpoint_out
            if args.checkpoint_out is not None
            else default_training_checkpoint(args.learning_rate)
        )
        save_checkpoint(model, checkpoint_out)
        status = "reached" if reached_threshold else "did not reach"
        print(
            f"{status} configured EWMA threshold {args.stop_ewma}; "
            f"saved checkpoint to {checkpoint_out}"
        )
    finally:
        if writer is not None:
            writer.close()
        env.close()


def evaluate(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = choose_device(args.device)
    env = make_env(args.env, render=args.render)

    try:
        observation_dim, action_dim = space_dimensions(env)
        model = Policy(observation_dim, action_dim).to(device)
        load_checkpoint(model, args.checkpoint, device)
        model.eval()

        returns: List[float] = []
        with torch.inference_mode():
            for episode in range(args.episodes):
                state = reset_env(env, args.seed + episode)
                episode_return = 0.0
                for _ in range(args.max_steps):
                    action = model.act(state, device, record=False)
                    state, reward, done = step_env(env, action)
                    episode_return += reward
                    if args.render and not hasattr(env, "render_mode"):
                        env.render()
                    if done:
                        break
                returns.append(episode_return)
                print(f"episode={episode + 1} return={episode_return:.3f}")

        print(f"mean_return={float(np.mean(returns)):.3f}")
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a policy")
    train_parser.add_argument("--env", default="CartPole-v0")
    train_parser.add_argument("--seed", type=seed_value, default=10)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--learning-rate", type=positive_float, default=0.008)
    train_parser.add_argument("--gamma", type=probability, default=0.999)
    train_parser.add_argument("--max-episodes", type=positive_int, default=10_000)
    train_parser.add_argument("--max-steps", type=positive_int, default=10_000)
    train_parser.add_argument("--stop-ewma", type=finite_float, default=195.0)
    train_parser.add_argument(
        "--checkpoint-out",
        type=Path,
    )
    train_parser.add_argument("--log-dir", type=Path)
    train_parser.set_defaults(handler=train)

    eval_parser = subparsers.add_parser("eval", help="evaluate a saved policy")
    eval_parser.add_argument("--env", default="CartPole-v0")
    eval_parser.add_argument("--seed", type=seed_value, default=10)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    eval_parser.add_argument("--episodes", type=positive_int, default=10)
    eval_parser.add_argument("--max-steps", type=positive_int, default=10_000)
    eval_parser.add_argument("--render", action="store_true")
    eval_parser.set_defaults(handler=evaluate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "eval" and args.seed + args.episodes - 1 > MAX_SEED:
        parser.error("--seed plus --episodes exceeds the supported seed range")
    args.handler(args)


if __name__ == "__main__":
    main()
