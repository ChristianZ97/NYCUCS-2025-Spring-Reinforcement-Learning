#!/usr/bin/env python3
"""REINFORCE with a learned state-value baseline for LunarLander-v2."""

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
DEFAULT_CHECKPOINT = PROJECT_DIR / "results" / "checkpoints" / "LunarLander_0.0001.pth"
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
    """Submitted shared actor-critic network with batch normalization."""

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        hidden_size = 512

        self.obs_layer1 = nn.Linear(observation_dim, hidden_size)
        self.obs_bn1 = nn.BatchNorm1d(hidden_size)
        self.obs_layer2 = nn.Linear(hidden_size, hidden_size)
        self.obs_bn2 = nn.BatchNorm1d(hidden_size)

        self.act_layer1 = nn.Linear(hidden_size, hidden_size)
        self.act_bn1 = nn.BatchNorm1d(hidden_size)
        self.act_layer2 = nn.Linear(hidden_size, action_dim)

        self.val_layer1 = nn.Linear(hidden_size, hidden_size)
        self.val_bn1 = nn.BatchNorm1d(hidden_size)
        self.val_layer2 = nn.Linear(hidden_size, 1)

        with torch.no_grad():
            for layer in (
                self.obs_layer1,
                self.obs_layer2,
                self.act_layer1,
                self.act_layer2,
                self.val_layer1,
                self.val_layer2,
            ):
                scale = 1.0 / np.sqrt(layer.in_features)
                layer.weight.copy_(torch.randn_like(layer.weight) * scale)

        self.saved_actions: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.rewards: List[float] = []

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)

        observation = F.relu(self.obs_bn1(self.obs_layer1(state)))
        observation = F.relu(self.obs_bn2(self.obs_layer2(observation)))
        actor = F.relu(self.act_bn1(self.act_layer1(observation)))
        critic = F.relu(self.val_bn1(self.val_layer1(observation)))
        return self.act_layer2(actor), self.val_layer2(critic)

    def act(self, state: np.ndarray, device: torch.device, record: bool) -> int:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        previous_mode = self.training
        self.eval()
        logits, state_value = self(state_tensor)
        self.train(previous_mode)

        distribution = Categorical(logits=logits)
        action = distribution.sample()
        if record:
            self.saved_actions.append(
                (distribution.log_prob(action).reshape(()), state_value.reshape(()))
            )
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
        values = torch.stack([item[1] for item in self.saved_actions])

        advantages = returns - values.detach()
        policy_loss = -(log_probs * advantages).sum()
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
    return PROJECT_DIR / "runs" / f"LunarLander_{learning_rate:g}.pth"


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
            step_size=200,
            gamma=0.95,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    train_parser.add_argument("--env", default="LunarLander-v2")
    train_parser.add_argument("--seed", type=seed_value, default=10)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--learning-rate", type=positive_float, default=0.0001)
    train_parser.add_argument("--gamma", type=probability, default=0.99)
    train_parser.add_argument("--max-episodes", type=positive_int, default=20_000)
    train_parser.add_argument("--max-steps", type=positive_int, default=10_000)
    train_parser.add_argument(
        "--stop-ewma",
        type=finite_float,
        default=120.0,
        help="historical run used 120; this is not the task's 200-return criterion",
    )
    train_parser.add_argument(
        "--checkpoint-out",
        type=Path,
    )
    train_parser.add_argument("--log-dir", type=Path)
    train_parser.set_defaults(handler=train)

    eval_parser = subparsers.add_parser("eval", help="evaluate a saved policy")
    eval_parser.add_argument("--env", default="LunarLander-v2")
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
