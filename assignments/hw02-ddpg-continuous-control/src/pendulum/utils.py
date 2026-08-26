"""Utilities shared by the portfolio Pendulum implementation."""

from __future__ import annotations

import random
from collections import namedtuple
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn


Transition = namedtuple(
    "Transition", ("state", "action", "mask", "next_state", "reward")
)
MAX_SEED = 2**32 - 1


def get_device(preference: str = "auto") -> torch.device:
    """Resolve a requested PyTorch device without assuming accelerator support."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False


def make_env(
    env_id: str = "Pendulum-v1",
    *,
    seed: int = 42,
    render_mode: str | None = None,
) -> gym.Env:
    """Create and seed a Gymnasium environment."""
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


@torch.no_grad()
def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy all source parameters into a target network."""
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_parameter.copy_(source_parameter)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-average source parameters into a target network."""
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


class ReplayMemory:
    """Fixed-capacity cyclic replay memory."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self.memory: list[Transition | None] = []
        self.position = 0

    def push(self, *transition: Any) -> None:
        if len(transition) != len(Transition._fields):
            raise ValueError(f"expected {len(Transition._fields)} transition fields")
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*transition)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if batch_size > len(self.memory):
            raise ValueError(
                f"cannot sample {batch_size} transitions from {len(self.memory)}"
            )
        return random.sample(self.memory, batch_size)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck action noise used by the retained experiment."""

    def __init__(
        self,
        action_dimension: int,
        *,
        scale: float = 0.1,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ) -> None:
        if action_dimension <= 0:
            raise ValueError("action dimension must be positive")
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.empty(action_dimension, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.state.fill(self.mu)

    def noise(self) -> np.ndarray:
        delta = self.theta * (self.mu - self.state)
        delta += self.sigma * np.random.randn(self.action_dimension)
        self.state = self.state + delta
        return self.state * self.scale
