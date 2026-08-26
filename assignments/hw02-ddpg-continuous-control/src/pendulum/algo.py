"""Deep Deterministic Policy Gradient update for Pendulum."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

try:
    from .model import Actor, Critic
    from .utils import OUNoise, Transition, get_device, hard_update, soft_update
except ImportError:  # Allow direct execution from this directory.
    from model import Actor, Critic
    from utils import OUNoise, Transition, get_device, hard_update, soft_update


@dataclass(frozen=True)
class UpdateMetrics:
    critic_loss: float
    actor_loss: float
    q_mean: float
    target_q_mean: float
    td_error_mean: float
    actor_grad_norm: float
    critic_grad_norm: float


def _gradient_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.norm().item())
    return total


class DDPG:
    """DDPG agent compatible with the retained April 2025 checkpoints."""

    def __init__(
        self,
        num_inputs: int,
        action_space: object,
        *,
        gamma: float = 0.999855778577656,
        tau: float = 0.02988295604632515,
        hidden_size: int = 128,
        actor_learning_rate: float = 0.0011115818565635618,
        critic_learning_rate: float = 0.0026822139156196297,
        device: torch.device | None = None,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if actor_learning_rate <= 0.0 or critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")

        self.device = device or get_device()
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space
        self.action_low = torch.as_tensor(
            np.asarray(getattr(action_space, "low"), dtype=np.float32),
            device=self.device,
        )
        self.action_high = torch.as_tensor(
            np.asarray(getattr(action_space, "high"), dtype=np.float32),
            device=self.device,
        )

        self.actor = Actor(hidden_size, num_inputs, action_space).to(self.device)
        self.actor_target = Actor(hidden_size, num_inputs, action_space).to(self.device)
        self.critic = Critic(hidden_size, num_inputs, action_space).to(self.device)
        self.critic_target = Critic(hidden_size, num_inputs, action_space).to(
            self.device
        )
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_learning_rate)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(
        self, state: torch.Tensor, action_noise: OUNoise | None = None
    ) -> torch.Tensor:
        action = self.actor(state.to(self.device))
        if action_noise is not None:
            noise = torch.as_tensor(
                action_noise.noise(), dtype=torch.float32, device=self.device
            )
            action = action + noise
        return action.clamp(self.action_low, self.action_high)

    def update_parameters(self, samples: list[Transition]) -> UpdateMetrics:
        if not samples:
            raise ValueError("cannot update from an empty batch")
        batch = Transition(*zip(*samples))
        states = torch.as_tensor(
            np.asarray(batch.state), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            np.asarray(batch.action), dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            np.asarray(batch.reward), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        masks = torch.as_tensor(
            np.asarray(batch.mask), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.asarray(batch.next_state), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            td_target = rewards + self.gamma * masks * target_q

        q_values = self.critic(states, actions)
        td_error = (td_target - q_values).detach()
        critic_loss = F.mse_loss(q_values, td_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        critic_grad_norm = _gradient_norm(self.critic.parameters())
        self.critic_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        policy_actions = self.actor(states)
        actor_loss = -self.critic(states, policy_actions).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        actor_grad_norm = _gradient_norm(self.actor.parameters())
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return UpdateMetrics(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            q_mean=float(q_values.mean().item()),
            target_q_mean=float(target_q.mean().item()),
            td_error_mean=float(td_error.abs().mean().item()),
            actor_grad_norm=actor_grad_norm,
            critic_grad_norm=critic_grad_norm,
        )

    def save_model(self, actor_path: Path, critic_path: Path) -> None:
        actor_path.parent.mkdir(parents=True, exist_ok=True)
        critic_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path: Path, critic_path: Path) -> None:
        if not actor_path.is_file() or not critic_path.is_file():
            raise FileNotFoundError(
                "both actor and critic checkpoint files are required"
            )
        actor_state = torch.load(
            actor_path, map_location=self.device, weights_only=True
        )
        critic_state = torch.load(
            critic_path, map_location=self.device, weights_only=True
        )
        self.actor.load_state_dict(actor_state, strict=True)
        self.critic.load_state_dict(critic_state, strict=True)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
