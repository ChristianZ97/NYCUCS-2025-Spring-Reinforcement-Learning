"""Portfolio DDPG implementation for HalfCheetah.

This target is a cleaned derivative of the exact submitted ``ddpg_cheetah.py``.
It preserves the three-layer, layer-normalized networks and parameter-space
exploration used there.  It does not implement clipped double Q learning.
"""

from __future__ import annotations

import argparse
import random
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


Transition = namedtuple(
    "Transition", ("state", "action", "mask", "next_state", "reward")
)
MAX_SEED = 2**32 - 1


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
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


@torch.no_grad()
def hard_update(target: nn.Module, source: nn.Module) -> None:
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_parameter.copy_(source_parameter)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


class ReplayMemory:
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


class Actor(nn.Module):
    """Submitted three-layer policy with LayerNorm and GELU."""

    def __init__(self, hidden_size: int, num_inputs: int, action_space: object) -> None:
        super().__init__()
        action_shape = getattr(action_space, "shape")
        action_high = np.asarray(getattr(action_space, "high"), dtype=np.float32)
        num_actions = int(action_shape[0])

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_actions)
        self.register_buffer(
            "action_high", torch.as_tensor(action_high), persistent=False
        )

        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = F.gelu(self.ln1(self.fc1(inputs)))
        features = F.gelu(self.ln2(self.fc2(features)))
        features = F.gelu(self.ln3(self.fc3(features)))
        return torch.tanh(self.fc_out(features)) * self.action_high


class Critic(nn.Module):
    """Submitted three-layer, layer-normalized Q network."""

    def __init__(self, hidden_size: int, num_inputs: int, action_space: object) -> None:
        super().__init__()
        num_actions = int(getattr(action_space, "shape")[0])
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

        # Preserve the submitted initialization, including its default fc3 init.
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, inputs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = torch.cat((inputs, actions), dim=-1)
        features = F.relu(self.ln1(self.fc1(features)))
        features = F.relu(self.ln2(self.fc2(features)))
        features = F.relu(self.ln3(self.fc3(features)))
        return self.fc_out(features)


@dataclass(frozen=True)
class UpdateMetrics:
    critic_loss: float
    actor_loss: float
    q_mean: float
    target_q_mean: float
    td_error_mean: float
    actor_grad_norm: float
    critic_grad_norm: float


def gradient_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.norm().item())
    return total


class DDPG:
    def __init__(
        self,
        num_inputs: int,
        action_space: object,
        *,
        gamma: float,
        tau: float,
        hidden_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        critic_weight_decay: float,
        device: torch.device,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if actor_learning_rate <= 0.0 or critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_low = torch.as_tensor(
            np.asarray(getattr(action_space, "low"), dtype=np.float32),
            device=device,
        )
        # The submission accidentally initialized this from action_space.low.
        self.action_high = torch.as_tensor(
            np.asarray(getattr(action_space, "high"), dtype=np.float32),
            device=device,
        )

        self.actor = Actor(hidden_size, num_inputs, action_space).to(device)
        self.actor_target = Actor(hidden_size, num_inputs, action_space).to(device)
        self.actor_perturbed = Actor(hidden_size, num_inputs, action_space).to(device)
        self.critic = Critic(hidden_size, num_inputs, action_space).to(device)
        self.critic_target = Critic(hidden_size, num_inputs, action_space).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=critic_learning_rate,
            weight_decay=critic_weight_decay,
        )
        hard_update(self.actor_target, self.actor)
        hard_update(self.actor_perturbed, self.actor)
        hard_update(self.critic_target, self.critic)

    @torch.no_grad()
    def perturb_policy(self, standard_deviation: float) -> None:
        if standard_deviation < 0.0:
            raise ValueError("parameter-noise standard deviation must be non-negative")
        # Preserve the submitted choice of perturbing the lagged target actor.
        hard_update(self.actor_perturbed, self.actor_target)
        for parameter in self.actor_perturbed.parameters():
            parameter.add_(torch.randn_like(parameter) * standard_deviation)

    def select_action(
        self, state: torch.Tensor, *, perturbed: bool = False
    ) -> torch.Tensor:
        policy = self.actor_perturbed if perturbed else self.actor
        action = policy(state.to(self.device))
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
            target_q = self.critic_target(next_states, next_actions).clamp(
                -1000.0, 1000.0
            )
            td_target = rewards + self.gamma * masks * target_q

        q_values = self.critic(states, actions)
        td_error = (td_target - q_values).detach()
        critic_loss = F.mse_loss(q_values, td_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        critic_grad_norm = gradient_norm(self.critic.parameters())
        self.critic_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        policy_actions = self.actor(states)
        actor_loss = -self.critic(states, policy_actions).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        actor_grad_norm = gradient_norm(self.actor.parameters())
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
        self.actor.load_state_dict(
            torch.load(actor_path, map_location=self.device, weights_only=True),
            strict=True,
        )
        self.critic.load_state_dict(
            torch.load(critic_path, map_location=self.device, weights_only=True),
            strict=True,
        )
        hard_update(self.actor_target, self.actor)
        hard_update(self.actor_perturbed, self.actor)
        hard_update(self.critic_target, self.critic)


@dataclass(frozen=True)
class TrainingConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    critic_weight_decay: float = 5e-3
    hidden_size: int = 256
    batch_size: int = 128
    replay_size: int = 1_000_000
    warm_up_steps: int = 20_000
    max_training_steps: int = 500_000
    updates_per_episode: int = 1
    parameter_noise_std: float = 0.2
    target_ewma_reward: float = 5000.0
    minimum_solve_episode: int = 30

    def validate(self) -> None:
        positive = (
            self.hidden_size,
            self.batch_size,
            self.replay_size,
            self.max_training_steps,
            self.updates_per_episode,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("sizes, budget, and update count must be positive")
        if self.replay_size < self.batch_size:
            raise ValueError("replay size must be at least the batch size")
        if self.warm_up_steps < 0 or self.parameter_noise_std < 0.0:
            raise ValueError("warm-up and parameter noise must be non-negative")


@dataclass(frozen=True)
class TrainingResult:
    solved: bool
    episodes_completed: int
    training_steps: int
    updates: int
    rewards: tuple[float, ...]
    ewma_rewards: tuple[float, ...]


def collect_episode(
    env: gym.Env,
    agent: DDPG,
    memory: ReplayMemory,
    *,
    total_steps: int,
    config: TrainingConfig,
) -> tuple[int, float]:
    state, _ = env.reset()
    episode_reward = 0.0
    while total_steps < config.max_training_steps:
        if total_steps < config.warm_up_steps:
            action = env.action_space.sample()
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            with torch.inference_mode():
                action = agent.select_action(state_tensor, perturbed=True).cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        memory.push(
            state,
            action,
            0.0 if terminated else 1.0,
            next_state,
            float(reward),
        )
        state = next_state
        episode_reward += float(reward)
        total_steps += 1
        if done:
            break
    return total_steps, episode_reward


def evaluate_agent(env: gym.Env, agent: DDPG) -> float:
    state, _ = env.reset()
    reward_sum = 0.0
    while True:
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.inference_mode():
            action = agent.select_action(state_tensor).cpu().numpy()
        state, reward, terminated, truncated, _ = env.step(action)
        reward_sum += float(reward)
        if terminated or truncated:
            return reward_sum


def log_update(writer: Any | None, metrics: UpdateMetrics, step: int) -> None:
    if writer is None:
        return
    writer.add_scalar("Train/Critic_Loss", metrics.critic_loss, step)
    writer.add_scalar("Train/Actor_Loss", metrics.actor_loss, step)
    writer.add_scalar("Train/Actor_Grad_Norm", metrics.actor_grad_norm, step)
    writer.add_scalar("Train/Critic_Grad_Norm", metrics.critic_grad_norm, step)
    writer.add_scalar("Train/Q_Eval", metrics.q_mean, step)
    writer.add_scalar("Train/Q_Target", metrics.target_q_mean, step)
    writer.add_scalar("Train/TD_Error", metrics.td_error_mean, step)


def train(
    env: gym.Env,
    config: TrainingConfig,
    *,
    device: torch.device,
    writer: Any | None = None,
) -> tuple[DDPG, TrainingResult]:
    config.validate()
    agent = DDPG(
        int(env.observation_space.shape[0]),
        env.action_space,
        gamma=config.gamma,
        tau=config.tau,
        hidden_size=config.hidden_size,
        actor_learning_rate=config.actor_learning_rate,
        critic_learning_rate=config.critic_learning_rate,
        critic_weight_decay=config.critic_weight_decay,
        device=device,
    )
    memory = ReplayMemory(config.replay_size)
    rewards: list[float] = []
    ewma_rewards: list[float] = []
    total_steps = 0
    total_updates = 0
    solved = False
    episode = 0

    while total_steps < config.max_training_steps:
        agent.perturb_policy(config.parameter_noise_std)
        total_steps, _ = collect_episode(
            env, agent, memory, total_steps=total_steps, config=config
        )
        if len(memory) >= max(config.warm_up_steps + 1, config.batch_size):
            for _ in range(config.updates_per_episode):
                metrics = agent.update_parameters(memory.sample(config.batch_size))
                total_updates += 1
            log_update(writer, metrics, total_steps)

        reward = evaluate_agent(env, agent)
        ewma = 0.05 * reward + 0.95 * ewma_rewards[-1] if ewma_rewards else reward
        rewards.append(reward)
        ewma_rewards.append(ewma)
        if writer is not None:
            writer.add_scalar("Eval/Episode_Reward", reward, episode)
            writer.add_scalar("Eval/EWMA_Reward", ewma, episode)
        print(
            f"episode={episode + 1} training_steps={total_steps} "
            f"reward={reward:.2f} ewma={ewma:.2f}"
        )

        if (
            episode + 1 >= config.minimum_solve_episode
            and ewma >= config.target_ewma_reward
        ):
            solved = True
            break
        episode += 1

    return agent, TrainingResult(
        solved=solved,
        episodes_completed=len(rewards),
        training_steps=total_steps,
        updates=total_updates,
        rewards=tuple(rewards),
        ewma_rewards=tuple(ewma_rewards),
    )


def make_environment(env_id: str, *, seed: int, render_mode: str | None) -> gym.Env:
    try:
        env = gym.make(env_id, render_mode=render_mode)
    except Exception as error:
        raise RuntimeError(
            f"unable to create {env_id!r}; install the Gymnasium MuJoCo extra "
            "and verify that the MuJoCo runtime is available"
        ) from error
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


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


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Train the submitted DDPG architecture on HalfCheetah. "
            "This is vanilla DDPG, not clipped double Q."
        )
    )
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    parser.add_argument("--seed", type=seed_value, default=42)
    parser.add_argument(
        "--max-training-steps",
        type=positive_int,
        default=defaults.max_training_steps,
    )
    parser.add_argument("--batch-size", type=positive_int, default=defaults.batch_size)
    parser.add_argument("--warm-up-steps", type=int, default=defaults.warm_up_steps)
    parser.add_argument(
        "--updates-per-episode",
        type=positive_int,
        default=defaults.updates_per_episode,
    )
    parser.add_argument(
        "--actor-learning-rate",
        type=positive_float,
        default=defaults.actor_learning_rate,
    )
    parser.add_argument(
        "--critic-learning-rate",
        type=positive_float,
        default=defaults.critic_learning_rate,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit destination for TensorBoard data and optional checkpoints.",
    )
    parser.add_argument("--save-model", action="store_true")
    return parser


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.warm_up_steps < 0:
        parser.error("--warm-up-steps must be non-negative")
    if args.save_model and args.output_dir is None:
        parser.error("--save-model requires --output-dir")

    set_seed(args.seed)
    device = get_device(args.device)
    writer = None
    env = None
    config = TrainingConfig(
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        max_training_steps=args.max_training_steps,
        batch_size=args.batch_size,
        warm_up_steps=args.warm_up_steps,
        updates_per_episode=args.updates_per_episode,
    )
    try:
        env = make_environment(
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

    status = "solved" if result.solved else "target not reached"
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
