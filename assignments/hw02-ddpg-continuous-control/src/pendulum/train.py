"""Episode collection, DDPG updates, and evaluation for Pendulum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from .algo import DDPG, UpdateMetrics
    from .utils import OUNoise, ReplayMemory
except ImportError:  # Allow direct execution from this directory.
    from algo import DDPG, UpdateMetrics
    from utils import OUNoise, ReplayMemory


@dataclass(frozen=True)
class TrainingConfig:
    gamma: float = 0.999855778577656
    tau: float = 0.02988295604632515
    noise_scale: float = 1.4068608877080406
    actor_learning_rate: float = 0.0011115818565635618
    critic_learning_rate: float = 0.0026822139156196297
    batch_size: int = 512
    num_episodes: int = 600
    replay_size: int = 100_000
    warm_up_steps: int = 5_000
    reward_scale: float = 0.1
    hidden_size: int = 128
    updates_per_episode: int = 1
    target_ewma_reward: float = -200.0
    minimum_solve_episode: int = 20

    def validate(self) -> None:
        if self.batch_size <= 0 or self.num_episodes <= 0:
            raise ValueError("batch size and episode count must be positive")
        if self.replay_size < self.batch_size:
            raise ValueError("replay size must be at least the batch size")
        if self.warm_up_steps < 0 or self.updates_per_episode <= 0:
            raise ValueError("warm-up must be non-negative and updates positive")
        if self.reward_scale <= 0.0 or self.noise_scale < 0.0:
            raise ValueError("reward scale must be positive and noise non-negative")


@dataclass(frozen=True)
class TrainingResult:
    solved: bool
    episodes_completed: int
    training_steps: int
    updates: int
    rewards: tuple[float, ...]
    ewma_rewards: tuple[float, ...]


def collect_episode(
    env: Any,
    agent: DDPG,
    memory: ReplayMemory,
    action_noise: OUNoise,
    *,
    total_steps: int,
    warm_up_steps: int,
    reward_scale: float,
) -> tuple[int, float]:
    state, _ = env.reset()
    episode_reward = 0.0
    while True:
        if total_steps < warm_up_steps:
            action = env.action_space.sample()
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            with torch.inference_mode():
                action = agent.select_action(state_tensor, action_noise)
            action = action.cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        memory.push(
            state,
            action,
            0.0 if terminated else 1.0,
            next_state,
            float(reward) * reward_scale,
        )
        state = next_state
        episode_reward += float(reward)
        total_steps += 1
        if done:
            return total_steps, episode_reward


def update_after_episode(
    agent: DDPG,
    memory: ReplayMemory,
    *,
    batch_size: int,
    updates_per_episode: int,
    total_steps: int,
    writer: Any | None = None,
) -> tuple[int, UpdateMetrics]:
    metrics: UpdateMetrics | None = None
    for _ in range(updates_per_episode):
        metrics = agent.update_parameters(memory.sample(batch_size))
    assert metrics is not None

    if writer is not None:
        writer.add_scalar("Train/Critic_Loss", metrics.critic_loss, total_steps)
        writer.add_scalar("Train/Actor_Loss", metrics.actor_loss, total_steps)
        writer.add_scalar("Train/Actor_Grad_Norm", metrics.actor_grad_norm, total_steps)
        writer.add_scalar(
            "Train/Critic_Grad_Norm", metrics.critic_grad_norm, total_steps
        )
        writer.add_scalar("Train/Q_Eval", metrics.q_mean, total_steps)
        writer.add_scalar("Train/Q_Target", metrics.target_q_mean, total_steps)
        writer.add_scalar("Train/TD_Error", metrics.td_error_mean, total_steps)
    return updates_per_episode, metrics


def evaluate_agent(env: Any, agent: DDPG) -> float:
    state, _ = env.reset()
    episode_reward = 0.0
    while True:
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.inference_mode():
            action = agent.select_action(state_tensor).cpu().numpy()
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += float(reward)
        if terminated or truncated:
            return episode_reward


def train(
    env: Any,
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
        device=device,
    )
    memory = ReplayMemory(config.replay_size)
    action_noise = OUNoise(int(env.action_space.shape[0]))
    rewards: list[float] = []
    ewma_rewards: list[float] = []
    total_steps = 0
    total_updates = 0
    solved = False

    for episode in range(config.num_episodes):
        action_noise.scale = config.noise_scale * (1.0 - episode / config.num_episodes)
        action_noise.reset()
        total_steps, _ = collect_episode(
            env,
            agent,
            memory,
            action_noise,
            total_steps=total_steps,
            warm_up_steps=config.warm_up_steps,
            reward_scale=config.reward_scale,
        )
        if len(memory) >= max(config.warm_up_steps, config.batch_size):
            updates, _ = update_after_episode(
                agent,
                memory,
                batch_size=config.batch_size,
                updates_per_episode=config.updates_per_episode,
                total_steps=total_steps,
                writer=writer,
            )
            total_updates += updates

        reward = evaluate_agent(env, agent)
        ewma = 0.05 * reward + 0.95 * ewma_rewards[-1] if ewma_rewards else reward
        rewards.append(reward)
        ewma_rewards.append(ewma)
        if writer is not None:
            writer.add_scalar("Eval/Episode_Reward", reward, episode)
            writer.add_scalar("Eval/EWMA_Reward", ewma, episode)

        print(
            f"episode={episode + 1} steps={total_steps} "
            f"reward={reward:.2f} ewma={ewma:.2f}"
        )
        if (
            episode + 1 >= config.minimum_solve_episode
            and ewma >= config.target_ewma_reward
        ):
            solved = True
            break

    result = TrainingResult(
        solved=solved,
        episodes_completed=len(rewards),
        training_steps=total_steps,
        updates=total_updates,
        rewards=tuple(rewards),
        ewma_rewards=tuple(ewma_rewards),
    )
    return agent, result
