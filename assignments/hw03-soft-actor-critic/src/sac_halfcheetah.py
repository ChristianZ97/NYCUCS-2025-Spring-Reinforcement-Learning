"""Soft Actor-Critic training for Gymnasium's HalfCheetah-v5 environment."""

import argparse
import random
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

MAX_SEED = 2**32 - 1


class MetricLogger:
    """Optional Weights & Biases logger; disabled unless explicitly requested."""

    def __init__(self, enabled: bool, project: str, run_name: str):
        self._wandb = None
        if not enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "Weights & Biases tracking requires the optional 'wandb' package."
            ) from exc

        self._wandb = wandb
        wandb.init(project=project, name=run_name, save_code=False)

    def log(self, metrics: dict) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Initialize a layer in the narrow range used by the original runs."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        """Build the stochastic policy network."""
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hidden1 = nn.Linear(in_dim, 512)
        self.hidden2 = nn.Linear(512, 512)

        self.log_std_layer = nn.Linear(512, out_dim)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)

        self.mu_layer = nn.Linear(512, out_dim)
        self.mu_layer = init_layer_uniform(self.mu_layer)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))

        mu = self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        # Correct the log probability for the tanh change of variables.
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob


class CriticQ(nn.Module):
    def __init__(self, in_dim: int):
        super(CriticQ, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 1)
        self.out = init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value


class CriticV(nn.Module):
    def __init__(self, in_dim: int):
        super(CriticV, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 1)
        self.out = init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class SACAgent:
    """SAC agent interacting with a continuous-control environment."""

    def __init__(self, env: gym.Env, args, logger=None):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.env = env
        self.log_metrics = logger.log if logger is not None else lambda _metrics: None
        self.memory_size = args.memory_size
        self.batch_size = args.batch_size
        self.memory = ReplayBuffer(
            obs_dim, action_dim, self.memory_size, self.batch_size
        )
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.lr = args.lr
        self.initial_random_steps = args.initial_random_steps
        self.policy_update_freq = args.policy_update_freq
        self.seed = args.seed
        self.num_steps = args.num_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        self.target_entropy = -np.prod((action_dim,)).item()  # heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

        self.actor = Actor(obs_dim, action_dim).to(self.device)

        self.vf = CriticV(obs_dim).to(self.device)
        self.vf_target = CriticV(obs_dim).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        self.qf_1 = CriticQ(obs_dim + action_dim).to(self.device)
        self.qf_2 = CriticQ(obs_dim + action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.lr)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=self.lr)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=self.lr)

        self.transition = list()
        self.total_step = 0

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        if self.total_step < self.initial_random_steps:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = (
                self.actor(torch.FloatTensor(state).to(self.device))[0]
                .detach()
                .cpu()
                .numpy()
            )

        self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        self.transition += [reward, next_state, done]
        self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by stochastic gradient descent."""
        device = self.device

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        new_action, log_prob = self.actor(state)

        # Automatic entropy-temperature update.
        alpha_loss = (
            -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        mask = 1 - done

        with torch.no_grad():
            v_target = self.vf_target(next_state)
            y = reward + mask * self.gamma * v_target

        q_1 = self.qf_1(state, action)
        q_2 = self.qf_2(state, action)
        qf_1_loss = F.mse_loss(q_1, y)
        qf_2_loss = F.mse_loss(q_2, y)

        new_q_1 = self.qf_1(state, new_action)
        new_q_2 = self.qf_2(state, new_action)
        q_hat = torch.min(new_q_1, new_q_2)

        v = self.vf(state)
        vf_loss = F.mse_loss(v, (q_hat - alpha * log_prob).detach())

        if self.total_step % self.policy_update_freq == 0:
            actor_loss = (alpha * log_prob - q_hat).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._target_soft_update()
        else:
            actor_loss = torch.zeros((), device=device)

        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()

        qf_loss = qf_1_loss + qf_2_loss

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        return (
            actor_loss.detach(),
            qf_loss.detach(),
            vf_loss.detach(),
            alpha_loss.detach(),
        )

    def train(self):
        """Train the agent."""
        state, _ = self.env.reset(seed=self.seed)
        score = 0
        ep = 0
        for self.total_step in tqdm(range(1, self.num_steps + 1)):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            actor_loss = 0.0
            qf_loss = 0.0
            vf_loss = 0.0
            alpha_loss = 0.0

            if done:
                state, _ = self.env.reset(seed=self.seed)
                ep += 1
                print(
                    f"Episode {ep} (Total step = {self.total_step}): Total Reward = {score}"
                )
                self.log_metrics({"episode": ep, "return": score})
                score = 0

            if (
                len(self.memory) >= self.batch_size
                and self.total_step > self.initial_random_steps
            ):
                losses = self.update_model()
                actor_loss, qf_loss, vf_loss, alpha_loss = [
                    float(loss.cpu()) for loss in losses
                ]

            self.log_metrics(
                {
                    "step": self.total_step,
                    "actor loss": actor_loss,
                    "q loss": qf_loss,
                    "v loss": vf_loss,
                    "alpha loss": alpha_loss,
                }
            )

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(self.vf_target.parameters(), self.vf.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def seed_value(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"must be between 0 and {MAX_SEED}")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be in the interval (0, 1]")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the portfolio SAC implementation on HalfCheetah-v5."
    )
    parser.add_argument("--lr", type=positive_float, default=0.0006344571440300171)
    parser.add_argument("--discount-factor", type=unit_interval, default=0.98)
    parser.add_argument("--tau", type=unit_interval, default=0.002244255539895659)
    parser.add_argument("--batch-size", type=positive_int, default=512)
    parser.add_argument("--initial-random-steps", type=nonnegative_int, default=10_000)
    parser.add_argument("--memory-size", type=positive_int, default=1_000_000)
    parser.add_argument("--num-steps", type=positive_int, default=1_000_000)
    parser.add_argument("--policy-update-freq", type=positive_int, default=1)
    parser.add_argument("--seed", type=seed_value, default=77)
    parser.add_argument(
        "--track",
        action="store_true",
        help="enable optional Weights & Biases metric logging",
    )
    parser.add_argument("--wandb-project", default="RL-HW3-SAC-HalfCheetah")
    parser.add_argument("--wandb-run-name", default="halfcheetah-sac")
    args = parser.parse_args()

    if args.batch_size > args.memory_size:
        parser.error("--batch-size cannot exceed --memory-size")
    return args


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    seed_torch(args.seed)

    try:
        logger = MetricLogger(args.track, args.wandb_project, args.wandb_run_name)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    env = None
    try:
        env = gym.make("HalfCheetah-v5")
        env = ActionNormalizer(env)
        agent = SACAgent(env, args, logger)
        agent.train()
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            logger.finish()


if __name__ == "__main__":
    main()
