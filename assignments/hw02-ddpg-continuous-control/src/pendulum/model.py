"""Actor and critic networks used by the retained Pendulum checkpoint."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    """Two-hidden-layer deterministic policy."""

    def __init__(self, hidden_size: int, num_inputs: int, action_space: object) -> None:
        super().__init__()
        if hidden_size <= 0 or num_inputs <= 0:
            raise ValueError("network dimensions must be positive")

        action_shape = getattr(action_space, "shape")
        action_high = np.asarray(getattr(action_space, "high"), dtype=np.float32)
        num_outputs = int(action_shape[0])

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_outputs)
        self.register_buffer(
            "action_high", torch.as_tensor(action_high), persistent=False
        )

        for layer in (self.fc1, self.fc2, self.fc_out):
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.fc1(inputs))
        features = F.relu(self.fc2(features))
        return torch.tanh(self.fc_out(features)) * self.action_high


class Critic(nn.Module):
    """Layer-normalized state-action value network."""

    def __init__(self, hidden_size: int, num_inputs: int, action_space: object) -> None:
        super().__init__()
        if hidden_size <= 0 or num_inputs <= 0:
            raise ValueError("network dimensions must be positive")

        action_shape = getattr(action_space, "shape")
        num_actions = int(action_shape[0])
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

        for layer in (self.fc1, self.fc2, self.fc_out):
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = torch.cat((inputs, actions), dim=-1)
        features = F.relu(self.ln1(self.fc1(features)))
        features = F.relu(self.ln2(self.fc2(features)))
        return self.fc_out(features)
