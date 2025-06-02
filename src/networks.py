from collections.abc import Callable
from typing import cast as typing_cast

import torch
import torch.nn as nn

from src.dtypes import State
from src.parameters import Hyperparameters


class ActorNetwork(nn.Module):
    def __init__(
        self,
        n_actions: int,
        input_dim: int,
        parameters: Hyperparameters
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, parameters.actor_fc1_dim),
            nn.Tanh(),
            nn.Linear(parameters.actor_fc1_dim, parameters.actor_fc2_dim),
            nn.Tanh(),
            nn.Linear(parameters.actor_fc2_dim, n_actions)
        )
        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=parameters.alpha
        )

    def forward(self, state: State) -> torch.Tensor:
        return self.net(state)

    __call__: Callable[[State], torch.Tensor]

class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        parameters: Hyperparameters
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, parameters.critic_fc1_dim),
            nn.Tanh(),
            nn.Linear(parameters.critic_fc1_dim, parameters.critic_fc2_dim),
            nn.Tanh(),
            nn.Linear(parameters.critic_fc2_dim, 1)
        )
        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=parameters.alpha
        )

    def forward(self, state: State) -> torch.Tensor:
        return typing_cast(torch.Tensor, self.net(state)).squeeze(-1)

    __call__: Callable[[State], torch.Tensor]
