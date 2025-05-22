from typing import Callable, cast as typing_cast

import torch
import torch.nn as nn

from src.parameters import Hyperparameters
from src.types_ import State

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
            nn.ReLU(),
            nn.Linear(parameters.actor_fc1_dim, parameters.actor_fc2_dim),
            nn.ReLU(),
            nn.Linear(parameters.actor_fc2_dim, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=parameters.alpha
        )

    def forward(self, state: State) -> torch.distributions.Distribution:
        probs: torch.Tensor = self.net(state)
        # @ Logits or probs? What even is the difference?
        return torch.distributions.Categorical(probs=probs)
    
    __call__: Callable[[State], torch.distributions.Distribution]

class CriticNetwork(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        parameters: Hyperparameters
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, parameters.critic_fc1_dim),
            nn.ReLU(),
            nn.Linear(parameters.critic_fc1_dim, parameters.critic_fc2_dim),
            nn.ReLU(),
            nn.Linear(parameters.critic_fc2_dim, 1)
        )
        self.optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=parameters.alpha
        )

    def forward(self, state: State) -> torch.Tensor:
        return typing_cast(torch.Tensor, self.net(state)) \
            .squeeze().to(torch.float64)
    
    __call__: Callable[[State], torch.Tensor]