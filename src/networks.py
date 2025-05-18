from typing import Callable

import torch
import torch.nn as nn

from src.types_ import State

class ActorNetwork(nn.Module):
    def __init__(self, n_actions: int, alpha: float, input_dims: tuple[int, int], fc1_dims: int = 256, fc2_dims: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=alpha)

    def forward(self, state: State) -> torch.distributions.Distribution:
        return self.layers(state)
    
    __call__: Callable[[State], torch.distributions.Distribution]

class CriticNetwork(nn.Module):
    def __init__(self, alpha: float, input_dims: tuple[int, int], fc1_dims: int = 256, fc2_dims: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=alpha)

    def forward(self, state: State) -> torch.Tensor:
        return self.layers(state)
    
    __call__: Callable[[State], torch.Tensor]