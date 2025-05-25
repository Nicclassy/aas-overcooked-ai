from dataclasses import dataclass


@dataclass
class Hyperparameters:
    alpha: float = 2e-05
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon: float = 0.2
    n_epochs: int = 10
    minibatch_size: int = 5
    critic_coefficient: float = 0.5

    actor_fc1_dim: int = 64
    actor_fc2_dim: int = 64
    critic_fc1_dim: int = 64
    critic_fc2_dim: int = 64

    def __post_init__(self):
        assert self.actor_fc2_dim <= self.actor_fc1_dim
        assert self.critic_fc2_dim <= self.critic_fc1_dim


@dataclass
class AlgorithmOptions:
    horizon: int = 400
    normalise_advantages: bool = True
