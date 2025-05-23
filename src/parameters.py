from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Hyperparameters:
    alpha: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon: float = 0.2  
    n_epochs: int = 10
    minibatch_size: int = 5
    critic_coefficient: float = 0.5

    actor_fc1_dim: int = 256
    actor_fc2_dim: int = 256
    critic_fc1_dim: int = 256
    critic_fc2_dim: int = 256

    def __post_init__(self):
        assert self.actor_fc2_dim <= self.actor_fc1_dim
        assert self.critic_fc2_dim <= self.critic_fc1_dim