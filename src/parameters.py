from dataclasses import dataclass


@dataclass
class Hyperparameters:
    alpha: float = 8e-4
    gamma: float = 0.95
    gae_lambda: float = 1
    epsilon: float = 0.2
    n_epochs: int = 12
    minibatch_size: int = 4000
    critic_coefficient: float = 0.5
    entropy_coefficient: float = 0.001

    actor_fc1_dim: int = 64
    actor_fc2_dim: int = 64
    critic_fc1_dim: int = 64
    critic_fc2_dim: int = 64

    def __post_init__(self):
        assert self.actor_fc2_dim <= self.actor_fc1_dim
        assert self.critic_fc2_dim <= self.critic_fc1_dim


@dataclass
class Options:
    rollout_episodes: int = 10
    learn_episodes: int = 12
    total_episodes: int = 1000
    horizon: int = 400
    use_tqdm: bool = True
    use_minibatches: bool = False
    use_training_shaped_rewards: bool = True
    normalise_advantages: bool = True
    reset_epochs_after_game: bool = True
