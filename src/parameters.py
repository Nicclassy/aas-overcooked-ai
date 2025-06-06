from dataclasses import dataclass
from typing import Optional

from src.misc import NonDestructiveMutation, non_destructive_mutation
from src.utils import jupyter_notebook_is_running


@non_destructive_mutation
@dataclass
class Hyperparameters(NonDestructiveMutation):
    alpha: float = 8e-4
    gamma: float = 0.95
    gae_lambda: float = 0.99
    epsilon: float = 0.2
    update_epochs: int = 15
    batch_size: int = 2048
    critic_coefficient: float = 0.5
    entropy_coefficient: float = 0.01

    actor_fc1_dim: int = 96
    actor_fc2_dim: int = 96
    critic_fc1_dim: int = 192
    critic_fc2_dim: int = 192

    def __post_init__(self):
        assert self.actor_fc2_dim <= self.actor_fc1_dim
        assert self.critic_fc2_dim <= self.critic_fc1_dim


@non_destructive_mutation
@dataclass
class Options(NonDestructiveMutation):
    rollout_episodes: int = 10
    total_episodes: int = 1000
    stop_shaped_rewards_after_episodes: Optional[int] = None
    horizon: int = 400
    checkpoints_dirname: Optional[str] = None
    tqdm_position: Optional[int] = None
    tqdm_description: Optional[str] = None
    use_tqdm: bool = True
    use_batches: bool = True
    use_training_shaped_rewards: bool = True
    normalise_advantages: bool = True
    reset_epochs_after_game: bool = False
    save_agent_after_training: bool = False
    is_running_jupyter_notebook: bool = jupyter_notebook_is_running()
