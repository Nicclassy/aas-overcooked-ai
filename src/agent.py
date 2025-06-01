from typing import final

import numpy as np
import torch
from numpy.typing import NDArray

from src.batching import RolloutExperiences
from src.misc import flatten_dim, numpy_getitem
from src.networks import ActorNetwork, CriticNetwork
from src.parameters import Hyperparameters, Options
from src.types_ import (
    Action,
    Done,
    Observation,
    Probability,
    Reward,
    StoredAction,
    StoredProbability,
    Value,
)
from src.utils import GlobalState, summary_writer_factory


@final
class Agent:
    def __init__(
        self,
        n_actions: int,
        input_dim: int | tuple[int],
        parameters: Hyperparameters,
        options: Options
    ):
        self.experiences = RolloutExperiences(
            parameters.minibatch_size,
            capacity=options.horizon * options.rollout_episodes
        )
        self.options = options
        self.gamma = parameters.gamma
        self.epsilon = parameters.epsilon
        self.n_epochs = parameters.n_epochs
        self.gae_lambda = parameters.gae_lambda
        self.c1 = parameters.critic_coefficient
        self.c2 = parameters.entropy_coefficient
        self.total_epochs = 0

        input_dim = flatten_dim(input_dim)
        assert input_dim >= parameters.actor_fc1_dim, \
            f"Input dim {input_dim} should be larger than layer 1 dim {parameters.actor_fc1_dim}"
        assert input_dim >= parameters.critic_fc1_dim, \
            f"Input dim {input_dim} should be larger than layer 1 dim {parameters.critic_fc1_dim}"
        self.actor = ActorNetwork(n_actions, input_dim, parameters)
        self.critic = CriticNetwork(input_dim * 2, parameters)

    def choose_action(
        self, observation: Observation
    ) -> tuple[Action, Probability, Value]:
        state = torch.from_numpy(observation)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        prob = dist.log_prob(action)
        return action.item(), prob.item(), value.item()

    def choose_actions(
        self, observations: Observation
    ) -> tuple[NDArray[StoredAction], NDArray[StoredProbability], Value]:
        states = torch.from_numpy(observations)
        dist = self.actor(states)
        value = self.critic(states.view(-1))
        actions = dist.sample()
        probs = dist.log_prob(actions)
        return actions.numpy(), probs.numpy(), value.item()

    def learn(self):
        def mean(values: list) -> torch.Tensor:
            return torch.tensor(values).mean()

        writer = summary_writer_factory(
            game_number=GlobalState.game_number,
            predicate=lambda: GlobalState.game_number == 1 or GlobalState.game_number % 10 == 0
        )

        advantages, returns = self.advantage_estimates(
            self.experiences.values, self.experiences.rewards, self.experiences.dones
        )

        if self.options.normalise_advantages:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        kl_values = []
        entropy_values = []
        actor_losses = []
        critic_losses = []
        total_losses = []

        for _ in range(self.n_epochs):
            states = torch.from_numpy(np.array(self.experiences.states)).float()
            actions = torch.from_numpy(np.array(self.experiences.actions)).float()
            probs = torch.tensor(np.array(self.experiences.probs)).float()

            for indicies in self.experiences.minibatch_indicies():
                b_states = numpy_getitem(states, indicies, index=self.options.use_minibatches)
                b_actions = numpy_getitem(actions, indicies, index=self.options.use_minibatches)
                b_advantages = numpy_getitem(advantages, indicies, index=self.options.use_minibatches)
                b_returns = numpy_getitem(returns, indicies, index=self.options.use_minibatches)
                old_probs = numpy_getitem(probs, indicies, index=self.options.use_minibatches)

                combined_action = b_states.view(b_states.shape[0], -1)
                # Critic values are the same for both agent
                new_values = self.critic(combined_action).unsqueeze(1).repeat(1, 2)
                dist = self.actor(b_states)

                new_probs = dist.log_prob(b_actions)
                probs_ratio = new_probs.exp() / old_probs.exp()

                weighted_ratio = probs_ratio * b_advantages
                clipped_ratio = (
                    torch.clamp(probs_ratio, 1 - self.epsilon, 1 + self.epsilon)
                    * advantages
                )

                entropy = dist.entropy().mean()
                actor_loss = -torch.min(weighted_ratio, clipped_ratio).mean()
                critic_loss = torch.nn.functional.mse_loss(new_values, b_returns)

                total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                kl_values.append((old_probs - new_probs).mean())
                entropy_values.append(entropy.item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                total_losses.append(total_loss.item())
                self.total_epochs += 1

        writer.add_scalar("Policy/KL", mean(kl_values), self.total_epochs)
        writer.add_scalar("Policy/Entropy", mean(entropy_values), self.total_epochs)
        writer.add_scalar("Loss/Actor", mean(actor_losses), self.total_epochs)
        writer.add_scalar("Loss/Critic", mean(critic_losses), self.total_epochs)
        writer.add_scalar("Loss/Total", mean(total_losses), self.total_epochs)

    def advantage_estimates(
        self, values: list[Value], rewards: list[Reward], dones: list[Done]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gae = 0.0
        n_experiences = len(self.experiences)
        advantages = [0.0] * n_experiences
        returns = [0.0] * n_experiences

        for t in reversed(range(n_experiences)):
            if t == n_experiences - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1] * (1 - dones[t])
            expected_return = rewards[t] + self.gamma * next_value
            delta = expected_return - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[n_experiences - t - 1] = expected_return

        return torch.stack(advantages).float(), torch.stack(returns).float()

    def reset(self):
        self.experiences.reset()
        if self.options.reset_epochs_after_game:
            self.total_epochs = 0
