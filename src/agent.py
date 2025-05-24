from functools import partial
from typing import final

import torch

from src.batching import AgentExperiences
from src.misc import flatten_dim
from src.networks import ActorNetwork, CriticNetwork
from src.parameters import Hyperparameters
from src.types_ import Action, Done, Observation, Probability, Reward, Value


@final
class Agent:
    def __init__(
        self, n_actions: int, input_dim: int | tuple[int], parameters: Hyperparameters
    ):
        self.experiences = AgentExperiences(parameters.minibatch_size)
        self.gamma = parameters.gamma
        self.epsilon = parameters.epsilon
        self.n_epochs = parameters.n_epochs
        self.gae_lambda = parameters.gae_lambda
        self.c1 = parameters.critic_coefficient

        # @ Why use separate optimizers for the actor and the critic?
        input_dim = flatten_dim(input_dim)
        self.actor = ActorNetwork(n_actions, input_dim, parameters)
        self.critic = CriticNetwork(input_dim, parameters)

    def choose_action(
        self, observation: Observation
    ) -> tuple[Action, Probability, Value]:
        state = torch.from_numpy(observation)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        prob = dist.log_prob(action)
        return action.item(), prob.item(), value.item()

    def learn(self):
        computed_advantages = self.compute_advantages(
            self.experiences.values, self.experiences.rewards, self.experiences.dones
        )
        # @ When should advantages be computed? Why?
        for _ in range(self.n_epochs):
            for minibatch_indicies in self.experiences.minibatch_indicies():
                minibatch = self.experiences[minibatch_indicies]
                states, actions, values, old_probs, *_ = map(
                    partial(torch.tensor, dtype=torch.float32), minibatch
                )

                dist = self.actor(states)
                critic_value = self.critic(states)
                advantages = computed_advantages[minibatch_indicies]

                new_probs = dist.log_prob(actions)
                probs_ratio = new_probs.exp() / old_probs.exp()
                weighted_ratio = probs_ratio * advantages
                clipped_ratio = (
                    torch.clamp(probs_ratio, 1 - self.epsilon, 1 + self.epsilon)
                    * advantages
                )
                returns = advantages + values

                actor_loss = -torch.min(weighted_ratio, clipped_ratio).mean()
                critic_loss = torch.nn.functional.mse_loss(returns, critic_value)

                # @ Should we include entropy?
                total_loss = actor_loss + self.c1 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # @ When do we clear this? Why?
        # @ Would it be more or less performant to discard earlier/later?
        self.experiences.clear()

    def compute_advantages(
        self, values: list[Value], rewards: list[Reward], dones: list[Done]
    ) -> torch.Tensor:
        result = torch.empty(len(self.experiences.rewards), dtype=torch.float32)
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (
                    rewards[k]
                    + self.gamma * values[k + 1] * (1 - int(dones[k]))
                    - values[k]
                )
                discount *= self.gamma * self.gae_lambda
            result[t] = a_t
        return result
