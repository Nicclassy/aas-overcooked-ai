from functools import partial
from typing import ClassVar, final

import torch
import torch.utils.tensorboard

from src.batching import AgentExperiences
from src.misc import flatten_dim
from src.networks import ActorNetwork, CriticNetwork
from src.parameters import AlgorithmOptions, Hyperparameters
from src.types_ import Action, Done, Observation, Probability, Reward, Value
from src.utils import get_summary_writer


@final
class Agent:
    _n_agents_created: ClassVar[int] = 0

    def __init__(
        self,
        n_actions: int,
        input_dim: int | tuple[int],
        parameters: Hyperparameters,
        options: AlgorithmOptions,
    ):
        self.experiences = AgentExperiences(parameters.minibatch_size, options.horizon)
        self.options = options
        self.gamma = parameters.gamma
        self.epsilon = parameters.epsilon
        self.n_epochs = parameters.n_epochs
        self.gae_lambda = parameters.gae_lambda
        self.c1 = parameters.critic_coefficient
        self.total_epochs = 0

        input_dim = flatten_dim(input_dim)
        assert input_dim > parameters.actor_fc1_dim
        assert input_dim > parameters.critic_fc1_dim
        self.actor = ActorNetwork(n_actions, input_dim, parameters)
        self.critic = CriticNetwork(input_dim, parameters)

        self.__class__._n_agents_created += 1
        self.agent_number = self.__class__._n_agents_created

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
        writer = get_summary_writer(write=self.agent_number == 1)
        computed_advantages = self.compute_advantages(
            self.experiences.values, self.experiences.rewards, self.experiences.dones
        )

        for _ in range(self.n_epochs):
            for minibatch_indicies in self.experiences.minibatch_indicies():
                minibatch = self.experiences[minibatch_indicies]
                states, actions, values, old_probs, *_ = map(
                    partial(torch.tensor, dtype=torch.float32), minibatch
                )

                dist = self.actor(states)
                critic_value = self.critic(states)
                advantages = computed_advantages[minibatch_indicies]
                if self.options.normalise_advantages:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-6
                    )

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
                # @ Why do we 'add' to the loss?
                total_loss = actor_loss + self.c1 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

            writer.add_scalar(
                "Policy/KL", (old_probs - new_probs).mean(), self.total_epochs
            )
            writer.add_scalar(
                "Policy/Entropy", dist.entropy().mean().item(), self.total_epochs
            )
            writer.add_scalar("Loss/Actor", actor_loss.item(), self.total_epochs)
            writer.add_scalar("Loss/Critic", critic_loss.item(), self.total_epochs)
            writer.add_scalar("Loss/Total", total_loss.item(), self.total_epochs)
            self.total_epochs += 1

    def compute_advantages(
        self, values: list[Value], rewards: list[Reward], dones: list[Done]
    ) -> torch.Tensor:
        n_experiences = len(self.experiences)
        advantages = torch.empty(n_experiences, dtype=torch.float32)
        gae = 0.0

        for t in reversed(range(n_experiences)):
            if t == n_experiences - 1:
                next_value = 0.0
                next_nonterminal = 0.0
            else:
                next_value = values[t + 1]
                next_nonterminal = 1 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantages[t] = gae

        return advantages
