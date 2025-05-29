from functools import partial
from typing import final

import torch

from src.batching import AgentExperiences
from src.misc import flatten_dim
from src.networks import ActorNetwork, CriticNetwork
from src.parameters import AlgorithmOptions, Hyperparameters
from src.types_ import Action, Done, Observation, Probability, Reward, Value
from src.utils import GlobalState, summary_writer_factory


@final
class Agent:
    def __init__(
        self,
        n_actions: int,
        input_dim: int | tuple[int],
        parameters: Hyperparameters,
        options: AlgorithmOptions,
        multiple_agents: bool = False
    ):
        self.experiences = AgentExperiences(
            parameters.minibatch_size,
            options.horizon * options.rollout_episodes
        )
        self.options = options
        self.gamma = parameters.gamma
        self.epsilon = parameters.epsilon
        self.n_epochs = parameters.n_epochs
        self.gae_lambda = parameters.gae_lambda
        self.c1 = parameters.critic_coefficient
        self.c2 = parameters.entropy_coefficient
        self.multiple_agents = multiple_agents
        self.total_epochs = 0

        input_dim = flatten_dim(input_dim)
        assert input_dim >= parameters.actor_fc1_dim, \
            f"Input dim {input_dim} should be larger than layer 1 dim {parameters.actor_fc1_dim}"
        assert input_dim >= parameters.critic_fc1_dim, \
            f"Input dim {input_dim} should be larger than layer 1 dim {parameters.critic_fc1_dim}"
        self.actor = ActorNetwork(n_actions, input_dim, parameters)
        self.critic = CriticNetwork(input_dim, parameters)

        GlobalState.n_agents_created += 1
        self.agent_number = GlobalState.n_agents_created

    def choose_action(
        self, observation: Observation
    ) -> tuple[Action, Probability, Value]:
        state = torch.from_numpy(observation)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        prob = dist.log_prob(action)
        return action.item(), prob.item(), value.item()

    def choose_actions(self, observations: Observation):
        state = torch.from_numpy(observations)
        dist = self.actor(state)
        values = self.critic(state)
        actions = dist.sample()
        probs = dist.log_prob(actions)
        return actions.numpy(), probs.numpy(), values.numpy()

    def learn(self):
        def mean(values: list) -> torch.Tensor:
            return torch.tensor(values).mean()

        writer = summary_writer_factory(
            agent_number=self.agent_number,
            write=self.agent_number == 1,
            game_number=GlobalState.game_number,
            predicate=lambda: GlobalState.game_number == 1 or GlobalState.game_number % 10 == 0
        )

        computed_advantages = self.compute_advantages(
            self.experiences.values, self.experiences.rewards, self.experiences.dones
        )

        kl_values = []
        entropy_values = []
        actor_losses = []
        critic_losses = []
        total_losses = []

        for _ in range(1):
            for minibatch_indicies in self.experiences.minibatch_indicies(self.options.use_minibatches):
                minibatch = self.experiences[minibatch_indicies]
                states, actions, values, old_probs, *_ = map(
                    partial(torch.tensor, dtype=torch.float32), minibatch
                )

                dist = self.actor(states)
                critic_value = self.critic(states)
                advantages = computed_advantages[minibatch_indicies]
                if self.options.normalise_advantages:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-10
                    )

                new_probs = dist.log_prob(actions)
                probs_ratio = new_probs.exp() / old_probs.exp()

                weighted_ratio = probs_ratio * advantages
                clipped_ratio = (
                    torch.clamp(probs_ratio, 1 - self.epsilon, 1 + self.epsilon)
                    * advantages
                )
                returns = advantages + values

                entropy = dist.entropy().mean()
                actor_loss = -torch.min(weighted_ratio, clipped_ratio).mean()
                critic_loss = torch.nn.functional.mse_loss(returns, critic_value)

                # We add entropy rather than subtract it, because we want to
                # maximise entropy
                total_loss = actor_loss + self.c1 * critic_loss #- self.c2 * entropy
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

        if self.options.reset_experiences_each_train:
            self.experiences.reset()

        writer.add_scalar("Policy/KL", mean(kl_values), self.total_epochs)
        writer.add_scalar("Policy/Entropy", mean(entropy_values), self.total_epochs)
        writer.add_scalar("Loss/Actor", mean(actor_losses), self.total_epochs)
        writer.add_scalar("Loss/Critic", mean(critic_losses), self.total_epochs)
        writer.add_scalar("Loss/Total", mean(total_losses), self.total_epochs)

    def compute_advantages(
        self, values: list[Value], rewards: list[Reward], dones: list[Done]
    ) -> torch.Tensor:
        gae = 0.0
        n_experiences = len(self.experiences)
        if self.multiple_agents:
            advantages = [0.0] * n_experiences
        else:
            advantages = torch.empty(n_experiences, dtype=torch.float32)

        with torch.no_grad():
            for t in reversed(range(n_experiences)):
                if t == n_experiences - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1] * (1 - dones[t])
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages[t] = gae

        if self.multiple_agents:
            return torch.stack(advantages).float()
        return advantages

    def evaluate(self, states, actions):
        values = self.critic(states)
        dist = self.actor(states)
        log_probs = dist.log_prob(actions)
        return values, log_probs

    def reset(self):
        if self.options.reset_experiences_each_train:
            self.experiences.reset()
        if self.options.reset_epochs_after_game:
            self.total_epochs = 0
