import numpy as np

import torch

from src.types_ import Done, Reward, Value
from src.batching import RolloutExperiences
from src.networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(
        self,
        n_actions: int,
        input_dims: tuple[int, int],
        gamma: float = 0.99,
        alpha: float = 0.0003,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        minibatch_size: int = 64,
        n_epochs: int = 10
    ):
        self.experiences = RolloutExperiences(minibatch_size)
        self.gamma = gamma
        self.epsilon = policy_clip
        self.n_epochs = n_epochs
        # Lower lambda is lower bias estimate
        self.gae_lambda = gae_lambda

        # @ Why use separate optimizers for the actor and the critic?
        self.actor = ActorNetwork(n_actions, alpha, input_dims)
        self.critic = CriticNetwork(alpha, input_dims)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        prob = dist.log_prob(action)
        # @ Do we need .squeeze()?
        return action.squeeze().item(), prob.squeeze().item(), value.squeeze().item()
    
    def learn(self):
        for _ in range(self.n_epochs):       
            # @ When should advantages be computed? Why?
            current_advantages = self.compute_advantages(
                self.experiences.values, 
                self.experiences.rewards, 
                self.experiences.dones
            )
                
            for minibatch_indicies in self.experiences.minibatch_indicies():
                minibatch = self.experiences[minibatch_indicies]
                states, actions, values, old_probs, *_ = minibatch
                states, actions, values, old_probs = map(
                    torch.tensor, (states, actions, values, old_probs)
                )

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()
                advantages = current_advantages[minibatch_indicies]

                new_probs = dist.log_prob(actions)
                probs_ratio = new_probs.exp() / old_probs.exp()
                weighted_ratio = probs_ratio * advantages
                clipped_ratio = torch.clamp(probs_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
                returns = advantages + values

                actor_loss = -torch.min(weighted_ratio, clipped_ratio).mean()
                # @ Is this the same as MSE loss
                critic_loss = torch.pow(returns - critic_value, 2).mean()

                # @ Should we include entropy?
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # @ When do we clear this? Why?
        # @ Would it be more or less performant to discard earlier/later?
        self.experiences.clear()

    def compute_advantages(self, values: list[Value], rewards: list[Reward], dones: list[Done]) -> torch.Tensor:
        result = torch.empty(len(self.experiences.rewards), dtype=torch.float)
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += (
                    discount * 
                    (rewards[k] + self.gamma * values[k + 1] 
                        * (1 - int(dones[k])) - values[k])
                )
                discount *= self.gamma * self.gae_lambda
            result[t] = a_t
        return result