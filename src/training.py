from dataclasses import dataclass
from typing import cast as typing_cast

import numpy as np
import torch
from overcooked_ai_py.mdp.overcooked_env import Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from tqdm import tqdm

from src.agent import Agent
from src.batching import AgentExperience
from src.misc import CHECKPOINTS_DIR
from src.parameters import AlgorithmOptions
from src.types_ import Observation, Reward


@dataclass
class GameResults:
    rewards: tuple[list[Reward], list[Reward]] | list[Reward]
    states: list[OvercookedState]

@dataclass
class RolloutResults:
    game_results: list[GameResults]

    def average_reward(self, agent_number: int = 1) -> int:
        if isinstance(self.game_results[0], tuple):
            return sum(
                sum(game_result.rewards[agent_number - 1]) / len(game_result.rewards[agent_number])
                for game_result in self.game_results
            )
        else:
            return sum(
                sum(game_result.rewards) / len(game_result.rewards)
                for game_result in self.game_results
            )


class MultiAgentTrainer:
    def __init__(
        self,
        env: Overcooked,
        agent1: Agent,
        agent2: Agent,
        options: AlgorithmOptions
    ):
        self.env = env
        self.options = options
        self.n_agents = 2
        self.agents = (agent1, agent2)

    def train_agents(self):
        current_episode = 0
        progress_bar = tqdm(total=self.options.total_episodes, desc="PPO Training", unit="step")
        while current_episode < self.options.total_episodes:
            results = self.rollout()
            for agent in self.agents:
                agent.learn()
                agent.reset()

            progress_bar.update(self.options.rollout_episodes)
            progress_bar.set_postfix(
                {"average reward (agent 1)": results.average_reward(agent_number=1)}
            )
            current_episode += self.options.rollout_episodes

    def rollout(self) -> RolloutResults:
        game_results: list[GameResults] = []

        for _ in range(self.options.rollout_episodes):
            info = self.env.reset()
            state = info["overcooked_state"]
            observations = list(info["both_agent_obs"])

            states = []
            rewards = ([], [])
            while True:
                states.append(state)
                actions = [None for _ in range(self.n_agents)]

                for i, observation in enumerate(observations):
                    observation: Observation = observation.astype(np.float32)
                    observations[i] = observation
                    with torch.no_grad():
                        actions[i], *_ = action, prob, value = self.agents[i].choose_action(
                            observation
                        )

                next_info, reward, done, env_info = self.env.step(actions)
                shaped_rewards = env_info["shaped_r_by_agent"]

                for i, observation in enumerate(observations):
                    individual_reward = reward + shaped_rewards[i]
                    experience = AgentExperience(
                        observation, action, value, prob, individual_reward, done
                    )
                    self.agents[i].experiences.add(experience)
                    rewards[i].append(individual_reward)

                if done:
                    break

                next_observations = list(next_info["both_agent_obs"])
                next_state = next_info["overcooked_state"]

                observations = next_observations
                state = next_state

            game_results.append(GameResults(rewards, states))

        return RolloutResults(game_results)

    def save_agents(self):
        if not CHECKPOINTS_DIR.exists():
            CHECKPOINTS_DIR.mkdir()

        for i, agent in enumerate(self.agents, start=1):
            actor_save_path = CHECKPOINTS_DIR.joinpath(f"actor{i}.pth")
            critic_save_path = CHECKPOINTS_DIR.joinpath(f"critic{i}.pth")
            torch.save(agent.actor.state_dict(), actor_save_path)
            torch.save(agent.critic.state_dict(), critic_save_path)

    def load_agents(self):
        for i, agent in enumerate(self.agents, start=1):
            actor_save_path = CHECKPOINTS_DIR.joinpath(f"actor{i}.pth")
            critic_save_path = CHECKPOINTS_DIR.joinpath(f"critic{i}.pth")
            agent.actor.load_state_dict(torch.load(actor_save_path))
            agent.critic.load_state_dict(torch.load(critic_save_path))


class SingleAgentTrainer:
    def __init__(
        self,
        env: Overcooked,
        agent: Agent,
        options: AlgorithmOptions,
    ):
        self.env = env
        self.agent = agent
        self.options = options

    def rollout(self) -> RolloutResults:
        game_results: list[GameResults] = []

        for _ in range(self.options.rollout_episodes):
            info = self.env.reset()
            state = info["overcooked_state"]
            observations = list(info["both_agent_obs"])

            states = []
            rewards = []
            while True:
                states.append(state)
                observations = np.array(
                    [observation.astype(np.float32) for observation in observations]
                )

                with torch.no_grad():
                    actions, probs, values = self.agent.choose_actions(np.array(observations))

                next_info, reward, done, env_info = self.env.step(actions.tolist())
                shaped_rewards = env_info["shaped_r_by_agent"]
                reward += torch.tensor(shaped_rewards)
                rewards.append(typing_cast(torch.Tensor, reward).sum().item())

                self.agent.experiences.add(
                    AgentExperience(observations, actions, values, probs, reward, done)
                )

                if done:
                    break

                next_observations = list(next_info["both_agent_obs"])
                next_state = next_info["overcooked_state"]

                observations = next_observations
                state = next_state

            game_results.append(GameResults(rewards, states))

        return RolloutResults(game_results)

    def train_agent(self):
        current_episode = 0
        if self.options.use_tqdm:
            progress_bar = tqdm(total=self.options.total_episodes, desc="PPO Training", unit="step")

        while current_episode < self.options.total_episodes:
            results = self.rollout()
            self.agent.learn()
            self.agent.reset()

            if self.options.use_tqdm:
                progress_bar.update(self.options.rollout_episodes)
                progress_bar.set_postfix(
                    {"average reward": results.average_reward()}
                )
            current_episode += self.options.rollout_episodes
