from dataclasses import dataclass

import log
import numpy as np
from overcooked_ai_py.mdp.overcooked_env import Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

from src.batching import AgentExperience
from src.agent import Agent
from src.types_ import Observation, Reward

@dataclass
class TrainingResults:
    rewards: tuple[list[Reward], list[Reward]]
    states: list[OvercookedState]

class AgentTrainer:
    def __init__(
        self,
        env: Overcooked,
        agent1: Agent,
        agent2: Agent,
        learn_episodes: int = 20,
        n_agents: int = 2
    ):
        assert n_agents == 2
        self.env = env
        self.learn_episodes = learn_episodes
        self.n_agents = n_agents
        self.agents = (agent1, agent2)

    def train_agents(self) -> TrainingResults:
        info = self.env.reset()
        observations = list(info["both_agent_obs"])
        state = info["overcooked_state"]

        states = []
        rewards = ([], [])
        episode = 1
        while True:
            states.append(state)
            actions = [None for _ in range(self.n_agents)]
            probs = [None for _ in range(self.n_agents)]
            values = [None for _ in range(self.n_agents)]

            for i, observation in enumerate(observations):
                observation: Observation = observation.astype(np.float32)
                observations[i] = observation
                actions[i], probs[i], values[i] = action, prob, value = self.agents[i].choose_action(observation)

            next_info, reward, done, env_info = self.env.step(actions)
            if done:
                break

            next_observations = list(next_info["both_agent_obs"])
            next_state = next_info["overcooked_state"]
            sparse_rewards = env_info["sparse_r_by_agent"]

            for i, observation in enumerate(observations):
                individual_reward = reward + sparse_rewards[i]
                experience = AgentExperience(observation, action, value, prob, individual_reward, done)
                self.agents[i].experiences.add(experience)
                rewards[i].append(individual_reward)

            if episode % self.learn_episodes == 0:
                for i, agent in enumerate(self.agents, start=1):
                    log("Agent", i, "is learning")
                    agent.learn()

            observations = next_observations
            state = next_state
            log("Episode:", episode, "agent 1 reward:", rewards[0][-1], "agent 2 reward:", rewards[1][-1])
            episode += 1

        return TrainingResults(rewards, states)

        