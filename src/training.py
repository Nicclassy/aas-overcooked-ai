from dataclasses import dataclass
from pathlib import Path

import log
import numpy as np
import torch
from overcooked_ai_py.mdp.overcooked_env import Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

from src.agent import Agent
from src.batching import AgentExperience
from src.parameters import AlgorithmOptions
from src.types_ import Observation, Reward

CHECKPOINTS_DIR = Path(__file__).parent.parent.joinpath("checkpoints")


@dataclass
class GameResults:
    rewards: tuple[list[Reward], list[Reward]]
    states: list[OvercookedState]


class AgentTrainer:
    def __init__(
        self,
        env: Overcooked,
        agent1: Agent,
        agent2: Agent,
        options: AlgorithmOptions,
        learn_episodes: int = 20,
        n_agents: int = 2,
    ):
        assert n_agents == 2
        self.env = env
        self.options = options
        self.learn_episodes = learn_episodes
        self.n_agents = n_agents
        self.agents = (agent1, agent2)

    def train_agents(self, n_games: int = 1) -> list[GameResults]:
        results = []
        for game_number in range(1, n_games + 1):
            game_results = self.play_game(game_number)
            results.append(game_results)
            log.rl(
                f"Game {game_number}",
                "agent 1 total reward:",
                sum(game_results.rewards[0]),
                "agent 2 total reward:",
                sum(game_results.rewards[1]),
            )
        return results

    def play_game(self, game_number: int) -> GameResults:
        info = self.env.reset()
        observations = list(info["both_agent_obs"])
        state = info["overcooked_state"]

        states = []
        rewards = ([], [])
        episode = 1
        while True:
            states.append(state)
            actions = [None for _ in range(self.n_agents)]

            for i, observation in enumerate(observations):
                observation: Observation = observation.astype(np.float32)
                observations[i] = observation
                actions[i], *_ = action, prob, value = self.agents[i].choose_action(
                    observation
                )

            next_info, reward, done, env_info = self.env.step(actions)
            if done:
                break

            next_observations = list(next_info["both_agent_obs"])
            next_state = next_info["overcooked_state"]
            shaped_rewards = env_info["shaped_r_by_agent"]

            for i, observation in enumerate(observations):
                individual_reward = reward + shaped_rewards[i]
                experience = AgentExperience(
                    observation, action, value, prob, individual_reward, done
                )
                self.agents[i].experiences.add(experience)
                rewards[i].append(individual_reward)

            if episode % self.learn_episodes == 0:
                for i, agent in enumerate(self.agents, start=1):
                    agent.learn()

            observations = next_observations
            state = next_state
            # log.rl(
            #     f"[Game {game_number}; Episode {episode}]",
            #     "agent 1 reward:",
            #     rewards[0][-1],
            #     "agent 2 reward:",
            #     rewards[1][-1],
            # )
            episode += 1

        for agent in self.agents:
            agent.experiences.reset()
        return GameResults(rewards, states)

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
