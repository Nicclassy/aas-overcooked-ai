import numpy as np

from src.agent import Agent
from src.dtypes import ObservationValue
from src.env import Overcookable
from src.parameters import Options
from src.training import OvercookedGame


class AgentTester:
    def __init__(self, env: Overcookable, agent: Agent, options: Options):
        self.env = env
        self.agent = agent
        self.options = options

    def play_game(self) -> OvercookedGame:
        info = self.env.reset()
        overcooked_state = info["overcooked_state"]
        observations = info["both_agent_obs"]

        rewards = []
        overcooked_states = []
        hud_datas = []
        game = OvercookedGame(rewards, overcooked_states, hud_datas, env=self.env)

        episode = 1
        done = False
        while not done:
            hud_datas.append({
                "episode": episode,
                "score": game.total_reward(),
                "soups": game.soups_made()
            })
            observations = np.array(
                [observation.astype(ObservationValue) for observation in observations]
            )

            actions = self.agent.choose_best_actions(observations)
            next_info, reward, done, _ = self.env.step(actions)

            overcooked_states.append(overcooked_state)
            rewards.append(reward)
            episode += 1

            next_observations = next_info["both_agent_obs"]
            next_state = next_info["overcooked_state"]

            observations = next_observations
            overcooked_state = next_state

        return game
