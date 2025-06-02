import itertools
from dataclasses import dataclass, field
from typing import Any, Optional
from typing import cast as typing_cast

import numpy as np
import torch
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from tqdm import tqdm

from src.agent import Agent
from src.batching import RolloutExperience
from src.dtypes import ObservationValue, Reward
from src.env import Overcookable, WrappedOvercookedEnv
from src.misc import iter_factory, timed
from src.parameters import Options


@dataclass(slots=True, frozen=True)
class OvercookedGame:
    rewards: list[Reward]
    overcooked_states: list[OvercookedState]
    hud_datas: list[dict[str, Any]] = field(default_factory=list)
    env: Optional[Overcookable] = None

    @property
    def base_mdp(self) -> Optional[OvercookedGridworld]:
        if isinstance(self.env, WrappedOvercookedEnv):
            return self.env.base_mdp
        else:
            return None

    def soups_made(self) -> int:
        return sum(reward // 20 for reward in self.rewards)

    def total_reward(self) -> int:
        return sum(self.rewards)

    def visualise(
        self,
        *,
        fps: int = 10,
        wait_after_last_frame: bool = True
    ):
        import pygame

        pygame.init()
        pygame.display.init()

        update_interval = 1000 // fps
        update_event = pygame.USEREVENT + 1

        base_mdp = self.base_mdp
        assert base_mdp is not None

        visualizer = StateVisualizer()
        hud_datas = self.hud_datas or itertools.repeat(None)
        next_rendered_state = iter_factory(
            visualizer.render_state(
                state,
                grid=base_mdp.terrain_mtx,
                hud_data=hud_data
            )
            for state, hud_data in zip(self.overcooked_states, hud_datas)
        )

        rendered_state = next_rendered_state()
        screen = pygame.display.set_mode(
            rendered_state.get_size(), flags=pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.event.set_allowed([pygame.QUIT, update_event])
        screen.blit(rendered_state, (0, 0))
        pygame.display.flip()

        pygame.time.set_timer(update_event, update_interval)
        pygame.display.set_caption("Overcooked-AI")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                elif event.type == update_event:
                    if (rendered_state := next_rendered_state()) is not None:
                        screen.blit(rendered_state, (0, 0))
                        pygame.display.flip()
                    elif not wait_after_last_frame:
                        running = False


@dataclass
class RolloutResults:
    games: list[OvercookedGame]

    def average_reward(self) -> int | float:
        average = 0
        for overcooked_game in self.games:
            average += overcooked_game.total_reward()

        if average == 0:
            return average

        average /= len(self.games)
        return average


class AgentTrainer:
    def __init__(
        self,
        env: Overcookable,
        agent: Agent,
        options: Options,
    ):
        self.env = env
        self.agent = agent
        self.options = options

    def rollout(self) -> RolloutResults:
        games: list[OvercookedGame] = []

        for _ in range(self.options.rollout_episodes):
            info = self.env.reset()
            overcooked_state = info["overcooked_state"]
            observations = info["both_agent_obs"]

            overcooked_states = []
            rewards = []
            done = False
            while not done:
                observations = np.array(
                    [observation.astype(ObservationValue) for observation in observations]
                )

                with torch.no_grad():
                    actions, probs, value = self.agent.choose_actions(np.array(observations))

                next_info, reward, done, env_info = self.env.step(actions.tolist())
                if self.options.use_training_shaped_rewards:
                    shaped_rewards = env_info["shaped_r_by_agent"]
                else:
                    shaped_rewards = [0, 0]
                reward += torch.tensor(shaped_rewards)
                episode_reward = typing_cast(torch.Tensor, reward).sum().item()

                self.agent.experiences.add(
                    RolloutExperience(observations, actions, value, probs, reward, done)
                )

                overcooked_states.append(overcooked_state)
                rewards.append(episode_reward)

                next_observations = next_info["both_agent_obs"]
                next_state = next_info["overcooked_state"]

                observations = next_observations
                overcooked_state = next_state

            games.append(OvercookedGame(rewards, overcooked_states))

        return RolloutResults(games)

    @timed
    def train_agent(self):
        current_episode = 0
        if self.options.use_tqdm:
            progress_bar = tqdm(
                total=self.options.total_episodes,
                desc="PPO Training",
                unit="step"
            )

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
