import argparse
from typing import Optional
from typing import cast as typing_cast

import numpy as np
import torch
from tqdm import tqdm

from src.agent import Agent
from src.batching import RolloutExperience
from src.dtypes import LayoutName, ObservationValue
from src.env import Overcookable, OvercookedEnvFactory
from src.game import OvercookedGame, RolloutResults
from src.misc import (
    CountedInstanceCreation,
    convert_to_cmd_args,
    dataclass_fieldnames,
    filter_attributes_only,
)
from src.parameters import Hyperparameters, Options
from src.utils import (
    GlobalState,
    create_writer,
    get_proper_layout_name,
    random_tqdm_colour,
)


class AgentTrainer:
    agent_number = CountedInstanceCreation()

    def __init__(
        self,
        env: Overcookable,
        agent: Agent
    ):
        self.env = env
        self.agent = agent
        self.options = agent.options

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
                stop_shaped_rewards = (
                    self.options.stop_shaped_rewards_after_episodes is not None
                    and GlobalState.current_episode > self.options.stop_shaped_rewards_after_episodes
                )
                if self.options.use_training_shaped_rewards and not stop_shaped_rewards:
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

    def train_agent(self):
        current_episode = 0
        if self.options.use_tqdm:
            progress_bar = tqdm(
                total=self.options.total_episodes,
                desc=self.options.tqdm_description,
                unit="step",
                position=int(self.options.tqdm_position or self.agent_number),
                colour=random_tqdm_colour()
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
            GlobalState.current_episode = current_episode


class ParallelAgentTrainer:
    def __init__(
        self,
        *layout_names: list[LayoutName],
        parameters: Optional[Hyperparameters] = None,
        options: Optional[Options] = None
    ):
        self.layout_names = layout_names
        self.parameters = parameters or Hyperparameters()
        self.options = options or Options()

    def train_agents(self):
        import subprocess

        cmds = []
        for tqdm_position, layout_name in enumerate(self.layout_names):
            self.options.tqdm_description = get_proper_layout_name(layout_name)
            extra_args = [
                *convert_to_cmd_args(self.options),
                *convert_to_cmd_args(self.parameters)
            ]

            cmd = [
                "python3",
                __file__,
                "--layout_name",
                repr(layout_name),
                "--tqdm_position",
                repr(tqdm_position),
                *extra_args
            ]
            cmds.append(cmd)

        print(cmds[0])
        exit()
        processes = [subprocess.Popen(cmd) for cmd in cmds]
        for process in processes:
            process.wait()


def main():
    parser = argparse.ArgumentParser()
    fieldnames = dataclass_fieldnames(Options) + dataclass_fieldnames(Hyperparameters)
    for fieldname in fieldnames:
        parser.add_argument(f"--{fieldname}", required=False)
    parser.add_argument("--layout_name", required=True)

    args = {
        key: eval(value) if value is not None else value
        for key, value in vars(parser.parse_args()).items()
    }
    layout_name = args["layout_name"]
    proper_layout_name = get_proper_layout_name(layout_name)
    options = Options(**filter_attributes_only(args, Options))
    parameters = Hyperparameters(**filter_attributes_only(args, Hyperparameters))
    env = OvercookedEnvFactory().create_env(layout_name)

    agent = Agent(
        parameters,
        options,
        env=env,
        writer_factory=lambda: create_writer(name=proper_layout_name)
    )

    trainer = AgentTrainer(env, agent)
    trainer.train_agent()

if __name__ == "__main__":
    main()
