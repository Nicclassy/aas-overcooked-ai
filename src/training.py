import argparse
import subprocess
from typing import Optional
from typing import cast as typing_cast

import numpy as np
import torch

from src.agent import Agent
from src.batching import RolloutExperience
from src.dtypes import LayoutName, ObservationValue
from src.env import MultipleOvercookedEnv, Overcookable, OvercookedEnvFactory
from src.game import OvercookedGame, RolloutResults
from src.misc import (
    CountedInstanceCreation,
    TerminalLineClear,
    convert_to_cmd_args,
    dataclass_fieldnames,
    filter_attributes_only,
)
from src.parameters import Hyperparameters, Options
from src.utils import (
    RESET,
    GlobalState,
    create_writer,
    get_proper_layout_name,
    random_tqdm_colour,
    terminal_supports_truecolor,
    tqdm_type_factory,
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

    def train_agent(self) -> list[int | float]:
        current_episode = 0
        if self.options.use_tqdm:
            colour = random_tqdm_colour()
            desc = self.options.tqdm_description
            if isinstance(self.env, MultipleOvercookedEnv):
                desc = get_proper_layout_name(self.env.layout_name)

            tqdm = tqdm_type_factory(
                force_jupyter=self.options.is_running_jupyter_notebook
            )
            progress_bar = tqdm(
                total=self.options.total_episodes,
                desc=desc,
                unit="step",
                position=self.options.tqdm_position or 0,
                colour=colour if not terminal_supports_truecolor() else None
            )

            if terminal_supports_truecolor():
                progress_bar.bar_format = (
                    f"{{l_bar}}{colour}{{bar}}{RESET}{{r_bar}}"
                )

        rewards = []
        while current_episode < self.options.total_episodes:
            results = self.rollout()
            rewards.extend(game.total_reward() for game in results.games)
            self.agent.learn()
            self.agent.reset()

            if self.options.use_tqdm:
                if isinstance(self.env, MultipleOvercookedEnv):
                    desc = get_proper_layout_name(self.env.layout_name)
                    progress_bar.set_description(desc=desc)
                progress_bar.update(self.options.rollout_episodes)
                progress_bar.set_postfix(
                    {"average reward": results.average_reward()}
                )
            current_episode += self.options.rollout_episodes
            GlobalState.current_episode = current_episode

        self.agent.finish_training()
        return rewards


class ParallelAgentTrainer:
    def __init__(
        self,
        *layout_names: list[LayoutName],
        parameters: Optional[Hyperparameters | list[Hyperparameters]] = None,
        options: Optional[Options | list[Options]] = None,
    ):
        for parameter in (parameters, options):
            if isinstance(parameter, list):
                assert len(parameter) == len(layout_names)
        self.layout_names = layout_names
        self.parameters = parameters or Hyperparameters()
        self.options = options or Options()
        self.output_dest: Optional[TerminalLineClear] = None
        self.processes: list[subprocess.Popen] = []

    def redirect_output(self, dest: TerminalLineClear):
        self.output_dest = dest

    def train_agents(self):
        cmds = []
        for i, layout_name in enumerate(self.layout_names):
            options = self.options
            if isinstance(options, list):
                options = options[i]

            parameters = self.parameters
            if isinstance(parameters, list):
                parameters = parameters[i]

            if options.tqdm_description is None:
                options.tqdm_description = get_proper_layout_name(layout_name)

            cmd = [
                "python3",
                __file__,
                "--layout_name",
                repr(layout_name),
                "--tqdm_position",
                repr(i),
                *convert_to_cmd_args(options),
                *convert_to_cmd_args(parameters)
            ]
            cmds.append(cmd)

        for cmd in cmds:
            stdout = subprocess.PIPE if self.output_dest is not None else None
            stderr = subprocess.STDOUT if self.output_dest is not None else None
            process = subprocess.Popen(
                cmd,
                stdout=stdout,
                stderr=stderr,
                text=True,
                bufsize=1
            )
            self.processes.append(process)
            if self.output_dest is None:
                continue
            self.output_dest.add_process(process)

    def wait_until_finished_training(self):
        for process in self.processes:
            process.wait()


def main():
    """
    Warning: this is a very hacky way of running multiple agents at once
    We are essentially using reflection to build an argument parser
    from the given dataclasses (so that we need not specify the arguments ourselves,
    especially should they ever change).
    It is done like this so that subprocess.Popen can be used, so as to
    take advantage of its performance. This is done as opposed to threading
    which is slow due to shared memory, and multiprocessing, which
    cannot be done because some objects are not pickable, namely
    some pygame objects. This code does not implement
    control over environment parameters but they are usually the same in
    every circumstance nonetheless.
    """
    parser = argparse.ArgumentParser()
    fieldnames = dataclass_fieldnames(Options) | dataclass_fieldnames(Hyperparameters)
    for fieldname in fieldnames:
        parser.add_argument(f"--{fieldname}", required=False)
    parser.add_argument("--layout_name", required=True)

    args = {
        key: eval(value) if value is not None else value
        for key, value in vars(parser.parse_args()).items()
    }
    layout_name = args["layout_name"]
    options = Options(**filter_attributes_only(args, Options))
    parameters = Hyperparameters(**filter_attributes_only(args, Hyperparameters))
    env = OvercookedEnvFactory().create_env(layout_name)

    agent = Agent(
        parameters,
        options,
        env=env,
        writer_factory=lambda: create_writer(name=options.tqdm_description)
    )

    trainer = AgentTrainer(env, agent)
    trainer.train_agent()

if __name__ == "__main__":
    main()
