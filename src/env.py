import random
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Protocol, final, overload, runtime_checkable
from typing import cast as typing_cast

import gymnasium as gym
from overcooked_ai_py.mdp.overcooked_env import Overcooked, OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from src.dtypes import Action, Done, Observation, Reward


@runtime_checkable
class Overcookable(Protocol):
    """
    Instead of using `gym.Env` or `gymnasium.Env`,
    we will define our own, 'more convenient' class at the top
    of the hierarchy with slightly more specialised types and one
    which does not require `Env.render`.
    We have also ommitted `truncated` for its absence in `OvercookedEnv`.

    This is necessary for standardisation and for (explicitly) following
    the type signatures in line with Overcooked-AI's way (rather than OpenAI's),
    while providing slightly more specificity and ommitting unneeded features
    """
    action_space: gym.spaces.Space[Action]
    observation_space: gym.spaces.Space[Observation]

    @abstractmethod
    def step(
        self, action: Action | list[Action]
    ) -> tuple[dict[str, Any], Reward, Done, dict[str, Any]]:
        ...

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class WrappedOvercookedEnv(Overcookable):
    """
    Simplify access of attributes greatly, keeping them
    together while still providing more or less the same
    'type' at runtime—the behaviour is functionally still
    the same
    """
    base_mdp: OvercookedGridworld
    base_env: OvercookedEnv
    env: Overcooked

    @property
    def action_space(self) -> gym.spaces.Space[Action]:
        return self.env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space[Observation]:
        return self.env.observation_space

    def step(
        self, action: Action | list[Action]
    ) -> tuple[dict[str, Any], Reward, Done, dict[str, Any]]:
        return self.env.step(action)

    def reset(self) -> dict[str, Any]:
        return self.env.reset()


@dataclass(frozen=True, kw_only=True)
class OvercookedEnvFactory:
    """Greatly simplifies the creation of environments"""
    old_dynamics: bool = True
    info_level: int = 0
    horizon: int = 400

    def create_env(self, layout_name: str) -> WrappedOvercookedEnv:
        base_mdp = OvercookedGridworld.from_layout_name(layout_name, old_dynamics=self.old_dynamics)
        base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=self.horizon)
        env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
        return WrappedOvercookedEnv(base_mdp, base_env, env)


@final
class MultipleOvercookedEnv(Overcookable):
    @overload
    def __init__(
        self,
        *layout_names: str,
        **kwargs: Any
    ):
        ...

    @overload
    def __init__(
        self,
        *envs: Overcookable,
        **kwargs: Any
    ):
        ...

    def __init__(
        self,
        *args: str | Overcookable,
        factory: Optional[OvercookedEnvFactory] = None,
        switch_env_interval: int = 1,
        allow_same_env_twice: bool = False
    ):
        args_it = iter(args)
        if (first_arg := next(args_it, None)) is None:
            raise ValueError("Expected a non empty list of arguments")

        expected_type = type(first_arg)
        assert all(type(arg) is expected_type for arg in args_it), \
            f"Expected a homogeneous argument type ({expected_type.__name__!r})"

        if isinstance(args[0], str):
            factory = factory or OvercookedEnvFactory()
            self.envs = list(map(factory.create_env, args))
        else:
            self.envs = typing_cast(list[Overcookable], args)

        self.env = self.envs[0]
        self.previous_index = len(self.envs)
        self.n_resets = 0
        self.reset_env_interval = switch_env_interval
        self.allow_same_env_twice = allow_same_env_twice

    @property
    def action_space(self) -> gym.spaces.Space[Action]:
        return self.env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space[Observation]:
        return self.env.observation_space

    def step(
        self, action: Action | list[Action]
    ) -> tuple[dict[str, Any], Reward, Done, dict[str, Any]]:
        return self.env.step(action)

    def reset(self) -> dict[str, Any]:
        randomise_env_index = partial(random.randint, 0, self.n_envs - 1)
        index = randomise_env_index()
        while not self.allow_same_env_twice and index != self.previous_index:
            index = randomise_env_index()
        self.previous_index = index
        self.env = self.envs[index]
        return self.env.reset()
