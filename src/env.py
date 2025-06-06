import random
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional, Protocol, final, overload, runtime_checkable
from typing import cast as typing_cast

import gymnasium as gym
from overcooked_ai_py.mdp.overcooked_env import Overcooked, OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from src.dtypes import Action, Done, LayoutName, Observation, Reward


@runtime_checkable
class HasLayoutName(Protocol):
    @property
    def layout_name(self) -> str:
        ...


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
    while providing slightly more specificity and ommitting unneeded features.
    """
    action_space: gym.spaces.Space[Action]
    observation_space: gym.spaces.Space[Observation]

    @abstractmethod
    def step(
        self, actions: list[Action]
    ) -> tuple[dict[str, Any], Reward, Done, dict[str, Any]]:
        ...

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class WrappedOvercookedEnv(Overcookable, HasLayoutName):
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
    def layout_name(self) -> str:
        return self.base_mdp.layout_name

    @property
    def action_space(self) -> gym.spaces.Space[Action]:
        return self.env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space[Observation]:
        return self.env.observation_space

    def step(
        self, actions: list[Action]
    ) -> tuple[dict[str, Any], Reward, Done, dict[str, Any]]:
        return self.env.step(actions)

    def reset(self) -> dict[str, Any]:
        return self.env.reset()


@dataclass(frozen=True)
class OvercookedEnvFactory:
    """Greatly simplifies the creation of environments"""
    layout_name: Optional[LayoutName] = None
    old_dynamics: bool = field(default=True, kw_only=True)
    info_level: int = field(default=0, kw_only=True)
    horizon: int = field(default=400, kw_only=True)

    def create_env(self, layout_name: Optional[LayoutName] = None) -> WrappedOvercookedEnv:
        layout_name = layout_name or self.layout_name
        assert layout_name is not None, "A layout name must be provided"

        base_mdp = OvercookedGridworld.from_layout_name(layout_name, old_dynamics=self.old_dynamics)
        base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=self.horizon)
        env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
        return WrappedOvercookedEnv(base_mdp, base_env, env)

    def create_envs(self, *layout_names: LayoutName) -> list[WrappedOvercookedEnv]:
        return list(map(self.create_env, layout_names))

    __call__ = create_envs


@final
class MultipleOvercookedEnv(Overcookable, HasLayoutName):
    @overload
    def __init__(
        self,
        *layout_names: LayoutName,
        factory: Optional[OvercookedEnvFactory] = None,
        reset_env_interval: int = 1,
        allow_same_env_twice: bool = False
    ):
        ...

    @overload
    def __init__(
        self,
        *envs: Overcookable,
        factory: Optional[OvercookedEnvFactory] = None,
        reset_env_interval: int = 1,
        allow_same_env_twice: bool = False
    ):
        ...

    def __init__(
        self,
        *args: LayoutName | Overcookable,
        factory: Optional[OvercookedEnvFactory] = None,
        reset_env_interval: int = 1,
        allow_same_env_twice: bool = False
    ):
        args_it = iter(args)
        if (first_arg := next(args_it, None)) is None:
            raise ValueError("Expected a non empty list of arguments")

        expected_type = type(first_arg)
        assert all(type(arg) is expected_type for arg in args_it), \
            f"Expected a homogeneous argument type ({expected_type.__name__!r})"

        if isinstance(args[0], Overcookable):
            self.envs = typing_cast(list[Overcookable], args)
        else:
            factory = factory or OvercookedEnvFactory()
            self.envs = list(map(factory.create_env, args))

        self.env = self.envs[0]
        self.n_envs = len(self.envs)
        self.previous_index = len(self.envs)
        self.n_resets = 0
        self.reset_env_interval = reset_env_interval
        self.allow_same_env_twice = allow_same_env_twice

    def __getitem__(self, index: int) -> Overcookable:
        return self.envs[index]

    def __len__(self) -> int:
        # This isn't reliable, but we are nonetheless assuming
        # that the caller does not modify the environments
        # prior to using this method
        return self.n_envs

    @property
    def layout_name(self) -> str:
        if isinstance(self.env, HasLayoutName):
            return self.env.layout_name
        raise ValueError(f"Unknown layout name for {self.env!r}")

    @property
    def action_space(self) -> gym.spaces.Space[Action]:
        return self.env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space[Observation]:
        return self.env.observation_space

    def step(
        self, actions: list[Action]
    ) -> tuple[dict[str, Any], Reward, Done, dict[str, Any]]:
        return self.env.step(actions)

    def reset(self) -> dict[str, Any]:
        if self.n_resets == 0 or self.n_resets % self.reset_env_interval != 0:
            self.n_resets += 1
            return self.env.reset()

        randomise_env_index = partial(random.randint, 0, self.n_envs - 1)
        if self.n_envs > 2:
            index = randomise_env_index()
            if not self.allow_same_env_twice:
                while index == self.previous_index:
                    index = randomise_env_index()
        else:
            index = (self.previous_index + 1) % self.n_envs
        self.previous_index = index
        self.env = self.envs[index]
        return self.env.reset()
