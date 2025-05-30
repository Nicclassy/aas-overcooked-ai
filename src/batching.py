from collections.abc import Iterator
from dataclasses import dataclass
from typing import final

import numpy as np
from numpy.typing import NDArray

from src.misc import AttributeIterable, AttributeUnpackable, assert_instance
from src.types_ import (
    Action,
    Done,
    Probability,
    Reward,
    State,
    StateValue,
    StoredAction,
    StoredDone,
    StoredProbability,
    StoredReward,
    StoredState,
    StoredValue,
    Value,
)

_runtime_type_checked_experience = [True]


@dataclass(slots=True, frozen=True)
class AgentExperience(AttributeUnpackable):
    state: State
    action: Action
    value: Value
    prob: Probability
    reward: Reward
    done: Done

    def __post_init__(self):
        if not _runtime_type_checked_experience[0]:
            assert_instance(self.state, StateValue)
            assert_instance(self.action, Action)
            assert_instance(self.value, Value)
            assert_instance(self.prob, Probability)
            assert_instance(self.reward, Reward)
            assert_instance(self.done, Done)
            _runtime_type_checked_experience[0] = True


@dataclass(slots=True, frozen=True)
class Minibatch(AttributeUnpackable):
    states: StoredState
    actions: NDArray[StoredAction]
    values: NDArray[StoredValue]
    probs: NDArray[StoredProbability]
    rewards: NDArray[StoredReward]
    dones: NDArray[StoredDone]


@final
class AgentExperiences(AttributeIterable[list]):
    def __init__(self, minibatch_size: int, size: int):
        self.minibatch_size = minibatch_size
        self.states: list[StoredState] = [0] * size
        self.actions: list[StoredAction] = [0] * size
        self.values: list[StoredValue] = [0] * size
        self.probs: list[StoredProbability] = [0] * size
        self.rewards: list[StoredReward] = [0] * size
        self.dones: list[StoredDone] = [0] * size
        self.index = 0

    def __getitem__(self, indicies: NDArray[np.int32]) -> Minibatch:
        return Minibatch(**{
            name: np.array(value)[indicies]
            for name, value in self.attributes_by_name()
        })

    def __len__(self) -> int:
        return self.index

    def minibatch_indicies(self, use_minibatches: bool = True) -> Iterator[NDArray[np.int32]]:
        n_experiences = len(self)
        if not use_minibatches:
            yield np.arange(n_experiences)

        batch_start = np.arange(0, n_experiences, self.minibatch_size)
        indices = np.arange(n_experiences, dtype=np.int32)
        np.random.shuffle(indices)
        yield from (
            indices[i:i + self.minibatch_size]
            for i in batch_start
        )

    def add(self, experience: AgentExperience):
        self.states[self.index] = experience.state
        self.actions[self.index] = experience.action
        self.values[self.index] = experience.value
        self.probs[self.index] = experience.prob
        self.rewards[self.index] = experience.reward
        self.dones[self.index] = experience.done
        self.index += 1

    def reset(self):
        self.index = 0
