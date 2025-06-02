from collections.abc import Iterator
from dataclasses import dataclass
from typing import final

import numpy as np
from numpy.typing import NDArray

from src.dtypes import (
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
from src.misc import AttributeIterable, AttributeUnpackable, assert_instance

_runtime_type_check_experience = False
_runtime_type_checked_experience = [False]


@dataclass(slots=True, frozen=True)
class RolloutExperience(AttributeUnpackable):
    state: State
    action: Action
    value: Value
    prob: Probability
    reward: Reward
    done: Done

    def __post_init__(self):
        if _runtime_type_check_experience and not _runtime_type_checked_experience[0]:
            assert_instance(self.state, StateValue)
            assert_instance(self.action, Action)
            assert_instance(self.value, Value)
            assert_instance(self.prob, Probability)
            assert_instance(self.reward, Reward)
            assert_instance(self.done, Done)
            _runtime_type_checked_experience[0] = True

@final
class RolloutExperiences(AttributeIterable[list]):
    def __init__(self, minibatch_size: int, capacity: int):
        self.minibatch_size = minibatch_size
        self.states: list[StoredState] = [0] * capacity
        self.actions: list[StoredAction] = [0] * capacity
        self.values: list[StoredValue] = [0] * capacity
        self.probs: list[StoredProbability] = [0] * capacity
        self.rewards: list[StoredReward] = [0] * capacity
        self.dones: list[StoredDone] = [0] * capacity
        self.index = 0

    def __len__(self) -> int:
        return self.index

    def minibatch_indicies(self) -> Iterator[NDArray[np.int32]]:
        n_experiences = len(self)
        batch_start = np.arange(0, n_experiences, self.minibatch_size)
        indices = np.arange(n_experiences, dtype=np.int32)
        np.random.shuffle(indices)
        yield from (
            indices[i:i + self.minibatch_size]
            for i in batch_start
        )

    def add(self, experience: RolloutExperience):
        self.states[self.index] = experience.state
        self.actions[self.index] = experience.action
        self.values[self.index] = experience.value
        self.probs[self.index] = experience.prob
        self.rewards[self.index] = experience.reward
        self.dones[self.index] = experience.done
        self.index += 1

    def reset(self):
        self.index = 0
