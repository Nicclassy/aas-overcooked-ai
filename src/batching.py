from collections.abc import Iterator
from dataclasses import dataclass
from typing import final

import numpy as np
from numpy.typing import NDArray

from src.dtypes import (
    StoredAction,
    StoredDone,
    StoredProbability,
    StoredReward,
    StoredState,
    StoredValue,
)
from src.misc import AttributeIterable, AttributeUnpackable


@dataclass(slots=True, frozen=True)
class RolloutExperience(AttributeUnpackable):
    state: StoredState
    action: StoredAction
    value: StoredValue
    prob: StoredProbability
    reward: StoredReward
    done: StoredDone


@final
class RolloutExperiences(AttributeIterable[list]):
    def __init__(self, batch_size: int, capacity: int):
        self.batch_size = batch_size
        self.states: list[StoredState] = [0] * capacity
        self.actions: list[StoredAction] = [0] * capacity
        self.values: list[StoredValue] = [0] * capacity
        self.probs: list[StoredProbability] = [0] * capacity
        self.rewards: list[StoredReward] = [0] * capacity
        self.dones: list[StoredDone] = [0] * capacity
        self.index = 0

    def __len__(self) -> int:
        return self.index

    def batch_indicies(self) -> Iterator[NDArray[np.int32]]:
        n_experiences = len(self)
        batch_start = np.arange(0, n_experiences, self.batch_size)
        indices = np.arange(n_experiences, dtype=np.int32)
        np.random.shuffle(indices)
        yield from (
            indices[i:i + self.batch_size]
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
