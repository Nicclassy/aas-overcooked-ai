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

_runtime_type_checked_experience = [False]


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
class AgentExperiences(AttributeIterable[np.ndarray]):
    def __init__(self, minibatch_size: int, size: int):
        self.minibatch_size = minibatch_size
        self.states: list[StoredState] = []
        self.actions: NDArray[StoredAction] = np.zeros(size, dtype=StoredAction)
        self.values: NDArray[StoredValue] = np.zeros(size, dtype=StoredValue)
        self.probs: NDArray[StoredProbability] = np.zeros(size, dtype=StoredProbability)
        self.rewards: NDArray[StoredReward] = np.zeros(size, dtype=StoredReward)
        self.dones: NDArray[StoredDone] = np.zeros(size, dtype=StoredDone)
        self.step = 0

    def __getitem__(self, indicies: NDArray[np.int64]) -> Minibatch:
        values = {name: value[indicies] for name, value in self.attributes_by_name()}
        values["states"] = np.array(self.states)[indicies]
        return Minibatch(**values)

    def __len__(self) -> int:
        # Make the reasonable inference that the amount of states
        # is the amount of actions is the amount of values etc.
        return len(self.states)

    def minibatch_indicies(self) -> Iterator[NDArray[np.int64]]:
        n_experiences = self.step
        batch_start = np.arange(0, n_experiences, self.minibatch_size)
        indices = np.arange(n_experiences, dtype=np.int64)
        np.random.shuffle(indices)
        yield from (
            indices[i:i + self.minibatch_size]
            for i in batch_start
        )

    def add(self, experience: AgentExperience):
        self.states.append(experience.state)
        self.actions[self.step] = experience.action
        self.values[self.step] = experience.value
        self.probs[self.step] = experience.prob
        self.rewards[self.step] = experience.reward
        self.dones[self.step] = experience.done
        self.step += 1

    def reset(self):
        self.states.clear()
        self.step = 0
