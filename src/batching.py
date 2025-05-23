from dataclasses import dataclass
from typing import Any, Iterator, final

import numpy as np
from numpy.typing import NDArray

from src.types_ import (
    Action, 
    Done, 
    Probability, 
    Reward, 
    State,
    StateValue,
    StoredState, 
    Value,
    StoredAction,
    StoredValue,
    StoredProbability,
    StoredReward,
    StoredDone
)
from src.misc import AttributeIterable, AttributeUnpackable, assert_instance

_runtime_type_checked_experience = [False]

def _numpy_getitem(value: list[Any], indices: NDArray[np.int64]) -> NDArray:
    return np.array(value)[indices]

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
    def __init__(self, minibatch_size: int):
        self.minibatch_size = minibatch_size
        self.states: list[State] = []
        self.actions: list[Action] = []
        self.values: list[Value] = []
        self.probs: list[Probability] = []
        self.rewards: list[Reward] = []
        self.dones: list[Done] = []

    def __getitem__(self, indicies: NDArray[np.int64]) -> Minibatch:
        return Minibatch(**{
            name: _numpy_getitem(value, indicies)
            for name, value in self.attributes_by_name()
        })
    
    def __len__(self) -> int:
        # Make the reasonable inference that the amount of states
        # is the amount of actions is the amount of values etc.
        return len(self.states)

    def minibatch_indicies(self) -> Iterator[NDArray[np.int64]]:
        n_experiences = len(self)
        batch_start = np.arange(0, n_experiences, self.minibatch_size)
        indices = np.arange(n_experiences, dtype=np.int64)
        np.random.shuffle(indices)
        yield from (
            indices[i:i + self.minibatch_size]
            for i in batch_start
        )
    
    def add(self, experience: AgentExperience):
        self.states.append(experience.state)
        self.actions.append(experience.action)
        self.values.append(experience.value)
        self.probs.append(experience.prob)
        self.rewards.append(experience.reward)
        self.dones.append(experience.done)

    def clear(self):
        for value in self.attribute_values():
            value.clear()