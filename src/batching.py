import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Generator, Iterator

from src.types_ import (
    Action, 
    Done, 
    Probability, 
    Reward, 
    State, 
    Value
)

def _numpy_getitem(value: list[object], indices: NDArray[np.int64]) -> NDArray:
    return np.array(value)[indices]

@dataclass(slots=True, frozen=True)
class RolloutExperience:
    state: State
    action: Action
    value: Value
    prob: Probability
    reward: Reward
    done: Done

    def __iter__(self) -> Iterator:
        yield from (self.state, self.action, self.value, self.prob, self.reward, self.done)

@dataclass(slots=True, frozen=True)
class Minibatch:
    states: State
    actions: NDArray[Action]
    values: NDArray[Value]
    probs: NDArray[Probability]
    rewards: NDArray[Reward]
    dones: NDArray[Done]

    def __iter__(self) -> Iterator:
        yield from (self.states, self.actions, self.values, self.probs, self.rewards, self.dones)

class RolloutExperiences:
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
            for name, value in self._attr_iter()
        })

    def minibatch_indicies(self) -> Generator[NDArray[np.int64]]:
        n_experiences = len(self.states)
        batch_start = np.arange(0, n_experiences, self.minibatch_size)
        indices = np.arange(n_experiences, dtype=np.int64)
        np.random.shuffle(indices)
        yield from (
            indices[i:i + self.minibatch_size]
            for i in batch_start
        )
    
    def add(self, experience: RolloutExperience):
        self.states.append(experience.state)
        self.actions.append(experience.action)
        self.values.append(experience.value)
        self.probs.append(experience.prob)
        self.rewards.append(experience.reward)
        self.dones.append(experience.done)

    def clear(self):
        for field in self._attr_values():
            field.clear()

    def _attr_values(self) -> Generator[list[object]]:
        yield from filter(lambda value: isinstance(value, list), self.__dict__.values())

    def _attr_iter(self, unmangle_names: bool = False) -> Generator[tuple[str, list[object]]]:
        def unmangle_attr_name(name: str) -> str:
            # Convert from something like '__Class__value' to 'value'
            return name.rsplit('_', maxsplit=1)[-1]
        
        for name, value in self.__dict__.items():
            if isinstance(value, list):
                if unmangle_names:
                    name = unmangle_attr_name(name)
                yield name, value