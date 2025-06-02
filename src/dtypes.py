from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ObservationValue: TypeAlias = np.float32
StateValue: TypeAlias = np.float32

Observation: TypeAlias = NDArray[ObservationValue]
State: TypeAlias = NDArray[StateValue]
Action: TypeAlias = int
Value: TypeAlias = float
Probability: TypeAlias = float
Reward: TypeAlias = int
Done: TypeAlias = bool

# These typealiases are used for the 'stored' versions of the
# above typealiases, that is, representing the type
# after it has been converted by numpy upon creating a new array
StoredState: TypeAlias = State
StoredAction: TypeAlias = np.int32
StoredValue: TypeAlias = np.float32
StoredProbability: TypeAlias = np.float32
StoredReward: TypeAlias = np.int32
StoredDone: TypeAlias = np.bool_
