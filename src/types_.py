import numpy as np
from numpy.typing import NDArray

type State = NDArray[np.float32]
type Action = np.int32
type Value = np.float32
type Probability = np.float32
type Reward = np.float32
type Done = np.bool