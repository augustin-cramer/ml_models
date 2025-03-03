import numpy as np
from typing import Callable, Dict


def entropy(p: np.ndarray) -> float:
    return -np.sum(p * np.log(p))

IMPURITY_FNS: Dict[str, Callable[[np.ndarray], float]] = {"entropy": entropy}
