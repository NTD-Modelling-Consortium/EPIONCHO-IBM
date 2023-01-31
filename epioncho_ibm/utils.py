from functools import cache
from typing import Optional, TypeVar

import numpy as np
from numpy.random import SFC64, Generator
from numpy.typing import NDArray

DType = TypeVar("DType", bound=np.generic)


def lag_array(fill: NDArray[DType], arr: NDArray[DType], n: int = 1) -> NDArray[DType]:
    """Lag array by `n` and back-stack with `fill`

    Args:
        fill (NDArray): Array to be filled at the beginning
        arr (NDArray): Array to be shifted/lagged
        n (int, optional): lag, must po positive. Defaults to 1.

    Returns:
        NDArray: shifted array
    """
    assert n > 0
    return np.vstack((fill, arr[:-n]))


def array_fully_equal(a1: NDArray[DType], a2: NDArray[DType]):
    return np.array_equal(a1, a2, equal_nan=True)


@cache
def random_generator(seed: Optional[int]):
    return Generator(SFC64(seed=seed))


def fast_binomial(
    n: NDArray[np.int_], p: NDArray[np.float_] | float, seed: Optional[int]
) -> NDArray[np.int_]:
    return random_generator(seed=seed).binomial(n, p)
