from typing import TypeVar

import numpy as np
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
