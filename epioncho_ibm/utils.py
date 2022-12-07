from typing import TypeVar

import numpy as np
from numpy.random import Generator, SFC64
from numpy.typing import NDArray
from functools import cache

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


class BlockBinomialGenerator:
    def __init__(
        self, 
        prob: float, 
        block_samples = 100
    ) -> None:
        # Regenerate row by row rather than as a block (except initial perhaps)
        # Consider as a list of rows, where each row has shape (block_samples, *shape)
        # Have values for current sample - when sample==block sample regenerate row
        # Excluding the initial generation means we can have this go to higher n
        # with less of a performance hit
        # requires [2,1,2,3,3,3,4]
        # converts to [(2,34),(1,56),(2,35),...]  or [134, 56, 135] in 1D
        # convert in one line - no need for shapes
        # each shape has a certain requirement from each line, and order to require in
        # increment array of required vals [10,5,1,0,0,0...]
        # (trials, blocks)
        self.block_samples = block_samples
        self.prob = prob
        self.array: np.ndarray = np.zeros((1, block_samples))
        self.next_access = np.zeros(1, dtype=int)


    def generate_row(self, n: int):
        existing_rows = self.array.shape[0]
        if n==0:
            pass
        elif n+1 > existing_rows:
            row_to_make_array = np.arange(existing_rows, n+1)
            full_rows_to_make = np.tile(row_to_make_array, (self.block_samples, 1)).T
            new_rows = np.random.binomial(n=full_rows_to_make, p=self.prob)
            self.array = np.concatenate((self.array,new_rows), axis=0)
            self.next_access = np.concatenate((self.next_access, np.zeros_like(row_to_make_array)), axis=0)
        else:
            new_row = np.repeat(n, self.block_samples)
            self.array[n] = np.random.binomial(n=new_row, p=self.prob)
            self.next_access[n] = 0


    def __call__(self, n_array: np.ndarray):
        old_shape = n_array.shape
        flat_n_array = n_array.flatten()
        if len(flat_n_array) > self.block_samples:
            raise ValueError("Too few block samples for array")
        # flatten on way in - reshape on way out
        max_val = np.amax(flat_n_array)
        if max_val+1 > self.array.shape[0]:
            self.generate_row(max_val)

        unique, counts = np.unique(flat_n_array, return_counts=True)

        # In case where requested block is outside sample limit, regenerate row
        last_requested_block = self.next_access[unique] + counts
        rows_to_regen = unique[last_requested_block > self.block_samples]
        for i in rows_to_regen:
            self.generate_row(i)

        part_start = unique*self.block_samples + self.next_access[unique]
        full_indices = np.array([], dtype = int)
        for i, start in enumerate(part_start):
            new_requested_blocks = np.arange(
                start = start, 
                stop = start + counts[i])

            full_indices = np.concatenate((full_indices, new_requested_blocks))

        binoms = np.take(self.array,full_indices)

        sorter = np.argsort(flat_n_array)
        flat_output = binoms[sorter]

        return flat_output.reshape(old_shape)


@cache
def random_generator():
    return Generator(SFC64())

def fast_binomial(n: NDArray[np.int_], p: NDArray[np.float_] | float) -> NDArray[np.int_]:
    return random_generator().binomial(n, p)
