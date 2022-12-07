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


def _get_indices(unique, counts, block_samples, next_access):
    part_start = unique*block_samples + next_access[unique]
    full_indices = np.array([], dtype = int)
    for i, start in enumerate(part_start):
        new_requested_blocks = np.arange(
            start = start, 
            stop = start + counts[i])

        full_indices = np.concatenate((full_indices, new_requested_blocks))
    return full_indices

class BlockBinomialGenerator:
    block_samples: int
    prob: float
    array: NDArray[np.int_] # (n_trials, block_samples)
    next_access: NDArray[np.int_]
    generator: Generator
    def __init__(
        self, 
        prob: float, 
        block_samples = 1000
    ) -> None:
        """
        Create block binomial generator. Creates rows of each trial n previously requested, and
        regenerates in blocks of block samples when each has been used up.

        Args:
            prob (float): The fixed probability
            block_samples (int, optional): The number of block samples. Defaults to 1000.
        """
        self.block_samples = block_samples
        self.prob = prob
        self.array = np.zeros((1, block_samples), dtype = int)
        self.next_access = np.zeros(1, dtype=int)
        self.generator = Generator(SFC64())


    def generate_row(self, n: int):
        """
        Regenerates a single row of the stored array, or if required, all rows up to
        specified row that are not already generated.

        Args:
            n (int): The number of row required.
        """
        existing_rows = self.array.shape[0]
        if n==0:
            pass
        elif n+1 > existing_rows:
            row_to_make_array = np.arange(existing_rows, n+1)
            full_rows_to_make = np.tile(row_to_make_array, (self.block_samples, 1)).T
            new_rows = self.generator.binomial(n=full_rows_to_make, p=self.prob)
            self.array = np.concatenate((self.array,new_rows), axis=0)
            self.next_access = np.concatenate((self.next_access, np.zeros_like(row_to_make_array)), axis=0)
        else:
            new_row = np.repeat(n, self.block_samples)
            self.array[n] = np.random.binomial(n=new_row, p=self.prob)
            self.next_access[n] = 0


    def __call__(self, n_array: np.ndarray) -> np.ndarray:
        """
        Generate Binomial array for fixed probability

        Args:
            n_array (np.ndarray): Array of n trials

        Raises:
            ValueError: Block samples must exceed array element number

        Returns:
            np.ndarray: binomial array for all values
        """
        old_shape = n_array.shape
        
        # Flatten on way in - reshape on way out
        flat_n_array = n_array.flatten()
        if len(flat_n_array) > self.block_samples:
            raise ValueError("Too few block samples for array")
        
        # Generate up to max value in array
        max_val = np.amax(flat_n_array)
        if max_val+1 > self.array.shape[0]:
            self.generate_row(max_val)

        unique, counts = np.unique(flat_n_array, return_counts=True)

        # In case where requested block is outside sample limit, regenerate row
        last_requested_block = self.next_access[unique] + counts
        rows_to_regen = unique[last_requested_block > self.block_samples]
        for i in rows_to_regen:
            self.generate_row(i)

        full_indices = _get_indices(unique, counts, self.block_samples, self.next_access)
        binoms = np.take(self.array,full_indices)

        sorter = np.argsort(flat_n_array)
        flat_output = binoms[sorter]

        return flat_output.reshape(old_shape)


@cache
def random_generator():
    return Generator(SFC64())

def fast_binomial(n: NDArray[np.int_], p: NDArray[np.float_] | float) -> NDArray[np.int_]:
    return random_generator().binomial(n, p)
