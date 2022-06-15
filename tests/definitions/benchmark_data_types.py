from typing import Dict, List, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from epioncho_ibm import Params


class BenchmarkArray(BaseModel):
    mean: float
    st_dev: float

    @classmethod
    def from_array(cls, array: Union[NDArray[np.float_], NDArray[np.int_]]):
        return cls(mean=np.mean(array), st_dev=np.std(array))


class NTDSettings(BaseModel):
    min_year: float
    max_year: float
    year_steps: int
    min_pop: int
    max_pop: int
    pop_steps: int
    max_pop_years: float
    benchmark_iters: int = 1


class OutputData(BaseModel):
    end_year: float
    params: Params
    people: Dict[str, BenchmarkArray]


class TestData(BaseModel):
    tests: List[OutputData]
