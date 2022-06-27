version = 1.0

from .params import (
    BlackflyParams,
    ExposureParams,
    HumanParams,
    MicrofilParams,
    Params,
    TreatmentParams,
    WormParams,
)
from .state import State, StateStats

# from enum import Enum

# class ParamSet(Enum):
#    param_set1 = "param_set1"
#    param_set2 = "param_set2"


def benchmarker_test_func(end_time: float, population: int) -> StateStats:
    state = State(params=Params(), n_people=population)
    state.run_simulation(start_time=0, end_time=end_time)
    return state.to_stats()
