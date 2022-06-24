version = 1.0
from epioncho_ibm.run_simulation import run_simulation

from .params import HumanParams, Params
from .state import RandomConfig, State, StateStats

# from enum import Enum

# class ParamSet(Enum):
#    param_set1 = "param_set1"
#    param_set2 = "param_set2"


def benchmarker_test_func(end_time: float, population: int) -> StateStats:
    params = Params(humans=HumanParams(human_population=population))
    random_config = RandomConfig()
    initial_state = State.generate_random(random_config=random_config, params=params)
    initial_state.dist_population_age(num_iter=15000)
    new_state = run_simulation(initial_state, start_time=0, end_time=end_time)
    stats = new_state.to_stats()
    return stats
