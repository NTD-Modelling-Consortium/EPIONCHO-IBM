import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from epioncho_ibm import Params, RandomConfig, State, run_simulation
from epioncho_ibm.state import NumericArrayStat, PeopleStats
from tests.benchmark_data_types import OutputData, TestData
from tests.utils import FlatDict, flatten_dict


class NTDSettings(BaseModel):
    min_year: float
    max_year: float
    year_steps: int
    min_pop: int
    max_pop: int
    pop_steps: int
    max_pop_years: float
    benchmark_iters: int = 1


def get_test_pairs(settings: NTDSettings) -> List[Tuple[float, int]]:
    def get_exponentially_spaced_steps(
        start: Union[int, float], end: Union[int, float], n_steps: int
    ) -> NDArray[np.float_]:
        log_start = math.log(start)
        log_end = math.log(end)
        exp_spaced = np.linspace(log_start, log_end, n_steps)
        return np.exp(exp_spaced)

    pop_spaces: NDArray[np.int_] = np.round(
        get_exponentially_spaced_steps(
            settings.min_pop, settings.max_pop, settings.pop_steps
        )
    )
    year_spaces = get_exponentially_spaced_steps(
        settings.min_year, settings.max_year, settings.year_steps
    )
    year_cols = year_spaces[:, None]
    pop_years = year_cols * pop_spaces
    valid_tests = pop_years < settings.max_pop_years
    coords = valid_tests.nonzero()
    years_for_test = year_spaces[coords[0]]
    pops_for_test = pop_spaces[coords[1]]
    return list(zip(years_for_test, pops_for_test))


def run_stochastic_test(end_time: float, params: Params) -> PeopleStats:
    random_config = RandomConfig()
    initial_state = State.generate_random(random_config=random_config, params=params)
    initial_state.dist_population_age(num_iter=15000)
    new_state = run_simulation(initial_state, start_time=0, end_time=end_time)
    stats = new_state.people.to_stats()
    return stats


def compute_mean_and_st_dev_of_pydantic(
    input_stats: List[PeopleStats],
) -> Dict[str, NumericArrayStat]:
    flat_dicts: List[FlatDict] = [
        flatten_dict(input_stat.dict()) for input_stat in input_stats
    ]
    dict_of_arrays: Dict[str, List[Any]] = {}
    for flat_dict in flat_dicts:
        for k, v in flat_dict.items():
            if k in dict_of_arrays:
                dict_of_arrays[k].append(v)
            else:
                dict_of_arrays[k] = [v]
    final_dict_of_arrays: Dict[str, NDArray[np.float_]] = {
        k: np.array(v) for k, v in dict_of_arrays.items()
    }
    return {k: NumericArrayStat.from_array(v) for k, v in final_dict_of_arrays.items()}


settings_path = Path("ntd_settings.json")
settings_model = NTDSettings.parse_file(settings_path)

test_pairs = get_test_pairs(settings_model)

print(f"Benchmark will run {len(test_pairs)} tests")

tests: List[OutputData] = []
for end_year, population in test_pairs:
    params = Params(human_population=population)

    list_of_stats: List[PeopleStats] = []
    for i in range(settings_model.benchmark_iters):
        stats = run_stochastic_test(end_year, params)
        list_of_stats.append(stats)
    people = compute_mean_and_st_dev_of_pydantic(list_of_stats)
    test_output = OutputData(end_year=end_year, params=params, people=people)
    tests.append(test_output)
test_data = TestData(tests=tests)
benchmark_file_path = Path("benchmark.json")
benchmark_file = open(benchmark_file_path, "w+")
json.dump(test_data.dict(), benchmark_file, indent=2)
