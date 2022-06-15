import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from epioncho_ibm import Params, RandomConfig, State, run_simulation
from epioncho_ibm.state import PeopleStats
from tests.definitions.benchmark_data_types import (
    BenchmarkArray,
    NTDSettings,
    OutputData,
    TestData,
)
from tests.definitions.pytest_config import PytestConfig
from tests.definitions.utils import FlatDict, flatten_dict


def get_test_pairs(settings: NTDSettings) -> Tuple[List[Tuple[float, int]], float]:
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
    total_pop_years = np.sum(pop_years[valid_tests])
    return list(zip(years_for_test, pops_for_test)), total_pop_years


def run_stochastic_test(end_time: float, params: Params) -> PeopleStats:
    random_config = RandomConfig()
    initial_state = State.generate_random(random_config=random_config, params=params)
    initial_state.dist_population_age(num_iter=15000)
    new_state = run_simulation(initial_state, start_time=0, end_time=end_time)
    stats = new_state.people.to_stats()
    return stats


def compute_mean_and_st_dev_of_pydantic(
    input_stats: List[PeopleStats],
) -> Dict[str, BenchmarkArray]:
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
    return {k: BenchmarkArray.from_array(v) for k, v in final_dict_of_arrays.items()}


def generate_settings_file_and_model(
    settings_folder: Path, settings_path: Path
) -> NTDSettings:
    while True:
        answer = input(
            "Settings path does not exist - would you like to make a settings file? (y/n): "
        )
        if answer == "y":
            break
        elif answer == "n":
            raise ValueError("Settings file not found or created")
        else:
            continue

    while True:
        constraints = input(
            "Please enter your end year constraints, comma separated (min_year, max_year, year_steps): "
        )
        items = constraints.split(",")
        if len(items) == 3:
            try:
                min_year = float(items[0])
                max_year = float(items[1])
                year_steps = int(items[2])
                if max_year < min_year:
                    continue
                break
            except ValueError:
                continue
        continue

    while True:
        constraints = input(
            "Please enter population constraints, comma separated (min_pop, max_pop, pop_steps): "
        )
        items = constraints.split(",")
        if len(items) == 3:
            try:
                min_pop = int(items[0])
                max_pop = int(items[1])
                pop_steps = int(items[2])
                if max_pop < min_pop:
                    continue
                break
            except ValueError:
                continue
        continue

    while True:
        pop_years_string = input("Please enter the max population-years: ")
        try:
            pop_years = float(pop_years_string)
            break
        except ValueError:
            continue

    while True:
        bench_iters_string = input("Please enter the number of benchmark iterations: ")
        try:
            bench_iters = int(bench_iters_string)
            break
        except ValueError:
            continue
    model = NTDSettings(
        min_year=min_year,
        max_year=max_year,
        year_steps=year_steps,
        min_pop=min_pop,
        max_pop=max_pop,
        pop_steps=pop_steps,
        max_pop_years=pop_years,
        benchmark_iters=bench_iters,
    )
    settings_folder.mkdir(parents=True)
    settings_file = open(settings_path, "w+")
    json.dump(model.dict(), settings_file, indent=2)
    return model


pytest_config = PytestConfig.parse_file("pytest_config.json")
settings_folder = Path(pytest_config.benchmark_path)
settings_path = Path(str(settings_folder) + os.sep + "settings.json")
if not settings_path.exists():
    settings_model = generate_settings_file_and_model(settings_folder, settings_path)
else:
    settings_model = NTDSettings.parse_file(settings_path)

test_pairs, total_pop_years = get_test_pairs(settings_model)
est_base_time = 0.46
est_test_time = est_base_time * total_pop_years
est_benchmark_time = est_test_time * settings_model.benchmark_iters
print(f"Benchmark will run {len(test_pairs)} tests")
print(f"Estimated benchmark calc time: {est_benchmark_time}")
print(f"Estimated test time (no reruns): {est_test_time}")

start = time.time()
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
end = time.time()
print(f"Benchmark calculated in: {end-start}")
test_data = TestData(tests=tests)
benchmark_file_path = Path(str(settings_folder) + os.sep + "benchmark.json")
benchmark_file = open(benchmark_file_path, "w+")
json.dump(test_data.dict(), benchmark_file, indent=2)
