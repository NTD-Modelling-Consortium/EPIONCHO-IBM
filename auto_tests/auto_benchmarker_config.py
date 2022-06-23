from auto_tests.definitions.auto_benchmarker import AutoBenchmarker
from epioncho_ibm import Params, RandomConfig, State, StateStats, run_simulation


def benchmarker_test_func_no_treat(end_time: float, population: int) -> StateStats:
    params = Params(human_population=population)
    random_config = RandomConfig()
    initial_state = State.generate_random(random_config=random_config, params=params)
    initial_state.dist_population_age(num_iter=15000)
    new_state = run_simulation(initial_state, start_time=0, end_time=end_time)
    stats = new_state.to_stats()
    return stats


def benchmarker_test_func_treat(end_time: float, population: int) -> StateStats:
    params = Params(
        human_population=population, treatment_start_time=0, treatment_interval_yrs=0.01
    )
    random_config = RandomConfig()
    initial_state = State.generate_random(random_config=random_config, params=params)
    initial_state.dist_population_age(num_iter=15000)
    new_state = run_simulation(initial_state, start_time=0, end_time=end_time)
    stats = new_state.to_stats()
    return stats


autobenchmarker = AutoBenchmarker(
    no_treatment=benchmarker_test_func_no_treat, treatment=benchmarker_test_func_treat
)
