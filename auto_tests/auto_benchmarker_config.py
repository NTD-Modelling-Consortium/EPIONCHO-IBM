from auto_tests.definitions.auto_benchmarker import AutoBenchmarker
from epioncho_ibm import HumanParams, Params, State, StateStats, TreatmentParams


def benchmarker_test_func_no_treat(end_time: float, population: int) -> StateStats:
    params = Params(humans=HumanParams(human_population=population))
    state = State(params=params)
    state.dist_population_age(num_iter=15000)
    state.run_simulation(start_time=0, end_time=end_time)
    stats = state.to_stats()
    return stats


def benchmarker_test_func_treat(end_time: float, population: int) -> StateStats:
    params = Params(
        humans=HumanParams(human_population=population),
        treatment=TreatmentParams(start_time=0, interval_years=0.01),
    )
    state = State(params=params)
    state.dist_population_age(num_iter=15000)
    state.run_simulation(start_time=0, end_time=end_time)
    stats = state.to_stats()
    return stats


autobenchmarker = AutoBenchmarker(
    no_treatment=benchmarker_test_func_no_treat, treatment=benchmarker_test_func_treat
)
