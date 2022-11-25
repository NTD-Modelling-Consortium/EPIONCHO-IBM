from epioncho_ibm import Params, State, StateStats, TreatmentParams
from epioncho_ibm.state import make_state_from_params
from tests.definitions.auto_benchmarker import AutoBenchmarker


def benchmarker_test_func_no_treat(end_time: float, population: int) -> StateStats:
    params = Params(treatment=None)
    state = make_state_from_params(params=params, n_people=int(population))
    state.run_simulation(start_time=0, end_time=end_time)
    return state.to_stats()


def benchmarker_test_func_treat(end_time: float, population: int) -> StateStats:
    params = Params(
        treatment=TreatmentParams(start_time=0, interval_years=0.01),
    )
    state = make_state_from_params(params=params, n_people=int(population))
    state.run_simulation(start_time=0, end_time=end_time)
    return state.to_stats()


autobenchmarker = AutoBenchmarker(
    no_treatment=benchmarker_test_func_no_treat, treatment=benchmarker_test_func_treat
)
