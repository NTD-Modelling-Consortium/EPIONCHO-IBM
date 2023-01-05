from pytest_trust_random import TrustRandomConfig, benchmark_test

from epioncho_ibm import Params, StateStats, TreatmentParams
from epioncho_ibm.state import make_state_from_params

pytest_config = TrustRandomConfig(
    acceptable_st_devs=2.5,
    re_runs=5,
    benchmark_path=f"benchmark",
)


@benchmark_test(pytest_config)
def no_treatment(end_time: float, population: int) -> StateStats:
    params = Params(treatment=None)
    state = make_state_from_params(params=params, n_people=int(population))
    state.run_simulation(start_time=0, end_time=end_time)
    return state.to_stats()


@benchmark_test(pytest_config)
def treatment(end_time: float, population: int) -> StateStats:
    params = Params(
        treatment=TreatmentParams(start_time=0, interval_years=0.01),
    )
    state = make_state_from_params(params=params, n_people=int(population))
    state.run_simulation(start_time=0, end_time=end_time)
    return state.to_stats()
