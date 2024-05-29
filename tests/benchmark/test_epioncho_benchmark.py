import json

import constants

from epioncho_ibm import Params, Simulation, TreatmentParams


# select the set of output data that corresponds with the end_time and population being tested
def find_benchmark_output(type, end_time, population):
    with open(constants.BENCHMARK_PATH + "/" + constants.BENCHMARK_FILE_NAME, "r") as f:
        return list(
            filter(
                lambda d: d["end_time"] == end_time and d["population"] == population,
                json.load(f)["tests"][type],
            )
        )[0]["data"]


# helper function to compare outputs between the expected mean/st. deviation and the simulated mean
def compare_outputs(test_run_stats, benchmark_stats):
    acceptable_st_devs = constants.ACCEPTABLE_ST_DEVS
    for key in test_run_stats.__dict__:
        value = test_run_stats.__dict__[key]
        if key not in benchmark_stats:
            raise RuntimeError(f"Key {key} not present in benchmark")
        mean = benchmark_stats[key]["mean"]
        st_dev = benchmark_stats[key]["st_dev"]
        lb = mean - (st_dev * acceptable_st_devs)
        ub = mean + (st_dev * acceptable_st_devs)
        if value < lb:
            raise ValueError(
                f"For key: {key} benchmark lower bound: {lb} surpassed by value {value}"
            )
        if value > ub:
            raise ValueError(
                f"For key: {key} benchmark upper bound: {ub} surpassed by value {value}"
            )


# testing the model if no treatment is applied
def test_no_treatment(end_time: float, population: int):
    benchmark_output = find_benchmark_output("no_treatment", end_time, population)
    params = Params(treatment=None, n_people=population)
    simulation = Simulation(start_time=0, params=params)
    simulation.run(end_time=end_time)
    compare_outputs(simulation.state.stats(), benchmark_output)


# testing the model if default treatment is applied
def test_treatment(end_time: float, population: int):
    benchmark_output = find_benchmark_output("treatment", end_time, population)
    params = Params(
        treatment=TreatmentParams(
            start_time=constants.TREAT_START,
            interval_years=constants.TREAT_INTERVAL,
            stop_time=constants.TREAT_END,
        ),
        n_people=population,
    )
    simulation = Simulation(start_time=0, params=params)
    simulation.run(end_time=end_time)
    compare_outputs(simulation.state.stats(), benchmark_output)
