import json
import os
from functools import partial

import constants
import numpy as np
from tqdm.contrib.concurrent import process_map

from epioncho_ibm import Params, Simulation, TreatmentParams


def runSims(i, end_time, population, treatment=None):
    params = Params(treatment=treatment, n_people=population)
    simulation = Simulation(start_time=0, params=params)
    simulation.run(end_time=end_time)
    stats = simulation.state.stats().__dict__
    return {"data": stats}


def runSimsParallel(end_time, population, num_runs, treatment=None):
    final_dict = {
        "datas": [],
        "end_time": end_time,
        "population": population,
    }
    max_workers = os.cpu_count() if num_runs > os.cpu_count() else num_runs
    all_data = process_map(
        partial(runSims, end_time=end_time, population=population, treatment=treatment),
        range(num_runs),
        max_workers=max_workers,
    )
    final_dict["datas"] = all_data
    return final_dict


def calc_std(data):
    output = {"tests": {}}
    for sim_key in data["tests"].keys():
        all_sims = data["tests"][sim_key]
        output["tests"][sim_key] = []
        for sim in all_sims:
            sim_summary_dict = {
                "data": {},
                "end_time": sim["end_time"],
                "population": sim["population"],
            }
            all_vals_dict = {}
            for run in sim["datas"]:
                for key in run["data"]:
                    value = run["data"][key]
                    if type(value) == dict:
                        value = value["mean"]
                    if key in all_vals_dict:
                        all_vals_dict[key].append(value)
                    else:
                        all_vals_dict[key] = [value]
            for key in all_vals_dict:
                mean = np.mean(all_vals_dict[key])
                std = np.std(all_vals_dict[key])
                sim_summary_dict["data"][key] = {"mean": mean, "st_dev": std}
            output["tests"][sim_key].append(sim_summary_dict)
    return output


if __name__ == "__main__":
    final_benchmark_dict = {
        "tests": {
            "no_treatment": [],
            "treatment": [],
        }
    }

    for end_time in constants.END_TIMES:
        for n in constants.POPULATIONS:
            if end_time * n >= 100:
                continue
            print("Testing " + str(end_time) + " " + str(n))
            final_benchmark_dict["tests"]["no_treatment"].append(
                runSimsParallel(end_time, n, constants.BENCHMARK_ITERS, None)
            )
            final_benchmark_dict["tests"]["treatment"].append(
                runSimsParallel(
                    end_time,
                    n,
                    constants.BENCHMARK_ITERS,
                    TreatmentParams(
                        start_time=constants.TREAT_START,
                        interval_years=constants.TREAT_INTERVAL,
                        stop_time=constants.TREAT_END,
                    ),
                )
            )

    with open(constants.BENCHMARK_PATH + "/" + constants.BENCHMARK_FILE_NAME, "w") as d:
        d.write(json.dumps(calc_std(final_benchmark_dict), indent=4))
