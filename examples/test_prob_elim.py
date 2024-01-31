import os
import random

import numpy as np
import pandas as pd

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from functools import partial
from multiprocessing import cpu_count

from tqdm.contrib.concurrent import process_map

from epioncho_ibm import Params, Simulation
from epioncho_ibm.state.params import (
    BlackflyParams,
    EpionchoEndgameModel,
    TreatmentParams,
)
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv
from examples.post_processing_function import post_processing_calculation

"""
{
    "parameters": {
        "initial": {
            "n_people": 400
        },
        "changes": []
    },
    "programs": [
        {
            "first_year": 1990,
            "last_year": 2025,
            "interventions": {
                "treatment_interval": 0.5,
                "correlation": 0.5,
            }
        },
        {
            "first_year": 2026,
            "last_year": 2040,
            "interventions": {
                "treatment_interval": 1,
                "min_age_of_treatment": 4,
                "correlation": 0.5,
                "microfilaricidal_nu": 0.04,
                "microfilaricidal_omega": 1.82,
                "embryostatic_lambda_max": 462,
                "embryostatic_phi": 4.83
            }
        }
    ]
}
"""


def run_sim(
    i,
    start_time=1950,
    mda_start=2000,
    mda_stop=2020,
    simulation_stop=2050,
    abr=2297,
    verbose=True,
    gamma_distribution=0.3,
    seed=None,
    mda_interval=1,
) -> tuple[Data, Data]:
    inputData = pd.read_csv("examples/InputPars_GHA0216121382.csv")
    index = random.randint(1, inputData.shape[0])
    seed = inputData.loc[index][0]
    gamma_distribution = inputData.loc[index][1]
    abr = inputData.loc[index][2]
    # print("ABR" +str(abr) + "kE" + str(kE) + "seed" + str(seed))
    params = Params(
        delta_time_days=1,
        year_length_days=365,
        n_people=400,
        blackfly=BlackflyParams(
            delta_h_zero=0.186,
            delta_h_inf=0.003,
            c_h=0.005,
            bite_rate_per_person_per_year=abr,
            gonotrophic_cycle_length=0.0096,
        ),
        gamma_distribution=gamma_distribution,
        seed=seed,
        sequela_active=[
            "HangingGroin",
            "Atrophy",
            "Blindness",
            "APOD",
            "CPOD",
            "RSD",
            "Depigmentation",
            "SevereItching",
        ],
        treatment=TreatmentParams(
            interval_years=mda_interval,
            start_time=mda_start,
            stop_time=mda_stop,
        ),
    )

    EpionchoEndgameModel.parse_raw
    simulation = Simulation(start_time=start_time, params=params, verbose=verbose)

    # Regression Testing - 70 iters
    # T1- Test it without any additional min/max age                                   | NA
    # T2- Test it changing the min+max age + age grouped for all ages + age grouped    | Passed
    # "age_groups": [[0,5], [5, 10], [10, 20], [20, 30], [30, 40], [40,50], [50,80]],
    #                 "prevalence":{
    #                     "age_min": 5,
    #                     "age_max": 80,
    #                 },
    # T3- Test it changing the specific min+max age for all ages                       | Passed
    # "prevalence":{
    #     "age_min": 5,
    #     "age_max": 80,
    # },
    # T4- Test it changing just the min age for all ages + age grouped                 | In-Progress
    # age_min=7
    # T5- Test it changing just the max age for all ages + age grouped                 | NA

    # Adding an additional object to store all age data
    age_grouped_run_data: Data = {}
    all_age_run_data: Data = {}
    for state in simulation.iter_run(
        end_time=simulation_stop,
        sampling_interval=0.5,
        # sampling_years=[i for i in range(mda_start, simulation_stop)],
    ):
        add_state_to_run_data(
            state,
            run_data=age_grouped_run_data,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=True,
            prevalence=True,
            mean_worm_burden=False,
            intensity=True,
        )
        add_state_to_run_data(
            state,
            run_data=all_age_run_data,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=False,
            prevalence=True,
            mean_worm_burden=False,
            intensity=True,
        )

    return (age_grouped_run_data, all_age_run_data)


run_iters = 10

if __name__ == "__main__":
    cpus_to_use = cpu_count() - 4

    # mda_start = 2000
    # mda_stop = 2040
    loopTimes = [(1990, 2025)]  # , (2000, 2020), (2000, 2030), (2000, 2040)]
    interval = 1
    for mda_start, mda_stop in loopTimes:
        # ~ 70% MFP
        rumSim = partial(
            run_sim,
            start_time=1900,
            mda_start=mda_start,
            mda_stop=mda_stop,
            simulation_stop=2025,
            verbose=False,
            mda_interval=1,
        )
        # Seperating the return data into two data arrays for separate processing
        # Look at the run_sim function definition/return value to see that the expected output is a tuple,
        # with index [0] containing the age_grouped data, and index [1] containing the all age data
        data: list[tuple[Data, Data]] = process_map(
            rumSim, range(run_iters), max_workers=cpus_to_use
        )

        age_grouped_data: list[Data] = [row[0] for row in data]
        all_age_data: list[Data] = [row[1] for row in data]

        write_data_to_csv(
            age_grouped_data,
            "test_outputs/mda-stop-" + str(mda_stop) + "-age_grouped_raw_data.csv",
        )
        write_data_to_csv(
            all_age_data,
            "test_outputs/mda-stop-" + str(mda_stop) + "-raw_all_age_data.csv",
        )
        post_processing_calculation(
            all_age_data,
            "test",
            "test-scenario",
            "test_outputs/mda-stop-" + str(mda_stop) + "-all_age_data.csv",
            mda_start,
            mda_stop,
            interval,
        )
