import random
from functools import partial

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import (
    EndgameSimulation,
    _times_of_change,
    endgame_to_params,
)
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


# Function to add model parameters (seed, exp, abr) and MDA history to endgame object
def get_endgame(iu_name="GHA0216121382"):
    treatment_program = []
    changes = []
    inputData = pd.read_csv("examples/inputParams/InputPars_" + iu_name + ".csv")
    index = random.randint(1, inputData.shape[0])
    seed = inputData.loc[index][0]
    gamma_distribution = inputData.loc[index][1]  # 0.3
    abr = inputData.loc[index][2]  # 2297
    if iu_name == "GHA0216121382":
        treatment_program.append(
            {
                "first_year": 1990,
                "last_year": 1996,
                "interventions": {
                    "treatment_interval": 1,
                    "total_population_coverage": 0.52,
                    "correlation": 0.5,
                },
            }
        )
        treatment_program.append(
            {
                "first_year": 1997,
                "last_year": 2017,
                "interventions": {
                    "treatment_interval": 1,
                    "total_population_coverage": 0.65,
                    "correlation": 0.5,
                },
            }
        )
        treatment_program.append(
            {
                "first_year": 2019,
                "last_year": 2019,
                "interventions": {
                    "treatment_interval": 1,
                    "total_population_coverage": 0.65,
                    "correlation": 0.5,
                },
            }
        )
        treatment_program.append(
            {
                "first_year": 2021,
                "last_year": 2025,
                "interventions": {
                    "treatment_interval": 1,
                    "total_population_coverage": 0.65,
                    "correlation": 0.5,
                },
            }
        )
    elif iu_name == "CIV0162715440":
        treatment_program.append(
            {
                "first_year": 1988,
                "last_year": 1996,
                "interventions": {
                    "treatment_interval": 1,
                    "total_population_coverage": 0.52,
                    "correlation": 0.5,
                },
            }
        )
        treatment_program.append(
            {
                "first_year": 1997,
                "last_year": 2000,
                "interventions": {
                    "treatment_interval": 1,
                    "total_population_coverage": 0.65,
                    "correlation": 0.5,
                },
            }
        )
        treatment_program.append(
            {
                "first_year": 2008,
                "last_year": 2025,
                "interventions": {
                    "treatment_interval": 1,
                    "total_population_coverage": 0.65,
                    "correlation": 0.5,
                },
            }
        )
        changes.append(
            {
                "year": 1977,
                "params": {"blackfly": {"bite_rate_per_person_per_year": abr * 0.2}},
            }
        )
        changes.append(
            {
                "year": 1993,
                "params": {"blackfly": {"bite_rate_per_person_per_year": abr}},
            }
        )
    treatment_program.append(
        {
            "first_year": 2026,
            "last_year": 2040,
            "interventions": {
                "treatment_interval": 1,
                "total_population_coverage": 0.65,
                "min_age_of_treatment": 4,
                "correlation": 0.5,
                "microfilaricidal_nu": 0.04,
                "microfilaricidal_omega": 1.82,
                "embryostatic_lambda_max": 462,
                "embryostatic_phi": 4.83,
            },
        }
    )
    changes.append({"year": 2026, "params": {"delta_time_days": 0.5}})

    return {
        "parameters": {
            "initial": {
                "n_people": 400,
                "seed": seed,
                "gamma_distribution": gamma_distribution,
                "delta_time_days": 1,
                "blackfly": {"bite_rate_per_person_per_year": abr},
            },
            # treatment doesn't seem to be occuring in 2026 (i.e the year in which delta_time_days is changed)
            # the problem comes with an edge case that breaks the logic in treatment.py - https://github.com/dreamingspires/EPIONCHO-IBM/blob/master/epioncho_ibm/advance/treatment.py#L36
            # basically we take an additional time step before we actually change the delta time values
            # this means we increment by 1 day, and then change the delta time days to 0.5 within that iteration
            # The logic linked is trying to determine whether treatment would have occured on that time step.
            # This happens whenever the current time is before the treatment stop time AND
            # when we are at the first iteration of a new round of treatment
            # this second part is determined by a) the current time is at or past one of the treatment times AND
            # b) it is only past that treatment time by at most delta_time (delta_time_days / year_length_days)
            # The bug here is that we have taken a 1 day step, but then change delta_time_days to 0.5
            # this that the logic will return false when we should be applying the first round of treatment
            "changes": changes,
        },
        "programs": treatment_program,
    }


# Function to run and save simulations
def run_sim(i, iu_name, verbose=False):
    endgame_structure = get_endgame(iu_name)
    # Read in endgame objects and set up simulation
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)

    endgame_sim = EndgameSimulation(
        start_time=1900, endgame=endgame, verbose=verbose, debug=True
    )
    # Run
    run_data: Data = {}
    for state in endgame_sim.iter_run(end_time=2041, sampling_interval=0.25):

        add_state_to_run_data(
            state,
            run_data=run_data,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=False,
            prevalence=True,
            mean_worm_burden=False,
            prevalence_OAE=False,
            intensity=False,
            with_sequela=False,
        )

    return run_data


# Wrapper
def wrapped_parameters(iu_name):
    # Run simulations and save output
    num_iter = 1  # 200
    max_workers = 10 if num_iter > 10 else num_iter
    rumSim = partial(run_sim, verbose=False, iu_name=iu_name)
    data = process_map(rumSim, range(num_iter), max_workers=max_workers)
    write_data_to_csv(
        data,
        "test_outputs/testing_" + iu_name + "-age_grouped_raw_data2.csv",
    )
    df = pd.read_csv("test_outputs/testing_" + iu_name + "-age_grouped_raw_data2.csv")
    df2 = df[
        ((df["year_id"] >= 2025) & (df["year_id"] <= 2040))
        & (df["measure"] == "prevalence")
    ]
    print(np.mean(np.array(df2.iloc[:, 4:]).astype(float), axis=1))


if __name__ == "__main__":
    # Run example
    # iu = "GHA0216121382"
    iu = "CIV0162715440"
    wrapped_parameters(iu)
