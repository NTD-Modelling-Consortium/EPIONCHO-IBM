from functools import partial

from plot_outputs import plot_outputs
from tqdm.contrib.concurrent import process_map

from epioncho_ibm.tools import Data, write_data_to_csv


# Function to add model parameters (seed, exp, abr) and MDA history to endgame object
def get_endgame(
    iter,
    iu_name="GHA0216121382",
    sample=True,
    mox_interval=1,
    abr=1641,
    immigration_rate=0,
    new_blackfly=0,
):
    treatment_program = []
    changes = []
    seed = iter + iter * 3758
    gamma_distribution = 0.3
    # TODO: Variable sampling during test
    # inputData = pd.read_csv("test_outputs/inputParams/InputPars_" + iu_name + ".csv")
    # index = random.randint(1, inputData.shape[0])
    # if sample:
    #     seed = inputData.loc[index][0]
    #     gamma_distribution = inputData.loc[index][1]  # 0.3
    #     abr = inputData.loc[index][2]
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
                "treatment_name": "MOX",
                "treatment_interval": mox_interval,
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
                "year_length_days": 366,
                "delta_h_zero": 0.186,
                "c_v": 0.005,
                "delta_h_inf": 0.003,
                "seed": seed,
                "gamma_distribution": gamma_distribution,
                "delta_time_days": 1,
                "immigration_rate": immigration_rate,
                "blackfly": {
                    "bite_rate_per_person_per_year": abr,
                    "immigrated_worm_count": new_blackfly,
                },
                "exposure": {"Q": 1.2},
                "sequela_active": [
                    "Blindness",
                    "SevereItching",
                    "RSD",
                    "APOD",
                    "CPOD",
                    "Atrophy",
                    "HangingGroin",
                    "Depigmentation",
                ],
            },
            "changes": changes,
        },
        "programs": treatment_program,
    }


def run_simulations(
    simulation_func,
    verbose,
    iu_name,
    sample,
    sample_interval,
    mox_interval,
    end_time,
    ranges,
    max_workers,
    desc,
    abr=1641,
    scenario_file="",
):
    rumSim = partial(
        simulation_func,
        verbose=verbose,
        iu_name=iu_name,
        sample=sample,
        samp_interval=sample_interval,
        mox_interval=mox_interval,
        abr=abr,
        end_time=end_time,
    )
    datas: list[tuple[Data, Data]] = process_map(
        rumSim, ranges, max_workers=max_workers
    )
    data: list[Data] = [row[0] for row in datas]
    age_data: list[Data] = [row[1] for row in datas]
    write_data_to_csv(
        data,
        "test_outputs/python_model_output/testing_"
        + iu_name
        + "-new_run_"
        + desc
        + ".csv",
    )
    write_data_to_csv(
        age_data,
        "test_outputs/python_model_output/testing_"
        + iu_name
        + "-new_run_"
        + desc
        + "_age-grouped"
        + ".csv",
    )
    plot_outputs("testing_" + iu_name + "-new_run_" + desc + "_age-grouped")
