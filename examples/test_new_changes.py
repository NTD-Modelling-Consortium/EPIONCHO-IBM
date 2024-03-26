import math
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


# Function to add model parameters (seed, exp, abr) and MDA history to endgame object
def get_endgame(iter, iu_name="GHA0216121382", sample=True, mox_interval=1):
    treatment_program = []
    changes = []
    seed = iter + iter * 3758
    gamma_distribution = 0.31
    abr = 1641
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
                "blackfly": {"bite_rate_per_person_per_year": abr},
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


# Function to run and save simulations
def run_sim(i, iu_name, verbose=False, sample=True, samp_interval=1, mox_interval=1):
    endgame_structure = get_endgame(
        i, iu_name, sample=sample, mox_interval=mox_interval
    )
    # Read in endgame objects and set up simulation
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)
    # print(endgame)
    endgame_sim = EndgameSimulation(
        start_time=1900, endgame=endgame, verbose=verbose, debug=True
    )
    # Run
    run_data: Data = {}
    run_data_age: Data = {}
    for state in endgame_sim.iter_run(end_time=2041, sampling_interval=samp_interval):

        add_state_to_run_data(
            state,
            run_data=run_data,
            number=True,
            n_treatments=True,
            achieved_coverage=False,
            with_age_groups=False,
            prevalence=True,
            mean_worm_burden=False,
            prevalence_OAE=False,
            intensity=True,
            with_sequela=True,
            with_pnc=True,
            saving_multiple_states=True,
        )
        add_state_to_run_data(
            state,
            run_data=run_data_age,
            number=True,
            n_treatments=True,
            achieved_coverage=False,
            with_age_groups=True,
            prevalence=True,
            mean_worm_burden=True,
            prevalence_OAE=True,
            intensity=True,
            with_sequela=True,
            with_pnc=True,
        )

    return (run_data, run_data_age)


def plot_outputs(file_name):
    df = pd.read_csv("test_outputs/python_model_output/" + file_name + ".csv")
    for measure in df["measure"].unique():
        measure_columns = "_" + measure
        filtered_df = df[((df["measure"] == measure))]
        numbers = df[((df["measure"] == "number"))]

        if measure == "number":
            measure_columns = "_numbers_2"
        new_combined_df = pd.merge(
            filtered_df,
            numbers,
            on=["year_id", "age_start", "age_end"],
            suffixes=(measure_columns, "_number"),
        )
        for col in df.columns:
            if col.startswith("draw"):
                new_combined_df[col] = (
                    new_combined_df[col + measure_columns]
                    * new_combined_df[col + "_number"]
                )
        new_combined_df = new_combined_df.drop(
            columns=[
                col + suffix
                for col in df.columns
                if col.startswith("draw")
                for suffix in [measure_columns]
            ]
        )

        age_grouped_df = new_combined_df.copy()

        for col in df.columns:
            if col.startswith("draw"):
                age_grouped_df[col] = (
                    age_grouped_df[col] / age_grouped_df[col + "_number"]
                )
        age_grouped_df = age_grouped_df.drop(
            columns=[
                col + suffix
                for col in df.columns
                if col.startswith("draw")
                for suffix in ["_number"]
            ]
        )

        age_groups = [
            str(x) + "_" + str(y)
            for x, y in zip(age_grouped_df["age_start"], age_grouped_df["age_end"])
        ]
        years = age_grouped_df["year_id"].values
        measure_values = age_grouped_df.iloc[:, 5:]

        new_combined_df = (
            new_combined_df.drop(
                columns=[
                    "age_start",
                    "age_end",
                    "measure" + measure_columns,
                    "measure_number",
                ]
            )
            .groupby("year_id")
            .sum()
            .reset_index()
        )

        for col in df.columns:
            if col.startswith("draw"):
                new_combined_df[col] = (
                    new_combined_df[col] / new_combined_df[col + "_number"]
                )
        new_combined_df = new_combined_df.drop(
            columns=[
                col + suffix
                for col in df.columns
                if col.startswith("draw")
                for suffix in ["_number"]
            ]
        )

        years_2 = new_combined_df["year_id"]
        measure_values_2 = new_combined_df.iloc[:, 1:]
        age_groups_2 = np.full(len(years_2), "0_80")

        def create_graphs(newDf, newDf_2, print_num):
            index = 0
            save_file_name = file_name
            square = math.floor(math.sqrt(len(newDf["age_groups"].unique()) + 1))
            fig, ax1 = plt.subplots(
                square, square + 1, figsize=(15, 15), sharex=True, sharey=True
            )
            ax1 = ax1.flatten()
            for age_group in newDf["age_groups"].unique():
                ax = ax1[index]
                group_data = newDf[newDf["age_groups"] == age_group]
                ax.plot(group_data["years"], group_data["measure"], label=age_group)
                ax.vlines(
                    x=1988, color="red", ymin=0, ymax=np.max(group_data["measure"])
                )
                ax.set_xlim(left=1950)
                ax.set_title(age_group)
                if index % (square + 1) == 0:
                    ax.set_ylabel(ylab)
                index += 1
            ax1[index].plot(
                newDf_2["years"],
                newDf_2["measure"],
                label=newDf_2["age_groups"].unique(),
            )
            ax1[index].vlines(
                x=1988, color="red", ymin=0, ymax=np.max(newDf_2["measure"])
            )
            ax1[index].set_xlim(left=1950)
            ax1[index].set_title(newDf_2["age_groups"].unique())
            for i in range(index + 1, len(ax1)):
                ax1[i].tick_params(
                    left=False, labelleft=False, bottom=False, labelbottom=False
                )
            plt.savefig(
                "test_outputs/" + measure + "_" + save_file_name + ".png",
                dpi=300,
            )
            plt.close(fig)

        newDf = pd.DataFrame(
            {
                "years": years,
                "measure": measure_values.mean(axis=1, skipna=True).tolist(),
                "age_groups": age_groups,
            }
        )

        newDf_2 = pd.DataFrame(
            {
                "years": years_2,
                "measure": measure_values_2.mean(axis=1, skipna=True).tolist(),
                "age_groups": age_groups_2,
            }
        )
        ylab = measure + " Prevalence"
        if measure == "number":
            ylab = "Population Count"
        create_graphs(newDf, newDf_2, print_num=False)


def run_simulations(
    verbose, iu_name, sample, sample_interval, mox_interval, ranges, max_workers, desc
):
    rumSim = partial(
        run_sim,
        verbose=verbose,
        iu_name=iu_name,
        sample=sample,
        samp_interval=sample_interval,
        mox_interval=mox_interval,
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
    # write_data_to_csv(
    #     age_data,
    #     "test_outputs/python_model_output/testing_"
    #     + iu_name
    #     + "-new_run_"
    #     + desc
    #     + "_age-grouped"
    #     + ".csv",
    # )
    # plot_outputs("testing_" + iu_name + "-new_run_" + desc + "_age-grouped")


# Wrapper
def wrapped_parameters(iu_name):
    # Run simulations and save output
    num_iter = 100
    max_workers = os.cpu_count() if num_iter > os.cpu_count() else num_iter
    run_simulations(
        False, iu_name, True, 1, 1, range(num_iter), max_workers, "mox_annual_1year"
    )

    run_simulations(
        False, iu_name, True, 1, 0.5, range(num_iter), max_workers, "mox_biannual_1year"
    )

    run_simulations(
        False,
        iu_name,
        True,
        1,
        0.25,
        range(num_iter),
        max_workers,
        "mox_quadannual_1year",
    )


if __name__ == "__main__":
    # Run example
    # iu = "GHA0216121382"
    iu = "CIV0162715440"
    wrapped_parameters(iu)
