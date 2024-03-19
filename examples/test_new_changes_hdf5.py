import math
import os
from functools import partial

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import EndgameSimulation, endgame_to_params
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


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


# Function to add model parameters (seed, exp, abr) and MDA history to endgame object
def get_endgame(mox_interval=1):
    treatment_program = []
    changes = []
    treatment_program.append(
        {
            "first_year": 2023,
            "last_year": 2025,
            "interventions": {
                "treatment_name": "IVM",
                "treatment_interval": 1,
                "total_population_coverage": 0.65,
                "min_age_of_treatment": 5,
                "correlation": 0.5,
            },
        }
    )
    treatment_program.append(
        {
            "first_year": 2026,
            "last_year": 2200,
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
            },
            "changes": [{"year": 2026, "params": {"delta_time_days": 0.5}}],
        },
        "programs": [
            {
                "first_year": 2023,
                "last_year": 2025,
                "interventions": {
                    "treatment_name": "IVM",
                    "treatment_interval": 1,
                    "correlation": 0.3,
                },
            },
            {
                "first_year": 2026,
                "last_year": 2040,
                "interventions": {
                    "treatment_name": "MOX",
                    "treatment_interval": 0.5,
                    "min_age_of_treatment": 4,
                    "total_population_coverage": 0.8,
                    "correlation": 0.3,
                    "microfilaricidal_nu": 0.04,
                    "microfilaricidal_omega": 1.82,
                    "embryostatic_lambda_max": 462,
                    "embryostatic_phi": 4.83,
                },
            },
        ],
    }


# Function to run and save simulations
def run_sim(i, verbose=False, samp_interval=1, mox_interval=1, end_time=2041):

    # Read in endgame objects and set up simulation
    hdf5_file = h5py.File(
        "/Users/adi/Downloads/hdf5_output/OutputVals_TestFeb2024_CIV0162715440.hdf5",
        "r",
    )
    restored_file_to_use = hdf5_file[f"draw_{i}"]
    restored_endgame_sim = EndgameSimulation.restore(restored_file_to_use)
    current_params = restored_endgame_sim.simulation.get_current_params()

    new_endgame_structure = get_endgame(mox_interval=mox_interval)
    new_endgame = EpionchoEndgameModel.parse_obj(new_endgame_structure)

    new_endgame.parameters.initial.blackfly.bite_rate_per_person_per_year = (
        current_params.blackfly.bite_rate_per_person_per_year
    )
    new_endgame.parameters.initial.gamma_distribution = (
        current_params.gamma_distribution
    )
    new_endgame.parameters.initial.seed = current_params.seed

    restored_endgame_sim.reset_endgame(new_endgame)

    # Run
    run_data: Data = {}
    run_data_age: Data = {}
    for state in restored_endgame_sim.iter_run(
        end_time=end_time,
        sampling_interval=samp_interval,
        make_time_backwards_compatible=True,
    ):

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


def run_simulations(
    verbose,
    iu_name,
    sample,
    sample_interval,
    mox_interval,
    end_time,
    ranges,
    max_workers,
    desc,
):
    rumSim = partial(
        run_sim,
        verbose=verbose,
        samp_interval=sample_interval,
        mox_interval=mox_interval,
        end_time=end_time,
    )
    datas: list[tuple[Data, Data]] = process_map(
        rumSim, ranges, max_workers=max_workers
    )
    data: list[Data] = [row[0] for row in datas]
    age_data: list[Data] = [row[1] for row in datas]
    write_data_to_csv(
        data,
        "test_outputs/python_model_output/hdf5_testing_"
        + iu_name
        + "-new_run_"
        + desc
        + ".csv",
    )
    # write_data_to_csv(
    #     age_data,
    #     "test_outputs/python_model_output/hdf5_testing_"
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
    end_time = 2041
    max_workers = os.cpu_count() if num_iter > os.cpu_count() else num_iter
    run_simulations(
        False,
        iu_name,
        True,
        1,
        1,
        end_time,
        range(num_iter),
        max_workers,
        "mox_annual_1year",
    )


if __name__ == "__main__":
    # Run example
    # iu = "GHA0216121382"
    iu = "CIV0162715440"
    wrapped_parameters(iu)
