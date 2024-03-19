import math
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from fast_binomial import SFC64, Generator
from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


# Function to add model parameters (seed, exp, abr) and MDA history to endgame object
def get_endgame(iter, iu_name="GHA0216121382", sample=True, use_mda=True):
    treatment_program = []
    changes = []
    seed = iter + iter * 3758
    gamma_distribution = 0.31
    abr = 1641
    # TODO: Variable sampling during test
    inputData = pd.read_csv(
        "/Users/adi/Downloads/test_inputs[15]/InputPars_" + iu_name + ".csv"
    )  # pd.read_csv("test_outputs/inputParams/InputPars_" + iu_name + ".csv")
    index = (
        iter if iter < inputData.shape[0] else iter % inputData.shape[0]
    )  # random.randint(1, inputData.shape[0])
    if sample:
        seed = inputData.loc[index][0]
        gamma_distribution = inputData.loc[index][1]
        abr = inputData.loc[index][2]
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
    elif iu_name == "GIN0227322616":
        treatment_program.append(
            {
                "first_year": 2014,
                "last_year": 2014,
                "interventions": {
                    "treatment_interval": 1,
                    "total_population_coverage": 0.65,
                    "correlation": 0.5,
                },
            }
        )
        treatment_program.append(
            {
                "first_year": 2015,
                "last_year": 2015,
                "interventions": {
                    "treatment_interval": 1,
                    "total_population_coverage": 0.25,
                    "correlation": 0.5,
                },
            }
        )
        treatment_program.append(
            {
                "first_year": 2016,
                "last_year": 2025,
                "interventions": {
                    "treatment_interval": 1,
                    "total_population_coverage": 0.65,
                    "correlation": 0.5,
                },
            }
        )
    # treatment_program.append(
    #     {
    #         "first_year": 2026,
    #         "last_year": 2030,
    #         "interventions": {
    #             "treatment_interval": 1,
    #             "total_population_coverage": 0.65,
    #             "min_age_of_treatment": 4,
    #             "correlation": 0.5,
    #             "microfilaricidal_nu": 0.04,
    #             "microfilaricidal_omega": 1.82,
    #             "embryostatic_lambda_max": 462,
    #             "embryostatic_phi": 4.83,
    #         },
    #     }
    # )
    changes.append({"year": 2026, "params": {"delta_time_days": 0.5}})
    if not (use_mda):
        treatment_program = []
        changes = []
    return {
        "parameters": {
            "initial": {
                "n_people": 400,
                "year_length_days": 365,
                "delta_h_zero": 0.186,
                "c_v": 0.005,
                "delta_h_inf": 0.003,
                "seed": seed,
                "gamma_distribution": gamma_distribution,
                "delta_time_days": 1,
                "blackfly": {"bite_rate_per_person_per_year": abr},
                "sequela_active": [
                    "Blindness",
                ],
            },
            "changes": changes,
        },
        "programs": treatment_program,
    }


# Function to run and save simulations
def run_sim(i, iu_name, verbose=False, sample=True, use_mda=True, start_time=1700):
    endgame_structure = get_endgame(i, iu_name, sample=sample, use_mda=use_mda)
    # Read in endgame objects and set up simulation
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)
    # print(endgame)
    endgame_sim = EndgameSimulation(
        start_time=start_time, endgame=endgame, verbose=verbose, debug=True
    )
    # Run
    run_data: Data = {}
    run_data_age: Data = {}
    for state in endgame_sim.iter_run(end_time=2025, sampling_interval=1):

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
            intensity=True,
            with_sequela=True,
            with_pnc=False,
        )
        add_state_to_run_data(
            state,
            run_data=run_data_age,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=True,
            prevalence=True,
            mean_worm_burden=False,
            prevalence_OAE=False,
            intensity=True,
            with_sequela=True,
            with_pnc=False,
        )

    return (run_data, run_data_age)


# Wrapper
def wrapped_parameters(iu_name):
    # Run simulations and save output
    num_iter = 50
    start_time = 1800
    use_mda = True
    sample = True
    fixed_code = "no_recreate_generator"
    sample_desc = "sample" if sample else "const_abr"
    mda_desc = "mda" if use_mda else "no_mda"
    file_name = (
        "sequelae_"
        + iu_name
        + "_start"
        + str(start_time)
        + "_"
        + str(num_iter)
        + "_"
        + sample_desc
        + "_"
        + mda_desc
        + "_"
        + fixed_code
    )
    file_name_age_grouped = file_name
    file_name_no_mda = (
        "sequelae_" + iu_name + "_start" + "1800_" + "200_" + sample_desc + "_no_mda"
    )
    max_workers = os.cpu_count() - 3 if num_iter > os.cpu_count() else num_iter
    rumSim = partial(
        run_sim,
        verbose=False,
        iu_name=iu_name,
        sample=sample,
        use_mda=use_mda,
        start_time=start_time,
    )
    datas: list[tuple[Data, Data]] = process_map(
        rumSim, range(num_iter), max_workers=max_workers
    )

    data: list[Data] = [row[0] for row in datas]
    age_data: list[Data] = [row[1] for row in datas]
    write_data_to_csv(
        data,
        "test_outputs/python_model_output/"
        + file_name
        + ".csv",  # testing_" + iu_name + "-new_run" + str(i) + ".csv",
    )
    write_data_to_csv(
        age_data,
        "test_outputs/python_model_output/"
        + file_name
        + "_age-grouped"
        + ".csv",  # testing_" + iu_name + "-new_run" + str(i) + ".csv",
    )
    plot_age_group = True
    if plot_age_group:
        file_name_age_grouped += "_age-grouped"
        df = pd.read_csv(
            "test_outputs/python_model_output/" + file_name_age_grouped + ".csv"
        )
        filtered_df = df[((df["measure"] == "Blindness"))]
        numbers = df[((df["measure"] == "number"))]

        years_num = numbers["year_id"]
        age_groups_num = [
            str(x) + "_" + str(y)
            for x, y in zip(numbers["age_start"], numbers["age_end"])
        ]
        number_age_group = numbers.iloc[:, 4:].mean(axis=1).tolist()

        new_combined_df = pd.merge(
            filtered_df,
            numbers,
            on=["year_id", "age_start", "age_end"],
            suffixes=("_blindness", "_number"),
        )
        for col in df.columns:
            if col.startswith("draw"):
                new_combined_df[col] = (
                    new_combined_df[col + "_blindness"]
                    * new_combined_df[col + "_number"]
                )
        new_combined_df = new_combined_df.drop(
            columns=[
                col + suffix
                for col in df.columns
                if col.startswith("draw")
                for suffix in ["_blindness"]
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
        ]  # [str(x) + "_" + str(y) for x, y in zip(filtered_df["age_start"], filtered_df["age_end"])]
        years = age_grouped_df["year_id"].values  # filtered_df["year_id"].values
        blindness = age_grouped_df.iloc[:, 5:]  # filtered_df.iloc[:, 4:]

        new_combined_df = (
            new_combined_df.drop(
                columns=["age_start", "age_end", "measure_blindness", "measure_number"]
            )
            .groupby("year_id")
            .sum()
            .reset_index()
        )

        years_num_all = new_combined_df["year_id"]
        age_groups_num_all = np.full(len(years_num_all), "0_80")
        number_age_group_all = (
            new_combined_df.filter(regex="^draw.*_number$").mean(axis=1).tolist()
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
        blindness_2 = new_combined_df.iloc[:, 1:]
        age_groups_2 = np.full(len(years_2), "0_80")
    else:
        df = pd.read_csv("test_outputs/python_model_output/" + file_name + ".csv")
        filtered_df = df[((df["measure"] == "Blindness"))]
        age_groups = [
            str(x) + "_" + str(y)
            for x, y in zip(filtered_df["age_start"], filtered_df["age_end"])
        ]
        years = filtered_df["year_id"].values
        blindness = filtered_df.iloc[:, 4:]  # .mean(axis=1, skipna=True).tolist()
        intensity = (
            (df[df["measure"] == "intensity"].iloc[:, 4:])
            .mean(axis=1, skipna=True)
            .tolist()
        )

    df_no_mda = pd.read_csv(
        "test_outputs/python_model_output/" + file_name_no_mda + ".csv"
    )  # "test_outputs/python_model_output/"  + file_name + ".csv")
    filtered_df_no_mda = df_no_mda[df_no_mda["measure"] == "Blindness"]
    years_no_mda = filtered_df_no_mda["year_id"]
    blindness_no_mda = filtered_df_no_mda.iloc[:, 4:].mean(axis=1, skipna=True).tolist()

    raiha_df = pd.read_csv(
        "/Users/adi/Downloads/csv_output_testfeb2024/model_output_TestFeb2024_GHA0216121382.csv"
    )
    filt_raiha_df = raiha_df[raiha_df["measure"] == "Blindness"]
    filt_raiha_df["year_id"]
    filt_raiha_df["mean"].astype(float)

    newDf = pd.DataFrame(
        {
            "years": years,
            "blindness": blindness.mean(axis=1, skipna=True).tolist(),
            "age_groups": age_groups,
        }
    )

    if not plot_age_group:
        fig, ax1 = plt.subplots()
        blindness_plot = blindness.mean(axis=1, skipna=True).tolist()
        ax1.plot(years, blindness_plot)
        ax1.plot(years_no_mda, blindness_no_mda, "r-")
        # ax1.plot(r_years, r_blindness, 'b-')
        ax1.vlines(x=1988, color="black", ymin=0, ymax=np.max(blindness_plot))

    if plot_age_group:

        def create_graphs(newDf, newDf_2, print_num):
            index = 0
            square = math.floor(math.sqrt(len(newDf["age_groups"].unique()) + 1))
            fig, ax1 = plt.subplots(square, square + 1, figsize=(10, 8), sharex=True)
            ax1 = ax1.flatten()
            for age_group in newDf["age_groups"].unique():
                ax = ax1[index]
                group_data = newDf[newDf["age_groups"] == age_group]
                ax.plot(group_data["years"], group_data["blindness"], label=age_group)
                ax.vlines(
                    x=1988, color="red", ymin=0, ymax=np.max(group_data["blindness"])
                )
                ax.set_xlim(left=1950)
                ax.set_title(age_group)
                if index % (square + 1) == 0:
                    ax.set_ylabel(ylab)
                index += 1
            ax1[index].plot(
                newDf_2["years"],
                newDf_2["blindness"],
                label=newDf_2["age_groups"].unique(),
            )
            ax1[index].vlines(
                x=1988, color="red", ymin=0, ymax=np.max(newDf_2["blindness"])
            )
            ax1[index].set_xlim(left=1950)
            ax1[index].set_title(newDf_2["age_groups"].unique())
            for i in range(index + 1, len(ax1)):
                ax1[i].tick_params(
                    left=False, labelleft=False, bottom=False, labelbottom=False
                )
            save_file_name = "_population" if print_num else ""
            # plt.show()
            plt.savefig(
                "test_outputs/testing_blindness_" + file_name + save_file_name + ".png",
                dpi=300,
            )

        print_num = False
        newDf_2 = pd.DataFrame(
            {
                "years": years_2,
                "blindness": blindness_2.mean(axis=1, skipna=True).tolist(),
                "age_groups": age_groups_2,
            }
        )
        ylab = "MF Prev"
        create_graphs(newDf, newDf_2, print_num)
        print_num = True
        if print_num:
            newDf = pd.DataFrame(
                {
                    "years": years_num,
                    "blindness": number_age_group,
                    "age_groups": age_groups_num,
                }
            )
            newDf_2 = pd.DataFrame(
                {
                    "years": years_num_all,
                    "blindness": number_age_group_all,
                    "age_groups": age_groups_num_all,
                }
            )
            ylab = "Population count"
            create_graphs(newDf, newDf_2, print_num)
    # for i in range(0, blindness.shape[0]):
    #     vals = blindness.iloc[i, :].values
    #     year = years[i]
    #     ax1.boxplot(vals, positions=[year], showfliers=False)
    # if vals[years == 1996] > vals[years == 1998]:
    #    ax1.plot(years, blindness.iloc[:, i].values)
    # ax1.set_xlabel('Years')
    # ax1.set_ylabel('Blindness Prev')
    # ax1.set_xlim(left=1900)
    # ax1.set_ylim(bottom=12, top=15)#0.05)

    # ymax = np.max(blindness)
    # ax1.vlines(x=1988, color="black", ymin=0, ymax=ymax)
    # ax1.vlines(x=1997, color="black", ymin=0, ymax=ymax)
    # plt.gca().set_yticks([10, 12, 14, 16, 18, 20])
    # ax1.vlines(x=2005, color="red", ymin=0, ymax=ymax)
    # ax1.vlines(x=2017, color="red", ymin=0, ymax=ymax)
    # ax1.vlines(x=2019, color="red", ymin=0, ymax=ymax)
    # ax1.vlines(x=2021, color="red", ymin=0, ymax=ymax)
    # plt.show()


if __name__ == "__main__":
    # Run example
    iu = "GHA0216121382"
    iu = "CIV0162715440"
    wrapped_parameters(iu)
