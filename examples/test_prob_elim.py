import os
import random
from collections import defaultdict

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
        year_length_days=366,
        n_people=440,
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


def calculate_probability_elimination(
    data: list[Data],
    iuName: str,
    scenario: str,
    mda_start_year: int | None,
    mda_stop_year: int,
    mda_interval: int,
    csv_file: str,
) -> None:
    # Arranging data into an easy to manipulate format (taken from tools.py)
    data_combined_runs: dict[
        tuple[float, float, float, str], list[float | int]
    ] = defaultdict(list)
    for run in data:
        for k, v in run.items():
            data_combined_runs[k].append(v)

    rows = sorted(
        (k + tuple(v) for k, v in data_combined_runs.items()),
        key=lambda r: (r[0], r[3], r[1]),
    )

    tmp = np.array(rows)
    # Data manipulation
    if mda_start_year:
        yrs_after_mda_start_mask = tmp[:, 0].astype(float) >= mda_start_year
        tmp = tmp[yrs_after_mda_start_mask, :]
    else:
        mda_start_year = 0

    # Calculating probability of elimination using mf_prev
    mf_prev_mask = tmp[:, 3] == "prevalence"
    mf_prev_vals = tmp[mf_prev_mask, 4:].astype(float)

    # Calculating the year where each run has < 1% prev
    mf_under_1_mask = mf_prev_vals <= 0.01
    under_1_prev_indices = np.argmax(mf_under_1_mask, axis=0)
    yearOfUnder1Prev = np.array(
        [
            float(tmp[mf_prev_mask][under_1_prev_indices[i], 0])
            if any(mf_under_1_mask[:, i])
            else np.nan
            for i in range(mf_under_1_mask.shape[1])
        ]
    )

    roundsTillUnder1Prev = (yearOfUnder1Prev - mda_start_year) / mda_interval

    # Probability of elimination for a given year = the average number of runs that reach 0 mf prev
    prob_elim = np.mean(mf_prev_vals == 0, axis=1)
    num_prob_elim = np.sum(mf_prev_mask)
    none_array = np.full(num_prob_elim, None)
    # combining results into a matrix format for output
    prob_elim_output = np.column_stack(
        (
            tmp[mf_prev_mask, :3],
            np.full(num_prob_elim, "prob_elim"),
            prob_elim,
            none_array,
            none_array,
            none_array,
        )
    )

    # Probability of getting < 1% mfp for a given year
    prob_under_1_mfp = np.mean(mf_under_1_mask, axis=1)
    under_1_prev_90_index = np.argmax(prob_under_1_mfp >= 0.90, axis=0)
    over_90_prob_elim_index = np.argmax(prob_elim >= 0.90, axis=0)

    prob_under_1_mfp_output = np.column_stack(
        (
            tmp[mf_prev_mask, :3],
            np.full(num_prob_elim, "prob_under_1_mfp"),
            prob_under_1_mfp,
            none_array,
            none_array,
            none_array,
        )
    )

    # Find the year where 90% of the runs have <1% mfp or have reached elimination completely
    yearOf90ProbElim = (
        tmp[mf_prev_mask, 0][under_1_prev_90_index]
        if np.any(prob_under_1_mfp >= 0.90)
        else ""
    )
    yearOf90Under1Prev = (
        tmp[mf_prev_mask, 0][over_90_prob_elim_index]
        if np.any(prob_elim >= 0.90)
        else ""
    )
    roundsTill90Under1Prev = (
        (float(yearOf90Under1Prev) - mda_start_year) / mda_interval
        if yearOf90Under1Prev != ""
        else ""
    )

    # Summarizing all other prevalence outputs (filtering to only mfp)
    other_prevs = tmp[mf_prev_mask, 4:].astype(float)
    other_prevs_output = np.column_stack(
        (
            tmp[mf_prev_mask, :4],
            np.mean(other_prevs, axis=1),
            np.percentile(other_prevs, 2.5, axis=1),
            np.percentile(other_prevs, 97.5, axis=1),
            np.median(other_prevs, axis=1),
        )
    )
    output = np.row_stack(
        (
            other_prevs_output,
            # probability of elim for each year
            prob_elim_output,
            # probability of 1% mf_prev for each year
            prob_under_1_mfp_output,
            # year of >=90% elim
            np.array(
                [
                    "",
                    np.nan,
                    np.nan,
                    "years_to_90_prob_elim",
                    yearOf90ProbElim,
                    None,
                    None,
                    None,
                ]
            ),
            # year of >=90% under 1% prev
            np.array(
                [
                    "",
                    np.nan,
                    np.nan,
                    "years_to_90_under_1_mfp",
                    yearOf90Under1Prev,
                    None,
                    None,
                    None,
                ]
            ),
            # rounds till >=90% under 1% prev
            np.array(
                [
                    "",
                    np.nan,
                    np.nan,
                    "rounds_to_90_under_1_mfp",
                    roundsTill90Under1Prev,
                    None,
                    None,
                    None,
                ]
            ),
            # avg year of <=1% mf prev
            np.array(
                [
                    "",
                    np.nan,
                    np.nan,
                    "years_to_1_mfp",
                    np.nanmean(yearOfUnder1Prev)
                    if not (np.isnan(yearOfUnder1Prev).all())
                    else None,
                    np.percentile(yearOfUnder1Prev, 2.5),
                    np.percentile(yearOfUnder1Prev, 97.5),
                    np.median(yearOfUnder1Prev),
                ]
            ),
            # all years to <1% mfp
            np.array(
                [
                    "",
                    np.nan,
                    np.nan,
                    "years_to_1_mfp_all_runs",
                    ",".join(yearOfUnder1Prev.astype(str)),
                    None,
                    None,
                    None,
                ]
            ),
            # avg rounds till <=1% mf prev
            np.array(
                [
                    "",
                    np.nan,
                    np.nan,
                    "rounds_to_1_mfp",
                    np.nanmean(roundsTillUnder1Prev)
                    if not (np.isnan(roundsTillUnder1Prev).all())
                    else None,
                    np.percentile(roundsTillUnder1Prev, 2.5),
                    np.percentile(roundsTillUnder1Prev, 97.5),
                    np.median(roundsTillUnder1Prev),
                ]
            ),
            # all rounds to <1% mfp
            np.array(
                [
                    "",
                    np.nan,
                    np.nan,
                    "rounds_to_1_mfp_all_runs",
                    ",".join(roundsTillUnder1Prev.astype(str)),
                    None,
                    None,
                    None,
                ]
            ),
        )
    )

    descriptor_output = np.column_stack(
        (
            np.full(output.shape[0], iuName),
            np.full(output.shape[0], scenario),
            np.full(output.shape[0], mda_start_year),
            np.full(output.shape[0], mda_stop_year),
            np.full(output.shape[0], mda_interval),
            output,
        )
    )

    pd.DataFrame(
        descriptor_output,
        columns=[
            "iu_name",
            "scenario",
            "mda_start_year",
            "mda_stop_year",
            "mda_interval",
            "year_id",
            "age_start",
            "age_end",
            "measure",
            "mean",
            "lower_bound",
            "upper_bound",
            "median",
        ],
    ).to_csv(csv_file)


run_iters = 10

if __name__ == "__main__":
    cpus_to_use = cpu_count() - 4

    # mda_start = 2000
    # mda_stop = 2040
    loopTimes = [(2000, 2010), (2000, 2020), (2000, 2030), (2000, 2040)]
    interval = 1
    for mda_start, mda_stop in loopTimes:
        # ~ 70% MFP
        rumSim = partial(
            run_sim,
            start_time=1990,
            mda_start=mda_start,
            mda_stop=mda_stop,
            simulation_stop=2051,
            # abr=abr,
            verbose=False,
            # seed=seed,
            # gamma_distribution=kE,
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
        calculate_probability_elimination(
            all_age_data,
            "test",
            "test-scenario",
            mda_start,
            mda_stop,
            interval,
            "test_outputs/mda-stop-" + str(mda_stop) + "-all_age_data.csv",
        )
