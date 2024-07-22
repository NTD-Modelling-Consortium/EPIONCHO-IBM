import csv
import glob
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from epioncho_ibm import State

Year = float
AgeStart = float
AgeEnd = float
Measurement = str

Data = dict[tuple[Year, AgeStart, AgeEnd, Measurement], float | int]


def add_state_to_run_data(
    state: State,
    run_data: Data,
    prevalence: bool = True,
    intensity: bool = True,
    number: bool = True,
    mean_worm_burden: bool = True,
    prevalence_OAE: bool = True,
    n_treatments: bool = True,
    achieved_coverage: bool = True,
    with_age_groups: bool = True,
    with_sequela: bool = True,
    with_pnc: bool = True,
    saving_multiple_states=False,
    age_range: tuple[int, int] = (0, 80),
) -> None:
    age_min = age_range[0]
    age_max = age_range[1]
    if prevalence or number or mean_worm_burden or intensity or with_pnc:
        if with_age_groups:
            for age_start in range(age_min, age_max):
                age_state = state.get_state_for_age_group(age_start, age_start + 1)
                partial_key = (round(state.current_time, 2), age_start, age_start + 1)
                if prevalence:
                    run_data[
                        (*partial_key, "prevalence")
                    ] = age_state.mf_prevalence_in_population(return_nan=True)
                if number:
                    run_data[(*partial_key, "number")] = age_state.n_people
                if mean_worm_burden:
                    run_data[
                        (*partial_key, "mean_worm_burden")
                    ] = age_state.mean_worm_burden()
                if intensity:
                    (
                        run_data[(*partial_key, "intensity")],
                        _,
                    ) = age_state.microfilariae_per_skin_snip(return_nan=True)
                if prevalence_OAE:
                    run_data[
                        (*partial_key, "OAE_prevalence")
                    ] = age_state.OAE_prevalence()
                if with_sequela:
                    seq = age_state.sequalae_prevalence()
                    for sequela, prev in seq.items():
                        run_data[(*partial_key, sequela)] = prev
                if with_pnc:
                    run_data[(*partial_key, "pnc")] = age_state.percent_non_compliant()
        else:
            partial_key = (round(state.current_time, 2), age_min, age_max)
            if prevalence:
                run_data[
                    (*partial_key, "prevalence")
                ] = state.mf_prevalence_in_population(return_nan=True)
            if number:
                run_data[(*partial_key, "number")] = state.n_people
            if mean_worm_burden:
                run_data[(*partial_key, "mean_worm_burden")] = state.mean_worm_burden()
            if intensity:
                (
                    run_data[(*partial_key, "intensity")],
                    _,
                ) = state.microfilariae_per_skin_snip(return_nan=True)
            if prevalence_OAE:
                run_data[(*partial_key, "OAE_prevalence")] = state.OAE_prevalence()
            if with_sequela:
                seq = state.sequalae_prevalence()
                for sequela, prev in seq.items():
                    run_data[(*partial_key, sequela)] = prev
            if with_pnc:
                run_data[(*partial_key, "pnc")] = state.percent_non_compliant()
    if n_treatments or achieved_coverage:
        if with_age_groups:
            for age_start in range(age_min, age_max, 1):
                age_state = state.get_state_for_age_group(age_start, age_start + 1)

                if n_treatments:
                    n_treatments_val = state.get_treatment_count_for_age_group(
                        age_start, (age_start + 1)
                    )
                    number_of_rounds = {}
                    for key, value in sorted(n_treatments_val.items()):
                        time_of_intervention = float(key.split(",")[0])
                        intervention_type = key.split(",")[1]
                        number_of_rounds[intervention_type] = (
                            number_of_rounds.get(intervention_type, 0) + 1
                        )

                        partial_key = (
                            math.floor(time_of_intervention),
                            age_start,
                            age_start + 1,
                        )

                        run_data[
                            (
                                *partial_key,
                                intervention_type
                                + " "
                                + str(number_of_rounds[intervention_type]),
                            )
                        ] = value

                if achieved_coverage:
                    achieved_coverage_val = state.get_achieved_coverage_for_age_group(
                        age_start, (age_start + 1)
                    )
                    number_of_rounds = {}
                    for key, value in sorted(achieved_coverage_val.items()):
                        time_of_intervention = float(key.split(",")[0])
                        intervention_type = key.split(",")[1]
                        number_of_rounds[intervention_type] = (
                            number_of_rounds.get(intervention_type, 0) + 1
                        )

                        partial_key = (
                            math.floor(time_of_intervention),
                            age_start,
                            age_start + 1,
                        )

                        run_data[
                            (
                                *partial_key,
                                "Coverage "
                                + intervention_type
                                + " "
                                + str(number_of_rounds[intervention_type]),
                            )
                        ] = value
        else:
            if n_treatments:
                n_treatments_val = state.get_treatment_count_for_age_group(
                    age_min, age_max
                )
                number_of_rounds = {}
                for key, value in sorted(n_treatments_val.items()):
                    time_of_intervention = float(key.split(",")[0])
                    intervention_type = key.split(",")[1]

                    number_of_rounds[intervention_type] = (
                        number_of_rounds.get(intervention_type, 0) + 1
                    )

                    partial_key = (math.floor(time_of_intervention), age_min, age_max)

                    run_data[
                        (
                            *partial_key,
                            intervention_type
                            + " "
                            + str(number_of_rounds[intervention_type]),
                        )
                    ] = value
            if achieved_coverage:
                achieved_coverage_val = state.get_achieved_coverage_for_age_group(
                    age_min, age_max
                )
                number_of_rounds = {}
                for key, value in sorted(achieved_coverage_val.items()):
                    time_of_intervention = float(key.split(",")[0])
                    intervention_type = key.split(",")[1]
                    number_of_rounds[intervention_type] = (
                        number_of_rounds.get(intervention_type, 0) + 1
                    )

                    partial_key = (math.floor(time_of_intervention), age_min, age_max)

                    run_data[
                        (
                            *partial_key,
                            "Coverage "
                            + intervention_type
                            + " "
                            + str(number_of_rounds[intervention_type]),
                        )
                    ] = value

    if not saving_multiple_states:
        state.reset_treatment_counter()


def write_data_to_csv(
    data: list[Data],
    csv_file: str,
) -> None:
    data_combined_runs: dict[
        tuple[Year, AgeStart, AgeEnd, Measurement], list[float | int]
    ] = defaultdict(list)
    for run in data:
        for k, v in run.items():
            data_combined_runs[k].append(v)

    rows = sorted(
        (k + tuple(v) for k, v in data_combined_runs.items()),
        key=lambda r: (r[0], r[3], r[1]),
    )
    with open(csv_file, "w") as f:
        # create the csv writer
        writer = csv.writer(f)
        first_elem: list[str] = ["year_id", "age_start", "age_end", "measure"] + [
            f"draw_{i}" for i in range(len(data))
        ]
        excel_data: list[tuple[str | float, ...]] = [tuple(first_elem)]
        writer.writerow(first_elem)
        for row in rows:
            excel_data.append(row)
            writer.writerow(row)


def post_processing_calculation(
    data: list[Data],
    iuName: str,
    scenario: str,
    prevalence_marker_name: str,
    csv_file: str,
    post_processing_start_time: int = 1970,
    mda_start_year: int = 2026,
    mda_stop_year: int = 2041,
    mda_interval: float = 1,
) -> None:
    """
    Takes in non-age-grouped model outputs and generates a summarized output file that summarizes
    the model outputs into mean, median, standard deviation, 2.5, 5, 10, 25, 50, 75, 90, 95, and 97.5 percentiles.
    It also calculates:
        (for each year) the probability that < 1% mf prevalence was reached across all iterations/draws
        (for each year) the probability that < 0% mf prevalence was reached across all iterations/draws
        the year at which < 1% mf prevalence is reached, calculated using the average prevalence across all runs
        the year at which 90% of runs reach < 1% mf prevalence is reached

    Args:
        data list[Data]: The raw data output from multiple runs of the model (a list of dictionaries, where each dictionary is the outputs for a single run of the model)
        iuName str: A name to define the parameters used for the model, typically the name of the IU being simulated
        scenario str: A name to define the scenario being tested
        prevalence_marker_name str: The name for the prevalence marker to be used to calculate the additional outputs
        post_processing_start_time int: The time at which we start the calculations of reaching the threshold/elimination
        csv_file str: The name you want the post processed data to be saved to.
        mda_start_year int: An optional variable to denote when MDA starts for a scenario
        mda_stop_year int: An optional variable to denote when MDA ends for a given scenario
        mda_interval int: An optional variable to denote the frequency of the MDA applied for a given scenario
    """
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
    # Making sure we only start the calculations from where MDA starts
    post_processing_start_mask = tmp[:, 0].astype(float) >= post_processing_start_time
    tmp_for_calc = tmp[post_processing_start_mask, :]

    # Calculating probability of elimination using mf_prev
    mf_prev_mask = tmp_for_calc[:, 3] == prevalence_marker_name
    mf_prev_vals = tmp_for_calc[mf_prev_mask, 4:].astype(float)

    # Probability of elimination for a given year = the average number of runs that reach 0 mf prev
    prob_elim = np.mean(mf_prev_vals == 0, axis=1)
    num_prob_elim = len(prob_elim)
    none_array = np.full(num_prob_elim, None)
    # combining results into a matrix format for output
    prob_elim_output = np.column_stack(
        (
            tmp_for_calc[mf_prev_mask, :3],
            np.full(num_prob_elim, "prob_elim"),
            prob_elim,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
        )
    )

    # Calculating the year where each run has < 1% prev
    mf_under_1_mask = mf_prev_vals < 0.01

    # Probability of getting < 1% mfp for a given year
    prob_under_1_mfp = np.mean(mf_under_1_mask, axis=1)
    num_prob_under_1_mfp = len(prob_under_1_mfp)
    none_array = np.full(num_prob_under_1_mfp, None)
    prob_under_1_mfp_output = np.column_stack(
        (
            tmp_for_calc[mf_prev_mask, :3],
            np.full(num_prob_under_1_mfp, "prob_under_1_mfp"),
            prob_under_1_mfp,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
            none_array,
        )
    )

    indeces_of_90_under_1_mfp = np.where(prob_under_1_mfp >= 0.90)[0]
    year_of_90_under_1_mfp = None
    if indeces_of_90_under_1_mfp.size > 0:
        year_of_90_under_1_mfp = tmp_for_calc[mf_prev_mask, :][
            indeces_of_90_under_1_mfp[0], 0
        ]
    year_90_under_1_mfp_output = np.column_stack(
        (
            "",
            np.nan,
            np.nan,
            "year_of_90_under_1_mfp_avg",
            year_of_90_under_1_mfp,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    )

    # Calculating the year where the avg of all runs has < 1% mf prev
    yearly_avg_mfp = np.mean(mf_prev_vals, axis=1)
    indeces_of_1_mfp_avg = np.where(yearly_avg_mfp < 0.01)[0]
    year_of_1_mfp_avg = None
    if indeces_of_1_mfp_avg.size > 0:
        year_of_1_mfp_avg = tmp_for_calc[mf_prev_mask, :][indeces_of_1_mfp_avg[0], 0]
    year_under_1_avg_mfp_output = np.column_stack(
        (
            "",
            np.nan,
            np.nan,
            "year_of_1_mfp_avg",
            year_of_1_mfp_avg,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    )

    # Summarizing all other outputs
    other_prevs = tmp[:, 4:].astype(float)
    other_prevs_output = np.column_stack(
        (
            tmp[:, :4],
            np.mean(other_prevs, axis=1),
            np.percentile(other_prevs, 2.5, axis=1),
            np.percentile(other_prevs, 5, axis=1),
            np.percentile(other_prevs, 10, axis=1),
            np.percentile(other_prevs, 25, axis=1),
            np.percentile(other_prevs, 50, axis=1),
            np.percentile(other_prevs, 75, axis=1),
            np.percentile(other_prevs, 90, axis=1),
            np.percentile(other_prevs, 95, axis=1),
            np.percentile(other_prevs, 97.5, axis=1),
            np.std(other_prevs, axis=1),
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
            # year that the avg mfp is < 1%
            year_under_1_avg_mfp_output,
            # year that the 90% of runs have < 1% mfp
            year_90_under_1_mfp_output,
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
            "2.5_percentile",
            "5_percentile",
            "10_percentile",
            "25_percentile",
            "50_percentile",
            "75_percentile",
            "90_percentile",
            "95_percentile",
            "97.5_percentile",
            "standard_deviation",
            "median",
        ],
    ).to_csv(csv_file)


def combineAndFilter(
    pathToOutputFiles=".",
    specific_files="*.csv",
    measure_filter=f'measure == "years_to_1_mfp" | measure == "rounds_to_1_mfp" | measure == "rounds_to_90_under_1_mfp" | measure == "years_to_90_under_1_mfp" | measure == "year_of_1_mfp_avg"',
    output_file_root=".",
):
    """
    Combines all data outputs in a given folder and filters as necessary. Saves it into two new files, one which is filtered, and one which is not.
    The data in the folder should be the output of `post_processing_calculation`

    Args:
        pathToOutputFiles - where all the data files are located
        specific_files - file name filter to only combine the files that are wanted
        measure_filter - data filter that filters the values within each data file based. This is directly passed into `pd.query()`

    """

    rows = []
    columns = []

    for filename in glob.glob(
        pathToOutputFiles + "**/" + specific_files, recursive=True
    ):
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            if len(columns) == 0:
                columns = next(reader)
            else:
                next(reader)
            rows.extend(reader)

    outputData = pd.DataFrame(rows, columns=columns)
    outputData.to_csv(f"{output_file_root}-combined_data.csv")

    filteredOutput = outputData.query(measure_filter)
    filteredOutput.to_csv(f"{output_file_root}-combined_filtered_data.csv")
