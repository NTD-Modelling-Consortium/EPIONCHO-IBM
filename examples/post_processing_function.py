from collections import defaultdict

import numpy as np
import pandas as pd

from epioncho_ibm.tools import Data


def post_processing_calculation(
    data: list[Data],
    iuName: str,
    scenario: str,
    csv_file: str,
    mda_start_year: int | None = 2026,
    mda_stop_year: int | None = 2041,
    mda_interval: float = 1,
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

    # Probability of elimination for a given year = the average number of runs that reach 0 mf prev
    prob_elim = np.mean(mf_prev_vals == 0, axis=1)
    num_prob_elim = len(prob_elim)
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

    # Calculating the year where each run has < 1% prev
    mf_under_1_mask = mf_prev_vals < 0.01

    # Probability of getting < 1% mfp for a given year
    prob_under_1_mfp = np.mean(mf_under_1_mask, axis=1)
    num_prob_under_1_mfp = len(prob_under_1_mfp)
    none_array = np.full(num_prob_under_1_mfp, None)
    prob_under_1_mfp_output = np.column_stack(
        (
            tmp[mf_prev_mask, :3],
            np.full(num_prob_under_1_mfp, "prob_under_1_mfp"),
            prob_under_1_mfp,
            none_array,
            none_array,
            none_array,
        )
    )

    # Calculating the year where the avg of all runs has < 1% mf prev
    yearly_avg_mfp = np.mean(mf_prev_vals, axis=1)
    indeces_of_1_mfp_avg = np.where(yearly_avg_mfp < 0.01)[0]
    year_of_1_mfp_avg = None
    if indeces_of_1_mfp_avg.size > 0:
        year_of_1_mfp_avg = tmp[mf_prev_mask, :][indeces_of_1_mfp_avg[0], 0]
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
        )
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
            # year_under_1_avg_mfp_output
            year_under_1_avg_mfp_output,
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
