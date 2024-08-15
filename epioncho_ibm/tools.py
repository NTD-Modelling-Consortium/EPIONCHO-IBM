import csv
import math
from collections import defaultdict

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
    custom_age_groups: list[tuple[int, int]] = None,
    age_range: tuple[int, int] = (0, 80),
) -> None:
    age_min = age_range[0]
    age_max = age_range[1]
    if custom_age_groups is None:
        custom_age_groups = [(i, i + 1) for i in range(age_max)]
    if prevalence or number or mean_worm_burden or intensity or with_pnc:
        if with_age_groups:
            for age_start, age_end in custom_age_groups:
                age_state = state.get_state_for_age_group(age_start, age_end)
                partial_key = (round(state.current_time, 2), age_start, age_end)
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
            for age_start, age_end in custom_age_groups:
                age_state = state.get_state_for_age_group(age_start, age_end)

                if n_treatments:
                    n_treatments_val = state.get_treatment_count_for_age_group(
                        age_start, age_end
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
                            age_end,
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
                        age_start, age_end
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
                            age_end,
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


def flatten_and_sort(
    data: list[Data],
) -> list[tuple]:
    """
    Converts the model outputs from multiple runs (using `add_state_to_run_data`) into a sorted 2d list, where each row represents a year, age group, measure, and value for all runs.

    Args:
        data (list[Data]): The model output from multiple runs of epioncho-ibm.

    Returns:
        A 2D list, of type list[tuple[Year, AgeStart, AgeEnd, Measurement, float | int, ...] where the value for each model run x is stored as a float in columns after "Measurement"
    """
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
    return rows


def convert_data_to_pandas(
    data: list[Data],
) -> pd.DataFrame:
    rows = flatten_and_sort(data)
    return pd.DataFrame(
        rows,
        columns=["year_id", "age_start", "age_end", "measure"]
        + [f"draw_{i}" for i in range(len(data))],
    )


def write_data_to_csv(
    data: list[Data],
    csv_file: str,
) -> None:
    rows = flatten_and_sort(data)
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
