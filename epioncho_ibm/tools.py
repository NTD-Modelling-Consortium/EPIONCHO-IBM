import csv
from collections import defaultdict

import numpy as np

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
    age_min: int = 0,
    age_max: int = 80,
    with_sequela: bool = True,
    output_age_range_map: dict = {},
) -> None:
    if prevalence or number or mean_worm_burden or intensity:
        if with_age_groups:
            age_groups = (
                output_age_range_map["age_groups"]
                if "age_groups" in output_age_range_map
                else [[i, i + 1] for i in range(age_min, age_max)]
            )
            for age_group in age_groups:
                add_state_to_run_data_helper(
                    run_data,
                    state,
                    age_group[0],
                    age_group[1],
                    prevalence,
                    number,
                    mean_worm_burden,
                    intensity,
                    prevalence_OAE,
                    with_sequela,
                    n_treatments=False,
                    achieved_coverage=False,
                )
        else:
            add_state_to_run_data_helper(
                run_data,
                state,
                age_min,
                age_max,
                prevalence,
                number,
                mean_worm_burden,
                intensity,
                prevalence_OAE,
                with_sequela,
                n_treatments=False,
                achieved_coverage=False,
                output_age_range_map=output_age_range_map,
            )
    if n_treatments or achieved_coverage:
        if with_age_groups:
            age_groups = (
                output_age_range_map["age_groups_cov"]
                if "cov_age_groups" in output_age_range_map
                else [[i, i + 5] for i in range(age_min, age_max, 5)]
            )
            for age_group in age_groups:
                add_state_to_run_data_helper(
                    run_data,
                    state,
                    age_group[0],
                    age_group[1],
                    prevalence,
                    number,
                    mean_worm_burden,
                    intensity,
                    prevalence_OAE,
                    with_sequela,
                    n_treatments,
                    achieved_coverage,
                    output_age_range_map,
                )
        else:
            add_state_to_run_data_helper(
                run_data,
                state,
                age_min,
                age_max,
                prevalence,
                number,
                mean_worm_burden,
                intensity,
                prevalence_OAE,
                with_sequela,
                n_treatments,
                achieved_coverage,
            )

    state.reset_treatment_counter()


def add_state_to_run_data_helper(
    run_data: Data,
    state: State,
    age_start: int,
    age_end: int,
    prevalence: bool,
    number: bool,
    mean_worm_burden: bool,
    intensity: bool,
    prevalence_OAE: bool,
    with_sequela: bool,
    n_treatments: bool,
    achieved_coverage: bool,
    output_age_range_map: dict = {},
):
    if prevalence:
        age_state, partial_key = return_age_state(
            state, output_age_range_map, "prevalence", age_start, age_end
        )
        run_data[(*partial_key, "prevalence")] = age_state.mf_prevalence_in_population(
            return_nan=True
        )
    if number:
        age_state, partial_key = return_age_state(
            state, output_age_range_map, "number", age_start, age_end
        )
        run_data[(*partial_key, "number")] = age_state.n_people
    if mean_worm_burden:
        age_state, partial_key = return_age_state(
            state, output_age_range_map, "mean_worm_burden", age_start, age_end
        )
        run_data[(*partial_key, "mean_worm_burden")] = age_state.mean_worm_burden()
    if intensity:
        age_state, partial_key = return_age_state(
            state, output_age_range_map, "intensity", age_start, age_end
        )
        (
            run_data[(*partial_key, "intensity")],
            _,
        ) = age_state.microfilariae_per_skin_snip(return_nan=True)
    if prevalence_OAE:
        age_state, partial_key = return_age_state(
            state, output_age_range_map, "prevalence_OAE", age_start, age_end
        )
        run_data[(*partial_key, "OAE_prevalence")] = age_state.OAE_prevalence()
    if with_sequela:
        age_state, partial_key = return_age_state(
            state, output_age_range_map, "with_sequela", age_start, age_end
        )
        seq = age_state.sequalae_prevalence()
        for sequela, prev in seq.items():
            run_data[(*partial_key, sequela)] = prev
    if not (n_treatments or achieved_coverage):
        return
    # Note: This is an approximation as it assumes the number of people in each category has not
    # changed since treatment
    n_treatments_val = state.get_treatment_count_for_age_group(age_start, age_end)
    if n_treatments:
        age_state, partial_key = return_age_state(
            state, output_age_range_map, "n_treatments", age_start, age_end
        )
        run_data[(*partial_key, "n_treatments")] = n_treatments_val
    if achieved_coverage:
        age_state, partial_key = return_age_state(
            state, output_age_range_map, "achieved_coverage", age_start, age_end
        )
        run_data[(*partial_key, "achieved_coverage")] = (
            n_treatments_val / state.n_people  # should this not be age_state.n_people?
            if age_state.n_people != 0
            else 0
        )


def return_age_state(
    state: State, output_age_range_map: dict, key: str, age_min: int, age_max: int
) -> tuple[State, tuple[float, int, int]]:
    if key in output_age_range_map:
        age_min = (
            output_age_range_map[key]["age_min"]
            if "age_min" in output_age_range_map[key]
            else age_min
        )
        age_max = (
            output_age_range_map[key]["age_max"]
            if "age_max" in output_age_range_map[key]
            else age_max
        )
    age_state = state.get_state_for_age_group(age_min, age_max)
    partial_key = (round(state.current_time, 2), age_min, age_max)
    return age_state, partial_key


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
