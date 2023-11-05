import csv
from collections import defaultdict

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
    both_prev_types: bool = False,
    age_min: int = 0,
    age_max: int = 80,
    with_sequela: bool = True,
) -> None:
    if prevalence or number or mean_worm_burden or intensity:
        if with_age_groups:
            for age_start in range(age_min, age_max):
                age_state = state.get_state_for_age_group(age_start, age_start + 1)
                partial_key = (round(state.current_time), age_start, age_start + 1)
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
        if not (with_age_groups) or both_prev_types:
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
    if n_treatments or achieved_coverage:
        if with_age_groups:
            for age_start in range(age_min, age_max, 5):
                age_state = state.get_state_for_age_group(age_start, age_start + 5)
                # Note: This is an approximation as it assumes the number of people in each category has not
                # changed since treatment
                partial_key = (round(state.current_time, 2), age_start, age_start + 5)
                n_treatments_val = state.get_treatment_count_for_age_group(
                    age_start, (age_start + 5)
                )
                if n_treatments:
                    run_data[(*partial_key, "n_treatments")] = n_treatments_val
                if achieved_coverage:
                    run_data[(*partial_key, "achieved_coverage")] = (
                        n_treatments_val / state.n_people
                        if age_state.n_people != 0
                        else 0
                    )
        else:
            partial_key = (round(state.current_time, 2), age_min, age_max)
            n_treatments_val = state.get_treatment_count_for_age_group(age_min, age_max)
            if n_treatments:
                run_data[(*partial_key, "n_treatments")] = n_treatments_val
            if achieved_coverage:
                run_data[(*partial_key, "achieved_coverage")] = (
                    n_treatments_val / state.n_people if state.n_people != 0 else 0
                )

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
