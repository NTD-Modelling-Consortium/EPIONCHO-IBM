import csv
from collections import defaultdict

from epioncho_ibm import EndgameSimulation, Simulation, State

Year = float
AgeStart = float
AgeEnd = float
Measurement = str

Data = dict[tuple[Year, AgeStart, AgeEnd, Measurement], float | int]


def add_state_to_run_data(
    state: State,
    run_data: Data,
    prevalence: bool = True,
    number: bool = True,
    n_treatments: bool = True,
    achieved_coverage: bool = True,
    with_age_groups: bool = True,
) -> None:
    if prevalence or number:
        if with_age_groups:
            for age_start in range(6, 100):
                age_state = state.get_state_for_age_group(age_start, age_start + 1)
                partial_key = (round(state.current_time), age_start, age_start + 1)
                if prevalence:
                    prev = age_state.mf_prevalence_in_population()
                    run_data[(*partial_key, "prevalence")] = prev
                if number:
                    run_data[(*partial_key, "number")] = age_state.n_people
        else:
            partial_key = (round(state.current_time), 6, 100)
            if prevalence:
                prev = state.mf_prevalence_in_population()
                run_data[(*partial_key, "prevalence")] = prev
            if number:
                run_data[(*partial_key, "number")] = state.n_people
    if n_treatments or achieved_coverage:
        if with_age_groups:
            for age_start in range(6, 100, 5):
                age_state = state.get_state_for_age_group(age_start, age_start + 5)
                # Note: This is an approximation as it assumes the number of people in each category has not
                # changed since treatment
                partial_key = (round(state.current_time), age_start, age_start + 5)
                n_treatments_val = state.get_treatment_count_for_age_group(
                    age_start, (age_start + 5)
                )
                if n_treatments:
                    run_data[(*partial_key, "n_treatments")] = n_treatments_val
                if achieved_coverage:
                    if age_state.n_people == 0:
                        run_data[(*partial_key, "achieved_coverage")] = 0
                    else:
                        run_data[(*partial_key, "achieved_coverage")] = (
                            n_treatments_val / age_state.n_people
                        )
        else:
            partial_key = (round(state.current_time), 6, 100)
            n_treatments_val = state.get_treatment_count_for_age_group(6, 100)
            if n_treatments:
                run_data[(*partial_key, "n_treatments")] = n_treatments_val
            if achieved_coverage:
                if state.n_people == 0:
                    run_data[(*partial_key, "achieved_coverage")] = 0
                else:
                    run_data[(*partial_key, "achieved_coverage")] = (
                        n_treatments_val / state.n_people
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
        excel_data: list[tuple[str | float]] = [tuple(first_elem)]
        writer.writerow(first_elem)
        for row in rows:
            excel_data.append(row)
            writer.writerow(row)
