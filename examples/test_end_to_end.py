import csv
from collections import defaultdict
from multiprocessing import cpu_count

import numpy as np
from tqdm.contrib.concurrent import process_map

from epioncho_ibm import EndgameSimulation, EpionchoEndgameModel

np.random.seed(0)

endgame = """
{
    "parameters": {
        "initial": {
            "n_people": 100
        },
        "changes": [
            {
                "year": 2022,
                "month": 1,
                "params": {
                    "delta_time_days": 1
                }
            },
            {
                "year": 2028,
                "month": 1,
                "params": {
                    "delta_time_days": 0.5
                }
            }
        ]
    },
    "programs": [
        {
            "first_year": 2025,
            "first_month": 2,
            "last_year": 2030,
            "last_month": 7,
            "interventions": {
                "treatment_interval": 0.5
            }
        }
    ]
}
"""

params = EpionchoEndgameModel.parse_raw(endgame)
simulation = EndgameSimulation(
    start_time=2020, endgame=params, verbose=True, debug=True
)

print("First years without treatment:")
simulation.run(end_time=2025.0)

Year = float
AgeStart = float
AgeEnd = float
Measurement = str


save_file = "./test.hdf5"
simulation.save(save_file)


def run_sim(i) -> dict[tuple[Year, AgeStart, AgeEnd, Measurement], float | int]:
    sim = EndgameSimulation.restore(save_file)
    run_data: dict[tuple[Year, AgeStart, AgeEnd, Measurement], float | int] = {}
    for state in sim.iter_run(end_time=2030, sampling_interval=1):
        for age_start in range(6, 100):
            age_state = state.get_state_for_age_group(age_start, age_start + 1)
            prev = age_state.mf_prevalence_in_population()
            partial_key = (round(state.current_time), age_start, age_start + 1)
            run_data[(*partial_key, "prevalence")] = prev
            run_data[(*partial_key, "number")] = age_state.n_people

        for age_start in range(6, 100, 5):
            n_treatments = state.get_treatment_count_for_age_group(
                age_start, (age_start + 5)
            )
            age_state = state.get_state_for_age_group(age_start, age_start + 5)
            # Note: This is an approximation as it assumes the number of people in each category has not
            # changed since treatment
            partial_key = (round(state.current_time), age_start, age_start + 5)
            run_data[(*partial_key, "n_treatments")] = n_treatments
            if age_state.n_people == 0:
                run_data[(*partial_key, "achieved_coverage")] = 0
            else:
                run_data[(*partial_key, "achieved_coverage")] = (
                    n_treatments / age_state.n_people
                )
        state.reset_treatment_counter()
    return run_data


run_iters = 5

if __name__ == "__main__":
    data: list[
        dict[tuple[Year, AgeStart, AgeEnd, Measurement], float | int]
    ] = process_map(run_sim, range(run_iters), max_workers=cpu_count())

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
    with open("test.csv", "w") as f:
        # create the csv writer
        writer = csv.writer(f)
        first_elem: list[str] = ["year_id", "age_start", "age_end", "measure"] + [
            f"draw_{i}" for i in range(run_iters)
        ]
        excel_data: list[tuple[str | float]] = [tuple(first_elem)]
        writer.writerow(first_elem)
        for row in rows:
            excel_data.append(row)
            writer.writerow(row)
    # write a row to the csv file

    print(excel_data)
