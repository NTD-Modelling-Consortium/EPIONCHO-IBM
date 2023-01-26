import csv
from multiprocessing import cpu_count

import numpy as np
from tqdm.contrib.concurrent import process_map

from epioncho_ibm import Params, Simulation, TreatmentParams

np.random.seed(0)

params = Params(treatment=TreatmentParams(start_time=5, stop_time=130), n_people=300)

simulation = Simulation(start_time=2020, params=params, verbose=True)
print("First years without treatment:")
simulation.run(end_time=2025.0)
print(simulation.state.mf_prevalence_in_population())

Year = float
AgeStart = float
Prevalence = float

save_file = "./test.hdf5"
run_iters = 5


def run_sim(i) -> dict[Year, dict[AgeStart, Prevalence]]:
    sim = Simulation.restore(save_file)
    run_data: dict[Year, dict[AgeStart, Prevalence]] = {}
    for state in sim.iter_run(end_time=2030, sampling_interval=1):
        print(state.current_time)
        state_data = {}
        for age_start in range(6, 100):
            age_state = state.get_state_for_age_group(age_start, age_start + 1)
            prev = age_state.mf_prevalence_in_population()
            state_data[age_start] = prev
        run_data[state.current_time] = state_data
    return run_data


def main():
    simulation.save(save_file)

    data: list[dict[Year, dict[AgeStart, Prevalence]]] = process_map(
        run_sim, range(run_iters), max_workers=cpu_count()
    )

    data_by_year: dict[Year, dict[AgeStart, list[Prevalence]]] = {}
    for i, run in enumerate(data):
        for year, year_data in run.items():
            if year not in data_by_year:
                data_by_year[year] = {
                    age_start: [prev] for age_start, prev in year_data.items()
                }
            else:
                rel_year = data_by_year[year]
                for age_start, prev in year_data.items():
                    if age_start not in rel_year:
                        rel_year[age_start] = [prev]
                    else:
                        rel_year[age_start] = rel_year[age_start] + [prev]
                data_by_year[year] = rel_year

    f = open("test.csv", "w")

    # create the csv writer
    writer = csv.writer(f)
    first_elem: list[str] = ["year_id", "age_start", "age_end", "measure"] + [
        f"draw_{i}" for i in range(run_iters)
    ]
    excel_data: list[list[str | float] | list[str] | list[float]] = [first_elem]
    writer.writerow(first_elem)
    for year, year_data in data_by_year.items():
        for age_start, prevalences in year_data.items():
            row = [year, age_start, age_start + 1, "prevalence"] + prevalences
            excel_data.append(row)
            writer.writerow(row)
    f.close()

    # write a row to the csv file

    print(excel_data)


if __name__ == "__main__":
    main()
