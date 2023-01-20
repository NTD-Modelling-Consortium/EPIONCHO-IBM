import numpy as np

from epioncho_ibm import EndgameSimulation, EpionchoEndgameModel, Simulation

import tempfile
import csv
import numpy as np

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
                    "delta_time": 0.0016
                }
            },
            {
                "year": 2028,
                "month": 1,
                "params": {
                    "delta_time": 0.003
                }
            }
        ]
    },
    "programs": [
        {
            "first_year": 2025,
            "first_month": 2,
            "last_year": 2030,
            "last_month": 8,
            "interventions": {
                "treatment_interval": 0.5
            }
        }
    ]
}
"""

params = EpionchoEndgameModel.parse_raw(endgame)
simulation = EndgameSimulation(start_time=2020, endgame=params, verbose=True, debug = True)

print("First years without treatment:")
simulation.run(end_time=2025.0)

Year = float
AgeStart = float
Prevalence = float

def run_sim(sim) -> dict[Year, dict[AgeStart, Prevalence]]:
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

run_iters = 5
with tempfile.TemporaryFile() as f:
    simulation.save(f)
    data = [run_sim(Simulation.restore(f)) for _ in range(run_iters)]


data_by_year: dict[Year, dict[AgeStart, list[Prevalence]]] = {}
for i, run in enumerate(data):
    for year, year_data in run.items():
        if year not in data_by_year:
            data_by_year[year] = {age_start: [prev] for age_start, prev in year_data.items()}
        else:
            rel_year = data_by_year[year]
            for age_start, prev in year_data.items():
                if age_start not in rel_year:
                    rel_year[age_start] = [prev]
                else:
                    rel_year[age_start] = rel_year[age_start] + [prev]
            data_by_year[year] = rel_year

f = open('test.csv', 'w')

# create the csv writer
writer = csv.writer(f)
first_elem: list[str] = ['year_id', 'age_start', 'age_end', 'measure'] + [f'draw_{i}' for i in range(run_iters)]
excel_data: list[list[str | float] | list[str] | list[float]] = [first_elem]
writer.writerow(first_elem)
for year, year_data in data_by_year.items():
    for age_start, prevalences in year_data.items():
        row = [year, age_start, age_start + 1, 'prevalence'] + prevalences
        excel_data.append(row)
        writer.writerow(row)
f.close()

# write a row to the csv file

print(excel_data)