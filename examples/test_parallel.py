from multiprocessing import cpu_count

import numpy as np
from tqdm.contrib.concurrent import process_map

from epioncho_ibm import Params, Simulation, TreatmentParams

benchmark_iters = 40


def run_sim(i):
    params = Params(treatment=TreatmentParams(start_time=0.1, interval_years=0.1))
    simulation = Simulation(start_time=0, params=params, n_people=400)
    simulation.run(end_time=1)
    return simulation.state.mf_prevalence_in_population()


list_of_stats: list[float] = process_map(
    run_sim, range(benchmark_iters), max_workers=cpu_count()
)

print(np.mean(list_of_stats))
print(np.std(list_of_stats))
