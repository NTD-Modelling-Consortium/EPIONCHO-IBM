from multiprocessing import cpu_count

import numpy as np
from tqdm.contrib.concurrent import process_map

from epioncho_ibm import Params, Simulation, TreatmentParams
import h5py
benchmark_iters = 10

new_file = h5py.File('test_one.hdf5', 'w')

def run_sim(i):
    params = Params(
        treatment=TreatmentParams(start_time=0.1, interval_years=0.1, stop_time=130),
        n_people=400,
    )
    simulation = Simulation(start_time=0, params=params)
    simulation.run(end_time=1)
    grp = new_file.create_group(f'iter_{str(i)}')
    simulation.save(grp)
    restored_grp = new_file[f'iter_{str(i)}']
    assert isinstance(restored_grp, h5py.Group)
    res_sim = Simulation.restore(restored_grp)
    return res_sim.state.mf_prevalence_in_population()


if __name__ == "__main__":
    list_of_stats: list[float] = process_map(
        run_sim, range(benchmark_iters), max_workers=cpu_count()
    )

    print(np.mean(list_of_stats))
    print(np.std(list_of_stats))
