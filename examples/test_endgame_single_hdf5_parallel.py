from multiprocessing import cpu_count

import h5py
import numpy as np
from tqdm.contrib.concurrent import process_map

from epioncho_ibm import EndgameSimulation, EpionchoEndgameModel


def get_endgame(seed, cov):
    return {
        "parameters": {
            "initial": {"n_people": 100, "seed": seed},
            "changes": []
        },
        "programs": [
            {
                "first_year": 3,
                "first_month": 1,
                "last_year": 4,
                "last_month": 12,
                "interventions": {
                    "treatment_interval": 0.5,
                    "total_population_coverage": cov,
                },
            },
            {
                "first_year": 5,
                "first_month": 2,
                "last_year": 10,
                "last_month": 7,
                "interventions": {
                    "treatment_interval": 0.5,
                    "total_population_coverage": cov,
                },
            },
        ],
    }


param_sets = [{"seed": 1, "cov": 0.5}, {"seed": 2, "cov": 0.6}, {"seed": 3, "cov": 0.7}]
new_file = h5py.File("test_one.hdf5", "w")


def run_sim(endgame_structure):
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)
    endgame_sim = EndgameSimulation(
        start_time=0, endgame=endgame, verbose=False, debug=True
    )
    endgame_sim.run(end_time=1)
    grp = new_file.create_group(
        f"iter_{str(endgame_sim.simulation.get_current_params().seed)}"
    )
    endgame_sim.save(grp)
    return endgame_sim.simulation.state.mf_prevalence_in_population()


endgame_structures = [
    get_endgame(param_set["seed"], param_set["cov"]) for param_set in param_sets
]


if __name__ == "__main__":
    list_of_stats: list[float] = process_map(
        run_sim, endgame_structures, max_workers=cpu_count()
    )

    print(np.mean(list_of_stats))
    print(np.std(list_of_stats))
