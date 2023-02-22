from multiprocessing import cpu_count

import numpy as np
from tqdm.contrib.concurrent import process_map

from epioncho_ibm import EndgameSimulation, EpionchoEndgameModel

benchmark_iters = 400

endgame = """
{
    "parameters": {
        "initial": {
            "n_people": 100
        },
        "changes": [
            {
                "year": 2,
                "month": 1,
                "params": {
                    "delta_time_days": 1
                }
            },
            {
                "year": 8,
                "month": 1,
                "params": {
                    "delta_time_days": 0.5
                }
            }
        ]
    },
    "programs": [
        {
            "first_year": 3,
            "first_month": 1,
            "last_year": 4,
            "last_month": 12,
            "interventions": {
                "treatment_interval": 0.5
            }
        },
        {
            "first_year": 5,
            "first_month": 2,
            "last_year": 10,
            "last_month": 7,
            "interventions": {
                "treatment_interval": 0.5
            }
        }
    ]
}
"""


def run_sim(i):
    params = EpionchoEndgameModel.parse_raw(endgame)
    endgame_sim = EndgameSimulation(
        start_time=0, endgame=params, verbose=False, debug=True
    )
    endgame_sim.run(end_time=1)
    return endgame_sim.simulation.state.mf_prevalence_in_population()


if __name__ == "__main__":
    list_of_stats: list[float] = process_map(
        run_sim, range(benchmark_iters), max_workers=cpu_count()
    )

    print(np.mean(list_of_stats))
    print(np.std(list_of_stats))
