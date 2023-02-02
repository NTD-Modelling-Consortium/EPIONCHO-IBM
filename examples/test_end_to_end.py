from multiprocessing import cpu_count

import numpy as np
from tqdm.contrib.concurrent import process_map

from epioncho_ibm import EndgameSimulation, EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv

endgame = """
{
    "parameters": {
        "initial": {
            "n_people": 100
        },
        "changes": [
            {
                "year": 2018,
                "month": 1,
                "params": {
                    "humans":{
                        "min_skinsnip_age": 5
                    }
                }
            },
            {
                "year": 2026,
                "month": 1,
                "params": {
                    "humans":{
                        "min_skinsnip_age": 2
                    }
                }
            }
        ]
    },
    "programs": [
        {
            "first_year": 2018,
            "last_year": 2025,
            "interventions": {
                "treatment_interval": 1
            }
        },
        {
            "first_year": 2027,
            "last_year": 2030,
            "interventions": {
                "treatment_interval": 0.5
            }
        }
    ]
}
"""

save_file = "./test.hdf5"


def run_sim(i) -> Data:
    sim = EndgameSimulation.restore(save_file)
    run_data: Data = {}
    for state in sim.iter_run(end_time=2030, sampling_interval=1):
        add_state_to_run_data(state, run_data=run_data)
    return run_data


params = EpionchoEndgameModel.parse_raw(endgame)
simulation = EndgameSimulation(
    start_time=2015, endgame=params, verbose=True, debug=True
)

print("First years without treatment:")
simulation.run(end_time=2018.0)


simulation.save(save_file)

run_iters = 5

if __name__ == "__main__":
    data: list[Data] = process_map(
        run_sim,
        range(run_iters),
        max_workers=cpu_count(),
    )
    write_data_to_csv(data, "test.csv")
