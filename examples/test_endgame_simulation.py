import numpy as np

from epioncho_ibm import EndgameSimulation, EpionchoEndgameModel, State

np.random.seed(0)

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
                    "delta_time": 0.001
                }
            },
            {
                "year": 8,
                "month": 1,
                "params": {
                    "delta_time": 0.003
                }
            }
        ]
    },
    "programs": [
        {
            "first_year": 3,
            "first_month": 1,
            "last_year": 5,
            "last_month": 1,
            "interventions": {
                "treatment_interval": 0.5
            }
        },
        {
            "first_year": 5,
            "first_month": 2,
            "last_year": 10,
            "last_month": 8,
            "interventions": {
                "treatment_interval": 0.5
            }
        }
    ]
}
"""

params = EpionchoEndgameModel.parse_raw(endgame)
simulation = EndgameSimulation(start_time=0, endgame=params, verbose=True, debug = True)
simulation.run(end_time=10)
