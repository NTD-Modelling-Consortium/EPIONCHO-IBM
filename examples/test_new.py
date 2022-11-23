from epioncho_ibm import HumanParams, Params, State, TreatmentParams

import numpy as np

np.random.seed(0)

params = Params()
state = State(params=params, n_people=10)
state.run_simulation(start_time=0, end_time=5, verbose=True)
