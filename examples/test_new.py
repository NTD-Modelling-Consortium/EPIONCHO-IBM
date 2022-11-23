import numpy as np

from epioncho_ibm import HumanParams, Params, State, TreatmentParams

np.random.seed(0)

params = Params(treatment=TreatmentParams(start_time=3))
state = State(params=params, n_people=10)
state.run_simulation(start_time=0, end_time=5, verbose=True)
print(state._people)
