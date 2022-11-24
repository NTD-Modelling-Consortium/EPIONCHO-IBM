import numpy as np

from epioncho_ibm import HumanParams, Params, State, TreatmentParams

np.random.seed(0)

params = Params(treatment=TreatmentParams(start_time=3))

prevalence = np.zeros(20)

# for i in range(20):
state = State(params=params, n_people=10)
state.run_simulation(start_time=0, end_time=5, verbose=True)
prevalence = state.mf_prevalence_in_population()
print(state._people)
print(prevalence)
