import numpy as np

from epioncho_ibm import (
    HumanParams,
    Params,
    State,
    TreatmentParams,
    make_state_from_hdf5,
    make_state_from_params,
)

np.random.seed(0)

params = Params(treatment=TreatmentParams(start_time=3))

prevalence = np.zeros(20)

# for i in range(20):
state = make_state_from_params(params=params, n_people=10)
state.run_simulation(start_time=0, end_time=5, verbose=True)
prevalence = state.mf_prevalence_in_population()
# print(state._people)
print(prevalence)
state.to_hdf5("test.hdf5")
new_state = make_state_from_hdf5("test.hdf5")
print(new_state.mf_prevalence_in_population())
