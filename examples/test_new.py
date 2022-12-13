import tempfile

import numpy as np

from epioncho_ibm import (
    Params,
    TreatmentParams,
    make_state_from_hdf5,
    make_state_from_params,
)

np.random.seed(0)

params = Params(treatment=TreatmentParams(start_time=3))

state = make_state_from_params(params=params, n_people=400)
state.run_simulation(start_time=0, end_time=5, verbose=True)
print(state._people)
print(state.mf_prevalence_in_population())

with tempfile.TemporaryFile() as f:
    state.to_hdf5(f)
    new_state = make_state_from_hdf5(f)
    print(new_state.mf_prevalence_in_population())
    assert state == new_state
