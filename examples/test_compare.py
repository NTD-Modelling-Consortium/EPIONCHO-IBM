from epioncho_ibm import (
    Params,
    TreatmentParams,
    make_state_from_params,
)
params = Params(treatment=TreatmentParams(start_time=3))

state = make_state_from_params(params=params, n_people=400)
state.run_simulation(start_time=0, end_time=100)
