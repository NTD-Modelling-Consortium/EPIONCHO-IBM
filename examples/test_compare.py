from epioncho_ibm import (
    Params,
    TreatmentParams,
    State,
)
params = Params(treatment=TreatmentParams(start_time=3))

state = State(params=params, n_people=400)
state.run_simulation(start_time=0, end_time=100)
