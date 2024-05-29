import numpy as np

from epioncho_ibm import Params, Simulation, TreatmentParams, make_state_from_hdf5

params = Params(
    delta_time_days=2,
    treatment=TreatmentParams(start_time=5, stop_time=130),
    n_people=400,
    seed=0,
)

simulation = Simulation(start_time=0, params=params, verbose=True)
simulation.run(end_time=10.0)

assert params.treatment is not None
params.treatment.microfilaricidal_nu = 0.01
params.treatment.microfilaricidal_omega = 1.5
params.treatment.embryostatic_lambda_max = 30
params.treatment.embryostatic_phi = 10
params.treatment.permanent_infertility = 0.4

simulation.reset_current_params(params)
simulation.run(end_time=12.0)

print(simulation.state.people.last_treatment)
