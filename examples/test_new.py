import tempfile

import numpy as np

from epioncho_ibm import Params, Simulation, TreatmentParams, make_state_from_hdf5

np.random.seed(0)

params = Params(treatment=TreatmentParams(start_time=5))

simulation = Simulation(start_time=0, params=params, n_people=400, verbose=True)
print("First years without treatment:")
simulation.run(end_time=5.0)
# print(simulation.state.people)
print(simulation.state.mf_prevalence_in_population())

print("Starting treatment:")
for state in simulation.iter_run(end_time=10, sampling_interval=0.1):
    print(
        f"Time: {state.current_time:.2f}. Prevalence: {state.mf_prevalence_in_population() * 100:.2f}%"
    )

with tempfile.TemporaryFile() as f:
    simulation.state.to_hdf5(f)
    new_state = make_state_from_hdf5(f)
    print("Final prevalence:", new_state.mf_prevalence_in_population())
    assert simulation.state == new_state
