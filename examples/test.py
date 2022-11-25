from epioncho_ibm import HumanParams, Params, State, TreatmentParams
from epioncho_ibm.state import make_state_from_params

params = Params(treatment=TreatmentParams(start_time=0))
state = make_state_from_params(params=params, n_people=440)

for i in range(12):
    state.run_simulation(start_time=i * 10, end_time=(i + 1) * 10, verbose=True)
    print("run", str(i))
    print(state.microfilariae_per_skin_snip())
    print(state.mf_prevalence_in_population())
