from epioncho_ibm import HumanParams, Params, State, TreatmentParams

params = Params()
state = State(params=params, n_people=10)
state.run_simulation(start_time=0, end_time=0.3, verbose=True)
