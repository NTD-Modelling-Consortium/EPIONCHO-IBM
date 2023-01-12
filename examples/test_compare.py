from epioncho_ibm import Params, Simulation, TreatmentParams

params = Params(treatment=TreatmentParams(start_time=3))
simulation = Simulation(start_time=0, params=params, n_people=400)
simulation.run(end_time=100)
