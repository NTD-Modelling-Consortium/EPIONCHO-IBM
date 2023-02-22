from epioncho_ibm import Params, Simulation, TreatmentParams

params = Params(treatment=TreatmentParams(start_time=3, stop_time=130), n_people=400)
simulation = Simulation(start_time=0, params=params)
simulation.run(end_time=100)
