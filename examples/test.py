from epioncho_ibm import Params, Simulation, TreatmentParams

params = Params(treatment=TreatmentParams(start_time=0), n_people=440)
simulation = Simulation(start_time=0, params=params, verbose=True)

for i in range(12):
    simulation.run(end_time=(i + 1) * 10)
    print("run", str(i))
    print(simulation.state.microfilariae_per_skin_snip())
    print(simulation.state.mf_prevalence_in_population())
