import csv

from epioncho_ibm import HumanParams, Params, State, TreatmentParams

params = Params(treatment=TreatmentParams(start_time=100))
state = State(params=params, n_people=440)


output_stats = state.run_simulation_output_stats(
    sampling_interval=0.5, start_time=0, end_time=120, verbose=True
)
with open("test.csv", "w+") as f:
    writer = csv.writer(f)
    writer.writerows(
        [[time, stat.population_prevalence] for time, stat in output_stats]
    )
