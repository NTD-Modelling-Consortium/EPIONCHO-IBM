import csv

from epioncho_ibm import Params, Simulation
from epioncho_ibm.state.params import BlackflyParams

params = Params(
    delta_time_days=1,
    n_people=400,
    blackfly=BlackflyParams(bite_rate_per_person_per_year=41922),
)

simulation = Simulation(start_time=0, params=params, verbose=True)

out_csv = open("epilepsy.csv", "w")
csv_write = csv.writer(out_csv)
for state in simulation.iter_run(end_time=200, sampling_interval=0.1):
    prevalence_OAE = state.count_OAE() / state.n_people
    csv_write.writerow([prevalence_OAE])
