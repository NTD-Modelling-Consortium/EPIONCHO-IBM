import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv

from epioncho_ibm import Params, Simulation
from epioncho_ibm.state.params import BlackflyParams

params = Params(
    delta_time_days=1,
    year_length_days=366,
    n_people=440,
    blackfly=BlackflyParams(
        delta_h_zero=0.186,
        delta_h_inf=0.003,
        c_h=0.005,
        bite_rate_per_person_per_year=2297,
        gonotrophic_cycle_length=0.0096,
    ),
    sequela_active=[
        "HangingGroin",
        "Atrophy",
        "Blindness",
        "APOD",
        "CPOD",
        "RSD",
        "Depigmentation",
        "SevereItching",
    ],
)

simulation = Simulation(start_time=0, params=params, verbose=True)

out_csv = open("epilepsy.csv", "w")
csv_write = csv.writer(out_csv)
for state in simulation.iter_run(end_time=100, sampling_interval=0.1):
    print(state.mf_prevalence_in_population())
    print(state.sequalae_prevalence())
