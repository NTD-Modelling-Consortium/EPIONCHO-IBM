import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from multiprocessing import cpu_count

from tqdm.contrib.concurrent import process_map

from epioncho_ibm import BlackflyParams, Params, Simulation
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


def run_sim(i) -> Data:
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

    sim = Simulation(start_time=0, params=params, debug=True)
    run_data: Data = {}
    for state in sim.iter_run(end_time=100, sampling_interval=1):
        add_state_to_run_data(
            state,
            run_data=run_data,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=True,
            intensity=False,
            prevalence=True,
            prevalence_OAE=True,
            mean_worm_burden=False,
            with_sequela=True,
        )
    return run_data


run_iters = 100

if __name__ == "__main__":
    data: list[Data] = process_map(
        run_sim,
        range(run_iters),
        max_workers=cpu_count() + 4,
    )
    write_data_to_csv(data, "test.csv")
