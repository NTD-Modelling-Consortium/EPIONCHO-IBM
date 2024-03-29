from multiprocessing import cpu_count

import numpy as np
from tqdm.contrib.concurrent import process_map

from epioncho_ibm import BlackflyParams, Params, Simulation, TreatmentParams
from epioncho_ibm.state.params import MicrofilParams
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


def run_sim(i) -> Data:
    params = Params(
        delta_time_days=1,
        year_length_days=366,
        treatment=TreatmentParams(start_time=80, stop_time=105),
        n_people=400,
        blackfly=BlackflyParams(
            delta_h_zero=0.186,
            delta_h_inf=0.003,
            c_h=0.005,
            bite_rate_per_person_per_year=294,
            gonotrophic_cycle_length=0.0096,
        ),
    )

    sim = Simulation(start_time=2020, params=params, debug=True)
    run_data: Data = {}
    for state in sim.iter_run(end_time=2060, sampling_interval=1):
        add_state_to_run_data(
            state,
            run_data=run_data,
            number=False,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=False,
            prevalence=True,
            mean_worm_burden=False,
        )
    return run_data


run_iters = 200

if __name__ == "__main__":
    data: list[Data] = process_map(
        run_sim,
        range(run_iters),
        max_workers=cpu_count(),
    )
    write_data_to_csv(data, "test.csv")
