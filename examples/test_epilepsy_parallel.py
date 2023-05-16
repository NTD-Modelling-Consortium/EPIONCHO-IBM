import csv
from multiprocessing import cpu_count

from tqdm.contrib.concurrent import process_map

from epioncho_ibm import Params, Simulation
from epioncho_ibm.state.params import BlackflyParams

benchmark_iters = 100


def run_sim(i) -> list[float]:
    params = Params(
        delta_time_days=1,
        year_length_days=366,
        n_people=400,
        blackfly=BlackflyParams(
            delta_h_zero=0.186,
            delta_h_inf=0.003,
            c_h=0.005,
            bite_rate_per_person_per_year=41922,
            gonotrophic_cycle_length=0.0096,
        ),
    )

    simulation = Simulation(start_time=0, params=params)
    prev = []
    for state in simulation.iter_run(end_time=100, sampling_interval=0.1):
        prevalence_OAE = state.count_OAE() / state.n_people
        prev.append(prevalence_OAE)

    return prev


if __name__ == "__main__":
    list_of_stats: list[list[float]] = process_map(
        run_sim, range(benchmark_iters), max_workers=cpu_count()
    )
    out_csv = open("epilepsy.csv", "w")
    csv_write = csv.writer(out_csv)
    new_stats: dict[int, list[float]] = {}
    for run in list_of_stats:
        for i, v in enumerate(run):
            if i in new_stats:
                new_stats[i].append(v)
            else:
                new_stats[i] = [v]

    csv_write.writerows(new_stats.values())
