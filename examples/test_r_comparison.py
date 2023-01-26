from multiprocessing import cpu_count

import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

from epioncho_ibm import Params, Simulation, TreatmentParams
from epioncho_ibm.state.params import BlackflyParams

mpl.use("Agg")

benchmark_iters = 1

save_file = "./test.hdf5"


def run_sim(i):
    simulation = Simulation.restore(save_file)
    x = []
    y = []
    for state in simulation.iter_run(end_time=115, sampling_interval=0.1):
        print(f"{state.current_time=}")
        print(f"prev = {state.mf_prevalence_in_population(return_nan=True)}")
        print()
        x.append(state.current_time)
        y.append(state.mf_prevalence_in_population(return_nan=True))

    return x, y


def main():
    params = Params(
        delta_time_days=1,
        year_length_days=366,
        treatment=TreatmentParams(start_time=80, stop_time=105),
        n_people=400,
        blackfly=BlackflyParams(bite_rate_per_person_per_year=290),
    )
    simulation = Simulation(start_time=0, params=params, verbose=True)
    simulation.run(end_time=75)
    simulation.save(save_file)

    list_of_stats: list[tuple[list[float], list[float]]] = process_map(
        run_sim, range(benchmark_iters), max_workers=cpu_count()
    )

    # print(np.mean(list_of_stats))
    # print(np.std(list_of_stats))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(list(list_of_stats))
    ax.plot(list_of_stats[0][0], list_of_stats[0][1])
    fig.savefig("temp.png")


if __name__ == "__main__":
    main()
