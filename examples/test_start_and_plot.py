import matplotlib.pyplot as plt

from epioncho_ibm import Params, Simulation
from epioncho_ibm.state.params import BlackflyParams


def run_sim(i):
    params = Params(
        delta_time_days=1,
        year_length_days=366,
        treatment=None,
        n_people=400,
        blackfly=BlackflyParams(
            delta_h_zero=0.186,
            delta_h_inf=0.003,
            c_h=0.005,
            bite_rate_per_person_per_year=294,
            gonotrophic_cycle_length=0.0096,
        ),
    )
    simulation = Simulation(start_time=0, params=params)
    x = []
    y = []
    for state in simulation.iter_run(end_time=25, sampling_interval=0.1):
        x.append(state.current_time)
        y.append(state.mf_prevalence_in_population(return_nan=True))
    return x, y


year, prev = run_sim(1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(year, prev)
fig.savefig("temp.png")
