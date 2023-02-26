import h5py

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv

hdf5_file = "test_one.hdf5"
scenario_file = "scenario2a.json"
n_sims = 3

group_names = [f"draw_{i}" for i in range(n_sims)]


new_file = h5py.File(hdf5_file, "r")
sims = []
output_data: list[Data] = []
for group_name in group_names:
    restored_grp = new_file[group_name]
    assert isinstance(restored_grp, h5py.Group)
    sim = EndgameSimulation.restore(restored_grp)
    new_endgame_model = EpionchoEndgameModel.parse_file(scenario_file)

    current_params = sim.simulation.get_current_params()
    sim.reset_endgame(new_endgame_model)
    new_params = sim.simulation.get_current_params()

    # Save out attributes to keep
    new_params.blackfly.bite_rate_per_person_per_year = (
        current_params.blackfly.bite_rate_per_person_per_year
    )
    new_params.gamma_distribution = current_params.gamma_distribution
    new_params.seed = current_params.seed

    sim.simulation.reset_current_params(new_params)

    run_data: Data = {}
    for state in sim.iter_run(end_time=2040, sampling_interval=1):
        add_state_to_run_data(
            state,
            run_data=run_data,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=True,
            prevalence=True,
            mean_worm_burden=False,
        )
    output_data.append(run_data)

write_data_to_csv(output_data, "test.csv")
