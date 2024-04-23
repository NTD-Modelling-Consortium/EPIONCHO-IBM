import json
import os

import h5py
from generate_model_input import get_endgame, run_simulations

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data


# Function to run and save simulations
def run_sim(
    i, verbose=False, samp_interval=1, mox_interval=1, end_time=2041, scenario_file=""
):

    # Read in endgame objects and set up simulation
    hdf5_file = h5py.File(
        "test_outputs/OutputVals_GHA0216121382_BC.hdf5",
        "r",
    )
    restored_file_to_use = hdf5_file[f"draw_{i}"]
    restored_endgame_sim = EndgameSimulation.restore(restored_file_to_use)
    current_params = restored_endgame_sim.simulation.get_current_params()

    new_endgame_structure = get_endgame(
        i, "hdf5_import", False, mox_interval=mox_interval
    )
    if scenario_file != "":
        with open(scenario_file, "r") as file:
            new_endgame_structure = json.load(file)

    new_endgame = EpionchoEndgameModel.parse_obj(new_endgame_structure)
    new_endgame.parameters.initial.blackfly.bite_rate_per_person_per_year = (
        current_params.blackfly.bite_rate_per_person_per_year
    )
    new_endgame.parameters.initial.gamma_distribution = (
        current_params.gamma_distribution
    )
    new_endgame.parameters.initial.seed = current_params.seed

    restored_endgame_sim.simulation.state.current_time = 2026
    restored_endgame_sim.reset_endgame(new_endgame)
    # Run
    run_data: Data = {}
    run_data_age: Data = {}
    for state in restored_endgame_sim.iter_run(
        end_time=end_time,
        sampling_interval=samp_interval,
        make_time_backwards_compatible=True,
    ):

        add_state_to_run_data(
            state,
            run_data=run_data,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=False,
            prevalence=True,
            mean_worm_burden=False,
            prevalence_OAE=False,
            intensity=False,
            with_sequela=False,
            with_pnc=False,
            saving_multiple_states=True,
        )
        add_state_to_run_data(
            state,
            run_data=run_data_age,
            number=True,
            n_treatments=True,
            achieved_coverage=True,
            with_age_groups=True,
            prevalence=True,
            mean_worm_burden=True,
            prevalence_OAE=True,
            intensity=True,
            with_sequela=True,
            with_pnc=True,
        )

    return (run_data, run_data_age)


# Wrapper
def wrapped_parameters(iu_name):
    # Run simulations and save output
    num_iter = 10
    end_time = 2040
    max_workers = os.cpu_count() if num_iter > os.cpu_count() else num_iter
    run_simulations(
        run_sim,
        False,
        iu_name,
        True,
        1,
        1,
        end_time,
        range(num_iter),
        max_workers,
        "mox_annual_1year",
        scenario_file="",
    )

    run_simulations(
        run_sim,
        False,
        iu_name,
        True,
        1,
        0.5,
        end_time,
        range(num_iter),
        max_workers,
        "mox_biannual_1year",
        scenario_file="",
    )

    run_simulations(
        run_sim,
        False,
        iu_name,
        True,
        1,
        1,
        end_time,
        range(num_iter),
        max_workers,
        "mox_quadannual_1year",
        scenario_file="",
    )

    run_simulations(
        run_sim,
        False,
        iu_name,
        True,
        1,
        1,
        end_time,
        range(num_iter),
        max_workers,
        "mox_annual_1year",
        scenario_file="test_outputs/scenario1di.json",
    )


if __name__ == "__main__":
    # Run example
    iu = "GHA0216121382"
    # iu = "CIV0162715440"
    wrapped_parameters(iu)
