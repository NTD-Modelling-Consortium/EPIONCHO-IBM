import os

import h5py
from generate_model_input import get_endgame, run_simulations

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data


# Function to run and save simulations
def run_sim(
    i,
    iu_name,
    verbose=False,
    sample=True,
    samp_interval=1,
    mox_interval=1,
    abr=1641,
    end_time=2041,
    scenario_file="",
):
    endgame_structure = get_endgame(
        i,
        iu_name,
        sample=sample,
        mox_interval=mox_interval,
        abr=abr,
    )
    # Read in endgame objects and set up simulation
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)
    # print(endgame)
    endgame_sim = EndgameSimulation(
        start_time=1900, endgame=endgame, verbose=verbose, debug=True
    )
    # Run
    run_data: Data = {}
    run_data_age: Data = {}
    for state in endgame_sim.iter_run(
        end_time=end_time, sampling_interval=samp_interval
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
            intensity=True,
            with_sequela=False,
            with_pnc=True,
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
    # testing saving to hdf5 for first 2 files
    if i <= 2:
        new_file = h5py.File(f"test_outputs/test_one{i}.hdf5", "w")
        grp = new_file.create_group(f"draw_{str(i)}")
        endgame_sim.save(grp)
    return (run_data, run_data_age)


# Wrapper
def wrapped_parameters(iu_name):
    # Run simulations and save output
    num_iter = 100
    max_workers = os.cpu_count() if num_iter > os.cpu_count() else num_iter
    run_simulations(
        run_sim,
        False,
        iu_name,
        True,
        1,
        1,
        range(num_iter),
        max_workers,
        "mox_annual",
        abr=1641,
    )

    run_simulations(
        run_sim,
        False,
        iu_name,
        True,
        0.5,
        0.5,
        range(num_iter),
        max_workers,
        "mox_biannual",
        abr=1641,
    )

    run_simulations(
        run_sim,
        False,
        iu_name,
        True,
        0.25,
        0.25,
        range(num_iter),
        max_workers,
        "mox_quadannual",
        abr=1641,
    )


if __name__ == "__main__":
    # Run example
    # iu = "GHA0216121382"
    iu = "CIV0162715440"
    wrapped_parameters(iu)
