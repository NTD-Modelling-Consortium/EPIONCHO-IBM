import os
from functools import partial

import h5py
import numpy as np
import pandas as pd
from generate_model_input import get_endgame
from plot_outputs import plot_outputs
from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


# Function to run and save simulations
def run_sim(
    i,
    iu_name,
    verbose=False,
    sample=True,
    samp_interval=1,
    mox_interval=1,
    abr=1641,
    immigration_rate=0,
    new_blackfly=0,
):
    endgame_structure = get_endgame(
        i,
        iu_name,
        sample=sample,
        mox_interval=mox_interval,
        abr=abr,
        immigration_rate=immigration_rate,
        new_blackfly=new_blackfly,
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
    for state in endgame_sim.iter_run(end_time=2041, sampling_interval=samp_interval):

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
            with_pnc=False,
            saving_multiple_states=True,
        )
        add_state_to_run_data(
            state,
            run_data=run_data_age,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=False,
            prevalence=True,
            mean_worm_burden=False,
            prevalence_OAE=False,
            intensity=True,
            with_sequela=False,
            with_pnc=False,
        )
    # new_file = h5py.File(f"test_one{i}.hdf5", "w")
    # grp = new_file.create_group(f"draw_{str(i)}")
    # endgame_sim.save(grp)
    return (run_data, run_data_age)


def run_simulations(
    verbose,
    iu_name,
    sample,
    sample_interval,
    mox_interval,
    ranges,
    max_workers,
    desc,
    abr=1641,
    immigration_rate=0,
    new_blackfly=0,
):
    rumSim = partial(
        run_sim,
        verbose=verbose,
        iu_name=iu_name,
        sample=sample,
        samp_interval=sample_interval,
        mox_interval=mox_interval,
        abr=abr,
        immigration_rate=immigration_rate,
        new_blackfly=new_blackfly,
    )
    datas: list[tuple[Data, Data]] = process_map(
        rumSim, ranges, max_workers=max_workers
    )
    data: list[Data] = [row[0] for row in datas]
    age_data: list[Data] = [row[1] for row in datas]
    write_data_to_csv(
        data,
        "test_outputs/python_model_output/testing_"
        + iu_name
        + "-new_run_"
        + desc
        + ".csv",
    )
    write_data_to_csv(
        age_data,
        "test_outputs/python_model_output/testing_"
        + iu_name
        + "-new_run_"
        + desc
        + "_age-grouped"
        + ".csv",
    )
    plot_outputs("testing_" + iu_name + "-new_run_" + desc + "_age-grouped")


# Wrapper
def wrapped_parameters(iu_name):
    abr_vals = [176, 300, 700, 1000]
    immigration_rate = 0.04
    # Run simulations and save output
    num_iter = 100
    max_workers = os.cpu_count() if num_iter > os.cpu_count() else num_iter
    for abr_val in abr_vals:

        # import_value = 0
        # run_simulations(
        #     False,
        #     iu_name,
        #     True,
        #     1,
        #     1,
        #     range(num_iter),
        #     max_workers,
        #     "no_immigration_" + str(abr_val),
        #     abr=abr_val,
        #     immigration_rate=0,
        #     new_blackfly=import_value,
        # )

        import_value = 2
        run_simulations(
            False,
            iu_name,
            True,
            1,
            1,
            range(num_iter),
            max_workers,
            "immigration_"
            + str(int(immigration_rate * 100))
            + "_num_worms_"
            + str(import_value)
            + "_"
            + str(abr_val),
            abr=abr_val,
            immigration_rate=immigration_rate,
            new_blackfly=import_value,
        )

        import_value = 4
        run_simulations(
            False,
            iu_name,
            True,
            1,
            1,
            range(num_iter),
            max_workers,
            "immigration_"
            + str(int(immigration_rate * 100))
            + "_num_worms_"
            + str(import_value)
            + "_"
            + str(abr_val),
            abr=abr_val,
            immigration_rate=immigration_rate,
            new_blackfly=import_value,
        )

        import_value = 6
        run_simulations(
            False,
            iu_name,
            True,
            1,
            1,
            range(num_iter),
            max_workers,
            "immigration_"
            + str(int(immigration_rate * 100))
            + "_num_worms_"
            + str(import_value)
            + "_"
            + str(abr_val),
            abr=abr_val,
            immigration_rate=immigration_rate,
            new_blackfly=import_value,
        )

        import_value = 8
        run_simulations(
            False,
            iu_name,
            True,
            1,
            1,
            range(num_iter),
            max_workers,
            "immigration_"
            + str(int(immigration_rate * 100))
            + "_num_worms_"
            + str(import_value)
            + "_"
            + str(abr_val),
            abr=abr_val,
            immigration_rate=immigration_rate,
            new_blackfly=import_value,
        )

        import_value = 10
        run_simulations(
            False,
            iu_name,
            True,
            1,
            1,
            range(num_iter),
            max_workers,
            "immigration_"
            + str(int(immigration_rate * 100))
            + "_num_worms_"
            + str(import_value)
            + "_"
            + str(abr_val),
            abr=abr_val,
            immigration_rate=immigration_rate,
            new_blackfly=import_value,
        )

        import_value = 25
        run_simulations(
            False,
            iu_name,
            True,
            1,
            1,
            range(num_iter),
            max_workers,
            "immigration_"
            + str(int(immigration_rate * 100))
            + "_num_worms_"
            + str(import_value)
            + "_"
            + str(abr_val),
            abr=abr_val,
            immigration_rate=immigration_rate,
            new_blackfly=import_value,
        )

        import_value = 50
        run_simulations(
            False,
            iu_name,
            True,
            1,
            1,
            range(num_iter),
            max_workers,
            "immigration_"
            + str(int(immigration_rate * 100))
            + "_num_worms_"
            + str(import_value)
            + "_"
            + str(abr_val),
            abr=abr_val,
            immigration_rate=immigration_rate,
            new_blackfly=import_value,
        )

        # import_value = 1000
        # run_simulations(
        #     False,
        #     iu_name,
        #     True,
        #     1,
        #     1,
        #     range(num_iter),
        #     max_workers,
        #     "immigration_"
        #     + str(int(immigration_rate * 100))
        #     + "_num_worms_"
        #     + str(import_value)
        #     + "_"
        #     + str(abr_val),
        #     abr=abr_val,
        #     immigration_rate=immigration_rate,
        #     new_blackfly=import_value,
        # )

        # import_value = 2000
        # run_simulations(
        #     False,
        #     iu_name,
        #     True,
        #     1,
        #     1,
        #     range(num_iter),
        #     max_workers,
        #     "immigration_"
        #     + str(int(immigration_rate * 100))
        #     + "_num_worms_"
        #     + str(import_value)
        #     + "_"
        #     + str(abr_val),
        #     abr=abr_val,
        #     immigration_rate=immigration_rate,
        #     new_blackfly=import_value,
        # )

    # run_simulations(
    #    False, iu_name, True, 1, 0.5, range(num_iter), max_workers, "save_hdf5_test_l3_176", abr=176
    # )


if __name__ == "__main__":
    # Run example
    # iu = "GHA0216121382"
    iu = "CIV0162715440"
    wrapped_parameters(iu)
