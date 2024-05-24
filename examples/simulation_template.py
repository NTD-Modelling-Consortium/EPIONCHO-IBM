import os
from functools import partial

import h5py
from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


# You can edit the inputs to this function to set more parameters dynamically
def get_parameters(iter, abr=1641, kE=0.3):
    # all treatment (MDA) that you want to apply will be stored as a list of dictionaries
    # Each dictionary will describe the MDA being applied
    # If you want to apply vector control, it is considered a model change (explained below)
    treatment_program = []

    # changes to the model parameters will also be stored as a list of dictionaries
    changes = []
    # setting the seed of the model is optional, but good practice
    seed = iter + iter * 3758

    # In this example, we are adding 10 years of MDA [1980-1990)
    # applied annually, at 52% total population coverage
    # correlation is a parameter [0, 1] that defines how
    # systematic the coverage is, with 0 being completely random
    # and 1 being completely systematic
    # The default treatment is IVM
    treatment_program.append(
        {
            "first_year": 1980,
            "last_year": 1990,
            # default treatment name is IVM
            "treatment_name": "IVM",
            "interventions": {
                # 1 = annual treatment, 0.5 = bi-annual, etc.
                "treatment_interval": 1,
                "total_population_coverage": 0.52,
                "correlation": 0.5,
            },
        }
    )

    # Now we are adding 10 years of MDA [1994-2005)
    # applied annually, at 65% total population coverage
    # We are also changing some of the parameters of
    # the effectiveness of the treatment
    # for a list of more parameters you can edit
    # see the following objects in their files
    # epioncho_ibm/state/params.py - `TreatmentParams` and `SpecificTreatmentParams`
    # these define the values inside of the "interventions" sub-dictionary
    # endgame-simulations/endgame-simulations/models.py - `Program`
    # these define the values in the first layer of the dictionary
    # note: endgame-simulations is a separate github repo
    treatment_program.append(
        {
            "first_year": 1994,
            "last_year": 2005,
            "interventions": {
                "treatment_name": "MOX",
                "treatment_interval": 1,
                "total_population_coverage": 0.65,
                "correlation": 0.5,
                "min_age_of_treatment": 4,
                "microfilaricidal_nu": 0.04,
                "microfilaricidal_omega": 1.82,
                "embryostatic_lambda_max": 462,
                "embryostatic_phi": 4.83,
            },
        }
    )

    # Vector Control is applied in EPIONCHO-IBM as a reduction in the ABR.
    # The year you want to start Vector Control, we can add a dictionary to the
    # changes list, giving it the year we want to make the change,
    # and the parameters we want to change, in this case the abr.
    changes.append(
        {
            "year": 1970,
            "params": {"blackfly": {"bite_rate_per_person_per_year": abr * 0.2}},
        }
    )

    # Typically, to account for the residual effect of Vector Control,
    # We change the ABR back the year AFTER Vector Control is stopped
    changes.append(
        {
            "year": 1993,
            "params": {"blackfly": {"bite_rate_per_person_per_year": abr}},
        }
    )

    # The MDA applied from 1994 - 2005 also requires us to change the delta time parameter.
    # Here we make that change.
    changes.append({"year": 1994, "params": {"delta_time_days": 0.5}})

    # Finally, we return back a dictionary full of all the parameters that we need to start the model.
    # A full list of the parameters can be found in `epioncho_ibm/state/params.py`
    return {
        "parameters": {
            "initial": {
                "n_people": 400,
                "year_length_days": 365,
                "delta_h_zero": 0.186,
                "c_v": 0.005,
                "delta_h_inf": 0.003,
                "seed": seed,
                "gamma_distribution": kE,
                "delta_time_days": 1,
                "blackfly": {
                    "bite_rate_per_person_per_year": abr,
                },
                "exposure": {"Q": 1.2},
                # Having this in the parameters makes sure that sequela prevalence is calculated
                "sequela_active": [
                    "Blindness",
                    "SevereItching",
                    "RSD",
                    "APOD",
                    "CPOD",
                    "Atrophy",
                    "HangingGroin",
                    "Depigmentation",
                ],
            },
            "changes": changes,
        },
        "programs": treatment_program,
    }


# Function to run and save simulations
def run_simulations(
    i,
    verbose=False,
    sampling_interval=1,
    abr=1641,
    kE=0.3,
    start_time=1900,
    end_time=2005,
):
    endgame_structure = get_parameters(i, abr=abr, kE=kE)

    # Read in endgame objects and set up simulation
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)

    # EndgameSimulation is a type of Simulation that allows for changes to the parameters
    endgame_sim = EndgameSimulation(
        start_time=start_time, endgame=endgame, verbose=verbose, debug=True
    )

    # Run the Simulation and store data
    run_data: Data = {}
    run_data_age: Data = {}
    for state in endgame_sim.iter_run(
        end_time=end_time, sampling_interval=sampling_interval
    ):
        # This is a list of all the default outputs you can get in at each sample point
        # You can also add custom outputs, but accessing the `state` object and outputting
        # of its attributes. This would need to be stored in a separate variable and returned
        # This specific call generates outputs for the full age range
        # However mf prevalence is limited to the minimum age of skin-snipping
        add_state_to_run_data(
            state,
            # variable to output data to
            run_data=run_data,
            # don't age group the data (outputs data for 0-80)
            with_age_groups=False,
            # output the number of people
            number=True,
            # output the number of treatments applied for each round of intervention since the last output
            n_treatments=True,
            # output the achieved coverage in the age group for each round of intervention since the last output
            achieved_coverage=True,
            # output the average mf prevalence
            prevalence=True,
            # output the mean worm burden
            mean_worm_burden=True,
            # output the OAE prevalence
            prevalence_OAE=True,
            # output the mf intensity
            intensity=True,
            # output all the sequela prevalences
            with_sequela=True,
            # Output the percent of people who don't comply
            with_pnc=True,
            # If we are also going to save data again using `add_state_to_run_data`
            # in the same timestep we need this to be True. Otherwise set it to false
            saving_multiple_states=True,
        )

        # Now we will generate output data in 1 year age groups [0, 1), [1, 2), ...., [79, 80)
        add_state_to_run_data(
            state,
            run_data=run_data_age,
            # now we want to age group the data
            with_age_groups=True,
            number=True,
            n_treatments=True,
            achieved_coverage=True,
            prevalence=True,
            mean_worm_burden=True,
            prevalence_OAE=True,
            intensity=True,
            with_sequela=True,
            with_pnc=True,
            # we are not going to use `add_state_to_run_data` at this timestep anymore
            saving_multiple_states=False,
        )
    # helper code for if you need to save the model state to be used to
    # initiate new runs
    # new_file = h5py.File(f"test_outputs/test_one{i}.hdf5", "w")
    # grp = new_file.create_group(f"draw_{str(i)}")
    # endgame_sim.save(grp)
    # return the generated data after the simulation is finished
    return (run_data, run_data_age)


# this is the function that python will start execution with when run
if __name__ == "__main__":

    # How many times we want to run the model for a given set of parameters
    # Typically this value is 200
    num_iters = 50
    # To use parallel processing we want to see how many cores our computer has.
    # If we have more cores than iterations, then we can just use as many cores as
    # we have iterations.
    # Otherwise, to maximize speed, we will use the maximum number of cores - 1
    # to preserve other functions
    max_workers = os.cpu_count() - 1 if num_iters > os.cpu_count() else num_iters
    # make sure at least 1 is selected
    max_workers = max(max_workers, 1)

    # To use parallel processing
    # We need to make a "partial" of the function that we want to run in parallel
    # In our case this is the `run_simulations` function, and we need to pass in certain
    # parameters that will be used to inform the simulation parameters
    # These can be customized to reflect the use case of your simulation
    runSimulations = partial(
        run_simulations,
        # Verbose Output
        verbose=False,
        # How often do we want to output? 1 = every year, 0.5 = every half year, etc.
        sampling_interval=1,
        # The ABR we want to initialize the model with
        abr=1641,
        # The kE value we want to initialize the model with
        kE=0.3,
        # The start time of the model
        start_time=1900,
        # The end time of the model
        end_time=2006,
    )

    # Now we use process_map to call the function we defined above
    # and tell it the number of runs we want to do
    # and the number of workers we can use.
    # It will output the data for each run in a list
    datas: list[tuple[Data, Data]] = process_map(
        run_simulations, range(num_iters), max_workers=max_workers
    )

    # this is what we use to seperate the age grouped outputs
    # from the all age outputs
    data: list[Data] = [row[0] for row in datas]
    age_data: list[Data] = [row[1] for row in datas]

    # We are then going to save this data to a csv file
    write_data_to_csv(
        data,
        "test_outputs/python_model_output/template_simulation_output.csv",
    )
    write_data_to_csv(
        age_data,
        "test_outputs/python_model_output/template_simulation_output_age-grouped.csv",
    )
