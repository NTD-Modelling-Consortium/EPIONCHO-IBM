import os

import numpy as np
import pandas as pd

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from functools import partial
from multiprocessing import cpu_count

from tqdm.contrib.concurrent import process_map

from epioncho_ibm import Params, Simulation
from epioncho_ibm.state.params import BlackflyParams, TreatmentParams
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


def run_sim(
    i,
    start_time=1950,
    mda_start=2000,
    mda_stop=2020,
    simulation_stop=2050,
    abr=2297,
    verbose=True,
) -> Data:
    params = Params(
        delta_time_days=1,
        year_length_days=366,
        n_people=440,
        blackfly=BlackflyParams(
            delta_h_zero=0.186,
            delta_h_inf=0.003,
            c_h=0.005,
            bite_rate_per_person_per_year=abr,
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
        treatment=TreatmentParams(
            interval_years=1,
            start_time=mda_start,
            stop_time=mda_stop,
        ),
    )

    simulation = Simulation(start_time=start_time, params=params, verbose=verbose)
    run_data: Data = {}
    for state in simulation.iter_run(
        end_time=simulation_stop,
        sampling_years=[i for i in range(mda_stop, simulation_stop)],
    ):
        add_state_to_run_data(
            state,
            run_data=run_data,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=True,
            prevalence=True,
            mean_worm_burden=False,
            intensity=True,
            output_age_range_map={"prevalence": {"age_min": 5, "age_max": 80}},
        )
    return run_data


# context object should look something like:
# {
#  "fileDescriptors": {"iu":"test", "scenario":"scenario_1", "mda_stop_year":2026}, # used for file name/output data]
#  "measures":["prevalence", "Blindness"], # measures to calculate all-ages prevalence
#  "calcProbElim":True # calculate probability of elimination across all runs
# }
def calcProbEliminationIUScenario(data, measureYears=[2040, 2050], context={}):
    if (
        ("calcProbElim" in context)
        & (context["calcProbElim"])
        & ("prevalence" not in context["measures"])
    ):
        context["measures"].append("prevalence")

    input_columns = ["year", "measure", "measure_value", "number", "run_num"]
    inputData = pd.DataFrame(
        [
            [key[0], key[3], value, 0, i]
            for i, run in enumerate(data)
            for key, value in run.items()
            if key[0] in measureYears and key[3] in context["measures"]
        ],
        columns=input_columns,
    )
    inputData["number"] = [
        value
        for run in data
        for key, value in run.items()
        if ((key[0] in measureYears) & (key[3] == "number"))
        for _ in range(len(context["measures"]))
    ]

    inputData["measure_value"] = np.array(inputData["measure_value"].values) * np.array(
        inputData["number"].values
    )
    tmp = (
        inputData.groupby(["year", "measure", "run_num"])
        .agg({"measure_value": "sum", "number": "sum"})
        .reset_index()
    )
    tmp["prevs"] = np.array(tmp["measure_value"].values) / np.array(
        tmp["number"].values
    )
    if context["calcProbElim"]:
        tmp["eliminated"] = np.where(
            tmp["measure"] == "prevalence", tmp["prevs"].values == 0, False
        )
        result = (
            tmp.groupby(["year", "measure"])
            .agg({"prevs": "mean", "eliminated": "mean"})
            .reset_index()
            .to_numpy()
        )
        prev_index = result[:, 1] == "prevalence"
        tmp2 = result[prev_index]
        prob_elims = tmp2[:, [0, 1, 3]]
        prob_elims[:, 1] = np.full(prob_elims.shape[0], "prob_elim")
        result = result[:, [0, 1, 2]]
        result = np.vstack((result, prob_elims))
    else:
        result = (
            tmp.groupby(["year", "measure"])
            .agg({"prevs": "mean"})
            .reset_index()
            .to_numpy()
        )

    numRows = result.shape[0]

    output_data = {}
    fileVals = []
    for label, value in context["fileDescriptors"].items():
        output_data[str(label)] = np.full(numRows, value)
        fileVals.append(str(value))
    output_data = {
        **output_data,
        **{
            "measure_year": result[:, 0],
            "measure": result[:, 1],
            "summary_value": result[:, 2],
        },
    }

    pd.DataFrame(output_data).to_csv("-".join(["prob_elim", *fileVals]) + ".csv")


run_iters = 1

if __name__ == "__main__":
    cpus_to_use = cpu_count() - 4

    # ~ 70% MFP
    rumSim = partial(
        run_sim,
        start_time=1950,
        mda_start=2000,
        mda_stop=2028,
        simulation_stop=2051,
        abr=2297,
    )
    data: list[Data] = process_map(rumSim, range(run_iters), max_workers=cpus_to_use)
    for i, row in enumerate(data):
        print(i, row)
        print("LINE SPACE")
    # write_data_to_csv(data, "testingChanges.csv")
    # calcProbEliminationIUScenario(data, context={
    #     "fileDescriptors": {"iu":"test", "scenario":"scenario_1", "mda_stop_year":2028},
    #     "measures":["prevalence", "Blindness"],
    #     "calcProbElim":True
    #     })
