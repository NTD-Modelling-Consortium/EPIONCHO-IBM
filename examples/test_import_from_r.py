import numpy as np
import pandas as pd

from epioncho_ibm import Simulation, State
from epioncho_ibm.advance.advance import advance_state
from epioncho_ibm.state.params import BlackflyParams, Params, mutable_to_immutable
from epioncho_ibm.state.people import BlackflyLarvae, DelayArrays, People, WormGroup
from epioncho_ibm.utils import array_fully_equal


def get_state_from_R(
    all_mats_temp_file: str,
    exvec_file: str,
    mfdelay_file: str,
    exposure_delay_file: str,
    lextras_file: str,
):
    n_people = 440

    allmatstemp = pd.read_csv(all_mats_temp_file).to_numpy()

    compliance = np.invert(allmatstemp[:, 1].astype(np.bool_))
    ages = allmatstemp[:, 2]
    sex_is_male = allmatstemp[:, 3].astype(np.bool_)

    L1 = allmatstemp[:, 4]
    L2 = allmatstemp[:, 5]
    L3 = allmatstemp[:, 6]
    blackfly_larvae = BlackflyLarvae(L1=L1, L2=L2, L3=L3)

    mf = allmatstemp[:, 7:28].T

    male_worms = allmatstemp[:, 28:49].T
    infertile_worms = allmatstemp[:, 49:70].T
    fertile_worms = allmatstemp[:, 70:91].T
    worms = WormGroup(male=male_worms, infertile=infertile_worms, fertile=fertile_worms)

    time_of_last_treatment = np.empty(n_people)
    time_of_last_treatment[:] = np.nan

    individual_exposure = pd.read_csv(exvec_file).to_numpy()[:, 1]

    mf_delay = pd.read_csv(mfdelay_file).to_numpy()[:, 1:].T
    exposure_delay = pd.read_csv(exposure_delay_file).to_numpy()[:, 1:].T
    worm_delay = pd.read_csv(lextras_file).to_numpy()[:, 1:].T
    delay_arrays = DelayArrays(
        _worm_delay=worm_delay,
        _exposure_delay=exposure_delay,
        _mf_delay=mf_delay,
        _worm_delay_current=worm_delay.shape[0] - 1,
        _exposure_delay_current=exposure_delay.shape[0] - 1,
        _mf_delay_current=mf_delay.shape[0] - 1,
    )
    # Note: Considered delay indexes

    people = People(
        compliance=compliance,
        sex_is_male=sex_is_male,
        blackfly=blackfly_larvae,
        ages=ages,
        mf=mf,
        worms=worms,
        time_of_last_treatment=time_of_last_treatment,
        delay_arrays=delay_arrays,
        individual_exposure=individual_exposure,
    )

    return State(
        people=people,
        _params=mutable_to_immutable(
            Params(
                n_people=n_people,
                seed=None,
                blackfly=BlackflyParams(
                    delta_h_zero=0.186,
                    delta_h_inf=0.003,
                    c_h=0.005,
                    bite_rate_per_person_per_year=294,
                ),
            )
        ),
        n_treatments=None,
        current_time=8,
    )


state = get_state_from_R(
    "allmatstemp.csv", "exvec.csv", "mfdelay.csv", "exposuredelay.csv", "lextras.csv"
)
state.to_hdf5("state.hdf5")
state2 = State.from_hdf5("state.hdf5")
print(f"state before: ", state == state2)
advance_state(state)
advance_state(state2)
print(f"state after: ", state == state2)
state.people.ages

print(
    f"compliance: ",
    array_fully_equal(state.people.compliance, state2.people.compliance),
)
print(
    f"sex: ",
    array_fully_equal(state.people.sex_is_male[12], state2.people.sex_is_male[12]),
)
print(f"blackfly: ", state.people.blackfly == state2.people.blackfly)
print(f"agesfull: ", array_fully_equal(state.people.ages, state2.people.ages))
print(f"ages1: ", state.people.ages[12] == state2.people.ages[12])
print(f"mf: ", array_fully_equal(state.people.mf, state2.people.mf))

print(
    f"time_of_last_treatment: ",
    array_fully_equal(
        state.people.time_of_last_treatment, state2.people.time_of_last_treatment
    ),
)

print(
    f"individual_ex: ",
    array_fully_equal(
        state.people.individual_exposure, state2.people.individual_exposure
    ),
)

sim = Simulation(input="state.hdf5", verbose=True, debug=True)
print(sim.state.mf_prevalence_in_population())
sim.run(end_time=40)
print(sim.state.mf_prevalence_in_population())
# print(state.people.worms == state2.people.worms)
# print(array_fully_equal(state.people.worms.male, state2.people.worms.male))
# print(array_fully_equal(state.people.worms.infertile, state2.people.worms.infertile))
# print(array_fully_equal(state.people.worms.fertile, state2.people.worms.fertile))
# print(state.people.delay_arrays == state2.people.delay_arrays)
# delta.hz <- 0.186
# delta.hinf <- 0.003
# c.h <-  0.005
