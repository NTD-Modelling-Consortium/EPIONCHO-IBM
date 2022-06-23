from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from epioncho_ibm.blackfly import calc_l1, calc_l2, calc_l3
from epioncho_ibm.microfil import change_in_microfil
from epioncho_ibm.state import People, State
from epioncho_ibm.worms import (
    WormGroup,
    calc_new_worms,
    change_in_worm_per_index,
    get_delayed_males_and_females,
)

from .params import Params


def _calc_coverage(
    people: People,
    # percent_non_compliant: float,
    measured_coverage: float,
    age_compliance: float,
) -> NDArray[np.bool_]:

    non_compliant_people = np.logical_or(
        people.ages < age_compliance, np.logical_not(people.compliance)
    )
    non_compliant_percentage = np.sum(non_compliant_people) / len(non_compliant_people)
    compliant_percentage = 1 - non_compliant_percentage
    coverage = measured_coverage / compliant_percentage  # TODO: Is this correct?
    out_coverage = np.repeat(coverage, len(people))
    out_coverage[non_compliant_people] = 0
    rand_nums = np.random.uniform(low=0, high=1, size=len(people))
    return rand_nums < out_coverage


def _calculate_total_exposure(
    params: Params, people: People, individual_exposure: NDArray[np.float_]
) -> NDArray[np.float_]:
    male_exposure_assumed = params.male_exposure * np.exp(
        -params.male_exposure_exponent * people.ages
    )
    male_exposure_assumed_of_males = male_exposure_assumed[people.sex_is_male]
    if len(male_exposure_assumed_of_males) == 0:
        # TODO: Is this correct?
        mean_male_exposure = 0
    else:
        mean_male_exposure: float = np.mean(male_exposure_assumed_of_males)
    female_exposure_assumed = params.female_exposure * np.exp(
        -params.female_exposure_exponent * people.ages
    )
    female_exposure_assumed_of_females = female_exposure_assumed[
        np.logical_not(people.sex_is_male)
    ]
    if len(female_exposure_assumed_of_females) == 0:
        # TODO: Is this correct?
        mean_female_exposure = 0
    else:
        mean_female_exposure: float = np.mean(female_exposure_assumed_of_females)

    sex_age_exposure = np.where(
        people.sex_is_male,
        male_exposure_assumed / mean_male_exposure,
        female_exposure_assumed / mean_female_exposure,
    )

    total_exposure = sex_age_exposure * individual_exposure
    total_exposure = total_exposure / np.mean(total_exposure)
    return total_exposure


def run_simulation(
    state: State, start_time: float = 0, end_time: float = 0, verbose: bool = False
) -> State:
    if end_time < start_time:
        raise ValueError("End time after start")

    current_time = start_time
    while current_time < end_time:
        if state.params.delta_time > current_time % 0.2 and verbose:
            print(current_time)

        if current_time >= state.params.treatment_start_time:
            coverage_in = _calc_coverage(
                state.people,
                state.params.total_population_coverage,
                state.params.min_skinsnip_age,
            )
        else:
            coverage_in = None

        total_exposure = _calculate_total_exposure(
            state.params, state.people, state.derived_params.individual_exposure
        )
        old_state = deepcopy(state)  # all.mats.cur
        # increase ages
        state.people.ages += state.params.delta_time

        people_to_die: NDArray[np.bool_] = np.logical_or(
            np.random.binomial(
                n=1,
                p=(1 / state.params.mean_human_age) * state.params.delta_time,
                size=state.params.human_population,
            )
            == 1,
            state.people.ages >= state.params.max_human_age,
        )

        # there is a delay in new parasites entering humans (from fly bites) and entering the first adult worm age class
        new_worms = calc_new_worms(state, total_exposure)
        # Take males and females from final column of worm_delay
        delayed_males, delayed_females = get_delayed_males_and_females(
            state.delay_arrays.worm_delay, state.params
        )
        # Move all columns in worm_delay along one
        state.delay_arrays.worm_delay = np.vstack(
            (new_worms, state.delay_arrays.worm_delay[:-1])
        )

        last_aging_worms = WormGroup.from_population(state.params.human_population)
        last_time_of_last_treatment = None
        for compartment in range(state.params.worm_age_stages):
            (
                last_total_worms,
                last_aging_worms,
                last_time_of_last_treatment,
            ) = change_in_worm_per_index(  # res
                params=state.params,
                state=state,
                delayed_females=delayed_females,
                delayed_males=delayed_males,
                worm_mortality_rate=state.derived_params.worm_mortality_rate,
                coverage_in=coverage_in,
                last_aging_worms=last_aging_worms,
                initial_treatment_times=state.derived_params.initial_treatment_times,
                current_time=current_time,
                compartment=compartment,
                time_of_last_treatment=state.people.time_of_last_treatment,
            )
            if np.any(
                np.logical_or(
                    np.logical_or(
                        last_total_worms.male < 0, last_total_worms.fertile < 0
                    ),
                    last_total_worms.infertile < 0,
                )
            ):
                candidate_people_male_worms = last_total_worms.male[
                    last_total_worms.male < 0
                ]
                candidate_people_fertile_worms = last_total_worms.fertile[
                    last_total_worms.fertile < 0
                ]
                candidate_people_infertile_worms = last_total_worms.infertile[
                    last_total_worms.infertile < 0
                ]

                raise RuntimeError(
                    f"Worms became negative: \nMales: {candidate_people_male_worms} \nFertile Females: {candidate_people_fertile_worms} \nInfertile Females: {candidate_people_infertile_worms}"
                )

            state.people.male_worms[compartment] = last_total_worms.male
            state.people.infertile_female_worms[
                compartment
            ] = last_total_worms.infertile
            state.people.fertile_female_worms[compartment] = last_total_worms.fertile

        assert last_time_of_last_treatment is not None
        if (
            state.derived_params.initial_treatment_times is not None
            and current_time >= state.params.treatment_start_time
        ):
            state.people.time_of_last_treatment = last_time_of_last_treatment

        for compartment in range(state.params.microfil_age_stages):
            state.people.mf[compartment] = change_in_microfil(
                state=old_state,
                params=state.params,
                microfillarie_mortality_rate=state.derived_params.microfillarie_mortality_rate,
                fecundity_rates_worms=state.derived_params.fecundity_rates_worms,
                time_of_last_treatment=state.people.time_of_last_treatment,
                compartment=compartment,
                current_time=current_time,
            )

        # inputs for delay in L1
        new_mf = np.sum(
            old_state.people.mf, axis=0
        )  # TODO: Should this be old state? mf.temp

        state.people.blackfly.L1 = calc_l1(
            state.params,
            new_mf,
            state.delay_arrays.mf_delay[-1],
            total_exposure,
            state.delay_arrays.exposure_delay[-1],
        )
        state.people.blackfly.L2 = calc_l2(
            state.params,
            state.delay_arrays.l1_delay,
            state.delay_arrays.mf_delay[-1],
            state.delay_arrays.exposure_delay[-1],
        )
        state.people.blackfly.L3 = calc_l3(state.params, old_state.people.blackfly.L2)

        state.delay_arrays.exposure_delay = np.vstack(
            (total_exposure, state.delay_arrays.exposure_delay[:-1])
        )
        state.delay_arrays.mf_delay = np.vstack(
            (new_mf, state.delay_arrays.mf_delay[:-1])
        )
        state.delay_arrays.l1_delay = state.people.blackfly.L1

        total_people_to_die: int = np.sum(people_to_die)
        if total_people_to_die > 0:
            state.delay_arrays.worm_delay[:, people_to_die] = 0
            state.delay_arrays.mf_delay[0, people_to_die] = 0
            state.delay_arrays.l1_delay[people_to_die] = 0
            state.people.time_of_last_treatment[people_to_die] = np.nan

            state.people.sex_is_male[people_to_die] = (
                np.random.uniform(low=0, high=1, size=total_people_to_die) < 0.5
            )  # TODO: Make adjustable
            state.people.ages[people_to_die] = 0
            state.people.blackfly.L1[people_to_die] = 0
            state.people.mf[:, people_to_die] = 0
            state.people.male_worms[:, people_to_die] = 0
            state.people.fertile_female_worms[:, people_to_die] = 0
            state.people.infertile_female_worms[:, people_to_die] = 0
        current_time += state.params.delta_time
    return state
