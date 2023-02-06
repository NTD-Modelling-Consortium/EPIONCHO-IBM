import numpy as np

from epioncho_ibm.state import Array, State

from .blackfly import calc_l1, calc_l2, calc_l3, calc_new_worms_from_blackfly
from .exposure import calculate_total_exposure
from .microfil import calculate_microfil_delta
from .treatment import get_treatment
from .worms import calculate_new_worms
from .old_microfil import change_in_microfil

def advance_state(state: State, debug: bool = False) -> None:
    """Advance the state forward one time step from t to t + dt"""

    treatment = get_treatment(
        state._params.treatment,
        state._params.humans,
        state._params.delta_time,
        state.current_time,
        state.derived_params.treatment_times,
        state.people.ages,
        state.people.compliance,
        state.numpy_bit_generator,
    )
    if treatment is not None and treatment.treatment_occurred:
        assert state.n_treatments is not None
        n_treatments_by_age, _ = np.histogram(
            state.people.ages[treatment.coverage_in],
            bins=len(state.n_treatments),
        )
        state.n_treatments += n_treatments_by_age

    total_exposure = calculate_total_exposure(
        state._params.exposure,
        state.people.ages,
        state.people.sex_is_male,
        state.people.individual_exposure,
    )
    state.people.ages += state._params.delta_time

    old_worms = state.people.worms.copy()

    # there is a delay in new parasites entering humans (from fly bites) and
    # entering the first adult worm age class
    new_worms = calc_new_worms_from_blackfly(
        state.people.blackfly.L3,
        state._params.blackfly,
        state._params.delta_time,
        total_exposure,
        state.n_people,
        old_worms,
        debug,
        state.numpy_bit_generator,
    )

    if state.people.delay_arrays.worm_delay is None:
        worm_delay: Array.Person.Int = new_worms
    else:
        worm_delay: Array.Person.Int = state.people.delay_arrays.worm_delay

    state.people.worms, last_time_of_last_treatment = calculate_new_worms(
        current_worms=state.people.worms,
        worm_params=state._params.worms,
        treatment=treatment,
        time_of_last_treatment=state.people.time_of_last_treatment,
        delta_time=state._params.delta_time,
        worm_delay_array=worm_delay,
        mortalities=state.derived_params.worm_mortality_rate,
        mortalities_generator=state.derived_params.worm_mortality_generator,
        current_time=state.current_time,
        debug=debug,
        worm_age_rate_generator=state.derived_params.worm_age_rate_generator,
        worm_sex_ratio_generator=state.derived_params.worm_sex_ratio_generator,
        worm_lambda_zero_generator=state.derived_params.worm_lambda_zero_generator,
        worm_omega_generator=state.derived_params.worm_omega_generator,
        numpy_bit_gen=state.numpy_bit_generator,
    )

    if (
        state._params.treatment is not None
        and state.current_time >= state._params.treatment.start_time
    ):
        state.people.time_of_last_treatment = last_time_of_last_treatment

    # inputs for delay in L1


    # state.people.mf += calculate_microfil_delta(
    #     current_microfil=state.people.mf,
    #     delta_time=state._params.delta_time,
    #     microfil_params=state._params.microfil,
    #     treatment_params=state._params.treatment,
    #     microfillarie_mortality_rate=state.derived_params.microfillarie_mortality_rate,
    #     fecundity_rates_worms=state.derived_params.fecundity_rates_worms,
    #     time_of_last_treatment=state.people.time_of_last_treatment,
    #     current_time=state.current_time,
    #     current_fertile_female_worms=old_worms.fertile,
    #     current_male_worms=old_worms.male,
    #     debug=debug,
    # )
    old_mf_mat = state.people.mf.copy()
    for compartment in range(state._params.microfil.microfil_age_stages):
        if compartment == 0:
            state.people.mf[compartment] = change_in_microfil(
                n_people=state.n_people,
                delta_time=state._params.delta_time,
                microfil_params=state._params.microfil,
                treatment_params=state._params.treatment,
                microfillarie_mortality_rate=state.derived_params.microfillarie_mortality_rate[
                    compartment
                ],
                fecundity_rates_worms=state.derived_params.fecundity_rates_worms,
                time_of_last_treatment=state.people.time_of_last_treatment,
                current_time=state.current_time,
                current_microfil=old_mf_mat[compartment],
                previous_microfil=None,
                current_fertile_female_worms=state.people.worms.fertile,
                current_male_worms=state.people.worms.male,
            )
        else:
            state.people.mf[compartment] = change_in_microfil(
                n_people=state.n_people,
                delta_time=state._params.delta_time,
                microfil_params=state._params.microfil,
                treatment_params=state._params.treatment,
                microfillarie_mortality_rate=state.derived_params.microfillarie_mortality_rate[
                    compartment
                ],
                fecundity_rates_worms=state.derived_params.fecundity_rates_worms,
                time_of_last_treatment=state.people.time_of_last_treatment,
                current_time=state.current_time,
                current_microfil=old_mf_mat[compartment],
                previous_microfil=old_mf_mat[compartment - 1],
                current_fertile_female_worms=state.people.worms.fertile,
                current_male_worms=state.people.worms.male,
            )
    old_mf: Array.Person.Float = np.sum(state.people.mf, axis=0)
    old_blackfly_L1 = state.people.blackfly.L1

    if state.people.delay_arrays.exposure_delay is None:
        exposure_delay: Array.Person.Float = total_exposure
    else:
        exposure_delay: Array.Person.Float = state.people.delay_arrays.exposure_delay

    if state.people.delay_arrays.mf_delay is None:
        mf_delay: Array.Person.Float = old_mf.copy()
    else:
        mf_delay: Array.Person.Float = state.people.delay_arrays.mf_delay

    state.people.blackfly.L1 = calc_l1(
        state._params.blackfly,
        old_mf,
        mf_delay,
        total_exposure,
        exposure_delay,
        state._params.year_length_days,
    )

    old_blackfly_L2 = state.people.blackfly.L2
    state.people.blackfly.L2 = calc_l2(
        state._params.blackfly,
        old_blackfly_L1,
        mf_delay,
        exposure_delay,
        state._params.year_length_days,
    )
    state.people.blackfly.L3 = calc_l3(state._params.blackfly, old_blackfly_L2)
    # TODO: Resolve new_mf=old_mf
    state.people.delay_arrays.lag_all_arrays(
        new_worms=new_worms, total_exposure=total_exposure, new_mf=old_mf
    )
    people_to_die: Array.Person.Bool = np.logical_or(
        state.derived_params.people_to_die_generator.binomial(
            np.repeat(1, state.n_people)
        )
        == 1,
        state.people.ages >= state._params.humans.max_human_age,
    )
    state.people.process_deaths(
        people_to_die, state._params.humans.gender_ratio, state.numpy_bit_generator
    )
