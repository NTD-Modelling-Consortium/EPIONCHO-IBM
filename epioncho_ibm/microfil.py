from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray

from epioncho_ibm.state import People

from .params import Params


def construct_derive_microfil_one(
    fertile_worms: NDArray[np.int_],
    microfil: NDArray[np.int_],
    fecundity_rates_worms: NDArray[np.float_],
    mortality: NDArray[np.float_],
    params: Params,
    person_has_worms: NDArray[np.bool_],
) -> Callable[[Union[float, NDArray[np.float_]]], NDArray[np.float_]]:
    """
    #function called during RK4 for first age class of microfilariae

    fertile_worms # fert.worms
    microfil #mf.in
    fecundity_rates_worms # ep.in
    mortality # mf.mort
    params.microfil_move_rate #mf.move
    person_has_worms # mp (once turned to 0 or 1)
    """
    new_in = np.einsum(
        "ij, i -> j", fertile_worms, fecundity_rates_worms
    )  # TODO: Check?

    def derive_microfil_one(k: Union[float, NDArray[np.float_]]) -> NDArray[np.float_]:
        mortality_temp = mortality * (microfil + k)
        assert np.sum(mortality_temp < 0) == 0
        move_rate_temp = params.microfil_move_rate * (microfil + k)
        assert np.sum(move_rate_temp < 0) == 0
        mortality_temp[mortality_temp < 0] = 0
        move_rate_temp[move_rate_temp < 0] = 0
        return person_has_worms * new_in - mortality_temp - move_rate_temp

    return derive_microfil_one


def construct_derive_microfil_rest(
    microfil: NDArray[np.int_],
    mortality: NDArray[np.float_],
    params: Params,
    microfil_compartment_minus_one: NDArray[np.int_],
) -> Callable[[Union[float, NDArray[np.float_]]], NDArray[np.float_]]:
    """
    #function called during RK4 for age classes of microfilariae > 1

    microfil #mf.in
    mortality # mf.mort
    params.microfil_move_rate #mf.move
    microfil_compartment_minus_one # mf.comp.minus.one
    """
    movement_last = microfil_compartment_minus_one * params.microfil_move_rate

    def derive_microfil_rest(k: Union[float, NDArray[np.float_]]) -> NDArray[np.float_]:
        mortality_temp = mortality * (microfil + k)
        assert np.sum(mortality_temp < 0) == 0
        move_rate_temp = params.microfil_move_rate * (microfil + k)
        assert np.sum(move_rate_temp < 0) == 0
        mortality_temp[mortality_temp < 0] = 0
        move_rate_temp[move_rate_temp < 0] = 0
        return movement_last - mortality_temp - move_rate_temp

    return derive_microfil_rest


def change_in_microfil(
    people: People,
    params: Params,
    microfillarie_mortality_rate: NDArray[np.float_],
    fecundity_rates_worms: NDArray[np.float_],
    time_of_last_treatment: Optional[NDArray[np.float_]],
    compartment: int,
    current_time: float,
) -> NDArray[np.float_]:
    """
    microfillarie_mortality_rate # mu.rates.mf
    fecundity_rates_worms # fec.rates
    params.delta_time "DT"
    worms.start/ws used to refer to start point in giant array for worms
    if initial_treatment_times is None give.treat is false etc
    params.treatment_start_time "treat.start"
    time_of_last_treatment # "treat.vec"
    "compartment" Corresponds to mf column mf.cpt
    "current_time" corresponds to iteration
    params.up up
    params.kap kap
    params.microfil_move_rate # mf.move.rate
    params.worm_age_stages "num.comps"
    params.microfil_age_stages "num.mf.comps"
    params.microfil_aging "time.each.comp"
    N is params.human_population
    people is dat
    """
    compartment_mortality = np.repeat(  # mf.mu
        microfillarie_mortality_rate[compartment], params.human_population
    )
    microfil: NDArray[np.int_] = people.mf[compartment]

    # increases microfilarial mortality if treatment has started
    if (
        time_of_last_treatment is not None
        and current_time >= params.treatment_start_time
    ):
        compartment_mortality_prime = (
            time_of_last_treatment + params.u_ivermectin
        ) ** (
            -params.shape_parameter_ivermectin
        )  # additional mortality due to ivermectin treatment
        compartment_mortality_prime = np.nan_to_num(compartment_mortality_prime)
        compartment_mortality += compartment_mortality_prime

    if compartment == 0:
        person_has_worms = np.sum(people.male_worms, axis=0) > 0
        derive_microfil = construct_derive_microfil_one(
            people.fertile_female_worms,
            microfil,
            fecundity_rates_worms,
            compartment_mortality,
            params,
            person_has_worms,
        )
    else:
        derive_microfil = construct_derive_microfil_rest(
            microfil, compartment_mortality, params, people.mf[compartment - 1]
        )
    k1 = derive_microfil(0.0)
    k2 = derive_microfil(params.delta_time * k1 / 2)
    k3 = derive_microfil(params.delta_time * k2 / 2)
    k4 = derive_microfil(params.delta_time * k3)
    return microfil + (params.delta_time / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
