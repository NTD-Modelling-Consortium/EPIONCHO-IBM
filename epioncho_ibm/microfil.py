from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from epioncho_ibm.state import People

from .params import MicrofilParams, Params, TreatmentParams


def construct_derive_microfil_one(
    fertile_worms: NDArray[np.int_],
    microfil: NDArray[np.int_],
    fecundity_rates_worms: NDArray[np.float_],
    mortality: NDArray[np.float_],
    microfil_move_rate: float,
    person_has_worms: NDArray[np.bool_],
) -> Callable[[Union[float, NDArray[np.float_]]], NDArray[np.float_]]:
    """
    #function called during RK4 for first age class of microfilariae

    fertile_worms # fert.worms
    microfil #mf.in
    fecundity_rates_worms # ep.in
    mortality # mf.mort
    microfil_move_rate #mf.move
    person_has_worms # mp (once turned to 0 or 1)
    """
    new_microfil = np.einsum(
        "ij, i -> j", fertile_worms, fecundity_rates_worms
    )  # TODO: Check?

    def derive_microfil_one(k: Union[float, NDArray[np.float_]]) -> NDArray[np.float_]:
        mortality_temp = mortality * (microfil + k)
        assert np.sum(mortality_temp < 0) == 0
        move_rate_temp = microfil_move_rate * (microfil + k)
        assert np.sum(move_rate_temp < 0) == 0
        mortality_temp[mortality_temp < 0] = 0
        move_rate_temp[move_rate_temp < 0] = 0
        return person_has_worms * new_microfil - mortality_temp - move_rate_temp

    return derive_microfil_one


def construct_derive_microfil_rest(
    microfil: NDArray[np.int_],
    mortality: NDArray[np.float_],
    microfil_move_rate: float,
    microfil_compartment_minus_one: NDArray[np.int_],
) -> Callable[[Union[float, NDArray[np.float_]]], NDArray[np.float_]]:
    """
    #function called during RK4 for age classes of microfilariae > 1

    microfil #mf.in
    mortality # mf.mort
    microfil_move_rate #mf.move
    microfil_compartment_minus_one # mf.comp.minus.one
    """
    movement_last = microfil_compartment_minus_one * microfil_move_rate

    def derive_microfil_rest(k: Union[float, NDArray[np.float_]]) -> NDArray[np.float_]:
        mortality_temp = mortality * (microfil + k)
        assert np.sum(mortality_temp < 0) == 0
        move_rate_temp = microfil_move_rate * (microfil + k)
        assert np.sum(move_rate_temp < 0) == 0
        mortality_temp[mortality_temp < 0] = 0
        move_rate_temp[move_rate_temp < 0] = 0
        return movement_last - mortality_temp - move_rate_temp

    return derive_microfil_rest


def change_in_microfil(
    n_people: int,
    delta_time: float,
    microfil_params: MicrofilParams,
    treatment_params: Optional[TreatmentParams],
    microfillarie_mortality_rate: float,
    fecundity_rates_worms: NDArray[np.float_],
    time_of_last_treatment: Optional[NDArray[np.float_]],
    current_time: float,
    current_microfil: NDArray[np.int_],
    previous_microfil: Optional[NDArray[np.int_]],
    current_fertile_female_worms: NDArray[np.int_],
    current_male_worms: NDArray[np.int_],
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
    compartment_mortality = np.repeat(microfillarie_mortality_rate, n_people)  # mf.mu
    microfil: NDArray[np.int_] = current_microfil

    # increases microfilarial mortality if treatment has started
    if treatment_params is not None and current_time >= treatment_params.start_time:
        assert time_of_last_treatment is not None
        compartment_mortality_prime = (
            time_of_last_treatment + microfil_params.u_ivermectin
        ) ** (
            -microfil_params.shape_parameter_ivermectin
        )  # additional mortality due to ivermectin treatment
        compartment_mortality += np.nan_to_num(compartment_mortality_prime)

    if previous_microfil is None:
        person_has_worms = np.sum(current_male_worms, axis=0) > 0
        derive_microfil = construct_derive_microfil_one(
            current_fertile_female_worms,
            microfil,
            fecundity_rates_worms,
            compartment_mortality,
            microfil_params.microfil_move_rate,
            person_has_worms,
        )
    else:
        derive_microfil = construct_derive_microfil_rest(
            microfil,
            compartment_mortality,
            microfil_params.microfil_move_rate,
            previous_microfil,
        )
    k1 = derive_microfil(0.0)
    k2 = derive_microfil(delta_time * k1 / 2)
    k3 = derive_microfil(delta_time * k2 / 2)
    k4 = derive_microfil(delta_time * k3)
    return microfil + (delta_time / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
