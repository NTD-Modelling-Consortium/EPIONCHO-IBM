from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .params import MicrofilParams, TreatmentParams


def construct_derive_microfil(
    fertile_worms: NDArray[np.int_],
    microfil: NDArray[np.float_],
    fecundity_rates_worms: NDArray[np.float_],
    mortality: NDArray[np.float_],
    microfil_move_rate: float,
    person_has_worms: NDArray[np.bool_],
) -> Callable[[float | NDArray[np.float_]], NDArray[np.float_]]:
    """
    fertile_worms # fert.worms
    microfil #mf.in
    fecundity_rates_worms # ep.in
    mortality # mf.mort
    microfil_move_rate #mf.move
    microfil_compartment_minus_one # mf.comp.minus.one
    person_has_worms # mp (once turned to 0 or 1)
    """

    assert np.all(mortality >= 0), "Mortality can't be negative"
    assert microfil_move_rate >= 0, "Mortality move rate can't be negative"

    lagged_microfil = np.roll(microfil, 1, axis=0)
    movement = lagged_microfil * microfil_move_rate

    movement[0, :] = (
        np.einsum("ij, i -> j", fertile_worms, fecundity_rates_worms) * person_has_worms
    )

    def derive_microfil_fn(k: float | NDArray[np.float_]) -> NDArray[np.float_]:
        microfil_adjusted = microfil + k
        assert np.all(microfil_adjusted >= 0)

        mortality_temp = mortality * microfil_adjusted
        move_rate_temp = microfil_move_rate * microfil_adjusted

        return movement - mortality_temp - move_rate_temp

    return derive_microfil_fn


def calculate_microfil_delta(
    stages: int,
    exiting_microfil: NDArray[np.float_],
    n_people: int,
    delta_time: float,
    microfil_params: MicrofilParams,
    treatment_params: TreatmentParams | None,
    microfillarie_mortality_rate: NDArray[np.float_],
    fecundity_rates_worms: NDArray[np.float_],
    time_of_last_treatment: NDArray[np.float_] | None,
    current_time: float,
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
    # mf.mu
    assert microfillarie_mortality_rate.shape == (stages,)
    mortality = np.repeat(microfillarie_mortality_rate, n_people).reshape(
        stages, n_people
    )

    # increases microfilarial mortality if treatment has started
    if treatment_params is not None and current_time >= treatment_params.start_time:
        assert time_of_last_treatment is not None
        # additional mortality due to ivermectin treatment
        mortality_prime = (
            time_of_last_treatment + microfil_params.u_ivermectin
        ) ** -microfil_params.shape_parameter_ivermectin
        mortality += np.nan_to_num(mortality_prime)

    derive_microfil = construct_derive_microfil(
        fertile_worms=current_fertile_female_worms,
        microfil=exiting_microfil,
        fecundity_rates_worms=fecundity_rates_worms,
        mortality=mortality,
        microfil_move_rate=microfil_params.microfil_move_rate,
        person_has_worms=np.sum(current_male_worms, axis=0) > 0,
    )

    k1 = derive_microfil(0.0)
    k2 = derive_microfil(delta_time * k1 / 2)
    k3 = derive_microfil(delta_time * k2 / 2)
    k4 = derive_microfil(delta_time * k3)
    return (delta_time / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
