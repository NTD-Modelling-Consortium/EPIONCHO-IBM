from typing import Callable

import numpy as np

from epioncho_ibm.types import Array

from .params import MicrofilParams, TreatmentParams


def _construct_derive_microfil(
    fertile_worms: Array.WormCat.Person.Int,
    microfil: Array.MFCat.Person.Float,
    fecundity_rates_worms: Array.WormCat.Float,
    mortality: Array.MFCat.Person.Float,
    microfil_move_rate: float,
    person_has_worms: Array.Person.Bool,
) -> Callable[[None | Array.MFCat.Person.Float], Array.MFCat.Person.Float]:
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

    # * lagged by one compartment
    movement: Array.MFCat.Person.Float = (
        np.roll(microfil, 1, axis=0) * microfil_move_rate
    )
    # TODO reconcile compartment size
    movement[0, :] = (
        np.einsum("ij, i -> j", fertile_worms, fecundity_rates_worms) * person_has_worms
    )

    def derive_microfil_fn(
        k: None | Array.MFCat.Person.Float,
    ) -> Array.MFCat.Person.Float:
        if k is None:
            microfil_adjusted = microfil
        else:
            microfil_adjusted = microfil + k
        assert np.all(microfil_adjusted >= 0)

        mortality_temp = mortality * microfil_adjusted
        move_rate_temp = microfil_move_rate * microfil_adjusted

        return movement - mortality_temp - move_rate_temp

    return derive_microfil_fn


def calculate_microfil_delta(
    current_microfil: Array.MFCat.Person.Float,
    delta_time: float,
    microfil_params: MicrofilParams,
    treatment_params: TreatmentParams | None,
    microfillarie_mortality_rate: Array.MFCat.Float,
    fecundity_rates_worms: Array.WormCat.Float,
    time_of_last_treatment: Array.Person.Float | None,
    current_time: float,
    current_fertile_female_worms: Array.WormCat.Person.Int,
    current_male_worms: Array.WormCat.Person.Int,
) -> Array.MFCat.Person.Float:
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
    mortality: Array.MFCat.Person.Float = np.repeat(
        microfillarie_mortality_rate, current_microfil.shape[1]
    ).reshape(current_microfil.shape)
    # increases microfilarial mortality if treatment has started
    if treatment_params is not None and current_time >= treatment_params.start_time:
        assert time_of_last_treatment is not None
        # additional mortality due to ivermectin treatment
        mortality_prime: Array.Person.Float = (
            time_of_last_treatment + microfil_params.u_ivermectin
        ) ** -microfil_params.shape_parameter_ivermectin

        mortality += np.nan_to_num(mortality_prime)

    derive_microfil = _construct_derive_microfil(
        fertile_worms=current_fertile_female_worms,
        microfil=current_microfil,
        fecundity_rates_worms=fecundity_rates_worms,
        mortality=mortality,
        microfil_move_rate=microfil_params.microfil_move_rate,
        person_has_worms=np.any(current_male_worms, axis=0),
    )

    k1 = derive_microfil(None)
    k2 = derive_microfil(k1 * (delta_time / 2))
    k3 = derive_microfil(k2 * (delta_time / 2))
    k4 = derive_microfil(k3 * delta_time)

    return (delta_time / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
