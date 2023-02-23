from typing import Callable

import numpy as np

from epioncho_ibm.state import Array, MicrofilParams, TreatmentParams
from epioncho_ibm.state.people import LastTreatment


def _construct_derive_microfil(
    fertile_worms: Array.WormCat.Person.Int,
    microfil: Array.MFCat.Person.Float,
    fecundity_rates_worms: Array.WormCat.Float,
    mortality: Array.MFCat.Person.Float,
    microfil_move_rate: float,
    person_has_worms: Array.Person.Bool,
    debug: bool,
) -> Callable[[None | Array.MFCat.Person.Float], Array.MFCat.Person.Float]:
    """
    Produces a function that takes a k value and produces the next in the sequence.

    Args:
        fertile_worms (Array.WormCat.Person.Int): The current number of fertile worms
        microfil (Array.MFCat.Person.Float): The current number of microfilariae
        fecundity_rates_worms (Array.WormCat.Float): The rate at which worms reproduce
        mortality (Array.MFCat.Person.Float): The death rate for microfilariae
        microfil_move_rate (float): The rate which microfil move compartments
        person_has_worms (Array.Person.Bool): Whether or not each person has worms
        debug (bool): Runs in debug mode

    Returns:
        Callable[[None | Array.MFCat.Person.Float], Array.MFCat.Person.Float]: A function that takes
         a k value and produces the next in the sequence.
    """
    if debug:
        assert np.all(mortality >= 0), "Mortality can't be negative"
        assert microfil_move_rate >= 0, "Mortality move rate can't be negative"

    # * lagged by one compartment
    movement: Array.MFCat.Person.Float = (
        np.roll(microfil, 1, axis=0) * microfil_move_rate
    )
    # TODO reconcile compartment size

    movement[0, :] = np.dot(fertile_worms.T, fecundity_rates_worms) * person_has_worms

    def derive_microfil_fn(
        k: None | Array.MFCat.Person.Float,
    ) -> Array.MFCat.Person.Float:
        if k is None:
            microfil_adjusted = microfil
        else:
            microfil_adjusted = microfil + k
        if debug:
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
    last_treatment: LastTreatment | None,
    current_time: float,
    current_fertile_female_worms: Array.WormCat.Person.Int,
    current_male_worms: Array.WormCat.Person.Int,
    debug: bool,
) -> Array.MFCat.Person.Float:
    """
    Calculates the change of microfilariae.

    Args:
        current_microfil (Array.MFCat.Person.Float): The current number of microfilariae in each person
        delta_time (float): dt - The amount of time advance in one time step
        microfil_params (MicrofilParams): The fixed parameters relating to microfilariae
        treatment_params (TreatmentParams | None): The fixed parameters relating to treatment
        microfillarie_mortality_rate (Array.MFCat.Float): The death rate for microfilariae
        fecundity_rates_worms (Array.WormCat.Float): The rate at which worms reproduce
        time_of_last_treatment (Array.Person.Float | None): The last time each person was treated
        current_time (float): The current time t in the model
        current_fertile_female_worms (Array.WormCat.Person.Int): The current number of fertile worms
        current_male_worms (Array.WormCat.Person.Int): The current number of male worms
        debug (bool): Runs in debug mode

    Returns:
        Array.MFCat.Person.Float: The change of microfilariae in each person and age category
    """
    mortality: Array.MFCat.Person.Float = np.tile(
        microfillarie_mortality_rate, (current_microfil.shape[1], 1)
    ).T
    # increases microfilarial mortality if treatment has started
    if treatment_params is not None and current_time >= treatment_params.start_time:
        assert last_treatment is not None
        # additional mortality due to ivermectin treatment
        time_since_last_treatment = current_time - last_treatment.time
        mortality_prime: Array.Person.Float = (
            time_since_last_treatment + last_treatment.u_ivermectin
        ) ** -last_treatment.shape_parameter_ivermectin

        mortality += np.nan_to_num(mortality_prime)

    derive_microfil = _construct_derive_microfil(
        fertile_worms=current_fertile_female_worms,
        microfil=current_microfil,
        fecundity_rates_worms=fecundity_rates_worms,
        mortality=mortality,
        microfil_move_rate=microfil_params.microfil_move_rate,
        person_has_worms=np.any(current_male_worms, axis=0),
        debug=debug,
    )

    k1 = derive_microfil(None)
    k2 = derive_microfil(k1 * (delta_time / 2))
    k3 = derive_microfil(k2 * (delta_time / 2))
    k4 = derive_microfil(k3 * delta_time)

    return (delta_time / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
