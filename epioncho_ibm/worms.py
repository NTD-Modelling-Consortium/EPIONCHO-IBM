from dataclasses import dataclass

import numpy as np
from fast_binomial import Generator

import epioncho_ibm.utils as utils
from epioncho_ibm.treatment import TreatmentGroup
from epioncho_ibm.types import Array

from .params import WormParams

__all__ = ["WormGroup", "calculate_new_worms"]


@dataclass
class WormGroup:
    """
    A group of worms, separated by sex and fertility
    """

    male: Array.WormCat.Person.Int
    infertile: Array.WormCat.Person.Int
    fertile: Array.WormCat.Person.Int

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WormGroup):
            return (
                np.array_equal(self.male, other.male)
                and np.array_equal(self.infertile, other.infertile)
                and np.array_equal(self.fertile, other.fertile)
            )
        else:
            return False

    @classmethod
    def from_population(cls, population: int):
        return cls(
            male=np.zeros(population, dtype=int),
            infertile=np.zeros(population, dtype=int),
            fertile=np.zeros(population, dtype=int),
        )


def _calc_dead_worms(
    current_worms: WormGroup,
    female_mortalities: Array.WormCat.Float | Array.WormCat.Person.Float,
    male_mortalities: Array.WormCat.Float,
    treatment_occurred: bool = False,
) -> WormGroup:
    """
    Calculates the number of worms dying in each compartment

    Args:
        current_worms (WormGroup): The current number of worms
        female_mortalities (Array.WormCat.Float | Array.WormCat.Person.Float): The rate of
            mortality for female worms, either assumed to be the same in each compartment,
            or by person in the case of treatment.
        male_mortalities (Array.WormCat.Float): The rate of mortality for male worms,
            assumed to be the same in each compartment
        treatment_occurred (bool): Whether or not treatment occurred in this time step

    Returns:
        WormGroup: The number of worms dying in each compartment
    """

    def _calc_dead_worms_single_group(
        current_worms: Array.WormCat.Person.Int,
        mortalities: Array.WormCat.Float | Array.WormCat.Person.Float,
        treatment_occurred_and_female: bool = False,
    ) -> Array.WormCat.Person.Int:
        assert current_worms.ndim == 2
        if treatment_occurred_and_female:
            return utils.fast_binomial(n=current_worms, p=mortalities)
        else:
            mortalities_by_person: Array.WormCat.Person.Float = np.tile(
                mortalities, (current_worms.shape[1], 1)
            ).T
            return utils.fast_binomial(
                n=current_worms,
                p=mortalities_by_person,
            )

    return WormGroup(
        male=_calc_dead_worms_single_group(
            current_worms=current_worms.male, mortalities=male_mortalities
        ),
        infertile=_calc_dead_worms_single_group(
            current_worms.infertile,
            female_mortalities,
            treatment_occurred_and_female=treatment_occurred,
        ),
        fertile=_calc_dead_worms_single_group(
            current_worms.fertile,
            female_mortalities,
            treatment_occurred_and_female=treatment_occurred,
        ),
    )


def _calc_outbound_worms(
    current_worms: WormGroup,
    worm_age_rate_generator: Generator,
    dead_worms: WormGroup,
) -> WormGroup:
    """
    Calculates the number of worms leaving each compartment due to aging.

    Args:
        current_worms (WormGroup): The current number of worms
        worm_age_rate_generator (Generator): Generates worms at a pre-defined rate.
            The rate at which worms move from one compartment to the next
        dead_worms (WormGroup): Worms dying in each compartment

    Returns:
        WormGroup: The number of worms leaving each compartment due to aging.
    """

    def _calc_outbound_worms_single_group(
        current_worms: Array.WormCat.Person.Int,
        dead_worms: Array.WormCat.Person.Int,
        worm_age_rate_generator: Generator,
    ) -> Array.WormCat.Person.Int:
        return worm_age_rate_generator.binomial(n=current_worms - dead_worms)

    return WormGroup(
        male=_calc_outbound_worms_single_group(
            dead_worms=dead_worms.male,
            current_worms=current_worms.male,
            worm_age_rate_generator=worm_age_rate_generator,
        ),
        infertile=_calc_outbound_worms_single_group(
            dead_worms=dead_worms.infertile,
            current_worms=current_worms.infertile,
            worm_age_rate_generator=worm_age_rate_generator,
        ),
        fertile=_calc_outbound_worms_single_group(
            dead_worms=dead_worms.fertile,
            current_worms=current_worms.fertile,
            worm_age_rate_generator=worm_age_rate_generator,
        ),
    )


def _calc_inbound_worms(
    worm_delay: Array.L3Delay.Person.Int,
    worm_sex_ratio_generator: Generator,
    outbound: WormGroup,
):
    """
    Calculates the inbound worms into each compartment, drawing from the final column of
    worm delay at random

    Args:
        worm_delay (Array.L3Delay.Person.Int): The array of worms delayed
        worm_sex_ratio_generator (Generator): Generates worms at a pre-defined sex ratio
        outbound (WormGroup): The outbound worms from each compartment

    Returns:
        WormGroup: The number of worms entering each compartment due to aging.
    """
    # Takes males and females from final column of worm_delay
    final_column: Array.Person.Int = worm_delay[-1]
    # Gets worms of each sex at random
    delayed_males = worm_sex_ratio_generator.binomial(n=final_column)
    delayed_females = final_column - delayed_males
    return WormGroup(
        male=utils.lag_array(delayed_males, outbound.male),
        infertile=utils.lag_array(delayed_females, outbound.infertile),
        fertile=utils.lag_array(
            np.zeros(outbound.fertile.shape[1], dtype="int"), outbound.fertile
        ),
    )


def _calc_delta_fertility(
    current_worms: WormGroup,
    dead_worms: WormGroup,
    outbound_worms: WormGroup,
    worm_params: WormParams,
    fertile_to_non_fertile_rate: Array.Person.Float | None,
    delta_time: float,
    worm_lambda_zero_generator: Generator,
    worm_omega_generator: Generator,
) -> Array.WormCat.Person.Int:
    """
    Calculates how many worms go from infertile to fertile.

    Args:
        current_worms (WormGroup): The current number of worms
        dead_worms (WormGroup): Worms dying in each compartment
        outbound_worms (WormGroup): Worms moving out of each age compartment
        worm_params (WormParams): The fixed parameters relating to worms
        fertile_to_non_fertile_rate (Array.Person.Float | None): The rate at which worms
            move from fertile to infertile, from treatment
        delta_time (float): dt - The amount of time advance in one time step
        worm_lambda_zero_generator (Generator): Generates infertile worms at a pre-defined rate
        worm_omega_generator (Generator): Generates fertile worms at a pre-defined rate

    Returns:
        Array.WormCat.Person.Int: The number of worms going from infertile to fertile in each compartment
    """

    def _calc_new_worms_fertility(
        current_worms: Array.WormCat.Person.Int,
        dead_worms: Array.WormCat.Person.Int,
        outbound_worms: Array.WormCat.Person.Int,
        prob: None | Array.Person.Float,
        worm_generator: Generator,
    ) -> Array.WormCat.Person.Int:

        remaining_female_worms = current_worms - dead_worms - outbound_worms
        remaining_female_worms[remaining_female_worms < 0] = 0

        if remaining_female_worms.any():
            if prob is None:
                return worm_generator.binomial(n=remaining_female_worms)
            else:
                return utils.fast_binomial(n=remaining_female_worms, p=prob)
        else:
            return np.zeros_like(current_worms)

    if fertile_to_non_fertile_rate is not None:
        lambda_zero_in = np.tile(
            worm_params.lambda_zero * delta_time + fertile_to_non_fertile_rate,
            (current_worms.fertile.shape[0], 1),
        )
    else:
        lambda_zero_in = None

    new_infertile_from_inside = _calc_new_worms_fertility(
        current_worms=current_worms.fertile,
        dead_worms=dead_worms.fertile,
        outbound_worms=outbound_worms.fertile,
        prob=lambda_zero_in,
        worm_generator=worm_lambda_zero_generator,
    )

    # approach assumes individuals which are moved from fertile to non
    # fertile class due to treatment re enter fertile class at standard rate
    new_fertile_from_inside = _calc_new_worms_fertility(
        current_worms=current_worms.infertile,
        dead_worms=dead_worms.infertile,
        outbound_worms=outbound_worms.infertile,
        prob=None,
        worm_generator=worm_omega_generator,
    )
    return new_fertile_from_inside - new_infertile_from_inside


def _calc_new_worms(
    inbound: WormGroup,
    outbound: WormGroup,
    dead: WormGroup,
    current_worms: WormGroup,
    delta_fertility: Array.WormCat.Person.Int,
    debug: bool,
) -> WormGroup:
    """
    Calculates the change in worms

    Args:
        inbound (WormGroup): Worms moving into each age compartment
        outbound (WormGroup): Worms moving out of each age compartment
        dead (WormGroup): Worms dying in each compartment
        current_worms (WormGroup): The current number of worms
        delta_fertility (Array.WormCat.Person.Int): Worms becoming fertile in each compartment
        debug (bool): Runs in debug mode

    Returns:
        WormGroup: The new value for total worms
    """
    transit_male = inbound.male - outbound.male
    new_male = current_worms.male + transit_male - dead.male

    transit_infertile = inbound.infertile - outbound.infertile
    new_infertile = (
        current_worms.infertile - delta_fertility - dead.infertile + transit_infertile
    )

    transit_fertile = inbound.fertile - outbound.fertile
    new_fertile = (
        current_worms.fertile + delta_fertility - dead.fertile + transit_fertile
    )
    if debug:
        assert np.all(
            (new_male >= 0) & (new_infertile >= 0) & (new_fertile >= 0)
        ), "Worms became negative!"
    return WormGroup(male=new_male, infertile=new_infertile, fertile=new_fertile)


def _calc_fertile_to_non_fertile_rate(
    current_time: float,
    lam_m: float,
    phi: float,
    time_of_last_treatment: Array.Person.Float,
    delta_time: float,
) -> Array.Person.Float:
    """
    Calculates the rate of conversion from fertile to non fertile worms based on treatment

    Args:
        current_time (float): The current time, t, in the model
        lam_m (float): From the effects of ivermectin
        phi (float): From the effects of ivermectin
        time_of_last_treatment (_type_): The last time each individual was treated
        delta_time (float): dt - The amount of time advance in one time step

    Returns:
        Array.Person.Float: The rate of conversion from fertile to non fertile worms
        based on treatment
    """
    # individuals which have been treated get additional infertility rate
    lam_m_temp = np.where(np.isnan(time_of_last_treatment), 0, lam_m)
    time_since_treatment = current_time - time_of_last_treatment
    return np.nan_to_num(delta_time * lam_m_temp * np.exp(-phi * time_since_treatment))


def _calc_female_mortalities(
    mortalities: Array.WormCat.Float,
    permanent_infertility: float,
    coverage_in: Array.Person.Bool,
) -> Array.WormCat.Person.Float:
    """
    Calculates the number of mortalities amongst female worms, under treatment.
    permanent_infertility is the proportion of female worms made permanently
    infertile. For simplicity these are killed.

    Args:
        mortalities (Array.WormCat.Float): The default worm mortality rate
        permanent_infertility (float): Permenent infertility in worms due to ivermectin
        coverage_in (Array.Person.Bool): An array stating if each person in the model is treated

    Returns:
        Array.WormCat.Person.Float: The mortality rate of female worms in each treated person
            and compartment
    """
    female_mortalities = np.tile(mortalities, (len(coverage_in), 1))
    female_mortalities[coverage_in] += permanent_infertility
    return female_mortalities.T


def calculate_new_worms(
    current_worms: WormGroup,
    worm_params: WormParams,
    treatment: TreatmentGroup | None,
    time_of_last_treatment: Array.Person.Float,
    delta_time: float,
    worm_delay_array: Array.L3Delay.Person.Int,
    mortalities: Array.WormCat.Float,
    current_time: float,
    debug: bool,
    worm_age_rate_generator: Generator,
    worm_sex_ratio_generator: Generator,
    worm_lambda_zero_generator: Generator,
    worm_omega_generator: Generator,
) -> tuple[WormGroup, Array.Person.Float]:
    """
    Calculates the new total worms in the model for one time step.

    Args:
        current_worms (WormGroup): The current worms in the model
        worm_params (WormParams): The fixed parameters relating to worms
        treatment_params (TreatmentParams | None): The fixed parameters relating to treatment
        time_of_last_treatment (Array.Person.Float): The last time a particular person was
            treated (None if treatment has not started).
        delta_time (float): dt - The amount of time advance in one time step
        worm_delay_array (Array.L3Delay.Person.Int): The array for the worms being delayed
        mortalities (Array.WormCat.Float): The default worm mortality rate
        current_time (float): The current time, t, in the model
        debug (bool): Runs in debug mode
        worm_age_rate_generator (Generator): Generates worms at a pre-defined aging rate
        worm_sex_ratio_generator (Generator): Generates worms at a pre-defined sex ratio
        worm_lambda_zero_generator (Generator): Generates infertile worms at a pre-defined rate
        worm_omega_generator (Generator): Generates fertile worms at a pre-defined rate
    Returns:
        tuple[WormGroup, Array.Person.Float]: Returns new total worms, last time people were treated, respectively
    """

    female_mortalities: Array.WormCat.Float | Array.WormCat.Person.Float = mortalities
    fertile_to_non_fertile_rate = None
    if treatment is not None:
        if treatment.treatment_occurred:
            female_mortalities = _calc_female_mortalities(
                mortalities, worm_params.permanent_infertility, treatment.coverage_in
            )
            time_of_last_treatment = time_of_last_treatment.copy()
            time_of_last_treatment[treatment.coverage_in] = current_time

        fertile_to_non_fertile_rate = _calc_fertile_to_non_fertile_rate(
            current_time=current_time,
            lam_m=worm_params.lam_m,
            phi=worm_params.phi,
            time_of_last_treatment=time_of_last_treatment,
            delta_time=delta_time,
        )

    dead = _calc_dead_worms(
        current_worms=current_worms,
        female_mortalities=female_mortalities,
        male_mortalities=mortalities,
        treatment_occurred=treatment is not None and treatment.treatment_occurred,
    )

    outbound = _calc_outbound_worms(
        current_worms=current_worms,
        worm_age_rate_generator=worm_age_rate_generator,
        dead_worms=dead,
    )

    inbound = _calc_inbound_worms(
        worm_delay=worm_delay_array,
        worm_sex_ratio_generator=worm_sex_ratio_generator,
        outbound=outbound,
    )

    delta_fertility = _calc_delta_fertility(
        current_worms,
        dead,
        outbound,
        worm_params,
        fertile_to_non_fertile_rate,
        delta_time,
        worm_lambda_zero_generator,
        worm_omega_generator,
    )
    return (
        _calc_new_worms(inbound, outbound, dead, current_worms, delta_fertility, debug),
        time_of_last_treatment,
    )
