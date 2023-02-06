import numpy as np
from numpy.random import Generator

from epioncho_ibm.state import Array, BlackflyParams, WormGroup

"""
L1, L2, L3 (parasite life stages) dynamics in the fly population
Assumed to be at equilibrium
Delay of 4 days for parasites moving from L1 to L2
"""


def calc_l1(
    blackfly_params: BlackflyParams,
    microfil: Array.Person.Float,
    last_microfil_delay: Array.Person.Float,
    total_exposure: Array.Person.Float,
    exposure_delay: Array.Person.Float,
    year_length: float,
) -> Array.Person.Float:
    """
    Calculates the amount of L1 larvae in the blackflies, associated by person

    Args:
        blackfly_params (BlackflyParams): A fixed set of parameters associated with blackflies
        microfil (Array.Person.Float): The amount of microfilariae at time t
        last_microfil_delay (Array.Person.Float): The final column of microfil delay
        total_exposure (Array.Person.Float): The overall exposure of each person to infection
        exposure_delay (Array.Person.Float): The final column of exposure delay
        year_length (float): The length of a year

    Returns:
        Array.Person.Float: The number of L1 Larvae associated with each person
    """
    # proportion of mf per mg developing into infective larvae within the vector
    delta_vv = blackfly_params.delta_v0 / (
        1 + blackfly_params.c_v * microfil * total_exposure
    )
    return (
        delta_vv
        * blackfly_params.bite_rate_per_fly_on_human
        * microfil
        * total_exposure
    ) / (
        blackfly_params.blackfly_mort_per_fly_per_year
        + (
            blackfly_params.blackfly_mort_from_mf_per_fly_per_year
            * microfil
            * total_exposure
        )
        + blackfly_params.l1_l2_per_larva_per_year
        * np.exp(
            -(blackfly_params.l1_delay / year_length)
            * (
                blackfly_params.blackfly_mort_per_fly_per_year
                + (
                    blackfly_params.blackfly_mort_from_mf_per_fly_per_year
                    * last_microfil_delay
                    * exposure_delay
                )
            )
        )
    )


def calc_l2(
    blackfly_params: BlackflyParams,
    l1: Array.Person.Float,
    last_microfil_delay: Array.Person.Float,
    exposure_delay: Array.Person.Float,
    year_length: float,
) -> Array.Person.Float:
    """
    Calculates the amount of L2 larvae in the blackflies, associated by person

    Args:
        blackfly_params (BlackflyParams): A fixed set of parameters associated with blackflies
        l1 (Array.Person.Float): The amount of L1 Larvae associated with each person at time t
        last_microfil_delay (Array.Person.Float): The final column of microfil delay
        exposure_delay (Array.Person.Float): The final column of exposure delay
        year_length (float): The length of a year

    Returns:
        Array.Person.Float: The number of L2 Larvae associated with each person
    """
    return (
        l1
        * (
            blackfly_params.l1_l2_per_larva_per_year
            * np.exp(
                -(blackfly_params.l1_delay / year_length)
                * (
                    blackfly_params.blackfly_mort_per_fly_per_year
                    + (
                        blackfly_params.blackfly_mort_from_mf_per_fly_per_year
                        * last_microfil_delay
                        * exposure_delay
                    )
                )
            )
        )
    ) / (
        blackfly_params.blackfly_mort_per_fly_per_year
        + blackfly_params.l2_l3_per_larva_per_year
    )


def calc_l3(
    blackfly_params: BlackflyParams,
    l2: Array.Person.Float,
) -> Array.Person.Float:
    """
    Calculates the amount of L3 larvae in the blackflies, associated by person

    Args:
        blackfly_params (BlackflyParams): A fixed set of parameters associated with blackflies
        l2 (Array.Person.Float): The amount of L2 Larvae associated with each person at time t

    Returns:
        Array.Person.Float: The number of L3 Larvae associated with each person
    """

    return (blackfly_params.l2_l3_per_larva_per_year * l2) / (
        (blackfly_params.a_H / blackfly_params.gonotrophic_cycle_length)
        + blackfly_params.blackfly_mort_per_fly_per_year
        + blackfly_params.mu_L3
    )


def _delta_h(
    blackfly_params: BlackflyParams,
    L3: float,
    total_exposure: Array.Person.Float,
    current_worms: WormGroup,
) -> Array.Person.Float:
    """
    Calculates the proportion of L3 larvae (final life stage in the fly population)
    developing into adult worms in humans

    Args:
        blackfly_params (BlackflyParams): A fixed set of parameters associated with blackflies.
            Parameters delta.hz, delta.hinf, c.h control the density dependent establishment of parasites
        L3 (float): The average amount of L3 larvae
        total_exposure (Array.Person.Float): The overall exposure of each person to infection
        current_worms (WormGroup): The current number of worms

    Returns:
        Array.Person.Float: The proportion of L3 larvae developing into adult worms for each person
    """
    annual_transm_potential = (
        blackfly_params.bite_rate_per_person_per_year
        / blackfly_params.bite_rate_per_fly_on_human
    )
    multiplier: Array.Person.Float = (
        blackfly_params.c_h
        * annual_transm_potential
        * blackfly_params.bite_rate_per_fly_on_human
        * L3
        * total_exposure
    )
    return (
        blackfly_params.delta_h_zero + (blackfly_params.delta_h_inf * multiplier)
    ) / (1 + multiplier)


def _calc_rate_of_l3_to_worms(
    blackfly_params: BlackflyParams,
    delta_time: float,
    L3: float,
    total_exposure: Array.Person.Float,
    current_worms: WormGroup,
) -> Array.Person.Float:
    """
    Calculates the rate at which L3 Larvae become worms in the human host
    AKA: W+1 rate

    Args:
        blackfly_params (BlackflyParams): A fixed set of parameters associated with blackflies
        delta_time (float): dt - one unit of time
        L3 (float): The average amount of L3 larvae
        total_exposure (Array.Person.Float): The overall exposure of each person to infection
        current_worms (WormGroup): The current number of worms

    Returns:
        Array.Person.Float: The rate at which L3 larvae become worms in each person
    """
    dh = _delta_h(blackfly_params, L3, total_exposure, current_worms=current_worms)
    annual_transm_potential = (
        blackfly_params.bite_rate_per_person_per_year
        / blackfly_params.bite_rate_per_fly_on_human
    )
    return (
        delta_time
        * annual_transm_potential
        * blackfly_params.bite_rate_per_fly_on_human
        * dh
        * total_exposure
        * L3
    )


def calc_new_worms_from_blackfly(
    L3: Array.Person.Float,
    blackfly_params: BlackflyParams,
    delta_time: float,
    total_exposure: Array.Person.Float,
    n_people: int,
    current_worms: WormGroup,
    debug: bool,
    numpy_bit_gen: Generator,
) -> Array.Person.Int:
    """
    Calculates the number of new worms produced based on the number of L3 larvae

    Args:
        L3 (Array.Person.Float): The number of L3 Larvae associated with each person
        blackfly_params (BlackflyParams): A fixed set of parameters associated with blackflies
        delta_time (float): dt - one unit of time
        total_exposure (Array.Person.Float): The overall exposure of each person to infection
        n_people (int): The total number of people
        current_worms (WormGroup): The current number of worms
        debug (bool): Runs in debug mode
        numpy_bit_gen: (Generator): The random number generator for numpy

    Returns:
        Array.Person.Int: The number of new worms produced by L3 larvae
    """
    new_rate = _calc_rate_of_l3_to_worms(
        blackfly_params, delta_time, float(np.mean(L3)), total_exposure, current_worms
    )
    if debug:
        assert not np.any(new_rate > 10**10)
    new_worms = numpy_bit_gen.poisson(lam=new_rate, size=n_people)
    return new_worms
