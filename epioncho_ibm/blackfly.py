import numpy as np
from numpy.typing import NDArray

from .params import BlackflyParams

# L1, L2, L3 (parasite life stages) dynamics in the fly population
# assumed to be at equilibrium
# delay of 4 days for parasites moving from L1 to L2


def calc_l1(
    blackfly_params: BlackflyParams,
    microfil: NDArray[np.float_],
    last_microfil_delay: NDArray[np.float_],
    total_exposure: NDArray[np.float_],
    exposure_delay: NDArray[np.float_],
    year_length: float,
) -> NDArray[np.float_]:
    """
    microfil # mf
    last_microfil_delay # mf.delay.in
    total_exposure # expos
    params.delta_v0 #delta.vo
    params.bite_rate_per_fly_on_human # beta
    params.c_v #c.v
    params.l1_l2_per_person_per_year # nuone
    params.blackfly_mort_per_person_per_year # mu.v
    params.blackfly_mort_from_mf_per_person_per_year # a.v
    exposure_delay # expos.delay
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
        blackfly_params.blackfly_mort_per_person_per_year
        + (
            blackfly_params.blackfly_mort_from_mf_per_person_per_year
            * microfil
            * total_exposure
        )
        + blackfly_params.l1_l2_per_person_per_year
        * np.exp(
            -(blackfly_params.l1_delay / year_length)
            * (
                blackfly_params.blackfly_mort_per_person_per_year
                + (
                    blackfly_params.blackfly_mort_from_mf_per_person_per_year
                    * last_microfil_delay
                    * exposure_delay
                )
            )
        )
    )


def calc_l2(
    blackfly_params: BlackflyParams,
    l1_delay: NDArray[np.float_],
    microfil: NDArray[np.float_],
    total_exposure: NDArray[np.float_],
    year_length: float,
) -> NDArray[np.float_]:
    """
    params.l1_l2_per_person_per_year # nuone
    params.blackfly_mort_per_person_per_year # mu.v
    params.l2_l3_per_person_per_year # nutwo
    params.blackfly_mort_from_mf_per_person_per_year # a.v
    l1_delay # L1.in
    microfil # mf
    total_exposure # expos
    """
    return (
        l1_delay
        * (
            blackfly_params.l1_l2_per_person_per_year
            * np.exp(
                -(blackfly_params.l1_delay / (year_length + 1))
                * (
                    blackfly_params.blackfly_mort_per_person_per_year
                    + (
                        blackfly_params.blackfly_mort_from_mf_per_person_per_year
                        * microfil
                        * total_exposure
                    )
                )
            )
        )
    ) / (
        blackfly_params.blackfly_mort_per_person_per_year
        + blackfly_params.l2_l3_per_person_per_year
    )


def calc_l3(
    blackfly_params: BlackflyParams,
    l2: NDArray[np.float_],
) -> NDArray[np.float_]:
    """
    params.l2_l3_per_person_per_year # nutwo
    l2 # L2.in
    params.a_H # a.H
    params.recip_gono_cycle # g
    params.blackfly_mort_per_person_per_year # mu.v
    params.sigma_L0 # sigma.L0
    """
    return (blackfly_params.l2_l3_per_person_per_year * l2) / (
        (blackfly_params.a_H / blackfly_params.recip_gono_cycle)
        + blackfly_params.blackfly_mort_per_person_per_year
        + blackfly_params.sigma_L0
    )


def delta_h(
    blackfly_params: BlackflyParams, L3: float, total_exposure: NDArray[np.float_]
) -> NDArray[np.float_]:
    # proportion of L3 larvae (final life stage in the fly population) developing into adult worms in humans
    # expos is the total exposure for an individual
    # delta.hz, delta.hinf, c.h control the density dependent establishment of parasites
    annual_transm_potential = (
        blackfly_params.bite_rate_per_person_per_year
        / blackfly_params.bite_rate_per_fly_on_human
    )
    multiplier = (
        blackfly_params.c_h
        * annual_transm_potential
        * blackfly_params.bite_rate_per_fly_on_human
        * L3
        * total_exposure
    )
    return (
        blackfly_params.delta_h_zero + (blackfly_params.delta_h_inf * multiplier)
    ) / (1 + multiplier)
