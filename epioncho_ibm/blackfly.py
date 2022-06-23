import numpy as np
from numpy.typing import NDArray

from .params import Params

# L1, L2, L3 (parasite life stages) dynamics in the fly population
# assumed to be at equilibrium
# delay of 4 days for parasites moving from L1 to L2


def calc_l1(
    params: Params,
    microfil: NDArray[np.float_],
    last_microfil_delay: NDArray[np.float_],
    total_exposure: NDArray[np.float_],
    exposure_delay: NDArray[np.float_],
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
    delta_vv = params.delta_v0 / (1 + params.c_v * microfil * total_exposure)
    return (
        delta_vv * params.bite_rate_per_fly_on_human * microfil * total_exposure
    ) / (
        params.blackfly_mort_per_person_per_year
        + (params.blackfly_mort_from_mf_per_person_per_year * microfil * total_exposure)
        + params.l1_l2_per_person_per_year
        * np.exp(
            -(4 / 365)
            * (
                params.blackfly_mort_per_person_per_year
                + (
                    params.blackfly_mort_from_mf_per_person_per_year
                    * last_microfil_delay
                    * exposure_delay
                )
            )
        )
    )


def calc_l2(
    params: Params,
    l1_delay: NDArray[np.float_],
    microfil: NDArray[np.float_],
    total_exposure: NDArray[np.float_],
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
            params.l1_l2_per_person_per_year
            * np.exp(
                -(4 / 366)
                * (
                    params.blackfly_mort_per_person_per_year
                    + (
                        params.blackfly_mort_from_mf_per_person_per_year
                        * microfil
                        * total_exposure
                    )
                )
            )
        )
    ) / (params.blackfly_mort_per_person_per_year + params.l2_l3_per_person_per_year)


def calc_l3(
    params: Params,
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
    return (params.l2_l3_per_person_per_year * l2) / (
        (params.a_H / params.recip_gono_cycle)
        + params.blackfly_mort_per_person_per_year
        + params.sigma_L0
    )
