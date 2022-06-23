from copy import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .params import Params


@dataclass
class WormGroup:
    male: NDArray[np.int_]
    infertile: NDArray[np.int_]
    fertile: NDArray[np.int_]

    @classmethod
    def from_population(cls, population: int):
        return cls(
            male=np.zeros(population, dtype=int),
            infertile=np.zeros(population, dtype=int),
            fertile=np.zeros(population, dtype=int),
        )


def _calc_dead_and_aging_worms(
    params: Params, current_worms: NDArray[np.int_], mortalities: NDArray[np.float_]
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    dead_worms = np.random.binomial(
        n=current_worms,
        p=mortalities,
        size=params.human_population,
    )
    aging_worms = np.random.binomial(
        n=current_worms - dead_worms,
        p=np.repeat(params.delta_time / params.worms_aging, params.human_population),
        size=params.human_population,
    )
    return dead_worms, aging_worms


def _calc_new_worms_from_inside(
    current_worms: NDArray[np.int_],
    dead_worms: NDArray[np.int_],
    aging_worms: NDArray[np.int_],
    human_population: int,
    prob: NDArray[np.float_],
) -> NDArray[np.int_]:
    delta_fertile_female_worms = current_worms - dead_worms - aging_worms  # trans.fc
    true_delta_fertile_female_worms = np.where(
        delta_fertile_female_worms > 0, delta_fertile_female_worms, 0
    )

    if np.sum(true_delta_fertile_female_worms) > 0:
        new_worms = np.random.binomial(
            n=true_delta_fertile_female_worms,
            p=prob,
            size=human_population,
        )
    else:
        new_worms = np.zeros(human_population, dtype=int)
    return new_worms


def change_in_worm_per_index(
    params: Params,
    state,
    delayed_females: NDArray[np.int_],
    delayed_males: NDArray[np.int_],
    worm_mortality_rate: NDArray[np.float_],
    coverage_in: Optional[NDArray[np.bool_]],
    last_aging_worms: WormGroup,
    initial_treatment_times: Optional[NDArray[np.float_]],
    current_time: float,
    compartment: int,
    time_of_last_treatment: Optional[NDArray[np.float_]],
) -> Tuple[WormGroup, WormGroup, Optional[NDArray[np.float_]],]:
    """
    params.delta_hz # delta.hz
    params.delta_hinf # delta.hinf
    params.c_h # c.h
    params.annual_transm_potential # "m"
    params.bite_rate_per_fly_on_human #"beta"
    "compartment" Corresponds to worm column
    params.worm_age_stages "num.comps"
    params.omega "omeg"
    params.lambda_zero "lambda.zero"
    params.human_population "N"
    params.lam_m "lam.m"
    params.phi "phi"
    last_males "new.worms.m"
    last_females "new.worms.nf.fo"
    total_exposure "tot.ex.ai"
    params.delta_time "DT"
    params.treatment_start_time "treat.start"
    params.treatment_stop_time "treat.stop"
    worm_mortality_rate "mort.rates.worms"
    params.total_population_coverage "treat.prob"
    params.treatment_interval "treat.int"
    coverage_in "onchosim.cov/inds.to.treat"
    last_change "w.f.l.c"
    params.permanent_infertility "cum.infer"
    worms.start/ws used to refer to start point in giant array for worms
    initial_treatment_times "times.of.treat.in"
    iteration/i now means current_time
    if initial_treatment_times is None give.treat is false etc
    N is params.human_population
    params.worms_aging "time.each.comp"
    """

    lambda_zero_in = np.repeat(
        params.lambda_zero * params.delta_time, params.human_population
    )  # loss of fertility lambda.zero.in
    omega = np.repeat(
        params.omega * params.delta_time, params.human_population
    )  # becoming fertile
    # male worms
    current_male_worms = state.people.male_worms[compartment]  # cur.Wm
    compartment_mortality = np.repeat(
        worm_mortality_rate[compartment], params.human_population
    )
    dead_male_worms, aging_male_worms = _calc_dead_and_aging_worms(
        params=params,
        current_worms=current_male_worms,
        mortalities=compartment_mortality,
    )
    if compartment == 0:
        total_male_worms = (
            current_male_worms + delayed_males - aging_male_worms - dead_male_worms
        )
    else:
        total_male_worms = (
            current_male_worms
            + last_aging_worms.male
            - aging_male_worms
            - dead_male_worms
        )

    # female worms

    current_female_worms_infertile = state.people.infertile_female_worms[
        compartment
    ]  # cur.Wm.nf
    current_female_worms_fertile = state.people.fertile_female_worms[
        compartment
    ]  # cur.Wm.f

    female_mortalities = copy(compartment_mortality)  # mort.fems
    #########
    # treatment
    #########

    # approach assumes individuals which are moved from fertile to non
    # fertile class due to treatment re enter fertile class at standard rate

    if (
        initial_treatment_times is not None
        and current_time > params.treatment_start_time
    ):
        assert time_of_last_treatment is not None
        during_treatment = np.any(
            np.logical_and(
                current_time <= initial_treatment_times,
                initial_treatment_times < current_time + params.delta_time,
            )
        )
        if during_treatment and current_time <= params.treatment_stop_time:
            assert coverage_in is not None
            # TODO: This only needs to be calculated at compartment 0 - all others repeat calc
            time_of_last_treatment[coverage_in] = current_time  # treat.vec
            # params.permanent_infertility is the proportion of female worms made permanently infertile, killed for simplicity
            female_mortalities[coverage_in] = (
                female_mortalities[coverage_in] + params.permanent_infertility
            )

        time_since_treatment = current_time - time_of_last_treatment  # tao

        # individuals which have been treated get additional infertility rate
        lam_m_temp = np.where(time_of_last_treatment == np.nan, 0, params.lam_m)
        fertile_to_non_fertile_rate = np.nan_to_num(
            params.delta_time * lam_m_temp * np.exp(-params.phi * time_since_treatment)
        )
        lambda_zero_in += fertile_to_non_fertile_rate  # update 'standard' fertile to non fertile rate to account for treatment
    ############################################################
    # .fi = 'from inside': worms moving from a fertile or infertile compartment
    # .fo = 'from outside': completely new adult worms
    dead_infertile_worms, aging_infertile_worms = _calc_dead_and_aging_worms(
        params=params,
        current_worms=current_female_worms_infertile,
        mortalities=female_mortalities,
    )
    dead_fertile_worms, aging_fertile_worms = _calc_dead_and_aging_worms(
        params=params,
        current_worms=current_female_worms_fertile,
        mortalities=female_mortalities,
    )

    new_worms_infertile_from_inside = _calc_new_worms_from_inside(
        current_worms=current_female_worms_fertile,
        dead_worms=dead_fertile_worms,
        aging_worms=aging_fertile_worms,
        human_population=params.human_population,
        prob=lambda_zero_in,
    )  # new.worms.nf.fi

    # females worms from infertile to fertile, this happens independent of males, but production of mf depends on males

    # individuals which still have non fertile worms in an age compartment after death and aging

    new_worms_fertile_from_inside = _calc_new_worms_from_inside(
        current_worms=current_female_worms_infertile,
        dead_worms=dead_infertile_worms,
        aging_worms=aging_infertile_worms,
        human_population=params.human_population,
        prob=omega,
    )  # new.worms.f.fi TODO: Are these the right way round?

    if compartment == 0:
        infertile_out = (
            current_female_worms_infertile
            + delayed_females
            + new_worms_infertile_from_inside
            - aging_infertile_worms
            - dead_infertile_worms
            - new_worms_fertile_from_inside
        )  # nf.out
        fertile_out = (
            current_female_worms_fertile
            + new_worms_fertile_from_inside
            - aging_fertile_worms
            - dead_fertile_worms
            - new_worms_infertile_from_inside
        )

    else:
        infertile_out = (
            current_female_worms_infertile
            + new_worms_infertile_from_inside
            - aging_infertile_worms
            - new_worms_fertile_from_inside
            + last_aging_worms.infertile
            - dead_infertile_worms
        )
        fertile_out = (
            current_female_worms_fertile
            + new_worms_fertile_from_inside
            - aging_fertile_worms
            - dead_fertile_worms
            - new_worms_infertile_from_inside
            + last_aging_worms.fertile
        )
    new_aging_worms = WormGroup(
        male=aging_male_worms,
        infertile=aging_infertile_worms,
        fertile=aging_fertile_worms,
    )
    new_total_worms = WormGroup(
        male=total_male_worms, infertile=infertile_out, fertile=fertile_out
    )
    return (
        new_total_worms,
        new_aging_worms,
        time_of_last_treatment,
    )


def get_delayed_males_and_females(
    worm_delay: NDArray[np.int_], params: Params
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    final_column = np.array(worm_delay[-1], dtype=int)
    assert len(final_column) == params.human_population
    last_males = np.random.binomial(
        n=final_column, p=0.5, size=len(final_column)
    )  # new.worms.m
    last_females = final_column - last_males  # new.worms.nf
    return last_males, last_females


def _delta_h(
    params: Params, L3: float, total_exposure: NDArray[np.float_]
) -> NDArray[np.float_]:
    # proportion of L3 larvae (final life stage in the fly population) developing into adult worms in humans
    # expos is the total exposure for an individual
    # delta.hz, delta.hinf, c.h control the density dependent establishment of parasites
    multiplier = (
        params.c_h
        * params.annual_transm_potential
        * params.bite_rate_per_fly_on_human
        * L3
        * total_exposure
    )
    return (params.delta_hz + (params.delta_hinf * multiplier)) / (1 + multiplier)


def _w_plus_one_rate(
    params: Params, L3: float, total_exposure: NDArray[np.float_]
) -> NDArray[np.float_]:
    """
    params.delta_hz # delta.hz
    params.delta_hinf # delta.hinf
    params.c_h # c.h
    params.annual_transm_potential # "m"
    params.bite_rate_per_fly_on_human #"beta"
    total_exposure # "expos"
    params.delta_time #"DT"
    """
    dh = _delta_h(params, L3, total_exposure)
    return (
        params.delta_time
        * params.annual_transm_potential
        * params.bite_rate_per_fly_on_human
        * dh
        * total_exposure
        * L3
    )


def calc_new_worms(state, total_exposure) -> NDArray[np.int_]:
    new_rate = _w_plus_one_rate(
        state.params, np.mean(state.people.blackfly.L3), total_exposure
    )
    if np.any(new_rate > 10**10):
        st_dev = np.sqrt(new_rate)
        new_worms: NDArray[np.int_] = np.round(
            np.random.normal(
                loc=new_rate, scale=st_dev, size=state.params.human_population
            )
        )
    else:
        new_worms = np.random.poisson(lam=new_rate, size=state.params.human_population)
    return new_worms
