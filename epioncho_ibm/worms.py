from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from epioncho_ibm.blackfly import delta_h

from .params import BlackflyParams, TreatmentParams, WormParams


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
    n_people: int,
    current_worms: NDArray[np.int_],
    mortalities: NDArray[np.float_],
    worm_age_rate: NDArray[np.float_] | float,
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    dead_worms = np.random.binomial(
        n=current_worms,
        p=mortalities,
        size=n_people,
    )
    aging_worms = np.random.binomial(
        n=current_worms - dead_worms,
        p=worm_age_rate,
        size=n_people,
    )

    return dead_worms, aging_worms


def _calc_new_worms_from_inside(
    current_worms: NDArray[np.int_],
    dead_worms: NDArray[np.int_],
    aging_worms: NDArray[np.int_],
    n_people: int,
    prob: NDArray[np.float_] | float,
) -> NDArray[np.int_]:
    # trans.fc
    delta_female_worms = current_worms - dead_worms - aging_worms
    delta_female_worms[delta_female_worms < 0] = 0

    if delta_female_worms.any():
        new_worms = np.random.binomial(
            n=delta_female_worms,
            p=prob,
            size=n_people,
        )
    else:
        new_worms = 0
    return new_worms


def change_in_worms(
    stages: int,
    current_worms: list[WormGroup],
    worm_params: WormParams,
    treatment_params: Optional[TreatmentParams],
    delta_time: float,
    n_people: int,
    delayed_females: NDArray[np.int_],
    delayed_males: NDArray[np.int_],
    mortalities: NDArray[np.float_],
    coverage_in: Optional[NDArray[np.bool_]],
    initial_treatment_times: Optional[NDArray[np.float_]],
    current_time: float,
    time_of_last_treatment: Optional[NDArray[np.float_]],
):
    last_aging_worms = WormGroup.from_population(n_people)
    all_male_worms = np.empty((stages, n_people), dtype=np.int_)
    all_infertile_female_worms = np.empty((stages, n_people), dtype=np.int_)
    all_fertile_female_worms = np.empty((stages, n_people), dtype=np.int_)

    for compartment in range(stages):
        (
            last_total_worms,
            last_aging_worms,
            last_time_of_last_treatment,
        ) = change_in_worm_per_index(  # res
            worm_params=worm_params,
            treatment_params=treatment_params,
            delta_time=delta_time,
            n_people=n_people,
            delayed_females=delayed_females,
            delayed_males=delayed_males,
            mortalities=mortalities,
            coverage_in=coverage_in,
            initial_treatment_times=initial_treatment_times,
            current_time=current_time,
            time_of_last_treatment=time_of_last_treatment,
            compartment=compartment,
            current_worms=current_worms[compartment],
            last_aging_worms=last_aging_worms,
        )
        check_no_worms_are_negative(last_total_worms)

        all_male_worms[compartment, :] = last_total_worms.male
        all_infertile_female_worms[compartment, :] = last_total_worms.infertile
        all_fertile_female_worms[compartment, :] = last_total_worms.fertile

    return (
        all_male_worms,
        all_infertile_female_worms,
        all_fertile_female_worms,
        last_time_of_last_treatment,
    )


def change_in_worm_per_index(
    worm_params: WormParams,
    treatment_params: Optional[TreatmentParams],
    delta_time: float,
    n_people: int,
    delayed_females: NDArray[np.int_],
    delayed_males: NDArray[np.int_],
    mortalities: NDArray[np.float_],
    coverage_in: Optional[NDArray[np.bool_]],
    initial_treatment_times: Optional[NDArray[np.float_]],
    current_time: float,
    time_of_last_treatment: Optional[NDArray[np.float_]],
    compartment: int,
    current_worms: WormGroup,
    last_aging_worms: WormGroup,
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
    # loss of fertility lambda.zero.in
    lambda_zero_in = worm_params.lambda_zero * delta_time
    omega = worm_params.omega * delta_time  # becoming fertile
    first_compartment = compartment == 0
    compartment_mortality = mortalities[compartment]

    worm_age_rate = delta_time / worm_params.worms_aging
    dead_male_worms, aging_male_worms = _calc_dead_and_aging_worms(
        n_people=n_people,
        current_worms=current_worms.male,
        mortalities=compartment_mortality,
        worm_age_rate=worm_age_rate,
    )

    total_male_worms = (
        current_worms.male
        + (delayed_males if first_compartment else last_aging_worms.male)
        - aging_male_worms
        - dead_male_worms
    )

    #########
    # treatment
    #########
    # approach assumes individuals which are moved from fertile to non
    # fertile class due to treatment re enter fertile class at standard rate
    female_mortalities = compartment_mortality  # mort.fems
    if treatment_params is not None and current_time > treatment_params.start_time:
        assert time_of_last_treatment is not None
        assert initial_treatment_times is not None
        during_treatment = np.any(
            np.logical_and(
                current_time <= initial_treatment_times,
                initial_treatment_times < current_time + delta_time,
            )
        )
        if during_treatment and current_time <= treatment_params.stop_time:
            female_mortalities = np.repeat(compartment_mortality, n_people)  # mort.fems
            assert coverage_in is not None
            # TODO: This only needs to be calculated at compartment 0 - all others repeat calc
            time_of_last_treatment[coverage_in] = current_time  # treat.vec
            # params.permanent_infertility is the proportion of female worms made permanently infertile, killed for simplicity
            female_mortalities[coverage_in] += worm_params.permanent_infertility

        time_since_treatment = current_time - time_of_last_treatment  # tao

        # individuals which have been treated get additional infertility rate
        lam_m_temp = np.where(time_of_last_treatment == np.nan, 0, worm_params.lam_m)
        fertile_to_non_fertile_rate = np.nan_to_num(
            delta_time * lam_m_temp * np.exp(-worm_params.phi * time_since_treatment)
        )
        lambda_zero_in += fertile_to_non_fertile_rate  # update 'standard' fertile to non fertile rate to account for treatment

    dead_infertile_worms, aging_infertile_worms = _calc_dead_and_aging_worms(
        n_people=n_people,
        current_worms=current_worms.infertile,
        mortalities=female_mortalities,
        worm_age_rate=worm_age_rate,
    )
    dead_fertile_worms, aging_fertile_worms = _calc_dead_and_aging_worms(
        n_people=n_people,
        current_worms=current_worms.fertile,
        mortalities=female_mortalities,
        worm_age_rate=worm_age_rate,
    )

    new_worms_infertile_from_inside = _calc_new_worms_from_inside(
        current_worms=current_worms.fertile,
        dead_worms=dead_fertile_worms,
        aging_worms=aging_fertile_worms,
        n_people=n_people,
        prob=lambda_zero_in,
    )  # new.worms.nf.fi

    # females worms from infertile to fertile, this happens independent of males, but production of mf depends on males

    # individuals which still have non fertile worms in an age compartment after death and aging

    new_worms_fertile_from_inside = _calc_new_worms_from_inside(
        current_worms=current_worms.infertile,
        dead_worms=dead_infertile_worms,
        aging_worms=aging_infertile_worms,
        n_people=n_people,
        prob=omega,
    )  # new.worms.f.fi TODO: Are these the right way round?

    delta_fertile = new_worms_fertile_from_inside - new_worms_infertile_from_inside

    infertile_excl_transiting = (
        current_worms.infertile - delta_fertile - dead_infertile_worms
    )
    fertile_excl_transiting = current_worms.fertile + delta_fertile - dead_fertile_worms

    infertile_out = (
        infertile_excl_transiting
        - aging_infertile_worms
        + (delayed_females if first_compartment else last_aging_worms.infertile)
    )

    fertile_out = (
        fertile_excl_transiting - aging_fertile_worms + last_aging_worms.fertile
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
    worm_delay: NDArray[np.int_], n_people: int, worm_sex_ratio: float
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    final_column = np.array(worm_delay[-1], dtype=int)
    assert len(final_column) == n_people
    last_males = np.random.binomial(
        n=final_column, p=worm_sex_ratio, size=len(final_column)
    )  # new.worms.m
    last_females = final_column - last_males  # new.worms.nf
    return last_males, last_females


def _w_plus_one_rate(
    blackfly_params: BlackflyParams,
    delta_time: float,
    L3: float,
    total_exposure: NDArray[np.float_],
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
    dh = delta_h(blackfly_params, L3, total_exposure)
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


def calc_new_worms(
    L3: NDArray[np.float_],
    blackfly_params: BlackflyParams,
    delta_time: float,
    total_exposure: NDArray[np.float_],
    n_people: int,
) -> NDArray[np.int_]:
    new_rate = _w_plus_one_rate(
        blackfly_params,
        delta_time,
        np.mean(L3),
        total_exposure,
    )
    assert not (np.any(new_rate > 10**10))
    new_worms = np.random.poisson(lam=new_rate, size=n_people)
    return new_worms


def check_no_worms_are_negative(worms: WormGroup):
    if np.any(
        np.logical_or(
            np.logical_or(worms.male < 0, worms.fertile < 0),
            worms.infertile < 0,
        )
    ):
        candidate_people_male_worms = worms.male[worms.male < 0]
        candidate_people_fertile_worms = worms.fertile[worms.fertile < 0]
        candidate_people_infertile_worms = worms.infertile[worms.infertile < 0]

        raise RuntimeError(
            f"Worms became negative: \nMales: {candidate_people_male_worms} \nFertile Females: {candidate_people_fertile_worms} \nInfertile Females: {candidate_people_infertile_worms}"
        )
