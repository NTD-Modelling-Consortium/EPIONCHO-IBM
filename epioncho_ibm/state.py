import math
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from .params import Params

np.seterr(all="ignore")


def negative_binomial_alt_interface(
    n: NDArray[np.float_], mu: NDArray[np.float_]
) -> NDArray[np.int_]:
    non_zero_n = n[n > 0]
    rel_prob = non_zero_n / (non_zero_n + mu[n > 0])
    temp_output = np.random.negative_binomial(
        n=non_zero_n, p=rel_prob, size=len(non_zero_n)
    )
    output = np.zeros(len(n), dtype=int)
    output[n > 0] = temp_output
    return output


class RandomConfig(BaseModel):
    gender_ratio: float = 0.5
    noncompliant_percentage: float = 0.05


@dataclass
class BlackflyLarvae:
    L1: NDArray[np.float_]  # 4: L1
    L2: NDArray[np.float_]  # 5: L2
    L3: NDArray[np.float_]  # 6: L3


class NumericArrayStat(BaseModel):
    mean: float
    # st_dev: float

    @classmethod
    def from_array(cls, array: Union[NDArray[np.float_], NDArray[np.int_]]):
        return cls(mean=np.mean(array))  # , st_dev=np.std(array))


class StateStats(BaseModel):
    percent_compliant: float
    percent_male: float
    L1: NumericArrayStat
    L2: NumericArrayStat
    L3: NumericArrayStat
    ages: NumericArrayStat
    mf: NumericArrayStat
    male_worms: NumericArrayStat
    infertile_female_worms: NumericArrayStat
    fertile_female_worms: NumericArrayStat
    mf_per_skin_snip: float
    population_prevalence: float


@dataclass
class People:
    compliance: NDArray[np.bool_]  # 1: 'column used during treatment'
    sex_is_male: NDArray[np.bool_]  # 3: sex
    blackfly: BlackflyLarvae
    ages: NDArray[np.float_]  # 2: current age
    mf: NDArray[np.float_]  # 2D Array, (N, age stage): microfilariae stages 7-28 (21)
    male_worms: NDArray[
        np.int_
    ]  # 2D Array, (N, age stage): worm stages 29-92 (63) males(21), infertile females(21), fertile females(21)
    infertile_female_worms: NDArray[np.int_]
    fertile_female_worms: NDArray[np.int_]

    # mf_current_quantity: NDArray[np.int_]
    # exposure: NDArray[np.float_]
    # new_worm_rate: NDArray[np.float_]
    time_of_last_treatment: NDArray[np.float_]  # treat.vec

    def __len__(self):
        return len(self.compliance)


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


class DerivedParams:
    worm_mortality_rate: NDArray[np.float_]
    fecundity_rates_worms: NDArray[np.float_]
    microfillarie_mortality_rate: NDArray[np.float_]
    initial_treatment_times: Optional[NDArray[np.float_]]
    individual_exposure: NDArray[np.float_]

    def __init__(self, params: Params) -> None:

        worm_age_categories = np.arange(
            start=0,
            stop=params.max_worm_age,
            step=params.max_worm_age / params.worm_age_stages,
        )  # age.cats
        self.worm_mortality_rate = weibull_mortality(
            params.delta_time, params.mu_worms1, params.mu_worms2, worm_age_categories
        )
        self.fecundity_rates_worms = (
            1.158305
            * params.fecundity_worms_1
            / (
                params.fecundity_worms_1
                + (params.fecundity_worms_2 ** (-worm_age_categories))
                - 1
            )
        )

        # TODO revisit +1 and -1
        microfillarie_age_categories = np.arange(
            start=0,
            stop=params.max_microfil_age + 1,
            step=params.max_microfil_age / (params.microfil_age_stages - 1),
        )  # age.cats.mf

        self.microfillarie_mortality_rate = weibull_mortality(
            params.delta_time,
            params.mu_microfillarie1,
            params.mu_microfillarie2,
            microfillarie_age_categories,
        )

        if params.give_treatment:
            treatment_number = (
                params.treatment_stop_time - params.treatment_start_time
            ) / params.treatment_interval_yrs
            if round(treatment_number) != treatment_number:
                raise ValueError(
                    f"Treatment times could not be found for start: {params.treatment_start_time}, stop: {params.treatment_stop_time}, interval: {params.treatment_interval_yrs}"
                )
            treatment_number_int: int = math.ceil(treatment_number)
            self.initial_treatment_times = np.linspace(  # "times.of.treat.in"
                start=params.treatment_start_time,
                stop=params.treatment_stop_time,
                num=treatment_number_int + 1,
            )
        else:
            self.initial_treatment_times = None

        individual_exposure = (
            np.random.gamma(  # individual level exposure to fly bites "ex.vec"
                shape=params.gamma_distribution,
                scale=params.gamma_distribution,
                size=params.human_population,
            )
        )
        self.individual_exposure = individual_exposure / np.mean(
            individual_exposure
        )  # normalise


class DelayArrays:
    worm_delay: NDArray[np.int_]
    exposure_delay: NDArray[np.float_]
    mf_delay: NDArray[np.int_]

    def __init__(self, params: Params, individual_exposure) -> None:
        number_of_worm_delay_cols = math.ceil(
            params.l3_delay * 28 / (params.delta_time * 365)
        )
        self.worm_delay = np.zeros(
            (number_of_worm_delay_cols, params.human_population), dtype=int
        )
        # matrix for exposure (to fly bites) for L1 delay
        number_of_exposure_columns = math.ceil(4 / (params.delta_time * 365))

        self.exposure_delay = np.tile(
            individual_exposure, (number_of_exposure_columns, 1)
        )  # exposure.delay

        # matrix for tracking mf for L1 delay
        number_of_mf_columns = math.ceil(4 / (params.delta_time * 365))
        self.mf_delay = (
            np.ones((number_of_mf_columns, params.human_population), dtype=int)
            * params.initial_mf
        )  # mf.delay
        # L1 delay in flies
        self.l1_delay = np.repeat(params.initial_L1, params.human_population)


class State:
    current_iteration: int = 0
    people: People
    _params: Params
    _derived_params: Optional[DerivedParams]
    _delay_arrays: Optional[DelayArrays]

    def __init__(self, people: People, params: Params) -> None:
        if len(people) != params.human_population:
            raise ValueError(
                f"People length ({len(people)}) inconsistent with params ({params.human_population})"
            )
        self.people = people
        self.params = params
        self._derived_params = None
        self._delay_arrays = None

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._derived_params = None
        self._params = value

    @property
    def derived_params(self) -> DerivedParams:
        if self._derived_params is None:
            self._derived_params = DerivedParams(self.params)
        return self._derived_params

    @property
    def delay_arrays(self) -> DelayArrays:
        if self._delay_arrays is None:
            self._delay_arrays = DelayArrays(
                self.params, self.derived_params.individual_exposure
            )
        return self._delay_arrays

    @delay_arrays.setter
    def delay_arrays(self, value):
        self._delay_arrays = value

    @classmethod
    def generate_random(cls, random_config: RandomConfig, params: Params) -> "State":
        n_people = params.human_population
        sex_array = (
            np.random.uniform(low=0, high=1, size=n_people) < random_config.gender_ratio
        )
        np.zeros(n_people)
        compliance_array = (
            np.random.uniform(low=0, high=1, size=n_people)
            > random_config.noncompliant_percentage
        )
        time_of_last_treatment = np.empty(n_people)
        time_of_last_treatment[:] = np.nan

        return cls(
            people=People(
                compliance=compliance_array,
                ages=np.zeros(n_people),
                sex_is_male=sex_array,
                blackfly=BlackflyLarvae(
                    L1=np.repeat(params.initial_L1, n_people),
                    L2=np.repeat(params.initial_L2, n_people),
                    L3=np.repeat(params.initial_L3, n_people),
                ),
                mf=np.ones((params.microfil_age_stages, n_people)) * params.initial_mf,
                male_worms=np.ones((params.worm_age_stages, n_people), dtype=int)
                * params.initial_worms,
                infertile_female_worms=np.ones(
                    (params.worm_age_stages, n_people), dtype=int
                )
                * params.initial_worms,
                fertile_female_worms=np.ones(
                    (params.worm_age_stages, n_people), dtype=int
                )
                * params.initial_worms,
                # mf_current_quantity=np.zeros(n_people, dtype=int),
                # exposure=np.zeros(n_people),
                # new_worm_rate=np.zeros(n_people),
                time_of_last_treatment=time_of_last_treatment,
            ),
            params=params,
        )

    def microfilariae_per_skin_snip(self: "State") -> Tuple[float, NDArray[np.float_]]:
        """
        #people are tested for the presence of mf using a skin snip, we assume mf are overdispersed in the skin
        #function calculates number of mf in skin snip for all people

        params.skin_snip_weight # ss.wt
        params.skin_snip_number # num.ss
        params.slope_kmf # slope.kmf
        params.initial_kmf # int.kMf
        params.human_population # pop.size
        Determined by new structure
        nfw.start,
        fw.end,
        mf.start,
        mf.end,
        """
        # rowSums(da... sums up adult worms for all individuals giving a vector of kmfs
        # TODO: Note that the worms used here were only female, not total - is this correct?
        kmf = self.params.slope_kmf * np.sum(
            self.people.fertile_female_worms + self.people.infertile_female_worms,
            axis=0,
        )
        mu = self.params.skin_snip_weight * np.sum(self.people.mf, axis=0)
        if self.params.skin_snip_number > 1:
            total_skin_snip_mf = np.zeros(
                (self.params.human_population, self.params.skin_snip_number)
            )
            for i in range(self.params.skin_snip_number):
                total_skin_snip_mf[:, i] = negative_binomial_alt_interface(n=kmf, mu=mu)
            mfobs = np.sum(total_skin_snip_mf, axis=1)
        else:
            mfobs = negative_binomial_alt_interface(n=kmf, mu=mu)
        mfobs = mfobs / (self.params.skin_snip_number * self.params.skin_snip_weight)
        return np.mean(mfobs), mfobs

    def mf_prevalence_in_population(self: "State") -> float:
        """
        Returns a decimal representation of mf prevalence in skinsnip aged population.
        """
        pop_over_min_age_array = self.people.ages >= self.params.min_skinsnip_age
        _, mf_skin_snip = self.microfilariae_per_skin_snip()
        infected_over_min_age = np.sum(mf_skin_snip[pop_over_min_age_array] > 0)
        total_over_min_age = np.sum(pop_over_min_age_array)
        return infected_over_min_age / total_over_min_age

    def dist_population_age(
        self,
        num_iter: int = 1,
        params: Optional[Params] = None,
    ):
        """
        Generate age distribution
        create inital age distribution and simulate stable age distribution
        """
        if params is None:
            params = self.params

        current_ages = self.people.ages
        size_population = len(self.people)
        delta_time_vector = np.ones(size_population) * params.delta_time
        for _ in range(num_iter):
            current_ages += delta_time_vector
            death_vector = np.random.binomial(
                n=1,
                p=(1 / params.mean_human_age) * params.delta_time,
                size=size_population,
            )
            current_ages[
                np.logical_or(death_vector == 1, current_ages >= params.max_human_age)
            ] = 0
        return current_ages

    def to_stats(self) -> StateStats:
        return StateStats(
            percent_compliant=np.sum(self.people.compliance)
            / len(self.people.compliance),
            percent_male=np.sum(self.people.sex_is_male) / len(self.people.compliance),
            L1=NumericArrayStat.from_array(self.people.blackfly.L1),
            L2=NumericArrayStat.from_array(self.people.blackfly.L2),
            L3=NumericArrayStat.from_array(self.people.blackfly.L3),
            ages=NumericArrayStat.from_array(self.people.ages),
            mf=NumericArrayStat.from_array(self.people.mf),
            male_worms=NumericArrayStat.from_array(self.people.male_worms),
            infertile_female_worms=NumericArrayStat.from_array(
                self.people.infertile_female_worms
            ),
            fertile_female_worms=NumericArrayStat.from_array(
                self.people.fertile_female_worms
            ),
            mf_per_skin_snip=self.microfilariae_per_skin_snip()[0],
            population_prevalence=self.mf_prevalence_in_population(),
        )


def calc_coverage(
    people: People,
    # percent_non_compliant: float,
    measured_coverage: float,
    age_compliance: float,
) -> NDArray[np.bool_]:

    non_compliant_people = np.logical_or(
        people.ages < age_compliance, np.logical_not(people.compliance)
    )
    non_compliant_percentage = np.sum(non_compliant_people) / len(non_compliant_people)
    compliant_percentage = 1 - non_compliant_percentage
    coverage = measured_coverage / compliant_percentage  # TODO: Is this correct?
    out_coverage = np.repeat(coverage, len(people))
    out_coverage[non_compliant_people] = 0
    rand_nums = np.random.uniform(low=0, high=1, size=len(people))
    return rand_nums < out_coverage


def weibull_mortality(
    delta_time: float, mu1: float, mu2: float, age_categories: np.ndarray
) -> NDArray[np.float_]:
    return delta_time * (mu1**mu2) * mu2 * (age_categories ** (mu2 - 1))


def calculate_total_exposure(
    params: Params, people: People, individual_exposure: NDArray[np.float_]
) -> NDArray[np.float_]:
    male_exposure_assumed = params.male_exposure * np.exp(
        -params.male_exposure_exponent * people.ages
    )
    male_exposure_assumed_of_males = male_exposure_assumed[people.sex_is_male]
    if len(male_exposure_assumed_of_males) == 0:
        # TODO: Is this correct?
        mean_male_exposure = 0
    else:
        mean_male_exposure: float = np.mean(male_exposure_assumed_of_males)
    female_exposure_assumed = params.female_exposure * np.exp(
        -params.female_exposure_exponent * people.ages
    )
    female_exposure_assumed_of_females = female_exposure_assumed[
        np.logical_not(people.sex_is_male)
    ]
    if len(female_exposure_assumed_of_females) == 0:
        # TODO: Is this correct?
        mean_female_exposure = 0
    else:
        mean_female_exposure: float = np.mean(female_exposure_assumed_of_females)

    sex_age_exposure = np.where(
        people.sex_is_male,
        male_exposure_assumed / mean_male_exposure,
        female_exposure_assumed / mean_female_exposure,
    )

    total_exposure = sex_age_exposure * individual_exposure
    total_exposure = total_exposure / np.mean(total_exposure)
    return total_exposure


def delta_h(
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


def w_plus_one_rate(
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
    dh = delta_h(params, L3, total_exposure)
    return (
        params.delta_time
        * params.annual_transm_potential
        * params.bite_rate_per_fly_on_human
        * dh
        * total_exposure
        * L3
    )


def get_last_males_and_females(
    worm_delay: NDArray[np.int_], params: Params
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    final_column = np.array(worm_delay[-1], dtype=int)
    assert len(final_column) == params.human_population
    last_males = np.random.binomial(
        n=final_column, p=0.5, size=len(final_column)
    )  # new.worms.m
    last_females = final_column - last_males  # new.worms.nf
    return last_males, last_females


def calc_dead_and_lost_worms(
    params: Params, current_worms: NDArray[np.int_], mortalities: NDArray[np.float_]
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    dead_worms = np.random.binomial(
        n=current_worms,
        p=mortalities,
        size=params.human_population,
    )
    lost_worms = np.random.binomial(
        n=current_worms - dead_worms,
        p=np.repeat(params.delta_time / params.worms_aging, params.human_population),
        size=params.human_population,
    )
    return dead_worms, lost_worms


def calc_new_worms_from_inside(
    current_worms: NDArray[np.int_],
    dead_worms: NDArray[np.int_],
    lost_worms: NDArray[np.int_],
    human_population: int,
    prob: NDArray[np.float_],
) -> NDArray[np.int_]:
    delta_fertile_female_worms = current_worms - dead_worms - lost_worms  # trans.fc
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
    state: State,
    last_females: NDArray[np.int_],
    last_males: NDArray[np.int_],
    worm_mortality_rate: NDArray[np.float_],
    coverage_in: Optional[NDArray[np.bool_]],
    last_lost_worms: WormGroup,
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
    dead_male_worms, lost_male_worms = calc_dead_and_lost_worms(
        params=params,
        current_worms=current_male_worms,
        mortalities=compartment_mortality,
    )
    if compartment == 0:
        total_male_worms = (
            current_male_worms + last_males - lost_male_worms - dead_male_worms
        )
    else:
        total_male_worms = (
            current_male_worms
            + last_lost_worms.male
            - lost_male_worms
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
                current_time < initial_treatment_times,
                initial_treatment_times <= current_time + params.delta_time,
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
    dead_infertile_worms, lost_infertile_worms = calc_dead_and_lost_worms(
        params=params,
        current_worms=current_female_worms_infertile,
        mortalities=female_mortalities,
    )
    dead_fertile_worms, lost_fertile_worms = calc_dead_and_lost_worms(
        params=params,
        current_worms=current_female_worms_fertile,
        mortalities=female_mortalities,
    )

    new_worms_infertile_from_inside = calc_new_worms_from_inside(
        current_worms=current_female_worms_fertile,
        dead_worms=dead_fertile_worms,
        lost_worms=lost_fertile_worms,
        human_population=params.human_population,
        prob=lambda_zero_in,
    )  # new.worms.nf.fi

    # females worms from infertile to fertile, this happens independent of males, but production of mf depends on males

    # individuals which still have non fertile worms in an age compartment after death and aging

    new_worms_fertile_from_inside = calc_new_worms_from_inside(
        current_worms=current_female_worms_infertile,
        dead_worms=dead_infertile_worms,
        lost_worms=lost_infertile_worms,
        human_population=params.human_population,
        prob=omega,
    )  # new.worms.f.fi TODO: Are these the right way round?

    if compartment == 0:
        infertile_out = (
            current_female_worms_infertile
            + last_females
            + new_worms_infertile_from_inside
            - lost_infertile_worms
            - dead_infertile_worms
            - new_worms_fertile_from_inside
        )  # nf.out
        fertile_out = (
            current_female_worms_fertile
            + new_worms_fertile_from_inside
            - lost_fertile_worms
            - dead_fertile_worms
            - new_worms_infertile_from_inside
        )

    else:
        infertile_out = (
            current_female_worms_infertile
            + new_worms_infertile_from_inside
            - lost_infertile_worms
            - new_worms_fertile_from_inside
            + last_lost_worms.infertile
            - dead_infertile_worms
        )
        fertile_out = (
            current_female_worms_fertile
            + new_worms_fertile_from_inside
            - lost_fertile_worms
            - dead_fertile_worms
            - new_worms_infertile_from_inside
            + last_lost_worms.fertile
        )
    new_lost_worms = WormGroup(
        male=lost_male_worms, infertile=lost_infertile_worms, fertile=lost_fertile_worms
    )
    new_total_worms = WormGroup(
        male=total_male_worms, infertile=infertile_out, fertile=fertile_out
    )
    return (
        new_total_worms,
        new_lost_worms,
        time_of_last_treatment,
    )


def construct_derive_microfil_one(
    fertile_worms: NDArray[np.int_],
    microfil: NDArray[np.int_],
    fecundity_rates_worms: NDArray[np.float_],
    mortality: NDArray[np.float_],
    params: Params,
    person_has_worms: NDArray[np.bool_],
) -> Callable[[Union[float, NDArray[np.float_]]], NDArray[np.float_]]:
    """
    #function called during RK4 for first age class of microfilariae

    fertile_worms # fert.worms
    microfil #mf.in
    fecundity_rates_worms # ep.in
    mortality # mf.mort
    params.microfil_move_rate #mf.move
    person_has_worms # mp (once turned to 0 or 1)
    """
    new_in = np.einsum(
        "ij, i -> j", fertile_worms, fecundity_rates_worms
    )  # TODO: Check?

    def derive_microfil_one(k: Union[float, NDArray[np.float_]]) -> NDArray[np.float_]:
        mortality_temp = mortality * (microfil + k)
        assert np.sum(mortality_temp < 0) == 0
        move_rate_temp = params.microfil_move_rate * (microfil + k)
        assert np.sum(move_rate_temp < 0) == 0
        mortality_temp[mortality_temp < 0] = 0
        move_rate_temp[move_rate_temp < 0] = 0
        return person_has_worms * new_in - mortality_temp - move_rate_temp

    return derive_microfil_one


def construct_derive_microfil_rest(
    microfil: NDArray[np.int_],
    mortality: NDArray[np.float_],
    params: Params,
    microfil_compartment_minus_one: NDArray[np.int_],
) -> Callable[[Union[float, NDArray[np.float_]]], NDArray[np.float_]]:
    """
    #function called during RK4 for age classes of microfilariae > 1

    microfil #mf.in
    mortality # mf.mort
    params.microfil_move_rate #mf.move
    microfil_compartment_minus_one # mf.comp.minus.one
    """
    movement_last = microfil_compartment_minus_one * params.microfil_move_rate

    def derive_microfil_rest(k: Union[float, NDArray[np.float_]]) -> NDArray[np.float_]:
        mortality_temp = mortality * (microfil + k)
        assert np.sum(mortality_temp < 0) == 0
        move_rate_temp = params.microfil_move_rate * (microfil + k)
        assert np.sum(move_rate_temp < 0) == 0
        mortality_temp[mortality_temp < 0] = 0
        move_rate_temp[move_rate_temp < 0] = 0
        return movement_last - mortality_temp - move_rate_temp

    return derive_microfil_rest


def change_in_microfil(
    state: State,
    params: Params,
    microfillarie_mortality_rate: NDArray[np.float_],
    fecundity_rates_worms: NDArray[np.float_],
    time_of_last_treatment: Optional[NDArray[np.float_]],
    compartment: int,
    current_time: float,
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
    state is dat
    """
    compartment_mortality = np.repeat(  # mf.mu
        microfillarie_mortality_rate[compartment], params.human_population
    )
    fertile_worms = state.people.fertile_female_worms  # fert.worms
    microfil: NDArray[np.int_] = state.people.mf[compartment]

    # increases microfilarial mortality if treatment has started
    if (
        time_of_last_treatment is not None
        and current_time >= params.treatment_start_time
    ):
        compartment_mortality_prime = (
            time_of_last_treatment + params.u_ivermectin
        ) ** (
            -params.shape_parameter_ivermectin
        )  # additional mortality due to ivermectin treatment
        compartment_mortality_prime = np.nan_to_num(compartment_mortality_prime)
        compartment_mortality += compartment_mortality_prime

    if compartment == 0:
        person_has_worms = np.sum(state.people.male_worms, axis=0) > 0
        derive_microfil = construct_derive_microfil_one(
            fertile_worms,
            microfil,
            fecundity_rates_worms,
            compartment_mortality,
            params,
            person_has_worms,
        )
    else:
        derive_microfil = construct_derive_microfil_rest(
            microfil, compartment_mortality, params, state.people.mf[compartment - 1]
        )
    k1 = derive_microfil(0.0)
    k2 = derive_microfil(params.delta_time * k1 / 2)
    k3 = derive_microfil(params.delta_time * k2 / 2)
    k4 = derive_microfil(params.delta_time * k3)
    return microfil + (params.delta_time / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


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


def run_simulation(
    state: State, start_time: float = 0, end_time: float = 0, verbose: bool = False
) -> State:
    if end_time < start_time:
        raise ValueError("End time after start")

    current_time = start_time
    while current_time < end_time:
        if state.params.delta_time > current_time % 0.2 and verbose:
            print(current_time)
        current_time += state.params.delta_time
        if current_time >= state.params.treatment_start_time:
            coverage_in = calc_coverage(
                state.people,
                state.params.total_population_coverage,
                state.params.min_skinsnip_age,
            )
        else:
            coverage_in = None

        total_exposure = calculate_total_exposure(
            state.params, state.people, state.derived_params.individual_exposure
        )
        old_state = deepcopy(state)  # all.mats.cur
        # increase ages
        state.people.ages += state.params.delta_time

        people_to_die: NDArray[np.bool_] = np.logical_or(
            np.random.binomial(
                n=1,
                p=(1 / state.params.mean_human_age) * state.params.delta_time,
                size=state.params.human_population,
            )
            == 1,
            state.people.ages >= state.params.max_human_age,
        )

        # there is a delay in new parasites entering humans (from fly bites) and entering the first adult worm age class

        new_rate = w_plus_one_rate(
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
            new_worms = np.random.poisson(
                lam=new_rate, size=state.params.human_population
            )
        # TODO: Check calculation change
        # new_worms = np.random.poisson(lam=new_rate, size=state.params.human_population)
        # Take males and females from final column of worm_delay
        last_males, last_females = get_last_males_and_females(
            state.delay_arrays.worm_delay, state.params
        )
        # Move all columns in worm_delay along one
        state.delay_arrays.worm_delay = np.vstack(
            (new_worms, state.delay_arrays.worm_delay[:-1])
        )

        last_lost_worms = WormGroup.from_population(state.params.human_population)
        last_time_of_last_treatment = None
        for compartment in range(state.params.worm_age_stages):
            (
                last_total_worms,
                last_lost_worms,
                last_time_of_last_treatment,
            ) = change_in_worm_per_index(  # res
                params=state.params,
                state=state,
                last_females=last_females,
                last_males=last_males,
                worm_mortality_rate=state.derived_params.worm_mortality_rate,
                coverage_in=coverage_in,
                last_lost_worms=last_lost_worms,
                initial_treatment_times=state.derived_params.initial_treatment_times,
                current_time=current_time,
                compartment=compartment,
                time_of_last_treatment=state.people.time_of_last_treatment,
            )
            if np.any(
                np.logical_or(
                    np.logical_or(
                        last_total_worms.male < 0, last_total_worms.fertile < 0
                    ),
                    last_total_worms.infertile < 0,
                )
            ):
                candidate_people_male_worms = last_total_worms.male[
                    last_total_worms.male < 0
                ]
                candidate_people_fertile_worms = last_total_worms.fertile[
                    last_total_worms.fertile < 0
                ]
                candidate_people_infertile_worms = last_total_worms.infertile[
                    last_total_worms.infertile < 0
                ]

                raise RuntimeError(
                    f"Worms became negative: \nMales: {candidate_people_male_worms} \nFertile Females: {candidate_people_fertile_worms} \nInfertile Females: {candidate_people_infertile_worms}"
                )

            state.people.male_worms[compartment] = last_total_worms.male
            state.people.infertile_female_worms[
                compartment
            ] = last_total_worms.infertile
            state.people.fertile_female_worms[compartment] = last_total_worms.fertile

        assert last_time_of_last_treatment is not None
        if (
            state.derived_params.initial_treatment_times is not None
            and current_time >= state.params.treatment_start_time
        ):
            state.people.time_of_last_treatment = last_time_of_last_treatment

        for compartment in range(state.params.microfil_age_stages):
            state.people.mf[compartment] = change_in_microfil(
                state=old_state,
                params=state.params,
                microfillarie_mortality_rate=state.derived_params.microfillarie_mortality_rate,
                fecundity_rates_worms=state.derived_params.fecundity_rates_worms,
                time_of_last_treatment=state.people.time_of_last_treatment,
                compartment=compartment,
                current_time=current_time,
            )

        # inputs for delay in L1
        new_mf = np.sum(
            old_state.people.mf, axis=0
        )  # TODO: Should this be old state? mf.temp

        state.people.blackfly.L1 = calc_l1(
            state.params,
            new_mf,
            state.delay_arrays.mf_delay[-1],
            total_exposure,
            state.delay_arrays.exposure_delay[-1],
        )
        state.people.blackfly.L2 = calc_l2(
            state.params,
            state.delay_arrays.l1_delay,
            state.delay_arrays.mf_delay[-1],
            state.delay_arrays.exposure_delay[-1],
        )
        state.people.blackfly.L3 = calc_l3(state.params, old_state.people.blackfly.L2)

        state.delay_arrays.exposure_delay = np.vstack(
            (total_exposure, state.delay_arrays.exposure_delay[:-1])
        )
        state.delay_arrays.mf_delay = np.vstack(
            (new_mf, state.delay_arrays.mf_delay[:-1])
        )
        state.delay_arrays.l1_delay = state.people.blackfly.L1

        total_people_to_die: int = np.sum(people_to_die)
        if total_people_to_die > 0:
            state.delay_arrays.worm_delay[:, people_to_die] = 0
            state.delay_arrays.mf_delay[0, people_to_die] = 0
            state.delay_arrays.l1_delay[people_to_die] = 0
            state.people.time_of_last_treatment[people_to_die] = np.nan

            state.people.sex_is_male[people_to_die] = (
                np.random.uniform(low=0, high=1, size=total_people_to_die) < 0.5
            )  # TODO: Make adjustable
            state.people.ages[people_to_die] = 0
            state.people.blackfly.L1[people_to_die] = 0
            state.people.mf[:, people_to_die] = 0
            state.people.male_worms[:, people_to_die] = 0
            state.people.fertile_female_worms[:, people_to_die] = 0
            state.people.infertile_female_worms[:, people_to_die] = 0
    return state
