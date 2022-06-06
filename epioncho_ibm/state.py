from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from .params import Params


class RandomConfig(BaseModel):
    gender_ratio: float = 0.5
    noncompliant_percentage: float = 0.05


@dataclass
class BlackflyLarvae:
    L1: NDArray[np.float_]  # 4: L1
    L2: NDArray[np.float_]  # 5: L2
    L3: NDArray[np.float_]  # 6: L3


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
    mf_current_quantity: NDArray[np.int_]
    exposure: NDArray[np.float_]
    new_worm_rate: NDArray[np.float_]

    def __len__(self):
        return len(self.compliance)


class State:
    current_iteration: int = 0
    people: People
    params: Params

    def __init__(self, people: People, params: Params) -> None:
        if len(people) != params.human_population:
            raise ValueError(
                f"People length ({len(people)}) inconsistent with params ({params.human_population})"
            )
        self.people = people
        self.params = params

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
                mf=np.zeros((params.microfil_age_stages, n_people)),
                male_worms=np.zeros((params.worm_age_stages, n_people), dtype=int),
                infertile_female_worms=np.zeros(
                    (params.worm_age_stages, n_people), dtype=int
                ),
                fertile_female_worms=np.zeros(
                    (params.worm_age_stages, n_people), dtype=int
                ),
                mf_current_quantity=np.zeros(n_people, dtype=int),
                exposure=np.zeros(n_people),
                new_worm_rate=np.zeros(n_people),
            ),
            params=params,
        )

    def prevelence(self: "State") -> float:
        raise NotImplementedError

    def microfilariae_per_skin_snip(self: "State") -> float:
        raise NotImplementedError

    def mf_prevalence_in_population(self: "State", min_age_skinsnip: int) -> float:
        """
        Returns a decimal representation of mf prevalence in skinsnip aged population.
        """
        pop_over_min_age_array = self.people.ages >= min_age_skinsnip
        pop_over_min_age = np.sum(pop_over_min_age_array)
        infected_over_min_age = np.sum(
            np.logical_and(pop_over_min_age_array, self.people.mf_current_quantity > 0)
        )
        return pop_over_min_age / infected_over_min_age

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
        for i in range(num_iter):
            current_ages += delta_time_vector
            death_vector = np.random.binomial(
                n=1,
                p=(1 / params.mean_human_age) * params.delta_time,
                size=size_population,
            )
            np.place(
                current_ages,
                np.logical_or(death_vector == 1, current_ages >= params.max_human_age),
                0,
            )
        return current_ages


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
    np.place(arr=out_coverage, mask=non_compliant_people, vals=0)
    rand_nums = np.random.uniform(low=0, high=1, size=len(people))
    final_coverage = rand_nums < out_coverage
    return final_coverage


def weibull_mortality(
    delta_time: float, mu1: float, mu2: float, age_categories: np.ndarray
) -> np.ndarray:
    return delta_time * (mu1**mu2) * mu2 * (age_categories ** (mu2 - 1))


def initialise_simulation(params: Params):
    give_treatment = True

    worm_max_age = 20
    microfillarie_max_age = 2.5
    number_of_microfillariae_age_categories = 20
    number_of_worm_age_categories = 20
    individual_exposure = (
        np.random.gamma(  # individual level exposure to fly bites "ex.vec"
            shape=params.gamma_distribution,
            scale=params.gamma_distribution,
            size=params.human_population,
        )
    )
    individual_exposure = individual_exposure / np.mean(
        individual_exposure
    )  # normalise
    if give_treatment:
        initial_treatment_times = np.arange(  # "times.of.treat.in"
            start=params.treatment_start_time,
            stop=(
                params.treatment_stop_time
                - params.treatment_interval / params.delta_time
            ),
        )
    else:
        initial_treatment_times = None

    # Cols to zero is a mechanism for resetting certain attributes to zero
    # columns_to_zero = np.arange( start = 1, stop = )

    # age-dependent mortality and fecundity rates of parasite life stages
    worm_age_categories = np.arange(
        start=0, stop=worm_max_age, step=worm_max_age / number_of_worm_age_categories
    )  # age.cats
    worm_mortality_rate = weibull_mortality(
        params.delta_time, params.mu_worms1, params.mu_worms2, worm_age_categories
    )
    print(worm_mortality_rate)
    fecundity_rates_worms = (
        1.158305
        * params.fecundity_worms_1
        / (
            params.fecundity_worms_1
            + (params.fecundity_worms_2 ** (-worm_age_categories))
            - 1
        )
    )

    microfillarie_age_categories = np.arange(
        start=0,
        stop=microfillarie_max_age,
        step=microfillarie_max_age / number_of_microfillariae_age_categories,
    )  # age.cats.mf

    microfillarie_mortality_rate = weibull_mortality(
        params.delta_time,
        params.mu_microfillarie1,
        params.mu_microfillarie2,
        microfillarie_age_categories,
    )
    # matrix for delay in L3 establishment in humans
    # RE-EVALUATE THIS SECTION
    # number_of_delay_cols = int(params.l3_delay * (28 / (delta_time*365)))
    # l_extras = np.zeros((number_of_delay_cols, params.human_population))
    # indices_l_mat = np.arange(2, number_of_delay_cols)

    # SET initial values in state
    number_of_delay_cols = round(params.l3_delay * 28 / (params.delta_time * 365))
    l_extras = np.zeros((number_of_delay_cols, params.human_population))

    return individual_exposure, l_extras, worm_mortality_rate, initial_treatment_times


def calculate_total_exposure(
    params: Params, people: People, individual_exposure: NDArray[np.float_]
) -> NDArray[np.float_]:
    male_exposure_assumed = params.male_exposure * np.exp(
        -params.male_exposure_exponent * people.ages
    )
    mean_male_exposure: float = np.mean(
        np.extract(people.sex_is_male, male_exposure_assumed)
    )
    female_exposure_assumed = params.female_exposure * np.exp(
        -params.female_exposure_exponent * people.ages
    )
    mean_female_exposure: float = np.mean(
        np.extract(np.logical_not(people.sex_is_male), female_exposure_assumed)
    )

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
    l_extras: NDArray[np.int_], params: Params
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    final_column = np.array(l_extras[-1], dtype=int)
    assert len(final_column) == params.human_population
    last_males = np.random.binomial(
        n=final_column, p=0.5, size=len(final_column)
    )  # new.worms.m
    last_females = final_column - last_males  # new.worms.nf
    return last_males, last_females


def change_in_worm_per_index(
    params: Params,
    state: State,
    L3: float,
    last_females: NDArray[np.float_],
    last_males: NDArray[np.float_],
    total_exposure: NDArray[np.float_],
    worm_mortality_rate: NDArray[np.float_],
    coverage_in: Optional[NDArray[np.bool_]],
    last_change: NDArray[np.float_],
    initial_treatment_times: Optional[NDArray[np.float_]],
    current_time: float,
    compartment: int,
) -> NDArray[np.float_]:
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
    coverage_in "onchosim.cov"
    last_change "w.f.l.c"
    permanent_infertility "cum.infer"
    worms.start/ws used to refer to start point in giant array for worms
    initial_treatment_times "times.of.treat.in"
    iteration/i now means current_time
    if initial_treatment_times is None give.treat is false etc
    N is params.human_population
    params.worms_aging "time.each.comp"
    """
    lambda_zero_in = np.repeat(
        params.lambda_zero * params.delta_time, params.human_population
    )  # loss of fertility
    omega = np.repeat(
        params.omega * params.delta_time, params.human_population
    )  # becoming fertile
    # male worms
    current_male_worms = state.people.male_worms[compartment]  # cur.Wm
    compartment_mortality = np.repeat(
        worm_mortality_rate[compartment], params.human_population
    )
    dead_male_worms = np.random.binomial(
        n=current_male_worms,
        p=compartment_mortality,
        size=params.human_population,
    )
    lost_male_worms = np.random.binomial(
        n=current_male_worms - dead_male_worms,
        p=np.repeat(params.delta_time / params.worms_aging, params.human_population),
        size=params.human_population,
    )
    if compartment == 0:
        # why are last males the only one not over all population?
        total_male_worms = (
            current_male_worms + last_males + lost_male_worms - dead_male_worms
        )
    else:
        total_male_worms = (
            current_male_worms + last_change - lost_male_worms - dead_male_worms
        )  # TODO: check

    # female worms

    return last_change


def run_simulation(state: State, start_time: float = 0, end_time: float = 0):

    if end_time < start_time:
        raise ValueError("End time after start")

    (
        individual_exposure,
        l_extras,
        worm_mortality_rate,
        initial_treatment_times,
    ) = initialise_simulation(state.params)
    treatment_vector_in = np.repeat(None, state.params.human_population)  # type:ignore
    # TODO: Move l_extras to state  - potentially move constants to params
    current_time = start_time
    while current_time < end_time:
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
            state.params, state.people, individual_exposure
        )

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

        L3_in = np.mean(state.people.blackfly.L3)
        new_rate = w_plus_one_rate(state.params, L3_in, total_exposure)
        new_worms = np.random.poisson(lam=new_rate, size=state.params.human_population)

        # Take males and females from final column of l_extras
        last_males, last_females = get_last_males_and_females(l_extras, state.params)
        # Move all columns in l_extras along one
        l_extras = np.vstack((new_worms, l_extras[:-1]))

        last_change = np.zeros(state.params.human_population)
        for compartment in range(state.params.worm_age_stages - 1):

            result = change_in_worm_per_index(
                state.params,
                state,
                L3_in,
                last_females,
                last_males,
                total_exposure,
                worm_mortality_rate,
                coverage_in,
                last_change,
                initial_treatment_times,
                current_time,
                compartment,
            )
            last_change = result  # assign output to use at next iteration, indexes 2, 5, 6 (worms moving through compartments)
