import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from epioncho_ibm.blackfly import calc_l1, calc_l2, calc_l3
from epioncho_ibm.microfil import change_in_microfil
from epioncho_ibm.worms import (
    WormGroup,
    calc_new_worms,
    change_in_worm_per_index,
    check_no_worms_are_negative,
    get_delayed_males_and_females,
)

from .derived_params import DerivedParams
from .params import ExposureParams, Params

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

    time_of_last_treatment: NDArray[np.float_]  # treat.vec

    def __len__(self):
        return len(self.compliance)


class DelayArrays:
    worm_delay: NDArray[np.int_]
    exposure_delay: NDArray[np.float_]
    mf_delay: NDArray[np.int_]

    def __init__(self, params: Params, individual_exposure) -> None:
        number_of_worm_delay_cols = math.ceil(
            params.blackfly.l3_delay * 28 / (params.delta_time * 365)
        )
        self.worm_delay = np.zeros(
            (number_of_worm_delay_cols, params.humans.human_population), dtype=int
        )
        # matrix for exposure (to fly bites) for L1 delay
        number_of_exposure_columns = math.ceil(4 / (params.delta_time * 365))

        self.exposure_delay = np.tile(
            individual_exposure, (number_of_exposure_columns, 1)
        )  # exposure.delay

        # matrix for tracking mf for L1 delay
        number_of_mf_columns = math.ceil(4 / (params.delta_time * 365))
        self.mf_delay = (
            np.ones((number_of_mf_columns, params.humans.human_population), dtype=int)
            * params.microfil.initial_mf
        )  # mf.delay
        # L1 delay in flies
        self.l1_delay = np.repeat(
            params.blackfly.initial_L1, params.humans.human_population
        )


def _calc_coverage(
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


def _calculate_total_exposure(
    exposure_params: ExposureParams,
    people: People,
    individual_exposure: NDArray[np.float_],
) -> NDArray[np.float_]:
    male_exposure_assumed = exposure_params.male_exposure * np.exp(
        -exposure_params.male_exposure_exponent * people.ages
    )
    male_exposure_assumed_of_males = male_exposure_assumed[people.sex_is_male]
    if len(male_exposure_assumed_of_males) == 0:
        # TODO: Is this correct?
        mean_male_exposure = 0
    else:
        mean_male_exposure: float = np.mean(male_exposure_assumed_of_males)
    female_exposure_assumed = exposure_params.female_exposure * np.exp(
        -exposure_params.female_exposure_exponent * people.ages
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
    return total_exposure / np.mean(total_exposure)


def _shift_delay_array(new_first_column, delay_array):
    return np.vstack((new_first_column, delay_array[:-1]))


class State:
    current_iteration: int = 0
    _people: People
    _params: Params
    _derived_params: DerivedParams
    _delay_arrays: DelayArrays

    def __init__(self, params: Params) -> None:
        n_people = params.humans.human_population
        sex_array = (
            np.random.uniform(low=0, high=1, size=n_people) < params.humans.gender_ratio
        )
        np.zeros(n_people)
        compliance_array = (
            np.random.uniform(low=0, high=1, size=n_people)
            > params.humans.noncompliant_percentage
        )
        time_of_last_treatment = np.empty(n_people)
        time_of_last_treatment[:] = np.nan
        self._people = People(
            compliance=compliance_array,
            ages=np.zeros(n_people),
            sex_is_male=sex_array,
            blackfly=BlackflyLarvae(
                L1=np.repeat(params.blackfly.initial_L1, n_people),
                L2=np.repeat(params.blackfly.initial_L2, n_people),
                L3=np.repeat(params.blackfly.initial_L3, n_people),
            ),
            mf=np.ones((params.microfil.microfil_age_stages, n_people))
            * params.microfil.initial_mf,
            male_worms=np.ones((params.worms.worm_age_stages, n_people), dtype=int)
            * params.worms.initial_worms,
            infertile_female_worms=np.ones(
                (params.worms.worm_age_stages, n_people), dtype=int
            )
            * params.worms.initial_worms,
            fertile_female_worms=np.ones(
                (params.worms.worm_age_stages, n_people), dtype=int
            )
            * params.worms.initial_worms,
            time_of_last_treatment=time_of_last_treatment,
        )
        self.params = params
        self._derived_params = DerivedParams(self.params)
        self._delay_arrays = DelayArrays(
            self.params, self._derived_params.individual_exposure
        )

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._derived_params = DerivedParams(value)
        self._delay_arrays = DelayArrays(
            value, self._derived_params.individual_exposure
        )
        self._params = value

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
        kmf = (
            self.params.microfil.slope_kmf
            * np.sum(
                self._people.fertile_female_worms + self._people.infertile_female_worms,
                axis=0,
            )
            + self.params.microfil.initial_kmf
        )

        mu = self.params.humans.skin_snip_weight * np.sum(self._people.mf, axis=0)
        if self.params.humans.skin_snip_number > 1:
            total_skin_snip_mf = np.zeros(
                (
                    self.params.humans.human_population,
                    self.params.humans.skin_snip_number,
                )
            )
            for i in range(self.params.humans.skin_snip_number):
                total_skin_snip_mf[:, i] = negative_binomial_alt_interface(n=kmf, mu=mu)
            mfobs = np.sum(total_skin_snip_mf, axis=1)
        else:
            mfobs = negative_binomial_alt_interface(n=kmf, mu=mu)
        mfobs = mfobs / (
            self.params.humans.skin_snip_number * self.params.humans.skin_snip_weight
        )
        return np.mean(mfobs), mfobs

    def mf_prevalence_in_population(self: "State") -> float:
        """
        Returns a decimal representation of mf prevalence in skinsnip aged population.
        """
        pop_over_min_age_array = (
            self._people.ages >= self.params.humans.min_skinsnip_age
        )
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

        current_ages = self._people.ages
        size_population = len(self._people)
        delta_time_vector = np.ones(size_population) * params.delta_time
        for _ in range(num_iter):
            current_ages += delta_time_vector
            death_vector = np.random.binomial(
                n=1,
                p=(1 / params.humans.mean_human_age) * params.delta_time,
                size=size_population,
            )
            current_ages[
                np.logical_or(
                    death_vector == 1, current_ages >= params.humans.max_human_age
                )
            ] = 0
        return current_ages

    def to_stats(self) -> StateStats:
        return StateStats(
            percent_compliant=np.sum(self._people.compliance)
            / len(self._people.compliance),
            percent_male=np.sum(self._people.sex_is_male)
            / len(self._people.compliance),
            L1=NumericArrayStat.from_array(self._people.blackfly.L1),
            L2=NumericArrayStat.from_array(self._people.blackfly.L2),
            L3=NumericArrayStat.from_array(self._people.blackfly.L3),
            ages=NumericArrayStat.from_array(self._people.ages),
            mf=NumericArrayStat.from_array(self._people.mf),
            male_worms=NumericArrayStat.from_array(self._people.male_worms),
            infertile_female_worms=NumericArrayStat.from_array(
                self._people.infertile_female_worms
            ),
            fertile_female_worms=NumericArrayStat.from_array(
                self._people.fertile_female_worms
            ),
            mf_per_skin_snip=self.microfilariae_per_skin_snip()[0],
            population_prevalence=self.mf_prevalence_in_population(),
        )

    def run_simulation(
        self: "State", start_time: float = 0, end_time: float = 0, verbose: bool = False
    ) -> None:
        if end_time < start_time:
            raise ValueError("End time after start")

        current_time = start_time
        while current_time < end_time:
            if self.params.delta_time > current_time % 0.2 and verbose:
                print(current_time)

            if (
                self.params.treatment is not None
                and current_time >= self.params.treatment.start_time
            ):
                coverage_in = _calc_coverage(
                    self._people,
                    self.params.humans.total_population_coverage,
                    self.params.humans.min_skinsnip_age,
                )
            else:
                coverage_in = None

            total_exposure = _calculate_total_exposure(
                self.params.exposure,
                self._people,
                self._derived_params.individual_exposure,
            )
            old_state = deepcopy(self)  # all.mats.cur
            # increase ages
            self._people.ages += self.params.delta_time

            people_to_die: NDArray[np.bool_] = np.logical_or(
                np.random.binomial(
                    n=1,
                    p=(1 / self.params.humans.mean_human_age) * self.params.delta_time,
                    size=self.params.humans.human_population,
                )
                == 1,
                self._people.ages >= self.params.humans.max_human_age,
            )

            # there is a delay in new parasites entering humans (from fly bites) and entering the first adult worm age class
            new_worms = calc_new_worms(
                self._people.blackfly.L3, self.params, total_exposure
            )
            # Take males and females from final column of worm_delay
            delayed_males, delayed_females = get_delayed_males_and_females(
                self._delay_arrays.worm_delay, self.params
            )
            # Move all columns in worm_delay along one
            self._delay_arrays.worm_delay = _shift_delay_array(
                new_worms, self._delay_arrays.worm_delay
            )

            last_aging_worms = WormGroup.from_population(
                self.params.humans.human_population
            )
            last_time_of_last_treatment = None
            for compartment in range(self.params.worms.worm_age_stages):
                (
                    last_total_worms,
                    last_aging_worms,
                    last_time_of_last_treatment,
                ) = change_in_worm_per_index(  # res
                    params=self.params,
                    people=self._people,
                    delayed_females=delayed_females,
                    delayed_males=delayed_males,
                    worm_mortality_rate=self._derived_params.worm_mortality_rate,
                    coverage_in=coverage_in,
                    last_aging_worms=last_aging_worms,
                    initial_treatment_times=self._derived_params.initial_treatment_times,
                    current_time=current_time,
                    compartment=compartment,
                    time_of_last_treatment=self._people.time_of_last_treatment,
                )
                check_no_worms_are_negative(last_total_worms)

                self._people.male_worms[compartment] = last_total_worms.male
                self._people.infertile_female_worms[
                    compartment
                ] = last_total_worms.infertile
                self._people.fertile_female_worms[
                    compartment
                ] = last_total_worms.fertile

            assert last_time_of_last_treatment is not None
            if (
                self.params.treatment is not None
                and current_time >= self.params.treatment.start_time
            ):
                self._people.time_of_last_treatment = last_time_of_last_treatment

            for compartment in range(self.params.microfil.microfil_age_stages):
                self._people.mf[compartment] = change_in_microfil(
                    people=old_state._people,
                    params=self.params,
                    microfillarie_mortality_rate=self._derived_params.microfillarie_mortality_rate,
                    fecundity_rates_worms=self._derived_params.fecundity_rates_worms,
                    time_of_last_treatment=self._people.time_of_last_treatment,
                    compartment=compartment,
                    current_time=current_time,
                )

            # inputs for delay in L1
            new_mf = np.sum(
                old_state._people.mf, axis=0
            )  # TODO: Should this be old state? mf.temp

            self._people.blackfly.L1 = calc_l1(
                self.params.blackfly,
                new_mf,
                self._delay_arrays.mf_delay[-1],
                total_exposure,
                self._delay_arrays.exposure_delay[-1],
            )
            self._people.blackfly.L2 = calc_l2(
                self.params.blackfly,
                self._delay_arrays.l1_delay,
                self._delay_arrays.mf_delay[-1],
                self._delay_arrays.exposure_delay[-1],
            )
            self._people.blackfly.L3 = calc_l3(
                self.params.blackfly, old_state._people.blackfly.L2
            )

            self._delay_arrays.exposure_delay = _shift_delay_array(
                total_exposure, self._delay_arrays.exposure_delay
            )
            self._delay_arrays.mf_delay = _shift_delay_array(
                new_mf, self._delay_arrays.mf_delay
            )
            self._delay_arrays.l1_delay = self._people.blackfly.L1

            total_people_to_die: int = np.sum(people_to_die)
            if total_people_to_die > 0:
                self._delay_arrays.worm_delay[:, people_to_die] = 0
                self._delay_arrays.mf_delay[0, people_to_die] = 0
                self._delay_arrays.l1_delay[people_to_die] = 0
                self._people.time_of_last_treatment[people_to_die] = np.nan

                self._people.sex_is_male[people_to_die] = (
                    np.random.uniform(low=0, high=1, size=total_people_to_die) < 0.5
                )  # TODO: Make adjustable
                self._people.ages[people_to_die] = 0
                self._people.blackfly.L1[people_to_die] = 0
                self._people.mf[:, people_to_die] = 0
                self._people.male_worms[:, people_to_die] = 0
                self._people.fertile_female_worms[:, people_to_die] = 0
                self._people.infertile_female_worms[:, people_to_die] = 0
            current_time += self.params.delta_time
        return None
