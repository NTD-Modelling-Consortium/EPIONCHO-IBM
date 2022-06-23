import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from .derived_params import DerivedParams
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
