import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from tqdm import tqdm

from epioncho_ibm.blackfly import calc_l1, calc_l2, calc_l3
from epioncho_ibm.microfil import calculate_microfil_delta
from epioncho_ibm.utils import lag_array
from epioncho_ibm.worms import (
    WormGroup,
    calc_new_worms,
    change_in_worms,
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


def truncated_geometric(N: int, prob: float, maximum: float) -> NDArray[np.float_]:
    output = np.repeat(maximum + 1, N)
    while np.any(output > maximum):
        output[output > maximum] = np.random.geometric(
            p=prob, size=len(output[output > maximum])
        )
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
    def from_array(cls, array: NDArray[np.float_] | NDArray[np.int_]):
        return cls(mean=float(np.mean(array)))  # , st_dev=np.std(array))


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
    # 2D Array, (N, age stage): worm stages 29-92 (63) males(21), infertile females(21), fertile females(21)
    male_worms: NDArray[np.int_]
    infertile_female_worms: NDArray[np.int_]
    fertile_female_worms: NDArray[np.int_]

    time_of_last_treatment: NDArray[np.float_]  # treat.vec

    def __len__(self):
        return len(self.compliance)


class DelayArrays:
    worm_delay: NDArray[np.int_]
    exposure_delay: NDArray[np.float_]
    mf_delay: NDArray[np.int_]

    def __init__(self, params: Params, n_people: int, individual_exposure) -> None:
        number_of_worm_delay_cols = math.ceil(
            params.blackfly.l3_delay
            * params.month_length_days
            / (params.delta_time * params.year_length_days)
        )
        self.worm_delay = np.zeros((number_of_worm_delay_cols, n_people), dtype=int)
        # matrix for exposure (to fly bites) for L1 delay
        number_of_exposure_columns = math.ceil(
            params.blackfly.l1_delay / (params.delta_time * params.year_length_days)
        )

        self.exposure_delay = np.tile(
            individual_exposure, (number_of_exposure_columns, 1)
        )

        # matrix for tracking mf for L1 delay
        number_of_mf_columns = math.ceil(
            params.blackfly.l1_delay / (params.delta_time * params.year_length_days)
        )

        # mf.delay
        self.mf_delay = (
            np.ones((number_of_mf_columns, n_people), dtype=int)
            * params.microfil.initial_mf
        )
        # L1 delay in flies
        self.l1_delay = np.repeat(params.blackfly.initial_L1, n_people)


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
        mean_male_exposure: float = float(np.mean(male_exposure_assumed_of_males))
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
        mean_female_exposure: float = float(np.mean(female_exposure_assumed_of_females))

    sex_age_exposure = np.where(
        people.sex_is_male,
        male_exposure_assumed / mean_male_exposure,
        female_exposure_assumed / mean_female_exposure,
    )

    total_exposure = sex_age_exposure * individual_exposure
    return total_exposure / np.mean(total_exposure)


class State:
    _people: People
    _params: Params
    _derived_params: DerivedParams
    _delay_arrays: DelayArrays

    def __init__(self, params: Params, n_people: int) -> None:
        sex_array = (
            np.random.uniform(low=0, high=1, size=n_people) < params.humans.gender_ratio
        )
        compliance_array = (
            np.random.uniform(low=0, high=1, size=n_people)
            > params.humans.noncompliant_percentage
        )
        time_of_last_treatment = np.empty(n_people)
        time_of_last_treatment[:] = np.nan
        self._people = People(
            compliance=compliance_array,
            ages=truncated_geometric(
                N=n_people,
                prob=params.delta_time / params.humans.mean_human_age,
                maximum=params.humans.max_human_age / params.delta_time,
            )
            * params.delta_time,
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

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._derived_params = DerivedParams(value, self.n_people)
        self._delay_arrays = DelayArrays(
            value, self.n_people, self._derived_params.individual_exposure
        )
        self._params = value

    @property
    def n_people(self):
        return len(self._people)

    def microfilariae_per_skin_snip(self: "State") -> tuple[float, NDArray[np.float_]]:
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
                    self.n_people,
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
        return float(np.mean(mfobs)), mfobs

    def mf_prevalence_in_population(self: "State") -> float:
        """
        Returns a decimal representation of mf prevalence in skinsnip aged population.
        """
        pop_over_min_age_array = (
            self._people.ages >= self.params.humans.min_skinsnip_age
        )
        _, mf_skin_snip = self.microfilariae_per_skin_snip()
        infected_over_min_age: float = float(
            np.sum(mf_skin_snip[pop_over_min_age_array] > 0)
        )
        total_over_min_age = float(np.sum(pop_over_min_age_array))
        return infected_over_min_age / total_over_min_age

    def to_stats(self) -> StateStats:
        return StateStats(
            percent_compliant=float(np.sum(self._people.compliance))
            / len(self._people.compliance),
            percent_male=float(np.sum(self._people.sex_is_male))
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

    def _advance(self: "State", current_time: float):
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
        # increase ages
        self._people.ages += self.params.delta_time

        people_to_die: NDArray[np.bool_] = np.logical_or(
            np.random.binomial(
                n=1,
                p=(1 / self.params.humans.mean_human_age) * self.params.delta_time,
                size=self.n_people,
            )
            == 1,
            self._people.ages >= self.params.humans.max_human_age,
        )

        # there is a delay in new parasites entering humans (from fly bites) and entering the first adult worm age class
        new_worms = calc_new_worms(
            self._people.blackfly.L3,
            self.params.blackfly,
            self.params.delta_time,
            total_exposure,
            self.n_people,
        )
        # Take males and females from final column of worm_delay
        delayed_males, delayed_females = get_delayed_males_and_females(
            self._delay_arrays.worm_delay,
            self.n_people,
            self.params.worms.sex_ratio,
        )

        # Move all rows in worm_delay along one
        self._delay_arrays.worm_delay = lag_array(
            new_worms, self._delay_arrays.worm_delay
        )

        old_fertile_female_worms = self._people.fertile_female_worms.copy()
        old_male_worms = self._people.male_worms.copy()

        current_worms = WormGroup(
            male=self._people.male_worms,
            infertile=self._people.infertile_female_worms,
            fertile=self._people.fertile_female_worms,
        )

        (
            self._people.male_worms,
            self._people.infertile_female_worms,
            self._people.fertile_female_worms,
            last_time_of_last_treatment,
        ) = change_in_worms(
            current_worms=current_worms,
            worm_params=self.params.worms,
            treatment_params=self.params.treatment,
            delta_time=self.params.delta_time,
            n_people=self.n_people,
            delayed_females=delayed_females,
            delayed_males=delayed_males,
            mortalities=self._derived_params.worm_mortality_rate,
            coverage_in=coverage_in,
            initial_treatment_times=self._derived_params.initial_treatment_times,
            current_time=current_time,
            time_of_last_treatment=self._people.time_of_last_treatment,
        )

        assert last_time_of_last_treatment is not None

        if (
            self.params.treatment is not None
            and current_time >= self.params.treatment.start_time
        ):
            self._people.time_of_last_treatment = last_time_of_last_treatment

        # inputs for delay in L1
        # TODO: Should this be the existing mf? mf.temp
        old_mf = np.sum(self._people.mf, axis=0)

        self._people.mf += calculate_microfil_delta(
            stages=self.params.microfil.microfil_age_stages,
            exiting_microfil=self._people.mf,
            n_people=self.n_people,
            delta_time=self.params.delta_time,
            microfil_params=self.params.microfil,
            treatment_params=self.params.treatment,
            microfillarie_mortality_rate=self._derived_params.microfillarie_mortality_rate,
            fecundity_rates_worms=self._derived_params.fecundity_rates_worms,
            time_of_last_treatment=self._people.time_of_last_treatment,
            current_time=current_time,
            current_fertile_female_worms=old_fertile_female_worms,
            current_male_worms=old_male_worms,
        )

        self._people.blackfly.L1 = calc_l1(
            self.params.blackfly,
            old_mf,
            self._delay_arrays.mf_delay[-1],
            total_exposure,
            self._delay_arrays.exposure_delay[-1],
            self.params.year_length_days,
        )

        old_blackfly_L2 = self._people.blackfly.L2
        self._people.blackfly.L2 = calc_l2(
            self.params.blackfly,
            self._delay_arrays.l1_delay,
            self._delay_arrays.mf_delay[-1],
            self._delay_arrays.exposure_delay[-1],
            self.params.year_length_days,
        )
        self._people.blackfly.L3 = calc_l3(self.params.blackfly, old_blackfly_L2)

        self._delay_arrays.exposure_delay = lag_array(
            total_exposure, self._delay_arrays.exposure_delay
        )
        self._delay_arrays.mf_delay = lag_array(old_mf, self._delay_arrays.mf_delay)
        self._delay_arrays.l1_delay = self._people.blackfly.L1

        total_people_to_die: int = int(np.sum(people_to_die))
        if total_people_to_die > 0:
            self._delay_arrays.worm_delay[:, people_to_die] = 0
            self._delay_arrays.mf_delay[0, people_to_die] = 0
            self._delay_arrays.l1_delay[people_to_die] = 0
            self._people.time_of_last_treatment[people_to_die] = np.nan

            self._people.sex_is_male[people_to_die] = (
                np.random.uniform(low=0, high=1, size=total_people_to_die)
                < self.params.humans.gender_ratio
            )
            self._people.ages[people_to_die] = 0
            self._people.blackfly.L1[people_to_die] = 0
            self._people.mf[:, people_to_die] = 0
            self._people.male_worms[:, people_to_die] = 0
            self._people.fertile_female_worms[:, people_to_die] = 0
            self._people.infertile_female_worms[:, people_to_die] = 0

    def run_simulation(
        self: "State", start_time: float = 0, end_time: float = 0, verbose: bool = False
    ) -> None:
        if end_time < start_time:
            raise ValueError("End time after start")

        current_time = start_time
        # total progress bar must be a bit over so that the loop doesn't exceed total
        with tqdm(
            total=end_time - start_time + self.params.delta_time, disable=not verbose
        ) as progress_bar:
            while current_time < end_time:
                progress_bar.update(self.params.delta_time)
                self._advance(current_time=current_time)
                current_time += self.params.delta_time

    def run_simulation_output_stats(
        self: "State",
        sampling_interval: float,
        start_time: float = 0,
        end_time: float = 0,
        verbose: bool = False,
    ) -> list[tuple[float, StateStats]]:
        if end_time < start_time:
            raise ValueError("End time after start")

        current_time = start_time
        output_stats: list[tuple[float, StateStats]] = []
        while current_time < end_time:
            if self.params.delta_time > current_time % 0.2 and verbose:
                print(current_time)
            if self.params.delta_time > current_time % sampling_interval:
                output_stats.append((current_time, self.to_stats()))
            self._advance(current_time=current_time)
            current_time += self.params.delta_time
        return output_stats
