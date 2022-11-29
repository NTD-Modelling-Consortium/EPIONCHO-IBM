import math
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Generic, TypeVar

import h5py
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from tqdm import tqdm

from epioncho_ibm.blackfly import calc_l1, calc_l2, calc_l3
from epioncho_ibm.microfil import calculate_microfil_delta
from epioncho_ibm.utils import array_fully_equal, lag_array
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

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BlackflyLarvae)
            and array_fully_equal(self.L1, other.L1)
            and array_fully_equal(self.L2, other.L2)
            and array_fully_equal(self.L3, other.L3)
        )

    def append_to_hdf5_group(self, group: h5py.Group):
        group.create_dataset("L1", data=self.L1)
        group.create_dataset("L2", data=self.L2)
        group.create_dataset("L3", data=self.L3)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group):
        return cls(np.array(group["L1"]), np.array(group["L2"]), np.array(group["L3"]))


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


class DelayArrays:
    worm_delay: NDArray[np.int_]
    exposure_delay: NDArray[np.float_]
    mf_delay: NDArray[np.int_]
    l1_delay: NDArray[np.float_]

    def __init__(self, worm_delay, exposure_delay, mf_delay, l1_delay) -> None:
        self.worm_delay = worm_delay
        self.exposure_delay = exposure_delay
        self.mf_delay = mf_delay
        self.l1_delay = l1_delay

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DelayArrays)
            and array_fully_equal(self.worm_delay, other.worm_delay)
            and array_fully_equal(self.exposure_delay, other.exposure_delay)
            and array_fully_equal(self.mf_delay, other.mf_delay)
            and array_fully_equal(self.l1_delay, other.l1_delay)
        )

    @classmethod
    def from_params(cls, params: Params, n_people: int, individual_exposure):
        number_of_worm_delay_cols = math.ceil(
            params.blackfly.l3_delay
            * params.month_length_days
            / (params.delta_time * params.year_length_days)
        )
        # matrix for tracking mf for L1 delay
        number_of_mf_columns = math.ceil(
            params.blackfly.l1_delay / (params.delta_time * params.year_length_days)
        )
        # matrix for exposure (to fly bites) for L1 delay
        number_of_exposure_columns: int = math.ceil(
            params.blackfly.l1_delay / (params.delta_time * params.year_length_days)
        )
        return cls(
            worm_delay=np.zeros((number_of_worm_delay_cols, n_people), dtype=int),
            exposure_delay=np.tile(
                individual_exposure, (number_of_exposure_columns, 1)
            ),
            mf_delay=(
                np.ones((number_of_mf_columns, n_people), dtype=int)
                * params.microfil.initial_mf
            ),
            l1_delay=np.repeat(params.blackfly.initial_L1, n_people),
        )

    def append_to_hdf5_group(self, group: h5py.Group):
        group.create_dataset("worm_delay", data=self.worm_delay)
        group.create_dataset("exposure_delay", data=self.exposure_delay)
        group.create_dataset("mf_delay", data=self.mf_delay)
        group.create_dataset("l1_delay", data=self.l1_delay)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group):
        return cls(
            np.array(group["worm_delay"]),
            np.array(group["exposure_delay"]),
            np.array(group["mf_delay"]),
            np.array(group["l1_delay"]),
        )

    def process_deaths(self, people_to_die: NDArray[np.bool_]):
        if np.any(people_to_die):
            self.worm_delay[:, people_to_die] = 0
            self.mf_delay[0, people_to_die] = 0
            self.l1_delay[people_to_die] = 0
            # TODO: Do we need self.exposure_delay = 0


@dataclass
class People:
    compliance: NDArray[np.bool_]  # 1: 'column used during treatment'
    sex_is_male: NDArray[np.bool_]  # 3: sex
    blackfly: BlackflyLarvae
    ages: NDArray[np.float_]  # 2: current age
    mf: NDArray[np.float_]  # 2D Array, (N, age stage): microfilariae stages 7-28 (21)
    worms: WormGroup
    time_of_last_treatment: NDArray[np.float_]  # treat.vec
    delay_arrays: DelayArrays
    individual_exposure: NDArray[np.float_]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, People)
            and array_fully_equal(self.compliance, other.compliance)
            and array_fully_equal(self.sex_is_male, other.sex_is_male)
            and self.blackfly == other.blackfly
            and array_fully_equal(self.ages, other.ages)
            and array_fully_equal(self.mf, other.mf)
            and self.worms == other.worms
            and array_fully_equal(
                self.time_of_last_treatment, other.time_of_last_treatment
            )
            and self.delay_arrays == other.delay_arrays
            and array_fully_equal(self.individual_exposure, other.individual_exposure)
        )

    def __len__(self):
        return len(self.compliance)

    def append_to_hdf5_group(self, group: h5py.Group):
        blackfly_group = group.create_group("blackfly")
        delay_arrays_group = group.create_group("delay_arrays")
        group.create_dataset("compliance", data=self.compliance)
        group.create_dataset("sex_is_male", data=self.sex_is_male)
        self.blackfly.append_to_hdf5_group(blackfly_group)
        group.create_dataset("ages", data=self.ages)
        group.create_dataset("mf", data=self.mf)
        group.create_dataset("male_worms", data=self.worms.male)
        group.create_dataset("infertile_female_worms", data=self.worms.infertile)
        group.create_dataset("fertile_female_worms", data=self.worms.fertile)
        group.create_dataset("time_of_last_treatment", data=self.time_of_last_treatment)
        self.delay_arrays.append_to_hdf5_group(delay_arrays_group)
        group.create_dataset("individual_exposure", data=self.individual_exposure)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group):
        blackfly_group = group["blackfly"]
        assert isinstance(blackfly_group, h5py.Group)
        delay_arrays_group = group["delay_arrays"]
        assert isinstance(delay_arrays_group, h5py.Group)
        return cls(
            np.array(group["compliance"]),
            np.array(group["sex_is_male"]),
            BlackflyLarvae.from_hdf5_group(blackfly_group),
            np.array(group["ages"]),
            np.array(group["mf"]),
            WormGroup(
                male=np.array(group["male_worms"]),
                infertile=np.array(group["infertile_female_worms"]),
                fertile=np.array(group["fertile_female_worms"]),
            ),
            np.array(group["time_of_last_treatment"]),
            DelayArrays.from_hdf5_group(delay_arrays_group),
            np.array(group["individual_exposure"]),
        )

    @classmethod
    def from_params(cls, params: Params, n_people: int, gamma_distribution=0.3):
        sex_array = (
            np.random.uniform(low=0, high=1, size=n_people) < params.humans.gender_ratio
        )
        compliance_array = (
            np.random.uniform(low=0, high=1, size=n_people)
            > params.humans.noncompliant_percentage
        )
        time_of_last_treatment = np.empty(n_people)
        time_of_last_treatment[:] = np.nan

        individual_exposure = (
            np.random.gamma(  # individual level exposure to fly bites "ex.vec"
                shape=gamma_distribution,
                scale=gamma_distribution,
                size=n_people,
            )
        )
        new_individual_exposure = individual_exposure / np.mean(
            individual_exposure
        )  # normalise
        new_individual_exposure.setflags(write=False)

        return cls(
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
            worms=WormGroup(
                male=np.ones((params.worms.worm_age_stages, n_people), dtype=int)
                * params.worms.initial_worms,
                infertile=np.ones((params.worms.worm_age_stages, n_people), dtype=int)
                * params.worms.initial_worms,
                fertile=np.ones((params.worms.worm_age_stages, n_people), dtype=int)
                * params.worms.initial_worms,
            ),
            time_of_last_treatment=time_of_last_treatment,
            delay_arrays=DelayArrays.from_params(
                params, n_people, new_individual_exposure
            ),
            individual_exposure=new_individual_exposure,
        )

    def process_deaths(self, people_to_die: NDArray[np.bool_], gender_ratio):
        if (total_people_to_die := int(np.sum(people_to_die))) > 0:
            self.sex_is_male[people_to_die] = (
                np.random.uniform(low=0, high=1, size=total_people_to_die)
                < gender_ratio
            )
            self.ages[people_to_die] = 0
            self.blackfly.L1[people_to_die] = 0
            self.mf[:, people_to_die] = 0
            self.worms.male[:, people_to_die] = 0
            self.worms.fertile[:, people_to_die] = 0
            self.worms.infertile[:, people_to_die] = 0
        self.delay_arrays.process_deaths(people_to_die)


def _calc_coverage(
    people: People,
    measured_coverage: float,
    age_compliance: float,
) -> NDArray[np.bool_]:

    non_compliant_people = (people.ages < age_compliance) | ~people.compliance
    compliant_percentage = 1 - np.mean(non_compliant_people)
    coverage = measured_coverage / compliant_percentage  # TODO: Is this correct?
    out_coverage = np.repeat(coverage, len(people))
    out_coverage[non_compliant_people] = 0
    rand_nums = np.random.uniform(low=0, high=1, size=len(people))
    return rand_nums < out_coverage


def _calculate_total_exposure(
    exposure_params: ExposureParams, people: People
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

    total_exposure = sex_age_exposure * people.individual_exposure
    return total_exposure / np.mean(total_exposure)


CallbackStat = TypeVar("CallbackStat")


class State(Generic[CallbackStat]):
    _people: People
    _params: Params
    _derived_params: DerivedParams

    def __init__(self, people: People, params: Params) -> None:
        self._people = people
        self.params = params

    @classmethod
    def from_params(
        cls, params: Params, n_people: int, gamma_distribution=0.3
    ):  # "gam.dis" individual level exposure heterogeneity
        return cls(
            people=People.from_params(params, n_people, gamma_distribution),
            params=params,
        )

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._derived_params = DerivedParams(value, self.n_people)
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
                self._people.worms.fertile + self._people.worms.infertile,
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
        infected_over_min_age = float(np.sum(mf_skin_snip[pop_over_min_age_array] > 0))
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
            male_worms=NumericArrayStat.from_array(self._people.worms.male),
            infertile_female_worms=NumericArrayStat.from_array(
                self._people.worms.infertile
            ),
            fertile_female_worms=NumericArrayStat.from_array(
                self._people.worms.fertile
            ),
            mf_per_skin_snip=self.microfilariae_per_skin_snip()[0],
            population_prevalence=self.mf_prevalence_in_population(),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, State)
            and self._people == other._people
            and self.params == other.params
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

        total_exposure = _calculate_total_exposure(self.params.exposure, self._people)
        # increase ages
        self._people.ages += self.params.delta_time

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
            self._people.delay_arrays.worm_delay,
            self.n_people,
            self.params.worms.sex_ratio,
        )

        # Move all rows in worm_delay along one
        self._people.delay_arrays.worm_delay = lag_array(
            new_worms, self._people.delay_arrays.worm_delay
        )

        old_fertile_female_worms = self._people.worms.fertile.copy()
        old_male_worms = self._people.worms.male.copy()

        self._people.worms, last_time_of_last_treatment = change_in_worms(
            current_worms=self._people.worms,
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
            self._people.delay_arrays.mf_delay[-1],
            total_exposure,
            self._people.delay_arrays.exposure_delay[-1],
            self.params.year_length_days,
        )

        old_blackfly_L2 = self._people.blackfly.L2
        self._people.blackfly.L2 = calc_l2(
            self.params.blackfly,
            self._people.delay_arrays.l1_delay,
            self._people.delay_arrays.mf_delay[-1],
            self._people.delay_arrays.exposure_delay[-1],
            self.params.year_length_days,
        )
        self._people.blackfly.L3 = calc_l3(self.params.blackfly, old_blackfly_L2)

        self._people.delay_arrays.exposure_delay = lag_array(
            total_exposure, self._people.delay_arrays.exposure_delay
        )
        self._people.delay_arrays.mf_delay = lag_array(
            old_mf, self._people.delay_arrays.mf_delay
        )
        self._people.delay_arrays.l1_delay = self._people.blackfly.L1

        people_to_die: NDArray[np.bool_] = np.logical_or(
            np.random.binomial(
                n=1,
                p=(1 / self.params.humans.mean_human_age) * self.params.delta_time,
                size=self.n_people,
            )
            == 1,
            self._people.ages >= self.params.humans.max_human_age,
        )
        self._people.process_deaths(people_to_die, self.params.humans.gender_ratio)

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

    def run_simulation_output_callback(
        self: "State",
        output_callback: Callable[[People, float], CallbackStat],
        sampling_interval: float,
        start_time: float = 0,
        end_time: float = 0,
        verbose: bool = False,
    ) -> list[CallbackStat]:
        if end_time < start_time:
            raise ValueError("End time after start")

        current_time = start_time
        output_stats: list[CallbackStat] = []
        while current_time < end_time:
            if self.params.delta_time > current_time % 0.2 and verbose:
                print(current_time)
            if self.params.delta_time > current_time % sampling_interval:
                output_stats.append(output_callback(self._people, current_time))
            self._advance(current_time=current_time)
            current_time += self.params.delta_time
        return output_stats

    @classmethod
    def from_hdf5(cls, input_file: str | Path | IO):
        f = h5py.File(input_file, "r")
        people_group = f["people"]
        assert isinstance(people_group, h5py.Group)
        params: str = str(f.attrs["params"])
        return cls(People.from_hdf5_group(people_group), Params.parse_raw(params))

    def to_hdf5(self, output_file: str | Path | IO):
        f = h5py.File(output_file, "w")
        group_people = f.create_group("people")
        self._people.append_to_hdf5_group(group_people)
        f.attrs["params"] = self._params.json()


def make_state_from_params(params: Params, n_people: int):
    return State.from_params(params, n_people)


def make_state_from_hdf5(input_file: str | Path | IO):
    return State.from_hdf5(input_file)
