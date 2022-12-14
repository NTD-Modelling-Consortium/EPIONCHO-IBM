from dataclasses import dataclass

import h5py
import numpy as np

from .params import Params
from .types import Array
from .utils import array_fully_equal
from .worms import WormGroup


def truncated_geometric(N: int, prob: float, maximum: float) -> Array.Person.Float:
    output = np.repeat(maximum + 1, N)
    while np.any(output > maximum):
        output[output > maximum] = np.random.geometric(
            p=prob, size=len(output[output > maximum])
        )
    return output


@dataclass
class BlackflyLarvae:
    L1: Array.Person.Float
    L2: Array.Person.Float
    L3: Array.Person.Float

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


class DelayArrays:
    _worm_delay: Array.L3Delay.Person.Int
    _exposure_delay: Array.L1Delay.Person.Float
    _mf_delay: Array.L1Delay.Person.Float
    _worm_delay_current: int
    _exposure_delay_current: int
    _mf_delay_current: int

    def __init__(
        self,
        worm_delay: Array.L3Delay.Person.Int,
        exposure_delay: Array.L1Delay.Person.Float,
        mf_delay: Array.L1Delay.Person.Float,
        worm_delay_current: int = 0,
        exposure_delay_current: int = 0,
        mf_delay_current: int = 0,
    ) -> None:
        self._worm_delay = worm_delay
        self._exposure_delay = exposure_delay
        self._mf_delay = mf_delay
        self._worm_delay_current = worm_delay_current
        self._exposure_delay_current = exposure_delay_current
        self._mf_delay_current = mf_delay_current

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DelayArrays)
            and array_fully_equal(self._worm_delay, other._worm_delay)
            and array_fully_equal(self._exposure_delay, other._exposure_delay)
            and array_fully_equal(self._mf_delay, other._mf_delay)
            and self._worm_delay_current == other._worm_delay_current
            and self._exposure_delay_current == other._exposure_delay_current
            and self._mf_delay_current == other._mf_delay_current
        )

    @property
    def worm_delay(self):
        if self._worm_delay.size:
            return self._worm_delay[self._worm_delay_current]
        else:
            return None

    @worm_delay.setter
    def worm_delay(self, value):
        assert self._worm_delay.size
        self._worm_delay[self._worm_delay_current] = value

    @property
    def exposure_delay(self):
        if self._exposure_delay.size:
            return self._exposure_delay[self._exposure_delay_current]
        else:
            return None

    @exposure_delay.setter
    def exposure_delay(self, value):
        assert self._exposure_delay.size
        self._exposure_delay[self._exposure_delay_current] = value

    @property
    def mf_delay(self):
        if self._mf_delay.size:
            return self._mf_delay[self._mf_delay_current]
        else:
            return None

    @mf_delay.setter
    def mf_delay(self, value):
        assert self._mf_delay.size
        self._mf_delay[self._mf_delay_current] = value

    @classmethod
    def from_params(
        cls, params: Params, n_people: int, individual_exposure: Array.Person.Float
    ):
        number_of_l3_delay_cols: int = round(
            params.blackfly.l3_delay
            * params.month_length_days
            / (params.delta_time * params.year_length_days)
        )
        number_of_l1_delay_columns: int = round(
            params.blackfly.l1_delay / (params.delta_time * params.year_length_days)
        )

        return cls(
            worm_delay=np.zeros((number_of_l3_delay_cols, n_people), dtype=int),
            exposure_delay=np.tile(
                individual_exposure, (number_of_l1_delay_columns, 1)
            ),
            mf_delay=(
                np.ones((number_of_l1_delay_columns, n_people), dtype=int)
                * params.microfil.initial_mf
            ),
        )

    def append_to_hdf5_group(self, group: h5py.Group):
        group.create_dataset("worm_delay", data=self._worm_delay)
        group.create_dataset("exposure_delay", data=self._exposure_delay)
        group.create_dataset("mf_delay", data=self._mf_delay)
        group.attrs["worm_delay_current"] = self._worm_delay_current
        group.attrs["exposure_delay_current"] = self._exposure_delay_current
        group.attrs["mf_delay_current"] = self._mf_delay_current

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group):
        return cls(
            np.array(group["worm_delay"]),
            np.array(group["exposure_delay"]),
            np.array(group["mf_delay"]),
            group.attrs["worm_delay_current"],
            group.attrs["exposure_delay_current"],
            group.attrs["mf_delay_current"],
        )

    def process_deaths(self, people_to_die: Array.Person.Bool):
        if np.any(people_to_die):
            if self._worm_delay.size:
                self._worm_delay[:, people_to_die] = 0
            if self._mf_delay.size:
                self._mf_delay[self._mf_delay_current, people_to_die] = 0
            # TODO: Do we need self.exposure_delay = 0

    def lag_all_arrays(
        self,
        new_worms: Array.Person.Int,
        total_exposure: Array.Person.Float,
        new_mf: Array.Person.Float,
    ):
        if self._worm_delay.size:
            self.worm_delay = new_worms
            self._worm_delay_current = (
                1 + self._worm_delay_current
            ) % self._worm_delay.shape[0]

        if self._exposure_delay.size:
            self.exposure_delay = total_exposure
            self._exposure_delay_current = (
                1 + self._exposure_delay_current
            ) % self._exposure_delay.shape[0]

        if self._mf_delay.size:
            self.mf_delay = new_mf
            self._mf_delay_current = (
                1 + self._mf_delay_current
            ) % self._mf_delay.shape[0]


@dataclass
class People:
    compliance: Array.Person.Bool
    sex_is_male: Array.Person.Bool
    blackfly: BlackflyLarvae
    ages: Array.Person.Float
    mf: Array.MFCat.Person.Float
    worms: WormGroup
    time_of_last_treatment: Array.Person.Float
    delay_arrays: DelayArrays
    individual_exposure: Array.Person.Float

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
    def from_params(
        cls, params: Params, n_people: int, gamma_distribution: float = 0.3
    ):
        sex_array = (
            np.random.uniform(low=0, high=1, size=n_people) < params.humans.gender_ratio
        )
        compliance_array = (
            np.random.uniform(low=0, high=1, size=n_people)
            > params.humans.noncompliant_percentage
        )
        time_of_last_treatment = np.empty(n_people)
        time_of_last_treatment[:] = np.nan

        # individual exposure to fly bites
        individual_exposure = np.random.gamma(
            shape=gamma_distribution,
            scale=gamma_distribution,
            size=n_people,
        )
        new_individual_exposure = individual_exposure / np.mean(individual_exposure)
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

    def process_deaths(self, people_to_die: Array.Person.Bool, gender_ratio: float):
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
