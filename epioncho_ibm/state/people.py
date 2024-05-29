from typing import Optional

import numpy as np
from hdf5_dataclass import HDF5Dataclass
from numpy.random import SFC64, Generator

from epioncho_ibm.utils import array_fully_equal

from .params import Params, TreatmentParams
from .types import Array


def truncated_geometric(
    N: int, prob: float, maximum: float, people_generator: Generator
) -> Array.Person.Float:
    output = np.repeat(maximum + 1, N)
    while np.any(output > maximum):
        output[output > maximum] = people_generator.geometric(
            p=prob, size=len(output[output > maximum])
        )
    return output


class BlackflyLarvae(HDF5Dataclass):
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


class DelayArrays(HDF5Dataclass):
    _worm_delay: Array.L3Delay.Person.Int
    _exposure_delay: Array.L1Delay.Person.Float
    _mf_delay: Array.L1Delay.Person.Float
    _worm_delay_current: int = 0
    _exposure_delay_current: int = 0
    _mf_delay_current: int = 0

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
    def from_params(cls, params: Params, individual_exposure: Array.Person.Float):
        number_of_l3_delay_cols: int = round(
            params.blackfly.l3_delay
            * params.month_length_days
            / (params.delta_time * params.year_length_days)
        )
        number_of_l1_delay_columns: int = round(
            params.blackfly.l1_delay / (params.delta_time * params.year_length_days)
        )

        return cls(
            _worm_delay=np.zeros((number_of_l3_delay_cols, params.n_people), dtype=int),
            _exposure_delay=np.tile(
                individual_exposure, (number_of_l1_delay_columns, 1)
            ),
            _mf_delay=(
                np.ones((number_of_l1_delay_columns, params.n_people), dtype=int)
                * params.microfil.initial_mf
            ),
        )

    def process_deaths(
        self, people_to_die: Array.Person.Bool, individual_exposure: Array.Person.Float
    ):
        if np.any(people_to_die):
            if self._worm_delay.size:
                self._worm_delay[:, people_to_die] = 0
            if self._mf_delay.size:
                self._mf_delay[self._mf_delay_current, people_to_die] = 0
            if self._exposure_delay.size:
                self._exposure_delay[:, people_to_die] = np.tile(
                    individual_exposure[people_to_die],
                    (self._exposure_delay.shape[0], 1),
                )

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


class WormGroup(HDF5Dataclass):
    """
    A group of worms, separated by sex and fertility
    """

    male: Array.WormCat.Person.Int
    infertile: Array.WormCat.Person.Int
    fertile: Array.WormCat.Person.Int

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WormGroup):
            return (
                np.array_equal(self.male, other.male)
                and np.array_equal(self.infertile, other.infertile)
                and np.array_equal(self.fertile, other.fertile)
            )
        else:
            return False

    @classmethod
    def from_population(cls, population: int):
        return cls(
            male=np.zeros(population, dtype=int),
            infertile=np.zeros(population, dtype=int),
            fertile=np.zeros(population, dtype=int),
        )

    def copy(self):
        return WormGroup(
            male=self.male.copy(),
            infertile=self.infertile.copy(),
            fertile=self.fertile.copy(),
        )


class LastTreatment(HDF5Dataclass):
    time: Array.Person.Float
    microfilaricidal_nu: Array.Person.Float
    microfilaricidal_omega: Array.Person.Float
    embryostatic_lambda_max: Array.Person.Float
    embryostatic_phi: Array.Person.Float
    permanent_infertility: Array.Person.Float

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, LastTreatment)
            and array_fully_equal(self.time, other.time)
            and array_fully_equal(self.microfilaricidal_nu, other.microfilaricidal_nu)
            and array_fully_equal(
                self.microfilaricidal_omega, other.microfilaricidal_omega
            )
            and array_fully_equal(
                self.embryostatic_lambda_max, other.embryostatic_lambda_max
            )
            and array_fully_equal(self.embryostatic_phi, other.embryostatic_phi)
            and array_fully_equal(
                self.permanent_infertility, other.permanent_infertility
            )
        )

    def copy(self):
        return LastTreatment(
            time=self.time.copy(),
            microfilaricidal_nu=self.microfilaricidal_nu.copy(),
            microfilaricidal_omega=self.microfilaricidal_omega.copy(),
            embryostatic_lambda_max=self.embryostatic_lambda_max.copy(),
            embryostatic_phi=self.embryostatic_phi.copy(),
            permanent_infertility=self.permanent_infertility.copy(),
        )


def dict_fully_equal(d1: dict[str, np.ndarray], d2: dict[str, np.ndarray]):
    if d1.keys() != d2.keys():
        return False
    for k, a1 in d1.items():
        if k not in d2:
            assert False
        else:
            a2 = d2[k]
            if not array_fully_equal(a1, a2):
                return False
    return True


class People(HDF5Dataclass):
    compliance: Array.Person.Float
    sex_is_male: Array.Person.Bool
    blackfly: BlackflyLarvae
    ages: Array.Person.Float
    mf: Array.MFCat.Person.Float
    worms: WormGroup
    last_treatment: LastTreatment
    delay_arrays: DelayArrays
    individual_exposure: Array.Person.Float
    was_infected: Array.Person.Bool
    tested_for_OAE: Array.Person.Bool
    has_OAE: Array.Person.Bool
    age_test_OAE: Array.Person.Float
    has_sequela: dict[str, Array.Person.Bool]
    countdown_sequela: dict[str, Array.Person.Float]
    has_been_treated: Optional[Array.Person.Bool]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, People):
            if self.compliance is not None and other.compliance is not None:
                compliance_equal = array_fully_equal(self.compliance, other.compliance)
            else:
                compliance_equal = (self.compliance is None) == (
                    other.compliance is None
                )
        else:
            compliance_equal = False
        return (
            isinstance(other, People)
            and compliance_equal
            and array_fully_equal(self.sex_is_male, other.sex_is_male)
            and self.blackfly == other.blackfly
            and array_fully_equal(self.ages, other.ages)
            and array_fully_equal(self.mf, other.mf)
            and self.worms == other.worms
            and self.last_treatment == other.last_treatment
            and self.delay_arrays == other.delay_arrays
            and array_fully_equal(self.individual_exposure, other.individual_exposure)
            and array_fully_equal(self.was_infected, other.was_infected)
            and array_fully_equal(self.tested_for_OAE, other.tested_for_OAE)
            and array_fully_equal(self.has_OAE, other.has_OAE)
            and array_fully_equal(self.age_test_OAE, other.age_test_OAE)
            and dict_fully_equal(self.has_sequela, other.has_sequela)
            and dict_fully_equal(self.countdown_sequela, other.countdown_sequela)
            and array_fully_equal(self.has_been_treated, other.has_been_treated)
        )

    def __len__(self):
        return len(self.sex_is_male)

    @classmethod
    def from_params(cls, params: Params):
        n_people = params.n_people
        if params.seed is not None:
            people_generator = Generator(SFC64(params.seed + 100))
        else:
            people_generator = Generator(SFC64())

        sex_array = (
            people_generator.uniform(low=0, high=1, size=n_people)
            < params.humans.gender_ratio
        )
        if params.treatment is None:
            compliance_array = np.zeros(n_people)
        else:
            compliance_array = People.draw_compliance_values(
                corr=params.treatment.correlation,
                cov=params.treatment.total_population_coverage,
                size=n_people,
                random_generator=people_generator,
            )
        last_treatment = np.empty(n_people)
        last_treatment[:] = np.nan
        last_treatment_full = LastTreatment(
            time=last_treatment.copy(),
            microfilaricidal_nu=last_treatment.copy(),
            microfilaricidal_omega=last_treatment.copy(),
            embryostatic_lambda_max=last_treatment.copy(),
            embryostatic_phi=last_treatment.copy(),
            permanent_infertility=last_treatment.copy(),
        )
        has_been_treated = np.full(n_people, False)
        # individual exposure to fly bites
        individual_exposure = people_generator.gamma(
            shape=params.gamma_distribution,
            scale=1 / params.gamma_distribution,
            size=n_people,
        )
        if params.microfil.initial_mf > 0 or params.worms.initial_worms > 0:
            was_infected = np.ones(n_people, dtype=bool)
        else:
            was_infected = np.zeros(n_people, dtype=bool)
        sequela_array = np.zeros(n_people, dtype=bool)
        has_sequela = {name: sequela_array.copy() for name in params.sequela_active}
        time_for_sequela = np.empty(n_people, dtype=float)
        time_for_sequela[:] = np.inf
        countdown_sequela = {
            name: time_for_sequela.copy() for name in params.sequela_active
        }
        return cls(
            compliance=compliance_array,
            ages=truncated_geometric(
                N=n_people,
                prob=params.delta_time / params.humans.mean_human_age,
                maximum=params.humans.max_human_age / params.delta_time,
                people_generator=people_generator,
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
            last_treatment=last_treatment_full,
            delay_arrays=DelayArrays.from_params(params, individual_exposure),
            individual_exposure=individual_exposure,
            was_infected=was_infected,
            tested_for_OAE=np.ones(n_people, dtype=bool),
            has_OAE=np.ones(n_people, dtype=bool),
            age_test_OAE=people_generator.uniform(3.0, 15.0, size=n_people),
            has_sequela=has_sequela,
            countdown_sequela=countdown_sequela,
            has_been_treated=has_been_treated,
        )

    @staticmethod
    def draw_compliance_values(
        corr: float,
        cov: float,
        size: int,
        random_generator: Generator,
    ):
        return random_generator.beta(
            a=cov * (1 - corr) / corr,
            b=(1 - cov) * (1 - corr) / corr,
            size=size,
        )

    def update_treatment_prob(self, corr: float, cov: float, numpy_bit_gen: Generator):
        """Draw new values for treatment probabilities.

        New treatment probability values are assigned to individuals
        ensuring that order in probablity value is kept across the
        population. In other words, individuals who had the highest
        probablity values before still do after.

        Args:
            corr (float): Treatment correlation value
            cov (float): Treatent coverage value
            numpy_bit_gen (Generator): A random number generator instance
                 from numpy.

        Returns:
            Array.Person.Float
        """
        new_probs = People.draw_compliance_values(
            corr, cov, size=len(self.ages), random_generator=numpy_bit_gen
        )
        self.compliance[np.argsort(self.compliance)] = np.sort(new_probs)

    def process_deaths(
        self,
        people_to_die: Array.Person.Bool,
        gender_ratio: float,
        numpy_bit_gen: Generator,
        treatment: Optional[TreatmentParams],
        gamma_distribution: float,
    ):
        if (total_people_to_die := int(np.sum(people_to_die))) > 0:
            self.sex_is_male[people_to_die] = (
                numpy_bit_gen.uniform(low=0, high=1, size=total_people_to_die)
                < gender_ratio
            )
            self.ages[people_to_die] = 0
            self.blackfly.L1[people_to_die] = 0
            self.mf[:, people_to_die] = 0
            self.worms.male[:, people_to_die] = 0
            self.worms.fertile[:, people_to_die] = 0
            self.worms.infertile[:, people_to_die] = 0
            self.was_infected[people_to_die] = False
            self.has_OAE[people_to_die] = False
            self.tested_for_OAE[people_to_die] = False
            self.age_test_OAE[people_to_die] = numpy_bit_gen.uniform(
                3.0, 15.0, size=total_people_to_die
            )
            self.individual_exposure[people_to_die] = numpy_bit_gen.gamma(
                shape=gamma_distribution,
                scale=1 / gamma_distribution,
                size=total_people_to_die,
            )
            self.has_been_treated[people_to_die] = False
            for arr in self.has_sequela.values():
                arr[people_to_die] = False
            for arr in self.countdown_sequela.values():
                arr[people_to_die] = np.inf

        self.delay_arrays.process_deaths(people_to_die, self.individual_exposure)
        if treatment:
            self.compliance[people_to_die] = People.draw_compliance_values(
                treatment.correlation,
                treatment.total_population_coverage,
                size=total_people_to_die,
                random_generator=numpy_bit_gen,
            )

    def get_people_for_age_group(self, age_start: float, age_end: float) -> "People":
        rel_ages = (self.ages >= age_start) & (self.ages < age_end)
        new_compliance = self.compliance[rel_ages]
        return People(
            compliance=new_compliance,
            sex_is_male=self.sex_is_male[rel_ages],
            blackfly=BlackflyLarvae(
                L1=self.blackfly.L1[rel_ages],
                L2=self.blackfly.L2[rel_ages],
                L3=self.blackfly.L3[rel_ages],
            ),
            ages=self.ages[rel_ages],
            mf=self.mf[:, rel_ages],
            worms=WormGroup(
                male=self.worms.male[:, rel_ages],
                fertile=self.worms.fertile[:, rel_ages],
                infertile=self.worms.infertile[:, rel_ages],
            ),
            last_treatment=LastTreatment(
                time=self.last_treatment.time[rel_ages],
                microfilaricidal_nu=self.last_treatment.microfilaricidal_nu[rel_ages],
                microfilaricidal_omega=self.last_treatment.microfilaricidal_nu[
                    rel_ages
                ],
                embryostatic_lambda_max=self.last_treatment.microfilaricidal_nu[
                    rel_ages
                ],
                embryostatic_phi=self.last_treatment.microfilaricidal_nu[rel_ages],
                permanent_infertility=self.last_treatment.microfilaricidal_nu[rel_ages],
            ),
            delay_arrays=DelayArrays(
                _worm_delay=self.delay_arrays._worm_delay[:, rel_ages],
                _exposure_delay=self.delay_arrays._exposure_delay[:, rel_ages],
                _mf_delay=self.delay_arrays._mf_delay[:, rel_ages],
                _worm_delay_current=self.delay_arrays._worm_delay_current,
                _exposure_delay_current=self.delay_arrays._exposure_delay_current,
                _mf_delay_current=self.delay_arrays._mf_delay_current,
            ),
            individual_exposure=self.individual_exposure[rel_ages],
            was_infected=self.was_infected[rel_ages],
            tested_for_OAE=self.tested_for_OAE[rel_ages],
            has_OAE=self.has_OAE[rel_ages],
            age_test_OAE=self.age_test_OAE[rel_ages],
            has_sequela={name: a[rel_ages] for name, a in self.has_sequela.items()},
            countdown_sequela={
                name: a[rel_ages] for name, a in self.countdown_sequela.items()
            },
            has_been_treated=self.has_been_treated[rel_ages],
        )

    def get_infected(self) -> Array.Person.Bool:
        total_male_worms = self.worms.male.sum(axis=0)
        total_female_worms = self.worms.fertile.sum(axis=0) + self.worms.infertile.sum(
            axis=0
        )
        return (total_male_worms > 0) & (total_female_worms > 0) & ~self.was_infected

    def get_current_tested_for_OAE(self) -> Array.Person.Bool:
        return (
            (self.age_test_OAE <= self.ages)
            & self.was_infected
            & np.logical_not(self.tested_for_OAE)
        )  # & np.logical_not(self.has_OAE)
