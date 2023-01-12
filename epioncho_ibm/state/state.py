from dataclasses import field
from pathlib import Path
from typing import IO

import numpy as np
from hdf5_dataclass import HDF5Dataclass
from pydantic import BaseModel

from .derived_params import DerivedParams
from .params import ImmutableParams, Params, immutable_to_mutable, mutable_to_immutable
from .people import People
from .types import Array

np.seterr(all="ignore")


class NumericArrayStat(BaseModel):
    mean: float
    # st_dev: float

    @classmethod
    def from_array(cls, array: Array.Person.Float | Array.Person.Int):
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


def negative_binomial_alt_interface(
    n: Array.General.Float, mu: Array.General.Float
) -> Array.General.Int:
    """
    Provides an alternate interface for random negative binomial.

    Args:
        n (Array.General.Float): Number of successes
        mu (Array.General.Float): Mean of the distribution

    Returns:
        Array.General.Int: Samples from a negative binomial distribution
    """
    non_zero_n = n[n > 0]
    rel_prob = non_zero_n / (non_zero_n + mu[n > 0])
    temp_output = np.random.negative_binomial(
        n=non_zero_n, p=rel_prob, size=len(non_zero_n)
    )
    output = np.zeros(len(n), dtype=int)
    output[n > 0] = temp_output
    return output


class State(HDF5Dataclass):
    people: People
    _params: ImmutableParams
    current_time: float = 0.0
    derived_params: DerivedParams = field(init=False, repr=False)

    @property
    def n_people(self):
        """
        The number of people simulated.
        """
        return len(self.people)

    def get_params(self) -> Params:
        return immutable_to_mutable(self._params)

    def reset_params(self, params: Params):
        """Reset the parameters

        Args:
            params (Params): New set of parameters
        """
        self._params = mutable_to_immutable(params)
        self._derive_params()

    def _derive_params(self) -> None:
        assert self._params
        self.derived_params = DerivedParams(immutable_to_mutable(self._params))

    @classmethod
    def from_params(
        cls,
        params: Params,
        n_people: int,
        gamma_distribution: float = 0.3,
        current_time: float = 0.0,
    ) -> "State":
        """Generate the initial state of the model, based on parameters.

        Args:
            params (Params): A set of fixed parameters for controlling the model.
            n_people (int): The number of people to be simulated
            gamma_distribution (float): Individual level exposure heterogeneity. Defaults to 0.3.
            current_time (float): Current time of the simulation's state. Defaults to 0

        Returns:
            State: The state of the model
        """
        return cls(
            people=People.from_params(params, n_people, gamma_distribution),
            _params=mutable_to_immutable(params),
            current_time=current_time,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, State)
            and self.people == other.people
            and self._params == other._params
            and self.current_time == other.current_time
        )

    def stats(self) -> StateStats:
        return StateStats(
            percent_compliant=float(np.sum(self.people.compliance))
            / len(self.people.compliance),
            percent_male=float(np.sum(self.people.sex_is_male))
            / len(self.people.compliance),
            L1=NumericArrayStat.from_array(self.people.blackfly.L1),
            L2=NumericArrayStat.from_array(self.people.blackfly.L2),
            L3=NumericArrayStat.from_array(self.people.blackfly.L3),
            ages=NumericArrayStat.from_array(self.people.ages),
            mf=NumericArrayStat.from_array(self.people.mf),
            male_worms=NumericArrayStat.from_array(self.people.worms.male),
            infertile_female_worms=NumericArrayStat.from_array(
                self.people.worms.infertile
            ),
            fertile_female_worms=NumericArrayStat.from_array(self.people.worms.fertile),
            mf_per_skin_snip=self.microfilariae_per_skin_snip()[0],
            population_prevalence=self.mf_prevalence_in_population(),
        )

    def microfilariae_per_skin_snip(self) -> tuple[float, Array.Person.Float]:
        """
        Calculates number of mf in skin snip for all people.

        People are tested for the presence of mf using a skin snip.
        We assume mf are overdispersed in the skin

        Returns:
            tuple[float, Array.Person.Float]: Mean mf, mf by person.
        """
        kmf = (
            self._params.microfil.slope_kmf
            * np.sum(
                self.people.worms.fertile + self.people.worms.infertile,
                axis=0,
            )
            + self._params.microfil.initial_kmf
        )

        mu = self._params.humans.skin_snip_weight * np.sum(self.people.mf, axis=0)
        if self._params.humans.skin_snip_number > 1:
            total_skin_snip_mf = np.zeros(
                (
                    self.n_people,
                    self._params.humans.skin_snip_number,
                )
            )
            for i in range(self._params.humans.skin_snip_number):
                total_skin_snip_mf[:, i] = negative_binomial_alt_interface(n=kmf, mu=mu)
            mfobs: Array.Person.Int = np.sum(total_skin_snip_mf, axis=1)
        else:
            mfobs: Array.Person.Int = negative_binomial_alt_interface(n=kmf, mu=mu)

        mfobs_percent: Array.Person.Float = mfobs / (
            self._params.humans.skin_snip_number * self._params.humans.skin_snip_weight
        )
        return float(np.mean(mfobs_percent)), mfobs_percent

    def mf_prevalence_in_population(self) -> float:
        """
        Calculates mf prevalence in population.

        Returns:
            float: mf_prevalence
        """
        pop_over_min_age_array = (
            self.people.ages >= self._params.humans.min_skinsnip_age
        )
        _, mf_skin_snip = self.microfilariae_per_skin_snip()
        infected_over_min_age = float(np.sum(mf_skin_snip[pop_over_min_age_array] > 0))
        total_over_min_age = float(np.sum(pop_over_min_age_array))
        return infected_over_min_age / total_over_min_age


def make_state_from_params(
    params: Params, n_people: int, gamma_distribution: float = 0.3
):
    return State.from_params(params, n_people, gamma_distribution=gamma_distribution)


def make_state_from_hdf5(input_file: str | Path | IO[bytes]):
    return State.from_hdf5(input_file)