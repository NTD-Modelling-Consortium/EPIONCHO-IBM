from typing import ClassVar, Optional, overload

import numpy as np

from .types import Array


@overload
def convert_prob(
    current_prob: Array.Person.Float, scale_factor: float
) -> Array.Person.Float:
    ...


@overload
def convert_prob(current_prob: float, scale_factor: float) -> float:
    ...


def convert_prob(
    current_prob: float | Array.Person.Float, scale_factor: float
) -> float | Array.Person.Float:
    return 1 - ((1 - current_prob) ** scale_factor)


class Sequela:
    """
    None for days remains positive means non-reversible
    """

    probability: ClassVar[float]
    probability_interval_years: ClassVar[float]
    days_remains_positive: ClassVar[Optional[float]] = None

    @classmethod
    def timestep_probability(cls, delta_time: float) -> float:
        scale_factor = delta_time / cls.probability_interval_years
        return convert_prob(current_prob=cls.probability, scale_factor=scale_factor)


class MFDependentSequela:
    probability_interval_years: ClassVar[float]
    days_remains_positive: ClassVar[Optional[float]] = None

    @overload
    @classmethod
    def _probability(cls, mf_count: float) -> float:
        ...

    @overload
    @classmethod
    def _probability(cls, mf_count: Array.Person.Float) -> Array.Person.Float:
        ...

    @classmethod
    def _probability(
        cls, mf_count: float | Array.Person.Float
    ) -> float | Array.Person.Float:
        raise NotImplementedError("Must implement prob method for mf dependent sequela")

    @classmethod
    def timestep_probability(
        cls, mf_count: float | Array.Person.Float, delta_time: float
    ) -> float | Array.Person.Float:
        scale_factor = delta_time / cls.probability_interval_years
        return convert_prob(
            current_prob=cls._probability(mf_count), scale_factor=scale_factor
        )


class Blindness(MFDependentSequela):
    probability_interval_years: float = 1.0
    prob_background_blindness: float = 0.003
    gamma1: float = 0.01

    @classmethod
    def _probability(cls, mf_count: float | Array.Person.Float):
        return cls.prob_background_blindness * np.exp(cls.gamma1 * mf_count)
