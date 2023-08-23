from typing import ClassVar, Literal, Optional, overload

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
    None for years countdown means non-reversible, and no lagged effects

    end_countdown_become_positive = True means lagged, when years countdown is active
    end_countdown_become_positive = False means reversible, when years countdown is active
    """

    probability_interval_years: ClassVar[float]
    years_countdown: ClassVar[Optional[float]] = None
    end_countdown_become_positive: Optional[bool] = None

    @classmethod
    def _probability(
        cls,
        mf_count: Array.Person.Float,
        ages: Array.Person.Float,
        existing_sequela: dict[str, Array.Person.Bool],
    ) -> float | Array.Person.Float:
        raise NotImplementedError("Must implement prob method for mf dependent sequela")

    @classmethod
    def timestep_probability(
        cls,
        delta_time: float,
        mf_count: Array.Person.Float,
        ages: Array.Person.Float,
        existing_sequela: dict[str, Array.Person.Bool],
    ) -> float | Array.Person.Float:
        scale_factor = delta_time / cls.probability_interval_years
        return convert_prob(
            current_prob=cls._probability(
                mf_count=mf_count, ages=ages, existing_sequela=existing_sequela
            ),
            scale_factor=scale_factor,
        )


class _BaseReversible(Sequela):
    probability_interval_years: float = 1 / 365
    years_countdown: float = 3 / 365
    end_countdown_become_positive = False
    prob: float

    @classmethod
    def _probability(
        cls,
        mf_count: Array.Person.Float,
        ages: Array.Person.Float,
        existing_sequela: dict[str, Array.Person.Bool],
    ) -> float | Array.Person.Float:
        new_probs = np.zeros_like(mf_count)
        mask = mf_count > 0 and ages >= 2
        new_probs[mask] = cls.prob
        return new_probs


class _BaseNonReversible(Sequela):
    probability_interval_years: float = 1.0
    prob: float

    @classmethod
    def _probability(
        cls,
        mf_count: Array.Person.Float,
        ages: Array.Person.Float,
        existing_sequela: dict[str, Array.Person.Bool],
    ) -> float | Array.Person.Float:
        new_probs = np.zeros_like(mf_count)
        mask = mf_count > 0
        new_probs[mask] = cls.prob
        return new_probs


class Blindness(Sequela):
    probability_interval_years: float = 1.0
    years_countdown: float = 2.0
    end_countdown_become_positive = True
    prob_background_blindness: float = 0.003
    gamma1: float = 0.01

    @classmethod
    def _probability(
        cls,
        mf_count: Array.Person.Float,
        ages: Array.Person.Float,
        existing_sequela: dict[str, Array.Person.Bool],
    ) -> float | Array.Person.Float:
        return cls.prob_background_blindness * np.exp(cls.gamma1 * mf_count)


class SevereItching(_BaseReversible):
    prob = 0.1636701


class RSD(_BaseReversible):
    prob = 0.04163095


class APOD(_BaseReversible):
    prob = 0.04163095  # TODO: Find prob


class CPOD(_BaseNonReversible):
    prob: float = 0.01  # TODO: Find prob

    @classmethod
    def _probability(
        cls,
        mf_count: Array.Person.Float,
        ages: Array.Person.Float,
        existing_sequela: dict[str, Array.Person.Bool],
    ) -> float | Array.Person.Float:
        if "APOD" not in existing_sequela:
            raise ValueError("CPOD active, but APOD is not")
        else:
            has_apod = existing_sequela["APOD"]
            new_prob = np.zeros_like(ages)
            new_prob[has_apod] = cls.prob
            return new_prob


class Atrophy(_BaseNonReversible):
    prob = 0.002375305


class HangingGroin(_BaseNonReversible):
    prob = 0.0007263018


class Depigmentation(_BaseNonReversible):
    prob = 0.001598305


SequelaType = list[
    Literal["Blindness"]
    | Literal["SevereItching"]
    | Literal["RSD"]
    | Literal["APOD"]
    | Literal["CPOD"]
    | Literal["Atrophy"]
    | Literal["HangingGroin"]
    | Literal["Depigmentation"]
]
