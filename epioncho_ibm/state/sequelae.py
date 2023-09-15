from typing import ClassVar, Literal, Optional, overload

import numpy as np

from .blindness_prob import blindness_prob
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
        has_this_sequela: Array.Person.Bool,
        countdown: Array.Person.Float,
    ) -> float | Array.Person.Float:
        raise NotImplementedError("Must implement prob method for mf dependent sequela")

    @classmethod
    def timestep_probability(
        cls,
        delta_time: float,
        mf_count: Array.Person.Float,
        ages: Array.Person.Float,
        existing_sequela: dict[str, Array.Person.Bool],
        has_this_sequela: Array.Person.Bool,
        countdown: Array.Person.Float,
    ) -> float | Array.Person.Float:
        scale_factor = delta_time / cls.probability_interval_years
        return convert_prob(
            current_prob=cls._probability(
                mf_count=mf_count,
                ages=ages,
                existing_sequela=existing_sequela,
                has_this_sequela=has_this_sequela,
                countdown=countdown,
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
        has_this_sequela: Array.Person.Bool,
        countdown: Array.Person.Float,
    ) -> float | Array.Person.Float:
        new_probs = np.zeros_like(mf_count)
        mask = np.logical_and(mf_count > 0, ~has_this_sequela, ages >= 2)
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
        has_this_sequela: Array.Person.Bool,
        countdown: Array.Person.Float,
    ) -> float | Array.Person.Float:
        new_probs = np.zeros_like(mf_count)
        mask = np.logical_and(mf_count > 0, np.logical_not(has_this_sequela))
        new_probs[mask] = cls.prob
        return new_probs


class Blindness(Sequela):
    probability_interval_years: float = 1.0
    years_countdown: float = 2.0
    end_countdown_become_positive = True
    prob_background_blindness: float = 0.003
    gamma1: float = 0.01
    prob_mapper = np.array(blindness_prob)

    @classmethod
    def _probability(
        cls,
        mf_count: Array.Person.Float,
        ages: Array.Person.Float,
        existing_sequela: dict[str, Array.Person.Bool],
        has_this_sequela: Array.Person.Bool,
        countdown: Array.Person.Float,
    ) -> float | Array.Person.Float:
        not_during_countdown = np.logical_or(countdown <= 0, countdown == np.inf)
        for_sample = np.logical_and(
            not_during_countdown, np.logical_not(has_this_sequela)
        )
        out = np.zeros_like(mf_count)
        # out[for_sample] = np.minimum(cls.prob_background_blindness * np.exp(
        #     cls.gamma1 * mf_count[for_sample]
        # ),1)
        mask = np.round(mf_count[for_sample]).astype(int)
        new_arr = np.ones_like(mask, dtype=float)
        valid_items = mask < len(cls.prob_mapper)
        new_arr[valid_items] = cls.prob_mapper[mask[valid_items]]
        out[for_sample] = new_arr
        return out


class SevereItching(_BaseReversible):
    prob = 0.1636701
    # prob = 0.129


class RSD(_BaseReversible):
    prob = 0.04163095
    # prob = 0.019


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
        has_this_sequela: Array.Person.Bool,
        countdown: Array.Person.Float,
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

sequela_mapper = {
    "Blindness": Blindness,
    "SevereItching": SevereItching,
    "RSD": RSD,
    "APOD": APOD,
    "CPOD": CPOD,
    "Atrophy": Atrophy,
    "HangingGroin": HangingGroin,
    "Depigmentation": Depigmentation,
}
