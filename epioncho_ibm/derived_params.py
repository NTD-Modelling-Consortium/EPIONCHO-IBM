import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .params import Params


def _weibull_mortality(
    delta_time: float, mu1: float, mu2: float, age_categories: np.ndarray
) -> NDArray[np.float_]:
    return delta_time * (mu1**mu2) * mu2 * (age_categories ** (mu2 - 1))


class DerivedParams:
    worm_mortality_rate: NDArray[np.float_]
    fecundity_rates_worms: NDArray[np.float_]
    microfillarie_mortality_rate: NDArray[np.float_]
    initial_treatment_times: Optional[NDArray[np.float_]]
    individual_exposure: NDArray[np.float_]

    def __init__(self, params: Params) -> None:

        worm_age_categories = np.arange(
            start=0,
            stop=params.max_worm_age,
            step=params.max_worm_age / params.worm_age_stages,
        )  # age.cats
        self.worm_mortality_rate = _weibull_mortality(
            params.delta_time, params.mu_worms1, params.mu_worms2, worm_age_categories
        )
        self.fecundity_rates_worms = (
            1.158305
            * params.fecundity_worms_1
            / (
                params.fecundity_worms_1
                + (params.fecundity_worms_2 ** (-worm_age_categories))
                - 1
            )
        )

        # TODO revisit +1 and -1
        microfillarie_age_categories = np.arange(
            start=0,
            stop=params.max_microfil_age + 1,
            step=params.max_microfil_age / (params.microfil_age_stages - 1),
        )  # age.cats.mf

        self.microfillarie_mortality_rate = _weibull_mortality(
            params.delta_time,
            params.mu_microfillarie1,
            params.mu_microfillarie2,
            microfillarie_age_categories,
        )

        if params.give_treatment:
            treatment_number = (
                params.treatment_stop_time - params.treatment_start_time
            ) / params.treatment_interval_yrs
            if round(treatment_number) != treatment_number:
                raise ValueError(
                    f"Treatment times could not be found for start: {params.treatment_start_time}, stop: {params.treatment_stop_time}, interval: {params.treatment_interval_yrs}"
                )
            treatment_number_int: int = math.ceil(treatment_number)
            self.initial_treatment_times = np.linspace(  # "times.of.treat.in"
                start=params.treatment_start_time,
                stop=params.treatment_stop_time,
                num=treatment_number_int + 1,
            )
        else:
            self.initial_treatment_times = None

        individual_exposure = (
            np.random.gamma(  # individual level exposure to fly bites "ex.vec"
                shape=params.gamma_distribution,
                scale=params.gamma_distribution,
                size=params.human_population,
            )
        )
        self.individual_exposure = individual_exposure / np.mean(
            individual_exposure
        )  # normalise
