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

    def __init__(self, params: Params, n_people: int) -> None:

        worm_age_categories = np.arange(
            start=0,
            stop=params.worms.max_worm_age,
            step=params.worms.max_worm_age / params.worms.worm_age_stages,
        )  # age.cats
        self.worm_mortality_rate = _weibull_mortality(
            params.delta_time,
            params.worms.mu_worms1,
            params.worms.mu_worms2,
            worm_age_categories,
        )
        self.fecundity_rates_worms = (
            1.158305
            * params.worms.fecundity_worms_1
            / (
                params.worms.fecundity_worms_1
                + (params.worms.fecundity_worms_2 ** (-worm_age_categories))
                - 1
            )
        )

        microfillarie_age_categories = np.arange(
            start=0,
            stop=params.microfil.max_microfil_age,
            step=params.microfil.max_microfil_age / params.microfil.microfil_age_stages,
        )  # age.cats.mf

        self.microfillarie_mortality_rate = _weibull_mortality(
            params.delta_time,
            params.microfil.mu_microfillarie1,
            params.microfil.mu_microfillarie2,
            microfillarie_age_categories,
        )

        if params.treatment is not None:
            treatment_number = (
                params.treatment.stop_time - params.treatment.start_time
            ) / params.treatment.interval_years
            if round(treatment_number) != treatment_number:
                raise ValueError(
                    f"Treatment times could not be found for start: {params.treatment.start_time}, stop: {params.treatment.stop_time}, interval: {params.treatment.interval_years}"
                )
            treatment_number_int: int = math.ceil(treatment_number)
            self.initial_treatment_times = np.linspace(  # "times.of.treat.in"
                start=params.treatment.start_time,
                stop=params.treatment.stop_time,
                num=treatment_number_int + 1,
            )
        else:
            self.initial_treatment_times = None
