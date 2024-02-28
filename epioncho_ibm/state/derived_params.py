import math
from typing import Optional

import numpy as np
from fast_binomial import SFC64, Generator

import epioncho_ibm.state.sequelae

from .params import Params
from .sequelae import Sequela, SequelaType, sequela_mapper
from .types import Array


def _weibull_mortality(
    mu1: float, mu2: float, age_categories: Array.General.Float
) -> Array.General.Float:
    return (mu1**mu2) * mu2 * (age_categories ** (mu2 - 1))


def get_sequela(
    sequela: SequelaType,
) -> dict[str, type[Sequela]]:
    return {name: sequela_mapper[name] for name in sequela if name in sequela_mapper}


class DerivedParams:
    worm_mortality_rate: Array.WormCat.Float
    fecundity_rates_worms: Array.WormCat.Float
    microfillarie_mortality_rate: Array.MFCat.Float
    treatment_times: Optional[Array.Treatments.Float]
    people_to_die_generator: Generator
    worm_age_rate_generator: Generator
    worm_sex_ratio_generator: Generator
    worm_lambda_zero_generator: Generator
    worm_omega_generator: Generator
    worm_mortality_generator: Generator
    sequela_classes: dict[str, type[Sequela]]

    def __init__(
        self, params: Params, oldGenerators: dict[str, Generator] = None
    ) -> None:
        worm_age_categories: Array.WormCat.Float = np.arange(
            start=0,
            stop=params.worms.max_worm_age,
            step=params.worms.max_worm_age / params.worms.worm_age_stages,
        )
        self.worm_mortality_rate = params.delta_time * _weibull_mortality(
            params.worms.mu_worms1,
            params.worms.mu_worms2,
            worm_age_categories,
        )

        self.fecundity_rates_worms = (
            params.worms.mf_production_per_worm
            * params.worms.fecundity_worms_1
            / (
                params.worms.fecundity_worms_1
                + (params.worms.fecundity_worms_2 ** (-worm_age_categories))
                - 1
            )
        )

        microfillarie_age_categories = np.linspace(
            start=0,
            stop=params.microfil.max_microfil_age,
            num=params.microfil.microfil_age_stages,
        )
        self.microfillarie_mortality_rate = _weibull_mortality(
            params.microfil.mu_microfillarie1,
            params.microfil.mu_microfillarie2,
            microfillarie_age_categories,
        )
        if params.treatment is not None:
            treatment_number = (
                params.treatment.stop_time - params.treatment.start_time
            ) / params.treatment.interval_years
            if round(treatment_number) != round(treatment_number, 10):
                raise ValueError(
                    f"Treatment times could not be found for start: {params.treatment.start_time}, stop: {params.treatment.stop_time}, interval: {params.treatment.interval_years}"
                )
            treatment_number_int: int = math.ceil(treatment_number)
            self.treatment_times = np.linspace(
                start=params.treatment.start_time,
                stop=params.treatment.stop_time,
                num=treatment_number_int + 1,
            )
        else:
            self.treatment_times = None

        if oldGenerators is None:
            seeds = [
                params.seed + i + 1 if params.seed is not None else None
                for i in range(6)
            ]
            self.people_to_die_generator = Generator(
                SFC64(seeds[0]), params.delta_time / params.humans.mean_human_age
            )
            self.worm_age_rate_generator = Generator(
                SFC64(seeds[1]), params.delta_time / params.worms.worms_aging
            )
            self.worm_sex_ratio_generator = Generator(
                SFC64(seeds[2]), params.worms.sex_ratio
            )
            self.worm_lambda_zero_generator = Generator(
                SFC64(seeds[3]), params.worms.lambda_zero * params.delta_time
            )
            self.worm_omega_generator = Generator(
                SFC64(seeds[4]), params.worms.omega * params.delta_time
            )
            self.worm_mortality_generator = Generator(
                SFC64(seeds[5]), self.worm_mortality_rate
            )
        else:
            self.people_to_die_generator = oldGenerators["people_to_die_generator"]
            self.worm_age_rate_generator = oldGenerators["worm_age_rate_generator"]
            self.worm_sex_ratio_generator = oldGenerators["worm_sex_ratio_generator"]
            self.worm_lambda_zero_generator = oldGenerators[
                "worm_lambda_zero_generator"
            ]
            self.worm_omega_generator = oldGenerators["worm_omega_generator"]
            self.worm_mortality_generator = oldGenerators["worm_mortality_generator"]

        self.sequela_classes = get_sequela(params.sequela_active)
