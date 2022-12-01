import numpy as np

from .params import ExposureParams
from .types import Array


def calculate_total_exposure(
    exposure_params: ExposureParams,
    ages: Array.Person.Float,
    sex_is_male: Array.Person.Bool,
    individual_exposure: Array.Person.Float,
) -> Array.Person.Float:
    male_exposure_assumed = exposure_params.male_exposure * np.exp(
        -exposure_params.male_exposure_exponent * ages
    )
    male_exposure_assumed_of_males = male_exposure_assumed[sex_is_male]
    if len(male_exposure_assumed_of_males) == 0:
        # TODO: Is this correct?
        mean_male_exposure = 0
    else:
        mean_male_exposure: float = float(np.mean(male_exposure_assumed_of_males))
    female_exposure_assumed = exposure_params.female_exposure * np.exp(
        -exposure_params.female_exposure_exponent * ages
    )
    female_exposure_assumed_of_females = female_exposure_assumed[
        np.logical_not(sex_is_male)
    ]
    if len(female_exposure_assumed_of_females) == 0:
        # TODO: Is this correct?
        mean_female_exposure = 0
    else:
        mean_female_exposure: float = float(np.mean(female_exposure_assumed_of_females))

    sex_age_exposure = np.where(
        sex_is_male,
        male_exposure_assumed / mean_male_exposure,
        female_exposure_assumed / mean_female_exposure,
    )

    total_exposure = sex_age_exposure * individual_exposure
    return total_exposure / np.mean(total_exposure)
