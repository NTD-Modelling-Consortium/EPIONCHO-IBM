import numpy as np

from epioncho_ibm.state import Array, ExposureParams


def calculate_total_exposure(
    exposure_params: ExposureParams,
    ages: Array.Person.Float,
    sex_is_male: Array.Person.Bool,
    individual_exposure: Array.Person.Float,
) -> Array.Person.Float:
    """
    Calculates how much each person is exposed to infection, taking
    into account their age, gender and individual exposure bias.

    Args:
        exposure_params (ExposureParams): A set of fixed parameters that control exposure
        ages (Array.Person.Float): An array of the ages of the people in the model
        sex_is_male (Array.Person.Bool): The gender of each person in the model
        individual_exposure (Array.Person.Float): The individual bias to exposure set at random

    Returns:
        Array.Person.Float: The overall exposure of each person to infection
    """
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
