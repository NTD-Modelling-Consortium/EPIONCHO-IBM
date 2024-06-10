import numpy as np

from epioncho_ibm.state import Array, ExposureParams


def calculate_total_exposure(
    exposure_params: ExposureParams,
    gender_ratio: float,
    ages: Array.Person.Float,
    sex_is_male: Array.Person.Bool,
    individual_exposure: Array.Person.Float,
    use_onchosim_mechanisms: str,
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

    def calc_exposure(exposure_intercept, exposure_exponent, ages, gender_selector):
        """
        Args:
            exposure_intercept: A derived parameter that controls the intercept of the exposure curve (sex-specific intercept)
            exposure_exponent: A float that controls the slope of the exposure curve (rate of change of contact rate in human)
            ages: An array of the ages of the people in the model

        Returns:
            Array.Person.Float: The sex-age exposure of each person to infection
        """
        raw_exposure = np.exp(-exposure_exponent * ages)
        gamma_s = 1 / (np.mean(raw_exposure[gender_selector]))
        return exposure_intercept * gamma_s * raw_exposure

    # E.F
    female_exposure = 1 / ((gender_ratio * (exposure_params.Q - 1)) + 1)
    # E.M
    male_exposure = exposure_params.Q * female_exposure

    male_exposure_array = calc_exposure(
        male_exposure, exposure_params.male_exposure_exponent, ages, sex_is_male
    )
    female_exposure_array = calc_exposure(
        female_exposure,
        exposure_params.female_exposure_exponent,
        ages,
        np.logical_not(sex_is_male),
    )

    onchosim_exposure = (
        use_onchosim_mechanisms == "both" or use_onchosim_mechanisms == "age-sex"
    )
    if onchosim_exposure:
        male_exposure_array = np.where(
            ages < exposure_params.male_exposure_const_age,
            ages
            * (
                exposure_params.male_exposure_max
                / exposure_params.male_exposure_const_age
            ),
            exposure_params.male_exposure_max,
        )

        female_exposure_array = np.where(
            ages < exposure_params.female_exposure_const_age,
            ages
            * (
                exposure_params.female_exposure_max
                / exposure_params.female_exposure_const_age
            ),
            exposure_params.female_exposure_max,
        )

    sex_age_exposure = np.where(
        sex_is_male,
        male_exposure_array,
        female_exposure_array,
    )

    total_exposure = sex_age_exposure * individual_exposure
    if onchosim_exposure:
        return total_exposure
    return total_exposure / np.mean(total_exposure)
