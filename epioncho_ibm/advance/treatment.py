from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator

from epioncho_ibm.state import Array, HumanParams, TreatmentParams


def _calc_coverage(
    ages: Array.Person.Float,
    compliance: Array.Person.Bool,
    measured_coverage: float,
    age_of_compliance: float,
    numpy_bit_gen: Generator,
) -> Array.Person.Bool:
    """
    Calculates whether each person in the model is covered by a treatment.

    Args:
        ages (Array.Person.Float): The ages of the people in the model
        compliance (Array.Person.Bool): Whether each person in the model is compliant
        measured_coverage (float): A measured value of coverage assuming all people are compliant.
        age_of_compliance (float): How old a person must be to be compliant
        numpy_bit_gen: (Generator): The random number generator for numpy

    Returns:
        Array.Person.Bool: An array stating if each person in the model is treated
    """
    non_compliant_people = (ages < age_of_compliance) | ~compliance
    compliant_percentage = 1 - np.mean(non_compliant_people)
    coverage = measured_coverage / compliant_percentage
    out_coverage = np.repeat(coverage, len(ages))
    out_coverage[non_compliant_people] = 0
    rand_nums = numpy_bit_gen.uniform(low=0, high=1, size=len(ages))
    return rand_nums < out_coverage


def _is_during_treatment(
    treatment: TreatmentParams,
    current_time: float,
    delta_time: float,
    treatment_times: Optional[Array.Treatments.Float],
) -> tuple[bool, bool]:
    """
    Returns two booleans describing if treatment has started, and if it occurred.

    Args:
        treatment_params (TreatmentParams | None): The fixed parameters relating to treatment
        current_time (float): The current time, t, in the model
        delta_time (float): dt - The amount of time advance in one time step
        treatment_times (Optional[Array.Treatments.Float]): The times for treatment across the model.

    Returns:
        tuple[bool, bool]: bool describing if treatment started,
            bool describing if treatment occurred, respectively
    """
    treatment_started = current_time >= treatment.start_time
    if treatment_started:
        assert treatment_times is not None
        treatment_occurred: bool = (
            bool(
                np.any(
                    (treatment_times <= current_time)
                    & (treatment_times > current_time - delta_time)
                )
            )
            and current_time <= treatment.stop_time
        )
    else:
        treatment_occurred = False
    return treatment_started, treatment_occurred


@dataclass
class TreatmentGroup:
    """
    treatment_params (TreatmentParams): The fixed parameters relating to treatment
    coverage_in (Array.Person.Bool): An array stating if each person in the model is treated
    treatment_times (Array.Treatments.Float): The times for treatment across the model.
    treatment_occurred (bool): A boolean stating if treatment occurred in this time step.
    """

    treatment_params: TreatmentParams
    coverage_in: Array.Person.Bool
    treatment_times: Array.Treatments.Float
    treatment_occurred: bool


def get_treatment(
    treatment_params: Optional[TreatmentParams],
    delta_time: float,
    current_time: float,
    treatment_times: Optional[Array.Treatments.Float],
    ages: Array.Person.Float,
    compliance: Optional[Array.Person.Bool],
    numpy_bit_gen: Generator,
) -> Optional[TreatmentGroup]:
    """
    Generates a treatment group, and calculates coverage, based on the current time

    Args:
        treatment_params (TreatmentParams | None): The fixed parameters relating to treatment
        delta_time (float): dt - The amount of time advance in one time step
        current_time (float): The current time, t, in the model
        treatment_times (Optional[Array.Treatments.Float]): The times for treatment across the model.
        ages (Array.Person.Float): The ages of the people
        compliance (Array.Person.Bool): The compliance of the people
        numpy_bit_gen: (Generator): The random number generator for numpy

    Returns:
        Optional[TreatmentGroup]: A treatment group containing information for later treatment
            calculation
    """
    if treatment_params is not None:
        assert compliance is not None
        assert treatment_times is not None
        treatment_started, treatment_occurred = _is_during_treatment(
            treatment_params,
            current_time,
            delta_time,
            treatment_times,
        )
        if treatment_started:
            coverage_in = _calc_coverage(
                ages,
                compliance,
                treatment_params.total_population_coverage,
                treatment_params.min_age_of_treatment,
                numpy_bit_gen,
            )
            return TreatmentGroup(
                treatment_params, coverage_in, treatment_times, treatment_occurred
            )
        else:
            return None
    else:
        return None
