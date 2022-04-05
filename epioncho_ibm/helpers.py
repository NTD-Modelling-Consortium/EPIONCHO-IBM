from random import random

from .types import Person


def is_treatable(person: Person, min_treatable_age: int) -> bool:
    """
    Treatable people are above the min_treatable_age, and have not already been treated.
    """
    return person.age >= min_treatable_age and not person.treated


def is_to_treat(
    person: Person, to_treat_probability: float, min_treatable_age: int
) -> bool:
    """
    Returns whether a given person is to be treated in a given iteration.
    """
    # TODO: Check with client how covrg (to_treat_probability) should be calculated based on implementation currently in os.cov()
    # covrg <- covrg / (1 - nc.age)

    if is_treatable(person, min_treatable_age):
        return random() < to_treat_probability
    return False


def get_L3_developing_in_human(
    person: Person,
    delta_hz: float,
    delta_hinf: float,
    c_h: float,
    L3,
    m: float,
    beta: float,
) -> float:
    expo = person.exposure
    out = (delta_hz + delta_hinf * c_h * m * beta * L3 * expo) / (
        1 + c_h * m * beta * L3 * expo
    )
    return out
