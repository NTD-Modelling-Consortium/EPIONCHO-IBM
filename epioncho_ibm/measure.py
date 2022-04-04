from typing import List

from .types import Person


def mf_prevalence_in_population(min_age_skinsnip: int, people: List[Person]) -> float:
    """
    Returns a decimal representation of mf prevalence in skinsnip aged population.
    """
    pop_over_min_age = 0
    infected_over_min_age = 0

    for person in people:
        if person.age >= min_age_skinsnip:
            pop_over_min_age += 1
            if person.mf_current_quantity > 0:
                infected_over_min_age += 1

    return pop_over_min_age / infected_over_min_age
