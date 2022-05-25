from random import random
from typing import List

import numpy as np
from numpy.typing import NDArray

from epioncho_ibm.state import People
"""
def is_treatable(person: People, min_treatable_age: int) -> bool:
    ""
    Treatable people are above the min_treatable_age, and are compliant.
    ""
    return person.ages >= min_treatable_age and person.compliance


def is_to_treat(
    person: Person, to_treat_probability: float, min_treatable_age: int
) -> bool:
    ""
    Returns whether a given person is to be treated in a given iteration.
    ""
    # TODO: Check with client how covrg (to_treat_probability) should be calculated based on implementation currently in os.cov()
    # covrg <- covrg / (1 - nc.age)

    if is_treatable(person, min_treatable_age):
        return random() < to_treat_probability
    return False
"""

def get_L3_developing_in_human(
    exposure: NDArray[np.float_],
    delta_hz: float,
    delta_hinf: float,
    c_h: float,
    L3,
    bite_rate_per_fly_on_human: float,
    beta: float,
) -> NDArray[np.float_]:
    """
    Returns the proportion of L3 larvae developing into works in a given person.
    Return value is between 0 and 1.
    """
    # TODO: verify why L3 is initially set to 0.03 when it appears it should be an integer
    out = (
        delta_hz + delta_hinf * c_h * bite_rate_per_fly_on_human * beta * L3 * exposure
    ) / (1 + c_h * bite_rate_per_fly_on_human * beta * L3 * exposure)
    return out


def get_new_worm_infections(
    people: People,
    delta_hz: float,
    delta_hinf: float,
    c_h: float,
    L3,
    bite_rate_per_fly_on_human: float,
    beta: float,
):
    """
    The rate of acquisition of new infections in each human line 222
    """


    people.new_worm_rate = get_L3_developing_in_human(
        people.exposure,
        delta_hz,
        delta_hinf,
        c_h,
        L3,
        bite_rate_per_fly_on_human,
        beta,
    )
    # TODO: Question 4 in questions txt
    # Calculate the value of new worms in the population
    # Why is the vector dh, created using m, beta, dh, expos, and L3 then again multiplied by each of these in Wplus1.rate?
    raise NotImplementedError
