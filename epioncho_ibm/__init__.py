version = 1.0
from typing import List

from pydantic import BaseModel

from .types import Person, RandomConfig


class State:
    current_iteration: int = 0
    people: List[Person]

    def __init__(self, people: List[Person]) -> None:
        self._people = people

    @classmethod
    def generate_random(cls, n_people: int, random_config: RandomConfig) -> "State":
        return cls([Person.generate_random(random_config) for _ in range(n_people)])

    def prevelence(self: "State") -> float:
        raise NotImplementedError

    def microfilariae_per_skin_snip(self: "State") -> float:
        raise NotImplementedError

    def mf_prevalence_in_population(self: "State", min_age_skinsnip: int) -> float:
        """
        Returns a decimal representation of mf prevalence in skinsnip aged population.
        """
        # TODO: Handle exceptions

        pop_over_min_age = 0
        infected_over_min_age = 0

        for person in self.people:
            if person.age >= min_age_skinsnip:
                pop_over_min_age += 1
                if person.mf_current_quantity > 0:
                    infected_over_min_age += 1

        return pop_over_min_age / infected_over_min_age


class Params(BaseModel):
    treatment_probability: float = 0.65  # The probability that a 'treatable' person is actually treated in an iteration
    treatment_start_iter: int  # The iteration upon which treatment commences (treat.start in R code)
    # See line 476 R code

    # So-called Hard coded params
    bite_rate_per_person_per_year = 1000  # Annual biting rate ABR
    bite_rate_per_fly_on_human = bite_rate_per_person_per_year * (
        (1 / 104) / 0.63
    )  # calculated as the product of the proportion of blackfly bites taken on humans 'm = ABR * [...]'

    # Params within blackfly vector...
    l1_l2_per_person_per_year = (
        201.6189  # Per capita development rate of larvae from stage L1 to L2 'nuone'
    )
    l2_l3_per_person_per_year = (
        207.7384  # Per capita development rate of larvae from stage L2 to L3 'nutwo'
    )
    blackfly_mort_per_person_per_year = (
        26  # Per capita mortality rate of blackfly vectors 'mu.v'
    )
    blackfly_mort_from_mf_per_person_per_year = (
        0.39  # Per capita microfilaria-induced mortality of blackfly vectors 'a.v'
    )


def advance_state(state: State, params: Params, n_iters: int = 1) -> State:
    def _next(state: State) -> State:
        state.current_iteration += 1
        # if state.current iteration >= params.treatmet_start_iter BEGIN TREATMENT THIS WAY
        raise NotImplementedError

    for i in range(n_iters):
        state = _next(state)

    return state
