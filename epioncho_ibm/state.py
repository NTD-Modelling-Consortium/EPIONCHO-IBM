from typing import List

import numpy as np

from .types import Person, RandomConfig
from .params import Params

class State:
    current_iteration: int = 0
    _people: List[Person]

    def __init__(self, people: List[Person]) -> None:
        self._people = people

    @classmethod
    def generate_random(cls, random_config: RandomConfig, params: Params) -> "State":
        return cls([Person.generate_random(random_config, params) for _ in range(params.human_population)])

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

        for person in self._people:
            if person.age >= min_age_skinsnip:
                pop_over_min_age += 1
                if person.mf_current_quantity > 0:
                    infected_over_min_age += 1

        return pop_over_min_age / infected_over_min_age

    def dist_population_age(
        self, num_iter: int = 1, DT: float = 1 / 366, mean_age: float = 50
    ):
        """
        function that updates age of the population in state by DT
        """
        number_of_people = len(self._people)
        for i in range(num_iter):
            for person in self._people:
                person.age += DT
            death_vec = np.random.binomial(
                1, ((1 / mean_age) * (1 / 366)), number_of_people
            )
            for i in range(number_of_people):
                if death_vec[i] == 1:
                    self._people[i].age = 0
                if self._people[i].age >= 80:
                    self._people[i].age = 0



def advance_state(state: State, params: Params, n_iters: int = 1) -> State:
    def _next(state: State) -> State:
        state.current_iteration += 1
        # if state.current iteration >= params.treatmet_start_iter BEGIN TREATMENT THIS WAY
        raise NotImplementedError

    for i in range(n_iters):
        state = _next(state)

    return state
