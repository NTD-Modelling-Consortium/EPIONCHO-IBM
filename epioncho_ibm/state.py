from typing import List, Optional

import numpy as np

from .types import Person, RandomConfig
from .params import Params
from numpy.typing import NDArray

class State:
    current_iteration: int = 0
    _people: List[Person]
    _params: Params

    def __init__(self, people: List[Person], params: Params) -> None:
        self._people = people
        self._params = params

    @classmethod
    def generate_random(cls, random_config: RandomConfig, params: Params) -> "State":
        return cls(
            [Person.generate_random(random_config, params) for _ in range(params.human_population)], 
            params=params
        )

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
        self, num_iter: int = 1, params: Optional[Params] = None,
    ):
        """
        Generate age distribution
        create inital age distribution and simulate stable age distribution
        """
        if params is None:
            params = self._params
        delta_time = params.delta_time
        mean_age = params.mean_human_age
        max_age = params.max_human_age
        number_of_people = len(self._people)
        current_ages = np.zeros(number_of_people)
        delta_time_vector = np.ones(number_of_people)*delta_time
        for i in range(num_iter):
            current_ages += delta_time_vector
            death_vector = np.random.binomial(
                n = 1, 
                p = (1/mean_age) * delta_time, 
                size = number_of_people
            )
            np.place(current_ages, death_vector == 1 or current_ages >=max_age , 0)
        for i, person in enumerate(self._people):
            person.age = current_ages[i]


def calc_coverage(people: List[Person], percent_non_compliant: float, coverage: float, age_compliance: float =5):
    
    non_compliant_people = [person for person in people if person.age < age_compliance or not person.compliant]
    non_compliant_percentage = len(non_compliant_people)/len(people)
    compliant_percentage = 1 - non_compliant_percentage
    new_coverage = coverage/compliant_percentage # TODO: Is this correct?


    ages = np.array([person.age for person in people])



def advance_state(state: State, params: Params, n_iters: int = 1) -> State:
    def _next(state: State) -> State:
        state.current_iteration += 1
        # if state.current iteration >= params.treatmet_start_iter BEGIN TREATMENT THIS WAY
        if(i >= params.treatment_start_iter):

             pass
             #{cov.in <- os.cov(all.dt = all.mats.cur, pncomp = pnc, covrg = treat.prob, N = N)}
    
        raise NotImplementedError

    for i in range(n_iters):
        state = _next(state)

    return state
