from typing import List, Optional

import numpy as np

from .params import Params
from numpy.typing import NDArray
from dataclasses import dataclass
from copy import copy
from pydantic import BaseModel

class RandomConfig(BaseModel):
    gender_ratio: float = 0.5
    noncompliant_percentage: float = 0.05

@dataclass
class BlackflyLarvae:
    L1: NDArray[np.float_]  # 4: L1
    L2: NDArray[np.float_]  # 5: L2
    L3: NDArray[np.float_]  # 6: L3


@dataclass
class People:
    compliance: NDArray[np.bool_] # 1: 'column used during treatment'
    sex_is_male: NDArray[np.bool_] # 3: sex
    blackfly: BlackflyLarvae 
    ages: NDArray[np.float_] # 2: current age
    mf: NDArray[np.float_] # 2D Array, (N, age stage): microfilariae stages 7-28 (21)
    worms: NDArray[np.float_] # 2D Array, (N, age stage): microfilariae stages 29-50 (21)
    mf_current_quantity: NDArray[np.int_]
    exposure: NDArray[np.float_]
    new_worm_rate: NDArray[np.float_]

    def __len__(self):
        return len(self.compliance)

class State:
    current_iteration: int = 0
    _people: People
    _params: Params

    def __init__(self, people: People, params: Params) -> None:
        self._people = people
        self._params = params

    @classmethod
    def generate_random(cls, random_config: RandomConfig, n_people: int, params: Params) -> "State":
        sex_array = np.random.uniform(low = 0, high = 1, size=n_people) < random_config.gender_ratio
        zeros_array = np.zeros(n_people)
        compliance_array = np.random.uniform(low = 0, high = 1, size=n_people) > random_config.noncompliant_percentage
        ones_array = np.ones(n_people)
        return cls( 
            people =People(
                compliance=compliance_array, 
                ages = np.zeros(n_people),
                sex_is_male = sex_array,
                blackfly = BlackflyLarvae(
                    L1 = ones_array * params.initial_L1, 
                    L2 = ones_array * params.initial_L2, 
                    L3 = ones_array * params.initial_L3
                ),
                mf = np.zeros((n_people, params.microfil_age_stages)),
                worms = np.zeros((n_people, params.worm_age_stages)),
                mf_current_quantity = np.zeros(n_people, dtype = int),
                exposure = np.zeros(n_people),
                new_worm_rate =  np.zeros(n_people)
            ), 
            params = params
        )

    def prevelence(self: "State") -> float:
        raise NotImplementedError

    def microfilariae_per_skin_snip(self: "State") -> float:
        raise NotImplementedError

    def mf_prevalence_in_population(self: "State", min_age_skinsnip: int) -> float:
        """
        Returns a decimal representation of mf prevalence in skinsnip aged population.
        """
        pop_over_min_age_array = self._people.ages >= min_age_skinsnip
        pop_over_min_age = np.sum(pop_over_min_age_array)
        infected_over_min_age = np.sum(np.logical_and(pop_over_min_age_array, self._people.mf_current_quantity > 0))
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
        
        current_ages = self._people.ages
        size_population = len(self._people)
        delta_time_vector = np.ones(size_population)*params.delta_time
        for i in range(num_iter):
            current_ages += delta_time_vector
            death_vector = np.random.binomial(n = 1, p = (1/params.mean_human_age) * params.delta_time, size = size_population)
            np.place(current_ages, np.logical_or(death_vector == 1, current_ages >=params.max_human_age), 0)
        return current_ages


def calc_coverage(people: People, percent_non_compliant: float, coverage: float, age_compliance: float =5):
    
    non_compliant_people = np.logical_or(people.ages < age_compliance, np.logical_not(people.compliance))
    non_compliant_percentage = np.sum(non_compliant_people)/len(non_compliant_people)
    compliant_percentage = 1 - non_compliant_percentage
    new_coverage = coverage/compliant_percentage # TODO: Is this correct?


    #ages = np.array([person.age for person in people])



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
