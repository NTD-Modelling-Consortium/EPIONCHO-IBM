version = 1.0
from typing import List

import numpy as np
from pydantic import BaseModel

from .types import Person, RandomConfig


class State:
    current_iteration: int = 0
    _people: List[Person]

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


class Params(BaseModel):
    # ep.equi.sim parameters (bottom of 'R' file)
    timestep_count: int = 10  # total number of timesteps of the simulation
    bite_rate_per_person_per_year: float = (
        1000  # Annual biting rate 'ABR' in paper and in R code
    )
    treatment_intrvl_yrs: float = (
        1  # 'trt.int' treatment interval (years, 0.5 gives biannual)
    )
    timestep_size: float = 1 / 366  # the timestep ('DT.in' and 'DT' in code)
    treatment_probability: float = 0.65  # The probability that a 'treatable' person is actually treated in an iteration
    # unclear what gv.trt / give.treat is, given that it is '1'. Might be flag to enable or disable treatment logic
    treatment_start_iter: int = (
        0  # The iteration upon which treatment commences (treat.start in R code)
    )
    treatment_stop_iter: int = 0  # the iteration up which treatment stops (treat.stop)
    # 'pnc' or percentage non compliant is in random config
    min_skinsnip_age: int = 5  # TODO: below
    min_treatable_age: int = 5  # TODO: check if skinsnip and treatable age differ or whether they are always the same value

    # See line 476 R code

    # So-called Hard coded params
    # '.' have been replaced with '_'
    human_population: int = 440  # 'N' in R file

    # TODO: find out from client what the origin is of these values (delta.hz, delta.hinf, and c.h)
    delta_hz: float = 0.1864987  # Proportion of L3 larvae developing to the adult stage within the human host, per bite when 𝐴𝑇𝑃(𝑡) → 0
    delta_hinf: float = 0.002772749  # Proportion of L3 larvae developing to the adult stage within the human host, per bite when 𝐴𝑇𝑃(𝑡) → ∞
    c_h: float = 0.004900419  # Severity of transmission intensity dependent parasite establishment within humans

    human_blood_index: float = 0.63  # 'h' in paper, used in 'm' and 'beta' in R code
    recip_gono_cycle: float = 1 / 104  # 'g' in paper, used in 'm' and 'beta' in R code
    bite_rate_per_fly_on_human: float = (
        human_blood_index / recip_gono_cycle
    )  # defined in table D in paper, is 'beta' in R code
    annual_transm_potential: float = (
        bite_rate_per_person_per_year * human_population
    ) / bite_rate_per_fly_on_human  # ATP in doc, possible this is 'm' - implemented based on doc calculation
    blackfly_mort_per_person_per_year: float = (
        26  # Per capita mortality rate of blackfly vectors 'mu.v'
    )
    int_mf: int = 0  # TODO: just 0, and doesn't change in program. What is the purpose?
    sigma_L0: int = 52  # TODO: unclear where this comes from, and what it means
    a_H: float = 0.8  # Time delay between L3 entering the host and establishing as adult worms in years
    # g is 'recip_gono_cycle'
    blackfly_mort_from_mf_per_person_per_year: float = (
        0.39  # Per capita microfilaria-induced mortality of blackfly vectors 'a.v'
    )
    max_age_person: int = 80  # 'real.max.age' in R file
    # human population is defined earlier
    mean_human_age: int = 50  # years 'mean.age' in R file

    # TODO: after establishing what int.L1/2/3 are, implement here
    lambda_zero: float = (
        1 / 3
    )  # Per capita rate of reversion from fertile to non-fertile adult female worms (lambda.zero / 0.33 in 'R' code)
    # omega
    omega: float = (
        0.59  # Per capita rate of progression from non-fertile to fertile adult female
    )
    delta_v_o: float = 0.0166  # TODO: verify with client, used in calc.L1
    c_v: float = 0.0205  # TODO: verify with client, used in calc.L1
    # sex.rat = 0.5 is in random config
    # num.mf.comps and num.comps.worm are both in the Person object

    # aging in parasites
    worms_aging: float = 1  # 'time.each.comp.worms'
    microfil_aging: float = 0.125  # 'time.each.comp.mf'
    microfil_move_rate: float = 8.13333  # 'mf.move.rate'

    l1_l2_per_person_per_year: float = (
        201.6189  # Per capita development rate of larvae from stage L1 to L2 'nuone'
    )
    l2_l3_per_person_per_year: float = (
        207.7384  # Per capita development rate of larvae from stage L2 to L3 'nutwo'
    )
    # Params within parasite


def advance_state(state: State, params: Params, n_iters: int = 1) -> State:
    def _next(state: State) -> State:
        state.current_iteration += 1
        # if state.current iteration >= params.treatmet_start_iter BEGIN TREATMENT THIS WAY
        raise NotImplementedError

    for i in range(n_iters):
        state = _next(state)

    return state
