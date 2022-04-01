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


class Params(BaseModel):
    treatment_probability: float = 0.65  # The probability that a 'treatable' person is actually treated in an iteration
    treatment_start_iter: int  # The iteration upon which treatment commences (treat.start in R code)
    # See line 476 R code


def advance_state(state: State, params: Params, n_iters: int = 1) -> State:
    def _next(state: State) -> State:
        state.current_iteration += 1
        # if state.current iteration >= params.treatmet_start_iter BEGIN TREATMENT THIS WAY
        raise NotImplementedError

    for i in range(n_iters):
        state = _next(state)

    return state
