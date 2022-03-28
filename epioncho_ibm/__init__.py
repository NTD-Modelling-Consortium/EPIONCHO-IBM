version = 1.0

from enum import Enum
from typing import List

from pydantic import conlist
from pydantic.main import BaseModel


class RandomConfig(BaseModel):
    gender_ration: float = 0.5


class Sex(Enum):
    male = "male"
    female = "female"


class BlackflyLarvae(BaseModel):
    L1: int  # 4: L1
    L2: int  # 5: L2
    L3: int  # 6: L3


worms_stages = 21
micro_stages = 21
WormsStageList = conlist(
    item_type=float, min_items=worms_stages, max_items=worms_stages
)
MicroStageList = conlist(
    item_type=float, min_items=micro_stages, max_items=micro_stages
)


class Person(BaseModel):
    treatment: int  # 1: column used during treatment
    current_age: int  # 2: current age
    sex: Sex  # 3: sex
    blackfly: BlackflyLarvae
    mf: MicroStageList  # microfilariae stages (21)
    worms: WormsStageList  # Worm stages (21)

    @classmethod
    def generate_random(
        cls, random_config: RandomConfig
    ) -> "Person":  # Other params here
        raise NotImplementedError


class State:
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


class Params(BaseModel):
    # See line 476 R code
    pass


def advance_state(state: State, params: Params, n_iters: int = 1) -> State:
    def _next(state: State) -> State:
        raise NotImplementedError

    for i in range(n_iters):
        state = _next(state)
    return state
