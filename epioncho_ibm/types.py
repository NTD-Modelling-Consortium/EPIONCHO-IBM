from enum import Enum
from random import uniform
from typing import Type

from pydantic import BaseModel, conlist

from .params import Params


class RandomConfig(BaseModel):
    gender_ratio: float = 0.5
    noncompliant_percentage: float = 0.05


worms_stages = 21

micro_stages = 21

WormsStageList = conlist(
    item_type=float, min_items=worms_stages, max_items=worms_stages
)

MicroStageList = conlist(
    item_type=float, min_items=micro_stages, max_items=micro_stages
)


class Sex(Enum):

    male = "male"

    female = "female"


class BlackflyLarvae(BaseModel):
    L1: float  # 4: L1
    L2: float  # 5: L2
    L3: float  # 6: L3


class Person(BaseModel):

    compliant: bool  # 1: 'column used during treatment'
    age: float = 0  # 2: current age
    sex: Sex  # 3: sex

    blackfly: BlackflyLarvae

    mf: MicroStageList = []  # microfilariae stages (21)

    worms: WormsStageList = []  # Worm stages (21)

    mf_current_quantity: int = 0

    exposure: float = 0

    new_worm_rate: float = 0

    # treated: bool  # TODO: check if exists

    @classmethod
    def generate_random(
        cls, random_config: RandomConfig, params: Params
    ) -> "Person":  # Other params here

        if uniform(0, 1) < random_config.gender_ratio:
            sex = Sex.male
        else:
            sex = Sex.female
        compliant = uniform(0, 1) > random_config.noncompliant_percentage
        # TODO: is this the right place for these?^^

        return Person(sex=sex, compliant=compliant, blackfly = BlackflyLarvae(L1 = params.initial_L1, L2 = params.initial_L2, L3 = params.initial_L3))
