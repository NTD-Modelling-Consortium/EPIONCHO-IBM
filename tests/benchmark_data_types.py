from typing import Dict, List

from pydantic import BaseModel

from epioncho_ibm import Params
from epioncho_ibm.state import NumericArrayStat


class OutputData(BaseModel):
    end_year: float
    params: Params
    people: Dict[str, NumericArrayStat]


class TestData(BaseModel):
    tests: List[OutputData]
