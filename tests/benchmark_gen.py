import math
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel


class NTDSettings(BaseModel):
    min_year: float
    max_year: float
    year_steps: int
    min_pop: int
    max_pop: int
    pop_steps: int
    max_pop_years: float


def get_test_pairs(settings: NTDSettings) -> List[Tuple[float, int]]:
    def get_exponentially_spaced_steps(
        start: Union[int, float], end: Union[int, float], n_steps: int
    ) -> NDArray[np.float_]:
        log_start = math.log(start)
        log_end = math.log(end)
        exp_spaced = np.linspace(log_start, log_end, n_steps)
        return np.exp(exp_spaced)

    pop_spaces: NDArray[np.int_] = np.round(
        get_exponentially_spaced_steps(
            settings.min_pop, settings.max_pop, settings.pop_steps
        )
    )
    year_spaces = get_exponentially_spaced_steps(
        settings.min_year, settings.max_year, settings.year_steps
    )
    year_cols = year_spaces[:, None]
    pop_years = year_cols * pop_spaces
    valid_tests = pop_years < settings.max_pop_years
    coords = valid_tests.nonzero()
    years_for_test = year_spaces[coords[0]]
    pops_for_test = pop_spaces[coords[1]]
    return list(zip(years_for_test, pops_for_test))


settings_path = Path("ntd_settings.json")
settings_model = NTDSettings.parse_file(settings_path)

test_pairs = get_test_pairs(settings_model)
print(test_pairs)
