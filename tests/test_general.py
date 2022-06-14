import numpy as np
import pytest

from epioncho_ibm import Params, RandomConfig, State, run_simulation

"""
@pytest.mark.asyncio
class TestGeneral:
    async def test_generate_random(self):
        random_config = RandomConfig()
        params = Params(human_population=5)
        initial_state = State.generate_random(
            random_config=random_config, params=params
        )

    async def test_distribute_random(self):
        random_config = RandomConfig()
        params = Params(human_population=5)
        initial_state = State.generate_random(
            random_config=random_config, params=params
        )
        initial_state.dist_population_age(num_iter=15000)

    async def test_run_simulation_short(self):
        random_config = RandomConfig()
        params = Params(human_population=5)
        initial_state = State.generate_random(
            random_config=random_config, params=params
        )
        initial_state.dist_population_age(num_iter=15000)
        new_state = run_simulation(initial_state, start_time=0, end_time=0.25)
        assert 50 <= np.sum(new_state.people.male_worms)
        assert np.sum(new_state.people.male_worms) <= 60
"""
