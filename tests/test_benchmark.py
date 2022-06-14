import pytest

from epioncho_ibm import RandomConfig, State, run_simulation
from tests.utils import flatten_dict


@pytest.mark.asyncio
class TestGeneral:
    async def test_run_simulation_short(self, benchmark_data):
        acceptable_st_devs = 3  # TODO: Figure out where to put this?

        random_config = RandomConfig()
        params = benchmark_data.params
        end_time = benchmark_data.end_year
        initial_state = State.generate_random(
            random_config=random_config, params=params
        )
        initial_state.dist_population_age(num_iter=15000)
        new_state = run_simulation(initial_state, start_time=0, end_time=end_time)

        people_stats = new_state.people.to_stats()
        people_stats_dict = flatten_dict(people_stats.dict())
        for k, v in people_stats_dict.items():
            if k not in benchmark_data.people:
                raise RuntimeError(f"Key {k} not present in benchmark")
            else:
                benchmark_item = benchmark_data.people[k]
            benchmark_item_mean = benchmark_item.mean
            benchmark_item_st_dev = benchmark_item.st_dev
            benchmark_lower_bound = (
                benchmark_item_mean - acceptable_st_devs * benchmark_item_st_dev
            )
            benchmark_upper_bound = (
                benchmark_item_mean + acceptable_st_devs * benchmark_item_st_dev
            )
            if v < benchmark_lower_bound:
                raise ValueError(
                    f"For key: {k} lower bound: {benchmark_lower_bound} surpassed by value {v}"
                )
            if v > benchmark_upper_bound:
                raise ValueError(
                    f"For key: {k} upper bound: {benchmark_upper_bound} surpassed by value {v}"
                )
