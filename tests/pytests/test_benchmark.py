import pytest

from epioncho_ibm import State
from tests.definitions.utils import flatten_dict
from tests.pytests.conftest import pytest_config


@pytest.mark.flaky(reruns=pytest_config.re_runs)
@pytest.mark.asyncio
class TestGeneral:
    async def test_benchmark(self, benchmark_data):
        acceptable_st_devs = (
            pytest_config.acceptable_st_devs
        )  # TODO: Figure out where to put this?

        params = benchmark_data.params
        end_time = benchmark_data.end_year
        state = State(params=params)
        state.dist_population_age(num_iter=15000)
        state.run_simulation(start_time=0, end_time=end_time)

        people_stats = state.to_stats()
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
