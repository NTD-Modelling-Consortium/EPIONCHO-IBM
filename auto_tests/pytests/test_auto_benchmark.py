import pytest

from auto_tests.definitions.auto_benchmarker import AutoBenchmarker
from auto_tests.pytests.conftest import pytest_config


@pytest.mark.flaky(reruns=pytest_config.re_runs)
@pytest.mark.asyncio
class TestGeneral:
    async def test_benchmark(self, benchmark_data, benchmarker: AutoBenchmarker):
        acceptable_st_devs = (
            pytest_config.acceptable_st_devs
        )  # TODO: Figure out where to put this?
        benchmarker.test_benchmark_data(
            benchmark_data=benchmark_data, acceptable_st_devs=acceptable_st_devs
        )
