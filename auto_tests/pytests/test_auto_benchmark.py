from typing import Tuple

import pytest

from auto_tests.definitions.auto_benchmarker import AutoBenchmarker, BaseOutputData
from auto_tests.pytests.conftest import pytest_config


@pytest.mark.flaky(reruns=pytest_config.re_runs)
@pytest.mark.asyncio
class TestGeneral:
    async def test_benchmark(
        self, benchmark_data: Tuple[str, BaseOutputData], benchmarker: AutoBenchmarker
    ):
        acceptable_st_devs = (
            pytest_config.acceptable_st_devs
        )  # TODO: Figure out where to put this?
        func_name, data = benchmark_data
        benchmarker.test_benchmark_data(
            benchmark_data=data,
            acceptable_st_devs=acceptable_st_devs,
            func_name=func_name,
        )
