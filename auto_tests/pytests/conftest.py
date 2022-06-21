import os
from pathlib import Path

import pytest

from auto_tests.definitions.auto_benchmarker import AutoBenchmarker, BaseOutputData
from auto_tests.definitions.pytest_config import PytestConfig
from epioncho_ibm import benchmarker_test_func

pytest_config = PytestConfig.parse_file("pytest_config.json")
benchmark_file_path = Path(
    str(Path(pytest_config.benchmark_path)) + os.sep + "benchmark.json"
)
autobenchmarker = AutoBenchmarker(benchmarker_test_func)

if not benchmark_file_path.exists():
    autobenchmarker.generate_benchmark(verbose=True)

test_data_model = autobenchmarker.test_model
benchmark_file: test_data_model = test_data_model.parse_file(benchmark_file_path)


@pytest.fixture(
    params=benchmark_file.tests,  # type:ignore
    # ids=[
    #    f"year: {str(i.end_year)} pop: {str(i.params.human_population)}"
    #    for i in benchmark_file.tests
    # ],
)
async def benchmark_data(request) -> BaseOutputData:
    return request.param


@pytest.fixture
async def benchmarker() -> AutoBenchmarker:
    return autobenchmarker
