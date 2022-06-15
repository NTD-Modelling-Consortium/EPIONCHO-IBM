import os
from pathlib import Path

import pytest

from tests.definitions.benchmark_data_types import OutputData, TestData
from tests.definitions.pytest_config import PytestConfig

pytest_config = PytestConfig.parse_file("pytest_config.json")
benchmark_file_path = (
    str(Path(pytest_config.benchmark_path)) + os.sep + "benchmark.json"
)

benchmark_file = TestData.parse_file(benchmark_file_path)


@pytest.fixture(
    params=benchmark_file.tests,
    ids=[
        f"year: {str(i.end_year)} pop: {str(i.params.human_population)}"
        for i in benchmark_file.tests
    ],
)
async def benchmark_data(request) -> OutputData:
    return request.param
