import pytest

from tests.benchmark_data_types import OutputData, TestData

benchmark_file = TestData.parse_file("benchmark.json")


@pytest.fixture(
    params=benchmark_file.tests,
    ids=[
        f"year: {str(i.end_year)} pop: {str(i.params.human_population)}"
        for i in benchmark_file.tests
    ],
)
async def benchmark_data(request) -> OutputData:
    return request.param
