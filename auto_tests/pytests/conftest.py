import os
from itertools import repeat
from pathlib import Path
from typing import List, Tuple

import pytest

from auto_tests.auto_benchmarker_config import autobenchmarker
from auto_tests.definitions.auto_benchmarker import (
    AutoBenchmarker,
    BaseOutputData,
    BaseTestModel,
)
from auto_tests.definitions.pytest_config import PytestConfig

"""
Note: In future it would be nice to generate the benchmark from the user arg, 
and re-run all pytests in one step. pytest-lazy-fixture may facilitate this.

We would need to somehow generate a list of lazy fixtures of the right length,
as determined by the settings file, then each fixture would take the appropriate
test from the benchmark. Since the fixtures would not be called until after 
configuration they could be generated after benchmark generation.

There would need to be a lazily loaded list of benchmark data, only imported on
the first getitem.

@pytest.fixture(params=[
    lazy_fixture('benchmark_item_1'),
    lazy_fixture('benchmark_item_2')
])
def get_benchmark_item(i):
    def benchmark_item():
        return i
    return benchmark_item


benchmark_items = [
    pytest.fixture(get_benchmark_item(1), name = 'benchmark_item_1'),
    pytest.fixture(get_benchmark_item(2), name = 'benchmark_item_2')
]

"""


def get_string_id(output_data: Tuple[str, BaseOutputData]) -> str:
    func_name, data = output_data
    data_dict = data.dict()
    del data_dict["data"]
    string_out = "func: " + func_name + ", "
    for k, v in data_dict.items():
        string_part = str(k) + ": " + str(v)
        string_out += string_part + ", "
    return string_out


def listify_tests(
    tests: BaseTestModel, autobenchmarker: AutoBenchmarker
) -> List[Tuple[str, BaseOutputData]]:
    tests_list = []
    for func_name in autobenchmarker.func_benchmarkers.keys():
        attribute: List[BaseOutputData] = getattr(tests, func_name)
        tests_list += list(zip(repeat(func_name), attribute))
    return tests_list


test_data_model = autobenchmarker.test_model


def pytest_configure(config):
    generate_benchmark = config.option.genbenchmark
    if generate_benchmark:
        autobenchmarker.generate_benchmark(verbose=True)
        raise RuntimeError("Benchmark regenerated - results of pytest must be re-run")


pytest_config = PytestConfig.parse_file("pytest_config.json")


def list_of_tests():
    benchmark_file_path = Path(
        str(Path(pytest_config.benchmark_path)) + os.sep + "benchmark.json"
    )
    if not benchmark_file_path.exists():
        autobenchmarker.generate_benchmark(verbose=True)
    benchmark_file: test_data_model = test_data_model.parse_file(benchmark_file_path)
    return listify_tests(getattr(benchmark_file, "tests"), autobenchmarker)


list_of_t = list_of_tests()


@pytest.fixture(
    params=list_of_t,
    ids=[get_string_id(i) for i in list_of_t],
)
async def benchmark_data(request) -> BaseOutputData:
    return request.param


@pytest.fixture
async def benchmarker() -> AutoBenchmarker:
    return autobenchmarker
