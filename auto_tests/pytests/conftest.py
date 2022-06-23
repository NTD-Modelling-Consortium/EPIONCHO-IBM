import os
from itertools import repeat
from pathlib import Path
from typing import List, Tuple

import pytest

from auto_tests.definitions.auto_benchmarker import (
    AutoBenchmarker,
    BaseOutputData,
    BaseTestModel,
)
from auto_tests.definitions.pytest_config import PytestConfig
from epioncho_ibm import benchmarker_test_func

pytest_config = PytestConfig.parse_file("pytest_config.json")
benchmark_file_path = Path(
    str(Path(pytest_config.benchmark_path)) + os.sep + "benchmark.json"
)
autobenchmarker = AutoBenchmarker(no_treatment=benchmarker_test_func)

if not benchmark_file_path.exists():
    autobenchmarker.generate_benchmark(verbose=True)

test_data_model = autobenchmarker.test_model
benchmark_file: test_data_model = test_data_model.parse_file(benchmark_file_path)


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


tests_list_form = listify_tests(getattr(benchmark_file, "tests"), autobenchmarker)


@pytest.fixture(
    params=tests_list_form,
    ids=[get_string_id(i) for i in tests_list_form],
)
async def benchmark_data(request) -> BaseOutputData:
    return request.param


@pytest.fixture
async def benchmarker() -> AutoBenchmarker:
    return autobenchmarker
