import benchmark.constants as constants
import numpy as np
import pytest


# input the proper endtimes and populations for the test
def pytest_generate_tests(metafunc):
    if "end_time" in metafunc.fixturenames:
        metafunc.parametrize("end_time", constants.END_TIMES)
    if "population" in metafunc.fixturenames:
        metafunc.parametrize("population", constants.POPULATIONS)


# set the re-run parameter for all tests
def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.flaky(reruns=constants.RE_RUNS))
