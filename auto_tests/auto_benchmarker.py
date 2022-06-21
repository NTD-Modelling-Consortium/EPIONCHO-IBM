from auto_tests.definitions.auto_benchmarker import AutoBenchmarker
from epioncho_ibm import StateStats, benchmarker_test_func
from epioncho_ibm.state import StateStats

benchmarker = AutoBenchmarker[StateStats](benchmarker_test_func)
mod = benchmarker.generate_benchmark(verbose=True)
