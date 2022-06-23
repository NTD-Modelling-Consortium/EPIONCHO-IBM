from auto_tests.definitions.auto_benchmarker import AutoBenchmarker
from epioncho_ibm import StateStats, benchmarker_test_func

benchmarker = AutoBenchmarker(no_treatment=benchmarker_test_func)
mod = benchmarker.generate_benchmark(verbose=True)
