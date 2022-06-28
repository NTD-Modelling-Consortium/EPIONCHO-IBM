from epioncho_ibm import StateStats, benchmarker_test_func
from tests.definitions.auto_benchmarker import AutoBenchmarker

autobenchmarker = AutoBenchmarker(
    no_treatment=benchmarker_test_func, treatment=benchmarker_test_func
)
mod = autobenchmarker.generate_benchmark(verbose=True)
