import scipy.stats
from pytest_trust_random import PytestConfig

pytest_config = PytestConfig.parse_file("pytest_config.json")

st_devs = pytest_config.acceptable_st_devs
re_runs = pytest_config.re_runs
independent_variables = 10
n_tests = 44
st_dev_prob = scipy.stats.norm(0, 1).cdf(-st_devs)
success_prob = 1 - st_dev_prob * 2
prob_of_one_test_fail = 1 - success_prob**independent_variables

prob_of_failing_all_re_runs = prob_of_one_test_fail ** (re_runs + 1)

prob_of_one_of_all_tests_failing = 1 - (1 - prob_of_one_test_fail) ** n_tests

prob_of_one_of_all_tests_failing_reruns = (
    1 - (1 - prob_of_failing_all_re_runs) ** n_tests
)

print("Fail probability per test (assuming no reruns): ", prob_of_one_test_fail)
print("Fail probability per test (assuming reruns): ", prob_of_failing_all_re_runs)
print(
    "Probability of one test failing: (assuming no reruns)",
    prob_of_one_of_all_tests_failing,
)
print(
    "Probability of one test failing: (assuming reruns)",
    prob_of_one_of_all_tests_failing_reruns,
)
