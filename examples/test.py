import numpy as np

from epioncho_ibm import Params, RandomConfig, State, run_simulation

random_config = RandomConfig()
params = Params(human_population=440, treatment_start_time=0)
initial_state = State.generate_random(random_config=random_config, params=params)

initial_state.dist_population_age(num_iter=15000)


male_count = np.sum(initial_state.people.sex_is_male) / len(initial_state.people)
compliant_count = np.sum(initial_state.people.compliance) / len(initial_state.people)
average_age = np.average(initial_state.people.ages)


print("Average age: " + str(average_age))
print("Male Percentage: " + str(male_count))
print("Compliant Percentage: " + str(compliant_count))

new_state = run_simulation(initial_state, start_time=0, end_time=10, verbose=True)
for i in range(12):
    new_state = run_simulation(
        new_state, start_time=(i + 1) * 10, end_time=(i + 2) * 10, verbose=True
    )
    print("run", str(i))
    print(new_state.microfilariae_per_skin_snip())
    print(new_state.mf_prevalence_in_population())
# new_state = advance_state(initial_state, params=params, n_iters=100)

# print(new_state.prevelence())
# print(new_state.microfilariae_per_skin_snip())
