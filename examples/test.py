from epioncho_ibm import Params, RandomConfig, State, advance_state
import numpy as np
random_config = RandomConfig()
params = Params()
initial_state = State.generate_random(random_config=random_config, params = params, n_people=100)

initial_state.dist_population_age(num_iter=15000)


male_count = np.sum(initial_state._people.sex_is_male)
compliant_count = np.sum(initial_state._people.compliance)
average_age = np.average(initial_state._people.ages)


print("Average age: " + str(average_age))
print("Male Percentage: " + str(male_count))
print("Compliant Percentage: " + str(compliant_count))


# new_state = advance_state(initial_state, params=params, n_iters=100)

# print(new_state.prevelence())
# print(new_state.microfilariae_per_skin_snip())
