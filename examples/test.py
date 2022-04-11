from epioncho_ibm import Params, RandomConfig, State, advance_state

random_config = RandomConfig()
params = Params()
initial_state = State.generate_random(n_people=100, random_config=random_config)

initial_state.dist_population_age(num_iter=15000)
compliantCount = 0
maleCount = 0
totalAge = 0
for person in initial_state._people:
    if person.compliant:
        compliantCount += 1
    if person.sex.value == "male":
        maleCount += 1
    totalAge += person.age

print("Average age: " + str(totalAge / 100))
print("Male Percentage: " + str(maleCount))
print("Compliant Percentage: " + str(compliantCount))


# new_state = advance_state(initial_state, params=params, n_iters=100)

# print(new_state.prevelence())
# print(new_state.microfilariae_per_skin_snip())
