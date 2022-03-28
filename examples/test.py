from epioncho_ibm import Params, RandomConfig, State, advance_state

random_config = RandomConfig()
params = Params()
initial_state = State.generate_random(n_people=10, random_config=random_config)

new_state = advance_state(initial_state, params=params, n_iters=100)

print(new_state.prevelence())
print(new_state.microfilariae_per_skin_snip())
