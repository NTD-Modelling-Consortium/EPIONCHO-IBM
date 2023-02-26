import h5py

from epioncho_ibm import EndgameSimulation, EpionchoEndgameModel


def get_endgame(seed, cov):
    return {
        "parameters": {"initial": {"n_people": 100, "seed": seed}, "changes": []},
        "programs": [],
    }


param_sets = [{"seed": 1, "cov": 0.5}, {"seed": 2, "cov": 0.6}, {"seed": 3, "cov": 0.7}]
new_file = h5py.File("test_one.hdf5", "w")


endgame_structures = [
    get_endgame(param_set["seed"], param_set["cov"]) for param_set in param_sets
]


if __name__ == "__main__":
    for i, param_set in enumerate(param_sets):
        endgame_structure = get_endgame(param_set["seed"], param_set["cov"])
        endgame = EpionchoEndgameModel.parse_obj(endgame_structure)
        endgame_sim = EndgameSimulation(
            start_time=2020, endgame=endgame, verbose=False, debug=True
        )
        endgame_sim.run(end_time=2021)
        grp = new_file.create_group(f"draw_{str(i)}")
        endgame_sim.save(grp)
