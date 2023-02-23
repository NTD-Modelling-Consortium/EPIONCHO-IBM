from epioncho_ibm import EndgameSimulation, EpionchoEndgameModel

# This is currently ignoring non compliant percent
endgame = EpionchoEndgameModel.parse_file("./scenario1.json")
simulation = EndgameSimulation(
    start_time=2015, endgame=endgame, verbose=True, debug=True
)
simulation.run(end_time=2040)
