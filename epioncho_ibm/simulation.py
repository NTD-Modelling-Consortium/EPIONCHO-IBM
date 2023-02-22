from endgame_simulations.simulations import GenericSimulation

from epioncho_ibm.advance import advance_state
from epioncho_ibm.state import Params, State


class Simulation(
    GenericSimulation[Params, State], state_class=State, advance_state=advance_state
):
    @property
    def _delta_time(self):
        return self.state._params.delta_time
