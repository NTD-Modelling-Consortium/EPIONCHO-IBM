import pytest

from epioncho_ibm import HumanParams, Params, Simulation, StateStats


@pytest.mark.asyncio
class TestGeneral:
    async def test_one_skin_snip(self):
        params = Params(humans=HumanParams(skin_snip_number=1))
        simulation = Simulation(start_time=0, params=params, n_people=10)
        simulation.state.microfilariae_per_skin_snip()

    async def test_output_stats(self):
        simulation = Simulation(start_time=0, params=Params(), n_people=10)
        output = [
            (s.current_time, s.stats())
            for s in simulation.iter_run(end_time=0.1, sampling_interval=0.1)
        ]
        assert len(output) == 1
        item = output[0]
        assert len(item) == 2
        assert item[0] == 0
        assert isinstance(item[1], StateStats)
