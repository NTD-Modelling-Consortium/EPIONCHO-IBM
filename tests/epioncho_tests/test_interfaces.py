import pytest

from epioncho_ibm import HumanParams, Params, State, StateStats
from epioncho_ibm.state import make_state_from_params


@pytest.mark.asyncio
class TestGeneral:
    async def test_one_skin_snip(self):
        state = make_state_from_params(
            params=Params(humans=HumanParams(skin_snip_number=1)), n_people=10
        )
        state.microfilariae_per_skin_snip()

    async def test_output_stats(self):
        params = Params()
        state = make_state_from_params(params=params, n_people=10)
        output = state.run_simulation_output_stats(
            sampling_interval=0.1, start_time=0, end_time=0.1
        )
        assert len(output) == 1
        item = output[0]
        assert len(item) == 2
        assert item[0] == 0
        assert isinstance(item[1], StateStats)

    async def test_verbose_output_stats(self, capfd):
        params = Params()
        state = make_state_from_params(params=params, n_people=10)
        state.run_simulation_output_stats(
            sampling_interval=0.1, start_time=0, end_time=0.3, verbose=True
        )
        out, err = capfd.readouterr()
        assert out == "0\n0.202739726027397\n"
