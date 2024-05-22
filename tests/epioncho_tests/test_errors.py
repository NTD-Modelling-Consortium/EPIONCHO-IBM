import pytest

from epioncho_ibm import Params, Simulation, TreatmentParams


@pytest.mark.asyncio
class TestGeneral:
    async def test_start_before_end(self):
        simulation = Simulation(start_time=10, params=Params(n_people=10))
        with pytest.raises(ValueError, match="End time 0 before start 10"):
            simulation.run(end_time=0)

    async def test_start_before_end_iter_run(self):
        simulation = Simulation(start_time=10, params=Params(n_people=10))
        with pytest.raises(ValueError, match="End time 0 before start 10"):
            next(simulation.iter_run(end_time=0, sampling_interval=1))

    async def test_set_n_people(self):
        simulation = Simulation(start_time=10, params=Params(n_people=10))

        with pytest.raises(
            AttributeError,
            match="can't set attribute 'n_people'",
        ):
            simulation.state.n_people = 4

    async def test_set_params(self):
        simulation = Simulation(start_time=0, params=Params(n_people=10))
        simulation.reset_current_params(Params(n_people=10))

    async def test_set_sub_params(self):
        simulation = Simulation(start_time=0, params=Params(n_people=10))
        with pytest.raises(
            TypeError,
            match='"ImmutableParams" is immutable and does not support item assignment',
        ):
            simulation.state._params.delta_time_days = 0.1

    async def test_set_sub_sub_params(self):
        simulation = Simulation(start_time=0, params=Params(n_people=10))
        with pytest.raises(
            TypeError,
            match='"ImmutableHumanParams" is immutable and does not support item assignment',
        ):
            simulation.state._params.humans.max_human_age = 80
