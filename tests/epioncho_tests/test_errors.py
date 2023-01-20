import pytest

from epioncho_ibm import Params, Simulation, TreatmentParams


@pytest.mark.asyncio
class TestGeneral:
    async def test_start_before_end(self):
        simulation = Simulation(start_time=10, params=Params(n_people=10))
        with pytest.raises(ValueError, match="End time before start"):
            simulation.run(end_time=0)

    async def test_start_before_end_iter_run(self):
        simulation = Simulation(start_time=10, params=Params(n_people=10))
        with pytest.raises(ValueError, match="End time before start"):
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


@pytest.mark.asyncio
class TestDerivedParams:
    async def test_treament_params_invalid_step(self):
        with pytest.raises(
            ValueError,
            match="Treatment times could not be found for start: 0.0, stop: 10.0, interval: 3.0",
        ):
            params = Params(
                treatment=TreatmentParams(start_time=0, stop_time=10, interval_years=3),
                n_people=10,
            )
            Simulation(start_time=0, params=params)
