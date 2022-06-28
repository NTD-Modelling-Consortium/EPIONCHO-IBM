import numpy as np
import pytest
from attr import s

from epioncho_ibm import Params, State
from epioncho_ibm.params import TreatmentParams
from epioncho_ibm.worms import WormGroup, check_no_worms_are_negative


@pytest.mark.asyncio
class TestGeneral:
    async def test_start_before_end(self):
        state = State(params=Params(), n_people=10)
        with pytest.raises(ValueError, match="End time after start"):
            state.run_simulation(start_time=10, end_time=0)

    async def test_start_before_end_output_stats(self):
        state = State(params=Params(), n_people=10)
        with pytest.raises(ValueError, match="End time after start"):
            state.run_simulation_output_stats(
                sampling_interval=1, start_time=10, end_time=0
            )

    async def test_set_n_people(self):
        state = State(params=Params(), n_people=10)

        with pytest.raises(
            ValueError,
            match="Setting n_people in state not allowed. Please re-initialise state.",
        ):
            state.n_people = 4

    async def test_set_params(self):
        state = State(params=Params(), n_people=10)
        state.params = Params()

    async def test_set_sub_params(self):
        state = State(params=Params(), n_people=10)
        with pytest.raises(
            ValueError, match="Cannot alter inner values of params in-place"
        ):
            state.params.delta_time = 0.1

    async def test_set_sub_sub_params(self):
        state = State(params=Params(), n_people=10)
        with pytest.raises(
            ValueError, match="Cannot alter inner values of params in-place"
        ):
            state.params.humans.max_human_age = 80

    async def test_negative_worms(self):
        with pytest.raises(RuntimeError, match="Worms became negative"):
            check_no_worms_are_negative(
                WormGroup(
                    male=np.zeros(3, dtype=int) - 1,
                    infertile=np.zeros(3, dtype=int),
                    fertile=np.zeros(3, dtype=int),
                )
            )


@pytest.mark.asyncio
class TestDerivedParams:
    async def test_treament_params_invalid_step(self):
        with pytest.raises(
            ValueError,
            match="Treatment times could not be found for start: 0.0, stop: 10.0, interval: 3.0",
        ):
            State(
                params=Params(
                    treatment=TreatmentParams(
                        start_time=0, stop_time=10, interval_years=3
                    )
                ),
                n_people=10,
            )
