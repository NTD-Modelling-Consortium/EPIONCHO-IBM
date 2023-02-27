from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

from endgame_simulations import apply_incremental_param_changes
from endgame_simulations.endgame_simulation import GenericEndgame

from epioncho_ibm.simulation import Simulation
from epioncho_ibm.state import Params, State
from epioncho_ibm.state.params import (
    EndgameParams,
    EpionchoEndgameModel,
    TreatmentParams,
)


@dataclass
class ParamChange:
    time: float
    params: Params


def _time_from_year_and_month(year: int, month: int, is_last: bool) -> float:
    if is_last:
        return year + (month) / 12
    else:
        return year + (month - 1) / 12


class ReasonForChange(IntEnum):
    # NOTE: the ordering is very important
    PARAMS_CHANGE = 1
    TREATMENT_ENDS = 2
    TREATMENT_STARTS = 3


def _times_of_change(
    endgame: EpionchoEndgameModel,
) -> list[tuple[float, ReasonForChange]]:
    """Generates times of changes with the event (either due to params or program change).

    The result may contain some timestamps twice - that means that both params and
    treatment changes at the same time."""
    changes: list[tuple[float, ReasonForChange]] = [
        (0.0, ReasonForChange.PARAMS_CHANGE)
    ]

    for change in endgame.parameters.changes:
        changes.append(
            (
                _time_from_year_and_month(change.year, change.month, is_last=False),
                ReasonForChange.PARAMS_CHANGE,
            )
        )

    for program in endgame.programs:
        start = _time_from_year_and_month(
            program.first_year, program.first_month, is_last=False
        )
        changes.append((start, ReasonForChange.TREATMENT_STARTS))
        if program.last_year:
            assert program.last_month
            end = _time_from_year_and_month(
                program.last_year, program.last_month, is_last=True
            )
            assert start < end
            changes.append((end, ReasonForChange.TREATMENT_ENDS))

    changes.sort()

    # TODO: If you don't like having potentially the same time for PARAMS_CHANGE and TREATMENT_STARTS
    # and relying on the ordering of these in the loop below, PARAMS_CHANGE and TREATMENT_STARTS could be
    # squashed together after sorting into PARAMS_CHANGE_WITH_TREATMENT or (not squashed)
    # PARAMS_CHANGE_WITHOUT_TREATMENT.

    return changes


def endgame_to_params(endgame: EpionchoEndgameModel) -> list[tuple[float, Params]]:
    def _params_over_time(model: EpionchoEndgameModel) -> Iterator[EndgameParams]:
        current = model.parameters.initial
        yield current
        for change in model.parameters.changes:
            current = apply_incremental_param_changes(current, change)
            yield current

    params_over_time = _params_over_time(endgame)
    programs = iter(endgame.programs or [])

    params: list[ParamChange] = []
    for time_of_change, reason in _times_of_change(endgame):
        if reason == ReasonForChange.PARAMS_CHANGE:
            new_params = Params.parse_obj(next(params_over_time).dict())
            if params:
                new_params.treatment = params[-1].params.treatment
            params.append(
                ParamChange(
                    time=time_of_change,
                    params=new_params,
                )
            )
        elif reason == ReasonForChange.TREATMENT_STARTS:
            assert params
            current_params = params[-1]
            current_treatment = current_params.params.treatment
            assert (
                not current_treatment or current_treatment.stop_time <= time_of_change
            ), f"Overlapping treatment found! \nCurrent_treatment: {current_treatment} \nStop time: {current_treatment.stop_time} \ntime_of_change: {time_of_change}"

            program = next(programs)

            assert not isinstance(program.interventions, list)

            treatment_dict = program.interventions.dict()
            interval_years = treatment_dict.pop("treatment_interval")
            treatment = TreatmentParams(
                **treatment_dict,
                interval_years=interval_years,
                start_time=time_of_change,
                stop_time=_time_from_year_and_month(
                    program.last_year, program.last_month, is_last=True
                ),
            )

            if current_params.time != time_of_change:
                assert time_of_change > current_params.time
                # we need to create new Params
                params.append(
                    ParamChange(
                        time=time_of_change, params=current_params.params.copy()
                    )
                )
            params[-1].params.treatment = treatment
        elif reason == ReasonForChange.TREATMENT_ENDS:
            # for Epioncho, we have to keep params.treatment in order to model
            # the effects of the old treatment. From this reason, we don't do
            # anything here, just assert that treatment is set already
            assert params and params[-1].params.treatment
        else:
            raise ValueError("Unsupported reason")

    return [(p.time, p.params) for p in params]


class EndgameSimulation(
    GenericEndgame[EpionchoEndgameModel, Simulation, State, Params],
    combined_params_model=Params,
    simulation_class=Simulation,
    convert_endgame=endgame_to_params,
):
    pass
