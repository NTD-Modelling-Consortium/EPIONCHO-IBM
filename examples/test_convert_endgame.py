from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from endgame_simulations import apply_incremental_param_changes
from endgame_simulations.models import EndgameModel, create_update_model

from epioncho_ibm.state import Params
from epioncho_ibm.state.params import EndgameParams, EndgameProgramParams, TreatmentParams


EpionchoEndgameModel = EndgameModel[EndgameParams, create_update_model(EndgameParams), EndgameProgramParams]
EpionchoEndgameModel.__name__ = 'EpionchoEndgameModel'

@dataclass
class ParamsAtTime:
    # TODO: rubbish name, maybe just stick to a tuple[float, params]...
    time: float
    params: Params


def _time_from_year_and_month(year: int, month: int) -> float:
    return year + (month - 1) / 12


class ReasonForChange(Enum):
    # NOTE: the ordering is very important
    PARAMS_CHANGE = 1
    TREATMENT_ENDS = 2
    TREATMENT_STARTS = 3


def _times_of_change(
    model: EpionchoEndgameModel,
) -> list[tuple[float, ReasonForChange]]:
    """Generates times of changes with the event (either due to params or program change).

    The result may contain some timestamps twice - that means that both params and
    treatment changes at the same time."""
    changes: list[tuple[float, ReasonForChange]] = [
        (0.0, ReasonForChange.PARAMS_CHANGE)
    ]

    for change in model.parameters.changes:
        changes.append(
            (
                _time_from_year_and_month(change.year, change.month),
                ReasonForChange.PARAMS_CHANGE,
            )
        )

    for program in model.programs:
        start = _time_from_year_and_month(program.first_year, program.first_month)
        changes.append((start, ReasonForChange.TREATMENT_STARTS))
        if program.last_year:
            assert program.last_month
            end = _time_from_year_and_month(program.last_year, program.last_month)
            assert start < end
            changes.append((end, ReasonForChange.TREATMENT_ENDS))

    changes.sort()
    return changes


def endgame_to_params(model: EpionchoEndgameModel) -> list[ParamsAtTime]:
    def _params_over_time(model: EpionchoEndgameModel) -> Iterator[EndgameParams]:
        current = model.parameters.initial
        yield current
        for change in model.parameters.changes:
            # TODO: apply_incr.... should maybe take a union of list and single one?
            current = apply_incremental_param_changes(current, [change])
            yield current

    params_over_time = _params_over_time(model)
    programs = iter(model.programs or [])

    params: list[ParamsAtTime] = []

    for time_of_change, reason in _times_of_change(model):
        if reason == ReasonForChange.PARAMS_CHANGE:
            params.append(
                ParamsAtTime(time=time_of_change, params=Params.parse_obj(next(params_over_time).dict()) )
            )
        elif reason == ReasonForChange.TREATMENT_ENDS:
            # for Epioncho, we have to keep params.treatment in order to model
            # the effects of the old treatment. From this reason, we don't do
            # anything here, just assert that treatment is set already
            assert params and params[-1].params.treatment
        elif reason == ReasonForChange.TREATMENT_STARTS:
            assert params
            current_params = params[-1]
            current_treatment = current_params.params.treatment
            assert (
                not current_treatment
                or current_treatment.stop_time < time_of_change
            ), "Overlapping treatment found!"

            program = next(programs)

            # TODO: what about changing other parameters? It was decided that they can't change I suppose...
            assert not isinstance(program.interventions, list)
            treatment = TreatmentParams(
                interval_years=program.interventions.treatment_interval,
                start_time=time_of_change,
                stop_time=_time_from_year_and_month(
                    program.last_year, program.last_month
                ),
            )

            if current_params.time == time_of_change:
                # treatment starts at the same time as params
                params_to_modify = current_params
            else:
                assert time_of_change > current_params.time
                # we need to create new Params
                params.append(
                    ParamsAtTime(
                        time=time_of_change, params=current_params.params.copy()
                    )
                )
                params_to_modify = params[-1]

            params_to_modify.params.treatment = treatment
        else:
            raise ValueError("Unsupported reason")

    return params
