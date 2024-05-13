from epioncho_ibm.state.derived_params import DerivedParams
from epioncho_ibm.state.params import Params, TreatmentParams


def test_treatment_times_uneven_spacing():
    params = Params(
        treatment=TreatmentParams(start_time=0, stop_time=10, interval_years=3),
        n_people=10,
        delta_time_days=0,
    )
    derived_params = DerivedParams(params, current_time=0)
    assert derived_params.treatment_times is not None
    assert derived_params.treatment_times.tolist() == [0, 3, 6, 9]
