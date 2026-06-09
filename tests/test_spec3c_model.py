"""Spec 3c — Model page. Foundations §3/§4.3. Fixtures in tests/conftest.py."""

# ---- Task 1: routing + column map ----------------------------------------

def test_consume_model_route_full(make_app):
    app = make_app()
    app.set_model_route("M1", "linearity_error")
    assert app.consume_model_route_full() == ("M1", "linearity_error")
    assert app.consume_model_route_full() == (None, None)


def test_consume_model_route_full_without_focus(make_app):
    app = make_app()
    app.set_model_route("M2")
    assert app.consume_model_route_full() == ("M2", None)


def test_track_metric_columns_public_and_linearity_maps_to_shifted():
    from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
    from laser_trim_analyzer.database.models import TrackResult as DBTR
    # Q4: the detector trains linearity_error on final_linearity_error_shifted; the UI must match.
    assert TRACK_METRIC_COLUMNS["linearity_error"] is DBTR.final_linearity_error_shifted
    assert "max_smoothness_value" not in TRACK_METRIC_COLUMNS  # lives on SmoothnessResult
