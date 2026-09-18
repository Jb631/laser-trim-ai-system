from laser_trim_analyzer.core.models import AnalysisStatus, TrackData


def _track(**kw):
    base = dict(track_id="TRK1", status=AnalysisStatus.PASS,
                travel_length=55.0, linearity_spec=0.01)
    base.update(kw)
    return TrackData(**base)


def test_trim_passes_defaults_to_empty_so_existing_callers_are_unaffected():
    assert _track().trim_passes == []


def test_trim_passes_round_trips():
    t = _track(trim_passes=[{"pass_index": 1, "label": "TRM1",
                             "positions": [0.0], "errors": [0.1]}])
    assert t.trim_passes[0]["label"] == "TRM1"


def test_analysis_result_setup_defaults_to_none():
    from laser_trim_analyzer.core.models import AnalysisResult, FileMetadata, SystemType
    from datetime import datetime
    r = AnalysisResult(
        metadata=FileMetadata(filename="x.xls", file_path="x.xls", model="8232-1",
                              serial="1", system=SystemType.B, file_date=datetime.now()),
        overall_status=AnalysisStatus.PASS, processing_time=0.1, tracks=[_track()])
    assert r.trim_setup is None
