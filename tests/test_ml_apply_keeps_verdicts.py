"""V5 Settings -> Apply may move SIGMA, never the linearity verdict.

2026-09-23: the bulk update graded status as 'both pass -> PASS, both fail -> FAIL, else
WARNING'. On the rebuilt work database that would have turned 1,701 linearity FAILs into
WARNING, taken all 235 ERRORs out of ERROR and turned 6,973 UNTRIMMED sweeps into WARNING.
"""
from datetime import datetime
from types import SimpleNamespace


def _db(tmp_path, monkeypatch):
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    db = mgr.DatabaseManager(tmp_path / "apply.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)
    return db


def _add(db, serial, overall, tracks):
    """tracks: (status, linearity_pass, sigma_pass, sigma_gradient)."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    when = datetime(2026, 5, 1)
    with db.session() as s:
        ar = DBAR(filename=f"M1-{serial}.xls", file_path=f"/f/M1/{serial}",
                  file_hash=f"M1{serial}".ljust(64, "0"), model="M1", serial=serial,
                  system=SystemType.A, file_date=when, timestamp=when,
                  overall_status=StatusType[overall], has_multi_tracks=len(tracks) > 1,
                  processing_time=0.1)
        s.add(ar); s.flush()
        for i, (st, lin, sig, grad) in enumerate(tracks):
            s.add(DBTR(analysis_id=ar.id, track_id=f"T{i+1}", status=StatusType[st],
                       linearity_pass=lin, sigma_pass=sig, sigma_gradient=grad))
        s.commit()
        return ar.id


def _apply(db, tmp_path, threshold):
    from laser_trim_analyzer.ml.manager import MLManager
    m = MLManager(db, ml_storage_path=tmp_path / "ml")
    m.trained_models = ["M1"]          # adapt to the real container type if it differs
    m.threshold_optimizers = {"M1": SimpleNamespace(is_calculated=True, threshold=threshold)}
    m.drift_detectors, m.predictors = {}, {}
    return m.apply_to_database(run_drift_detection=False)


def _state(db, aid):
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, TrackResult as DBTR
    with db.session() as s:
        a = s.get(DBAR, aid)
        tr = s.query(DBTR).filter(DBTR.analysis_id == aid).order_by(DBTR.track_id).all()
        return (a.overall_status.name,
                [(t.status.name, t.linearity_pass, t.sigma_pass) for t in tr])


def test_apply_moves_sigma_but_never_the_linearity_verdict(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    # threshold 0.5: sigma 0.1 passes, 0.9 fails
    lin_fail = _add(db, "s1", "FAIL", [("FAIL", False, True, 0.1)])
    watch = _add(db, "s2", "PASS", [("PASS", True, True, 0.9)])
    err = _add(db, "s3", "ERROR", [("ERROR", None, None, 999.999)])
    untrimmed = _add(db, "s4", "UNTRIMMED", [("UNTRIMMED", None, None, 0.1)])
    no_tracks = _add(db, "s5", "ERROR", [])
    mixed = _add(db, "s6", "WARNING", [("PASS", True, True, 0.1), ("UNTRIMMED", None, None, 0.9)])
    # A pre-Task-1 (b4f640f) ERROR row: linearity_pass is NOT NULL (legacy data
    # written before _create_failed_track started nulling it). The
    # linearity_pass.isnot(None) guard alone would not protect this row from
    # being re-graded to FAIL -- only status.notin_(_UNGRADED) does.
    legacy_err = _add(db, "s7", "ERROR", [("ERROR", False, True, 999.999)])

    _apply(db, tmp_path, threshold=0.5)

    assert _state(db, lin_fail) == ("FAIL", [("FAIL", False, True)])          # never WARNING
    assert _state(db, watch) == ("WARNING", [("WARNING", True, False)])       # sigma moved
    assert _state(db, err) == ("ERROR", [("ERROR", None, None)])              # untouched
    assert _state(db, untrimmed) == ("UNTRIMMED", [("UNTRIMMED", None, None)])
    assert _state(db, no_tracks)[0] == "ERROR"                                # never PASS
    assert _state(db, mixed) == ("PASS", [("PASS", True, True), ("UNTRIMMED", None, None)])
    assert _state(db, legacy_err) == ("ERROR", [("ERROR", False, True)])      # legacy row, never FAIL


def test_apply_writes_no_threshold_onto_an_ungraded_track_and_does_not_count_it(
        tmp_path, monkeypatch):
    """The threshold is a grading input. An ERROR track failed processing (its sigma is
    the analyser's 999.999 marker) and an UNTRIMMED sweep has no trim verdict, so neither
    is graded against it -- and neither may be stamped with it either, or counted in
    `updated` as if it had been. Until 2026-09-24 the first bulk UPDATE wrote
    sigma_threshold onto every track of the model and counted them all."""
    from laser_trim_analyzer.database.models import TrackResult as DBTR
    db = _db(tmp_path, monkeypatch)
    graded = _add(db, "s1", "PASS", [("PASS", True, True, 0.1)])
    warned = _add(db, "s2", "WARNING", [("WARNING", True, False, 0.9)])
    err = _add(db, "s3", "ERROR", [("ERROR", None, None, 999.999)])
    failed = _add(db, "s4", "ERROR", [("PROCESSING_FAILED", None, None, 999.999)])
    untrimmed = _add(db, "s5", "UNTRIMMED", [("UNTRIMMED", None, None, 0.1)])

    counts = _apply(db, tmp_path, threshold=0.5)

    with db.session() as s:
        thresholds = {t.analysis_id: t.sigma_threshold for t in s.query(DBTR).all()}
    assert thresholds[graded] == 0.5 and thresholds[warned] == 0.5
    assert thresholds[err] is None, "an ERROR track is never stamped with a threshold"
    assert thresholds[failed] is None, "nor a PROCESSING_FAILED one"
    assert thresholds[untrimmed] is None, "nor an UNTRIMMED sweep"
    assert counts["updated"] == 2, f"only the two graded tracks were updated: {counts}"
