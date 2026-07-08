"""WARNING status recompute (M4, 2026-07-07).

Re-grades overall_status from stored track pass flags under the current rule:
track = FAIL if linearity_pass False, else PASS if sigma_pass True, else
WARNING; analysis = all-PASS -> PASS, any-FAIL -> FAIL, else WARNING.
NULL linearity_pass rows are skipped (Fix Missing Tracks population).
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _add(db, model, serial, overall, tracks, when=datetime(2026, 5, 1)):
    """tracks: list of (track_status, linearity_pass, sigma_pass)."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    with db.session() as s:
        ar = DBAR(filename=f"{model}-{serial}.xls", file_path=f"/f/{model}/{serial}",
                  file_hash=f"{model}{serial}".ljust(64, "0"), model=model,
                  serial=serial, system=SystemType.A, file_date=when, timestamp=when,
                  overall_status=StatusType[overall], has_multi_tracks=len(tracks) > 1,
                  processing_time=0.1)
        s.add(ar); s.flush()
        for i, (ts, lin, sig) in enumerate(tracks):
            s.add(DBTR(analysis_id=ar.id, track_id=f"T{i+1}", status=StatusType[ts],
                       linearity_pass=lin, sigma_pass=sig))
        s.commit()
        return ar.id


def _status_of(db, aid):
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR
    with db.session() as s:
        return s.query(DBAR.overall_status).filter(DBAR.id == aid).scalar().name


def test_recompute_rules_and_dry_run(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "r.db")
    # The misclassification this exists for: linearity-FAIL labeled WARNING.
    a1 = _add(db, "M1", "s1", "WARNING", [("FAIL", False, True)])
    # Both-pass labeled WARNING -> should become PASS.
    a2 = _add(db, "M1", "s2", "WARNING", [("PASS", True, True)])
    # Correctly-labeled WARNING (lin pass, sigma fail) -> unchanged.
    a3 = _add(db, "M1", "s3", "WARNING", [("WARNING", True, False)])
    # NULL linearity flag -> skipped, even though labeled WARNING.
    a4 = _add(db, "M1", "s4", "WARNING", [("WARNING", None, None)])
    # UNTRIMMED analysis -> untouched.
    a5 = _add(db, "M1", "s5", "UNTRIMMED", [("UNTRIMMED", None, None)])
    # Multi-track: one FAIL among passes -> FAIL; UNTRIMMED track ignored.
    a6 = _add(db, "M1", "s6", "PASS",
              [("PASS", True, True), ("FAIL", False, True), ("UNTRIMMED", None, None)])

    preview = db.recompute_overall_statuses(dry_run=True)
    assert preview["changed"] == 3            # a1, a2, a6
    assert preview["skipped_null_flags"] == 1  # a4
    assert preview["transitions"] == {
        "WARNING->FAIL": 1, "WARNING->PASS": 1, "PASS->FAIL": 1}
    # Dry run must not write.
    assert _status_of(db, a1) == "WARNING"

    res = db.recompute_overall_statuses(dry_run=False)
    assert res["changed"] == 3
    assert _status_of(db, a1) == "FAIL"
    assert _status_of(db, a2) == "PASS"
    assert _status_of(db, a3) == "WARNING"
    assert _status_of(db, a4) == "WARNING"   # skipped
    assert _status_of(db, a5) == "UNTRIMMED"
    assert _status_of(db, a6) == "FAIL"

    # Idempotent: second run changes nothing.
    again = db.recompute_overall_statuses(dry_run=True)
    assert again["changed"] == 0


def test_scale_anomaly_ingest_guard():
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.core.models import TrackData, AnalysisStatus

    def track(lin_err, band=0.05):
        return TrackData(
            track_id="T1", status=AnalysisStatus.PASS, linearity_spec=band,
            travel_length=12.0,
            position_data=list(range(12)), error_data=[0.01] * 12,
            upper_limits=[band] * 12, lower_limits=[-band] * 12,
            linearity_error=lin_err, linearity_pass=True,
        )

    # 10.007 against a ±0.05 band (the production case) -> flagged.
    issues = Processor._validate_track_data([track(10.007)])
    assert any("scale-anomalous" in i for i in issues), issues
    # Real-but-bad error inside 10x band -> NOT flagged (that's a FAIL, not junk).
    assert not any("scale-anomalous" in i
                   for i in Processor._validate_track_data([track(0.3)]))
