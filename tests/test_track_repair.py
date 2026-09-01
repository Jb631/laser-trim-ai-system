"""Fix Missing Tracks — detection rules and the unreachable-source contract.

Two shapes are "missing tracks", and both are repaired by re-parsing:
  * zero track rows;
  * every track row array-less while the parent claims a gradeable verdict.

UNTRIMMED and ERROR parents are deliberately NOT missing-tracks: an untrimmed
test sweep has no trim result to store (the parser moves its sweep into the
untrimmed_* columns and clears the trimmed arrays), and an ERROR row never
measured anything. On the work database those account for ~5,666 of the ~5,811
array-less track rows — flagging them would bury real defects and send the
repair tool re-parsing thousands of files to no effect.

The trap these tests exist for: SafeJSON writes Python None through
SQLAlchemy's JSON type, which stores the JSON literal 'null' — TEXT, not SQL
NULL. Every `IS NULL` detector misses those rows entirely.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _add_analysis(db, serial, overall, tracks, file_path=None):
    """tracks: list of (track_status, position_data, error_data)."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    when = datetime(2026, 5, 1)
    with db.session() as s:
        ar = DBAR(filename=f"M1-{serial}.xls",
                  file_path=file_path or f"/nowhere/M1/{serial}.xls",
                  file_hash=f"h{serial}".ljust(64, "0"), model="M1", serial=serial,
                  system=SystemType.A, file_date=when, timestamp=when,
                  overall_status=StatusType[overall],
                  has_multi_tracks=len(tracks) > 1, processing_time=0.1)
        s.add(ar); s.flush()
        for i, (ts, pos, err) in enumerate(tracks):
            s.add(DBTR(analysis_id=ar.id, track_id=f"TRK{i + 1}",
                       status=StatusType[ts], position_data=pos, error_data=err))
        s.commit()
        return ar.id


def _detected_ids(db):
    return {r["id"] for r in db.get_trim_records_missing_tracks(linked_only=False)}


def test_none_arrays_are_stored_as_the_json_literal_null(tmp_path):
    """The premise of every other test here: None does NOT become SQL NULL."""
    from sqlalchemy import text
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "t.db")
    _add_analysis(db, "s1", "WARNING", [("UNTRIMMED", None, None)])
    with db.session() as s:
        raw = s.execute(text(
            "SELECT typeof(position_data), CAST(position_data AS TEXT), "
            "position_data IS NULL FROM track_results")).fetchone()
    assert raw[0] == "text" and raw[1] == "null", raw
    assert raw[2] == 0, "stored as SQL NULL — the 'null' literal trap is gone"


def test_gradeable_record_with_all_null_arrays_is_detected(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "t.db")
    warn_id = _add_analysis(db, "s1", "WARNING", [("WARNING", None, None)])
    fail_id = _add_analysis(db, "s2", "FAIL",
                            [("FAIL", None, None), ("FAIL", None, None)])
    pass_id = _add_analysis(db, "s3", "PASS", [("PASS", None, None)])

    assert _detected_ids(db) == {warn_id, fail_id, pass_id}


def test_untrimmed_and_error_parents_are_not_detected(tmp_path):
    """Array-lessness is expected for these — by design, not a defect."""
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "t.db")
    _add_analysis(db, "s1", "UNTRIMMED", [("UNTRIMMED", None, None)])
    _add_analysis(db, "s2", "ERROR", [("ERROR", None, None)])
    # The real work-database shape: a 2-track unit whose untrimmed-only track
    # has no trimmed arrays while the parent is graded from the other track.
    _add_analysis(db, "s3", "WARNING",
                  [("UNTRIMMED", None, None), ("WARNING", [0.0, 1.0], [0.1, 0.2])])

    assert _detected_ids(db) == set()


def test_zero_track_records_still_detected(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "t.db")
    empty_id = _add_analysis(db, "s1", "PASS", [])
    # An UNTRIMMED parent with no rows at all is still nothing-stored, so the
    # zero-track branch keeps it (unlike the array-less branch).
    empty_untrimmed = _add_analysis(db, "s2", "UNTRIMMED", [])
    _add_analysis(db, "s3", "PASS", [("PASS", [0.0, 1.0], [0.1, 0.2])])

    assert _detected_ids(db) == {empty_id, empty_untrimmed}


def test_partially_populated_record_is_not_flagged(tmp_path):
    """One real track + one array-less track keeps its measurement and its
    verdict. Re-parsing it would rewrite good rows."""
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "t.db")
    _add_analysis(db, "s1", "FAIL",
                  [("FAIL", [0.0, 1.0], [0.1, 0.2]), ("WARNING", None, None)])

    assert _detected_ids(db) == set()


def test_empty_list_arrays_count_as_absent(tmp_path):
    """SafeJSON binds [] to SQL NULL, so an empty list is array-less too."""
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "t.db")
    aid = _add_analysis(db, "s1", "PASS", [("PASS", [], [])])

    assert _detected_ids(db) == {aid}


def test_unreachable_source_is_reported_not_raised(tmp_path):
    """Off the plant network every source is unreachable. That must be a
    per-record outcome, never an exception and never a silent skip."""
    from laser_trim_analyzer.core.track_repair import (
        repair_missing_tracks, SOURCE_UNREACHABLE)
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "t.db")
    _add_analysis(db, "s1", "PASS", [], file_path=r"\\192.168.66.9\Share\gone.xls")
    _add_analysis(db, "s2", "WARNING", [("WARNING", None, None)],
                  file_path=str(tmp_path / "also-missing.xls"))

    report = repair_missing_tracks(db=db, linked_only=False)

    assert report.examined == 2, "the batch must not abort on the first miss"
    assert report.unreachable == 2
    assert report.repaired == 0
    assert {o.result for o in report.outcomes} == {SOURCE_UNREACHABLE}
    assert "unreachable" in report.summary()


def test_progress_callback_failure_does_not_abort_the_batch(tmp_path):
    from laser_trim_analyzer.core.track_repair import repair_missing_tracks
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "t.db")
    _add_analysis(db, "s1", "PASS", [], file_path="/nowhere/a.xls")

    def boom(done, total, phase):
        raise RuntimeError("widget destroyed")

    report = repair_missing_tracks(db=db, progress=boom, linked_only=False)
    assert report.examined == 1 and report.unreachable == 1


# --- ingest guard: a verdict must be backed by the measurement -------------
# Nothing may create the shape the repair tool exists to fix. A track that
# claims PASS/WARNING/FAIL while carrying no arrays is not a quiet grade, it
# is an ungradeable read — the same call the parser already makes when the
# limit columns are corrupt (linearity_spec_warning -> ERROR, verdict None).

def _track(status, positions, errors):
    from laser_trim_analyzer.core.models import AnalysisStatus, TrackData
    return TrackData(
        track_id="TRK1", status=AnalysisStatus[status], travel_length=1.0,
        linearity_spec=0.05, sigma_gradient=0.001, sigma_pass=True,
        linearity_error=0.01, linearity_pass=True,
        position_data=positions, error_data=errors)


def test_gradeable_track_without_arrays_is_demoted_to_error():
    from laser_trim_analyzer.core.models import AnalysisStatus
    from laser_trim_analyzer.core.processor import enforce_measurement_backed_verdict

    for arrays in ((None, None), ([], []), ([0.0, 1.0], None), (None, [0.1])):
        for status in ("PASS", "WARNING", "FAIL"):
            t = _track(status, *arrays)
            reason = enforce_measurement_backed_verdict(t)
            assert reason, f"{status} with arrays={arrays} passed the guard"
            assert t.status == AnalysisStatus.ERROR
            # The verdict is withdrawn, never inverted: an ungraded unit must
            # not read as a linearity rejection it never earned.
            assert t.linearity_pass is None and t.sigma_pass is None


def test_guard_leaves_measured_tracks_and_untrimmed_sweeps_alone():
    from laser_trim_analyzer.core.models import AnalysisStatus
    from laser_trim_analyzer.core.processor import enforce_measurement_backed_verdict

    good = _track("PASS", [0.0, 1.0], [0.1, 0.2])
    assert enforce_measurement_backed_verdict(good) is None
    assert good.status == AnalysisStatus.PASS and good.linearity_pass is True

    # An untrimmed-only sweep legitimately has no trimmed arrays and claims no
    # verdict — the by-design shape that must never be demoted or flagged.
    sweep = _track("UNTRIMMED", None, None)
    sweep.linearity_pass = None
    sweep.sigma_pass = None
    assert enforce_measurement_backed_verdict(sweep) is None
    assert sweep.status == AnalysisStatus.UNTRIMMED

    # ERROR tracks are already ungraded; the guard has nothing to add.
    err = _track("ERROR", None, None)
    assert enforce_measurement_backed_verdict(err) is None
