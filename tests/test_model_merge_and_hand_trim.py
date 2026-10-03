"""Two decisions James made on 2026-10-02.

1. "yea you can mearge 7953A with 7953-A. and 7953B with 7953-B" -- the same parts, named both
   ways by their files (the only two such twins in the fleet). One name from now on: a file named
   either way is stored under the hyphenated name, and the rows already stored under the glued
   name are renamed once, at start-up.
2. Of 8232-1 and 8340-1: "yes those are hand trim models and we should use 20X". A linearity error
   up to 20x its spec band is a REAL laser-stage error on a model that is hand-trimmed afterwards
   (their normal tracks run up to 5-10x), so it is not "suspect" there: the line is 20x on the
   hand-trim models, 10x everywhere else. Files already stored as suspect for no other reason than
   a 10-20x error are re-judged once, at start-up.

Every dataset here is INVENTED.
"""
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
T0 = datetime(2025, 3, 3)


def _db(tmp_path, name="m.db"):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / name)


# =============================================================================================
# 1 -- one name for 7953-A and 7953-B

def test_the_glued_names_read_as_their_hyphenated_twins():
    from laser_trim_analyzer.core.model_names import canonical_model
    assert canonical_model("7953A") == "7953-A"
    assert canonical_model("7953B") == "7953-B"
    for other in ("7953-A", "7953-B", "7953-1A", "7953-1B", "8340-1", "Unknown", "", None):
        assert canonical_model(other) == other


def test_a_trim_file_named_7953A_is_stored_as_7953_A(tmp_path):
    from laser_trim_analyzer.core.parser import ExcelParser
    dst = tmp_path / "7953A_18_TEST DATA_9-5-2025_9-00 AM.xls"
    shutil.copy(FIXTURES / "trim" / "dlts_8074_18.xls", dst)
    parsed = ExcelParser().parse_file(dst)
    assert parsed["metadata"].model == "7953-A"


def _analysis_row(s, model, serial, day, *, status="PASS", unit_id=None, data_quality="good",
                  issues=None):
    from laser_trim_analyzer.database.models import (
        AnalysisResult as AR, StatusType, SystemType)
    ar = AR(filename=f"{model}_{serial}_{day:%Y%m%d}.xls", file_path=f"/f/{model}/{serial}",
            file_hash=f"{model}-{serial}-{day:%Y%m%d}", model=model, serial=serial,
            system=SystemType.A, file_date=day, timestamp=day,
            overall_status=getattr(StatusType, status), has_multi_tracks=False,
            processing_time=0.1, unit_id=unit_id, data_quality=data_quality,
            data_quality_issues=json.dumps(issues) if issues is not None else None)
    s.add(ar)
    s.flush()
    return ar


def test_rows_already_stored_under_the_glued_name_are_renamed_once(tmp_path):
    from sqlalchemy import text
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as AR, ModelMetricState, ModelProcessFacts)
    path = tmp_path / "old.db"
    db = DatabaseManager(path)
    with db.session() as s:
        _analysis_row(s, "7953A", "12", T0, unit_id="7953A/12/2025-03-03")
        _analysis_row(s, "7953-A", "13", T0, unit_id="7953-A/13/2025-03-03")
        _analysis_row(s, "7953B", "4", T0, unit_id="7953B/4/2025-03-03")
        s.add(ModelMetricState(model="7953A", metric="linearity_error", is_trained=True,
                               baseline_count=9, last_updated=T0))
        s.add(ModelMetricState(model="7953-A", metric="linearity_error", is_trained=True,
                               baseline_count=9, last_updated=T0))
        s.add(ModelProcessFacts(model="7953B", facts={}, computed_at=T0))
        s.commit()
        # A spec under the glued name only, and a requalification under each name.
        s.execute(text("INSERT INTO baseline_requalifications (model, effective_date, note, set_at)"
                       " VALUES ('7953A', '2024-01-01', 'old', '2024-01-02')"))
        s.commit()
    with db.session() as s:              # forget that the merge already ran on this new file
        s.execute(text("DELETE FROM app_meta WHERE key = 'model_aliases'"))
        s.commit()
    db.close()

    db = DatabaseManager(path)           # start-up: the migration runs
    with db.session() as s:
        models = sorted(r[0] for r in s.query(AR.model))
        units = sorted(r[0] for r in s.query(AR.unit_id))
        states = sorted((r.model, r.metric) for r in s.query(ModelMetricState))
        facts = sorted(r.model for r in s.query(ModelProcessFacts))
        requal = [r[0] for r in s.execute(text("SELECT model FROM baseline_requalifications"))]
    assert models == ["7953-A", "7953-A", "7953-B"]
    assert units == ["7953-A/12/2025-03-03", "7953-A/13/2025-03-03", "7953-B/4/2025-03-03"]
    # the glued name's per-model state goes (it is rebuilt for the merged model); the
    # hyphenated name's own state stays until it is rebuilt
    assert states == [("7953-A", "linearity_error")]
    assert facts == []
    assert requal == ["7953-A"]             # moved over: the hyphenated name had none


def test_the_merge_runs_once(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import AnalysisResult as AR
    path = tmp_path / "once.db"
    db = DatabaseManager(path)           # a new database: nothing to rename, the merge is recorded
    with db.session() as s:
        _analysis_row(s, "7953A", "12", T0)      # a glued name stored after the merge ran
        s.commit()
    db.close()
    db = DatabaseManager(path)
    with db.session() as s:
        assert [r[0] for r in s.query(AR.model)] == ["7953A"]


def test_a_final_test_and_a_smoothness_record_are_saved_under_the_merged_name(tmp_path):
    # Through the real save paths, whatever name the parser handed them.
    from laser_trim_analyzer.database.models import FinalTestResult as FT, SmoothnessResult as SR
    db = _db(tmp_path)
    meta = {"filename": "7953B-sn4.xls", "file_path": str(tmp_path / "7953B-sn4.xls"),
            "model": "7953B", "serial": "4", "file_date": T0, "test_date": T0}
    db.save_final_test(meta, [], {"overall_status": "PASS"}, file_hash="ft-7953B-4")
    db.save_smoothness_result({**meta, "filename": "7953A-sn9.xls", "model": "7953A",
                               "serial": "9"}, [], file_hash="sm-7953A-9")
    with db.session() as s:
        assert [r[0] for r in s.query(FT.model)] == ["7953-B"]
        assert [r[0] for r in s.query(SR.model)] == ["7953-A"]


# =============================================================================================
# 2 -- the hand-trim models' line is 20x

def _track(error, band):
    from laser_trim_analyzer.core.models import AnalysisStatus, TrackData
    pts = [i / 20 for i in range(21)]
    return TrackData(track_id="TRK1", travel_length=1.0, linearity_spec=band,
                     status=AnalysisStatus.FAIL, linearity_error=error,
                     position_data=pts, error_data=[0.001] * len(pts),
                     upper_limits=[band] * len(pts), lower_limits=[-band] * len(pts))


def test_a_15x_error_is_suspect_on_an_ordinary_model():
    from laser_trim_analyzer.core.processor import Processor as P
    issues = P._validate_track_data([_track(0.15, 0.01)], model="8074")
    assert any("scale-anomalous linearity error" in i for i in issues)


def test_a_15x_error_is_a_real_error_on_a_hand_trim_model():
    from laser_trim_analyzer.core.processor import Processor as P
    assert P._validate_track_data([_track(0.15, 0.01)], model="8232-1") == []
    assert P._validate_track_data([_track(0.15, 0.01)], model="8340-1") == []


def test_a_25x_error_is_suspect_on_a_hand_trim_model_too():
    from laser_trim_analyzer.core.processor import Processor as P
    issues = P._validate_track_data([_track(0.25, 0.01)], model="8232-1")
    assert any("scale-anomalous linearity error" in i for i in issues)


def _suspect_file(s, model, serial, error, band, *, other_issue=None):
    from laser_trim_analyzer.database.models import TrackResult as TR, StatusType
    issues = [f"Track A: scale-anomalous linearity error ({error:.4g} vs spec band ±{band:.4g})"]
    if other_issue:
        issues.append(other_issue)
    ar = _analysis_row(s, model, serial, T0, status="FAIL", data_quality="suspect", issues=issues)
    s.add(TR(analysis_id=ar.id, track_id="TRK1", status=StatusType.FAIL,
             final_linearity_error_shifted=error, upper_limits=[band] * 21,
             lower_limits=[-band] * 21))
    return ar.id


def test_stored_hand_trim_files_are_re_judged_once_at_the_20x_line(tmp_path):
    from sqlalchemy import text
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import AnalysisResult as AR
    path = tmp_path / "suspects.db"
    db = DatabaseManager(path)
    with db.session() as s:
        real = _suspect_file(s, "8232-1", "1", 0.15, 0.01)             # 15x on a hand-trim model
        broken = _suspect_file(s, "8232-1", "2", 0.25, 0.01)           # 25x: still a fault
        other = _suspect_file(s, "8340-1", "3", 0.15, 0.01,
                              other_issue="TRK1: all-zero error data")   # another reason too
        ordinary = _suspect_file(s, "8074", "4", 0.15, 0.01)           # 15x, not hand-trimmed
        s.commit()
        s.execute(text("DELETE FROM app_meta WHERE key = 'hand_trim_suspect_line'"))
        s.commit()
    db.close()

    db = DatabaseManager(path)           # start-up: the re-judging runs
    with db.session() as s:
        dq = {r.id: (r.data_quality, r.data_quality_issues) for r in s.query(AR)}
    assert dq[real][0] == "good"
    assert json.loads(dq[real][1]) == []
    assert dq[broken][0] == "suspect"
    assert dq[other][0] == "suspect"
    assert dq[ordinary][0] == "suspect"


def test_the_drift_state_is_rebuilt_for_these_changes():
    # Both change what feeds drift (the merged models' runs; the hand-trim models' real errors),
    # so the stored drift state is rebuilt once more at the next start.
    from laser_trim_analyzer.ml.drift_training import DRIFT_RULES_VERSION
    assert DRIFT_RULES_VERSION != "2026-10-02"
