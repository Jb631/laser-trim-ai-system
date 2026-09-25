"""`scripts/app_qa_sweep.py`'s database-level checks, held to scratch databases.

A sweep check is only worth its PASS if it can FAIL, and only if its FAIL means a bug:
the 2026-09-24 final review found one that would FAIL on correct data the first time
work ingested a laser-2/3 file after the pull (`check_initial_trim_value_on_database`)
and one that PASSed over zero rows (`check_increment_volts_on_database`). Each case
here builds the rows by the real schema, then runs ONE check on them.

The sweep stubs `tkinter`/`customtkinter` into `sys.modules` at import (see
`test_harness_db_guard.py`), so importing it would poison this pytest process: the
check runs in a subprocess that imports the sweep, opens the scratch file READ-ONLY,
and prints the check's PASS/FAIL/WARN lines back as JSON. Invented values throughout.
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# argv: scratch db, check function name, a tmp dir for the injected guard database.
_RUNNER = r"""
import json, sqlite3, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import app_qa_sweep as sweep
import laser_trim_analyzer.database.manager as _m, laser_trim_analyzer.database as _d
_guard = _m.DatabaseManager(Path(sys.argv[3]) / "guard.db")
_m._db_manager = _guard; _d._db_manager = _guard     # nothing may reach the configured DB
conn = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
getattr(sweep, sys.argv[2])(conn)
print("RESULTS_JSON=" + json.dumps(sweep.RESULTS))
"""


def _run_check(db, name, tmp_path):
    """[(PASS|FAIL|WARN, check name, detail)] from one sweep check run on `db`'s file."""
    db.close()                    # every row in the main file before another process reads it
    guard = tmp_path / "runner"
    guard.mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, "-c", _RUNNER, str(db.database_path), name, str(guard)],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULTS_JSON=")), None)
    assert r.returncode == 0 and line, (
        f"{name} did not run (rc={r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return [tuple(x) for x in json.loads(line[len("RESULTS_JSON="):])]


def _scratch_db(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / "scratch.db")


def _track(s, *, system, model, serial, file_path=None):
    """One analysis + one track; returns the track row (flushed, so it has an id)."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    ar = DBAR(model=model, serial=serial, system=SystemType[system],
              filename=f"{model}_{serial}.xls", file_path=file_path or f"/fake/{model}_{serial}.xls",
              file_date=datetime(2026, 2, 3), overall_status=StatusType.PASS)
    s.add(ar)
    s.flush()
    tr = DBTR(analysis_id=ar.id, track_id="TRK1" if system != "B" else "Track A",
              status=StatusType.PASS, travel_length=1.0, linearity_spec=0.01)
    s.add(tr)
    s.flush()
    return tr


# ============================================== check_initial_trim_value_on_database

# (initial_trim_value column, recipe) per laser-2 pass row -- the shapes each writer leaves.
OLD = (None, {"initial_trim_value": [0.101, 0.202, None], "laser_speed_high": 10.0})
NEW = ([0.303, 0.404], {"laser_speed_high": 10.0})       # today's writer: key not in recipe
SHEET_HAD_NONE = (None, {"initial_trim_value": [None, None], "laser_speed_high": 10.0})
NEITHER = (None, {"laser_speed_high": 10.0})              # the capture lost on the way


def _laser2_passes(tmp_path, passes):
    from sqlalchemy import null as sql_null
    from laser_trim_analyzer.database.models import TrimPass
    db = _scratch_db(tmp_path)
    with db.session() as s:
        tr = _track(s, system="A", model="9993", serial="1")
        for i, (column, recipe) in enumerate(passes, start=1):
            s.add(TrimPass(track_result_id=tr.id, pass_index=i, sheet=f"SEC1 TRK1 {i} TRM1",
                           initial_trim_value=column if column is not None else sql_null(),
                           recipe=recipe))
    return db


def _itv(results):
    return [r for r in results if r[1].startswith("initial trim value")]


def test_initial_trim_value_old_and_new_rows_together_pass(tmp_path):
    """5 rows written before the column existed (value in recipe) + 2 written by today's
    code (value in the column only) + 1 whose sheet genuinely had nothing: all correct.
    The old check held all 7 real values against the recipe's 5 and FAILed
    "helper=7 recipe=5" -- the first ingest of a laser-2/3 file after the pull."""
    db = _laser2_passes(tmp_path, [OLD] * 5 + [NEW] * 2 + [SHEET_HAD_NONE])
    results = _itv(_run_check(db, "check_initial_trim_value_on_database", tmp_path))
    assert [r for r in results if r[0] != "PASS"] == [], results
    assert len(results) == 4, results          # coverage, no loss, recipe agreement, read-back
    coverage = next(r for r in results if "carries its capture" in r[1])
    assert "2 in the column" in coverage[2] and "6 in recipe" in coverage[2], coverage
    assert "1 whose sheet had none" in coverage[2], coverage


def test_initial_trim_value_a_pass_captured_nowhere_fails_coverage(tmp_path):
    db = _laser2_passes(tmp_path, [OLD] * 5 + [NEW] * 2 + [NEITHER])
    results = _itv(_run_check(db, "check_initial_trim_value_on_database", tmp_path))
    failed = [r for r in results if r[0] == "FAIL"]
    assert len(failed) == 1 and "carries its capture" in failed[0][1], results
    assert "1 in neither place" in failed[0][2], failed


def test_initial_trim_value_never_passes_over_rows_it_does_not_have(tmp_path):
    """Old rows only (today's copy): the column read-back has nothing to hold, so it is
    not run -- no PASS over zero rows."""
    db = _laser2_passes(tmp_path, [OLD] * 3)
    results = _itv(_run_check(db, "check_initial_trim_value_on_database", tmp_path))
    assert [r[0] for r in results] == ["PASS"] * 3, results
    assert not any("reads back each column value" in r[1] for r in results), results


# ================================================= check_increment_volts_on_database

LTS_FIXTURE = REPO / "tests" / "fixtures" / "trim" / "lts_8232-1_193.xls"   # TrimVolts1 has readings


def _laser1_passes(tmp_path, passes):
    """passes: (captured?, processed after the capture started?, file_path or None).
    The scratch database records its own capture start at its first start-up, exactly
    as a work database does; a pass 'before' it is dated 2020."""
    from sqlalchemy import null as sql_null
    from laser_trim_analyzer.database.models import TrimPass
    db = _scratch_db(tmp_path)
    with db.session() as s:
        for i, (captured, after, path) in enumerate(passes, start=1):
            tr = _track(s, system="B", model="9994", serial=str(i),
                        file_path=str(path) if path else None)
            row = TrimPass(track_result_id=tr.id, pass_index=1, sheet="Trim 1",
                           increment_volts=[[0.111, 0.222]] if captured else sql_null(),
                           increment_volts_first_row=2 if captured else None,
                           increment_volts_truncated=False if captured else None)
            if not after:
                row.created_date = datetime(2020, 1, 1)
            s.add(row)
    return db


def _iv(results):
    return [r for r in results if r[1].startswith("increment volts")]


def test_increment_volts_never_passes_over_zero_passes_processed_since(tmp_path):
    """Today's copy: every laser-1 pass predates the capture. There is nothing to hold,
    so the carry-check is a WARN with the counts -- the old code printed PASS."""
    db = _laser1_passes(tmp_path, [(True, False, None), (False, False, None)])
    results = _iv(_run_check(db, "check_increment_volts_on_database", tmp_path))
    carries = [r for r in results if "carries its TrimVolts curves" in r[1]]
    assert carries == [], f"no PASS/FAIL over zero passes: {results}"
    warned = [r for r in results if r[0] == "WARN" and "no laser-1 Trim N pass processed" in r[1]]
    assert len(warned) == 1, results
    assert "0 processed since the capture started" in warned[0][2], warned
    assert "2 predate it (1 back-filled, 1 not yet)" in warned[0][2], warned


def test_increment_volts_passes_when_a_pass_processed_since_carries_its_curves(tmp_path):
    db = _laser1_passes(tmp_path, [(False, False, None), (True, True, None)])
    results = _iv(_run_check(db, "check_increment_volts_on_database", tmp_path))
    carries = [r for r in results if "carries its TrimVolts curves" in r[1]]
    assert [r[0] for r in carries] == ["PASS"], results
    assert "1 processed since the capture started: 1 carry their curves" in carries[0][2]


def test_increment_volts_still_fails_on_a_missed_pass(tmp_path):
    """Processed since the capture started, no curves stored, and its workbook HAS a
    TrimVolts reading beside `Trim 1` (the real fixture): missed -- a FAIL."""
    assert LTS_FIXTURE.is_file()
    db = _laser1_passes(tmp_path, [(True, True, None), (False, True, LTS_FIXTURE)])
    results = _iv(_run_check(db, "check_increment_volts_on_database", tmp_path))
    carries = [r for r in results if "carries its TrimVolts curves" in r[1]]
    assert [r[0] for r in carries] == ["FAIL"], results
    assert f"{LTS_FIXTURE.name} Trim 1" in carries[0][2], carries


def test_increment_volts_a_pass_refused_at_ingest_is_a_warn_not_a_miss(tmp_path):
    """Since 2026-09-24 ingest REFUSES a capture its workbook's own VOLTAGES sheet
    contradicts (a wrong position is worse than none), so a pass processed since the
    capture can correctly hold no curves beside a TrimVolts sheet full of readings.
    Settled by the back-fill's reader, that is a refusal -- a WARN naming the file --
    never "missed": a FAIL there would fail correct data, the first time work ingests
    such a file. Built from the back-fill tests' own misplaced file."""
    from test_backfill_increment_volts import _bump_initial_points_ignored, _write_modified_copy
    shifted = _write_modified_copy(tmp_path / "shifted.xlsx", mutate=_bump_initial_points_ignored)
    db = _laser1_passes(tmp_path, [(True, True, None), (False, True, shifted)])
    results = _iv(_run_check(db, "check_increment_volts_on_database", tmp_path))
    carries = [r for r in results if "carries its TrimVolts curves" in r[1]]
    assert [r[0] for r in carries] == ["PASS"], results
    assert "1 refused" in carries[0][2] and "0 missed" in carries[0][2], carries
    refused = [r for r in results if r[0] == "WARN" and "REFUSED at ingest" in r[1]]
    assert len(refused) == 1 and "shifted.xlsx Trim 1" in refused[0][2], results


# ============================================== a check that crashes (facelift F4, TRACKER C2)
# Several of the sweep's check blocks had no try/except, so ONE exception ended the whole sweep:
# every check after it silently never ran, and the tally line never printed. A block that crashes
# is now ONE FAIL naming the block and the exception, and the sweep goes on.

_RUNNER_CODE = r"""
import json, sys
sys.path.insert(0, "scripts")
import app_qa_sweep as sweep
exec(sys.argv[1])
print("RESULTS_JSON=" + json.dumps(sweep.RESULTS))
"""


def _run_code(code):
    r = subprocess.run([sys.executable, "-c", _RUNNER_CODE, code], cwd=REPO, capture_output=True,
                       text=True, timeout=300)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULTS_JSON=")), None)
    return r, ([tuple(x) for x in json.loads(line[len("RESULTS_JSON="):])] if line else None)


def test_a_block_that_crashes_is_one_fail_and_the_sweep_goes_on():
    r, results = _run_code(
        "with sweep._guard('invented block'):\n"
        "    sweep.check('invented block: a check before the crash', True)\n"
        "    raise ValueError('invented crash')\n"
        "sweep.check('the next block still runs', True)\n")
    assert r.returncode == 0 and results is not None, r.stdout[-2000:] + r.stderr[-2000:]
    assert results == [
        ("PASS", "invented block: a check before the crash", ""),
        ("FAIL", "invented block (the check itself crashed)", "ValueError: invented crash"),
        ("PASS", "the next block still runs", "")]
    assert "Traceback" in r.stdout and "invented crash" in r.stdout     # where, not only what


def test_stopping_the_run_is_never_swallowed():
    r, results = _run_code("with sweep._guard('invented block'):\n    raise KeyboardInterrupt\n")
    assert r.returncode != 0 and results is None                     # it propagated
    assert "KeyboardInterrupt" in r.stderr


# ============================================== check_error_rows_have_a_reason

def _error_analysis(s, *, file_path, error_reason=None):
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, StatusType, SystemType
    ar = DBAR(model="7000", serial="1", system=SystemType.A, filename=file_path.replace("\\", "/")
              .rsplit("/", 1)[-1], file_path=file_path, file_date=datetime(2026, 2, 3),
              overall_status=StatusType.ERROR, error_reason=error_reason)
    s.add(ar)
    s.flush()
    return ar


def _marker(s, *, file_path, reason):
    """A per-path failure marker, the shape _write_failure_marker leaves (invented values)."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import ProcessedFile, UNREADABLE_PREFIX
    s.add(ProcessedFile(filename=file_path.replace("\\", "/").rsplit("/", 1)[-1],
                        file_path=file_path, file_hash=DatabaseManager.skip_marker_hash(file_path),
                        file_size=1, file_modified_date=datetime(2026, 2, 3),
                        error_message=UNREADABLE_PREFIX + reason, analysis_id=None, success=True))


def _reason(results):
    return [r for r in results if r[1].startswith("every ERROR row has a reason")]


def test_a_marker_spelled_another_way_still_explains_its_error_row(tmp_path):
    """The marker is matched to its ERROR row by PATH. It used to be exact string equality -- true
    today only because one write path writes both -- so the same file spelled with forward
    slashes (a forward-slash config root) or in another case (Windows paths are case-insensitive)
    read as an ERROR row with no reason at all: a FAIL for a reason that is there."""
    db = _scratch_db(tmp_path)
    with db.session() as s:
        _error_analysis(s, file_path="C:\\Shop\\Trim Data\\7000_1.xls")
        _marker(s, file_path="c:/shop/trim data/7000_1.xls", reason="No valid track data found")
        s.commit()
    results = _reason(_run_check(db, "check_error_rows_have_a_reason", tmp_path))
    assert results[0][0] == "PASS", results
    assert [r[0] for r in results] == ["PASS", "WARN"], results          # marker-only: a WARN
    assert "1 of 1" in results[1][2]


def test_an_error_row_with_no_reason_anywhere_still_fails(tmp_path):
    db = _scratch_db(tmp_path)
    with db.session() as s:
        _error_analysis(s, file_path="C:\\Shop\\Trim Data\\7000_1.xls")
        _marker(s, file_path="C:\\Shop\\Trim Data\\7000_2.xls", reason="another file entirely")
        _error_analysis(s, file_path="C:\\Shop\\Trim Data\\7000_3.xls",
                        error_reason="Insufficient data points")               # explained
        s.commit()
    results = _reason(_run_check(db, "check_error_rows_have_a_reason", tmp_path))
    assert [r[0] for r in results] == ["FAIL"], results
    assert "reasonless=1 of 2" in results[0][2]


# ============================================== check_screens_count_what_they_draw (M9 wording)

_RUNNER_DB = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import app_qa_sweep as sweep
import laser_trim_analyzer.database.manager as _m, laser_trim_analyzer.database as _d
_guard = _m.DatabaseManager(Path(sys.argv[3]) / "guard.db")
_m._db_manager = _guard; _d._db_manager = _guard     # nothing may reach the configured DB
getattr(sweep, sys.argv[2])(_m.DatabaseManager(Path(sys.argv[1])))
print("RESULTS_JSON=" + json.dumps(sweep.RESULTS))
"""


def test_the_screens_check_names_what_it_counts(tmp_path):
    """Re-review Minor 4: the check said it matched "the rows its section draws" and printed
    "drawn=5" -- but it counts the rows a page HANDS its FindingsView, and the view draws at most
    3 of a group (rows_per_group) behind "Show all 5". Five yield findings: the count holds, and
    the line says what it counted."""
    db = _scratch_db(tmp_path)
    yields = [{"model": "7000", "analyzer": "ink_target", "category": "Ink target", "lever": "ink",
               "title": f"Invented finding {i}", "summary": "invented", "n_units": 10 + i,
               "tracks_per_year": 100.0 - i, "expected_gain_points": 2.0, "evidence": {}}
              for i in range(5)]
    db.replace_process_findings("7000", {"tracks": 1, "errors": {}}, yields)
    db.close()
    guard = tmp_path / "runner"
    guard.mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, "-c", _RUNNER_DB, str(db.database_path),
                        "check_screens_count_what_they_draw", str(guard)],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULTS_JSON=")), None)
    assert r.returncode == 0 and line, r.stdout[-3000:] + r.stderr[-3000:]
    results = [tuple(x) for x in json.loads(line[len("RESULTS_JSON="):])]
    home = [x for x in results if x[1].startswith("home: 'N worth changing'")]
    assert len(home) == 1 and home[0][0] == "PASS", results
    assert "draws" not in home[0][1] and "hands its view" in home[0][1], home
    assert "drawn=" not in home[0][2] and "handed=5" in home[0][2], home
    model = [x for x in results if x[1].startswith("model page: the 'Worth changing' count")]
    assert len(model) == 1 and model[0][0] == "PASS", results
    assert "draws" not in model[0][1] and "hands its view" in model[0][1], model
    assert "drawn=" not in model[0][2] and "handed=5" in model[0][2], model
