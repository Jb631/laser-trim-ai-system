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
