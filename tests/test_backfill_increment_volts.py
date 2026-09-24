"""`scripts/backfill_increment_volts.py` -- the resumable back-fill for laser 1's TrimVolts
capture (Task 4, `49f865e`). Brief: `.superpowers/sdd/2026-09-23-parse-fixes/task-5-brief.md`.

Built the way the brief's Step 1 asks: a tmp database built by processing the real LTS
fixtures with Task 4's own capture (it already ships), then the three columns are NULLed out
by hand to reproduce "written before the back-fill existed" -- and the script is run and
compared back against what Task 4's capture put there in the first place (itself tied to an
independent, fresh `ExcelParser().parse_file(...)` call, not just to the DB's own earlier
write).

Six behaviours, each a test group below:
  1. refuses without a DB path, and tells you to snapshot first
  2. selects laser-1 (system 'B') Trim-N passes with increment_volts IS NULL, oldest file
     first; opens each file read-only; reads only Model Parameters + TrimVolts N; matches by
     (track_result_id, sheet)
  3. resumable, --limit bounds a run, progress every N files with a rate and an ETA
  4. a missing/unreadable file is counted and named, never fatal, and changes nothing else
  5. --dry-run reads and reports without writing
  6. commits in batches (files, not passes)
"""
import contextlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import scripts.backfill_increment_volts as biv          # noqa: E402
from laser_trim_analyzer.core.parser import ExcelParser  # noqa: E402

LTS = Path("tests/fixtures/trim/lts_8232-1_193.xls")
LTS_194 = Path("tests/fixtures/trim/lts_8232-1_194.xls")
DLTS = Path("tests/fixtures/trim/dlts_8232-1_243.xls")


# --------------------------------------------------------------------- helpers

def _inject(monkeypatch, db):
    """BOTH globals: a Processor must never reach the configured database."""
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)


def _build_db(tmp_path, monkeypatch, files, name="t.db"):
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    db = mgr.DatabaseManager(tmp_path / name)
    _inject(monkeypatch, db)
    proc = Processor(use_ml=False)
    for f in files:
        db.save_analysis(proc.process_file(f))
    return db


def _null_out(db):
    """Reproduce 'written before the capture existed' on every trim_passes row."""
    with db.session() as s:
        s.execute(sa.text(
            "UPDATE trim_passes SET increment_volts = NULL, "
            "increment_volts_first_row = NULL, increment_volts_truncated = NULL"))


def _target_columns_snapshot(db):
    """{trim_passes.id: (increment_volts, first_row, truncated)} -- the three columns
    this script may touch, exactly as stored (JSON text or NULL, not decoded)."""
    with db.session() as s:
        rows = s.execute(sa.text(
            "SELECT id, increment_volts, increment_volts_first_row, "
            "increment_volts_truncated FROM trim_passes ORDER BY id")).all()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


_TP_COLUMNS = None  # filled lazily -- needs the ORM models, which need sys.path set up


def _all_trim_pass_columns():
    global _TP_COLUMNS
    if _TP_COLUMNS is None:
        from laser_trim_analyzer.database.models import TrimPass
        _TP_COLUMNS = list(TrimPass.__table__.columns.keys())
    return _TP_COLUMNS


def _other_columns_snapshot(db):
    """Every trim_passes column EXCEPT the three target ones, ordered and keyed by id --
    what 'nothing else changed' is checked against."""
    cols = _all_trim_pass_columns()
    others = [c for c in cols if c not in
             ("increment_volts", "increment_volts_first_row", "increment_volts_truncated")]
    with db.session() as s:
        rows = s.execute(sa.text(
            f"SELECT {', '.join(others)} FROM trim_passes ORDER BY id")).all()
    return [tuple(r) for r in rows]


def _other_tables_snapshot(db):
    """Every column of track_results and analysis_results -- the back-fill must be
    trim_passes-only."""
    with db.session() as s:
        tracks = s.execute(sa.text("SELECT * FROM track_results ORDER BY id")).all()
        analyses = s.execute(sa.text("SELECT * FROM analysis_results ORDER BY id")).all()
    return [tuple(r) for r in tracks], [tuple(r) for r in analyses]


def _trim_n_increment_volts(db):
    """increment_volts of every 'Trim N' row only -- excludes 'Lin Error', which never gets
    a TrimVolts capture and must stay NULL forever; a test that expects it to fill in is
    wrong about what this script does, not the script."""
    with db.session() as s:
        rows = s.execute(sa.text(
            "SELECT increment_volts FROM trim_passes WHERE sheet LIKE 'Trim %'")).all()
    return [r[0] for r in rows]


def _set_file_path(db, filename, new_path):
    with db.session() as s:
        s.execute(sa.text("UPDATE analysis_results SET file_path = :p WHERE filename = :fn"),
                 {"p": str(new_path), "fn": filename})


def _null_out_path(db_path):
    """Same as `_null_out`, but against a path rather than a live DatabaseManager -- for the
    CLI tests, which open their OWN DatabaseManager inside `main()`."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE trim_passes SET increment_volts = NULL, "
                     "increment_volts_first_row = NULL, increment_volts_truncated = NULL")
        conn.commit()
    finally:
        conn.close()


_FAKE_CAPTURE = {"increment_volts": [[0.1, 0.2]], "increment_volts_first_row": 0,
                 "increment_volts_truncated": False}


def _fake_files(n, passes_per_file=1):
    return [biv.FileWork(
                analysis_id=i, file_path=f"fake/share/file{i}.xls", filename=f"file{i}.xls",
                passes=[biv.PassRef(pass_id=i * passes_per_file + k, sheet=f"Trim {k + 1}")
                       for k in range(passes_per_file)])
           for i in range(n)]


def _counting_session(db, calls):
    """Wrap db.session so each ENTRY (== one commit, since DatabaseManager.session commits
    once per successful `with` block) is counted -- the same idea as
    test_ft_regrade_throughput.py's _CountingSession, simplified: we only need the count."""
    real_session = db.session

    @contextlib.contextmanager
    def counting():
        calls.append(1)
        with real_session() as s:
            yield s
    return counting


# ============================================================== 2. selection + real reads

@pytest.mark.parametrize("fixture", [LTS, LTS_194], ids=lambda p: p.name)
def test_backfill_matches_a_fresh_parse_of_the_real_file(tmp_path, monkeypatch, fixture):
    db = _build_db(tmp_path, monkeypatch, [fixture])
    before = _target_columns_snapshot(db)
    assert any(v[0] is not None for v in before.values()), \
        "sanity: Task 4's own capture must have put something there first"

    _null_out(db)
    assert all(v == (None, None, None) for v in _target_columns_snapshot(db).values())

    report = biv.backfill(db)
    assert report.dry_run is False
    assert report.files_missing == 0
    assert report.files_unreadable == 0
    assert report.passes_filled > 0

    after = _target_columns_snapshot(db)
    assert after == before, "back-filled values must equal what Task 4's own capture wrote"

    # Tied to an INDEPENDENT fresh parse too, not only to this DB's own earlier write.
    fresh = ExcelParser().parse_file(fixture)
    fresh_by_sheet = {p["sheet"]: p for t in fresh["tracks"] for p in t.get("trim_passes", [])
                      if p["sheet"].lower().startswith("trim ")}
    with db.session() as s:
        rows = s.execute(sa.text(
            "SELECT p.sheet, p.increment_volts, p.increment_volts_first_row, "
            "p.increment_volts_truncated FROM trim_passes p "
            "JOIN track_results t ON t.id = p.track_result_id "
            "JOIN analysis_results a ON a.id = t.analysis_id "
            "WHERE a.filename = :fn AND p.sheet LIKE 'Trim %'"), {"fn": fixture.name}).all()
    assert len(rows) == 2
    for sheet, volts, first_row, truncated in rows:
        want = fresh_by_sheet[sheet]
        assert json.loads(volts) == want["increment_volts"]
        assert first_row == want["increment_volts_first_row"]
        assert bool(truncated) == want["increment_volts_truncated"]


def test_a_real_fill_changes_only_the_three_columns(tmp_path, monkeypatch):
    """The missing-file test below also proves 'nothing else changes', but trivially --
    no write happens there at all. This is the same check where writes actually DO happen,
    which is the case a mutation (e.g. touching created_date, or the wrong row) would slip
    past a check that only ever runs on a no-op path."""
    db = _build_db(tmp_path, monkeypatch, [LTS, LTS_194])
    _null_out(db)
    before_other = _other_columns_snapshot(db)
    before_tr, before_ar = _other_tables_snapshot(db)

    report = biv.backfill(db)
    assert report.passes_filled > 0
    assert report.files_updated == 2

    assert _other_columns_snapshot(db) == before_other, \
        "a real fill must not touch any OTHER trim_passes column"
    assert _other_tables_snapshot(db) == (before_tr, before_ar), \
        "a real fill must not touch track_results or analysis_results at all"


def test_laser_2_and_lin_error_rows_are_never_candidates_or_touched(tmp_path, monkeypatch):
    db = _build_db(tmp_path, monkeypatch, [LTS, DLTS])
    _null_out(db)

    candidates = biv._select_candidates(db)
    touched_files = {Path(f.file_path).name for f in candidates}
    assert DLTS.name not in touched_files, "laser 2 has no Trim-N TrimVolts capture at all"
    for f in candidates:
        assert all(p.sheet.lower() != "lin error" for p in f.passes)

    biv.backfill(db)
    with db.session() as s:
        dlts_rows = s.execute(sa.text(
            "SELECT p.increment_volts FROM trim_passes p "
            "JOIN track_results t ON t.id = p.track_result_id "
            "JOIN analysis_results a ON a.id = t.analysis_id WHERE a.filename = :fn"),
            {"fn": DLTS.name}).all()
        lin_error_rows = s.execute(sa.text(
            "SELECT p.increment_volts FROM trim_passes p "
            "JOIN track_results t ON t.id = p.track_result_id "
            "JOIN analysis_results a ON a.id = t.analysis_id "
            "WHERE a.filename = :fn AND p.sheet = 'Lin Error'"), {"fn": LTS.name}).all()
    assert dlts_rows and all(r[0] is None for r in dlts_rows)
    assert lin_error_rows and all(r[0] is None for r in lin_error_rows)


# ==================================================================== 3. resume, limit, ETA

def test_a_second_run_touches_nothing_already_filled(tmp_path, monkeypatch):
    db = _build_db(tmp_path, monkeypatch, [LTS, LTS_194])
    _null_out(db)

    first = biv.backfill(db)
    assert first.passes_filled > 0
    targets_after_1 = _target_columns_snapshot(db)
    other_after_1 = _other_columns_snapshot(db)

    second = biv.backfill(db)
    assert second.candidates_total == 0
    assert second.files_run == 0
    assert second.passes_filled == 0
    assert second.files_updated == 0

    assert _target_columns_snapshot(db) == targets_after_1, "resumed run rewrote a filled row"
    assert _other_columns_snapshot(db) == other_after_1


def test_limit_bounds_a_run_and_a_second_run_finishes_the_rest(tmp_path, monkeypatch):
    db = _build_db(tmp_path, monkeypatch, [LTS, LTS_194])
    _null_out(db)

    first = biv.backfill(db, limit=1)
    assert first.candidate_files_total == 2
    assert first.files_run == 1
    assert first.passes_filled > 0
    remaining_null = sum(1 for v in _target_columns_snapshot(db).values() if v[0] is None)
    assert remaining_null > 0, "the second file must still be untouched"

    second = biv.backfill(db)
    assert second.files_run == 1
    assert second.passes_filled > 0

    third = biv.backfill(db)
    assert third.candidates_total == 0
    assert all(v is not None for v in _trim_n_increment_volts(db))


def test_progress_fires_every_progress_every_files_and_once_at_the_end(tmp_path, monkeypatch):
    from laser_trim_analyzer.database import manager as mgr
    db = mgr.DatabaseManager(tmp_path / "prog.db")
    monkeypatch.setattr(biv, "_select_candidates", lambda _db: _fake_files(7))
    monkeypatch.setattr(biv, "_capture_file",
                        lambda path, sheets: ({s: _FAKE_CAPTURE for s in sheets}, None))

    seen = []
    biv.backfill(db, progress_every=3, progress=lambda *a: seen.append(a))

    dones = [c[0] for c in seen]
    assert dones == [3, 6, 7], f"expected progress at 3, 6 and the final 7, got {dones}"
    done, total, name, rate, eta_text = seen[-1]
    assert (done, total) == (7, 7)
    assert isinstance(name, str) and name
    assert rate is None or isinstance(rate, float)
    assert isinstance(eta_text, str)
    assert eta_text == "", "nothing left to predict at the final call"


# ============================================================ 4. missing / unreadable files

def test_a_missing_file_is_counted_named_and_changes_nothing_else(tmp_path, monkeypatch):
    db = _build_db(tmp_path, monkeypatch, [LTS])
    _null_out(db)
    _set_file_path(db, LTS.name, tmp_path / "share" / "gone-missing.xls")

    before_other = _other_columns_snapshot(db)
    before_targets = _target_columns_snapshot(db)
    before_tr, before_ar = _other_tables_snapshot(db)

    report = biv.backfill(db)
    assert report.files_missing == 1
    assert report.files_unreadable == 0
    assert report.passes_filled == 0
    assert report.files_updated == 0
    assert any("gone-missing.xls" in n for n in report.missing_files)

    assert _other_columns_snapshot(db) == before_other
    assert _target_columns_snapshot(db) == before_targets, "still all NULL -- nothing filled"
    assert _other_tables_snapshot(db) == (before_tr, before_ar)


def test_an_unreadable_file_is_counted_and_named_and_is_not_fatal(tmp_path, monkeypatch):
    db = _build_db(tmp_path, monkeypatch, [LTS, LTS_194])
    _null_out(db)
    junk = tmp_path / "not_really_excel.xls"
    junk.write_text("this is not a workbook")
    _set_file_path(db, LTS.name, junk)

    report = biv.backfill(db)          # must not raise -- LTS_194 must still be processed
    assert report.files_unreadable == 1
    assert any("not_really_excel.xls" in n for n in report.unreadable_files)
    assert report.passes_filled > 0, "the OTHER file must still be backfilled"

    with db.session() as s:
        still_null = s.execute(sa.text(
            "SELECT p.increment_volts FROM trim_passes p "
            "JOIN track_results t ON t.id = p.track_result_id "
            "JOIN analysis_results a ON a.id = t.analysis_id "
            "WHERE a.filename = :fn AND p.sheet LIKE 'Trim %'"), {"fn": LTS.name}).all()
    assert all(r[0] is None for r in still_null)


# =============================================================================== 5. dry-run

def test_dry_run_reports_without_writing(tmp_path, monkeypatch):
    db = _build_db(tmp_path, monkeypatch, [LTS])
    _null_out(db)
    before = _target_columns_snapshot(db)

    report = biv.backfill(db, dry_run=True)
    assert report.dry_run is True
    assert report.passes_filled > 0
    assert report.files_updated > 0

    after = _target_columns_snapshot(db)
    assert after == before
    assert all(v == (None, None, None) for v in after.values())


# ============================================================= 6. batching (files, not rows)

def test_a_small_run_commits_in_one_transaction(tmp_path, monkeypatch):
    from laser_trim_analyzer.database import manager as mgr
    db = mgr.DatabaseManager(tmp_path / "batch_small.db")
    monkeypatch.setattr(biv, "_select_candidates", lambda _db: _fake_files(7))
    monkeypatch.setattr(biv, "_capture_file",
                        lambda path, sheets: ({s: _FAKE_CAPTURE for s in sheets}, None))
    calls = []
    monkeypatch.setattr(db, "session", _counting_session(db, calls))

    report = biv.backfill(db, batch_size=200)
    assert report.files_run == 7
    assert len(calls) == 1, f"7 files should be ONE transaction, was {len(calls)}"


def test_the_batch_writer_chunks_at_two_hundred_files(tmp_path, monkeypatch):
    from laser_trim_analyzer.database import manager as mgr
    db = mgr.DatabaseManager(tmp_path / "batch_big.db")
    monkeypatch.setattr(biv, "_select_candidates", lambda _db: _fake_files(450))
    monkeypatch.setattr(biv, "_capture_file",
                        lambda path, sheets: ({s: _FAKE_CAPTURE for s in sheets}, None))
    calls = []
    monkeypatch.setattr(db, "session", _counting_session(db, calls))

    report = biv.backfill(db, batch_size=200)
    assert report.files_run == 450
    assert report.passes_filled == 450
    assert len(calls) == 3, f"450 files at batch_size=200 should be 3 transactions, was {len(calls)}"


def test_a_batch_boundary_mid_run_still_leaves_every_file_filled(tmp_path, monkeypatch):
    """Not just the transaction COUNT -- the batching must not drop or duplicate work."""
    db = _build_db(tmp_path, monkeypatch, [LTS, LTS_194])
    _null_out(db)
    report = biv.backfill(db, batch_size=1)     # every single file is its own transaction
    assert report.files_run == 2
    assert all(v is not None for v in _trim_n_increment_volts(db))


# ======================================================================== 1. CLI: the guard

def test_refuses_without_a_db_path(capsys):
    rc = biv.main([])
    assert rc == 2
    out = capsys.readouterr().out
    assert "snapshot_db.py" in out


def test_refuses_a_nonexistent_db_path_and_creates_nothing(tmp_path, capsys):
    missing = tmp_path / "nope.db"
    rc = biv.main([str(missing)])
    assert rc == 1
    assert not missing.exists(), "must never CREATE a database just by naming it"
    assert "FATAL" in capsys.readouterr().out


def test_main_prints_what_it_will_touch_and_a_snapshot_command(tmp_path, monkeypatch, capsys):
    _build_db(tmp_path, monkeypatch, [LTS])
    _null_out_path(tmp_path / "t.db")

    rc = biv.main([str(tmp_path / "t.db")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "snapshot_db.py" in out
    assert "still missing their curves" in out
    assert "This run will touch" in out


def test_main_dry_run_flag_writes_nothing(tmp_path, monkeypatch, capsys):
    db = _build_db(tmp_path, monkeypatch, [LTS])
    _null_out(db)
    before = _target_columns_snapshot(db)

    rc = biv.main([str(tmp_path / "t.db"), "--dry-run"])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert _target_columns_snapshot(db) == before
