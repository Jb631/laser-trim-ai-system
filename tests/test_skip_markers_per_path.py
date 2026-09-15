"""Skip markers are keyed by PATH, not by content hash (2026-09-14).

The bug these pin down: `processed_files` has UNIQUE(file_hash), and
`mark_file_skipped` used to return early whenever ANY row already held the
same hash. Every empty file on the work share hashes to the SHA-256 of b"",
so the one row for `~$7029-72.xlsx` matched all 832 other empty files and
none of them was recorded under its own path. The processor logged
"recorded as skipped" for each while the write was silently dropped, and the
scan re-offered them as "new" every day forever (~930 files a run).

The fix stores a synthetic per-path `skip_marker_hash()` instead, so the
existing unique constraint now means "one marker per path" — no schema
change. These tests cover the fix AND its four consequences: the scan skips
the path next run, a CHANGED file is still re-parsed, the hash-keyed stat
heal never touches markers, and the trim recording path (which does
`scalar_one_or_none()` by content hash) cannot see a marker.
"""

import hashlib
import os
from datetime import datetime
from pathlib import Path

import pytest

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import ProcessedFile

EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def _db(tmp_path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "markers.db")


class _Row:
    """A detached snapshot — ORM instances expire when the session closes."""

    def __init__(self, r):
        self.file_path = r.file_path
        self.filename = r.filename
        self.file_hash = r.file_hash
        self.file_size = r.file_size
        self.file_modified_date = r.file_modified_date
        self.error_message = r.error_message
        self.success = r.success
        self.analysis_id = r.analysis_id


def _markers(db):
    with db.session() as s:
        return {r.file_path: _Row(r) for r in s.query(ProcessedFile).all()}


def _mark_empty(db, path: Path, reason=None):
    st = path.stat()
    db.mark_file_skipped(
        filename=path.name, file_path=str(path),
        file_hash=EMPTY_SHA, file_size=st.st_size,
        file_modified_date=datetime.fromtimestamp(st.st_mtime),
        error_message=reason,
    )


# --------------------------------------------------------------------------
# (a) two different-named empty files both get skip markers
# --------------------------------------------------------------------------

def test_two_empty_files_both_get_markers(tmp_path):
    a = tmp_path / "Final Test 8084-sn100_8-24-2011_2-44 PM.xls"
    b = tmp_path / "Final Test 8084-sn107_8-27-2011_9-17 AM.xls"
    a.write_bytes(b"")
    b.write_bytes(b"")
    assert hashlib.sha256(a.read_bytes()).hexdigest() == EMPTY_SHA

    db = _db(tmp_path)
    _mark_empty(db, a)
    _mark_empty(db, b)

    rows = _markers(db)
    # THE bug: before the fix this was 1 — b collided with a on file_hash
    # and was dropped without an error.
    assert len(rows) == 2, f"both empty files must be recorded, got {list(rows)}"
    assert str(a) in rows and str(b) in rows
    # Each carries its own synthetic hash, and neither is the content hash.
    assert rows[str(a)].file_hash != rows[str(b)].file_hash
    for r in rows.values():
        assert r.file_hash.startswith("skip:")
        assert len(r.file_hash) == 64, "check_pf_hash_length demands exactly 64"
        assert r.file_hash != EMPTY_SHA
        assert r.success is True and r.analysis_id is None  # a skip marker
        # The real content hash is not lost, just not the identity.
        assert EMPTY_SHA in (r.error_message or "")


def test_remarking_same_path_updates_rather_than_duplicates(tmp_path):
    p = tmp_path / "empty.xls"
    p.write_bytes(b"")
    db = _db(tmp_path)
    _mark_empty(db, p, reason="first reason")
    _mark_empty(db, p, reason="second reason")

    rows = _markers(db)
    assert len(rows) == 1
    assert "second reason" in rows[str(p)].error_message


def test_marker_records_the_reason(tmp_path):
    p = tmp_path / "junk.xls"
    p.write_bytes(b"")
    db = _db(tmp_path)
    _mark_empty(db, p, reason="Excel file format cannot be determined")
    row = _markers(db)[str(p)]
    assert "Excel file format cannot be determined" in row.error_message


# --------------------------------------------------------------------------
# (b) a marked file is classified "processed" by a fresh scan next load
# --------------------------------------------------------------------------

def _processor_loaded_from(db, monkeypatch):
    """A fresh Processor whose caches are loaded from `db`."""
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.config import Config
    import laser_trim_analyzer.database as db_mod

    monkeypatch.setattr(db_mod, "get_database", lambda *a, **k: db)
    proc = Processor(Config(), use_ml=False)
    proc._load_processed_hashes()
    return proc


def test_marked_file_is_processed_on_next_scan(tmp_path, monkeypatch):
    # The SECOND empty file, so the scenario is the real one: its content
    # hash collides with the first file's, and before the fix it therefore
    # had no row of its own and came back as "new" on every scan.
    other = tmp_path / "Final Test 7029-81.xls"
    other.write_bytes(b"")
    p = tmp_path / "Final Test 7029-82.xls"
    p.write_bytes(b"")
    db = _db(tmp_path)
    _mark_empty(db, other, reason="Excel file format cannot be determined")
    _mark_empty(db, p, reason="Excel file format cannot be determined")

    proc = _processor_loaded_from(db, monkeypatch)
    st = p.stat()
    proc._disk_stats = {str(p): (st.st_size, st.st_mtime)}

    # Memory alone, no hashing: the path is known and its stat still matches.
    assert proc._classify_scan(p) == "processed"


def test_marked_file_needs_no_hash_read(tmp_path, monkeypatch):
    other = tmp_path / "Final Test 7029-81.xls"
    other.write_bytes(b"")
    p = tmp_path / "Final Test 7029-83.xls"
    p.write_bytes(b"")
    db = _db(tmp_path)
    _mark_empty(db, other)
    _mark_empty(db, p)

    proc = _processor_loaded_from(db, monkeypatch)
    st = p.stat()
    proc._disk_stats = {str(p): (st.st_size, st.st_mtime)}

    def boom(*a, **k):
        raise AssertionError("a marked file was hashed on a clean re-scan")

    monkeypatch.setattr(
        "laser_trim_analyzer.core.processor.calculate_file_hash", boom)
    assert proc._is_processed(p) is True


# --------------------------------------------------------------------------
# (d) a marker whose file CHANGED is not treated as processed
# --------------------------------------------------------------------------

def test_changed_file_is_reoffered(tmp_path, monkeypatch):
    other = tmp_path / "Final Test 7029-81.xls"
    other.write_bytes(b"")
    p = tmp_path / "Final Test 7029-84.xls"
    p.write_bytes(b"")
    db = _db(tmp_path)
    _mark_empty(db, other)
    _mark_empty(db, p)

    proc = _processor_loaded_from(db, monkeypatch)
    # Precondition: the marker exists at all (it did not, before the fix).
    assert str(p) in proc._processed_filenames

    # The file is replaced with real content (someone re-exported it).
    p.write_bytes(b"a real workbook now, with content")
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 3600))
    st = p.stat()
    proc._disk_stats = {str(p): (st.st_size, st.st_mtime)}

    # Stat mismatch -> memory can't settle it.
    assert proc._classify_scan(p) == "needs_hash"
    # …and the I/O resolution computes the REAL hash, which is not in the
    # cache (the marker contributed only its synthetic one), so it is new.
    is_processed, repair = proc._resolve_scan_io(p)
    assert is_processed is False, "a changed file must be re-parsed"
    assert proc._is_processed(p) is False


# --------------------------------------------------------------------------
# (c) the hash-keyed stat heal never touches marker rows
# --------------------------------------------------------------------------

def test_stat_heal_does_not_touch_markers(tmp_path):
    p = tmp_path / "empty.xls"
    p.write_bytes(b"")
    db = _db(tmp_path)
    _mark_empty(db, p)

    # The heal keys on the REAL content hash. A marker must be invisible to
    # it — otherwise one empty file's heal would rewrite another's stat.
    n = db.update_processed_file_stats([(EMPTY_SHA, 999, datetime(2026, 7, 6))])
    assert n["processed_files"] == 0
    row = _markers(db)[str(p)]
    assert row.file_size == 0, "marker stat was overwritten by a content heal"


# --------------------------------------------------------------------------
# (d, cont.) the trim recording path cannot trip over markers
# --------------------------------------------------------------------------

def test_trim_recording_unaffected_by_markers(tmp_path):
    """manager.py's record path does scalar_one_or_none() BY CONTENT HASH.

    Two rows sharing a hash would raise MultipleResultsFound and break the
    main trim path — which is why the marker hash is synthetic and unique
    per path rather than the real content hash.
    """
    db = _db(tmp_path)
    for name in ("empty_a.xls", "empty_b.xls", "empty_c.xls"):
        f = tmp_path / name
        f.write_bytes(b"")
        _mark_empty(db, f)
    assert len(_markers(db)) == 3

    # A real trim file that happens to also be empty-hashed content-wise:
    # look it up the way the recording path does.
    from sqlalchemy import select
    with db.session() as s:
        got = s.execute(
            select(ProcessedFile.id).where(ProcessedFile.file_hash == EMPTY_SHA)
        ).scalar_one_or_none()          # must not raise MultipleResultsFound
    assert got is None, "markers must be invisible to content-hash lookups"

    # And is_file_processed (also content-keyed) is likewise unconfused.
    real = tmp_path / "real_unit.xls"
    real.write_bytes(b"")
    assert db.is_file_processed(real) is False


# --------------------------------------------------------------------------
# (e) trim ERROR rows stay success=False and keep being retried
# --------------------------------------------------------------------------

def test_error_rows_stay_retryable(tmp_path, monkeypatch):
    """The 112 DLTS junk rows are retried BY DESIGN so a parser upgrade can
    fix them. Nothing here may quietly convert one into a skip marker."""
    p = tmp_path / "dlts_junk.xls"
    p.write_bytes(b"broken")
    db = _db(tmp_path)
    st = p.stat()
    with db.session() as s:
        s.add(ProcessedFile(
            filename=p.name, file_path=str(p),
            file_hash="cd" * 32, file_size=st.st_size,
            file_modified_date=datetime.fromtimestamp(st.st_mtime),
            error_message="Could not find data start",
            analysis_id=None, success=False,
        ))
        s.commit()

    # Marking the same path must not flip success to True.
    db.mark_file_skipped(
        filename=p.name, file_path=str(p), file_hash="cd" * 32,
        file_size=st.st_size,
        file_modified_date=datetime.fromtimestamp(st.st_mtime),
        error_message="still broken",
    )
    row = _markers(db)[str(p)]
    assert row.success is False, "an ERROR row must stay retryable"
    assert row.analysis_id is None

    # And the scan still offers it: success=False rows are excluded from the
    # processed caches entirely.
    proc = _processor_loaded_from(db, monkeypatch)
    proc._disk_stats = {str(p): (st.st_size, st.st_mtime)}
    assert proc._classify_scan(p) == "new"


# --------------------------------------------------------------------------
# (c) a duplicate FT save under a new path gets a marker
# --------------------------------------------------------------------------

def _ft_metadata(path: Path, serial="sn1"):
    return {
        "filename": path.name,
        "file_path": str(path),
        "model": "8084",
        "serial": serial,
        "file_date": datetime(2011, 8, 24),
        "test_date": datetime(2011, 8, 24),
    }


def _ft_tracks():
    return [{
        "track_id": "default",
        "positions": [0.0, 1.0, 2.0],
        "errors": [0.001, 0.002, 0.001],
        "linearity_pass": True,
    }]


def test_duplicate_ft_save_marks_the_new_path(tmp_path):
    """The IntegrityError ("race condition") branch, which is what the work
    log shows 60 times a run: the content hash DIFFERS (so the up-front hash
    check misses) but uq_final_test_file — filename+file_date+model+serial —
    still matches, so the row is already on record under another path."""
    db = _db(tmp_path)
    first = tmp_path / "Final Test 8084-sn1_8-24-2011_2-44 PM.xls"
    first.write_bytes(b"ft workbook bytes")
    second = tmp_path / "copy" / first.name          # same NAME, new folder
    second.parent.mkdir()
    second.write_bytes(b"ft workbook bytes, re-exported with an edit")

    st = first.stat()
    first_id = db.save_final_test(
        metadata=_ft_metadata(first), tracks=_ft_tracks(), test_results={},
        file_hash=hashlib.sha256(first.read_bytes()).hexdigest(),
        file_size=st.st_size,
        file_modified_date=datetime.fromtimestamp(st.st_mtime),
    )
    assert first_id

    st2 = second.stat()
    dup_id = db.save_final_test(
        metadata=_ft_metadata(second), tracks=_ft_tracks(), test_results={},
        file_hash=hashlib.sha256(second.read_bytes()).hexdigest(),
        file_size=st2.st_size,
        file_modified_date=datetime.fromtimestamp(st2.st_mtime),
    )
    assert dup_id == first_id, "the duplicate must resolve to the existing row"

    rows = _markers(db)
    assert str(second) in rows, "the duplicate's path must be marked"
    row = rows[str(second)]
    assert f"duplicate of final_test_results id {first_id}" in row.error_message
    assert row.file_hash.startswith("skip:")
    assert row.success is True and row.analysis_id is None
    # The ORIGINAL path is untouched — it is a real record, not a marker.
    assert str(first) not in rows


# item 4 — the summary line says how many were recorded as unreadable
# --------------------------------------------------------------------------

def test_summary_line_reports_unreadable_count():
    from laser_trim_analyzer.core.ingest_run import (
        FolderResult, IngestReport, format_ingest_summary)

    rep = IngestReport(results=[FolderResult(folder="f", ok=True, new_files=4)],
                       seconds=12.0)
    assert "unreadable" not in format_ingest_summary(rep)  # silent at zero

    rep.marked_unreadable = 931
    line = format_ingest_summary(rep)
    assert "931 files could not be read" in line
    assert "recorded as unreadable" in line

    rep.marked_unreadable = 1
    assert "1 file could not be read and was" in format_ingest_summary(rep)
