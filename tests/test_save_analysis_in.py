"""The trim save writes into the session it is given (ingest-speed Task 5; spec §3, rulings 7, 10).

`_save_analysis_in(session, analysis, stat, file_hash)` holds what `save_analysis` always did --
the analysis row, its tracks, their trim passes, the setup row and the processed-files marker --
but writes into a session the CALLER owns, so a batch writer (Task 6) can put twenty files in one
transaction. `save_analysis` is now a thin wrapper around it: its own session, one commit, exactly
today's behaviour.

Ruling 10: no file I/O inside a write transaction. The marker records the (size, mtime) and the
content hash it is HANDED -- the parse's own, in the ingest to come -- instead of statting and
hashing the file while the one database connection is held (a ~113 ms share round trip per file,
and a file rewritten between parse and save used to be recorded with the new stat on the old
content). `save_analysis`, which is not handed them, takes them BEFORE it opens its session.

Every stored value must be what today's code stored: the rows are compared WHOLE against the golden
captured before the move (see `save_rows.py`).
"""
from datetime import datetime
from pathlib import Path

import pytest

import save_rows
from save_rows import FIXED_DATE, MTIME


@pytest.fixture
def db(tmp_path, monkeypatch):
    from laser_trim_analyzer.database.manager import DatabaseManager
    d = DatabaseManager(tmp_path / "save.db")
    save_rows.inject(d, monkeypatch)
    yield d
    d.close()


def _pass_result(path: Path, serial="7"):
    from laser_trim_analyzer.core.models import (
        AnalysisResult, AnalysisStatus, FileMetadata, SystemType, TrackData)
    return AnalysisResult(
        metadata=FileMetadata(filename=path.name, file_path=path, file_date=FIXED_DATE,
                              model="8003", serial=serial, system=SystemType.A),
        overall_status=AnalysisStatus.PASS, processing_time=0.01,
        tracks=[TrackData(track_id="TRK1", status=AnalysisStatus.PASS, travel_length=1.0,
                          linearity_spec=0.01, optimal_offset=0.0, linearity_pass=True,
                          linearity_fail_points=0)])


def _processed_rows(db):
    from sqlalchemy import text
    with db.session() as s:
        return [dict(r._mapping) for r in s.execute(text(
            "SELECT file_path, file_hash, file_size, file_modified_date, success, error_message, "
            "analysis_id FROM processed_files ORDER BY id"))]


# ---- the reference: today's rows, whole ---------------------------------------------------------

def test_save_analysis_stores_exactly_todays_rows_for_every_kind(db, tmp_path):
    """Every trim kind, the ERROR shapes, a file gone before its save, and both update paths --
    through the public `save_analysis`, compared column by column with the golden captured from the
    code BEFORE this change. With LTA_REGENERATE_SAVE_ROWS=1 it (re)writes that golden instead."""
    steps = save_rows.build_scenario(tmp_path)
    ids = [(label, db.save_analysis(result)) for label, result in steps]
    snap = save_rows.snapshot(Path(db.database_path), tmp_path, ids)
    if save_rows.REGENERATE:
        save_rows.write_golden(snap)
        pytest.skip(f"regenerated {save_rows.GOLDEN}")
    save_rows.assert_matches_golden(snap)


def test_the_golden_covers_what_it_claims(db, tmp_path):
    """A golden that silently lost a kind would pass forever: pin what it holds."""
    golden = save_rows.load_golden()
    t = golden["tables"]
    systems = {r["system"] for r in t["analysis_results"]}
    statuses = {r["overall_status"] for r in t["analysis_results"]}
    assert {"A", "B", "C"} <= systems and {"UNTRIMMED", "ERROR", "FAIL"} <= statuses
    assert any(r["increment_volts"] not in (None, "null") for r in t["trim_passes"])
    assert any(r["track2_parameters"] is not None for r in t["trim_setup"])
    assert sum(1 for r in t["track_results"] if r["analysis_id"] == 4) == 2       # the two-track file
    pf = t["processed_files"]
    assert [r["id"] for r in pf] == [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12], \
        "row 7 was the first ERROR's retry marker: its clean re-read must delete it"
    kept = [r for r in pf if r["file_hash"] == "<skip marker of this path>"]
    assert [(r["id"], r["analysis_id"], r["file_size"], r["file_modified_date"],
             r["error_message"]) for r in kept] == [
        (9, None, "<size of file>", "<mtime of file>",
         "unreadable: Excel file format cannot be determined, you must specify an engine manually.")]
    failed = [r for r in pf if not r["success"]]
    assert [r["error_message"] for r in failed] == [
        "Excel file format cannot be determined, you must specify an engine manually.",
        "[Errno 13] Permission denied: '8001_1_TEST DATA_1-2-2026_9-00 AM.xls'",
        "TRK1: Insufficient data points"]
    twin = pf[-1]
    assert (twin["file_path"].endswith("8011_1_TEST DATA_1-2-2026_9-00 AM.xls")
            and twin["analysis_id"] == 11), "the same bytes' one row goes to the LAST path saved"
    assert not any("moved_away" in r["file_path"] for r in pf)
    assert [rid for _, rid in golden["ids"]] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 6]


# ---- the session-taking body --------------------------------------------------------------------

def test_the_body_with_carried_values_stores_the_same_rows_and_touches_no_file(db, tmp_path, monkeypatch):
    """The whole scenario through `_save_analysis_in`, each file in a session the TEST owns, with
    every stat, open and hash trapped: identical rows, and not one file touched."""
    steps = save_rows.build_scenario(tmp_path)
    carried = [db._file_identity(result.metadata.file_path) for _, result in steps]
    ids = []
    with save_rows.no_file_io(monkeypatch) as touched:
        for (label, result), (stat, file_hash) in zip(steps, carried):
            with db.session() as s:
                ids.append((label, db._save_analysis_in(s, result, stat, file_hash)))
    assert touched == [], f"the save touched the file system: {touched}"
    save_rows.assert_matches_golden(save_rows.snapshot(Path(db.database_path), tmp_path, ids))


def test_the_marker_records_the_carried_stat_and_hash_not_the_disk(db, tmp_path):
    """Ruling 10: the row describes the bytes that were PARSED. The file on disk has since been
    rewritten (different size, mtime and content); the carried values are what is recorded -- on
    the processed-files row and on an ERROR's retry marker alike."""
    from laser_trim_analyzer.core.processor import Processor
    ok_path, bad_path = tmp_path / "8004_1_TEST DATA.xls", tmp_path / "8005_1_TEST DATA.xls"
    for p in (ok_path, bad_path):
        p.write_bytes(b"rewritten after the parse -- 43 bytes long!")
    parsed_stat = (1234, MTIME)                              # invented: what the parse saw
    parsed_hash, bad_hash = "ab" * 32, "cd" * 32
    proc = Processor(use_ml=False)
    meta = proc._create_minimal_metadata(bad_path)
    meta.file_date = FIXED_DATE
    bad = proc._create_error_result(meta, "No valid track data found", 0.0)
    with db.session() as s:
        db._save_analysis_in(s, _pass_result(ok_path), parsed_stat, parsed_hash)
        db._save_analysis_in(s, bad, parsed_stat, bad_hash)
    parsed_mtime = datetime.fromtimestamp(MTIME).strftime("%Y-%m-%d %H:%M:%S.%f")
    rows = _processed_rows(db)
    assert [(r["file_hash"], r["file_size"], r["file_modified_date"], bool(r["success"]))
            for r in rows] == [
        (parsed_hash, 1234, parsed_mtime, True),
        (bad_hash, 1234, parsed_mtime, False),
        (db.skip_marker_hash(str(bad_path)), 1234, parsed_mtime, True),      # the retry marker
    ]


def test_a_file_gone_since_its_parse_is_recorded_with_the_stat_it_was_parsed_with(db, tmp_path):
    """The carried values need no file: a file moved after its parse is still marked processed.
    (`save_analysis`, which must stat it itself, records nothing for it -- today's behaviour,
    pinned by the golden's 'gone from disk' step.)"""
    gone = tmp_path / "moved" / "8006_2_TEST DATA.xls"
    with db.session() as s:
        aid = db._save_analysis_in(s, _pass_result(gone), (99, MTIME), "ef" * 32)
    assert [(r["file_hash"], r["file_size"], r["analysis_id"]) for r in _processed_rows(db)] == [
        ("ef" * 32, 99, aid)]


def test_no_stat_means_no_processed_row_and_a_stat_without_a_hash_is_refused(db, tmp_path):
    """stat=None is today's "could not stat it": the rows are saved, no marker is. A stat WITHOUT
    a hash is a caller's bug: refused loudly, and the session rolls the rows back with it."""
    from sqlalchemy import text
    a, b = tmp_path / "8007_1_TEST DATA.xls", tmp_path / "8008_1_TEST DATA.xls"
    with db.session() as s:
        db._save_analysis_in(s, _pass_result(a), None, None)
    with pytest.raises(ValueError, match="hash"):
        with db.session() as s:
            db._save_analysis_in(s, _pass_result(b), (5, MTIME), None)
    with db.session() as s:
        names = [r[0] for r in s.execute(text("SELECT filename FROM analysis_results ORDER BY id"))]
    assert names == [a.name] and _processed_rows(db) == []


def test_save_analysis_takes_the_stat_and_hash_before_its_transaction(db, tmp_path, monkeypatch):
    """Ruling 10 for the wrapper too: the stat and the hash happen BEFORE `session()` opens -- so
    the one database connection is never held across a share round trip."""
    from contextlib import contextmanager
    from laser_trim_analyzer.database import manager as mgr
    inside, calls = [False], []
    real_session = mgr.DatabaseManager.session

    @contextmanager
    def watched(self):
        with real_session(self) as s:
            inside[0] = True
            try:
                yield s
            finally:
                inside[0] = False

    def spy(name, real):
        def wrapper(*a, **k):
            calls.append((name, inside[0]))
            return real(*a, **k)
        return wrapper

    monkeypatch.setattr(mgr.DatabaseManager, "session", watched)
    monkeypatch.setattr(mgr, "stat_once", spy("stat_once", mgr.stat_once))
    monkeypatch.setattr(mgr, "calculate_file_hash", spy("calculate_file_hash", mgr.calculate_file_hash))
    f = tmp_path / "8009_1_TEST DATA.xls"
    f.write_bytes(b"invented bytes")
    db.save_analysis(_pass_result(f))
    assert calls == [("stat_once", False), ("calculate_file_hash", False)], calls
    assert [r["file_size"] for r in _processed_rows(db)] == [len(b"invented bytes")]
