"""Final Test / smoothness rows carry (size, mtime) so the incremental scan
never re-reads them (2026-08-29 processing-speed fix).

The trim path has had the stat fast-path since 2026-07-06, but
final_test_results and smoothness_results stored no size/mtime. Every known FT
file therefore fell through to calculate_file_hash — a full content read over
the network share — on EVERY scan, and the heal pass only repaired
processed_files, so it never got better. These tests pin:
  * save_final_test / save_smoothness_result persist the disk stat
  * _load_processed_hashes exposes it in _processed_stat
  * _is_processed then skips WITHOUT hashing
  * update_processed_file_stats heals legacy FT/smoothness rows by hash
"""

from datetime import datetime
from pathlib import Path

import pytest


def _make_processor():
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.processor import Processor
    return Processor(Config(), use_ml=False)


def _ft_meta(path: Path):
    return {
        "filename": path.name,
        "file_path": str(path),
        "model": "6607",
        "serial": "s1",
        "file_date": datetime(2026, 8, 1),
        "test_date": datetime(2026, 8, 1),
    }


@pytest.fixture()
def ft_file(tmp_path):
    p = tmp_path / "6607_s1_FINAL TEST_8-1-2026_9-00 AM.xls"
    p.write_bytes(b"final test data v1")
    return p


def test_save_final_test_records_disk_stats(tmp_path, ft_file):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import FinalTestResult

    db = DatabaseManager(tmp_path / "ft.db")
    st = ft_file.stat()
    db.save_final_test(
        metadata=_ft_meta(ft_file), tracks=[], test_results={},
        file_hash="ab" * 32,
        file_size=st.st_size,
        file_modified_date=datetime.fromtimestamp(st.st_mtime),
    )
    with db.session() as s:
        row = s.query(FinalTestResult).one()
        assert row.file_size == st.st_size
        assert row.file_modified_date == datetime.fromtimestamp(st.st_mtime)


def test_known_ft_file_skips_without_hashing(tmp_path, ft_file, monkeypatch):
    """The whole point: a recorded FT file is recognised from memory alone."""
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "ft.db")
    st = ft_file.stat()
    db.save_final_test(
        metadata=_ft_meta(ft_file), tracks=[], test_results={},
        file_hash="ab" * 32,
        file_size=st.st_size,
        file_modified_date=datetime.fromtimestamp(st.st_mtime),
    )

    monkeypatch.setattr("laser_trim_analyzer.database.get_database", lambda: db)
    proc = _make_processor()
    proc._load_processed_hashes()
    assert proc._processed_stat.get(str(ft_file)) is not None

    # Discovery supplies the stat (scandir gives it with the listing), so the
    # check must not even stat(), let alone read the file.
    proc._disk_stats = {str(ft_file): (st.st_size, st.st_mtime)}

    def boom(*a, **k):
        raise AssertionError("should not hash")

    monkeypatch.setattr("laser_trim_analyzer.core.processor.calculate_file_hash", boom)
    assert proc._is_processed(ft_file) is True


def test_known_ft_file_skips_under_new_path_form(tmp_path, ft_file, monkeypatch):
    """Same file reached by a different path form (mapped drive vs UNC) is
    rescued by basename + stat — still without hashing."""
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "ft.db")
    st = ft_file.stat()
    meta = _ft_meta(ft_file)
    meta["file_path"] = "/mnt/share/ft/" + ft_file.name   # recorded under the old mount
    db.save_final_test(
        metadata=meta, tracks=[], test_results={}, file_hash="cd" * 32,
        file_size=st.st_size,
        file_modified_date=datetime.fromtimestamp(st.st_mtime),
    )

    monkeypatch.setattr("laser_trim_analyzer.database.get_database", lambda: db)
    proc = _make_processor()
    proc._load_processed_hashes()
    proc._disk_stats = {str(ft_file): (st.st_size, st.st_mtime)}
    monkeypatch.setattr(
        "laser_trim_analyzer.core.processor.calculate_file_hash",
        lambda *a, **k: pytest.fail("hashed a basename+stat match"),
    )
    assert proc._is_processed(ft_file) is True


def test_changed_ft_content_is_not_skipped(tmp_path, ft_file, monkeypatch):
    """Safety: a re-export to the same path with new content still processes."""
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "ft.db")
    st = ft_file.stat()
    db.save_final_test(
        metadata=_ft_meta(ft_file), tracks=[], test_results={},
        file_hash="ab" * 32,
        file_size=st.st_size,
        file_modified_date=datetime.fromtimestamp(st.st_mtime),
    )
    monkeypatch.setattr("laser_trim_analyzer.database.get_database", lambda: db)
    proc = _make_processor()
    proc._load_processed_hashes()
    # New content: size and mtime both differ from what was recorded.
    proc._disk_stats = {str(ft_file): (st.st_size + 500, st.st_mtime + 3600)}
    ft_file.write_bytes(b"final test data v2 -- different and longer")
    assert proc._is_processed(ft_file) is False


def test_smoothness_row_records_and_uses_disk_stats(tmp_path, monkeypatch):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import SmoothnessResult

    p = tmp_path / "6607_s2_OUTPUT SMOOTHNESS_8-1-2026.xls"
    p.write_bytes(b"smoothness data")
    db = DatabaseManager(tmp_path / "os.db")
    st = p.stat()
    db.save_smoothness_result(
        metadata={"filename": p.name, "file_path": str(p), "model": "6607",
                  "serial": "s2", "file_date": datetime(2026, 8, 1)},
        tracks=[{"track_id": "default", "smoothness_pass": True,
                 "max_smoothness": 0.01, "avg_smoothness": 0.005}],
        file_hash="ef" * 32,
        file_size=st.st_size,
        file_modified_date=datetime.fromtimestamp(st.st_mtime),
    )
    with db.session() as s:
        row = s.query(SmoothnessResult).one()
        assert row.file_size == st.st_size
        assert row.file_modified_date == datetime.fromtimestamp(st.st_mtime)

    monkeypatch.setattr("laser_trim_analyzer.database.get_database", lambda: db)
    proc = _make_processor()
    proc._load_processed_hashes()
    proc._disk_stats = {str(p): (st.st_size, st.st_mtime)}
    monkeypatch.setattr(
        "laser_trim_analyzer.core.processor.calculate_file_hash",
        lambda *a, **k: pytest.fail("hashed a known smoothness file"),
    )
    assert proc._is_processed(p) is True


def test_update_processed_file_stats_heals_ft_and_smoothness(tmp_path):
    """First post-deploy scan hash-confirms once; the heal must land on the
    FT/smoothness row too, or every later scan pays the hash again."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        FinalTestResult, SmoothnessResult)

    db = DatabaseManager(tmp_path / "heal.db")
    ft_hash, os_hash = "11" * 32, "22" * 32
    db.save_final_test(
        metadata=_ft_meta(tmp_path / "legacy_ft.xls"), tracks=[],
        test_results={}, file_hash=ft_hash)          # legacy row: NULL stats
    db.save_smoothness_result(
        metadata={"filename": "legacy_os.xls", "file_path": "/legacy/os.xls",
                  "model": "6607", "serial": "s3"},
        tracks=[{"track_id": "default", "smoothness_pass": True}],
        file_hash=os_hash)

    n = db.update_processed_file_stats([
        (ft_hash, 4321, datetime(2026, 8, 29)),
        (os_hash, 8765, datetime(2026, 8, 28)),
    ])
    assert n["final_test_results"] == 1 and n["smoothness_results"] == 1
    assert n["processed_files"] == 0 and n["total"] == 2

    with db.session() as s:
        ft = s.query(FinalTestResult).filter_by(file_hash=ft_hash).one()
        assert (ft.file_size, ft.file_modified_date) == (4321, datetime(2026, 8, 29))
        os_row = s.query(SmoothnessResult).filter_by(file_hash=os_hash).one()
        assert (os_row.file_size, os_row.file_modified_date) == (8765, datetime(2026, 8, 28))


def _drain(gen):
    """Run a process_batch generator to completion, returning its summary."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def test_parallel_filter_settles_known_files_in_memory(tmp_path, ft_file, monkeypatch):
    """The parallel path's phase 1 must answer a known folder without touching
    the disk at all — this is what turns a 70-minute FT scan into seconds."""
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "ft.db")
    second = tmp_path / "6607_s2_FINAL TEST_8-1-2026_9-30 AM.xls"
    second.write_bytes(b"final test data 2")
    disk_stats = {}
    for i, f in enumerate((ft_file, second)):
        st = f.stat()
        meta = _ft_meta(f)
        meta["serial"] = f"s{i}"
        db.save_final_test(metadata=meta, tracks=[], test_results={},
                           file_hash=f"{i}" * 64, file_size=st.st_size,
                           file_modified_date=datetime.fromtimestamp(st.st_mtime))
        disk_stats[str(f)] = (st.st_size, st.st_mtime)

    monkeypatch.setattr("laser_trim_analyzer.database.get_database", lambda: db)
    monkeypatch.setattr(
        "laser_trim_analyzer.core.processor.calculate_file_hash",
        lambda *a, **k: pytest.fail("hashed a known file in the parallel filter"),
    )
    proc = _make_processor()
    proc.config.processing.turbo_mode_threshold = 1     # force the parallel path
    summary = _drain(proc.process_batch([ft_file, second], incremental=True,
                                        disk_stats=disk_stats))
    assert summary.skipped == 2 and summary.processed == 0
    assert proc.last_scan_stats["needs_hash"] == 0
    assert proc.last_scan_stats["memory_hits"] == 2


def test_parallel_filter_verifies_and_heals_legacy_rows(tmp_path, ft_file, monkeypatch):
    """Post-deploy first pass: rows with NULL stats are hash-confirmed once in
    the verify pool, then stamped so later scans are memory-only."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import FinalTestResult
    from laser_trim_analyzer.utils.hashing import calculate_file_hash

    db = DatabaseManager(tmp_path / "ft.db")
    real_hash = calculate_file_hash(ft_file, use_cache=False)
    db.save_final_test(metadata=_ft_meta(ft_file), tracks=[], test_results={},
                       file_hash=real_hash)             # legacy row: no stats

    monkeypatch.setattr("laser_trim_analyzer.database.get_database", lambda: db)
    proc = _make_processor()
    proc.config.processing.turbo_mode_threshold = 1
    st = ft_file.stat()
    summary = _drain(proc.process_batch(
        [ft_file], incremental=True,
        disk_stats={str(ft_file): (st.st_size, st.st_mtime)}))
    assert summary.skipped == 1 and summary.processed == 0
    assert proc.last_scan_stats["needs_hash"] == 1      # one verification…
    assert proc.last_scan_stats["heal_updated"] == 1    # …that repaired the row
    with db.session() as s:
        row = s.query(FinalTestResult).one()
        assert row.file_size == st.st_size and row.file_modified_date is not None
