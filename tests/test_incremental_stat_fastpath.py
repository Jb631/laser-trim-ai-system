"""Incremental scan stat fast-path (2026-07-06 processing-hang fix).

The "checking against database" stage used to hash (fully read) every file
whose path was already known. On network shares that reads every byte of
every previously-processed file per scan. Now: a path-hit whose recorded
size+mtime still match the file on disk is skipped WITHOUT reading it;
hash-confirm remains the fallback, and a successful hash-confirm repairs
the stat record so the next scan is fast (self-healing).
"""

import os
from datetime import datetime
from pathlib import Path

import pytest

from laser_trim_analyzer.utils.hashing import calculate_file_hash


def _make_processor():
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.config import Config
    return Processor(Config(), use_ml=False)


def _prime(proc, path: Path, size=None, mtime=None, file_hash=None):
    """Simulate a loaded cache containing `path`."""
    proc._processed_filenames = {str(path)}
    proc._processed_hashes = {file_hash} if file_hash else set()
    proc._processed_stat = {}
    if size is not None and mtime is not None:
        proc._processed_stat[str(path)] = (size, mtime)


@pytest.fixture()
def sample(tmp_path):
    p = tmp_path / "unit.xls"
    p.write_bytes(b"trim data v1")
    return p


def test_stat_match_skips_without_reading(sample, monkeypatch):
    proc = _make_processor()
    st = sample.stat()
    _prime(proc, sample, size=st.st_size, mtime=st.st_mtime)

    def boom(*a, **k):
        raise AssertionError("file was read (hashed) despite stat match")

    monkeypatch.setattr("laser_trim_analyzer.core.processor.calculate_file_hash", boom)
    assert proc._is_processed(sample) is True


def test_unknown_path_is_new_without_reading(sample, monkeypatch):
    proc = _make_processor()
    proc._processed_filenames = set()
    proc._processed_hashes = set()
    proc._processed_stat = {}
    monkeypatch.setattr(
        "laser_trim_analyzer.core.processor.calculate_file_hash",
        lambda *a, **k: pytest.fail("hashed a brand-new path"),
    )
    assert proc._is_processed(sample) is False


def test_stale_stat_falls_back_to_hash_and_heals(sample):
    proc = _make_processor()
    real_hash = calculate_file_hash(sample, use_cache=False)
    st = sample.stat()
    # Recorded stat is stale (wrong size) but content hash matches.
    _prime(proc, sample, size=st.st_size + 999, mtime=st.st_mtime, file_hash=real_hash)

    assert proc._is_processed(sample) is True
    # Heal queued with the real stat, keyed by content hash…
    assert len(proc._stat_heal) == 1
    healed_hash, healed_size, healed_mtime = proc._stat_heal[0]
    assert healed_hash == real_hash
    assert healed_size == st.st_size
    assert isinstance(healed_mtime, datetime)
    # …and the in-memory cache is repaired so this scan won't re-hash it.
    assert proc._processed_stat[str(sample)] == (st.st_size, st.st_mtime)


def test_changed_content_is_not_skipped(sample):
    proc = _make_processor()
    old_hash = calculate_file_hash(sample, use_cache=False)
    st = sample.stat()
    _prime(proc, sample, size=st.st_size, mtime=st.st_mtime, file_hash=old_hash)

    # Re-export new content to the same filename: size+mtime forced different.
    sample.write_bytes(b"trim data v2 -- different and longer")
    os.utime(sample, (st.st_atime, st.st_mtime + 3600))

    assert proc._is_processed(sample) is False
    assert proc._stat_heal == []


def test_missing_stat_record_hash_confirms(sample):
    proc = _make_processor()
    real_hash = calculate_file_hash(sample, use_cache=False)
    _prime(proc, sample, file_hash=real_hash)  # no stat record at all

    assert proc._is_processed(sample) is True
    assert len(proc._stat_heal) == 1  # healed for next time


def test_update_processed_file_stats_persists(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "t.db")
    src = tmp_path / "a.xls"
    src.write_bytes(b"x" * 100)
    fh = "ab" * 32  # valid 64-char SHA-256 shape
    db.mark_file_skipped(
        filename="a.xls", file_path=str(src),
        file_hash=fh, file_size=1,
        file_modified_date=datetime(2020, 1, 1),
    )
    n = db.update_processed_file_stats([(fh, 100, datetime(2026, 7, 6))])
    assert n == 1

    from laser_trim_analyzer.database.models import ProcessedFile
    with db.session() as s:
        row = s.query(ProcessedFile).filter_by(file_hash=fh).one()
        assert row.file_size == 100
        assert row.file_modified_date == datetime(2026, 7, 6)
