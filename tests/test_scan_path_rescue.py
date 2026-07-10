"""Incremental-scan path rescue + runaway-reprocess guard (work incident 2026-07-09).

The work share browsed under a different path form (mapped drive vs UNC vs a
new root) made every FULL-PATH lookup miss: the app set out to reprocess the
entire known history and then bounced off the duplicate constraints saving.
Recognition now rescues by BASENAME (filenames carry model_serial_datetime):
  * recorded (size, mtime) match  -> skip, re-bind path, NO file read
  * unique legacy record, no stat -> adopt (databases predating the stat
    fast-path), NO file read — hashing 78k files over SMB IS the lockup
  * ambiguous                     -> hash identity check (unchanged)
And a non-empty database that recognizes NOTHING in a big batch aborts loudly
instead of reprocessing the world.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _make_processor():
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.config import Config
    return Processor(Config(), use_ml=False)


def _no_read(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("file was read (hashed) — the rescue must be stat-only")
    monkeypatch.setattr("laser_trim_analyzer.core.processor.calculate_file_hash", boom)


@pytest.fixture()
def sample(tmp_path):
    p = tmp_path / "8340-1_7_TEST DATA_3-12-2026_4-56 AM.xls"
    p.write_bytes(b"trim data v1")
    return p


def test_rebinds_new_path_form_by_stat_without_reading(sample, monkeypatch):
    """Same file reached via a new path form: recognized via (size, mtime)."""
    proc = _make_processor()
    st = sample.stat()
    stored = "Z:\\\\Production\\\\" + sample.name          # the OLD path form
    proc._processed_filenames = {stored}
    proc._processed_hashes = set()
    proc._processed_stat = {}
    proc._processed_basename = {sample.name: [(stored, st.st_size, st.st_mtime)]}
    _no_read(monkeypatch)
    assert proc._is_processed(sample) is True
    assert proc._scan_rebound == 1
    # Re-bound: the new path now answers via the exact fast path.
    assert str(sample) in proc._processed_filenames


def test_adopts_unique_legacy_record_without_reading(sample, monkeypatch):
    """Database predates the stat fast-path (all stats NULL): a unique
    basename match is adopted with the observed stat, no hashing."""
    proc = _make_processor()
    stored = "Z:\\\\Production\\\\" + sample.name
    proc._processed_filenames = {stored}
    proc._processed_hashes = set()
    proc._processed_stat = {}
    proc._processed_basename = {sample.name: [(stored, None, None)]}
    _no_read(monkeypatch)
    assert proc._is_processed(sample) is True
    assert proc._scan_adopted == 1
    assert proc._processed_stat[str(sample)][0] == sample.stat().st_size


def test_ambiguous_basename_falls_back_to_hash(sample, monkeypatch):
    """Two folders, same filename, no stats: identity must come from the hash."""
    from laser_trim_analyzer.utils.hashing import calculate_file_hash as real_hash
    proc = _make_processor()
    entries = [("Z:\\\\A\\\\" + sample.name, None, None),
               ("Z:\\\\B\\\\" + sample.name, None, None)]
    proc._processed_filenames = {e[0] for e in entries}
    proc._processed_hashes = {real_hash(sample)}
    proc._processed_stat = {}
    proc._processed_basename = {sample.name: entries}
    assert proc._is_processed(sample) is True          # hash matched
    assert proc._scan_rebound == 1

    # Different content under the ambiguous name -> NOT recognized.
    other = sample.parent / "other" ; other.mkdir()
    p2 = other / sample.name
    p2.write_bytes(b"totally different content")
    proc2 = _make_processor()
    proc2._processed_filenames = {e[0] for e in entries}
    proc2._processed_hashes = {real_hash(sample)}
    proc2._processed_stat = {}
    proc2._processed_basename = {sample.name: entries}
    assert proc2._is_processed(p2) is False


def test_unknown_basename_is_new(sample):
    proc = _make_processor()
    proc._processed_filenames = {"Z:\\\\x\\\\somethingelse.xls"}
    proc._processed_hashes = set()
    proc._processed_stat = {}
    proc._processed_basename = {"somethingelse.xls": [("Z:\\\\x\\\\somethingelse.xls", None, None)]}
    assert proc._is_processed(sample) is False


def test_zero_match_guard_aborts_instead_of_reprocessing(tmp_path):
    """Non-empty DB + zero recognition on a big batch = wrong path/database.
    The batch must abort with a clear message, not reprocess the world."""
    from laser_trim_analyzer.core.models import BatchSummary
    proc = _make_processor()
    proc._processed_filenames = {f"Z:\\\\known\\\\k{i}.xls" for i in range(1500)}
    proc._processed_hashes = set()
    proc._processed_stat = {}
    proc._processed_basename = {}
    files = []
    for i in range(120):                                # >= turbo threshold
        f = tmp_path / f"new_{i}.xls"
        f.write_bytes(b"x")
        files.append(f)
    # Batch start re-loads the cache from the DB; pin the primed state.
    proc._load_processed_hashes = lambda: None
    gen = proc.process_batch(files, incremental=True)
    with pytest.raises(RuntimeError, match="different path|wrong database"):
        list(gen)


def test_guard_silent_when_files_are_recognized(tmp_path, monkeypatch):
    """All files recognized -> no guard, nothing processed, all skipped."""
    proc = _make_processor()
    files, basemap = [], {}
    for i in range(120):
        f = tmp_path / f"k{i}.xls"
        f.write_bytes(b"x")
        st = f.stat()
        stored = f"Z:\\\\old\\\\k{i}.xls"
        basemap[f.name] = [(stored, st.st_size, st.st_mtime)]
        files.append(f)
    proc._processed_filenames = {e[0][0] for e in basemap.values()} | {"pad%d" % i for i in range(1000)}
    proc._processed_hashes = set()
    proc._processed_stat = {}
    proc._processed_basename = basemap
    proc._load_processed_hashes = lambda: None   # pin the primed cache
    _no_read(monkeypatch)
    results = list(proc.process_batch(files, incremental=True))
    assert results == []                                # nothing reprocessed
    assert proc._scan_rebound == 0                      # counters reset after batch log
