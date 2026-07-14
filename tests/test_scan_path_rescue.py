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


def test_guard_fires_when_known_filenames_fail_recognition(tmp_path):
    """Many KNOWN filenames failing recognition = wrong path/database.
    The batch must abort with a clear message, not reprocess the world."""
    proc = _make_processor()
    # DB knows these very filenames — but with unmatchable stats and no
    # hashes, recognition fails for all of them (the wrong-DB signature).
    files = []
    basemap = {}
    for i in range(120):                                # >= turbo threshold
        f = tmp_path / f"k{i}.xls"
        f.write_bytes(b"x")
        basemap[f.name] = [(f"Z:\\\\old\\\\k{i}.xls", 999999, 1.0)]  # never matches
        files.append(f)
    proc._processed_filenames = ({e[0][0] for e in basemap.values()}
                                 | {f"pad{i}" for i in range(1000)})
    proc._processed_hashes = set()
    proc._processed_stat = {}
    proc._processed_basename = basemap
    proc._load_processed_hashes = lambda: None
    gen = proc.process_batch(files, incremental=True)
    with pytest.raises(RuntimeError, match="different path|wrong database"):
        list(gen)


def test_all_new_folder_passes_the_guard(tmp_path):
    """Work regression 2026-07-10: a folder of 349 genuinely NEW files was
    blocked by guard v1 ('matched 0 of N'). Unknown filenames must process."""
    proc = _make_processor()
    proc._processed_filenames = {f"Z:\\\\known\\\\k{i}.xls" for i in range(1500)}
    proc._processed_hashes = set()
    proc._processed_stat = {}
    proc._processed_basename = {}                       # none of the new names known
    files = []
    for i in range(120):
        f = tmp_path / f"new_{i}.xls"
        f.write_bytes(b"x")
        files.append(f)
    proc._load_processed_hashes = lambda: None
    # Must NOT raise; the junk files just flow through as non-trim skips.
    list(proc.process_batch(files, incremental=True))


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


def test_nan_measurements_become_none_not_validation_errors():
    """Work incident 2026-07-10: pydantic >=2.12 rejects NaN on ge=0 fields —
    2,106 files errored in one day. NaN means 'not measured' -> None."""
    from laser_trim_analyzer.core.models import TrackData, AnalysisStatus
    t = TrackData(track_id="T1", travel_length=1.0, linearity_spec=0.01,
                  status=AnalysisStatus.PASS,
                  linearity_error=float("nan"), raw_linearity_error=float("nan"),
                  optimized_linearity_error=float("nan"), max_deviation=float("nan"))
    assert t.linearity_error is None
    assert t.max_deviation is None
    # Real values pass through untouched.
    ok = TrackData(track_id="T1", travel_length=1.0, linearity_spec=0.01,
                   status=AnalysisStatus.PASS, linearity_error=0.005)
    assert ok.linearity_error == 0.005


def test_permanent_failures_are_not_retried_forever(tmp_path, monkeypatch):
    """Full-log taxonomy 2026-07-10: ~4,000 files fail for reasons that can
    never succeed on retry (no serial, pre-2003 Excel, duplicates). They must
    be recorded as skipped after the first attempt; transient errors must NOT."""
    proc = _make_processor()
    marked = []
    monkeypatch.setattr(proc, "_mark_file_skipped", lambda p: marked.append(p.name))
    assert proc._is_permanent_failure(ValueError("Serial cannot be empty"))
    assert proc._is_permanent_failure(ValueError(
        "Excel file format cannot be determined, you must specify an engine manually."))
    assert proc._is_permanent_failure(Exception(
        "(sqlite3.IntegrityError) UNIQUE constraint failed: final_test_results.filename"))
    assert not proc._is_permanent_failure(OSError("network path unavailable"))
    assert not proc._is_permanent_failure(MemoryError("out of memory"))


def test_disk_stats_map_eliminates_per_file_io(sample, monkeypatch):
    """V4-speed regression guard (2026-07-10): with discovery-captured stats,
    the incremental check must touch NEITHER stat() NOR the file content."""
    proc = _make_processor()
    st = sample.stat()
    stored = "Z:\\\\Production\\\\" + sample.name
    proc._processed_filenames = {stored}
    proc._processed_hashes = set()
    proc._processed_stat = {}
    proc._processed_basename = {sample.name: [(stored, st.st_size, st.st_mtime)]}
    proc._disk_stats = {str(sample): (st.st_size, st.st_mtime)}
    _no_read(monkeypatch)

    def no_stat(self_):
        raise AssertionError("stat() called despite discovery-captured stats")
    monkeypatch.setattr(Path, "stat", no_stat)
    assert proc._is_processed(sample) is True
    assert proc._scan_rebound == 1


def test_baseline_requalification_floors_training(tmp_path):
    """Design-change policy + LOT MODE (2026-07-13): after requalifying at
    date D, training uses only LOTS from D forward. Lots are weekly clusters
    of 5 units; old design ~10, new design ~20."""
    from datetime import datetime, timedelta
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType,
        ModelMetricState)
    from laser_trim_analyzer.ml.drift_training import train_drift_detector

    db = DatabaseManager(tmp_path / "rq.db")
    t0 = datetime(2025, 1, 6)
    with db.session() as s:
        i = 0
        for lot in range(24):                       # 12 old-design + 12 new
            lot_day = t0 + timedelta(weeks=lot)
            level = 10.0 if lot < 12 else 20.0
            for u in range(5):
                when = lot_day + timedelta(hours=u)
                ar = DBAR(filename=f"RQ-{i}.xls", file_path=f"/f/{i}",
                          file_hash=f"rq{i}".ljust(64, "0"), model="RQ",
                          serial=f"s{i}", system=SystemType.A, file_date=when,
                          timestamp=when, overall_status=StatusType.PASS,
                          has_multi_tracks=False, processing_time=0.1)
                s.add(ar); s.flush()
                s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                           untrimmed_sigma_gradient=level + (u % 3) * 0.01,
                           travel_length=1.0, sigma_pass=True, linearity_pass=True))
                i += 1
        s.commit()

    # No requalification: baseline lots span both designs -> mean far from 20.
    train_drift_detector(db, model="RQ")
    with db.session() as s:
        ms = s.query(ModelMetricState).filter_by(
            model="RQ", metric="untrimmed_sigma_gradient").first()
        assert ms is not None and ms.is_trained
        assert ms.baseline_mean < 19.0

    # Requalified at the design change: baseline = new-design LOTS only (~20).
    db.set_baseline_requalification("RQ", (t0 + timedelta(weeks=12)).date().isoformat(),
                                    "design change")
    train_drift_detector(db, model="RQ")
    with db.session() as s:
        ms = s.query(ModelMetricState).filter_by(
            model="RQ", metric="untrimmed_sigma_gradient").first()
        assert ms.is_trained, "12 post-change lots must be enough to train"
        assert ms.baseline_mean > 19.5, f"baseline still polluted: {ms.baseline_mean}"
        # baseline_count is LOTS now (12 closed - 3 replay = 9 baseline lots)
        assert ms.baseline_count == 9


def test_lot_mode_one_alarm_per_lot_not_per_unit(tmp_path):
    """The core reason for lot mode: a single shifted 60-unit lot must push
    the detector ONCE (one lot median), not 60 times. Stable lots stay
    Stable; a sustained multi-lot step accumulates."""
    from datetime import datetime, timedelta
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType,
        ModelMetricState)
    from laser_trim_analyzer.ml.drift_training import train_drift_detector, advance_drift_state

    db = DatabaseManager(tmp_path / "lots.db")
    t0 = datetime(2025, 1, 6)

    def add_lot(s, lot_idx, level, n, start_i):
        lot_day = t0 + timedelta(weeks=lot_idx)
        for u in range(n):
            when = lot_day + timedelta(hours=u % 48)
            i = start_i + u
            ar = DBAR(filename=f"L-{i}.xls", file_path=f"/l/{i}",
                      file_hash=f"lt{i}".ljust(64, "0"), model="LM",
                      serial=f"s{i}", system=SystemType.A, file_date=when,
                      timestamp=when, overall_status=StatusType.PASS,
                      has_multi_tracks=False, processing_time=0.1)
            s.add(ar); s.flush()
            s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                       untrimmed_sigma_gradient=level + (u % 5) * 0.02,
                       travel_length=1.0, sigma_pass=True, linearity_pass=True))
        return start_i + n

    with db.session() as s:
        i = 0
        for lot in range(12):                 # stable history, medians ~10.04
            i = add_lot(s, lot, 10.0 + (lot % 3) * 0.02, 8, i)
        s.commit()
    train_drift_detector(db, model="LM")
    with db.session() as s:
        ms = s.query(ModelMetricState).filter_by(
            model="LM", metric="untrimmed_sigma_gradient").first()
        assert ms.is_trained
        base_cpos = ms.cusum_pos or 0.0

    # ONE big shifted lot (60 units at +2 lot-σ-ish): exactly one detector
    # step — cusum_pos rises by AT MOST (median - mean - k)/σ once, NOT 60x.
    with db.session() as s:
        i = add_lot(s, 13, 10.5, 60, 1000)    # clearly above the ~10.02 band
        s.commit()
    advanced = advance_drift_state(db, model="LM")
    assert advanced == 1
    with db.session() as s:
        ms = s.query(ModelMetricState).filter_by(
            model="LM", metric="untrimmed_sigma_gradient").first()
        one_lot_cpos = (ms.cusum_pos or 0.0) - base_cpos
        # A unit-mode detector would have accumulated ~60 pushes; lot mode
        # accumulates one bounded push (< suspect gate = 8σ).
        assert 0.0 < one_lot_cpos <= 8.0, f"one lot pushed {one_lot_cpos}σ"
        assert ms.last_row_id is None          # lot-mode watermark stays date-based
