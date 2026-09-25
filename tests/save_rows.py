"""Every kind of trim result through the save path, and the rows it leaves, as plain data.

Shared by `test_save_analysis_in.py` (ingest-speed Task 5) and `test_write_batch.py` (Task 6).
Those tasks move the save body -- where every stored number is written -- into a function that
writes into the session it is given, and then into one transaction per batch of files. The
promise is that NOT ONE stored value changes, so the tests compare WHOLE ROWS, every column of
every row of the five tables a trim save writes, never a count.

THE REFERENCE is `fixtures/save_rows_golden.json`: the rows `save_analysis` wrote for the scenario
below at 39636e2, BEFORE the save body moved. Every save path -- `save_analysis`, the session-taking
body with carried (size, mtime, hash), `write_batch`, `save_batch` -- must reproduce it. Regenerate
it only on purpose, and say why in the commit:

    LTA_REGENERATE_SAVE_ROWS=1 python -m pytest tests/test_save_analysis_in.py -k golden

THE SCENARIO covers every trim kind: laser 2 (DLTS, System A), laser 1 (LTS, System B, with its
TrimVolts curves), laser 3 (LTS3: laser 2's format under an LTS3 folder, System C), a two-track file
(with its Track 2 setup block), a no-cut UNTRIMMED file, three ERROR shapes (a file-level reason that
earns a retry marker -- twice: one kept, one later cleared -- a transient one that must not, and a
track-level one that never has), two paths
holding the same bytes (one content-keyed processed-files row between them), a file gone from disk
before its save, a re-process of a stored file (the update path), and an ERROR later read cleanly
(which clears its marker). Real fixture files are copied under `root` with an invented, pinned
mtime; the synthetic results carry invented values.

WHAT IS NORMALISED, and why each is still checked: the clock columns (`timestamp`, `created_date`,
`processed_date`, `processing_time`) become "<volatile>"; paths become relative to `root`; a size,
mtime, content hash or skip-marker hash is replaced by its NAME ("<sha256 of file>") only when it
EQUALS what the file on disk says -- so the golden still pins that the right value was stored,
without pinning the tmp directory, the time zone, or the bytes of a workbook generated per run.
JSON columns are compared as data; the JSON text 'null' stays distinct from SQL NULL on purpose
(the save writes each deliberately -- see `_write_trim_passes`). Floats are compared to 1e-6,
absolute and relative: `tests/test_parse_all_models.py`'s tolerance, for the same reason (another
platform's BLAS).
"""
from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
GOLDEN = FIXTURES / "save_rows_golden.json"
REGENERATE = os.environ.get("LTA_REGENERATE_SAVE_ROWS") == "1"

# Invented, pinned: 2026-01-01 12:00:00 UTC -- noon, so its LOCAL date is the same calendar day in
# every time zone from UTC-11 to UTC+11 (unit_id carries a date).
MTIME = 1767268800.0
# Invented date for the synthetic results.
FIXED_DATE = datetime(2026, 1, 2, 9, 0)

TABLES = ("analysis_results", "track_results", "trim_passes", "trim_setup", "processed_files")
VOLATILE = {
    "analysis_results": {"timestamp", "processing_time"},
    "trim_passes": {"created_date"},
    "trim_setup": {"created_date"},
    "processed_files": {"processed_date"},
}
_SQLITE_DT = "%Y-%m-%d %H:%M:%S.%f"


def inject(db, monkeypatch) -> None:
    """`db` into BOTH `_db_manager` globals: the processor's spec lookups reach `get_database()`."""
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    monkeypatch.setattr(mgr, "_db_manager", db)
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)


def _pinned_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    os.utime(dst, (MTIME, MTIME))
    return dst


def build_scenario(root: Path) -> List[Tuple[str, Any]]:
    """[(label, AnalysisResult)] in SAVE order. Parses real fixtures, so a manager must already be
    injected (see `inject`); nothing here writes to any database."""
    from laser_trim_analyzer.core.models import (
        AnalysisResult, AnalysisStatus, FileMetadata, SystemType, TrackData)
    from laser_trim_analyzer.core.processor import Processor, error_reason_of
    from test_no_cut_template import _make_workbook

    inp = root / "in"
    dlts = _pinned_copy(FIXTURES / "trim" / "dlts_8232-1_242.xls", inp / "dlts_8232-1_242.xls")
    lts = _pinned_copy(FIXTURES / "trim" / "lts_8232-1_193.xls", inp / "lts_8232-1_193.xls")
    lts3 = _pinned_copy(FIXTURES / "trim" / "dlts_8232-1_243.xls",
                        inp / "LTS3" / "8232-1" / "dlts_8232-1_243.xls")
    two = _pinned_copy(FIXTURES / "trim" / "dlts_8074_18.xls", inp / "dlts_8074_18.xls")
    nocut = _make_workbook(inp / "9999_12_TA_Test Data_1-2-2026_9-00 AM.xlsx", lin_error="template")
    os.utime(nocut, (MTIME, MTIME))

    proc = Processor(use_ml=False)
    # The composite-risk models load from a CWD-relative data/ml_models (spec 4.4): in the main
    # checkout they exist, in a worktree they do not. Point it at nothing, so the rows never
    # depend on where the suite was started.
    proc.ml_storage_path = root / "no_ml_models"
    r = {k: proc.process_file(p) for k, p in
         (("dlts", dlts), ("lts", lts), ("lts3", lts3), ("two", two), ("nocut", nocut))}
    assert r["dlts"].metadata.system is SystemType.A and r["dlts"].file_type == "trim"
    assert r["lts"].metadata.system is SystemType.B
    assert r["lts3"].metadata.system is SystemType.C, "an LTS3 folder must label it laser 3"
    assert len(r["two"].tracks) == 2 and r["two"].trim_setup.get("_track2")
    assert r["nocut"].overall_status is AnalysisStatus.UNTRIMMED
    assert any(p.get("increment_volts") for t in r["lts"].tracks for p in t.trim_passes), \
        "the LTS fixture must carry TrimVolts curves"

    def invented_file(name: str, content: bytes) -> Path:
        f = inp / name
        f.write_bytes(content)                          # invented; unreadable on purpose
        os.utime(f, (MTIME, MTIME))
        return f

    def error_file(name: str, reason: str):
        meta = proc._create_minimal_metadata(invented_file(name, f"not a workbook: {name}".encode()))
        meta.file_date = FIXED_DATE                     # _create_minimal_metadata stamps now()
        return proc._create_error_result(meta, reason, 0.0)

    def pass_result(path: Path, model: str, serial: str):
        return AnalysisResult(
            metadata=FileMetadata(filename=path.name, file_path=path, file_date=FIXED_DATE,
                                  model=model, serial=serial, system=SystemType.A),
            overall_status=AnalysisStatus.PASS, processing_time=0.01,
            tracks=[TrackData(track_id="TRK1", status=AnalysisStatus.PASS, travel_length=1.0,
                              linearity_spec=0.01, optimal_offset=0.0, linearity_pass=True,
                              linearity_fail_points=0)])

    err = error_file("8000_1_TEST DATA_1-2-2026_9-00 AM.xls", "No valid track data found")
    err_kept = error_file("8003_1_TEST DATA_1-2-2026_9-00 AM.xls",
                          "Excel file format cannot be determined, you must specify an engine manually.")
    transient = error_file("8001_1_TEST DATA_1-2-2026_9-00 AM.xls",
                           "[Errno 13] Permission denied: '8001_1_TEST DATA_1-2-2026_9-00 AM.xls'")
    track_err_path = invented_file("8856_1_TEST DATA_1-2-2026_9-00 AM.xls", b"x")
    bad_track = TrackData(track_id="TRK1", status=AnalysisStatus.ERROR, travel_length=1.0,
                          linearity_spec=0.01, anomaly_reason="Insufficient data points")
    track_err = AnalysisResult(
        metadata=FileMetadata(filename=track_err_path.name, file_path=track_err_path,
                              file_date=FIXED_DATE, model="8856", serial="1", system=SystemType.A),
        overall_status=AnalysisStatus.ERROR, processing_time=0.01, tracks=[bad_track],
        error_reason=error_reason_of([bad_track], AnalysisStatus.ERROR))
    # processed_files is keyed by CONTENT: a second path holding the same bytes relinks the one
    # row to its own analysis (today's behaviour -- pinned here, not endorsed).
    twin_a = pass_result(invented_file("8010_1_TEST DATA_1-2-2026_9-00 AM.xls", b"same bytes"),
                         "8010", "1")
    twin_b = pass_result(invented_file("8011_1_TEST DATA_1-2-2026_9-00 AM.xls", b"same bytes"),
                         "8011", "1")
    gone = pass_result(inp / "moved_away" / "8002_3_TEST DATA_1-2-2026_9-00 AM.xls",  # never created
                       "8002", "3")
    # The same identity as `err` (filename, file_date, model, serial), now read cleanly.
    clean = AnalysisResult(metadata=err.metadata.model_copy(), overall_status=AnalysisStatus.PASS,
                           processing_time=0.01, tracks=gone.tracks)
    return [
        ("laser 2 (DLTS)", r["dlts"]),
        ("laser 1 (LTS), TrimVolts", r["lts"]),
        ("laser 3 (LTS3)", r["lts3"]),
        ("two-track", r["two"]),
        ("no cut (UNTRIMMED)", r["nocut"]),
        ("ERROR, file-level reason", err),
        ("ERROR, file-level reason, never read cleanly", err_kept),
        ("ERROR, transient reason", transient),
        ("ERROR, track-level", track_err),
        ("same bytes as the next file", twin_a),
        ("same bytes as the last file", twin_b),
        ("gone from disk before the save", gone),
        ("laser 1 re-processed (update path)", r["lts"]),
        ("the ERROR read cleanly (update path)", clean),
    ]


# ---------------------------------------------------------------------------------------------
# rows as data

def _file_facts(path_text: str) -> Dict[str, Any] | None:
    from laser_trim_analyzer.database.manager import DatabaseManager
    p = Path(path_text)
    if not p.is_file():
        return None
    st = p.stat()
    return {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).strftime(_SQLITE_DT),
            "mtime_date": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d"),
            "skip": DatabaseManager.skip_marker_hash(path_text)}


def _relative(path_text: str, root: Path) -> str:
    try:
        return "<root>/" + Path(path_text).relative_to(root).as_posix()
    except ValueError:
        return path_text


def dump_rows(db_path: Path, root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Every row of the five tables, normalised as the module docstring says."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = {}
        for table in TABLES:
            cur = con.execute(f"SELECT * FROM {table} ORDER BY id")
            cols = [d[0] for d in cur.description]
            rows = []
            for values in cur.fetchall():
                row = dict(zip(cols, values))
                facts = _file_facts(row["file_path"]) if row.get("file_path") else None
                norm = {}
                for col, val in row.items():
                    if col in VOLATILE.get(table, ()):
                        val = None if val is None else "<volatile>"
                    elif isinstance(val, str) and val[:1] in "[{":
                        try:
                            val = json.loads(val)
                        except ValueError:
                            pass                  # text that merely starts with a bracket
                    norm[col] = val
                if row.get("file_path"):
                    norm["file_path"] = _relative(row["file_path"], root)
                if facts and table == "processed_files":
                    if norm["file_size"] == facts["size"]:
                        norm["file_size"] = "<size of file>"
                    if norm["file_modified_date"] == facts["mtime"]:
                        norm["file_modified_date"] = "<mtime of file>"
                    if norm["file_hash"] == facts["sha256"]:
                        norm["file_hash"] = "<sha256 of file>"
                    elif norm["file_hash"] == facts["skip"]:
                        norm["file_hash"] = "<skip marker of this path>"
                    if isinstance(norm["error_message"], str):
                        norm["error_message"] = norm["error_message"].replace(
                            facts["sha256"], "<sha256 of file>")
                if facts and table == "analysis_results" and norm["file_date"] == facts["mtime"]:
                    norm["file_date"] = "<mtime of file>"
                    if isinstance(norm["unit_id"], str):
                        norm["unit_id"] = norm["unit_id"].replace(facts["mtime_date"], "<mtime date>")
                rows.append(norm)
            out[table] = rows
        return out
    finally:
        con.close()


def snapshot(db_path: Path, root: Path, ids) -> Dict[str, Any]:
    return {"ids": [[label, rid] for label, rid in ids], "tables": dump_rows(db_path, root)}


def differences(got, want, where: str = "", out=None, limit: int = 40) -> List[str]:
    """Every place `got` differs from `want`, as readable lines (at most `limit`)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    num = (int, float)
    if (isinstance(got, num) and isinstance(want, num)
            and not isinstance(got, bool) and not isinstance(want, bool)):
        both_nan = (isinstance(got, float) and isinstance(want, float)
                    and math.isnan(got) and math.isnan(want))     # a stored NaN is a value too
        if not both_nan and not math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6):
            out.append(f"{where}: {want!r} -> {got!r}")
    elif isinstance(got, dict) and isinstance(want, dict):
        for k in sorted(set(got) | set(want)):
            if k not in got or k not in want:
                out.append(f"{where}.{k}: {'missing now' if k not in got else 'new column'}")
            else:
                differences(got[k], want[k], f"{where}.{k}", out, limit)
    elif isinstance(got, list) and isinstance(want, list):
        if len(got) != len(want):
            out.append(f"{where}: {len(want)} items -> {len(got)}")
        for i, (g, w) in enumerate(zip(got, want)):
            differences(g, w, f"{where}[{i}]", out, limit)
    elif type(got) is not type(want) or got != want:
        out.append(f"{where}: {want!r} -> {got!r}")
    return out


def load_golden() -> Dict[str, Any]:
    return json.loads(GOLDEN.read_text())


def write_golden(snap: Dict[str, Any]) -> None:
    # One value per line (indent=1): a model name and a number never share a line, which is what
    # scripts/check_no_customer_values.py reads as "a model beside its price".
    GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")


def assert_matches_golden(snap: Dict[str, Any]) -> None:
    diffs = differences(snap, load_golden())
    assert not diffs, ("the save path stored different rows than today's save_analysis did "
                       "(tests/fixtures/save_rows_golden.json):\n  " + "\n  ".join(diffs))


# ---------------------------------------------------------------------------------------------
# a trap for any file I/O

@contextlib.contextmanager
def no_file_io(monkeypatch):
    """Every way the save path could touch a file -- os.stat/lstat, open, and the manager's own
    stat_once/calculate_file_hash (which a warm hash cache would answer WITHOUT any I/O, so they
    are trapped by name) -- records its call and raises. Yields the record: a caller that swallows
    the exception is still caught by it."""
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.utils import hashing
    touched: List[str] = []

    def trap(name):
        def refused(*args, **kwargs):
            touched.append(f"{name}{args[:1]!r}")
            raise AssertionError(f"file I/O inside the save: {name}{args[:1]!r}")
        return refused

    with monkeypatch.context() as m:
        for owner, name in ((os, "stat"), (os, "lstat"), (os, "open"), (builtins, "open"),
                            (io, "open"), (mgr, "stat_once"), (mgr, "calculate_file_hash"),
                            (hashing, "stat_once"), (hashing, "calculate_file_hash")):
            m.setattr(owner, name, trap(f"{owner.__name__}.{name}"))
        yield touched
