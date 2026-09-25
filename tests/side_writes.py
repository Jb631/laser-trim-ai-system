"""The ingest workers' OTHER writes -- final tests, smoothness results, skip markers -- as one
scenario, and the rows it leaves (ingest-speed Task 7).

Task 7 moves each of these saves into a body that writes into the session it is GIVEN, so the
batch writer can run it inside one transaction; the public method keeps its session and its
commit. The promise is Task 5's: not one stored value changes. So, as there, the rows are compared
WHOLE -- every column of every row of the tables below -- against `fixtures/save_rows_side_golden.json`,
the rows TODAY's public methods wrote for this scenario at e9a5df8, before any body moved; and
every path run in one process is compared EXACTLY with the public methods run beside it. The
machinery (normalisation, tolerances, the I/O trap) is `save_rows.py`'s. Regenerate only on purpose:

    LTA_REGENERATE_SAVE_ROWS=1 python -m pytest tests/test_side_write_bodies.py -k golden

THE SCENARIO. The final-test payloads are exactly what the PROCESSOR hands `save_final_test` for
each fixture -- captured by standing in for that one method while `process_file` runs -- so the
grading that produced them is today's. It covers every final-test format the repository holds
(Format 1 x4, Format 3's three tracks, Format 4 without a serial, a synthetic Format 2 `Rout_`
workbook) and every branch of the save: a fuzzy and an exact trim link, a header-only row with no
stat, the same content again (a legacy row's stat stamped, the verdict re-graded), the same content
at another path (that path marked, the verdict re-graded), a file re-exported in place (the UNIQUE
identity fires; the row's identity refreshed) and new content under an existing identity at another
path (that path marked). Smoothness (invented payloads -- the repository holds no smoothness file):
a trim link, two tracks, an upsert, and the same identity with new content (nothing stored). Skip
markers: not test data, a read failure, the same path again (stat and reason refreshed), no reason
(the old one kept), and a stored trim's own row (stat and reason refreshed, success kept).
All values not read from a fixture are invented.
"""
from __future__ import annotations

import copy
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import save_rows
from save_rows import FIXTURES, MTIME, _pinned_copy

SIDE_TABLES = ("final_test_results", "final_test_tracks", "smoothness_results",
               "smoothness_tracks", "processed_files", "analysis_results")
SIDE_GOLDEN = FIXTURES / "save_rows_side_golden.json"
FT_FIXTURES = ("7458-sn7_4-2-2026_3-19 PM.xls", "7539-2-sn23_1-22-2026_2-44 PM.xls",
               "8232-1-sn176_3-28-2026_7-59 AM.xls", "8232-1-sn180_3-28-2026_9-37 AM.xls",
               "8434ct-1118D.xls", "8639-30-sn88b_11-2-2015_9-19 AM.xls")
ROUT = "Rout_9990_sn5_vo.xlsx"          # invented model and serial


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _capture_ft_payloads(db, paths) -> Dict[str, Dict[str, Any]]:
    """{filename: the kwargs the processor passes save_final_test}, from the real processor."""
    from laser_trim_analyzer.core.processor import Processor
    proc = Processor(use_ml=False)
    proc.ml_storage_path = Path(paths[0]).parent / "no_ml_models"
    got: Dict[str, Dict[str, Any]] = {}

    def stand_in(**kwargs):
        got[Path(kwargs["metadata"]["file_path"]).name] = copy.deepcopy(kwargs)
        return -7                        # an id nothing stores

    db.save_final_test = stand_in        # what the processor's get_database() hands back
    try:
        for p in paths:
            proc.process_file(p)
    finally:
        del db.save_final_test
    missing = [Path(p).name for p in paths if Path(p).name not in got]
    assert not missing, f"the processor saved no final test for {missing}"
    return got


def build_side_scenario(root: Path, db) -> List[Tuple[str, str, Dict[str, Any]]]:
    """[(label, kind, kwargs)] in SAVE order. kind: 'trim' ({'analysis': result}), 'final_test',
    'smoothness' or 'marker', whose kwargs are exactly the public method's. `db` must already be
    injected (see save_rows.inject): the processor grades against it, and nothing is written."""
    from laser_trim_analyzer.core.models import (
        AnalysisResult, AnalysisStatus, FileMetadata, SystemType, TrackData)
    from test_ft_polyfit_degenerate import clean_series, write_format2

    inp = root / "side"
    # Under a "Test Station" folder, as on the share: the Format 4 file carries no final-test
    # filename pattern, and the detector routes it by that folder.
    ft_dir = inp / "Test Station" / "ft"
    ft_paths = [_pinned_copy(FIXTURES / "final_test" / n, ft_dir / n) for n in FT_FIXTURES]
    rout = write_format2(ft_dir / ROUT, *clean_series())
    os.utime(rout, (MTIME, MTIME))
    ft_paths.append(rout)
    p = _capture_ft_payloads(db, ft_paths)

    def trim(name: str, model: str, serial: str, day: datetime):
        f = inp / "trim" / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(f"invented trim workbook {name}".encode())
        os.utime(f, (MTIME, MTIME))
        return AnalysisResult(
            metadata=FileMetadata(filename=name, file_path=f, file_date=day, model=model,
                                  serial=serial, system=SystemType.A),
            overall_status=AnalysisStatus.PASS, processing_time=0.01,
            tracks=[TrackData(track_id="TRK1", status=AnalysisStatus.PASS, travel_length=1.0,
                              linearity_spec=0.01, optimal_offset=0.0, linearity_pass=True,
                              linearity_fail_points=0)])

    def ft(name: str, **changes):
        kw = copy.deepcopy(p[name])
        for key, value in changes.items():
            if key == "file_path":
                kw["metadata"]["file_path"] = value
            else:
                kw[key] = value
        return kw

    elsewhere = ft_dir / "Voltage Output"
    f1, f2, f3, f4 = FT_FIXTURES[0], FT_FIXTURES[1], FT_FIXTURES[2], FT_FIXTURES[3]

    def smooth(name: str, model: str, serial: str, day: datetime, content: str, *values):
        tracks = [{"track_id": f"TRK{i + 1}", "smoothness_pass": v < 0.5, "max_smoothness": v,
                   "avg_smoothness": round(v / 3, 6), "smoothness_spec": 0.5,
                   "positions": [0.0, 0.5, 1.0], "smoothness_values": [v / 4, v, v / 2]}
                  for i, v in enumerate(values)]
        return {"metadata": {"filename": name, "file_path": str(inp / "os" / name),
                             "model": model, "serial": serial, "file_date": day,
                             "test_date": day + timedelta(hours=9), "element_label": "A",
                             "smoothness_spec": 0.5},
                "tracks": tracks, "file_hash": _sha(content), "file_size": 2345,
                "file_modified_date": datetime(2026, 2, 3, 4, 5, 6)}

    def marker(name: str, content, size: int, when: datetime, reason=None, failed_read=False,
               path=None):
        return {"filename": name, "file_path": str(path or (inp / "misc" / name)),
                "file_hash": _sha(content) if content else "", "file_size": size,
                "file_modified_date": when, "error_message": reason, "failed_read": failed_read}

    trim_a = trim("7458_007_TEST DATA_3-30-2026_8-00 AM.xls", "7458", "007", datetime(2026, 3, 30, 8))
    trim_b = trim("7539-2_23_TEST DATA_1-20-2026_8-00 AM.xls", "7539-2", "23", datetime(2026, 1, 20, 8))
    smooth_b = smooth("9991-sn4_2-3-2026.xlsx", "9991", "4", datetime(2026, 2, 3), "os b", 0.21, 0.62)
    return [
        ("trim 7458/007: a fuzzy-serial link target", "trim", {"analysis": trim_a}),
        ("trim 7539-2/23: an exact link target", "trim", {"analysis": trim_b}),
        ("FT format 1, fuzzy trim link", "final_test", ft(f1)),
        ("FT format 1, exact trim link", "final_test", ft(f2)),
        ("FT format 1, no trim", "final_test", ft(f3)),
        ("FT format 3, three tracks", "final_test", ft(FT_FIXTURES[5])),
        ("FT format 4, no serial", "final_test", ft(FT_FIXTURES[4])),
        ("FT format 2 (Rout_)", "final_test", ft(ROUT)),
        ("FT header only, no stat recorded", "final_test",
         ft(f4, tracks=[], file_size=None, file_modified_date=None)),
        ("FT same content again: the legacy row's stat stamped, verdict re-graded", "final_test",
         ft(f4)),
        ("FT same content at another path: that path marked, verdict re-graded", "final_test",
         ft(f2, file_path=str(elsewhere / f2))),
        ("FT re-exported in place: the identity refreshed", "final_test",
         ft(f3, file_hash=_sha("re-exported in place"))),
        ("FT new content under a stored identity, another path: that path marked", "final_test",
         ft(f1, file_hash=_sha("new content elsewhere"), file_path=str(elsewhere / f1))),
        ("smoothness, exact trim link", "smoothness",
         smooth("7539-2-sn23_1-25-2026.xlsx", "7539-2", "23", datetime(2026, 1, 25), "os a", 0.18)),
        ("smoothness, two tracks, no trim", "smoothness", smooth_b),
        ("smoothness, same content again (upsert)", "smoothness",
         {**smooth("9991-sn4_2-3-2026.xlsx", "9991", "4", datetime(2026, 2, 3), "os b", 0.33, 0.44),
          "file_size": 3456}),
        ("smoothness, same identity, new content (nothing stored)", "smoothness",
         smooth("9991-sn4_2-3-2026.xlsx", "9991", "4", datetime(2026, 2, 3), "os b v2", 0.1)),
        ("marker: not test data", "marker",
         marker("summary report.xlsx", "summary", 5120, datetime(2026, 1, 5, 8))),
        ("marker: a read failure", "marker",
         marker("8003_2_TEST DATA_1-2-2026_9-00 AM.xls", "unreadable", 77, datetime(2026, 1, 6, 8),
                reason="Excel file format cannot be determined, you must specify an engine manually.",
                failed_read=True)),
        ("marker: the same path again, stat and reason refreshed", "marker",
         marker("summary report.xlsx", "summary v2", 6144, datetime(2026, 1, 7, 8),
                reason="not test data: a summary workbook")),
        ("marker: the same path, no reason and no hash: the reason kept", "marker",
         marker("summary report.xlsx", None, 7168, datetime(2026, 1, 8, 8))),
        ("marker on a stored trim's own row: stat and reason refreshed, success kept", "marker",
         marker(trim_a.metadata.filename, "trim copy", 999, datetime(2026, 1, 9, 8),
                reason="invented: same content as another path", path=trim_a.metadata.file_path)),
    ]


def run_public(db, steps) -> List[Tuple[str, Any]]:
    """Every step through its PUBLIC method (its own session and commit): what each returned --
    or, for a save the method refuses (Format 4 without a serial), what it raised."""
    method = {"trim": lambda kw: db.save_analysis(kw["analysis"]),
              "final_test": lambda kw: db.save_final_test(**kw),
              "smoothness": lambda kw: db.save_smoothness_result(**kw),
              "marker": lambda kw: db.mark_file_skipped(**kw)}
    out = []
    for label, kind, kw in steps:
        try:
            out.append((label, method[kind](copy.deepcopy(kw))))
        except Exception as e:
            out.append((label, f"raised {type(e).__name__}: {e}"))
    return out


def public_snapshot(steps, root: Path) -> Dict[str, Any]:
    """The scenario through the public methods, into a database of its own, in THIS process: what
    every other path must store EXACTLY."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    path = root / "side_reference.db"
    ref = DatabaseManager(path)
    try:
        ids = run_public(ref, steps)
    finally:
        ref.close()
    return save_rows.snapshot(path, root, ids, SIDE_TABLES)
