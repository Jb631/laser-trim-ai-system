"""System C / LTS3 support (2026-07-06).

The third trim system writes files FORMAT-identical to an existing system
(A or B — the parser reads the format from each file's sheets). Identity is
the 'LTS3' parent folder in the storage path. Sheets decide FORMAT (drives
extraction); path decides SYSTEM (tags the record C).
"""
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---- path-based identity -----------------------------------------------------

def test_lts3_directory_marks_system_c():
    from laser_trim_analyzer.core.parser import ExcelParser
    from laser_trim_analyzer.core.models import SystemType

    resolve = ExcelParser._resolve_system_identity
    # POSIX and Windows-style paths, case variants, suffixed folder names.
    assert resolve(Path("/data/LTS3/8475-1_A123.xls"), SystemType.A) == SystemType.C
    assert resolve(Path("/data/lts3/sub/8475-1_A123.xls"), SystemType.B) == SystemType.C
    assert resolve(Path(r"C:\lasers\LTS3\2026\file.xls"), SystemType.A) == SystemType.C
    assert resolve(Path("/mnt/LTS3 Data/file.xls"), SystemType.B) == SystemType.C


def test_non_lts3_paths_keep_format_identity():
    from laser_trim_analyzer.core.parser import ExcelParser
    from laser_trim_analyzer.core.models import SystemType

    resolve = ExcelParser._resolve_system_identity
    assert resolve(Path("/data/laser2/file.xls"), SystemType.B) == SystemType.B
    assert resolve(Path("/data/system_a/file.xls"), SystemType.A) == SystemType.A
    # A FILENAME starting with LTS3 must not match — directories only.
    assert resolve(Path("/data/laser2/LTS3_report.xls"), SystemType.B) == SystemType.B


# ---- DB round-trip -------------------------------------------------------------

def _add_row(db, model, path, system_value, when=datetime(2026, 7, 1)):
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType as DBSystemType,
        StatusType)
    with db.session() as s:
        ar = DBAR(filename=Path(path).name, file_path=path,
                  file_hash=f"h{abs(hash(path)) % 10**12:012d}" + "0" * 52,
                  model=model, serial="sn1", system=DBSystemType(system_value),
                  file_date=when, timestamp=when, overall_status=StatusType.PASS,
                  has_multi_tracks=False, processing_time=0.1)
        s.add(ar)
        s.flush()
        # get_system_comparison inner-joins tracks — every analysis needs one.
        s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                   sigma_pass=True, linearity_pass=True))
        s.commit()


def test_system_c_saves_and_loads(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SystemType as DBSystemType

    db = DatabaseManager(tmp_path / "c.db")
    _add_row(db, "9001-1", "/data/LTS3/9001-1_X1.xls", "C")
    with db.session() as s:
        row = s.query(DBAR).filter(DBAR.model == "9001-1").one()
        assert row.system == DBSystemType.C


def test_system_comparison_includes_c(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "cmp.db")
    _add_row(db, "M-A", "/data/a/f1.xls", "A")
    _add_row(db, "M-B", "/data/b/f2.xls", "B")
    _add_row(db, "M-C", "/data/LTS3/f3.xls", "C")
    cmp_result = db.get_system_comparison(days_back=3650)
    assert "system_c" in cmp_result
    assert cmp_result["system_c"] is not None
    assert cmp_result["system_c"]["total_files"] == 1


# ---- backfill migration --------------------------------------------------------

def test_migration_retags_lts3_rows(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SystemType as DBSystemType

    dbfile = tmp_path / "mig.db"
    db = DatabaseManager(dbfile)
    # Rows processed BEFORE path-based detection: stored under their format id.
    _add_row(db, "OLD-1", "/prod/LTS3/2026/OLD-1_S1.xls", "B")
    _add_row(db, "OLD-2", r"D:\prod\LTS3 Data\OLD-2_S2.xls", "A")
    _add_row(db, "OLD-3", "/prod/laser2/OLD-3_S3.xls", "B")          # untouched
    _add_row(db, "OLD-4", "/prod/laser2/LTS3_named_file.xls", "B")   # filename only — untouched
    db.close() if hasattr(db, "close") else None

    # Re-init: migrations run on construction.
    db2 = DatabaseManager(dbfile)
    with db2.session() as s:
        got = {r.model: r.system for r in s.query(DBAR).all()}
    assert got["OLD-1"] == DBSystemType.C
    assert got["OLD-2"] == DBSystemType.C
    assert got["OLD-3"] == DBSystemType.B
    assert got["OLD-4"] == DBSystemType.B
