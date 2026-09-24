"""Laser 3 (LTS3) writes laser 2's sheets, not laser 1's — and the parser must route on FORMAT.

James, 2026-09-20: "laser 2 & 3 use the same style sheet not 1 & 3." He is right, and several
comments in this repository said the opposite (they have been corrected). The evidence in the work
database: all 547 LTS3 tracks carry `track_id = 'TRK1'`, which only `_extract_system_a_tracks`
produces — the System B reader would have written "default" for those filenames, which carry no
`_TA_` segment. LTS3 filenames also follow the DLTS convention (`2475-8_4_TEST DATA_8-4-2026…`),
not the LTS one (`8232-1_104_TA_Test Data_…`).

What makes this safe is that `parse_file` passes the FORMAT to `_extract_tracks` and keeps the
LTS3 folder only as an identity label. These tests pin that, because the bug it prevents is silent:
an A-format file read by the B reader does not raise, it produces wrong numbers.
"""
from pathlib import Path

import pytest

from laser_trim_analyzer.core.models import SystemType
from laser_trim_analyzer.core.parser import ExcelParser

A_SHEETS = ["SEC1 TRK1 0", "SEC1 TRK1 TRM", "SEC1 TRK1 1"]
B_SHEETS = ["Model Parameters", "Trim Parameters", "Lin Error", "test", "VOLTAGES", "ERRORS"]


def test_the_lts3_folder_changes_the_label_not_the_reader():
    p = ExcelParser()
    a_like = Path(r"\\srv\TEST_DATA\LTS3\2475-8\2475-8_4_TEST DATA_8-4-2026_12-19 PM.xls")
    assert p._detect_system_from_sheets(A_SHEETS) is SystemType.A          # the FORMAT
    assert p._resolve_system_identity(a_like, SystemType.A) is SystemType.C  # the LABEL
    # ...and a B-format file that happened to sit under LTS3 would still be read as B
    assert p._detect_system_from_sheets(B_SHEETS) is SystemType.B
    assert p._resolve_system_identity(a_like, SystemType.B) is SystemType.C


def test_extraction_dispatches_on_format_never_on_the_lts3_label(monkeypatch):
    """The whole point: SystemType.C must never reach `_extract_tracks`, because the dispatch there
    sends everything that is not A to the System B reader."""
    p = ExcelParser()
    seen = []
    monkeypatch.setattr(ExcelParser, "_extract_system_a_tracks", lambda self, xl, fp: seen.append("A") or [])
    # `*_`: the B reader also takes laser 1's TrimVolts frame (2026-09-24).
    monkeypatch.setattr(ExcelParser, "_extract_system_b_tracks", lambda self, xl, fp, *_: seen.append("B") or [])
    p._extract_tracks(None, Path("x"), SystemType.A)
    p._extract_tracks(None, Path("x"), SystemType.B)
    assert seen == ["A", "B"]
    # If SystemType.C were ever passed here it would land in the B reader — which is why it is not.
    p._extract_tracks(None, Path("x"), SystemType.C)
    assert seen == ["A", "B", "B"], "C falls through to the B reader; pass the FORMAT, never the label"


def test_a_real_lts3_file_would_be_read_by_the_laser_2_reader(tmp_path, monkeypatch):
    """End to end on a synthetic A-format workbook living under an LTS3 folder: the stored system is
    C (the label) while the reader that ran is A (the format)."""
    from openpyxl import Workbook
    folder = tmp_path / "LTS3" / "2475-8"
    folder.mkdir(parents=True)
    path = folder / "2475-8_4_TEST DATA_8-4-2026_12-19 PM.xlsx"
    wb = Workbook()
    wb.active.title = "SEC1 TRK1 0"
    wb.create_sheet("SEC1 TRK1 TRM")
    wb.save(path)
    used = []
    real_a = ExcelParser._extract_system_a_tracks
    monkeypatch.setattr(ExcelParser, "_extract_system_a_tracks",
                        lambda self, xl, fp: used.append("A") or real_a(self, xl, fp))
    monkeypatch.setattr(ExcelParser, "_extract_system_b_tracks",
                        lambda self, xl, fp, *_: used.append("B") or [])
    try:
        result = ExcelParser().parse_file(path)
    except Exception as exc:                       # an empty A-format workbook may yield no tracks
        assert used == ["A"], f"the B reader ran on an A-format LTS3 file ({exc})"
        return
    assert used == ["A"], "an LTS3 file must be read by the laser-2 (System A) reader"
    assert result["metadata"].system is SystemType.C, "…while still being labelled laser 3"


def test_the_no_cut_template_rule_is_a_laser_1_rule():
    """The blank-`Lin Error` rule lives in the System B reader, so it can only ever fire on laser 1
    (and on any B-format file). That matches where the problem is: of the 1,182 tracks stored as
    flawless with an all-zero error column, 1,180 are laser 1 and 2 are laser 2 — none is laser 3.
    Laser 3 has no `Lin Error` sheet at all."""
    import inspect
    src = inspect.getsource(ExcelParser._extract_system_b_tracks)
    assert "_lin_error_is_template" in src
    assert "_lin_error_is_template" not in inspect.getsource(ExcelParser._extract_system_a_tracks)
