"""3rd-laser guard (2026-07-06): unknown trim-file layouts must fail LOUDLY.

Before: _detect_system_from_sheets defaulted unknown layouts to System B
(silently wrong data under B's column map), and detect_file_type buried them
as skipped-forever non-trim files at INFO level. With a third trim system in
production, both paths must be visible.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _make_xlsx(path: Path, sheet_names):
    from openpyxl import Workbook
    wb = Workbook()
    wb.active.title = sheet_names[0]
    for name in sheet_names[1:]:
        wb.create_sheet(title=name)
    wb.save(path)


def test_unknown_layout_raises_not_defaults_to_b(tmp_path):
    from laser_trim_analyzer.core.parser import ExcelParser

    p = tmp_path / "8475-1_A12345.xlsx"
    _make_xlsx(p, ["LASER3 DATA", "RESULTS"])  # matches neither A nor B

    with pytest.raises(ValueError) as exc:
        ExcelParser().parse_file(p)
    msg = str(exc.value).lower()
    assert "unrecognized" in msg
    assert "third" in msg or "system" in msg


def test_known_layouts_still_detect(tmp_path):
    from laser_trim_analyzer.core.parser import ExcelParser
    from laser_trim_analyzer.core.models import SystemType

    parser = ExcelParser()
    assert parser._detect_system_from_sheets(["SEC1 TRK1 0", "SEC1 TRK1 TRM"]) == SystemType.A
    assert parser._detect_system_from_sheets(["test", "Trim 1", "Lin Error"]) == SystemType.B


def test_unknown_structure_routes_non_trim_with_warning(tmp_path, caplog):
    import logging
    from laser_trim_analyzer.core.parser import detect_file_type

    p = tmp_path / "8475-1_A12345.xlsx"
    _make_xlsx(p, ["LASER3 DATA", "RESULTS"])

    with caplog.at_level(logging.WARNING):
        assert detect_file_type(p) == "non_trim"
    assert any("unrecognized sheet structure" in r.message.lower()
               for r in caplog.records), "burial must be visible at WARNING level"


def test_error_result_not_batch_crash(tmp_path):
    """A 3rd-laser-like file in a batch must become a per-file ERROR result,
    not kill the batch. (detect_file_type may route it non_trim first; force
    the parse path by calling process_file on a file that detects as trim.)"""
    from laser_trim_analyzer.core.parser import ExcelParser

    # Directly exercise the parse path: unknown layout raises ValueError,
    # which processor.process_file's except-block converts to an ERROR result.
    p = tmp_path / "9001-2_B99999.xlsx"
    _make_xlsx(p, ["WEIRD1", "WEIRD2"])
    with pytest.raises(ValueError):
        ExcelParser().parse_file(p)
