"""A blank THEORY cell must not become an error of 0.0.

CLAUDE.md: "Blank cells are ungraded, never 0.0." The final-test parser honours that
in the branch that uses the file's own error column, and broke it in the sibling
branch 20 lines below, which substituted the measured value for a blank theory cell --
making `measured - measured` exactly 0.0, dead centre of every band, on a
zero-tolerance metric.

Found 2026-09-20 while reading the build ledgers. It was described there as dormant;
it is not. In a 1,500-workbook sample of the real share, 611 parsed and one carried a
blank theory cell. At fleet scale the rebuild re-runs this over ~151,000 final-test
records.
"""
import pytest

from laser_trim_analyzer.core.final_test_parser import FinalTestParser


def workbook(tmp_path, theory_blank_at=None, n=20):
    """A minimal Format 1 final-test sheet, with column D (the error column) empty.

    Leaving D empty is what routes the parser into the measured-minus-theory branch --
    the one under test. Column layout is the parser's own: A measured, B index,
    C theory, D error, E electrical angle, G upper, H lower.
    """
    import pandas as pd
    rows = []
    for i in range(n):
        theory = 1.0 + i * 0.1 + (0.002 if i % 4 == 0 else 0.0)
        if i == theory_blank_at:
            theory = None
        rows.append([1.0 + i * 0.1, i + 1, theory, None, i * 0.03, None, 0.05, -0.05])
    path = tmp_path / "8340-1-sn999_1-22-2026_2-44 PM.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="Sheet1", header=False, index=False)
    return path


def test_a_blank_theory_cell_yields_no_error_not_a_zero(tmp_path):
    track = FinalTestParser().parse_file(workbook(tmp_path, theory_blank_at=5))["tracks"][0]
    assert track["theory_values"][5] is None
    assert track["errors"][5] is None, "a blank theory cell must be ungraded, never 0.0"
    # its neighbours are unaffected
    assert track["errors"][4] == pytest.approx(-0.002)
    assert track["errors"][6] == pytest.approx(0.0)


def test_without_a_blank_every_point_keeps_its_error(tmp_path):
    track = FinalTestParser().parse_file(workbook(tmp_path))["tracks"][0]
    assert all(e is not None for e in track["errors"])
    assert track["errors"][0] == pytest.approx(-0.002)


def test_a_genuine_zero_error_is_still_reported_as_zero(tmp_path):
    # The fix must not turn every 0.0 into None -- only the ones with no theory.
    # Points 1, 2, 3 of the fixture have theory == measured exactly.
    track = FinalTestParser().parse_file(workbook(tmp_path))["tracks"][0]
    assert track["errors"][1] == pytest.approx(0.0)
    assert track["errors"][2] == pytest.approx(0.0)


def test_an_ungraded_point_inside_the_window_is_a_fail_not_a_pass():
    # This is why the 0.0 mattered: the parser's own grader counts an unmeasured point
    # inside the graded window as a fail. A fabricated 0.0 would have passed it.
    p = FinalTestParser()
    errors = [0.001, None, 0.001]
    upper, lower = [0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]
    fail_points, passed = p._grade_points(errors, upper, lower, (0, 2))
    assert fail_points >= 1 and passed is False
    # and with a real measurement in that slot the same track passes
    ok_points, ok = p._grade_points([0.001, 0.001, 0.001], upper, lower, (0, 2))
    assert ok_points == 0 and ok is True


def test_fewer_than_two_points_yields_no_errors_rather_than_a_perfect_sweep():
    # The linear-fit fallback cannot fit a line to one point; it used to write 0.0 for
    # every point, handing a one-point track a flawless result.
    import inspect
    src = inspect.getsource(FinalTestParser._extract_format1_tracks)
    assert "errors = [None] * len(measured_values)" in src
    assert "errors = [0.0] * len(measured_values)" not in src
