"""A recomputed final-test error is a VOLTAGE, because the limits are volts.

Format 1 sheets carry the station's own error column (D) and its per-point
limit columns (G/H). The parser uses column D only when it is at least 90%
populated; otherwise it recomputes the errors, either from measured minus
theory or from a linear fit on electrical angle. Both fallbacks used to divide
the result by the sweep's full scale, which turns a voltage into a fraction of
full scale -- while G/H stayed in volts. On a 10 V part that shrinks every
error about 10x and drops it inside a +/-0.01..0.02 V band, so the app records
a PASS for a unit the station rejected.

`7539-2-sn23_1-22-2026_2-44 PM.xls` is that file: column D populated on 161 of
its 181 rows (88.95%, just under the gate, because the station leaves the rows
it does not grade blank), full scale 9.822 V, true max |measured - theory|
0.1429 V, stored as 0.01455 with zero fail points and linearity_pass True
against a station verdict of FAIL on 149 points.

Linearity is the zero-tolerance customer disposition, so a unit error here is
not a cosmetic one: it is the difference between shipping and not shipping.

The synthetic sheets pin the arithmetic that the fixture can only pin for one
file -- one per fallback branch.
"""
import statistics

import numpy as np
import pandas as pd
import pytest

from laser_trim_analyzer.core.final_test_parser import FinalTestParser
from laser_trim_analyzer.utils.constants import FINAL_TEST_FORMAT1_COLUMNS

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "final_test"
SN23 = FIXTURES / "7539-2-sn23_1-22-2026_2-44 PM.xls"

# The synthetic sheet: a 0-10 V ramp with one deliberate bump.
N_POINTS = 100
BUMP_INDEX = 50
BUMP_VOLTS = 0.05      # outside the band in volts, comfortably inside it at 1/10th
LIMIT_VOLTS = 0.02
FULL_SCALE = 10.0      # the bump sits mid-sweep, so it moves neither end


@pytest.fixture(scope="module")
def parser():
    return FinalTestParser()


def _measured_errors(track):
    return [e for e in track["errors"]
            if e is not None and not (isinstance(e, float) and np.isnan(e))]


# ---- the real file --------------------------------------------------------

def test_the_7539_fixture_the_station_failed_is_not_stored_as_a_pass(parser):
    """The file the station failed on 149 points must not be recorded a PASS.

    This is the parser's own uncorrected grade. The processor still applies the
    analyzer's offset/slope correction on top of it, and that correction may
    legitimately disagree with the station -- but it cannot start from errors
    that are in the wrong unit.
    """
    parsed = parser.parse_file(SN23)
    track = parsed["tracks"][0]
    errors = track["errors"]

    # The gate really is the reason this file recomputes: 161 of 181 rows.
    assert len(errors) == 181
    assert track["station_fail_points"] == 149
    assert track["station_linearity_pass"] is False

    assert track["linearity_pass"] is False
    assert track["linearity_fail_points"] > 0
    # Volts. Normalised by the 9.822 V full scale this was 0.01455 -- inside
    # the 0.02 V band, which is exactly how the FAIL became a PASS.
    assert max(abs(e) for e in _measured_errors(track)) == pytest.approx(
        0.142935, abs=1e-5)
    assert max(abs(e) for e in _measured_errors(track)) > 0.1

    # ---- errors and limits are ONE unit ----------------------------------
    #
    # The obvious check -- "the largest |error| is within a factor of 20 of
    # the largest |limit|" -- does NOT discriminate on this file and would be
    # a check that passes on the broken code: normalised, max |error| is
    # 0.01455 against a 0.02 V limit, a ratio of 0.73, already well inside a
    # factor of 20 (in volts it is 7.15). A band-relative magnitude cannot
    # separate "too small by 10x" from "in spec".
    #
    # What does separate them is the sheet's own error column. On this file
    # column D equals (measured - theory) + 0.046127 with slope 1.0000, and
    # "D outside G/H" reproduces the station's column-I flag on all 161 graded
    # rows -- so D is, by the station's own arithmetic, in the unit of G/H. An
    # error trace in that unit therefore differs from D by a CONSTANT (the
    # station's compensation offset). Dividing by full scale destroys that:
    # the difference stops being constant and starts tracking the error.
    #
    #     volts : spread of (errors - D) = 0.0     (constant offset -0.046127)
    #     old   : spread of (errors - D) = 0.0656  (stdev 0.0102)
    sheet = pd.read_excel(SN23, sheet_name="Sheet1", header=None)
    column_d = pd.to_numeric(
        sheet.iloc[:, FINAL_TEST_FORMAT1_COLUMNS["error"]], errors="coerce")
    # The parser keeps every row of this sheet, in sheet order (its sweep runs
    # -90 -> +90, so nothing is re-sorted), which is what lets the two line up.
    assert len(track["errors"]) == len(column_d)

    paired = [(e, d) for e, d in zip(track["errors"], column_d)
              if e is not None and not np.isnan(d)]
    assert len(paired) == 161
    offsets = [e - d for e, d in paired]
    assert statistics.pstdev(offsets) < 1e-9
    assert statistics.fmean(offsets) == pytest.approx(-0.046127, abs=1e-6)


# ---- the arithmetic, one synthetic sheet per fallback branch ---------------

def _write_format1(path, *, with_theory=True):
    """A minimal Format 1 sheet whose error column is under the 90% gate.

    Columns are taken from FINAL_TEST_FORMAT1_COLUMNS, not hard-coded: a
    0-10 V ramp in A, the ideal ramp in C, a +0.05 V bump at one mid-sweep
    point in A, +/-0.02 V limits in G/H, and the station's per-point flags in
    I so the graded window is read off the sheet the way a real one is.

    Column D carries the true volt error on 76 of the 100 rows (76%), which is
    under the parser's 90% gate -- the same, entirely normal, situation a
    station creates when it leaves more than 10% of the sweep ungraded.
    """
    cols = FINAL_TEST_FORMAT1_COLUMNS
    ideal = [FULL_SCALE * i / (N_POINTS - 1) for i in range(N_POINTS)]
    measured = list(ideal)
    measured[BUMP_INDEX] += BUMP_VOLTS
    electrical_angle = [-90.0 + 180.0 * i / (N_POINTS - 1) for i in range(N_POINTS)]

    file_errors = [None] * N_POINTS
    for i in range(12, 88):
        file_errors[i] = measured[i] - ideal[i]

    flags = [None] * N_POINTS
    for i in range(6, 94):
        flags[i] = 0
    flags[BUMP_INDEX] = 1

    by_column = {
        cols["measured"]: measured,
        cols["index"]: list(range(1, N_POINTS + 1)),
        cols["theory"]: (list(ideal) if with_theory else [None] * N_POINTS),
        cols["error"]: file_errors,
        cols["electrical_angle"]: electrical_angle,
        cols["upper_limit"]: [LIMIT_VOLTS] * N_POINTS,
        cols["lower_limit"]: [-LIMIT_VOLTS] * N_POINTS,
        cols["station_flag"]: flags,
    }
    width = max(by_column) + 1
    frame = pd.DataFrame(
        {c: by_column.get(c, [None] * N_POINTS) for c in range(width)})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Sheet1", header=False, index=False)
    return path


def test_recomputed_errors_equal_measured_minus_theory_in_volts(tmp_path, parser):
    """measured - theory, full stop. 0.05 V, not 0.05/10 = 0.005."""
    path = _write_format1(tmp_path / "9999-1-sn1_1-2-2026_10-00 AM.xlsx")
    parsed = parser.parse_file(path)
    track = parsed["tracks"][0]

    assert parsed["format"] == "format1"
    assert len(track["errors"]) == N_POINTS
    # The station's flags are what bound the grade, and the bump is inside.
    assert track["graded_window"] == (6, 93)
    assert track["graded_window_source"] == "flags"

    assert track["errors"][BUMP_INDEX] == pytest.approx(BUMP_VOLTS, abs=1e-6)
    # Normalised it would have been exactly 0.005 -- inside the +/-0.02 band.
    assert track["errors"][BUMP_INDEX] != pytest.approx(
        BUMP_VOLTS / FULL_SCALE, abs=1e-6)
    assert all(e == pytest.approx(0.0, abs=1e-9)
               for i, e in enumerate(track["errors"]) if i != BUMP_INDEX)

    assert track["linearity_fail_points"] == 1
    assert track["linearity_pass"] is False


def test_linear_fit_residuals_are_in_volts_when_theory_is_missing(tmp_path, parser):
    """With no theory column the parser fits an ideal line -- still in volts.

    The fit absorbs a little of a single outlier, so the expected residual is
    computed here with the same polyfit rather than asserted as a round number.
    """
    path = _write_format1(tmp_path / "9999-1-sn2_1-2-2026_10-05 AM.xlsx",
                          with_theory=False)
    track = parser.parse_file(path)["tracks"][0]

    assert all(t is None for t in track["theory_values"])

    angles = np.array(track["electrical_angles"])
    measured = np.array(track["measured_values"])
    expected = (measured - np.polyval(np.polyfit(angles, measured, 1), angles))
    assert track["errors"] == pytest.approx(list(expected), abs=1e-12)

    # ~0.0495 V: the bump minus the sliver the fit took. Normalised it was
    # 0.00495, a quarter of the band instead of two and a half times it.
    assert track["errors"][BUMP_INDEX] == pytest.approx(0.0495, abs=1e-4)
    assert track["errors"][BUMP_INDEX] > LIMIT_VOLTS

    assert track["linearity_fail_points"] == 1
    assert track["linearity_pass"] is False
