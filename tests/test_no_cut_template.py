"""A `Lin Error` sheet that copies its own theory column is a template, not a sweep.

When the operator sweeps a unit and makes NO cut (a check before, between or
after trimming sessions), laser 1 (`LTS`, System B) and laser 3 (`LTS3`,
System C) still write a `Lin Error` sheet -- but it is the blank TEMPLATE: its
measured column is a verbatim copy of its theory column, so its error column is
identically zero. No real sweep has zero noise.

The parser used to take that template as the final sweep ("a lone `Lin Error`
implies a single trim pass"), so the unit was stored as a finished trim with
zero error at every point and linearity PASS. 1,182 tracks in the work database
landed that way, 1,181 of them "PASS", and they sit inside every laser yield,
median and drift baseline. Their only real measurement is the `test` sweep, so
they belong on the existing untrimmed-only path (`trim_pass_count == 0`).

The rule is deliberately NARROW. It needs BOTH halves:
  * no `Trim N` sheet at all, AND
  * `Lin Error` measured == theory to 1e-9 over at least 10 graded rows.
A lone `Lin Error` carrying REAL data keeps today's behaviour, and a file that
HAS `Trim N` sheets is never touched whatever its `Lin Error` looks like. Both
of those are pinned below (`test_real_lone_lin_error_*`, `test_trim_sheet_*`)
and must not move.

Every workbook here is SYNTHETIC, built under `tmp_path`. Nothing is copied out
of `Work Files/` (real customer data).
"""
import logging
import math

import pytest

# System B data sheet layout (see utils.constants.SYSTEM_B_COLUMNS):
#   A measured volts | B index | C theory volts | D error | E position
#   F upper limit    | G lower limit
_MEASURED, _INDEX, _THEORY, _ERROR, _POSITION, _UPPER, _LOWER = range(7)

_N_POINTS = 61          # matches the real 8340 sweeps
_TRAVEL = 0.5           # position step
_TEST_VOLTS = 10.0
_SPEC = 0.2             # +/- linearity limit
_BLANK_LEAD = 6         # blank rows the station leaves at the top of column D


def _theory(i: int) -> float:
    """The ideal output ramp the station writes into column C."""
    return _TEST_VOLTS * i / (_N_POINTS - 1)


def _position(i: int) -> float:
    return -((_N_POINTS - 1) / 2.0) * _TRAVEL + i * _TRAVEL


def _noise(i: int, amplitude: float) -> float:
    """A small deterministic sinusoid -- what a real sweep's error looks like."""
    return amplitude * math.sin(2.0 * math.pi * i / 17.0)


def _write_sweep(ws, measured_of, *, error_column, limits: bool = True):
    """Fill one System B data sheet. Row 1 carries data AND the metadata cells,
    exactly as the real station workbooks do.

    `error_column` is one of:
      "real"   -- the station computed measured - theory, as on a trimmed sheet
      "blank"  -- column D left entirely empty
      "zeros"  -- the shape the station actually writes on a no-cut file: blank
                  for the first few rows, then a literal 0.0 all the way down,
                  whatever the real measured - theory difference is. Confirmed
                  on 75 of the 136 no-cut files in the LTS development slice.
    """
    assert error_column in ("real", "blank", "zeros")
    for i in range(_N_POINTS):
        row = i + 1                       # openpyxl is 1-based; data starts at row 1
        measured = measured_of(i)
        ws.cell(row=row, column=_MEASURED + 1, value=measured)
        ws.cell(row=row, column=_INDEX + 1, value=i)
        ws.cell(row=row, column=_THEORY + 1, value=_theory(i))
        if error_column == "real":
            ws.cell(row=row, column=_ERROR + 1, value=measured - _theory(i))
        elif error_column == "zeros" and i >= _BLANK_LEAD:
            ws.cell(row=row, column=_ERROR + 1, value=0.0)
        ws.cell(row=row, column=_POSITION + 1, value=_position(i))
        if limits:
            ws.cell(row=row, column=_UPPER + 1, value=_SPEC)
            ws.cell(row=row, column=_LOWER + 1, value=-_SPEC)
    # Metadata row 0: K1 measured angle, L1 unit length, R1 resistance.
    ws.cell(row=1, column=11, value=30.0)
    ws.cell(row=1, column=12, value=30.0)
    ws.cell(row=1, column=18, value=10500.0)


def _make_workbook(path, *, lin_error, trim_sheets=(), test_noise=0.03,
                   test_error="blank"):
    """Build a System B workbook.

    `lin_error` is one of:
      "template"  -- measured column IS the theory column (the no-cut template)
      "real"      -- measured = theory + a small sinusoid (a genuine final sweep)
      "blank"     -- measured column empty, error column populated (the shape of
                     the real 6126 "Touch-up_Lin Out" file in the sample corpus)
      None        -- no `Lin Error` sheet at all

    `test_error` picks the shape of the `test` sheet's own error column --
    see `_write_sweep`.
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws_test = wb.active
    ws_test.title = "test"
    # The pre-trim sweep: real, noisy, and the ONLY real measurement in a
    # no-cut file.
    _write_sweep(ws_test, lambda i: _theory(i) + _noise(i, test_noise),
                 error_column=test_error)

    for n in trim_sheets:
        ws = wb.create_sheet(title=f"Trim {n}")
        _write_sweep(ws, lambda i: _theory(i) + _noise(i, 0.01), error_column="real")

    if lin_error is not None:
        ws = wb.create_sheet(title="Lin Error")
        if lin_error == "template":
            # measured == theory at every point; error column identically 0.0
            _write_sweep(ws, _theory, error_column="real")
        elif lin_error == "real":
            _write_sweep(ws, lambda i: _theory(i) + _noise(i, 0.01),
                         error_column="real")
        elif lin_error == "blank":
            for i in range(_N_POINTS):
                row = i + 1
                ws.cell(row=row, column=_INDEX + 1, value=i)
                ws.cell(row=row, column=_THEORY + 1, value=_theory(i))
                ws.cell(row=row, column=_ERROR + 1, value=_noise(i, 0.01))
                ws.cell(row=row, column=_POSITION + 1, value=_position(i))
                ws.cell(row=row, column=_UPPER + 1, value=_SPEC)
                ws.cell(row=row, column=_LOWER + 1, value=-_SPEC)
        else:  # pragma: no cover - test author error
            raise AssertionError(f"unknown lin_error kind {lin_error!r}")

    wb.save(path)
    return path


def _parse_one_track(path):
    from laser_trim_analyzer.core.parser import ExcelParser
    from laser_trim_analyzer.core.models import SystemType

    result = ExcelParser().parse_file(path)
    assert result["metadata"].system == SystemType.B, (
        "fixture must be recognised as a System B workbook")
    tracks = result["tracks"]
    assert len(tracks) == 1, f"expected one track, got {len(tracks)}"
    return tracks[0]


def _name(tmp_path):
    # System B naming the parser expects: model_serial_track_...
    return tmp_path / "9999_12_TA_Test Data_1-2-2026_9-00 AM.xlsx"


# --------------------------------------------------------------------------
# (a) the defect: a template `Lin Error` and no `Trim N` is a no-cut file
# --------------------------------------------------------------------------

def test_template_lin_error_with_no_trim_sheets_is_untrimmed_only(tmp_path):
    """The whole point. A no-cut file must not be recorded as a finished trim."""
    path = _make_workbook(_name(tmp_path), lin_error="template")
    track = _parse_one_track(path)

    assert track["trim_pass_count"] == 0, (
        "no cut was made, so no trim pass happened -- the lone template "
        f"`Lin Error` must not imply one (got {track['trim_pass_count']})")
    assert track.get("is_untrimmed_only") is True, (
        "the file's only real measurement is the `test` sweep, so it belongs "
        "on the untrimmed-only path")


def test_template_lin_error_reports_no_trimmed_error_array(tmp_path):
    """A column of zeros must never reach the app as a final result: that is
    what turned 1,181 of 1,182 such tracks into 'linearity PASS'."""
    path = _make_workbook(_name(tmp_path), lin_error="template")
    track = _parse_one_track(path)

    errors = track.get("errors")
    assert not errors, (
        "the template's zero error column must not be reported as the trimmed "
        f"result (got {len(errors or [])} points)")
    assert not track.get("positions"), (
        "no trimmed sweep exists, so there are no trimmed positions either")


def test_template_lin_error_keeps_the_real_test_sweep(tmp_path):
    """Rerouting must PRESERVE the real measurement, not discard the file."""
    path = _make_workbook(_name(tmp_path), lin_error="template")
    track = _parse_one_track(path)

    untrimmed = track.get("untrimmed_errors") or []
    assert len(untrimmed) == _N_POINTS, (
        f"the real `test` sweep must survive (got {len(untrimmed)} points)")
    assert any(abs(e) > 1e-6 for e in untrimmed), (
        "the preserved sweep must be the REAL noisy one, not another row of zeros")


def test_template_lin_error_keeps_the_real_sweep_through_a_zeroed_error_column(tmp_path):
    """The no-cut station leaves BOTH error columns zero, not just the
    template's. Rerouting must still hand the app the real sweep, recovered
    from measured - theory, rather than swapping a fake-zero trimmed result for
    a fake-zero pre-trim one.

    Measured on the LTS development slice: 75 of the 136 no-cut files have a
    `test` error column that is blank-then-literal-zero while their real
    measured - theory reaches 0.19 V -- far outside any linearity spec."""
    path = _make_workbook(_name(tmp_path), lin_error="template",
                          test_error="zeros")
    track = _parse_one_track(path)

    assert track.get("is_untrimmed_only") is True
    untrimmed = track.get("untrimmed_errors") or []
    assert len(untrimmed) == _N_POINTS, (
        f"the real `test` sweep must survive (got {len(untrimmed)} points)")
    non_zero = [e for e in untrimmed if e is not None and not math.isnan(e)
                and abs(e) > 1e-9]
    assert len(non_zero) >= _N_POINTS // 2, (
        "the station's zero error column was taken at face value: the "
        f"preserved sweep has only {len(non_zero)} real points of "
        f"{len(untrimmed)}. It must be recovered from measured - theory.")
    assert max(abs(e) for e in non_zero) == pytest.approx(0.03, abs=5e-3), (
        "the recovered sweep must be the sinusoid the fixture measured")


def test_template_lin_error_is_logged_once_at_info(tmp_path, caplog):
    """A rebuild log has to show how often this fired."""
    path = _make_workbook(_name(tmp_path), lin_error="template")
    with caplog.at_level(logging.INFO, logger="laser_trim_analyzer.core.parser"):
        _parse_one_track(path)

    hits = [r for r in caplog.records
            if "blank template" in r.message and path.name in r.message]
    assert len(hits) == 1, (
        f"expected exactly one INFO line naming the file, got {len(hits)}: "
        f"{[r.message for r in caplog.records]}")
    assert hits[0].levelno == logging.INFO


# --------------------------------------------------------------------------
# (b) + the blank-measured shape: behaviour that must NOT move
# --------------------------------------------------------------------------

def test_real_lone_lin_error_still_counts_as_a_trim_pass(tmp_path):
    """PINS TODAY'S BEHAVIOUR. A lone `Lin Error` with real data is a real
    final sweep -- such files exist in the sample corpus and must be untouched."""
    path = _make_workbook(_name(tmp_path), lin_error="real")
    track = _parse_one_track(path)

    assert track["trim_pass_count"] == 1
    assert not track.get("is_untrimmed_only")
    errors = track.get("errors") or []
    assert len(errors) == _N_POINTS
    assert any(abs(e) > 1e-6 for e in errors), "the real sweep's error must survive"


def test_real_lone_lin_error_with_blank_measured_column_is_untouched(tmp_path):
    """PINS TODAY'S BEHAVIOUR for the other lone-`Lin Error` shape in the
    sample corpus (`6126_114_..._Touch-up_Lin Out.xls`): the measured column is
    empty but the error column is real. Too few comparable rows to call it a
    template, so the rule must not fire."""
    path = _make_workbook(_name(tmp_path), lin_error="blank")
    track = _parse_one_track(path)

    assert track["trim_pass_count"] == 1
    assert not track.get("is_untrimmed_only")
    assert len(track.get("errors") or []) == _N_POINTS


# --------------------------------------------------------------------------
# (c) a file WITH `Trim N` sheets is never rerouted, whatever `Lin Error` holds
# --------------------------------------------------------------------------

def test_trim_sheet_present_is_never_rerouted_even_if_lin_error_looks_blank(tmp_path):
    """PINS TODAY'S BEHAVIOUR. The rule requires BOTH halves; a cut demonstrably
    happened here, so this file stays a trimmed track."""
    path = _make_workbook(_name(tmp_path), lin_error="template", trim_sheets=(1,))
    track = _parse_one_track(path)

    assert track["trim_pass_count"] == 1, "the `Trim 1` sheet is the pass count"
    assert not track.get("is_untrimmed_only")
    assert len(track.get("positions") or []) == _N_POINTS, (
        "the trimmed sweep must still be extracted")


def test_trim_sheet_present_keeps_lin_error_as_the_final_sweep(tmp_path):
    """PINS TODAY'S BEHAVIOUR, and is what gives the "no `Trim N`" half of the
    condition its teeth.

    Sheet priority is `Lin Error` > highest `Trim N`. This fixture's `Lin Error`
    is the template (error identically 0.0) while its `Trim 1` carries a ~0.01 V
    sinusoid, so the two are told apart by the error array alone. If the rule
    ever fired on a file that HAS `Trim N` sheets, `Lin Error` would be
    discarded and `Trim 1` would silently become the final sweep -- a different
    answer for every multi-pass file in the database, not just the no-cut ones.
    """
    path = _make_workbook(_name(tmp_path), lin_error="template", trim_sheets=(1,))
    track = _parse_one_track(path)

    errors = [e for e in (track.get("errors") or []) if e is not None and not math.isnan(e)]
    assert len(errors) == _N_POINTS
    assert max(abs(e) for e in errors) == 0.0, (
        "`Lin Error` must still win the sheet priority: a non-zero error array "
        "means `Trim 1` was read instead, so the rule fired on a file that has "
        "`Trim N` sheets")


def test_two_trim_sheets_with_template_lin_error_keep_their_pass_count(tmp_path):
    path = _make_workbook(_name(tmp_path), lin_error="template", trim_sheets=(1, 2))
    track = _parse_one_track(path)

    assert track["trim_pass_count"] == 2
    assert not track.get("is_untrimmed_only")
    errors = [e for e in (track.get("errors") or []) if e is not None and not math.isnan(e)]
    assert max(abs(e) for e in errors) == 0.0, "`Lin Error` still wins the priority"


# --------------------------------------------------------------------------
# (d) the helper on its own
# --------------------------------------------------------------------------

def _frame(measured, theory):
    """A minimal System B-shaped frame: col 0 measured, col 2 theory."""
    import pandas as pd
    return pd.DataFrame({0: measured, 1: list(range(len(measured))), 2: theory})


def test_helper_says_true_for_a_template():
    from laser_trim_analyzer.core.parser import ExcelParser
    theory = [_theory(i) for i in range(_N_POINTS)]
    assert ExcelParser._lin_error_is_template(_frame(list(theory), theory)) is True


def test_helper_says_false_for_a_real_sweep():
    from laser_trim_analyzer.core.parser import ExcelParser
    theory = [_theory(i) for i in range(_N_POINTS)]
    measured = [t + _noise(i, 0.01) for i, t in enumerate(theory)]
    assert ExcelParser._lin_error_is_template(_frame(measured, theory)) is False


def test_helper_rejects_a_difference_far_below_any_real_noise_floor():
    """1e-9 V is four orders of magnitude under the tightest spec: anything
    that differs at all is a measurement, not a copied column."""
    from laser_trim_analyzer.core.parser import ExcelParser
    theory = [_theory(i) for i in range(_N_POINTS)]
    measured = list(theory)
    measured[30] += 1e-6
    assert ExcelParser._lin_error_is_template(_frame(measured, theory)) is False


def test_helper_needs_at_least_ten_comparable_rows():
    """Too little evidence must mean 'leave it alone', never 'template'."""
    from laser_trim_analyzer.core.parser import ExcelParser
    theory = [_theory(i) for i in range(5)]
    assert ExcelParser._lin_error_is_template(_frame(list(theory), theory)) is False


def test_helper_says_false_for_a_non_numeric_sheet():
    from laser_trim_analyzer.core.parser import ExcelParser
    labels = [f"row {i}" for i in range(_N_POINTS)]
    assert ExcelParser._lin_error_is_template(_frame(labels, labels)) is False


def test_helper_says_false_for_a_blank_measured_column():
    """The real 6126 shape: nothing to compare, so no claim is made."""
    from laser_trim_analyzer.core.parser import ExcelParser
    import pandas as pd
    theory = [_theory(i) for i in range(_N_POINTS)]
    assert ExcelParser._lin_error_is_template(
        _frame([None] * _N_POINTS, theory)) is False


def test_helper_says_false_for_a_frame_with_no_theory_column():
    """A layout the parser has never seen must not be guessed at."""
    from laser_trim_analyzer.core.parser import ExcelParser
    import pandas as pd
    assert ExcelParser._lin_error_is_template(
        pd.DataFrame({0: [1.0] * _N_POINTS})) is False


def test_helper_says_false_for_an_empty_frame():
    from laser_trim_analyzer.core.parser import ExcelParser
    import pandas as pd
    assert ExcelParser._lin_error_is_template(pd.DataFrame()) is False


def test_helper_uses_the_parsers_own_column_constants():
    """Guard against the 0/2 column indices being hard-coded here and drifting
    apart from the parser's map."""
    from laser_trim_analyzer.utils.constants import SYSTEM_B_COLUMNS
    assert SYSTEM_B_COLUMNS["measured_volts"] == _MEASURED
    assert SYSTEM_B_COLUMNS["theory_volts"] == _THEORY
