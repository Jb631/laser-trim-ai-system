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


def _write_sweep(ws, measured_of, *, error_column, limits: bool = True, theory=True):
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
        if theory is True:
            ws.cell(row=row, column=_THEORY + 1, value=_theory(i))
        elif theory == "text":
            # A theory column of text: `_get_column_data` yields NOTHING, which is the shape of the
            # real mis-mapped 8504-2 shop files in the sample corpus (theory_volts=0 beside a 23-point sweep).
            ws.cell(row=row, column=_THEORY + 1, value="n/a")
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


def _write_ragged_test_sweep(ws, *, measured_short: int):
    """A `test` sweep whose per-point columns do NOT all end together.

    The measured column stops `measured_short` rows early, and the error column
    carries one blank at exactly that row. Both halves are needed to make the
    readers disagree: the trimmed reader reads the error column with
    `allow_nan=True` (a mid-column blank keeps the length, so positions/limits
    stay full-length) while the pre-trim reader stops dead at the first blank
    and then cannot recover past the end of the measured column. A measured
    column that is merely short self-heals, because the trimmed reader truncates
    positions to match and the limits follow.
    """
    end = _N_POINTS - measured_short
    for i in range(_N_POINTS):
        row = i + 1
        measured = _theory(i) + _noise(i, 0.03)
        if i < end:
            ws.cell(row=row, column=_MEASURED + 1, value=measured)
        ws.cell(row=row, column=_INDEX + 1, value=i)
        ws.cell(row=row, column=_THEORY + 1, value=_theory(i))
        if i != end:
            ws.cell(row=row, column=_ERROR + 1, value=measured - _theory(i))
        ws.cell(row=row, column=_POSITION + 1, value=_position(i))
        ws.cell(row=row, column=_UPPER + 1, value=_SPEC)
        ws.cell(row=row, column=_LOWER + 1, value=-_SPEC)
    ws.cell(row=1, column=11, value=30.0)
    ws.cell(row=1, column=12, value=30.0)
    ws.cell(row=1, column=18, value=10500.0)


# The double-resolution family (confirmed models 1844202 / 1844204 / 1844205)
# records the measured column at DOUBLE the position/theory grid: coarse
# position[i] lines up with fine measured[2i]. Subtracting row-by-row compares
# values at different physical positions and produces a garbage ramp.
_DR_COARSE = 32                      # position/theory points, as on 1844205
_DR_FINE = 2 * _DR_COARSE - 1        # measured points
_DR_NOISE = 0.02


def _dr_theory(i: int) -> float:
    return _TEST_VOLTS * i / (_DR_COARSE - 1)


def _dr_position(i: int) -> float:
    return -((_DR_COARSE - 1) / 2.0) * _TRAVEL + i * _TRAVEL


def _dr_measured(j: int) -> float:
    """Fine grid: measured[2i] is the real reading at coarse point i."""
    return _TEST_VOLTS * j / (2 * (_DR_COARSE - 1)) + _noise(j, _DR_NOISE)


def _write_double_resolution_test_sweep(ws):
    for j in range(_DR_FINE):
        ws.cell(row=j + 1, column=_MEASURED + 1, value=_dr_measured(j))
    for i in range(_DR_COARSE):
        row = i + 1
        ws.cell(row=row, column=_INDEX + 1, value=i)
        ws.cell(row=row, column=_THEORY + 1, value=_dr_theory(i))
        # The station's own error column: fine measured[i] minus coarse
        # theory[i], row by row -- the garbage this branch exists to replace.
        ws.cell(row=row, column=_ERROR + 1, value=_dr_measured(i) - _dr_theory(i))
        ws.cell(row=row, column=_POSITION + 1, value=_dr_position(i))
        ws.cell(row=row, column=_UPPER + 1, value=_SPEC)
        ws.cell(row=row, column=_LOWER + 1, value=-_SPEC)
    ws.cell(row=1, column=11, value=30.0)
    ws.cell(row=1, column=12, value=30.0)
    ws.cell(row=1, column=18, value=10500.0)


def _write_double_resolution_template(ws):
    """The no-cut template at the coarse grid: measured IS theory."""
    for i in range(_DR_COARSE):
        row = i + 1
        ws.cell(row=row, column=_MEASURED + 1, value=_dr_theory(i))
        ws.cell(row=row, column=_INDEX + 1, value=i)
        ws.cell(row=row, column=_THEORY + 1, value=_dr_theory(i))
        ws.cell(row=row, column=_ERROR + 1, value=0.0)
        ws.cell(row=row, column=_POSITION + 1, value=_dr_position(i))
        ws.cell(row=row, column=_UPPER + 1, value=_SPEC)
        ws.cell(row=row, column=_LOWER + 1, value=-_SPEC)


def _make_workbook(path, *, lin_error, trim_sheets=(), test_noise=0.03,
                   test_error="blank", measured_short=0,
                   double_resolution=False, test_theory=True):
    """Build a System B workbook.

    `lin_error` is one of:
      "template"  -- measured column IS the theory column (the no-cut template)
      "real"      -- measured = theory + a small sinusoid (a genuine final sweep)
      "blank"     -- measured column empty, error column populated (the shape of
                     the real 6126 "Touch-up_Lin Out" file in the sample corpus)
      None        -- no `Lin Error` sheet at all

    `test_error` picks the shape of the `test` sheet's own error column --
    see `_write_sweep`. `measured_short` and `double_resolution` select the
    ragged and double-resolution `test` sweeps instead.
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws_test = wb.active
    ws_test.title = "test"
    # The pre-trim sweep: real, noisy, and the ONLY real measurement in a
    # no-cut file.
    if double_resolution:
        _write_double_resolution_test_sweep(ws_test)
    elif measured_short:
        _write_ragged_test_sweep(ws_test, measured_short=measured_short)
    else:
        _write_sweep(ws_test, lambda i: _theory(i) + _noise(i, test_noise),
                     error_column=test_error, theory=test_theory)

    for n in trim_sheets:
        ws = wb.create_sheet(title=f"Trim {n}")
        _write_sweep(ws, lambda i: _theory(i) + _noise(i, 0.01), error_column="real")

    if lin_error is not None:
        ws = wb.create_sheet(title="Lin Error")
        if lin_error == "template" and double_resolution:
            _write_double_resolution_template(ws)
        elif lin_error == "template":
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
# the rerouted track must not carry a laser pass it never ran
# --------------------------------------------------------------------------

def _passes(track):
    return [(p.get("pass_index"), p.get("sheet")) for p in (track.get("trim_passes") or [])]


def test_rerouted_track_carries_no_trim_pass(tmp_path):
    """`trim_passes` is the table that will teach a model what a cut does to a
    curve, so a row in it is an assertion that a cut happened.

    `_read_trim_passes` re-derives its sheet list from the workbook, and
    `core.trim_passes.pass_sheets` appends `Lin Error` as a pass for lasers 1
    and 3 unconditionally -- so nulling the local `lin_error_sheet` did not
    reach it. Every rerouted track still came back carrying one pass: the
    template's ~zero error sweep attached to a full ~3000-unit cut recipe, on a
    unit the same parse had just declared un-cut. Confirmed on 40 of the first
    40 fired real files.
    """
    path = _make_workbook(_name(tmp_path), lin_error="template")
    track = _parse_one_track(path)

    assert track.get("is_untrimmed_only") is True
    assert _passes(track) == [], (
        "a file with no cut has no pass: the template sheet must not be "
        f"captured as one (got {_passes(track)})")


def test_rerouted_track_has_no_pass_carrying_a_cut_recipe(tmp_path):
    """The specific falsehood: zero linearity error attributed to a real cut."""
    path = _make_workbook(_name(tmp_path), lin_error="template")
    track = _parse_one_track(path)

    for p in (track.get("trim_passes") or []):
        raise AssertionError(
            f"pass {p.get('pass_index')} from sheet {p.get('sheet')!r} asserts a "
            f"cut (laser_cut_length={p.get('laser_cut_length')!r}) with "
            f"{len(p.get('errors') or [])} error points on an un-cut unit")


def test_trim_sheets_still_capture_their_passes(tmp_path):
    """PINS TODAY'S BEHAVIOUR. A file that really was cut keeps every pass,
    in order, with its real sweep -- the exclusion must reach the template
    sheet only."""
    path = _make_workbook(_name(tmp_path), lin_error="real", trim_sheets=(1,))
    track = _parse_one_track(path)

    assert _passes(track) == [(1, "Trim 1"), (2, "Lin Error")]
    first = (track.get("trim_passes") or [])[0]
    errors = [e for e in (first.get("errors") or []) if e is not None]
    assert len(errors) == _N_POINTS
    assert max(abs(e) for e in errors) == pytest.approx(0.01, abs=1e-3), (
        "the first pass must still carry `Trim 1`'s real sweep")


def test_lone_real_lin_error_still_captures_its_pass(tmp_path):
    """PINS TODAY'S BEHAVIOUR for case (b): a lone `Lin Error` with real data
    is a real final sweep and is still captured as pass 1."""
    path = _make_workbook(_name(tmp_path), lin_error="real")
    track = _parse_one_track(path)

    assert _passes(track) == [(1, "Lin Error")]
    first = (track.get("trim_passes") or [])[0]
    errors = [e for e in (first.get("errors") or []) if e is not None]
    assert len(errors) == _N_POINTS
    assert max(abs(e) for e in errors) == pytest.approx(0.01, abs=1e-3)


# --------------------------------------------------------------------------
# every per-point array that travels with the track must be the same length
# --------------------------------------------------------------------------

def test_per_point_arrays_end_together_on_a_ragged_sweep(tmp_path, caplog):
    """The spec band has to line up with the sweep it grades -- and the sweep has to survive.

    Originally this asserted that a ragged sweep was TRUNCATED to its shortest array. A final review
    showed what that cost: the pre-trim reader stops at the first blank, so one dropped cell cut a
    40-point sweep to 5, and the arrays it was being aligned with (the limits) are not even stored on
    this path. The sheet is read twice; the longer read is kept, so the arrays line up at FULL length.
    """
    path = _make_workbook(_name(tmp_path), lin_error="template", measured_short=5)
    with caplog.at_level(logging.WARNING, logger="laser_trim_analyzer.core.parser"):
        track = _parse_one_track(path)

    assert track.get("is_untrimmed_only") is True
    per_point = ("untrimmed_positions", "untrimmed_errors", "upper_limits",
                 "lower_limits", "theory_volts")
    lengths = {k: len(track[k]) for k in per_point if track.get(k) is not None}
    assert len(lengths) == len(per_point), f"an array went missing: {lengths}"
    assert set(lengths.values()) == {_N_POINTS}, f"the sweep was cut short: {lengths}"
    assert not [r for r in caplog.records if "truncating every one" in r.message]

def test_a_well_formed_sweep_is_not_warned_about(tmp_path, caplog):
    """The warning must mean something: an ordinary no-cut file must not emit it."""
    path = _make_workbook(_name(tmp_path), lin_error="template")
    with caplog.at_level(logging.WARNING, logger="laser_trim_analyzer.core.parser"):
        track = _parse_one_track(path)

    assert len(track["untrimmed_positions"]) == _N_POINTS
    assert not [r for r in caplog.records if "per-point arrays disagree" in r.message]


# --------------------------------------------------------------------------
# the double-resolution family (1844202 / 1844204 / 1844205)
# --------------------------------------------------------------------------

def test_double_resolution_no_cut_file_stores_the_corrected_pre_trim_error(tmp_path):
    """1844205 alone has 92 no-cut tracks in the work database, and no real
    file of this family takes the untrimmed-only path in the corpus or the
    slice -- so this is the only cover it has.

    These models record the measured column at 2x the position/theory grid.
    Subtracting row by row pairs each reading with a point half a grid step
    away and yields a garbage ramp (pre-trim sigma ~1.6 instead of ~0.1). The
    pre-trim reader re-pairs coarse position[i] with fine measured[2i]. A
    rerouted no-cut file of this family must get the CORRECTED error, which is
    what every normal file of the same model already gets."""
    path = _make_workbook(_name(tmp_path), lin_error="template",
                          double_resolution=True)
    track = _parse_one_track(path)

    assert track["trim_pass_count"] == 0
    assert track.get("is_untrimmed_only") is True
    assert _passes(track) == []

    errors = [e for e in (track.get("untrimmed_errors") or [])
              if e is not None and not math.isnan(e)]
    assert len(errors) == _DR_COARSE, (
        f"expected one error per coarse point, got {len(errors)}")
    worst = max(abs(e) for e in errors)
    assert worst == pytest.approx(_DR_NOISE, abs=5e-3), (
        f"max|error| is {worst:.4f}; the corrected value is ~{_DR_NOISE}. A "
        f"value of order 1 V means the station's row-by-row garbage was stored")

    # And the arrays still end together.
    lengths = {k: len(track[k]) for k in
               ("untrimmed_positions", "untrimmed_errors", "upper_limits",
                "lower_limits", "theory_volts") if track.get(k) is not None}
    assert len(set(lengths.values())) == 1, lengths


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


_MISMAPPED = [
    "Test Station/8504-2/8504-2-shop34_9-19-2024 12-27-51 PM.xlsx",
    "Test Station/8504-2/8504-2-shop35_9-19-2024 12-36-31 PM.xlsx",
]


def test_an_empty_per_point_array_does_not_delete_the_sweep(caplog):
    """Re-review finding, on the two real files that reproduce it.

    The alignment truncation keyed on `is not None`, so an EMPTY array joined the comparison with
    length 0, became the shortest, and every other array was cut to nothing -- deleting the one real
    measurement an untrimmed-only file has. `theory_volts` is the array that can legitimately be
    empty (a theory column that reads as no numbers at all). These two corpus files are a mis-mapped
    layout whose sweep was already junk, but the mechanism is general: ANY untrimmed-only track whose
    theory column reads empty lost its sweep.
    """
    import logging
    from pathlib import Path
    from laser_trim_analyzer.core.parser import ExcelParser
    root = Path("Work Files/Sample_Base_2026-04-10")
    paths = [root / rel for rel in _MISMAPPED]
    if not all(p.exists() for p in paths):
        pytest.skip("sample corpus not present")
    parser = ExcelParser()
    for path in paths:
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            (track,) = parser.parse_file(path)["tracks"]
        assert track.get("is_untrimmed_only") is True
        assert len(track["untrimmed_positions"]) == 23, path.name
        assert len(track["untrimmed_errors"]) == 23, path.name
        assert len(track["upper_limits"]) == len(track["lower_limits"]) == 23
        assert not track.get("theory_volts")                 # the empty array stays empty...
        said = [r.getMessage() for r in caplog.records if "truncating every one" in r.getMessage()]
        assert said == [], said                              # ...and nothing is truncated over it


def test_one_dropped_cell_does_not_cut_the_sweep_short(tmp_path, caplog):
    """Final review, demonstrated: the pre-trim reader stops at the first blank in the measured or
    theory column, so a single dropout turned a 40-point sweep into 5 points. The sheet is read twice;
    keep the longer read, and say so."""
    import logging
    path = _name(tmp_path)
    _make_workbook(path, lin_error="template", measured_short=5)
    with caplog.at_level(logging.WARNING):
        track = _parse_one_track(path)
    assert len(track["untrimmed_errors"]) == _N_POINTS, "a dropped cell truncated the stored sweep"
    assert len(track["untrimmed_positions"]) == _N_POINTS
    said = [r.getMessage() for r in caplog.records if "recovered pre-trim sweep stops at" in r.getMessage()]
    assert len(said) == 1 and f"of {_N_POINTS} points" in said[0]


def test_a_no_cut_file_still_keeps_the_recovered_sweep_over_its_column_of_zeros(tmp_path):
    """The tie case, which is why the sheet is read twice at all: the test sheet's own error column is
    a full-length run of literal zeros, and the recovered one is the same length and real."""
    path = _name(tmp_path)
    _make_workbook(path, lin_error="template", test_error="zeros")
    track = _parse_one_track(path)
    real = [e for e in track["untrimmed_errors"] if isinstance(e, float) and e == e and e != 0.0]
    assert len(track["untrimmed_errors"]) == _N_POINTS and len(real) > _N_POINTS // 2
