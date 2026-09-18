import logging
from pathlib import Path

import pandas as pd
import pytest
from laser_trim_analyzer.core.models import SystemType
from laser_trim_analyzer.core.parser import ExcelParser
from laser_trim_analyzer.core.trim_passes import pass_sheets, read_pass

DLTS = "tests/fixtures/trim/dlts_8232-1_243.xls"
LTS = "tests/fixtures/trim/lts_8232-1_194.xls"


def test_system_a_orders_passes_and_drops_the_untrimmed_sweep():
    names = ["Model Parameters", "SEC1 TRK1 0", "SEC1 TRK1 1 TRM1",
             "SEC1 TRK1 2 TRM2", "SEC1 TRK1 3 TRM2"]
    assert pass_sheets(names, SystemType.A, "TRK1") == [
        (1, "SEC1 TRK1 1 TRM1"), (2, "SEC1 TRK1 2 TRM2"), (3, "SEC1 TRK1 3 TRM2")]


def test_system_a_older_naming_puts_trm_last():
    names = ["SEC1 TRK1 0", "SEC1 TRK1 1", "SEC1 TRK1 TRM"]
    got = pass_sheets(names, SystemType.A, "TRK1")
    assert [s for _, s in got] == ["SEC1 TRK1 1", "SEC1 TRK1 TRM"]


def test_system_a_ignores_the_other_track():
    names = ["SEC1 TRK1 1 TRM1", "SEC1 TRK2 1 TRM1"]
    assert pass_sheets(names, SystemType.A, "TRK2") == [(1, "SEC1 TRK2 1 TRM1")]


def test_system_b_orders_trim_sheets_then_lin_error():
    names = ["test", "Trim 1", "Trim 2", "Lin Error", "Notes"]
    assert pass_sheets(names, SystemType.B, "default") == [
        (1, "Trim 1"), (2, "Trim 2"), (3, "Lin Error")]


def test_system_c_reads_as_system_b():
    names = ["test", "Trim 1", "Lin Error"]
    assert pass_sheets(names, SystemType.C, "default") == [(1, "Trim 1"), (2, "Lin Error")]


def test_no_pass_sheets_is_empty_not_an_error():
    assert pass_sheets(["Notes", "hold"], SystemType.B, "default") == []


def test_read_pass_returns_a_full_sweep():
    df = pd.read_excel(DLTS, sheet_name="SEC1 TRK1 2 TRM2", header=None)
    sweep = read_pass(df, SystemType.A, start_row=1)
    assert len(sweep["positions"]) == len(sweep["errors"]) > 50
    assert max(abs(e) for e in sweep["errors"] if e is not None) == pytest.approx(0.06262, abs=1e-4)


def test_read_pass_keeps_blank_limits_as_none():
    """Ignored points carry no limit. They must stay None, never 0.0 —
    a 0.0 limit grades every point as failing."""
    df = pd.read_excel(DLTS, sheet_name="SEC1 TRK1 2 TRM2", header=None)
    sweep = read_pass(df, SystemType.A, start_row=1)
    assert any(v is None for v in sweep["upper_limits"])
    assert 0.0 not in [v for v in sweep["upper_limits"] if v is not None]


def test_system_a_pass_sheets_carry_per_point_process_data():
    """The richest thing in these files: what was cut at EVERY position, not
    one cut length for the whole pass. This is what a cut-length model learns
    from, so losing it would gut the deferred work."""
    df = pd.read_excel(DLTS, sheet_name="SEC1 TRK1 2 TRM2", header=None)
    sweep = read_pass(df, SystemType.A, start_row=1)
    assert len(sweep["cut_lengths"]) == len(sweep["positions"])
    assert any(v is not None for v in sweep["cut_lengths"])
    assert any(v is not None for v in sweep["trim_currents"])
    # Predicted vs actually-applied correction: the machine's own accuracy.
    assert "pred_deltas" in sweep and "used_deltas" in sweep


def test_system_b_has_no_per_point_process_data_and_says_so():
    """System B sheets are the plain sweep layout. The extra keys must be
    absent rather than present-and-empty, so a consumer can tell the
    difference between 'not recorded' and 'recorded as nothing'."""
    df = pd.read_excel(LTS, sheet_name="Trim 1", header=None)
    sweep = read_pass(df, SystemType.B, start_row=0)
    assert "cut_lengths" not in sweep
    assert len(sweep["positions"]) > 50


def test_read_pass_aligns_every_key_to_positions_length():
    """positions is the anchor column: it is recorded for every tested point,
    including points that carry no error/limit because they are ignored (an
    ungraded head or tail of the sweep). Both real fixtures below have such
    a gap — DLTS `SEC1 TRK1 2 TRM2` leads with 6 ignored points before the
    graded window opens; LTS `Trim 1` trails with 6 (rows with a real
    position/measured-volts but no error or limit, i.e. positions 23-27.5).
    A column reader that strips its OWN trailing blanks independently of
    positions silently shortens errors/upper_limits/lower_limits relative
    to positions in exactly this situation, shifting every value recorded
    after the gap out of point-for-point alignment without raising or even
    changing the type of what comes back. That is worse than the 0.0-limit
    bug this module's docstring warns about, because nothing downstream
    would even look wrong until the cut-length model trained on it."""
    a = pd.read_excel(DLTS, sheet_name="SEC1 TRK1 2 TRM2", header=None)
    a_sweep = read_pass(a, SystemType.A, start_row=1)
    assert len(a_sweep["errors"]) == len(a_sweep["positions"])
    assert len(a_sweep["upper_limits"]) == len(a_sweep["positions"])
    assert len(a_sweep["lower_limits"]) == len(a_sweep["positions"])

    b = pd.read_excel(LTS, sheet_name="Trim 1", header=None)
    b_sweep = read_pass(b, SystemType.B, start_row=0)
    assert len(b_sweep["errors"]) == len(b_sweep["positions"])
    assert len(b_sweep["upper_limits"]) == len(b_sweep["positions"])
    assert len(b_sweep["lower_limits"]) == len(b_sweep["positions"])
    # Rows 51-56 of this fixture's Trim 1 sheet have a real recorded
    # position (23 .. 27.5) but no graded error: an ungraded tail, not a
    # missing one. It must read back as None at the SAME index as that
    # position, not be dropped and let an earlier value slide into its slot.
    assert b_sweep["positions"][-1] == pytest.approx(27.5)
    assert b_sweep["errors"][-1] is None


def test_aligned_column_longer_than_positions_logs_a_warning(caplog):
    """The alignment fix above pads a SHORT column to positions' length.
    The other direction matters just as much: a column LONGER than
    positions must not be silently truncated, because a future key read
    through the same helper could reproduce the exact bug just fixed with
    nothing to catch it.

    This isn't hypothetical: SYSTEM_B_COLUMNS['measured_volts'] runs 2 rows
    past where positions goes blank on every System-B pass sheet checked,
    with genuine data in the overflow (e.g. lts_8232-1_194.xls Trim 1 has
    5.021913 / 5.021891 past position's last real row). read_pass doesn't
    read that column today, so today's output is fine either way — but the
    guard has to be in the shared helper, not in a comment, because the
    next key added through it won't come with a fixture to catch it by
    hand. This test exercises the guard through `errors`, a key read_pass
    already returns, by making its column artificially longer than
    positions in a synthetic sheet shaped like System B's."""
    df = pd.DataFrame({
        0: [1.1, 2.2, 3.3],        # measured_volts (not read by read_pass)
        1: [0, 1, 2],
        2: [0.0, 0.5, 1.0],
        3: [0.01, 0.02, 0.03],     # error: 3 real values...
        4: [-1.0, 0.0, None],      # ...but position blank on row 2 -> n=2
        5: [0.05, 0.05, None],
        6: [-0.05, -0.05, None],
    })
    with caplog.at_level(logging.WARNING):
        sweep = read_pass(df, SystemType.B, start_row=0)

    # The guard doesn't change the output contract: still aligned, overflow
    # dropped, not raised. It only has to stop being SILENT about it.
    assert len(sweep["positions"]) == 2
    assert sweep["errors"] == [0.01, 0.02]

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("errors" in msg and "3" in msg and "2" in msg for msg in warnings), (
        f"expected a warning naming the key and both lengths, got: {warnings}")


# ---------------------------------------------------------------------------
# Parser wiring: the two reader modules above are only useful once parse_file
# actually calls them. These exercise that integration.
# ---------------------------------------------------------------------------

def test_parser_attaches_passes_to_the_track():
    result = ExcelParser().parse_file(Path(DLTS))
    track = result["tracks"][0]
    assert len(track["trim_passes"]) >= 2
    first = track["trim_passes"][0]
    assert first["pass_index"] == 1
    assert len(first["errors"]) > 50
    assert first["laser_cut_length_mm"] == 0.75   # recipe joined to sweep


def test_parser_attaches_setup_to_the_file():
    result = ExcelParser().parse_file(Path(DLTS))
    setup = result["trim_setup"]
    assert setup["initial_resistance_lower_limit"] == 4200
    assert setup["final_resistance_upper_limit"] == 5500


def test_system_b_setup_carries_the_final_resistance_spec():
    result = ExcelParser().parse_file(Path(LTS))
    setup = result["trim_setup"]
    assert setup["min_resistance"] == 5000
    assert setup["max_resistance"] == 5500


def test_everything_the_parser_returned_before_is_unchanged():
    """The keys that existed before capture must still be present."""
    track = ExcelParser().parse_file(Path(DLTS))["tracks"][0]
    for key in ("track_id", "positions", "errors", "upper_limits", "lower_limits",
                "untrimmed_positions", "untrimmed_errors", "travel_length",
                "linearity_spec", "untrimmed_resistance", "trimmed_resistance"):
        assert key in track, f"capture dropped the pre-existing key {key}"


def test_pass_reusing_an_earlier_recipe_is_joined_by_trm_label_not_position():
    """dlts_8232-1_243.xls has 3 pass sheets ("SEC1 TRK1 1 TRM1",
    "SEC1 TRK1 2 TRM2", "SEC1 TRK1 3 TRM2") but only 2 recipes
    ("SEC1-TRK1-TRM1", "SEC1-TRK1-TRM2") — pass 3 reuses TRM2's recipe
    rather than getting one of its own.

    A positional join (recipes[idx - 1]) leaves pass 3 unjoined, since
    there is no recipes[2]. The sheet name's own TRM token is what
    recovers the correct recipe: pass 3 must carry TRM2's recipe, with
    the same cut length as pass 2.
    """
    result = ExcelParser().parse_file(Path(DLTS))
    passes = result["tracks"][0]["trim_passes"]
    assert len(passes) == 3

    second, third = passes[1], passes[2]
    assert third["pass_index"] == 3
    assert "laser_cut_length_mm" in third, "pass 3 must carry a recipe, not go unjoined"
    assert third["laser_cut_length_mm"] == second["laser_cut_length_mm"] == 0.88


def test_pass_count_matching_recipe_count_still_joins_correctly():
    """dlts_8232-1_242.xls uses the same Trim Parameters sheet as the 243
    fixture above (recipes SEC1-TRK1-TRM1=0.75, SEC1-TRK1-TRM2=0.88) but
    stopped after 2 passes, so pass count equals recipe count exactly and
    a plain positional join would also happen to get this right. Covered
    so the TRM-label join can't silently break the common 1:1 case while
    fixing the reuse case above."""
    result = ExcelParser().parse_file(Path("tests/fixtures/trim/dlts_8232-1_242.xls"))
    passes = result["tracks"][0]["trim_passes"]
    assert len(passes) == 2
    assert passes[0]["laser_cut_length_mm"] == 0.75
    assert passes[1]["laser_cut_length_mm"] == 0.88


# ---------------------------------------------------------------------------
# Fix round: start_row must never be inherited from the wrong sheet.
#
# _extract_track_data's start_row is only trustworthy when it was computed
# against a REAL trimmed sheet. A track can reach _read_trim_passes via
# _extract_untrimmed_only_track instead -- reachable on System A whenever
# numbered pass sheets exist ("SEC1 TRK1 1", "SEC1 TRK1 2", ...) but none of
# them (and no separate sheet) carries a "TRM" suffix: has_trm stays False,
# highest_trm_sheet stays None, so _extract_system_a_tracks never picks a
# trimmed_sheet at all even though trim_pass_count is > 0. In that path the
# only sheet _extract_track_data ever reads is the untrimmed one, so its
# start_row describes THAT sheet's layout, not the pass sheets'.
#
# No fixture on disk exhibits this sheet-naming gap, so it is constructed
# directly here with a minimal synthetic workbook rather than hunted for.
# ---------------------------------------------------------------------------

def _write_min_system_a_sheet(writer, sheet_name, n_meta_rows, n_data_rows):
    """Minimal System A sheet: `n_meta_rows` of non-numeric filler in the
    position column (so _find_data_start must skip past them), then
    `n_data_rows` of real position/error/limit data. Column indices match
    SYSTEM_A_COLUMNS (error=6, position=7, upper_limit=8, lower_limit=9).
    Returns the position values actually written, for the caller to assert
    against.
    """
    grid = []
    positions = []
    for r in range(n_meta_rows + n_data_rows):
        row = [None] * 10
        if r < n_meta_rows:
            row[0] = "META"          # non-numeric -> not a data-start candidate
        else:
            i = r - n_meta_rows
            pos = float(i)
            row[6] = round(0.01 * (i + 1), 4)   # error
            row[7] = pos                         # position
            row[8] = 0.05                        # upper_limit
            row[9] = -0.05                        # lower_limit
            positions.append(pos)
        grid.append(row)
    pd.DataFrame(grid).to_excel(writer, sheet_name=sheet_name, header=False, index=False)
    return positions


def test_untrimmed_only_track_computes_start_row_from_the_first_pass_sheet(tmp_path):
    """Construct the gap directly: sheets "SEC1 TRK1 0/1/2" with 1/2 carrying
    no TRM suffix at all, so the track is routed through
    _extract_untrimmed_only_track even though it has 2 real pass sheets.

    The untrimmed sheet ("...0") is given ONE EXTRA meta row the pass sheets
    don't have, so its start_row (2) genuinely differs from the pass
    sheets' own (1). Reusing the untrimmed sheet's start_row for the pass
    sheets -- the pre-fix behaviour -- would read pass 1 one row late,
    silently dropping its first real point (5 -> 4). The fix must recompute
    start_row from the first pass sheet itself and recover all 5.
    """
    path = tmp_path / "9999_1_TEST.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _write_min_system_a_sheet(writer, "SEC1 TRK1 0", n_meta_rows=2, n_data_rows=5)
        pass1_positions = _write_min_system_a_sheet(writer, "SEC1 TRK1 1", n_meta_rows=1, n_data_rows=5)
        _write_min_system_a_sheet(writer, "SEC1 TRK1 2", n_meta_rows=1, n_data_rows=5)

    result = ExcelParser().parse_file(path)
    track = result["tracks"][0]
    # Confirms the gap is real, not a stale assumption about the fixture.
    assert track["is_untrimmed_only"] is True
    assert track["trim_pass_count"] == 2

    passes = track["trim_passes"]
    assert len(passes) == 2, "the fix must still find both pass sheets"
    first = passes[0]
    assert len(first["positions"]) == 5, (
        "start_row must be computed from the pass sheet itself, not "
        "inherited from the untrimmed sheet's different (longer-header) layout"
    )
    assert first["positions"][0] == pytest.approx(pass1_positions[0])
    assert first["positions"] == pytest.approx(pass1_positions)


def test_read_trim_passes_gives_up_when_no_start_row_is_determinable(tmp_path):
    """If start_row is None (the untrimmed-only path) and even the first
    pass sheet has no findable position data, _read_trim_passes must return
    no passes rather than guess one (e.g. 0) -- a missing field is
    recoverable, a silently wrong one is not."""
    path = tmp_path / "9999_2_TEST.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # No numeric data anywhere in the position column (col 7) -->
        # _find_data_start can't locate a start row on this sheet.
        pd.DataFrame([["META"] * 10] * 5).to_excel(
            writer, sheet_name="SEC1 TRK1 1 TRM1", header=False, index=False)

    xl = pd.ExcelFile(path)
    assert ExcelParser()._first_pass_start_row(xl, SystemType.A, "SEC1 TRK1 1 TRM1") is None

    out = ExcelParser()._read_trim_passes(xl, SystemType.A, "TRK1", None, recipes=[])
    assert out == []


def test_first_pass_start_row_matches_the_known_offset_on_a_real_fixture():
    """Sanity check the new helper against a real, already-verified sheet:
    SEC1 TRK1 2 TRM2 of the DLTS fixture is read at start_row=1 throughout
    this file (see test_read_pass_returns_a_full_sweep above); the helper
    must independently arrive at the same answer."""
    xl = pd.ExcelFile(DLTS)
    assert ExcelParser()._first_pass_start_row(xl, SystemType.A, "SEC1 TRK1 2 TRM2") == 1


def test_a_numpy_integer_column_is_not_silently_dropped():
    """np.int64 is neither int nor float, so `isinstance(v, (int, float))`
    read every cell of an int64-typed column as blank and the column
    vanished with no error. Real instance: `8340_147_TA_Test Data_...xls`,
    `Lin Error`, 61 integer-zero cells. Rare only because a header string
    usually forces object dtype -- a model writing whole-number cut lengths
    would lose the lot.
    """
    import numpy as np
    import pandas as pd
    from laser_trim_analyzer.core.trim_passes import _col
    df = pd.DataFrame({0: np.array([0, 1, 2, 3], dtype=np.int64)})
    assert df.dtypes[0] == np.int64
    assert _col(df, 0, 0) == [0.0, 1.0, 2.0, 3.0]


def test_blanks_and_booleans_are_still_blanks():
    """The widened numeric test must not start accepting non-measurements.
    bool is an Integral, so True would read as 1.0 without the explicit guard.
    """
    import numpy as np
    import pandas as pd
    from laser_trim_analyzer.core.trim_passes import _col
    df = pd.DataFrame({0: [1.5, True, None, float("nan"), "x", np.float64(2.5)]})
    assert _col(df, 0, 0) == [1.5, None, None, None, None, 2.5]
