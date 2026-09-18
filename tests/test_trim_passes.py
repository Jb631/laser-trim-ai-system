import pandas as pd
import pytest
from laser_trim_analyzer.core.models import SystemType
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
