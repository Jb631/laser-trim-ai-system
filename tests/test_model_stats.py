"""INVESTIGATE stats table — the numbers that replace the Excel round-trip.

Fixture style follows tests/test_spc_db.py: a real DatabaseManager on a tmp
DB, seeded with the exact shapes the real data has (a 1e12 Ω open-circuit
reading, a negative electrical angle, NULL metrics, a WARNING unit that is
accepted product and a FAIL unit that is not).
"""
from datetime import datetime, timedelta

import pytest

from laser_trim_analyzer.core.model_stats import (
    POSITIVE_RATIO, RATIO_BAND, compute_model_stats, metric_policy)
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import (
    AnalysisResult, StatusType, SystemType, TrackResult)

D0 = datetime(2026, 1, 5)


@pytest.fixture()
def db(tmp_path):
    return DatabaseManager(tmp_path / "t.db")


_SEQ = [0]


def _add(db, model, day, status, **track):
    """One unit with one track. `track` overrides any TrackResult column.

    Serials are sequenced because (filename, file_date, model, serial) is
    UNIQUE — and because a repeated serial is VALID in this domain (a unit
    trimmed twice), so the fixture must not lean on serial to distinguish rows.
    """
    _SEQ[0] += 1
    with db.session() as s:
        ar = AnalysisResult(
            model=model, serial=f"{model}-{_SEQ[0]}",
            system=SystemType.A, filename=f"{model}_{_SEQ[0]}_{day:%m-%d-%Y}.xls",
            file_date=day, overall_status=status)
        s.add(ar)
        s.flush()
        cols = {"track_id": "default", "status": status}
        cols.update(track)
        s.add(TrackResult(analysis_id=ar.id, **cols))


def _row(stats, key):
    return next(r for r in stats.rows if r.key == key)


# ---------------------------------------------------------------------------
# D2 — the ALL | LIN-PASSING split
# ---------------------------------------------------------------------------

def test_warning_unit_is_lin_passing_and_fail_unit_is_not(db):
    """WARNING is the internal sigma watch on ACCEPTED product; FAIL is the
    customer linearity rejection. The split has to follow the disposition."""
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=1000.0)
    _add(db, "M", D0, StatusType.WARNING, untrimmed_resistance=1100.0)
    _add(db, "M", D0, StatusType.FAIL, untrimmed_resistance=9000.0)
    st = compute_model_stats(db, "M")
    r = _row(st, "untrimmed_resistance")
    assert r.all_.n == 3 and r.lin_passing.n == 2
    assert r.lin_passing.high == 1100.0          # the FAIL unit is not in here
    assert r.all_.high == 9000.0
    assert r.lin_passing.avg == pytest.approx(1050.0)


def test_untrimmed_and_error_rows_are_in_all_but_not_lin_passing(db):
    """ALL is every track row carrying a usable value, whatever the status."""
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=1000.0)
    _add(db, "M", D0, StatusType.UNTRIMMED, untrimmed_resistance=1200.0)
    _add(db, "M", D0, StatusType.ERROR, untrimmed_resistance=1400.0)
    st = compute_model_stats(db, "M")
    r = _row(st, "untrimmed_resistance")
    assert r.all_.n == 3 and r.lin_passing.n == 1


# ---------------------------------------------------------------------------
# D3 — plausibility filtering (the 6607 case, in miniature)
# ---------------------------------------------------------------------------

def test_open_circuit_reading_excluded_count_disclosed_average_unmoved(db):
    """A 1e12 Ω reading must not move the average, and must be COUNTED.

    This is 6607 in miniature: on the real database seven readings of that
    class out of 9,943 drag the raw average from 4,282 Ω to 32,079 Ω and the
    max from 29.6 kΩ to 2.1e8 Ω. Filtering it silently would be the same lie
    told quietly, so the cell has to carry the count it dropped.
    """
    for v in (4000.0, 4200.0, 4400.0):
        _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=v)
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=1e12)
    st = compute_model_stats(db, "M")
    r = _row(st, "untrimmed_resistance")
    assert r.all_.n == 3
    assert r.all_.avg == pytest.approx(4200.0)
    assert r.all_.high == 4400.0
    assert r.all_.excluded == 1                  # disclosed, never hidden
    assert st.excluded_total >= 1


def test_zero_and_negative_resistance_are_excluded_not_averaged(db):
    """0 Ω is not a resistance reading — it is a reading that did not happen."""
    for v in (1000.0, 1100.0, 1200.0):
        _add(db, "M", D0, StatusType.PASS, trimmed_resistance=v)
    _add(db, "M", D0, StatusType.FAIL, trimmed_resistance=0.0)
    st = compute_model_stats(db, "M")
    r = _row(st, "trimmed_resistance")
    assert r.all_.n == 3 and r.all_.excluded == 1
    assert r.all_.low == 1000.0


def test_negative_electrical_angle_is_kept(db):
    """`measured_electrical_angle` has 2,416 legitimate negatives in the real
    database. The ratio rule must NOT reach it — this guards the policy table
    against being over-applied to every numeric column."""
    assert metric_policy("measured_electrical_angle") != POSITIVE_RATIO
    _add(db, "M", D0, StatusType.PASS, measured_electrical_angle=-0.05)
    _add(db, "M", D0, StatusType.PASS, measured_electrical_angle=0.55)
    _add(db, "M", D0, StatusType.PASS, measured_electrical_angle=0.60)
    st = compute_model_stats(db, "M")
    r = _row(st, "measured_electrical_angle")
    assert r.all_.n == 3 and r.all_.excluded == 0
    assert r.all_.low == -0.05


def test_ratio_band_is_the_model_median_hundredfold(db):
    """The band is [median/100, median*100] over the model's own positives."""
    for v in (100.0, 100.0, 100.0):
        _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=v)
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=100.0 * RATIO_BAND)
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=100.0 / RATIO_BAND)
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=100.0 * RATIO_BAND * 1.01)
    st = compute_model_stats(db, "M")
    r = _row(st, "untrimmed_resistance")
    assert r.all_.n == 5                     # the two band EDGES are kept
    assert r.all_.excluded == 1              # only the one past the edge


def test_lin_passing_uses_the_same_band_as_all(db):
    """One band per (model, metric, window). A reading the ALL column calls
    corrupt cannot be quietly kept in the LIN-PASSING column."""
    for v in (4000.0, 4200.0, 4400.0):
        _add(db, "M", D0, StatusType.WARNING, untrimmed_resistance=v)
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=1e12)
    st = compute_model_stats(db, "M")
    r = _row(st, "untrimmed_resistance")
    assert r.lin_passing.n == 3 and r.lin_passing.excluded == 1
    assert r.lin_passing.high == 4400.0


# ---------------------------------------------------------------------------
# D4 — per-cell NULL exclusion with the surviving n visible
# ---------------------------------------------------------------------------

def test_null_metric_excluded_per_cell_with_visible_n(db):
    """NULL is never imputed and never silently zero: the cell reports the n
    it actually had and how many rows had nothing recorded."""
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=1000.0,
         sigma_gradient=0.01)
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=None,
         sigma_gradient=0.02)
    st = compute_model_stats(db, "M")
    res = _row(st, "untrimmed_resistance")
    sig = _row(st, "sigma_gradient")
    assert res.all_.n == 1 and res.all_.missing == 1
    assert res.all_.avg == pytest.approx(1000.0)     # not 500.0
    assert sig.all_.n == 2 and sig.all_.missing == 0


def test_empty_cell_reports_nothing_rather_than_zero(db):
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=None)
    st = compute_model_stats(db, "M")
    r = _row(st, "untrimmed_resistance")
    assert r.all_.n == 0
    assert r.all_.avg is None and r.all_.low is None and r.all_.high is None


# ---------------------------------------------------------------------------
# D4 — the two rate rows
# ---------------------------------------------------------------------------

def test_trim_pass_rate_row(db):
    _add(db, "M", D0, StatusType.PASS, linearity_pass=True)
    _add(db, "M", D0, StatusType.WARNING, linearity_pass=True)
    _add(db, "M", D0, StatusType.FAIL, linearity_pass=False)
    _add(db, "M", D0, StatusType.ERROR, linearity_pass=None)
    st = compute_model_stats(db, "M")
    r = _row(st, "trim_passed_linearity")
    assert r.kind == "rate"
    assert r.all_.n == 3 and r.all_.count == 2
    assert r.all_.pct == pytest.approx(200.0 / 3.0)
    assert r.all_.missing == 1
    assert r.lin_passing.n == 2 and r.lin_passing.count == 2
    assert r.lin_passing.pct == pytest.approx(100.0)


def test_already_met_spec_before_trim_rate_row(db):
    """untrimmed_error_max <= linearity_spec, both non-NULL (see the module
    docstring: this definition is an inference from the column set)."""
    _add(db, "M", D0, StatusType.PASS, untrimmed_error_max=0.1, linearity_spec=0.5)
    _add(db, "M", D0, StatusType.PASS, untrimmed_error_max=0.5, linearity_spec=0.5)
    _add(db, "M", D0, StatusType.FAIL, untrimmed_error_max=0.9, linearity_spec=0.5)
    _add(db, "M", D0, StatusType.PASS, untrimmed_error_max=None, linearity_spec=0.5)
    _add(db, "M", D0, StatusType.PASS, untrimmed_error_max=0.2, linearity_spec=None)
    st = compute_model_stats(db, "M")
    r = _row(st, "already_met_spec")
    assert r.all_.n == 3 and r.all_.count == 2        # <= is inclusive
    assert r.all_.missing == 2
    assert r.all_.pct == pytest.approx(200.0 / 3.0)
    assert r.lin_passing.n == 2 and r.lin_passing.count == 2


# ---------------------------------------------------------------------------
# window / lot edges
# ---------------------------------------------------------------------------

def test_cutoff_edge_is_inclusive(db):
    _add(db, "M", D0 - timedelta(days=1), StatusType.PASS, untrimmed_resistance=1000.0)
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=1100.0)
    _add(db, "M", D0 + timedelta(days=1), StatusType.PASS, untrimmed_resistance=1200.0)
    st = compute_model_stats(db, "M", cutoff=D0)
    r = _row(st, "untrimmed_resistance")
    assert r.all_.n == 2 and r.all_.low == 1100.0


def test_lot_window_includes_both_end_days(db):
    _add(db, "M", D0 - timedelta(days=1), StatusType.PASS, untrimmed_resistance=900.0)
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=1000.0)
    _add(db, "M", D0 + timedelta(days=2), StatusType.PASS, untrimmed_resistance=1100.0)
    _add(db, "M", D0 + timedelta(days=3), StatusType.PASS, untrimmed_resistance=1200.0)
    st = compute_model_stats(db, "M", lot=(D0, D0 + timedelta(days=2)))
    r = _row(st, "untrimmed_resistance")
    assert r.all_.n == 2 and r.all_.low == 1000.0 and r.all_.high == 1100.0


def test_window_narrows_the_plausibility_band_population(db):
    """The band's median comes from the values IN THE WINDOW — a model that
    changed scale between windows must not be judged against the other one."""
    _add(db, "M", D0 - timedelta(days=10), StatusType.PASS, untrimmed_resistance=1.0)
    _add(db, "M", D0 - timedelta(days=10), StatusType.PASS, untrimmed_resistance=1.0)
    for v in (4000.0, 4200.0, 4400.0):
        _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=v)
    st = compute_model_stats(db, "M", cutoff=D0)
    assert _row(st, "untrimmed_resistance").all_.n == 3


# ---------------------------------------------------------------------------
# honest degradation
# ---------------------------------------------------------------------------

def test_unknown_model_degrades_honestly(db):
    st = compute_model_stats(db, "NOPE")
    assert st.tracks == 0
    assert all(r.all_.n == 0 for r in st.rows)
    assert "no" in st.note.lower()


def test_empty_database_does_not_raise(db):
    st = compute_model_stats(db, "")
    assert st.tracks == 0 and st.rows


def test_unreadable_database_degrades_instead_of_raising():
    """Never raises into the GUI (the sibling rule from core/spec_alignment)."""
    class Boom:
        def session(self):
            raise RuntimeError("database is locked")
    st = compute_model_stats(Boom(), "M")
    assert st.tracks == 0 and st.rows and st.note


def test_rows_are_ordered_and_labelled(db):
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=1000.0)
    st = compute_model_stats(db, "M")
    assert [r.key for r in st.rows] == [
        "untrimmed_resistance", "trimmed_resistance", "measured_electrical_angle",
        "final_linearity_error_shifted", "margin_to_spec", "sigma_gradient",
        "trim_passed_linearity", "already_met_spec"]
    assert all(r.label and r.kind in ("distribution", "rate") for r in st.rows)
    assert [r.key for r in st.distribution_rows] == [r.key for r in st.rows[:6]]
    assert [r.key for r in st.rate_rows] == [r.key for r in st.rows[6:]]
