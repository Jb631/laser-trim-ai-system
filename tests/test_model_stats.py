"""INVESTIGATE stats table — the numbers that replace the Excel round-trip.

Fixture style follows tests/test_spc_db.py: a real DatabaseManager on a tmp
DB, seeded with the exact shapes the real data has (a 1e12 Ω open-circuit
reading, a negative electrical angle, NULL metrics, a WARNING unit that is
accepted product and a FAIL unit that is not).
"""
from datetime import datetime, timedelta

import pytest

from laser_trim_analyzer.core.model_stats import (
    POSITIVE_RATIO, RATIO_BAND, Cell, StatRow, compute_model_stats,
    metric_policy)
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


def test_untrimmed_rows_are_in_all_but_not_lin_passing(db):
    """ALL is every track row carrying a usable value, whatever the
    DISPOSITION — an untrimmed test sweep really did measure the part."""
    _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=1000.0)
    _add(db, "M", D0, StatusType.UNTRIMMED, untrimmed_resistance=1200.0)
    _add(db, "M", D0, StatusType.FAIL, untrimmed_resistance=1400.0)
    st = compute_model_stats(db, "M")
    r = _row(st, "untrimmed_resistance")
    assert r.all_.n == 3 and r.lin_passing.n == 1


def test_a_record_that_failed_processing_is_dropped_and_disclosed(db):
    """The 8856 case: all 94 ERROR records carry sigma_gradient = 999.999, the
    analyser's saturation sentinel, and averaging them in put 8856's sigma at
    433.5 against a true 0.0012. No value policy catches that — a 100x band
    around a 0.001 median would delete 8340-1's real 1.x values on FAILED
    units. The record's own status is what says these are not measurements."""
    for v in (0.001, 0.0012, 0.0014):
        _add(db, "M", D0, StatusType.PASS, sigma_gradient=v)
    _add(db, "M", D0, StatusType.ERROR, sigma_gradient=999.999)
    st = compute_model_stats(db, "M")
    r = _row(st, "sigma_gradient")
    assert r.all_.n == 3
    assert r.all_.avg == pytest.approx(0.0012)   # not 250.0
    assert r.all_.high == pytest.approx(0.0014)
    assert r.all_.errored == 1                   # disclosed, never hidden
    assert r.lin_passing.errored == 0            # never in that population
    assert st.errored == 1 and st.tracks == 3
    assert "processing failed" in st.note


def test_a_failed_record_is_dropped_from_every_row_including_the_rates(db):
    _add(db, "M", D0, StatusType.PASS, linearity_pass=True,
         untrimmed_error_max=0.1, linearity_spec=0.5)
    _add(db, "M", D0, StatusType.ERROR, linearity_pass=False,
         untrimmed_error_max=0.1, linearity_spec=0.5)
    st = compute_model_stats(db, "M")
    rate = _row(st, "trim_passed_linearity")
    assert rate.all_.n == 1 and rate.all_.count == 1 and rate.all_.pct == 100.0
    assert rate.all_.errored == 1
    assert _row(st, "already_met_spec").all_.n == 1


def test_a_failed_record_cannot_stretch_the_plausibility_band(db):
    """The band's median is taken AFTER the failed records are removed."""
    for v in (1000.0, 1100.0, 1200.0):
        _add(db, "M", D0, StatusType.PASS, untrimmed_resistance=v)
    _add(db, "M", D0, StatusType.ERROR, untrimmed_resistance=1e12)
    st = compute_model_stats(db, "M")
    r = _row(st, "untrimmed_resistance")
    assert r.all_.n == 3 and r.all_.errored == 1 and r.all_.excluded == 0


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
    # A track that recorded no verdict at all — `missing`, never a 0 in the
    # denominator ("10% of what we measured" is not "10% of what we ran").
    _add(db, "M", D0, StatusType.UNTRIMMED, linearity_pass=None)
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


# ---------------------------------------------------------------------------
# display formatting — one scale per row, kohms where that reads better
# ---------------------------------------------------------------------------

def test_ohms_read_as_kilohms_above_a_thousand():
    from laser_trim_analyzer.core.model_stats import display_unit, format_in
    assert display_unit("ohms", 4281.8) == "kΩ"
    assert display_unit("ohms", 929.0) == "Ω"
    assert format_in(4281.8, "kΩ") == "4.28 kΩ"
    assert format_in(29576.0, "kΩ") == "29.6 kΩ"
    assert format_in(422.0, "Ω") == "422 Ω"


def test_a_row_uses_one_scale_for_all_its_cells():
    """min 422 Ω next to max 29.6 kΩ is unreadable — the ROW picks the scale."""
    from laser_trim_analyzer.core.model_stats import row_unit
    row = StatRow(key="untrimmed_resistance", label="R", unit="ohms",
                  kind="distribution",
                  all_=Cell(n=3, excluded=0, missing=0, avg=4281.8,
                            low=422.0, high=29576.0),
                  lin_passing=Cell(n=0, excluded=0, missing=0))
    assert row_unit(row) == "kΩ"


def test_other_units_and_missing_values():
    from laser_trim_analyzer.core.model_stats import format_in
    assert format_in(None, "kΩ") == "—"
    assert format_in(46.789, "°") == "46.8°"
    assert format_in(13.5786, "%") == "13.6%"
    assert format_in(0.0056266, "") == "0.005627"


# ---------------------------------------------------------------------------
# lot-vs-history verdicts (the shared SPC core writes "normal")
# ---------------------------------------------------------------------------

def _seed_lots(db, model, n_lots=12, value=1000.0, last_value=None, spacing=7):
    """One lot a week, `n_lots` of them, 6 units each. Lot boundaries come from
    ml/lots.py (a gap > LOT_GAP_DAYS starts a new lot), so weekly spacing gives
    one lot per day-cluster.

    Lots jitter by 0-2 ohms with no trend: the SPC core refuses to judge a
    baseline with ZERO lot-to-lot spread (±3*0 would flag every lot that is not
    exactly the centre), so a fixture of identical lots would test nothing.
    """
    for k in range(n_lots):
        v = (value + k % 3) if (k < n_lots - 1 or last_value is None) else last_value
        day = D0 + timedelta(days=spacing * k)
        for i in range(6):
            _add(db, model, day, StatusType.PASS,
                 untrimmed_resistance=v + i, linearity_pass=True)
    return D0 + timedelta(days=spacing * (n_lots - 1))


def test_lot_within_its_normal(db):
    from laser_trim_analyzer.core.model_stats import compute_lot_verdicts
    last = _seed_lots(db, "M", value=1000.0)
    v = compute_lot_verdicts(db, "M", (last, last))["untrimmed_resistance"]
    assert v.status == "within"
    assert "within its normal" in v.text


def test_lot_above_everything_this_model_has_done(db):
    from laser_trim_analyzer.core.model_stats import compute_lot_verdicts
    # Lot medians wobble by 1 ohm between lots, so a 500-ohm jump is far
    # outside the model's own lot-to-lot spread.
    last = _seed_lots(db, "M", value=1000.0, last_value=1500.0)
    v = compute_lot_verdicts(db, "M", (last, last))["untrimmed_resistance"]
    assert v.status == "above"
    assert "above everything this model has done" in v.text
    assert "typically 1.50 kΩ" in v.text     # this lot's value, in the row's unit
    assert "kΩ–" not in v.text               # the band names its unit once
    assert v.normal_low is not None and v.normal_high is not None


def test_lot_below_everything_this_model_has_done(db):
    from laser_trim_analyzer.core.model_stats import compute_lot_verdicts
    last = _seed_lots(db, "M", value=1000.0, last_value=500.0)
    v = compute_lot_verdicts(db, "M", (last, last))["untrimmed_resistance"]
    assert v.status == "below"
    assert "below anything this model has done" in v.text


def test_short_history_says_so_instead_of_inventing_a_band(db):
    """Below MIN_LOTS_TRAIN the SPC core refuses to judge — and so do we."""
    from laser_trim_analyzer.core.model_stats import compute_lot_verdicts
    last = _seed_lots(db, "M", n_lots=3, value=1000.0)
    v = compute_lot_verdicts(db, "M", (last, last))["untrimmed_resistance"]
    assert v.status == "unjudged"
    assert "not run enough lots" in v.text
    assert v.normal_low is None


def test_metric_with_no_readings_in_the_lot_says_so(db):
    from laser_trim_analyzer.core.model_stats import compute_lot_verdicts
    last = _seed_lots(db, "M", value=1000.0)
    v = compute_lot_verdicts(db, "M", (last, last))["margin_to_spec"]
    assert v.status == "no data"
    assert v.lot_typical is None


def test_lot_verdicts_never_raise_on_a_dead_database():
    from laser_trim_analyzer.core.model_stats import compute_lot_verdicts
    class Boom:
        def session(self):
            raise RuntimeError("database is locked")
    out = compute_lot_verdicts(Boom(), "M", (D0, D0))
    assert out == {}


# ---------------------------------------------------------------------------
# the lot selector's choices (the SHARED clustering, never a second one)
# ---------------------------------------------------------------------------

def test_model_lots_are_newest_first_and_labelled(db):
    from laser_trim_analyzer.core.model_stats import model_lots
    last = _seed_lots(db, "M", n_lots=4)
    lots = model_lots(db, "M")
    assert len(lots) == 4
    assert lots[0].end == last                       # newest first
    assert lots[0].n == 6 and lots[0].label
    assert all(lots[i].end > lots[i + 1].end for i in range(len(lots) - 1))


def test_open_lot_is_marked_and_is_the_default(db):
    from laser_trim_analyzer.core.model_stats import default_lot_index, model_lots
    _seed_lots(db, "M", n_lots=4)
    lots = model_lots(db, "M")
    # The newest lot ends on the DB's own newest date, so it is still open.
    assert lots[0].is_open and "current lot" in lots[0].label
    assert default_lot_index(lots) == 0


def test_no_open_lot_defaults_to_all_history(db):
    from laser_trim_analyzer.core.model_stats import default_lot_index, model_lots
    _seed_lots(db, "M", n_lots=4)
    # A newer model moves the DB's clock past M's last lot, closing it. The
    # date stays in the PAST: the app's anchor throws out file-date junk more
    # than a day ahead of the wall clock, so a "future" row would move nothing.
    _add(db, "OTHER", D0 + timedelta(days=120), StatusType.PASS,
         untrimmed_resistance=1000.0)
    lots = model_lots(db, "M")
    assert not lots[0].is_open
    assert default_lot_index(lots) is None           # None = all history


def test_model_lots_on_an_unreadable_database():
    from laser_trim_analyzer.core.model_stats import model_lots
    class Boom:
        def session(self):
            raise RuntimeError("database is locked")
    assert model_lots(Boom(), "M") == []


def test_a_resistance_band_never_prints_a_negative_normal(db):
    """±3σ on lot medians dips below zero when the spread is wide; a
    resistance "normal -1.02 kΩ" is not something an engineer can read.
    The SENTENCE clamps, the comparison does not (evidence.py's precedent)."""
    from laser_trim_analyzer.core.model_stats import compute_lot_verdicts
    # Lots alternating 1 kohm / 3 kohm: a lot-to-lot spread that wide next to
    # the centre puts the lower 3-sigma limit below zero.
    for k in range(12):
        day = D0 + timedelta(days=7 * k)
        for i in range(6):
            _add(db, "M", day, StatusType.PASS,
                 untrimmed_resistance=(3000.0 if k % 2 else 1000.0) + i)
    last = D0 + timedelta(days=7 * 11)
    v = compute_lot_verdicts(db, "M", (last, last))["untrimmed_resistance"]
    assert v.normal_low is not None and v.normal_low < 0    # math untouched
    assert "normal under" in v.text and "-" not in v.text.split("normal")[-1]


def test_electrical_angle_keeps_its_negative_band(db):
    """The one metric that IS legitimately negative must keep both bounds."""
    from laser_trim_analyzer.core.model_stats import compute_lot_verdicts
    for k in range(12):
        day = D0 + timedelta(days=7 * k)
        for i in range(6):
            _add(db, "M", day, StatusType.PASS,
                 measured_electrical_angle=(2.0 if k % 2 else 0.5) + 0.01 * i)
    last = D0 + timedelta(days=7 * 11)
    v = compute_lot_verdicts(db, "M", (last, last))["measured_electrical_angle"]
    assert v.normal_low is not None and v.normal_low < 0
    assert "normal under" not in v.text and "–" in v.text


def test_lot_verdicts_ignore_records_that_failed_processing(db):
    """8856's 75 failed records carry 999.999 in the linearity column too — a
    lot median built from those would produce a confident, wrong sentence."""
    from laser_trim_analyzer.core.model_stats import compute_lot_verdicts
    for k in range(12):
        day = D0 + timedelta(days=7 * k)
        for i in range(6):
            _add(db, "M", day, StatusType.PASS,
                 final_linearity_error_shifted=0.01 + 0.001 * (k % 3) + 0.0001 * i)
        # Two sentinel rows in every lot: enough to own the median if kept.
        for _ in range(2):
            _add(db, "M", day, StatusType.ERROR,
                 final_linearity_error_shifted=999.999)
    last = D0 + timedelta(days=7 * 11)
    v = compute_lot_verdicts(db, "M", (last, last))["final_linearity_error_shifted"]
    assert v.lot_typical is not None and v.lot_typical < 1.0
    assert v.lot_n == 6                       # the sentinels are not readings
    assert "999" not in v.text
