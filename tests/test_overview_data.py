"""The Overview's one loader -- gui/v6/overview_data.load_overview (Graphite redesign, 2026-10-02;
option B, 2026-10-04).

James: "keep all 16 cards (fail rate up, or a signal moved), each with its reason". The cards are
the union of the "Drifting now" fail-rate list (ml/spc.compute_focus_list) and the drift
detector's flags (ml/manager.get_drifting_models); "Everything else" is every other active model;
the inactive models are counted, never hidden. Pass % is the app's headline yield, with suspect
files left out like the drift watch and the FOCUS list leave them out. Option B adds each model's
lasers and the dollars lost at final test (the Company trends formula, over this page's own 90
days), and the one header line that says them.

Every database here is a tmp file; every model, serial, date and PRICE is INVENTED.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from laser_trim_analyzer.gui.v6 import overview_data as od
from laser_trim_analyzer.ml.drift_types import (
    AlertType, DriftTier, MetricStatus, ModelAlertSummary, ModelDriftStatus)
from laser_trim_analyzer.ml.spc import FocusResult

ANCHOR = datetime(2026, 3, 20, 15, 30)       # the newest trim file in most tests


def _db(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / "overview.db")


_serial = iter(range(1, 10 ** 7))


def _trims(db, model, when, *, passes=0, warnings=0, fails=0, errors=0, failed_processing=0,
           untrimmed=0, suspect_fails=0, suspect_passes=0):
    """`model`'s trim files dated `when`, so many of each status. The suspect ones are files the
    processor marked data_quality='suspect'."""
    from laser_trim_analyzer.database.models import AnalysisResult, StatusType, SystemType
    groups = (("PASS", passes, "good"), ("WARNING", warnings, "good"), ("FAIL", fails, "good"),
              ("ERROR", errors, "good"), ("PROCESSING_FAILED", failed_processing, "good"),
              ("UNTRIMMED", untrimmed, "good"), ("FAIL", suspect_fails, "suspect"),
              ("PASS", suspect_passes, "suspect"))
    with db.session() as s:
        for status, count, quality in groups:
            for _ in range(count):
                n = next(_serial)
                s.add(AnalysisResult(model=model, serial=f"{model}-{n}", system=SystemType.A,
                                     filename=f"{model}_{n}.xls", file_date=when,
                                     overall_status=StatusType[status], data_quality=quality))


def _final_tests(db, model, when, *, passes=0, fails=0):
    from laser_trim_analyzer.database.models import FinalTestResult, StatusType
    with db.session() as s:
        for status, count in (("PASS", passes), ("FAIL", fails)):
            for _ in range(count):
                n = next(_serial)
                s.add(FinalTestResult(filename=f"FT_{model}_{n}.xls", model=model,
                                      serial=f"{model}-ft-{n}", file_date=when,
                                      overall_status=StatusType[status]))


def _entry(model, p_base, p_recent):
    """A FOCUS row as the loader reads one: the model and its two fail rates."""
    return SimpleNamespace(model=model, p_base=p_base, p_recent=p_recent)


def _flag(model, tier, magnitude, metric="untrimmed_error_max"):
    return ModelAlertSummary(model=model, tier=tier, alert_type=AlertType.SLOW_DRIFT,
                             worst_metric=metric, magnitude=magnitude)


def _status(model, metric, baseline, recent, tier=DriftTier.DRIFT):
    ms = MetricStatus(metric=metric, tier=tier, alert_type=AlertType.SLOW_DRIFT, magnitude=1.0,
                      baseline_mean=baseline, baseline_std=0.01, recent_mean=recent,
                      recent_count=5, is_trained=True)
    return ModelDriftStatus(model=model, overall_tier=tier, worst_metric=metric,
                            worst_alert_type=AlertType.SLOW_DRIFT, per_metric={metric: ms})


def _fake_sources(monkeypatch, *, focus=(), flags=(), statuses=None):
    """Stand in for the two card sources: the FOCUS list (through the one focus loader) and the
    drift detector (its flags, and each flagged model's per-metric status)."""
    statuses = statuses or {}
    monkeypatch.setattr(od, "load_focus", lambda db, *a, **k: (
        FocusResult(focus=list(focus), chronic=[], anchor=None), None))
    monkeypatch.setattr(od, "get_drifting_models", lambda db, *a, **k: list(flags))
    monkeypatch.setattr(od, "get_model_drift_status",
                        lambda db, model, reference_date=None: statuses[model])


# ---- which models get a card, in which order, and why ------------------------------------------

def test_the_cards_are_the_fail_rate_list_then_the_detectors_other_flags_by_tier(tmp_path,
                                                                                monkeypatch):
    db = _db(tmp_path)
    _trims(db, "ANY", ANCHOR, passes=1)
    _fake_sources(
        monkeypatch,
        focus=[_entry("F1", 0.05, 0.30), _entry("F2", 0.10, 0.40)],
        # handed over out of order on purpose: the loader orders the detector's own models
        flags=[_flag("D2", DriftTier.WARNING, 3.0), _flag("F2", DriftTier.DRIFT, 1.5),
               _flag("D3", DriftTier.OUT_OF_CONTROL, 1.1), _flag("D1", DriftTier.OUT_OF_CONTROL, 2.0)],
        statuses={m: _status(m, "untrimmed_error_max", 0.1, 0.2) for m in ("D1", "D2", "D3", "F2")})
    ov = od.load_overview(db)
    assert [c.model for c in ov.cards] == ["F1", "F2", "D1", "D3", "D2"]
    assert od.card_count(ov) == 5
    assert ov.failed == {}


def test_each_card_says_why_it_is_there_and_both_lists_join_with_a_dot(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _trims(db, "ANY", ANCHOR, passes=1)
    _fake_sources(
        monkeypatch, focus=[_entry("F1", 0.04, 0.12), _entry("BOTH", 0.10, 0.40)],
        flags=[_flag("BOTH", DriftTier.DRIFT, 1.5),
               _flag("RES", DriftTier.OUT_OF_CONTROL, 2.0, metric="untrimmed_resistance"),
               _flag("FT", DriftTier.WARNING, 1.0, metric="ft_fail_fraction")],
        statuses={"BOTH": _status("BOTH", "untrimmed_error_max", 0.1, 0.2),
                  "RES": _status("RES", "untrimmed_resistance", 4693.04, 5896.67),
                  "FT": _status("FT", "ft_fail_fraction", 0.025, 0.1875)})
    reasons = {c.model: c.reason for c in od.load_overview(db).cards}
    assert reasons == {
        "F1": "Fail rate 4% → 12%",
        "BOTH": "Fail rate 10% → 40% · Untrimmed error (max) 0.1 → 0.2",
        "RES": "Untrimmed resistance 4,693 → 5,897",       # the theme's fmt_measure
        "FT": "Final-test lot fail rate 2.5% → 18.8%",     # a fraction reads as a percent
    }


def test_a_signal_that_moved_on_a_model_still_passing_says_so(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _trims(db, "CLEAN", ANCHOR, passes=200)                 # 100% over the 90 days
    _trims(db, "SLIP", ANCHOR, passes=98, fails=2)          # 98%: not "still passing"
    _trims(db, "EDGE", ANCHOR, passes=99, fails=1)          # 99%: the line, inclusive
    _fake_sources(monkeypatch, flags=[_flag(m, DriftTier.DRIFT, 1.0) for m in ("CLEAN", "SLIP", "EDGE")],
                  statuses={m: _status(m, "untrimmed_error_max", 0.1, 0.2)
                            for m in ("CLEAN", "SLIP", "EDGE")})
    reasons = {c.model: c.reason for c in od.load_overview(db).cards}
    assert reasons["CLEAN"] == "Untrimmed error (max) 0.1 → 0.2 · still passing"
    assert reasons["EDGE"].endswith(" · still passing")
    assert reasons["SLIP"] == "Untrimmed error (max) 0.1 → 0.2"


def test_a_real_fail_rate_excursion_is_a_card_with_its_rates(tmp_path):
    """Through the real compute_focus_list: weekly runs of 20 at 10% failing, then one at 60%."""
    db = _db(tmp_path)
    start = ANCHOR - timedelta(days=7 * 11)
    for k in range(11):
        _trims(db, "HOT", start + timedelta(days=7 * k), passes=18, fails=2)
    _trims(db, "HOT", ANCHOR, passes=8, fails=12)
    ov = od.load_overview(db)
    hot = [c for c in ov.cards if c.model == "HOT"]
    assert len(hot) == 1, [c.model for c in ov.cards]
    assert hot[0].reason == "Fail rate 10% → 60%"
    assert hot[0].units == 240 and hot[0].pass_pct == pytest.approx(100 * 206 / 240)


def test_a_real_detector_flag_is_a_card_with_its_signal(tmp_path):
    """Through real drift training (tests/test_drift_trust_rules.py's invented history): twelve
    ordinary runs, then two whose linearity error is far above them -- every unit still PASS."""
    from test_drift_trust_rules import _history, _lot, _train
    db = _db(tmp_path)
    day = _history(db, "M1")
    _lot(db, "M1", day, [0.5] * 5, tag="x")
    _lot(db, "M1", day + timedelta(days=7), [0.5] * 5, tag="y")
    _train(db)
    cards = {c.model: c for c in od.load_overview(db).cards}
    assert "M1" in cards, list(cards)
    assert cards["M1"].reason.startswith("Linearity error ")
    assert " → " in cards["M1"].reason and cards["M1"].reason.endswith(" · still passing")


# ---- pass %: the headline yield, suspect files out ----------------------------------------------

def test_pass_is_pass_and_warning_over_everything_graded(tmp_path):
    """(PASS + WARNING) / (PASS + WARNING + FAIL). Not graded: ERROR, PROCESSING_FAILED, UNTRIMMED.
    Left out: a file marked suspect, and a file dated more than a day ahead."""
    db = _db(tmp_path)
    _trims(db, "M", ANCHOR, passes=6, warnings=2, fails=2, errors=3, failed_processing=1,
           untrimmed=2, suspect_fails=5, suspect_passes=4)
    _trims(db, "M", datetime.now() + timedelta(days=30), passes=50)
    ov = od.load_overview(db)
    (row,) = [r for r in ov.others if r.model == "M"]
    assert (row.units, row.pass_pct) == (10, pytest.approx(80.0))
    assert ov.anchor == ANCHOR, "a future-dated file became the newest file"


def test_a_suspect_newest_file_is_not_the_anchor(tmp_path):
    db = _db(tmp_path)
    _trims(db, "M", ANCHOR, passes=3)
    _trims(db, "M", ANCHOR + timedelta(days=5), suspect_passes=2)
    assert od.load_overview(db).anchor == ANCHOR


def test_the_windows_are_ninety_days_the_year_before_and_twelve_months(tmp_path):
    db = _db(tmp_path)
    day = ANCHOR
    _trims(db, "W", day, passes=30)                              # in the 90 days (its last day)
    _trims(db, "W", day - timedelta(days=89), fails=10)          # in the 90 days (its first day)
    _trims(db, "W", day - timedelta(days=90), passes=10)         # the year before
    _trims(db, "W", day - timedelta(days=454), fails=30)         # the year before (its first day)
    _trims(db, "W", day - timedelta(days=455), passes=100)       # neither
    ov = od.load_overview(db)
    (row,) = [r for r in ov.others if r.model == "W"]
    assert row.units == 40 and row.pass_pct == pytest.approx(75.0)
    assert row.was_pct == pytest.approx(25.0)
    assert (row.trend, row.tone) == ("up 50 pts", "up")
    # Apr 2025 .. Mar 2026, oldest first: Dec 2025 holds -89 (10 FAIL) and -90 (10 PASS) days
    assert row.months == [None] * 8 + [pytest.approx(50.0), None, None, pytest.approx(100.0)]
    # ...and the header line names the clock: the newest counted file's day
    assert od.header_line(ov).endswith(" · newest file 20 Mar 2026")


def test_every_models_counts_come_from_one_grouped_query(tmp_path):
    from sqlalchemy import event
    db = _db(tmp_path)
    for i in range(6):
        _trims(db, f"M{i}", ANCHOR - timedelta(days=i), passes=3, fails=1)
    seen = []

    def spy(conn, cursor, statement, parameters, context, executemany):
        if "date(analysis_results.file_date)" in statement:
            seen.append(statement)
    event.listen(db._engine, "before_cursor_execute", spy)
    try:
        ov = od.load_overview(db)
    finally:
        event.remove(db._engine, "before_cursor_execute", spy)
    assert len(ov.others) == 6
    assert len(seen) == 1, seen


def test_a_model_with_no_trims_shows_its_final_test_pass(tmp_path, monkeypatch):
    """8506's shape on the work data: its trims are stored under other names, so its card can
    only show final test -- and says so."""
    db = _db(tmp_path)
    _trims(db, "OTHER", ANCHOR, passes=5)
    _final_tests(db, "FTONLY", ANCHOR - timedelta(days=3), passes=8, fails=2)
    _final_tests(db, "FTONLY", ANCHOR - timedelta(days=200), passes=10)
    _fake_sources(monkeypatch, flags=[_flag("FTONLY", DriftTier.DRIFT, 1.0, "ft_fail_fraction")],
                  statuses={"FTONLY": _status("FTONLY", "ft_fail_fraction", 0.02, 0.2)})
    (card,) = od.load_overview(db).cards
    assert card.final_test is True
    assert (card.units, card.pass_pct, card.was_pct) == (10, pytest.approx(80.0), pytest.approx(100.0))
    assert card.months[-1] == pytest.approx(80.0)


def test_a_card_with_trims_shows_its_trims_not_its_final_tests(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _trims(db, "BOTHSTATIONS", ANCHOR, passes=9, fails=1)
    _final_tests(db, "BOTHSTATIONS", ANCHOR, fails=10)
    _fake_sources(monkeypatch, flags=[_flag("BOTHSTATIONS", DriftTier.DRIFT, 1.0)],
                  statuses={"BOTHSTATIONS": _status("BOTHSTATIONS", "untrimmed_error_max", 0.1, 0.2)})
    (card,) = od.load_overview(db).cards
    assert card.final_test is False and card.pass_pct == pytest.approx(90.0)


def test_the_hand_trim_models_are_tagged_on_cards_and_rows(tmp_path, monkeypatch):
    # James's list so far (2026-09). On these the laser PASS rate is hand-trim workload.
    assert od.HAND_TRIM_MODELS == frozenset({"8232-1", "8340-1"})
    db = _db(tmp_path)
    for m in ("8232-1", "8340-1", "7000"):
        _trims(db, m, ANCHOR, passes=5, fails=5)
    _fake_sources(monkeypatch, flags=[_flag("8232-1", DriftTier.DRIFT, 1.0)],
                  statuses={"8232-1": _status("8232-1", "untrimmed_error_max", 0.1, 0.2)})
    ov = od.load_overview(db)
    assert [(c.model, c.hand_trim) for c in ov.cards] == [("8232-1", True)]
    assert {r.model: r.hand_trim for r in ov.others} == {"8340-1": True, "7000": False}


# ---- "Everything else" ---------------------------------------------------------------------------

def test_everything_else_is_every_other_active_model_busiest_first(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _trims(db, "SMALL", ANCHOR, passes=5)
    _trims(db, "BUSY", ANCHOR, passes=20)
    _trims(db, "MIDDLE", ANCHOR, passes=10)
    _trims(db, "CARDED", ANCHOR, passes=50)
    _trims(db, "OLD", ANCHOR - timedelta(days=120), passes=40)    # nothing in the 90 days
    _trims(db, "BROKEN", ANCHOR, errors=30)                       # nothing GRADED in the 90 days
    _fake_sources(monkeypatch, flags=[_flag("CARDED", DriftTier.DRIFT, 1.0)],
                  statuses={"CARDED": _status("CARDED", "untrimmed_error_max", 0.1, 0.2)})
    ov = od.load_overview(db)
    assert [r.model for r in ov.others] == ["BUSY", "MIDDLE", "SMALL"]


@pytest.mark.parametrize("now,was,expected", [
    (80.0, None, ("new", "new")),
    (80.0, 76.0, ("steady", "steady")),
    (80.0, 75.0, ("steady", "steady")),          # within +/-5 points, inclusive
    (80.0, 85.0, ("steady", "steady")),
    (80.0, 74.0, ("up 6 pts", "up")),
    (70.0, 80.0, ("down 10 pts", "down")),
    (80.4, 74.6, ("up 6 pts", "up")),            # whole points, then the +/-5 rule
])
def test_the_trend_words(now, was, expected):
    assert od.trend_words(now, was) == expected


# ---- the inactive models: counted, never hidden (F5) ------------------------------------------

def test_the_inactive_models_come_from_core_activity(tmp_path):
    from test_model_activity import NEWEST, _file
    db = _db(tmp_path)
    _file(db, "LIVE", NEWEST)
    _file(db, "OLD", NEWEST - timedelta(days=900))
    _file(db, "NOTRIM", NEWEST - timedelta(days=10), statuses=("UNTRIMMED",))
    ov = od.load_overview(db)
    assert ov.inactive == {"OLD": NEWEST - timedelta(days=900), "NOTRIM": None}


# ---- a part that fails is NAMED; the rest still loads -------------------------------------------

def _boom(*_a, **_k):
    raise RuntimeError("invented crash")


@pytest.mark.parametrize("target,part", [
    ("focus", od.PART_FOCUS), ("drift", od.PART_DRIFT),
    ("rates", od.PART_RATES), ("activity", od.PART_ACTIVITY)])
def test_a_part_that_fails_is_named_and_the_rest_still_loads(tmp_path, monkeypatch, target, part):
    import laser_trim_analyzer.gui.v6.focus_data as fd
    db = _db(tmp_path)
    _trims(db, "F1", ANCHOR, passes=9, fails=1)
    _trims(db, "D1", ANCHOR, passes=10)
    _trims(db, "PLAIN", ANCHOR, passes=4)
    monkeypatch.setattr(fd, "compute_focus_list",
                        lambda db: FocusResult(focus=[_entry("F1", 0.05, 0.3)], chronic=[], anchor=None))
    monkeypatch.setattr(od, "get_drifting_models", lambda db, *a, **k: [_flag("D1", DriftTier.DRIFT, 1.0)])
    monkeypatch.setattr(od, "get_model_drift_status",
                        lambda db, model, reference_date=None: _status(model, "untrimmed_error_max", 0.1, 0.2))
    if target == "focus":
        monkeypatch.setattr(fd, "compute_focus_list", _boom)
    elif target == "drift":
        monkeypatch.setattr(od, "get_drifting_models", _boom)
    elif target == "rates":
        monkeypatch.setattr(od, "_rates_by_day", _boom)
    else:
        monkeypatch.setattr(od, "load_activity", _boom)
    ov = od.load_overview(db)
    assert set(ov.failed) == {part}
    assert ov.failed[part] == "RuntimeError: invented crash"
    models = [c.model for c in ov.cards]
    if target == "focus":
        assert models == ["D1"] and od.card_count(ov) is None
    elif target == "drift":
        assert models == ["F1"] and od.card_count(ov) is None
    else:
        assert models == ["F1", "D1"] and od.card_count(ov) == 2
    # A model the failed source would have carded is still an active model: it is listed with
    # the rest, never dropped (the banner says the list it would have been on could not load).
    if target == "rates":
        assert ov.others == [] and all(c.pass_pct is None for c in ov.cards)
        assert all(c.units is None for c in ov.cards)     # a count it does not have: none
        assert ov.anchor is None
    else:
        expected = {"focus": ["F1", "PLAIN"], "drift": ["D1", "PLAIN"]}.get(target, ["PLAIN"])
        assert [r.model for r in ov.others] == expected
    assert (ov.inactive is None) == (target == "activity")
    # ...and with no pass rates the page cannot tell an active model from a quiet one: unknown.
    assert (ov.quiet is None) == (target in ("activity", "rates"))


def test_a_database_that_cannot_answer_anything_still_returns_an_overview():
    class _Gone:
        def session(self):
            raise RuntimeError("invented: the database is gone")

        def __getattr__(self, name):
            return _boom
    ov = od.load_overview(_Gone())
    assert set(ov.failed) == {od.PART_FOCUS, od.PART_DRIFT, od.PART_RATES, od.PART_ACTIVITY,
                              od.PART_TREND}
    assert ov.cards == [] and ov.others == [] and od.card_count(ov) is None
    assert ov.trend is None


def test_the_two_data_health_counts_come_through(tmp_path, monkeypatch):
    db = _db(tmp_path)
    monkeypatch.setattr(type(db), "count_legacy_ft_verdicts", lambda self: 12, raising=False)
    monkeypatch.setattr(type(db), "count_failed_file_markers", lambda self: 70, raising=False)
    ov = od.load_overview(db)
    assert (ov.legacy_ft, ov.unreadable) == (12, 70)


# ---- the words the page prints -------------------------------------------------------------------

@pytest.mark.parametrize("count,text", [
    (None, "Models that need a look"), (0, "No models need a look"),
    (1, "1 model needs a look"), (13, "13 models need a look")])
def test_the_heading_never_prints_a_count_it_does_not_have(count, text):
    assert od.need_a_look(count) == text


@pytest.mark.parametrize("value,text", [
    (None, "—"), (0.0, "0%"), (0.3, "1%"), (71.6, "72%"), (99.6, "99%"), (100.0, "100%")])
def test_a_pass_rate_never_rounds_to_all_or_nothing(value, text):
    assert od.pct_text(value) == text


def test_an_empty_database_has_no_cards_no_rows_and_says_so(tmp_path):
    ov = od.load_overview(_db(tmp_path))
    assert ov.failed == {} and ov.cards == [] and ov.others == [] and ov.anchor is None
    assert od.card_count(ov) == 0
    assert od.header_line(ov) == "No models need a look · no trim files on record yet"


def test_a_failed_final_test_read_is_named_on_its_own_and_spoils_nothing_else(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _trims(db, "OTHER", ANCHOR, passes=5)
    _final_tests(db, "FTONLY", ANCHOR, passes=8, fails=2)
    _fake_sources(monkeypatch, flags=[_flag("FTONLY", DriftTier.DRIFT, 1.0, "ft_fail_fraction")],
                  statuses={"FTONLY": _status("FTONLY", "ft_fail_fraction", 0.02, 0.2)})
    monkeypatch.setattr(od, "_ft_rates_by_day", _boom)
    ov = od.load_overview(db)
    assert ov.failed == {od.PART_FT: "RuntimeError: invented crash"}
    (card,) = ov.cards
    assert card.pass_pct is None and card.final_test is False
    assert card.units is None                        # never "0 units" over a read that failed
    assert [r.model for r in ov.others] == ["OTHER"] and ov.anchor == ANCHOR
    assert od.card_count(ov) == 1                    # the cards themselves are all known


def test_the_model_page_header_and_the_overview_count_the_same_90_days(tmp_path):
    # One window, two screens (2026-10-02 merge): the Overview counts the 90 CALENDAR days ending
    # on the newest file's day; the Model page's header counted 90 days to the minute before the
    # newest file -- so files on the day before the Overview's first day, later in the day than the
    # newest file, counted on the header only. Same model, two different pass rates.
    from datetime import time as _time
    from laser_trim_analyzer.gui.v6.overview_data import load_overview
    from laser_trim_analyzer.gui.v6.pages.model_page import header_facts
    db = _db(tmp_path)
    _trims(db, "M1", ANCHOR, passes=5)
    _trims(db, "M1", datetime.combine((ANCHOR - timedelta(days=89)).date(), _time(9, 0)), passes=2)
    _trims(db, "M1", datetime.combine((ANCHOR - timedelta(days=90)).date(), _time(20, 0)), fails=3)
    overview = load_overview(db)
    row = next(r for r in list(overview.cards) + list(overview.others) if r.model == "M1")
    facts = header_facts(db, "M1")
    assert (facts["units"], facts["passed"]) == (row.units, round(row.pass_pct * row.units / 100))
    assert row.units == 7


def test_a_final_test_after_the_newest_trim_counts_on_neither_screen(tmp_path, monkeypatch):
    # A model graded on final test (no trims of its own in the window): the Overview stops at the
    # newest TRIM file's day, so the header must too -- a final test dated after it counted on the
    # header only.
    from laser_trim_analyzer.gui.v6.pages.model_page import header_facts
    db = _db(tmp_path)
    _trims(db, "OTHER", ANCHOR, passes=5)
    _final_tests(db, "FTONLY", ANCHOR - timedelta(days=3), passes=8, fails=2)
    _final_tests(db, "FTONLY", ANCHOR + timedelta(hours=10), fails=4)     # the next morning
    _fake_sources(monkeypatch, flags=[_flag("FTONLY", DriftTier.DRIFT, 1.0, "ft_fail_fraction")],
                  statuses={"FTONLY": _status("FTONLY", "ft_fail_fraction", 0.02, 0.2)})
    (card,) = od.load_overview(db).cards
    facts = header_facts(db, "FTONLY", now=ANCHOR + timedelta(days=1))
    assert card.final_test is True and facts["basis"] == "final test"
    assert (facts["units"], facts["passed"]) == (card.units, 8) == (10, 8)


# ---- the final review of the redesign (2026-10-02) ---------------------------------------------

def _focus_entry(model, p_base, p_recent, metric="linearity_fail_fraction"):
    """A FOCUS row with its series, as compute_focus_list hands one over."""
    return SimpleNamespace(model=model, p_base=p_base, p_recent=p_recent,
                           series=SimpleNamespace(metric=metric))


def test_a_card_names_the_signal_its_click_should_chart(tmp_path, monkeypatch):
    """A click opened the Model page charting whatever the previous model charted. The card now
    names its signal: a fail-rate card its fail rate (the FOCUS series' own metric -- also on a
    model on both lists, whose reason reads the fail rate first), a detector card its worst."""
    db = _db(tmp_path)
    _trims(db, "ANY", ANCHOR, passes=1)
    _fake_sources(
        monkeypatch, focus=[_focus_entry("F1", 0.04, 0.12), _focus_entry("BOTH", 0.1, 0.4)],
        flags=[_flag("BOTH", DriftTier.DRIFT, 1.5),
               _flag("RES", DriftTier.DRIFT, 1.0, metric="untrimmed_resistance")],
        statuses={"BOTH": _status("BOTH", "untrimmed_error_max", 0.1, 0.2),
                  "RES": _status("RES", "untrimmed_resistance", 4693.0, 5897.0)})
    metrics = {c.model: c.metric for c in od.load_overview(db).cards}
    assert metrics == {"F1": "linearity_fail_fraction", "BOTH": "linearity_fail_fraction",
                       "RES": "untrimmed_resistance"}


def test_every_model_on_file_is_somewhere_on_the_overview(tmp_path):
    """A model trimmed before the 90 days but within two years was on no card, in no list and not
    inactive: on file, and nowhere on the page. It is counted in a collapsed line of its own
    now, with its last trim, as the inactive ones are (F5, James: "i dont want to hide them")."""
    from test_model_activity import NEWEST, _file
    db = _db(tmp_path)
    _file(db, "LIVE", NEWEST)
    _file(db, "QUIET", NEWEST - timedelta(days=200))
    _file(db, "OLD", NEWEST - timedelta(days=900))
    ov = od.load_overview(db)
    assert [r.model for r in ov.others] == ["LIVE"]
    assert ov.quiet == {"QUIET": NEWEST - timedelta(days=200)}
    assert set(ov.inactive) == {"OLD"}


def test_the_overview_never_pays_for_a_last_processed_stamp_it_does_not_show(tmp_path, monkeypatch):
    """load_focus's "last processed" stamp walks every model (list_known_models runs the drift
    detector over all of them again); the Overview never prints it."""
    import laser_trim_analyzer.gui.v6.focus_data as fd
    import laser_trim_analyzer.ml.manager as mlm
    db = _db(tmp_path)
    _trims(db, "ANY", ANCHOR, passes=1)
    calls = []
    monkeypatch.setattr(fd, "compute_focus_list",
                        lambda db: FocusResult(focus=[], chronic=[], anchor=None))
    monkeypatch.setattr(mlm, "list_known_models", lambda *a, **k: calls.append(1) or [])
    ov = od.load_overview(db)
    assert calls == [] and ov.failed == {}



def test_no_model_is_on_the_page_twice_and_none_on_file_is_missing(tmp_path, monkeypatch):
    """Every model on file, in exactly one place: a card, "Everything else", "Other models on
    file" or "Inactive models". A model with no trim file at all -- smoothness records only (8213-1
    and 8508 on the work data, each with over a thousand final tests too) -- was in the Models
    picker and nowhere on the page (re-review, 2026-10-02); it reads "no trim file on record". A
    name found only on final tests is not a model the picker opens: not listed."""
    from test_model_activity import NEWEST, _file
    db = _db(tmp_path)
    _file(db, "LIVE", NEWEST)
    _file(db, "CARDED", NEWEST - timedelta(days=200))      # a real laser file -- and on a card
    _file(db, "QUIET", NEWEST - timedelta(days=200))
    _file(db, "OLD", NEWEST - timedelta(days=900))
    db.save_smoothness_result({"filename": "SMONLY-1.xls", "file_path": str(tmp_path / "SMONLY-1.xls"),
                               "model": "SMONLY", "serial": "1", "file_date": NEWEST,
                               "test_date": NEWEST}, [], file_hash="sm-SMONLY-1")   # no trim file
    _final_tests(db, "FTNAME", NEWEST - timedelta(days=5), passes=3)     # a name on final tests only
    _fake_sources(monkeypatch, focus=[_focus_entry("CARDED", 0.1, 0.5)])
    ov = od.load_overview(db)
    placed = ([c.model for c in ov.cards] + [r.model for r in ov.others] + list(ov.quiet)
              + list(ov.inactive))
    assert sorted(placed) == ["CARDED", "LIVE", "OLD", "QUIET", "SMONLY"]       # each once
    assert ov.quiet == {"QUIET": NEWEST - timedelta(days=200), "SMONLY": None}


def test_when_the_models_on_file_cannot_be_read_the_line_is_unknown_and_named(tmp_path, monkeypatch):
    from test_model_activity import NEWEST, _file
    db = _db(tmp_path)
    _file(db, "LIVE", NEWEST)
    monkeypatch.setattr(od, "_models_on_file", _boom)
    ov = od.load_overview(db)
    assert ov.quiet is None and ov.failed == {od.PART_ON_FILE: "RuntimeError: invented crash"}
    assert [r.model for r in ov.others] == ["LIVE"]              # the rest still loads


def test_the_card_rule_names_its_ninety_days():
    """The FOCUS half drops a model whose newest bad run is over 90 days old, and the drift half
    flags only on a run within 90 days: the printed rule says so (re-review, 2026-10-02)."""
    from laser_trim_analyzer.ml.drift_types import RECENT_LOT_DAYS
    assert f"a run of the last {RECENT_LOT_DAYS} days" in od.CARD_RULE



# ---- each laser, charted (James, 2026-10-04) ----------------------------------------------------

def test_the_overview_charts_each_laser_with_the_company_trend(tmp_path):
    """James, 2026-10-04: "on the overveiw screen i no longer have each laser charted overall?" --
    the chart of each laser and the company lived on the Dashboard (since the redesign, the
    "Company trends" link). The Overview loads that same chart: the last 12 months, by month."""
    db = _db(tmp_path)
    _trims(db, "M1", datetime.now() - timedelta(days=10), passes=3, fails=1)
    ov = od.load_overview(db)
    assert od.TREND_DAYS == 365
    assert ov.trend == db.get_company_yield_trend(days_back=365, period="month")
    assert ov.trend["by_system"]["A"][-1]["linearity_yield"] == 75.0


def test_a_chart_that_cannot_load_is_named_and_the_rest_still_loads(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _trims(db, "M1", ANCHOR, passes=3)
    monkeypatch.setattr(db, "get_company_yield_trend", _boom)
    ov = od.load_overview(db)
    assert ov.trend is None and ov.failed == {od.PART_TREND: "RuntimeError: invented crash"}
    assert [r.model for r in ov.others] == ["M1"]


# ---- option B (2026-10-04): the dollars lost at final test, each model's lasers, the header line --
# Every price and cost ratio below is INVENTED: real prices are customer data (CLAUDE.md).

PRICES = {"PRICED": 10.0, "CHEAP": 2.5, "NOFAILS": 7.0, "ZERO": 0.0}


def _trims_on(db, model, when, system, *, passes=0, fails=0, errors=0, suspect_passes=0):
    """`model`'s trim files on one laser (`system`, the code's letter)."""
    from laser_trim_analyzer.database.models import AnalysisResult, StatusType, SystemType
    with db.session() as s:
        for status, count, quality in (("PASS", passes, "good"), ("FAIL", fails, "good"),
                                       ("ERROR", errors, "good"), ("PASS", suspect_passes, "suspect")):
            for _ in range(count):
                n = next(_serial)
                s.add(AnalysisResult(model=model, serial=f"{model}-{n}", system=SystemType[system],
                                     filename=f"{model}_{n}.xls", file_date=when,
                                     overall_status=StatusType[status], data_quality=quality))


def _ft(db, model, when, status, count=1):
    from laser_trim_analyzer.database.models import FinalTestResult, StatusType
    with db.session() as s:
        for _ in range(count):
            n = next(_serial)
            s.add(FinalTestResult(filename=f"FT_{model}_{n}.xls", model=model,
                                  serial=f"{model}-ft-{n}", file_date=when,
                                  overall_status=StatusType[status]))


def _by_model(ov):
    return {x.model: x for x in list(ov.cards) + list(ov.others)}


def test_the_money_is_final_test_fails_in_the_ninety_days_times_price_times_ratio(tmp_path):
    """The Company trends formula -- FT fails x unit price x cost ratio -- over the Overview's OWN
    window: the 90 calendar days ending on the newest counted trim file's day."""
    db = _db(tmp_path)
    for m in ("PRICED", "CHEAP"):
        _trims(db, m, ANCHOR, passes=5)
    _ft(db, "PRICED", ANCHOR - timedelta(days=2), "FAIL", 3)               # counted
    _ft(db, "PRICED", (ANCHOR - timedelta(days=89)).replace(hour=0, minute=1), "FAIL", 1)  # first day
    _ft(db, "PRICED", ANCHOR - timedelta(days=90), "FAIL", 2)              # the day before: out
    _ft(db, "PRICED", ANCHOR + timedelta(hours=10), "FAIL", 4)             # after the anchor's day
    _ft(db, "PRICED", ANCHOR, "PASS", 6)                                   # passes cost nothing
    _ft(db, "PRICED", ANCHOR, "WARNING", 2)                                # a watch, not a fail
    _ft(db, "CHEAP", ANCHOR - timedelta(days=10), "FAIL", 4)
    ov = od.load_overview(db, prices=PRICES, cost_ratio=0.4)
    got = _by_model(ov)
    assert (got["PRICED"].ft_fails, got["PRICED"].money) == (4, pytest.approx(4 * 10.0 * 0.4))
    assert (got["CHEAP"].ft_fails, got["CHEAP"].money) == (4, pytest.approx(4 * 2.5 * 0.4))
    assert ov.money_total == pytest.approx(16.0 + 4.0) and ov.unpriced == 0
    assert ov.failed == {}


def test_a_model_with_no_price_has_no_money_and_is_counted_only_when_it_failed(tmp_path):
    db = _db(tmp_path)
    for m in ("NOPRICE", "NOPRICECLEAN", "NOFAILS", "ZERO"):
        _trims(db, m, ANCHOR, passes=5)
    _ft(db, "NOPRICE", ANCHOR, "FAIL", 2)
    _ft(db, "NOPRICECLEAN", ANCHOR, "PASS", 2)
    _ft(db, "ZERO", ANCHOR, "FAIL", 3)                # a loaded price of 0 is a price, not "none"
    ov = od.load_overview(db, prices=PRICES, cost_ratio=0.5)
    got = _by_model(ov)
    assert (got["NOPRICE"].money, got["NOPRICE"].ft_fails) == (None, 2)
    assert (got["NOPRICECLEAN"].money, got["NOPRICECLEAN"].ft_fails) == (None, 0)
    assert got["NOFAILS"].money == 0.0 and got["ZERO"].money == 0.0
    assert ov.money_total == 0.0
    assert ov.unpriced == 1                           # NOPRICE: it failed final test, no price


def test_the_cost_ratio_defaults_to_half_and_prices_may_be_written_any_way(tmp_path):
    db = _db(tmp_path)
    _trims(db, "PRICED", ANCHOR, passes=5)
    _trims(db, "9090", ANCHOR, passes=5)
    _ft(db, "PRICED", ANCHOR, "FAIL", 2)
    _ft(db, "9090", ANCHOR, "FAIL", 1)
    ov = od.load_overview(db, prices={"PRICED": "10", 9090: 4.0, "BAD": "not a price"})
    got = _by_model(ov)
    assert got["PRICED"].money == pytest.approx(2 * 10.0 * 0.5)       # YAML may hand a string
    assert got["9090"].money == pytest.approx(1 * 4.0 * 0.5)          # ...or a number for a key
    bare = od.load_overview(db)                       # no prices handed over at all
    assert (bare.money_total, bare.unpriced, bare.no_prices) == (0.0, 2, True)
    assert od.load_overview(db, prices={"PRICED": 1.0}).no_prices is False


def test_the_header_total_is_the_full_cost_every_priced_model_on_the_page_or_not(tmp_path):
    """The header's dollars are every final-test FAIL in the Overview's 90 days on a model with a
    price -- a model on no card and in no list too (a name found only on final tests, a model
    trimmed before the window): the full cost, as Company trends counts it. A row's dollars stay
    its own (coordinator, 2026-10-04)."""
    db = _db(tmp_path)
    _trims(db, "PRICED", ANCHOR, passes=5)                            # on the page
    _ft(db, "PRICED", ANCHOR, "FAIL", 2)
    _ft(db, "OFFPAGE", ANCHOR - timedelta(days=5), "FAIL", 3)         # final tests only: no row
    _ft(db, "OFFPAGE", ANCHOR - timedelta(days=95), "FAIL", 9)        # before the 90 days
    _ft(db, "OFFPAGE", ANCHOR + timedelta(hours=10), "FAIL", 4)       # after the anchor's day
    _ft(db, "STRAY", ANCHOR - timedelta(days=1), "FAIL", 4)           # no price, no row
    _ft(db, "CLEANOFF", ANCHOR, "PASS", 5)                            # no price, never failed
    ov = od.load_overview(db, prices={"PRICED": 10.0, "OFFPAGE": 2.0}, cost_ratio=0.5)
    (row,) = ov.others
    assert row.model == "PRICED" and row.money == pytest.approx(2 * 10.0 * 0.5)
    assert ov.money_total == pytest.approx(2 * 10.0 * 0.5 + 3 * 2.0 * 0.5)
    assert ov.unpriced == 1                                           # STRAY: failed, no price
    assert ov.no_prices is False


def test_a_final_test_card_is_charged_for_its_own_final_test_fails(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _trims(db, "OTHER", ANCHOR, passes=5)
    _final_tests(db, "FTONLY", ANCHOR - timedelta(days=3), passes=8, fails=2)
    _fake_sources(monkeypatch, flags=[_flag("FTONLY", DriftTier.DRIFT, 1.0, "ft_fail_fraction")],
                  statuses={"FTONLY": _status("FTONLY", "ft_fail_fraction", 0.02, 0.2)})
    (card,) = od.load_overview(db, prices={"FTONLY": 3.0}, cost_ratio=1.0).cards
    assert card.final_test is True and (card.ft_fails, card.money) == (2, pytest.approx(6.0))


def test_final_tests_past_the_window_or_undated_are_not_counted(tmp_path):
    db = _db(tmp_path)
    _trims(db, "PRICED", datetime.now() - timedelta(days=1), passes=5)
    _ft(db, "PRICED", datetime.now() - timedelta(days=2), "FAIL", 1)
    _ft(db, "PRICED", datetime.now() + timedelta(days=30), "FAIL", 9)      # more than a day ahead
    _ft(db, "PRICED", None, "FAIL", 5)                                    # no date at all
    (row,) = od.load_overview(db, prices=PRICES).others
    assert row.ft_fails == 1


def test_a_failed_money_read_is_named_and_never_reads_as_zero(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _trims(db, "PRICED", ANCHOR, passes=5)
    _ft(db, "PRICED", ANCHOR, "FAIL", 2)
    monkeypatch.setattr(od, "_ft_fails_by_model", _boom)
    ov = od.load_overview(db, prices=PRICES)
    assert ov.failed == {od.PART_MONEY: "RuntimeError: invented crash"}
    assert ov.money_total is None
    (row,) = ov.others
    assert row.money is None and row.ft_fails is None and row.units == 5    # the rest still loads
    assert od.money_text(ov, row) == "could not be worked out"


def test_without_the_pass_rates_the_money_is_unknown_and_only_the_rates_are_named(tmp_path,
                                                                                monkeypatch):
    """The dollars are counted over the Overview's own window, which the pass rates anchor."""
    db = _db(tmp_path)
    _trims(db, "PRICED", ANCHOR, passes=5)
    monkeypatch.setattr(od, "_rates_by_day", _boom)
    ov = od.load_overview(db, prices=PRICES)
    assert set(ov.failed) == {od.PART_RATES} and ov.money_total is None
    assert "could not be worked out" in od.header_line(ov)


def test_each_models_lasers_are_the_ones_with_graded_trims_in_the_window_in_laser_order(tmp_path):
    db = _db(tmp_path)
    _trims_on(db, "TWO", ANCHOR, "C", passes=2)                            # laser 3
    _trims_on(db, "TWO", ANCHOR - timedelta(days=5), "A", fails=1)         # laser 2
    _trims_on(db, "TWO", ANCHOR - timedelta(days=200), "B", passes=4)      # laser 1, before the window
    _trims_on(db, "TWO", ANCHOR, "B", errors=3, suspect_passes=2)          # laser 1: none counted
    _trims_on(db, "ONE", ANCHOR, "A", passes=1)
    _trims_on(db, "ONE", ANCHOR, "B", passes=1)
    got = _by_model(od.load_overview(db))
    assert got["TWO"].lasers == ["A", "C"]                  # laser 2, laser 3 -- never the code's ABC
    assert got["ONE"].lasers == ["B", "A"]                  # laser 1 first
    assert got["ONE"].units == 2                            # one day, two lasers: both count
    assert got["TWO"].newest == ANCHOR.date()


def test_a_final_test_card_has_no_lasers_and_an_unknown_read_has_none_known(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _trims(db, "OTHER", ANCHOR, passes=5)
    _final_tests(db, "FTONLY", ANCHOR - timedelta(days=3), passes=8, fails=2)
    _fake_sources(monkeypatch, flags=[_flag("FTONLY", DriftTier.DRIFT, 1.0, "ft_fail_fraction")],
                  statuses={"FTONLY": _status("FTONLY", "ft_fail_fraction", 0.02, 0.2)})
    (card,) = od.load_overview(db).cards
    assert card.lasers == [] and card.newest == (ANCHOR - timedelta(days=3)).date()
    monkeypatch.setattr(od, "_rates_by_day", _boom)
    (card,) = od.load_overview(db).cards
    assert card.lasers is None and card.newest is None


@pytest.mark.parametrize("kw,text", [
    (dict(cards=13, money_total=1234.4),
     "13 models need a look · $1,234 lost at final test in the last 90 days · newest file 20 Mar 2026"),
    (dict(cards=1, money_total=0.0, unpriced=3),
     "1 model needs a look · $0 lost at final test in the last 90 days · 3 without a price · "
     "newest file 20 Mar 2026"),
    (dict(cards=2, money_total=0.4),
     "2 models need a look · <$1 lost at final test in the last 90 days · newest file 20 Mar 2026"),
    (dict(cards=2, money_total=None, failed={od.PART_MONEY: "RuntimeError: x"}),
     "2 models need a look · dollars lost at final test could not be worked out · newest file 20 Mar 2026"),
    (dict(cards=2, money_total=50.0, failed={od.PART_DRIFT: "RuntimeError: x"}),
     "Models that need a look · $50 lost at final test in the last 90 days · newest file 20 Mar 2026"),
    (dict(cards=0, money_total=None, anchor=None),
     "No models need a look · no trim files on record yet"),
    (dict(cards=2, money_total=None, anchor=None, failed={od.PART_RATES: "RuntimeError: x"}),
     "2 models need a look · dollars lost at final test could not be worked out"),
    # No price loaded at all: a missing input never reads as "$0" (coordinator, 2026-10-04).
    (dict(cards=2, money_total=0.0, unpriced=3, no_prices=True),
     "2 models need a look · add prices in Settings → Backlog to see the dollars lost at final "
     "test · newest file 20 Mar 2026"),
    (dict(cards=2, money_total=None, no_prices=True, failed={od.PART_MONEY: "RuntimeError: x"}),
     "2 models need a look · dollars lost at final test could not be worked out · newest file "
     "20 Mar 2026"),
])
def test_the_header_line(kw, text):
    kw = dict(kw)
    cards = [od.Card(model=f"M{i}", reason="") for i in range(kw.pop("cards"))]
    kw.setdefault("anchor", ANCHOR)
    assert od.header_line(od.Overview(cards=cards, **kw)) == text


def test_the_money_words_for_one_model():
    ov = od.Overview(anchor=ANCHOR, money_total=10.0)
    assert od.money_text(ov, od.Card(model="A", reason="", money=1234.5, ft_fails=12)) == \
        "$1,234 · 12 failed final test"
    assert od.money_text(ov, od.Card(model="A", reason="", money=None, ft_fails=1)) == \
        "no price · 1 failed final test"
    assert od.money_text(ov, od.Card(model="A", reason="", money=0.0, ft_fails=0)) == "$0"
    assert od.money_text(ov, od.Card(model="A", reason="", money=None, ft_fails=0)) == "no price"


# ---- the sweep holds the dollars to independent SQL (scripts/app_qa_sweep.py) -------------------
# check_overview_money_on_database reads the CONFIG's prices; these hand it invented ones. The sweep
# stubs tkinter at import, so it runs in a subprocess (tests/test_sweep_db_checks.py's runner).

# Invented. "NOPRICE" and "STRAY" have none; "OFFPAGE" and "STRAY" are on no card and in no list
# (final tests only) -- the header's total and its unpriced count are every model's.
SWEEP_PRICES = {"PRICED": 12.5, "CHEAP": 3.25, "OFFPAGE": 6.75}
SWEEP_RATIO = 0.4
SWEEP_FAILS = {"PRICED": 7, "CHEAP": 9, "OFFPAGE": 2}    # each priced model's fails in the window


def _money_scratch(tmp_path):
    db = _db(tmp_path)
    for m in ("PRICED", "CHEAP", "NOPRICE", "CLEAN"):
        _trims(db, m, ANCHOR, passes=5)
    _ft(db, "PRICED", ANCHOR - timedelta(days=3), "FAIL", 7)
    _ft(db, "PRICED", ANCHOR - timedelta(days=95), "FAIL", 5)          # before the 90 days
    _ft(db, "PRICED", ANCHOR, "WARNING", 2)
    _ft(db, "CHEAP", ANCHOR - timedelta(days=40), "FAIL", 9)
    _ft(db, "NOPRICE", ANCHOR - timedelta(days=1), "FAIL", 4)
    _ft(db, "CLEAN", ANCHOR, "PASS", 6)
    _ft(db, "OFFPAGE", ANCHOR - timedelta(days=6), "FAIL", 2)          # priced, not on the page
    _ft(db, "STRAY", ANCHOR - timedelta(days=2), "FAIL", 3)            # no price, not on the page
    return db


def _run_money_check(db, patch="", prices=None):
    prices = SWEEP_PRICES if prices is None else prices
    from test_sweep_db_checks import _run_code
    db.close()
    code = (
        "import sqlite3\n"
        "from types import SimpleNamespace\n"
        "import laser_trim_analyzer.database.manager as _m, laser_trim_analyzer.database as _d\n"
        f"_db = _m.DatabaseManager(r'{db.database_path}'); _m._db_manager = _db; _d._db_manager = _db\n"
        f"{patch}\n"
        f"cfg = SimpleNamespace(active_models=SimpleNamespace(model_prices={prices!r},"
        f" cost_ratio={SWEEP_RATIO!r}))\n"
        f"raw = sqlite3.connect('file:{db.database_path}?mode=ro', uri=True)\n"
        "sweep.check_overview_money_on_database(_db, raw, cfg)\n")
    r, results = _run_code(code)
    assert r.returncode == 0 and results is not None, r.stdout[-3000:] + r.stderr[-3000:]
    return results


def _never_a_price(results):
    """A check may name models and counts -- never a price or a dollar figure."""
    said = " ".join(f"{n} {d}" for _v, n, d in results)
    dollars = [n * SWEEP_PRICES[m] * SWEEP_RATIO for m, n in SWEEP_FAILS.items()]
    for figure in list(SWEEP_PRICES.values()) + dollars + [sum(dollars)]:
        assert str(figure) not in said and f"{figure:.2f}" not in said, (figure, said)
    assert "$" not in said, said


def test_the_money_check_passes_when_the_dollars_match_their_definition(tmp_path):
    results = _run_money_check(_money_scratch(tmp_path))
    assert not [x for x in results if x[0] == "FAIL"], results
    passed = {n for v, n, _ in results if v == "PASS"}
    assert any(n.startswith("overview money: each card's and row's final-test fails") for n in passed)
    assert any(n.startswith("overview money: each card's and row's dollars") for n in passed)
    assert any(n.startswith("overview money: the header's total") for n in passed)
    assert any(n.startswith("overview money: 'without a price'") for n in passed)
    assert any("3 priced models failed final test, 1 of them not on the page" in d
               for _v, _n, d in results), results
    _never_a_price(results)


def test_the_money_check_fails_when_fails_before_the_window_are_counted(tmp_path):
    results = _run_money_check(
        _money_scratch(tmp_path),
        patch="import laser_trim_analyzer.gui.v6.overview_data as od\n"
              "od._window = lambda anchor_day: (anchor_day - od.timedelta(days=120),"
              " anchor_day - od.timedelta(days=485))")
    failed = [(n, d) for v, n, d in results if v == "FAIL"]
    assert any("final-test fails" in n and "PRICED" in d for n, d in failed), results
    _never_a_price(results)


def test_the_money_check_fails_when_the_cost_ratio_is_ignored(tmp_path):
    results = _run_money_check(
        _money_scratch(tmp_path),
        patch="import laser_trim_analyzer.gui.v6.overview_data as od\nod._clean_ratio = lambda r: 1.0")
    failed = [(n, d) for v, n, d in results if v == "FAIL"]
    assert any("dollars" in n and "PRICED" in d and "CHEAP" in d for n, d in failed), results
    assert any("total" in n for n, d in failed), results
    _never_a_price(results)


def test_the_money_check_fails_when_an_unpriced_model_is_not_counted(tmp_path):
    results = _run_money_check(
        _money_scratch(tmp_path),
        patch="import laser_trim_analyzer.gui.v6.overview_data as od\n"
              "_real = od._charge\n"
              "def _charge(ov, fails, prices, ratio):\n"
              "    _real(ov, fails, prices, ratio)\n"
              "    ov.unpriced = 0\n"
              "od._charge = _charge")
    failed = [(n, d) for v, n, d in results if v == "FAIL"]
    assert any("without a price" in n and "NOPRICE" in d for n, d in failed), results


def test_the_money_check_never_passes_on_nothing(tmp_path):
    db = _db(tmp_path)
    _trims(db, "PRICED", ANCHOR, passes=5)                     # trims, but no final test at all
    results = _run_money_check(db)
    assert not any(v == "PASS" for v, _n, _d in results), results
    assert any(v == "WARN" for v, _n, _d in results), results


def test_the_money_check_fails_when_the_total_counts_only_the_models_on_the_page(tmp_path):
    results = _run_money_check(
        _money_scratch(tmp_path),
        patch="import laser_trim_analyzer.gui.v6.overview_data as od\n"
              "_real = od._charge\n"
              "def _charge(ov, fails, prices, ratio):\n"
              "    _real(ov, fails, prices, ratio)\n"
              "    ov.money_total = sum(x.money or 0.0 for x in ov.cards + ov.others)\n"
              "od._charge = _charge")
    failed = [(n, d) for v, n, d in results if v == "FAIL"]
    assert any(n.startswith("overview money: the header's total") for n, d in failed), results
    _never_a_price(results)


def test_the_money_check_fails_when_without_a_price_counts_only_the_models_on_the_page(tmp_path):
    results = _run_money_check(
        _money_scratch(tmp_path),
        patch="import laser_trim_analyzer.gui.v6.overview_data as od\n"
              "_real = od._charge\n"
              "def _charge(ov, fails, prices, ratio):\n"
              "    _real(ov, fails, prices, ratio)\n"
              "    ov.unpriced = sum(1 for x in ov.cards + ov.others if x.ft_fails and x.money is None)\n"
              "od._charge = _charge")
    failed = [(n, d) for v, n, d in results if v == "FAIL"]
    assert any("without a price" in n and "STRAY" in d for n, d in failed), results


def test_with_no_prices_the_money_check_holds_the_header_to_asking_for_them(tmp_path):
    asks = "overview money: with no price loaded the header asks for prices, never '$0'"
    results = _run_money_check(_money_scratch(tmp_path), prices={})
    assert asks in {n for v, n, _ in results if v == "PASS"}, results
    assert not [x for x in results if x[0] == "FAIL"], results
    again = tmp_path / "again"
    again.mkdir()
    results = _run_money_check(
        _money_scratch(again), prices={},
        patch="import laser_trim_analyzer.gui.v6.overview_data as od\n"
              "od.money_words = lambda ov: '$0 lost at final test in the last 90 days'")
    assert asks in {n for v, n, _ in results if v == "FAIL"}, results
