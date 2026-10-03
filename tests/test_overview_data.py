"""The Overview's one loader -- gui/v6/overview_data.load_overview (Graphite redesign, 2026-10-02).

James: "keep all 16 cards (fail rate up, or a signal moved), each with its reason". The cards are
the union of the "Drifting now" fail-rate list (ml/spc.compute_focus_list) and the drift
detector's flags (ml/manager.get_drifting_models); "Everything else" is every other active model;
the inactive models are counted, never hidden. Pass % is the app's headline yield, with suspect
files left out like the drift watch and the FOCUS list leave them out.

Every database here is a tmp file; every model, serial and date is INVENTED.
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
    monkeypatch.setattr(od, "load_focus", lambda db, models=None: (
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
    assert od.window_caption(ov) == "Last 90 days · newest file 20 Mar 2026"


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
        assert ov.anchor is None
    else:
        expected = {"focus": ["F1", "PLAIN"], "drift": ["D1", "PLAIN"]}.get(target, ["PLAIN"])
        assert [r.model for r in ov.others] == expected
    assert (ov.inactive is None) == (target == "activity")


def test_a_database_that_cannot_answer_anything_still_returns_an_overview():
    class _Gone:
        def session(self):
            raise RuntimeError("invented: the database is gone")

        def __getattr__(self, name):
            return _boom
    ov = od.load_overview(_Gone())
    assert set(ov.failed) == {od.PART_FOCUS, od.PART_DRIFT, od.PART_RATES, od.PART_ACTIVITY}
    assert ov.cards == [] and ov.others == [] and od.card_count(ov) is None


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
    assert od.window_caption(ov) == "No trim files on record yet"


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
