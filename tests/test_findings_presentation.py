"""Every rule the Findings screens follow, tested without a window (spec section 3)."""
import logging

import pytest

from laser_trim_analyzer.findings import presentation as P


def f(analyzer, model="M", **kw):
    base = {"analyzer": analyzer, "model": model, "category": kw.pop("category", ""),
            "title": kw.pop("title", "t"), "summary": "s", "systems": kw.pop("systems", ["B"]),
            "n_units": kw.pop("n_units", 100), "tracks_per_year": kw.pop("tpy", None),
            "evidence": kw.pop("evidence", {}), "computed_at": kw.pop("computed_at", None)}
    base.update(kw)
    return base


def cut(model="M", best=6800.0, current=6900.0, tpy=100.0, grade="two_periods", track="Track A", **kw):
    return f("cut_setting", model, category="Cut setting", tpy=tpy,
             evidence={"best": best, "current": current, "grade": grade, "track": track}, **kw)


def test_every_analyzer_the_engine_runs_has_a_group():
    import pkgutil
    import laser_trim_analyzer.findings.analyzers as pkg
    names = {m.name for m in pkgutil.iter_modules(pkg.__path__)}
    missing = sorted(names - set(P.ANALYZER_GROUP))
    assert not missing, f"analyzers with no group on the Findings page: {missing}"


def test_an_unknown_analyzer_is_shown_under_other_never_dropped(caplog):
    with caplog.at_level(logging.ERROR):
        groups = P.arrange([f("brand_new")])
    other = [g for g in groups if g.spec.key == "other"]
    assert other and other[0].rows and other[0].rows[0].findings[0]["analyzer"] == "brand_new"
    assert any("brand_new" in r.getMessage() for r in caplog.records)


def test_the_four_groups_come_in_order_and_other_only_when_needed():
    keys = [g.spec.key for g in P.arrange([])]
    assert keys == ["yield", "laser_time", "check", "history"]


def test_a_model_tab_hides_empty_groups():
    keys = [g.spec.key for g in P.arrange([cut()], include_empty=False)]
    assert keys == ["yield"]


@pytest.mark.parametrize("category,field,value", [
    ("Trim avoidance", "arrive_in_spec_n", 1008),
    ("Pass effectiveness", "multi_cut_n", 420),
    ("Multi-pass burden", "tracks_over_recipe", 124),
])
def test_laser_time_readouts_count_tracks(category, field, value):
    x = f("trim_effort" if category != "Multi-pass burden" else "pass_burden",
          category=category, evidence={"facts": {field: value, "unplanned_passes": 999}})
    assert P.readout(x) == value


def test_a_history_readout_is_the_pass_rate_move():
    x = f("recipe_change", evidence={"before": {"trim_pass_pct": 80.0}, "after": {"trim_pass_pct": 58.0}})
    assert P.readout(x) == pytest.approx(-22.0)
    assert P.value_text("history", -22.0) == "-22" and P.value_tone("history", -22.0) == "down"
    assert P.value_tone("history", 21.0) == "up"


def test_two_tracks_saying_the_same_thing_become_one_row():
    rows = P.arrange([cut(tpy=182.0, track="Track A"), cut(tpy=118.0, track="Track B")])[0].rows
    assert len(rows) == 1 and rows[0].merged
    assert rows[0].value == pytest.approx(300.0)
    assert "both tracks" in rows[0].tags


def test_different_recommendations_never_merge():
    rows = P.arrange([cut(best=6800.0), cut(best=6700.0)])[0].rows
    assert len(rows) == 2


def test_a_merged_row_shows_its_weakest_evidence():
    rows = P.arrange([cut(grade="same_days"), cut(grade="two_periods", track="Track B")])[0].rows
    assert "two periods · test first" in rows[0].tags and "same days" not in rows[0].tags


def test_yield_sorts_by_rate_then_size_with_no_rate_last():
    rows = P.arrange([cut("A", tpy=None, best=1.0), cut("B", tpy=5.0, best=2.0),
                      cut("C", tpy=50.0, best=3.0)])[0].rows
    assert [r.model for r in rows] == ["C", "B", "A"]
    assert P.value_text("yield", None) == "—" and P.value_text("yield", 300.4) == "~300"


def test_history_is_newest_first():
    old = f("recipe_change", "OLD", evidence={"after": {"first": "2023-07-17"}})
    new = f("recipe_change", "NEW", evidence={"after": {"first": "2025-01-13"}})
    rows = [g for g in P.arrange([old, new]) if g.spec.key == "history"][0].rows
    assert [r.model for r in rows] == ["NEW", "OLD"]


def test_statements_read_like_the_approved_design():
    assert P.statement(cut()) == "Laser 1 (LTS): cut 6900 → try 6800"
    x = f("recipe_change", title="Laser 1 (LTS): recipe changed from 1 cut (cut length 6800) "
                                 "to 1 cut (cut length 6900)",
          evidence={"after": {"first": "2024-10-04"}})
    assert P.statement(x) == "Oct 2024 · Laser 1 (LTS): cut 6800 → 6900"
    y = f("recipe_change", title="Laser 1 (LTS): recipe changed from 1 cut (cut length 2950) "
                                 "to 2 cuts (cut length 2950, 4500)",
          evidence={"after": {"first": "2025-01-13"}})
    assert P.statement(y) == "Jan 2025 · Laser 1 (LTS): 1 cut → 2 cuts"


def test_the_caption_counts_rows_after_merging_and_dates_portably():
    fs = [cut(computed_at="2026-09-23 11:52:51"), cut(track="Track B"),
          f("limit_tables"), f("trim_effort", category="Trim avoidance")]
    cap = P.caption(P.arrange(fs), fs)
    assert cap.startswith("1 change worth testing · 1 way to save laser time · 1 test to check")
    assert cap.endswith("worked out 23 Sep")          # never strftime("%-d"): it raises on Windows


def test_value_text_for_counts():
    assert P.value_text("laser_time", 1008.0) == "1,008" and P.value_text("check", 22.0) == "22"
