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


def cut(model="M", best=6800.0, current=6900.0, tpy=100.0, grade="two_periods", track="Track A",
        ev=None, **kw):
    """A cut_setting finding. `ev` adds evidence keys (table, stale, ran_on_laser_since...)."""
    return f("cut_setting", model, category="Cut setting", tpy=tpy,
             evidence={"best": best, "current": current, "grade": grade, "track": track, **(ev or {})},
             **kw)


def lt(track, model="6607", title="Laser 1 (LTS): the limit table changed"):
    """limit_tables writes one finding per (laser, track); its title names neither."""
    return f("limit_tables", model, category="Limit table", title=title, evidence={"track": track},
             summary=f"the evidence for {track}")


def rows_of(groups, key):
    return [g for g in groups if g.spec.key == key][0].rows


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
    assert P.value_text("history", -22.0) == "-22" and P.value_tone("history", -22.0, [x]) == "down"
    assert P.value_tone("history", 21.0, [x]) == "up"


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


# ---- Every row has its own key (final review, 2026-09-24) ----------------------------------------
# limit_tables writes one finding per (laser, track) and its title names neither; the row key was
# (group, model, statement), so two tracks made two identical rows under ONE key -- clicking the
# first opened the second's detail, and Track A's evidence could never be opened.

def test_two_findings_that_read_the_same_are_two_rows_each_naming_its_track():
    rows = rows_of(P.arrange([lt("Track A"), lt("Track B")]), "check")
    assert len(rows) == 2 and len({r.key for r in rows}) == 2
    assert sorted(r.statement for r in rows) == ["Laser 1 (LTS) · Track A: the limit table changed",
                                                 "Laser 1 (LTS) · Track B: the limit table changed"]


def test_a_row_keeps_its_own_key_whatever_order_the_findings_arrive_in():
    # A refresh can return the same findings in another order. A key that only numbered the twins
    # would hand Track A's key to Track B -- and the open row would silently become the other one.
    a, b = lt("Track A"), lt("Track B")
    first = {r.findings[0]["evidence"]["track"]: r.key for r in rows_of(P.arrange([a, b]), "check")}
    again = {r.findings[0]["evidence"]["track"]: r.key for r in rows_of(P.arrange([b, a]), "check")}
    assert first == again


def test_naming_the_track_does_not_change_a_rows_key():
    # The key is the statement as first worded: a row keeps its key when a same-reading sibling
    # appears (and its displayed statement gains the track).
    alone = rows_of(P.arrange([lt("Track A")]), "check")[0]
    assert alone.statement == "Laser 1 (LTS): the limit table changed"        # nothing to tell apart
    with_twin = [r for r in rows_of(P.arrange([lt("Track A"), lt("Track B")]), "check")
                 if r.findings[0]["evidence"]["track"] == "Track A"][0]
    assert with_twin.key == alone.key


def test_multi_pass_twins_name_their_track_and_their_cut():
    # pass_burden splits one track by first cut -- its facts label is "Laser 1 (LTS) · Track A · cut 4000".
    def pb(track, cut_setting):
        return f("pass_burden", "8340", category="Multi-pass burden",
                 title="Laser 1 (LTS): 34% of tracks need more than the 1 cut the recipe asks for",
                 evidence={"facts": {"tracks_over_recipe": 50, "cut_setting": cut_setting}, "track": track})
    rows = rows_of(P.arrange([pb("Track A", 4000.0), pb("Track A", 4100.0)]), "laser_time")
    assert len({r.key for r in rows}) == 2
    assert sorted(r.statement for r in rows) == [
        "Laser 1 (LTS) · Track A · cut 4000: 34% of tracks need more than the 1 cut the recipe asks for",
        "Laser 1 (LTS) · Track A · cut 4100: 34% of tracks need more than the 1 cut the recipe asks for"]


def test_rows_nothing_tells_apart_still_get_a_key_each():
    # A cache written before the track was stored, or an analyzer this page does not know: the same
    # words twice and no identity at all. Each row must still open ITSELF, never its twin.
    x = f("brand_new", title="the same words", summary="one")
    y = f("brand_new", title="the same words", summary="two")
    rows = rows_of(P.arrange([x, y]), "other")
    assert len(rows) == 2 and len({r.key for r in rows}) == 2


# ---- A pass-rate move across a limit-table change is not coloured (final review, item 4) ----------

@pytest.mark.parametrize("tables", [
    {"limit_table_changed": True, "limit_tables_mixed": True},      # the busiest table changed
    {"limit_table_changed": False, "limit_tables_mixed": True},     # a mix of tables on a side
    {"limit_table_changed": True},                                   # either flag alone is enough
])
def test_a_move_across_a_limit_table_change_keeps_its_number_but_no_colour(tables):
    x = f("recipe_change", evidence={"before": {"trim_pass_pct": 80.0},
                                     "after": {"trim_pass_pct": 58.0, "first": "2025-01-13"}, **tables})
    row = rows_of(P.arrange([x]), "history")[0]
    assert row.value == pytest.approx(-22.0)                 # the recorded move is still shown
    assert P.value_tone("history", row.value, row.findings) is None
    assert "different test" in row.tags


def test_a_like_for_like_move_keeps_its_colour_and_no_tag():
    x = f("recipe_change", evidence={"before": {"trim_pass_pct": 58.0},
                                     "after": {"trim_pass_pct": 80.0, "first": "2025-01-13"},
                                     "limit_table_changed": False, "limit_tables_mixed": False})
    row = rows_of(P.arrange([x]), "history")[0]
    assert P.value_tone("history", row.value, row.findings) == "up"
    assert "different test" not in row.tags


# ---- cut_setting merges only one TEST's tracks (final review, item 5) -----------------------------

def test_one_track_on_two_limit_tables_is_two_rows_never_both_tracks():
    rows = rows_of(P.arrange([cut(ev={"table": "t1"}), cut(ev={"table": "t2"})]), "yield")
    assert len(rows) == 2 and not any("both tracks" in r.tags for r in rows)
    assert len({r.key for r in rows}) == 2


def test_two_tracks_graded_on_different_tables_are_not_merged():
    rows = rows_of(P.arrange([cut(track="Track A", ev={"table": "t1"}),
                              cut(track="Track B", ev={"table": "t2"})]), "yield")
    assert len(rows) == 2


def test_two_tracks_on_the_same_table_still_merge():
    rows = rows_of(P.arrange([cut(track="Track A", ev={"table": "t1"}),
                              cut(track="Track B", ev={"table": "t1"})]), "yield")
    assert len(rows) == 1 and "both tracks" in rows[0].tags


def test_one_track_is_never_merged_with_itself_even_without_a_table():
    # A cache written after the track was stored but before the table was: nothing but the track
    # says these are one track on two tables -- which is enough never to call them "both tracks".
    rows = rows_of(P.arrange([cut(track="Track A"), cut(track="Track A")]), "yield")
    assert len(rows) == 2 and not any("both tracks" in r.tags for r in rows)


def test_a_table_the_model_moved_on_from_says_so_in_its_row():
    moved = cut(ev={"stale": True, "ran_on_laser_since": True, "last_ran": "2026-01-12", "table": "t1"})
    assert P.statement(moved) == ("Laser 1 (LTS): 6800 did better than 6900, Track A last ran on "
                                  "that limit table Jan 2026")
    gone = cut(ev={"stale": True, "ran_on_laser_since": False, "last_ran": "2025-04-02", "table": "t1"})
    assert P.statement(gone) == "Laser 1 (LTS): 6800 did better than 6900, last run Apr 2025"


def test_two_tracks_whose_rows_would_say_different_things_are_not_merged():
    moved = cut(track="Track A", ev={"stale": True, "ran_on_laser_since": True, "last_ran": "2026-01-12",
                                     "table": "t1"})
    gone = cut(track="Track B", ev={"stale": True, "ran_on_laser_since": False, "last_ran": "2026-01-12",
                                    "table": "t1"})
    assert len(rows_of(P.arrange([moved, gone]), "yield")) == 2
