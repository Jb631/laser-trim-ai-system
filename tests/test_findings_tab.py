import pytest
import customtkinter as ctk

FACTS = {"tracks": 2351, "yardstick": {"n": 2351, "agreement": 1.0, "faithful": True},
         "trim_effort": {"B": {"tracks_cut": 2351, "cuts": {"1": 1059, "2": 1263, "3+": 29},
                               "graded_untrimmed_n": 2351, "arrive_in_spec_pct": 4.6,
                               "in_limits_after_cut1_pct": 21.0, "multi_cut_n": 1292,
                               "multi_in_limits_after_first_pct": 18.0,
                               "multi_in_limits_after_last_pct": 52.0}},
         "recipe_history": [{"system": "B", "first": "2023-09-22", "last": "2024-06-30",
                             "recipe": "1 cut (cut length 2950)", "n": 817, "trim_pass_pct": 24.0,
                             "median_incoming_r": 4601.0}]}
FINDING = {"model": "HOT", "analyzer": "ink_target", "title": "Incoming resistance: aim lower",
           "category": "Ink target", "lever": "ink", "lever_label": "Ink formulation (incoming resistance)",
           "lead_time": "next lot", "expected_gain_points": 3.9, "tracks_per_year": 34.0,
           "annual_volume": 873, "summary": "Within Laser 1 (LTS) running 2 cuts…",
           "strength_name": "Spearman", "strength_value": -0.15, "n_units": 346, "evidence": {}}


def _texts(widget):
    out = []
    for c in widget.winfo_children():
        if isinstance(c, (ctk.CTkLabel, ctk.CTkButton)):
            out.append(c.cget("text"))
        out.extend(_texts(c))
    return [x for x in out if x]          # CTkScrollableFrame owns one empty label of its own


def _tab(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.findings_tab import FindingsTab
    return FindingsTab(tk_root, theme=ThemeManager())


def test_before_anything_is_computed_it_says_so(tk_root):
    texts = _texts(_tab(tk_root))
    assert len(texts) == 1 and "No process findings have been computed" in texts[0]


def test_no_findings_reads_as_nothing_to_act_on_not_as_a_gap(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": FACTS, "findings": []})
    text = " | ".join(_texts(tab))
    assert "Nothing to act on" in text
    assert "Laser 1 (LTS)" in text and "System B" not in text      # the shop's names, never the letters
    assert "18%" in text and "52%" in text and "Recipe history" in text


def test_findings_render_through_the_shared_view_not_a_card(tk_root):
    """The tab no longer draws its own finding cards (lever/lead-time/gain layout) -- that
    job moved to FindingsView (Task 7), already covered by tests/test_findings_view.py. Here
    we only need: findings replace "Nothing to act on", and the shared view's own group
    heading is the one that shows up."""
    tab = _tab(tk_root)
    tab.set_data({"facts": FACTS, "findings": [FINDING, {**FINDING, "title": "Recipe changed",
                                                         "tracks_per_year": None,
                                                         "expected_gain_points": None}]})
    text = " | ".join(_texts(tab))
    assert "Nothing to act on" not in text
    assert "Incoming resistance: aim lower" in text and "Recipe changed" in text
    assert "Change a setting to raise yield" in text


def test_a_model_the_yardstick_cannot_vouch_for_says_why(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": dict(FACTS, trim_effort=None,
                                yardstick={"n": 40, "agreement": 0.9, "faithful": False}),
                  "findings": []})
    assert any("not graded for this model" in x for x in _texts(tab))


def test_set_data_none_returns_to_the_empty_state(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": FACTS, "findings": [FINDING]})
    tab.set_data(None)
    texts = _texts(tab)
    assert len(texts) == 1 and "No process findings" in texts[0]


def test_the_model_page_loads_the_tab_from_the_cache(make_app):
    # `_seed` gives the model real analyses so the page has something to open;
    # tests/ is on sys.path, so a sibling test module's helper is importable.
    from test_spec3c_model import _seed
    app = make_app()
    _seed(app.db, "HOT", fails_last=12)
    app.db.replace_process_findings("HOT", FACTS, [FINDING])
    app.set_model_route("HOT")
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None       # suppress on_show's BACKGROUND reload (see test_spec3c_model)
    app.show_page("model")
    del page._reload
    page.reload_now()                      # the synchronous path
    assert "Incoming resistance: aim lower" in " | ".join(_texts(page._findings_tab))


# ---- appended by fix round 1: a value that is not there, and an analyzer that crashed ----

def test_a_crashed_analyzer_is_named_and_the_yardstick_is_not_blamed(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": dict(FACTS, trim_effort=None, recipe_history=None,
                                errors={"trim_effort": "RuntimeError: boom",
                                        "recipe_change": "KeyError: 'x'"}),
                  "findings": []})
    text = " | ".join(_texts(tab))
    assert "Could not be worked out this time" in text
    assert "what each cut buys" in text and "RuntimeError: boom" in text
    assert "the recipe history" in text and "KeyError" in text
    assert "not graded for this model" not in text      # the yardstick agreed 100% -- it is not the reason
    assert "Recipe history" not in text                 # None history: no heading, no crash


def test_a_thin_model_is_told_the_bar_not_only_100_percent(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": dict(FACTS, trim_effort=None,
                                yardstick={"n": 12, "agreement": 1.0, "faithful": False,
                                           "min_n": 30, "min_agreement": 0.99}),
                  "findings": []})
    text = " | ".join(_texts(tab))
    assert "not graded for this model" in text
    assert "at least 99% of at least 30 tracks" in text and "checked on 12 tracks" in text
    assert "only 100%" not in text


def test_a_model_with_no_trim_tracks_says_so(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": {"model": "EMPTY", "tracks": 0, "annual_volume": 0, "latest": None,
                            "yardstick": None, "recipe_history": None, "trim_effort": None, "errors": {}},
                  "findings": []})
    text = " | ".join(_texts(tab))
    assert "No laser trim tracks are stored" in text and "not graded" not in text


def test_a_value_that_is_not_there_reads_as_a_dash_never_as_zero_and_never_crashes(tk_root):
    tab = _tab(tk_root)
    effort = {"B": {"tracks_cut": None, "cuts": None, "graded_untrimmed_n": None,
                    "arrive_in_spec_pct": None, "in_limits_after_cut1_pct": None, "multi_cut_n": None}}
    history = [{"system": "B", "first": None, "last": None, "recipe": None, "n": None,
                "trim_pass_pct": None, "median_incoming_r": None}]
    tab.set_data({"facts": dict(FACTS, trim_effort=effort, recipe_history=history),
                  "findings": [{**FINDING, "n_units": None, "expected_gain_points": None}]})
    text = " | ".join(_texts(tab))
    assert "Recipe history" in text                     # it rendered all the way to the last section
    assert "median incoming — Ω" in text and "0 Ω" not in text
    assert "— tracks cut" in text and "()" not in text
    assert "Incoming resistance: aim lower" in text      # the finding itself still rendered, via the shared view
    assert "None" not in text


def test_more_than_one_limit_table_gets_its_own_section_and_one_does_not(tk_root):
    tab = _tab(tk_root)
    two = [{"system": "B", "track": "Track A", "rows": 111, "graded": 89, "n": 1649, "first": "2023-09-22",
            "last": "2026-01-12", "trim_pass_pct": 34.0},
           {"system": "B", "track": "Track A", "rows": 57, "graded": 45, "n": 823, "first": "2023-10-05",
            "last": "2026-09-11", "trim_pass_pct": None}]
    tab.set_data({"facts": dict(FACTS, limit_tables=two), "findings": []})
    text = " | ".join(_texts(tab))
    assert "Limit tables" in text and "89 graded points of 111 rows" in text and "1,649 tracks" in text
    assert "45 graded points of 57 rows" in text and "— left the laser inside limits" in text
    assert "Laser 1 (LTS)" in text and "System B" not in text
    tab.set_data({"facts": dict(FACTS, limit_tables=two[:1]), "findings": []})
    assert "Limit tables" not in " | ".join(_texts(tab))
    tab.set_data({"facts": dict(FACTS, limit_tables=None, errors={"limit_tables": "RuntimeError: bad table"}),
                  "findings": []})
    text = " | ".join(_texts(tab))
    assert "the limit tables" in text and "RuntimeError: bad table" in text


def test_fact_sections_draw_their_heading_through_group_header_with_a_matching_count(tk_root):
    """Step 1(c) (design doc 2026-09-24-facelift-step2-pages-design.md §1 item 6): the four
    enumerable fact sections -- limit tables, the pass-burden group, the cut-setting group,
    recipe history -- draw their heading through blocks.group_header (the same block the
    Findings page's own groups use), not the tab's bare _heading() label, so the row count
    sits next to the title exactly like every other list in the app.

    group_header's shape is title-label + count-pill-label as SIBLINGS inside a small "top"
    frame nested under a "wrap" frame; _heading()'s old shape was a single bare label packed
    straight into the tab's scrollable body. Checking the count sits next to the title (not
    merely "somewhere in the tab") and that the title's parent is no longer the body itself
    tells the two apart structurally -- it fails on the old bare-label code, not only on
    missing text."""
    tab = _tab(tk_root)
    two_tables = [{"system": "B", "track": "Track A", "rows": 111, "graded": 89, "n": 1649,
                   "first": "2023-09-22", "last": "2026-01-12", "trim_pass_pct": 34.0},
                  {"system": "B", "track": "Track A", "rows": 57, "graded": 45, "n": 823,
                   "first": "2023-10-05", "last": "2026-09-11", "trim_pass_pct": None}]
    burden = {"B": {"n": 40, "normal_cuts": 1, "share_over_recipe": 12.0,
                    "unplanned_passes": 5, "unplanned_passes_per_100_tracks": 12.5}}
    cuts = {"B": {"n": 40, "window": "2023-09 → 2026-09", "current_setting": 2950,
                  "days_with_more_than_one_setting_pct": None,
                  "settings": [{"setting": 2950, "n": 40, "pass_pct": 34.0,
                                "median_incoming_resistance": 4600.0, "window": "…"}]}}
    tab.set_data({"facts": dict(FACTS, limit_tables=two_tables, pass_burden=burden, cut_setting=cuts),
                  "findings": []})

    def _count_beside(heading_text, expected_count):
        label = None
        def walk(w):
            nonlocal label
            for c in w.winfo_children():
                if isinstance(c, ctk.CTkLabel) and c.cget("text") == heading_text:
                    label = c
                walk(c)
        walk(tab)
        assert label is not None, f"no heading {heading_text!r} found"
        assert label.master is not tab._body, (
            f"{heading_text!r} is packed straight into the body -- still a bare _heading() label")
        siblings = [c.cget("text") for c in label.master.winfo_children() if isinstance(c, ctk.CTkLabel)]
        assert f"{expected_count:,}" in siblings, (heading_text, siblings)

    _count_beside("Limit tables this model has been graded against", len(two_tables))
    _count_beside("Cuts the recipe did not ask for (last year)", len(burden))
    _count_beside("Cut settings this model has been run at", len(cuts))
    _count_beside("Recipe history", len(FACTS["recipe_history"]))


def test_the_tab_draws_findings_with_the_same_groups_as_the_page(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.findings_tab import FindingsTab
    tab = FindingsTab(tk_root, ThemeManager())
    cut = {"analyzer": "cut_setting", "model": "6607", "category": "Cut setting", "title": "t",
           "summary": "s", "systems": ["B"], "n_units": 1, "tracks_per_year": 182.0,
           "evidence": {"best": 6800.0, "current": 6900.0, "grade": "two_periods", "track": "Track A"}}
    tab.set_data({"facts": {"tracks": 10, "errors": {}}, "findings": [cut]})
    texts = _texts(tab)
    assert "Change a setting to raise yield" in texts
    assert "What changed" not in texts                  # empty groups hidden on one model's tab
    assert not any(x.startswith("Open ") for x in texts)  # already on the model



def test_a_section_heading_is_no_smaller_than_the_group_headings_inside_it(tk_root):
    """Final review, 2026-09-24: the tab's section headings ("What to do about it") were caption
    size, smaller than the FindingsView group headers (SIZE_HEADING) drawn inside them."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    tab = _tab(tk_root)
    tab.set_data({"facts": FACTS, "findings": [FINDING]})

    def size_of(text):
        found = []

        def walk(w):
            for c in w.winfo_children():
                if isinstance(c, ctk.CTkLabel) and c.cget("text") == text:
                    found.append(c.cget("font").cget("size"))
                walk(c)
        walk(tab)
        assert found, f"no label reads {text!r}"
        return found[0]

    group = size_of("Change a setting to raise yield")
    assert group == ThemeManager().SIZE_HEADING
    for section in ("What was measured", "What to do about it", "Recipe history"):
        assert size_of(section) >= group, section


# ---- facelift F4 (2026-09-25): a FAILED load is its own state on the tab too ------------------
# The Model page's "Worth changing" section already had three states (failed / not computed /
# empty); the tab did not -- a crashed facts load drew "No process findings have been computed
# for this model yet..." under the banner that named the crash.

def test_a_failed_load_says_it_failed_never_not_computed(tk_root):
    from laser_trim_analyzer.gui.v6.widgets.findings_tab import NOT_COMPUTED_TEXT
    tab = _tab(tk_root)
    tab.set_data({"facts": FACTS, "findings": [FINDING]})       # a good load first
    tab.set_data(None, failed="RuntimeError: invented database crash")
    text = " | ".join(_texts(tab))
    assert NOT_COMPUTED_TEXT not in text
    assert "could not be loaded" in text and "RuntimeError: invented database crash" in text
    assert "Nothing to act on" not in text and "Incoming resistance" not in text   # no stale rows
    def labels(w):
        for c in w.winfo_children():
            if isinstance(c, ctk.CTkLabel):
                yield c
            yield from labels(c)
    banners = [w for w in labels(tab._body) if w.cget("fg_color") == tab.theme.CHECK_TINT]
    assert banners, "a failure is a check-tone banner, never a quiet line"


def test_the_three_states_follow_each_other_and_recover(tk_root):
    from laser_trim_analyzer.gui.v6.widgets.findings_tab import NOT_COMPUTED_TEXT
    tab = _tab(tk_root)
    tab.set_data(None, failed="OperationalError: database is locked")
    assert "could not be loaded" in " | ".join(_texts(tab))
    tab.set_data(None)                                             # not computed
    assert _texts(tab) == [NOT_COMPUTED_TEXT]
    tab.set_data({"facts": FACTS, "findings": []})                 # computed, nothing found
    text = " | ".join(_texts(tab))
    assert "Nothing to act on" in text and "could not be loaded" not in text


# ---- I3 (final review, 2026-09-25): the tab shows the five new facts keys ----------------------
# Example numbers are invented.

LOSS = {"Laser 1 (LTS)": {"n": 1290, "fails": 465, "auc_error": 0.736, "auc_resistance": 0.61,
                          "limit_table": {"key": "k1", "graded_points": 57, "tracks": 1290},
                          "other_tables_n": 40},
        "Laser 2 (DLTS)": {"n": 12, "fails": 0, "auc_error": None, "auc_resistance": None,
                           "limit_table": {"key": "k2", "graded_points": 45, "tracks": 12},
                           "other_tables_n": 0}}
REWORK = {"linked": 1300, "rework_unit_days": 300, "skipped_pairs": 3, "junk_readings": 0,
          "unpaired_final_tests": 0, "reduction": "r", "confirmed": True,
          "by_laser": {
              "Laser 1 (LTS)": {"rework_unit_days": 261, "rework_ratio_n": 244, "control_n": 415,
                                "control_top_third_n": 139, "control_top_third_min_laser_error": 0.05,
                                "mann_whitney_u": 1.0, "p_value": 5.2e-06, "median_ratio_rework": 0.41,
                                "median_ratio_control_top_third": 0.51, "effect_ratio": 0.805,
                                "confirmed": True},
              "Laser 2 (DLTS)": {"rework_unit_days": 39, "rework_ratio_n": 12, "control_n": 5,
                                 "control_top_third_n": 2, "control_top_third_min_laser_error": None,
                                 "mann_whitney_u": None, "p_value": None, "median_ratio_rework": 0.5,
                                 "median_ratio_control_top_third": None, "effect_ratio": None,
                                 "confirmed": False, "note": "12 reworked unit-days and 2 comparable "
                                 "pass/pass units could be read -- the test needs 30 and 20"}}}
STATION = {"status": "differs", "pct_positions_differing": 1.0, "matched_positions": 225,
           "trim_typ_band": 0.05, "ft_typ_band": 0.03, "sampled_lasers": ["Laser 1 (LTS)"],
           "differing_ratio": 1.7, "differing_wider_share": 1.0,
           "note": "100% of the positions both stations measure are graded to different limits "
                   "(trim ±0.050 V, final test ±0.030 V)"}
MACHINE = {"window": {"first": "2024-10", "last": "2026-09", "months": 24, "anchored_to": "2026-09-22"},
           "comparisons": [
               {"table": "t1", "graded_points": 31, "in_window": False, "months": ["2023-07", "2024-09"],
                "by_laser": {"Laser 1 (LTS)": {"n": 480, "pass_pct": 75.2},
                             "Laser 2 (DLTS)": {"n": 314, "pass_pct": 99.1}}},
               {"table": "t1", "graded_points": 31, "in_window": True, "months": ["2024-10", "2025-02"],
                "by_laser": {"Laser 1 (LTS)": {"n": 130, "pass_pct": 82.3},
                             "Laser 2 (DLTS)": {"n": 174, "pass_pct": 97.7}}}]}
NEW = dict(FACTS, loss_origin=LOSS, rework_load=REWORK, station_setup=STATION,
           machine_compare=MACHINE, setup_change=[], errors={})


def test_every_analyzer_the_engine_runs_has_a_name_a_person_would_say():
    from laser_trim_analyzer.findings.engine import compute_for_model
    from laser_trim_analyzer.gui.v6.widgets.findings_tab import analyzer_name
    from findings_helpers import START
    import inspect
    names = {"recipe_change", "setup_change", "ink_target", "limit_tables", "cut_setting",
             "pass_burden", "machine_compare", "loss_origin", "station_setup", "rework_load",
             "trim_effort"}
    src = inspect.getsource(compute_for_model)
    assert all(f'failed("{n}"' in src for n in names)          # the list is the engine's own
    for n in names:
        assert analyzer_name(n) != n and "_" not in analyzer_name(n), n


def test_loss_origin_sits_last_beside_the_predictors_own_auc(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": NEW, "findings": [], "predictor_auc": 0.78})
    texts = _texts(tab)
    text = " | ".join(texts)
    assert "Laser 1 (LTS) · 1,290 tracks, 465 failed at the laser · AUC 0.74 from incoming " \
           "linearity, 0.61 from incoming resistance" in text
    assert "on the 57-point limit table (40 tracks on other tables left out)" in text
    assert "Laser 2 (DLTS) · 12 tracks, 0 failed at the laser · AUC — from incoming linearity, " \
           "— from incoming resistance" in text                    # None is a dash, never 0
    assert any("predictor's own AUC" in x and "0.78" in x for x in texts)
    heading = texts.index("Where the loss is made (last year)")
    assert all(not x.startswith(("Hand trim", "Two lasers", "Laser and final test"))
               for x in texts[heading:])                          # the last section, above the panel


def test_no_trained_predictor_and_an_unreadable_one_are_told_apart(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": NEW, "findings": [], "predictor_auc": None})
    assert any("No final-test predictor is trained" in x for x in _texts(tab))
    tab.set_data({"facts": NEW, "findings": [], "predictor_auc_error": "OperationalError: locked"})
    text = " | ".join(_texts(tab))
    assert "could not be read" in text and "OperationalError: locked" in text
    assert "No final-test predictor is trained" not in text


def test_rework_says_its_verdict_per_laser_with_both_sizes_p_and_the_effect(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": NEW, "findings": []})
    text = " | ".join(_texts(tab))
    assert "Hand trim after a laser fail (last year)" in text
    assert "1,300 final tests linked" in text and "300 unit-days" in text
    assert ("Laser 1 (LTS) · confirmed as hand trim · 244 reworked unit-days against the 139 "
            "untouched units with the largest laser errors (of 415)") in text
    assert "effect 0.81" in text and "p = 5.2e-06" in text
    assert "Laser 2 (DLTS) · not confirmed: 12 reworked unit-days and 2 comparable" in text
    assert "effect —" in text and "p = —" in text


def test_station_setup_shows_its_note(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": NEW, "findings": []})
    text = " | ".join(_texts(tab))
    assert "Laser and final test limits" in text
    assert "Laser 1 (LTS) against final test: 100% of the positions both stations measure" in text
    assert "225 positions" in text


def test_machine_compare_shows_its_per_table_rates_and_dates_the_old_ones(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": NEW, "findings": []})
    text = " | ".join(_texts(tab))
    assert "Two lasers on the same test" in text
    assert ("Oct 2024 – Feb 2025 · 31-point limit table · Laser 1 (LTS) 82% of 130 · "
            "Laser 2 (DLTS) 98% of 174") in text
    assert ("Jul 2023 – Sep 2024 · 31-point limit table · Laser 1 (LTS) 75% of 480 · Laser 2 "
            "(DLTS) 99% of 314 · before the 24 months to Sep 2026, so not a finding") in text


# ---- facts cached by an older version (controller ruling, 2026-09-25) ---------------------------
# At work every model's facts were written by the code before this pull, and only the models an
# ingest touches are refreshed -- so the tab WILL meet facts keys that are absent (the work
# database's 319 rows hold none of the five new ones) or in an older shape (57fdcdb's, the close-out
# copy). Such a key must never crash, and never show a number: it reads as not worked out.
# Captured read-only on 2026-09-25: tests/fixtures/findings_cache/before_this_version.json.

import json as _json
from pathlib import Path as _Path

CACHE = _json.loads((_Path(__file__).resolve().parent / "fixtures" / "findings_cache"
                     / "before_this_version.json").read_text())
NOT_WORKED_OUT = "Not worked out yet by this version"


def _cached(version, model):
    return {"facts": CACHE[version]["facts"][model], "findings": CACHE[version]["findings"][model]}


def test_the_captured_blobs_are_the_shapes_they_claim():
    pre = CACHE["pre_pull"]["facts"]["6607"]
    assert not {"machine_compare", "loss_origin", "station_setup", "rework_load",
                "setup_change"} & set(pre)
    old = CACHE["fix_57fdcdb"]["facts"]
    assert "by_laser" not in old["6607"]["rework_load"]                      # flat, pooled lasers
    assert "limit_table" not in old["6607"]["loss_origin"]["Laser 1 (LTS)"]    # tables pooled
    assert "sampled_lasers" not in old["6607"]["station_setup"]
    assert "comparisons" not in old["6126"]["machine_compare"]                 # keyed by table
    assert "before" in old["6607"]["setup_change"][0]                          # per-key runs


def test_facts_from_before_the_pull_read_as_not_worked_out_and_the_rest_still_renders(tk_root):
    tab = _tab(tk_root)
    tab.set_data(_cached("pre_pull", "6607"))
    text = " | ".join(_texts(tab))
    assert (f"{NOT_WORKED_OUT} — the station limits comparison, the laser comparison, the rework "
            "count and where the loss is made. Refresh findings (Settings) to see them.") in text
    for heading in ("Laser and final test limits", "Two lasers on the same test",
                    "Hand trim after a laser fail", "Where the loss is made"):
        assert heading not in text
    assert "Recipe history" in text                     # what this version reads the same way


def test_facts_from_the_pre_fix_code_never_show_their_numbers(tk_root):
    tab = _tab(tk_root)
    tab.set_data(_cached("fix_57fdcdb", "6607"))
    text = " | ".join(_texts(tab))
    assert (f"{NOT_WORKED_OUT} — the station limits comparison, the rework count and where the "
            "loss is made. Refresh findings (Settings) to see them.") in text
    old = CACHE["fix_57fdcdb"]["facts"]["6607"]
    assert f"{old['rework_load']['rework_ratio_n']} reworked" not in text
    # The facts LINE (the cached loss_origin FINDING's own title may still say "AUC 0.74" in
    # "What to do about it": it is a finding row, and the same call stands in this version).
    assert "from incoming linearity" not in text and "from incoming resistance" not in text
    assert old["station_setup"]["note"] not in text
    assert "Two lasers on the same test" not in text    # {} is "nothing comparable" in both shapes


def test_a_table_keyed_machine_comparison_never_shows_its_rates(tk_root):
    tab = _tab(tk_root)
    tab.set_data(_cached("fix_57fdcdb", "6126"))
    text = " | ".join(_texts(tab))
    assert "the laser comparison" in text and NOT_WORKED_OUT in text
    (table,) = CACHE["fix_57fdcdb"]["facts"]["6126"]["machine_compare"].values()
    for laser, rate in table["by_laser"].items():
        assert f"{laser} {rate['pass_pct']:.0f}% of {rate['n']:,}" not in text


def test_current_facts_say_nothing_about_an_older_version(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": NEW, "findings": []})
    assert NOT_WORKED_OUT not in " | ".join(_texts(tab))


def test_a_model_with_no_trim_tracks_is_not_told_its_facts_are_old(tk_root):
    """compute_for_model returns before any analyzer for a model with no tracks: its new keys are
    None because there was nothing to work out, not because an older version wrote them."""
    tab = _tab(tk_root)
    tab.set_data({"facts": {"tracks": 0, "errors": {}, "machine_compare": None, "loss_origin": None,
                            "station_setup": None, "rework_load": None, "setup_change": None},
                  "findings": []})
    text = " | ".join(_texts(tab))
    assert "No laser trim tracks are stored" in text and NOT_WORKED_OUT not in text


def test_the_model_pages_worth_changing_section_reads_older_caches(make_app):
    """Its other reader of the facts: it reads only whether facts exist and their errors."""
    from test_spec3c_model import _seed
    app = make_app()
    _seed(app.db, "HOT", fails_last=12)
    page = app.page_container.get_page("model")
    for version, model in (("pre_pull", "6607"), ("fix_57fdcdb", "6607"), ("fix_57fdcdb", "6126")):
        page._set_findings_section(_cached(version, model), [])
        texts = _texts(page._worth_section)
        assert "Worth changing on this model" in texts
        assert not any("Could not be worked out" in x for x in texts)


def test_the_errors_every_screen_reads_come_through_from_older_caches(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    db = DatabaseManager(tmp_path / "old_cache.db")
    for version in ("pre_pull", "fix_57fdcdb"):
        for model, facts in CACHE[version]["facts"].items():
            db.replace_process_findings(f"{version}-{model}", dict(facts, errors={}), [])
    db.replace_process_findings("BROKE", dict(CACHE["pre_pull"]["facts"]["6607"],
                                              errors={"cut_setting": "RuntimeError: x"}), [])
    assert db.get_process_errors() == {"BROKE": {"cut_setting": "RuntimeError: x"}}


def _sum_rates(findings):
    rates = [f["tracks_per_year"] for f in findings if f.get("tracks_per_year") is not None]
    return sum(rates) if rates else None


def test_the_presentation_layer_and_both_counts_read_findings_cached_by_older_versions():
    """Findings rows cached by the code before this pull (the six analyzers it had) and by 57fdcdb
    (the new analyzers' pre-fix evidence): arranged without a crash, and every row's number is the
    one its own evidence holds -- never a 0 or a blank made up for a field it lacks."""
    from laser_trim_analyzer.findings import presentation as P
    from laser_trim_analyzer.gui.v6.pages.home_page import _yield_findings_count
    from laser_trim_analyzer.gui.v6.pages.model_page import _worth_changing_count
    for version in ("pre_pull", "fix_57fdcdb"):
        for model, findings in CACHE[version]["findings"].items():
            groups = P.arrange(findings, include_empty=False)
            rows = [r for g in groups for r in g.rows]
            assert sum(len(r.findings) for r in rows) == len(findings)
            for g in groups:
                for r in g.rows:
                    assert r.statement, (version, model)
                    f, ev = r.findings[0], r.findings[0].get("evidence") or {}
                    key = g.spec.key
                    if key == "yield":            # a rate only where the finding claims one: "—" else
                        assert r.value == _sum_rates(r.findings)
                    elif key == "laser_time":     # the readout's own field, never the n_units fallback
                        field = P._LASER_TIME_FIELD[f["category"]]
                        assert r.value == sum(x["evidence"]["facts"][field] for x in r.findings)
                    elif key == "history":        # the move between the two sides it recorded
                        assert r.value == pytest.approx(ev["after"]["trim_pass_pct"]
                                                        - ev["before"]["trim_pass_pct"])
                    else:
                        assert r.value == sum(x["n_units"] for x in r.findings)
                    shown = P.value_text(key, r.value, r.findings)
                    assert shown == "—" if r.value is None else shown not in ("", "—")
            assert _yield_findings_count(findings) == sum(
                len(g.rows) for g in groups if g.spec.key == "yield")
            assert _worth_changing_count(findings) == sum(
                len(g.rows) for g in groups if g.spec.key in ("yield", "laser_time", "check"))


def test_a_crashed_new_analyzer_is_named_and_its_section_is_not_drawn(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": dict(NEW, rework_load=None, errors={"rework_load": "RuntimeError: locked"}),
                  "findings": []})
    text = " | ".join(_texts(tab))
    assert "Could not be worked out this time — the rework count (RuntimeError: locked)" in text
    assert "Hand trim after a laser fail" not in text
    assert "Where the loss is made (last year)" in text           # the rest still drawn


def test_the_model_page_puts_the_stored_predictor_auc_beside_loss_origin(make_app):
    from test_spec3c_model import _seed
    from laser_trim_analyzer.database.models import ModelMLState
    app = make_app()
    _seed(app.db, "HOT", fails_last=12)
    app.db.replace_process_findings("HOT", NEW, [FINDING])
    with app.db.session() as s:
        s.add(ModelMLState(model="HOT", predictor_trained=True, predictor_auc=0.8123))
    app.set_model_route("HOT")
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None
    app.show_page("model")
    del page._reload
    page.reload_now()
    assert any("predictor's own AUC" in x and "0.81" in x for x in _texts(page._findings_tab))
