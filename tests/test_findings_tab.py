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
