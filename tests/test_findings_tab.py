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
           "lead_time": "next lot", "expected_gain_points": 3.9, "units_per_year": 34.0,
           "annual_volume": 873, "summary": "Within Laser 1 (LTS) running 2 cuts…",
           "strength_name": "Spearman", "strength_value": -0.15, "n_units": 346, "evidence": {}}


def _texts(widget):
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
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
    assert "18%" in text and "52%" in text and "RECIPE HISTORY" in text


def test_a_finding_shows_its_lever_lead_time_and_gain(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": FACTS, "findings": [FINDING, {**FINDING, "title": "Recipe changed",
                                                         "units_per_year": None,
                                                         "expected_gain_points": None}]})
    text = " | ".join(_texts(tab))
    assert "Nothing to act on" not in text
    assert "34 units a year" in text and "no gain claimed" in text and "next lot" in text


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
