import customtkinter as ctk


def _finding(model, title, units_per_year, volume=100):
    return {"model": model, "analyzer": "a", "category": "Ink target", "lever": "ink",
            "lever_label": "Ink formulation (incoming resistance)", "lead_time": "next lot",
            "title": title, "summary": "s",
            "expected_gain_points": None if units_per_year is None else 5.0,
            "units_per_year": units_per_year, "annual_volume": volume, "n_units": 300, "evidence": {}}


def _buttons(widget):
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkButton):
            out.append(c.cget("text"))
        out.extend(_buttons(c))
    return out


def test_findings_is_in_the_sidebar_right_after_investigate():
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar
    keys = [k for k, _ in Sidebar.ITEMS]
    assert ("findings", "Findings") in Sidebar.ITEMS
    assert keys.index("findings") == keys.index("model") + 1


def test_the_list_is_ranked_and_a_row_opens_the_model(make_app, monkeypatch):
    app = make_app()
    app.db.replace_process_findings("SMALL", {"tracks": 1}, [_finding("SMALL", "no gain here", None, volume=9000)])
    app.db.replace_process_findings("BIG", {"tracks": 1}, [_finding("BIG", "the big one", 500.0)])
    page = app.page_container.get_page("findings")
    page.reload_now()
    rows = _buttons(page)
    assert len(rows) == 2
    assert rows[0].startswith("BIG") and "500 units a year" in rows[0]        # a stated gain outranks volume
    assert rows[1].startswith("SMALL") and "no gain claimed" in rows[1]
    shown = []
    monkeypatch.setattr(app, "show_page", lambda name: shown.append(name))
    page._open("BIG")
    assert shown == ["model"] and app.consume_model_route() == "BIG"


def test_an_empty_cache_says_so(make_app):
    app = make_app()
    page = app.page_container.get_page("findings")
    page.reload_now()
    labels = [c.cget("text") for c in page._list.winfo_children() if isinstance(c, ctk.CTkLabel)]
    assert _buttons(page._list) == [] and any("No findings yet" in x for x in labels)
