import customtkinter as ctk


def _finding(model, title, tracks_per_year, size=100):
    return {"model": model, "analyzer": "a", "category": "Ink target", "lever": "ink",
            "lever_label": "Ink formulation (incoming resistance)", "lead_time": "next lot",
            "title": title, "summary": "s",
            "expected_gain_points": None if tracks_per_year is None else 5.0,
            "tracks_per_year": tracks_per_year, "annual_volume": size, "n_units": size, "evidence": {}}


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
    app.db.replace_process_findings("SMALL", {"tracks": 1}, [_finding("SMALL", "no rate here", None, size=9000)])
    app.db.replace_process_findings("BIG", {"tracks": 1}, [_finding("BIG", "the big one", 500.0)])
    page = app.page_container.get_page("findings")
    page.reload_now()
    rows = _buttons(page)
    assert len(rows) == 2
    assert rows[0].startswith("BIG") and "500 tracks a year" in rows[0]        # a stated rate outranks sample size
    assert rows[1].startswith("SMALL") and "no rate claimed" in rows[1]
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


def _labels(page):
    return [c.cget("text") for c in page._list.winfo_children() if isinstance(c, ctk.CTkLabel)]


def test_a_load_failure_is_an_error_not_an_empty_list(make_app, monkeypatch):
    app = make_app()
    page = app.page_container.get_page("findings")

    def boom():
        raise RuntimeError("database is locked")
    monkeypatch.setattr(app.db, "get_process_findings", boom)
    page.reload_now()
    text = " | ".join(_labels(page))
    assert "could not be loaded" in text and "RuntimeError: database is locked" in text
    assert "No findings yet" not in text


def test_models_whose_analyzers_failed_are_named_under_the_list(make_app):
    app = make_app()
    app.db.replace_process_findings("BIG", {"tracks": 1, "errors": {}}, [_finding("BIG", "the big one", 500.0)])
    app.db.replace_process_findings("HURT", {"tracks": 9, "errors": {"trim_effort": "ValueError: x"}}, [])
    page = app.page_container.get_page("findings")
    page.reload_now()
    assert len(_buttons(page)) == 1
    text = " | ".join(_labels(page))
    assert "1 model(s) could not be fully worked out" in text and "HURT" in text


def test_a_healthy_cache_says_nothing_about_failures(make_app):
    app = make_app()
    app.db.replace_process_findings("BIG", {"tracks": 1, "errors": {}}, [_finding("BIG", "the big one", 500.0)])
    page = app.page_container.get_page("findings")
    page.reload_now()
    assert not any("could not be" in x for x in _labels(page))


def test_a_failure_of_the_secondary_read_does_not_throw_away_the_list(make_app, monkeypatch):
    """Round B review: the list and "which models failed" are two separate reads. When only the
    second one fails, the findings the first one returned are still true -- show them, and say that
    the OTHER thing is unknown. (Before: one try block, so a good list was replaced by an error page.)"""
    app = make_app()
    app.db.replace_process_findings("BIG", {"tracks": 1, "errors": {}}, [_finding("BIG", "the big one", 500.0)])

    def boom():
        raise RuntimeError("database is locked")
    monkeypatch.setattr(app.db, "get_process_errors", boom)
    page = app.page_container.get_page("findings")
    page.reload_now()
    rows = _buttons(page)
    assert len(rows) == 1 and rows[0].startswith("BIG")
    text = " | ".join(_labels(page))
    assert "Findings could not be loaded" not in text
    assert "could not be checked" in text and "RuntimeError: database is locked" in text
