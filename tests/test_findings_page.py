"""The Findings page: every model's findings in four groups (Task 7 rebuild on FindingsView).

Same behaviours as the old flat ranked list -- a failed load reads as an error, an empty
cache says so, model-level errors are still named -- now read from page._notices (banners)
and page._view (the shared FindingsView) instead of a column of CTkButtons.
"""
import customtkinter as ctk


def _finding(model, title, tracks_per_year, size=100, analyzer="ink_target", category="Ink target"):
    # "lever" is a required column on ProcessFinding (database/manager.replace_process_findings
    # reads it as d["lever"], not .get) -- every persisted finding needs one even though the
    # Findings view itself never shows it.
    return {"model": model, "analyzer": analyzer, "category": category, "lever": "ink",
            "title": title, "summary": f"summary for {model}", "systems": ["B"], "n_units": size,
            "tracks_per_year": tracks_per_year, "evidence": {}}


def _texts(widget):
    out = []
    for c in widget.winfo_children():
        if isinstance(c, (ctk.CTkLabel, ctk.CTkButton)):
            out.append(c.cget("text"))
        out.extend(_texts(c))
    return out


def _labels(widget):
    """Recursive CTkLabel text -- used on page._notices, which only ever holds banners."""
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
            out.append(c.cget("text"))
        out.extend(_labels(c))
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
    # Both findings are "Ink target" -> the "yield" group; a stated rate (BIG, 500/yr)
    # outranks a row with none (SMALL, sample size only) -- same ordering rule as before,
    # now inside the group rather than across one flat list.
    assert len(page._view.row_widgets) == 2
    texts = _texts(page._view)
    assert texts.index("BIG") < texts.index("SMALL")
    assert "~500" in texts                  # BIG's readout: a stated rate
    assert "—" in texts                     # SMALL's readout: no rate claimed
    caption = page._caption.cget("text")
    assert "changes worth testing" in caption
    shown = []
    monkeypatch.setattr(app, "show_page", lambda name: shown.append(name))
    page._open("BIG")
    assert shown == ["model"]
    assert app.consume_model_route() == "BIG"
    assert app.consume_model_tab() == "findings"        # opens straight onto the model's Findings tab


def test_an_empty_cache_says_so(make_app):
    app = make_app()
    page = app.page_container.get_page("findings")
    page.reload_now()
    assert page._view.row_widgets == {}
    # winfo_manager(), not winfo_ismapped(): make_app WITHDRAWS the window, so nothing in it is ever
    # mapped and "not mapped" was true whether or not the view was hidden (final review: with
    # pack_forget made a no-op both tests still passed -- and the page then showed four empty groups
    # saying "Nothing here yet" under the error banner, a failure drawn as a result).
    assert page._view.winfo_manager() == ""
    assert any("No findings yet" in x for x in _labels(page._notices))
    assert page._caption.cget("text") == ""


def test_a_load_failure_is_an_error_not_an_empty_list(make_app, monkeypatch):
    app = make_app()
    page = app.page_container.get_page("findings")

    def boom():
        raise RuntimeError("database is locked")
    monkeypatch.setattr(app.db, "get_process_findings", boom)
    page.reload_now()
    text = " | ".join(_labels(page._notices))
    assert "could not be loaded" in text and "RuntimeError: database is locked" in text
    assert "No findings yet" not in text
    assert page._view.winfo_manager() == ""          # hidden -- see test_an_empty_cache_says_so


def test_a_good_load_after_a_failed_one_shows_the_list_again(make_app, monkeypatch):
    """The failed load hid the view; the next good load must put it back (findings_page._apply
    re-packs it only when it is not laid out -- winfo_manager(), which a withdrawn test window
    cannot fake the way it fakes "not mapped")."""
    app = make_app()
    app.db.replace_process_findings("BIG", {"tracks": 1, "errors": {}}, [_finding("BIG", "the big one", 500.0)])
    page = app.page_container.get_page("findings")
    real = app.db.get_process_findings

    def boom():
        raise RuntimeError("database is locked")
    monkeypatch.setattr(app.db, "get_process_findings", boom)
    page.reload_now()
    assert page._view.winfo_manager() == ""
    monkeypatch.setattr(app.db, "get_process_findings", real)
    page.reload_now()
    assert page._view.winfo_manager() == "pack"
    assert len(page._view.row_widgets) == 1
    assert not any("could not be loaded" in x for x in _labels(page._notices))


def test_models_whose_analyzers_failed_are_named_under_the_list(make_app):
    app = make_app()
    app.db.replace_process_findings("BIG", {"tracks": 1, "errors": {}}, [_finding("BIG", "the big one", 500.0)])
    app.db.replace_process_findings("HURT", {"tracks": 9, "errors": {"trim_effort": "ValueError: x"}}, [])
    page = app.page_container.get_page("findings")
    page.reload_now()
    assert len(page._view.row_widgets) == 1
    text = " | ".join(_labels(page._notices))
    assert "1 model(s) could not be fully worked out" in text and "HURT" in text


def test_a_healthy_cache_says_nothing_about_failures(make_app):
    app = make_app()
    app.db.replace_process_findings("BIG", {"tracks": 1, "errors": {}}, [_finding("BIG", "the big one", 500.0)])
    page = app.page_container.get_page("findings")
    page.reload_now()
    assert not any("could not be" in x for x in _labels(page._notices))


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
    assert len(page._view.row_widgets) == 1
    notices = " | ".join(_labels(page._notices))
    assert "Findings could not be loaded" not in notices
    assert "could not be checked" in notices and "RuntimeError: database is locked" in notices


def test_all_four_groups_show_even_when_only_one_has_rows(make_app):
    """Spec section 3 'States': all four groups are always on the page; an empty one says
    what would fill it."""
    app = make_app()
    app.db.replace_process_findings("BIG", {"tracks": 1, "errors": {}}, [_finding("BIG", "the big one", 500.0)])
    page = app.page_container.get_page("findings")
    page.reload_now()
    texts = _texts(page._view)
    assert "Change a setting to raise yield" in texts
    assert "Laser time you could save" in texts
    assert "Check the test" in texts
    assert "What changed" in texts
    assert any(x.startswith("No recipe or setting changes found") for x in texts)   # the empty "What changed" group
