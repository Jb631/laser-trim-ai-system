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


# ---- F4 review (out of scope 1): an older load never overwrites a newer one --------------------
# Home, Triage and the Model page drop a load that is no longer the newest; the Findings page did
# not, so a stale "Findings could not be loaded" could land over a newer, healthy list.

import pytest


@pytest.mark.parametrize("newer", ("async", "sync"))
def test_an_older_findings_page_load_never_overwrites_a_newer_one(make_app, monkeypatch, newer):
    from test_spec3f_home import _older_then_newer, _pump_ui, _pump_until, _settle_workers
    app = make_app()
    page = app.page_container.get_page("findings")
    _settle_workers(app)
    load = _older_then_newer(
        lambda: {"rows": [], "errors": {}, "failed": "RuntimeError: invented findings crash",
                 "errors_failed": None},                                  # older: crashed
        lambda: {"rows": [], "errors": {}, "failed": None, "errors_failed": None})  # newer: healthy
    monkeypatch.setattr(page, "_query", load)
    page.on_show()                             # the older load, still in its query...
    load.started()
    if newer == "async":                       # ...when a newer one starts and finishes first
        page.on_show()
        assert _pump_until(app, lambda: any("No findings yet" in x for x in _labels(page._notices)))
    else:
        page.reload_now()
    load.release()                             # the older load finishes LAST
    _pump_ui(app)
    notices = " | ".join(_labels(page._notices))
    assert "could not be loaded" not in notices, "an older load overwrote a newer one"
    assert "No findings yet" in notices


# ---- F5 (2026-09-25): "Inactive" models -- labelled, never hidden ------------------------------

def _seed_inactive(app):
    """LIVE trimmed at the fleet's newest file; OLD last trimmed 900 days before it."""
    from datetime import timedelta
    from test_model_activity import NEWEST, _file
    _file(app.db, "LIVE", NEWEST)
    old_last = NEWEST - timedelta(days=900)
    _file(app.db, "OLD", old_last)
    return f"Inactive · last trimmed {old_last:%b %Y}"


def test_an_inactive_models_finding_is_tagged_listed_after_the_active_ones_and_still_counted(make_app):
    app = make_app()
    tag = _seed_inactive(app)
    app.db.replace_process_findings("OLD", {"tracks": 1, "errors": {}},
                                    [_finding("OLD", "the old one", 900.0)])
    for i in range(5):
        app.db.replace_process_findings(f"LIVE{i}", {"tracks": 1, "errors": {}},
                                        [_finding(f"LIVE{i}", f"live {i}", 800.0 - i)])
    page = app.page_container.get_page("findings")
    page.reload_now()
    assert [k[1] for k in page._view.row_widgets] == [f"LIVE{i}" for i in range(5)]
    assert "Show all 6" in _texts(page._view)
    assert page._caption.cget("text").startswith("6 changes worth testing")      # still counted
    page._view.show_all("yield")
    assert [k[1] for k in page._view.row_widgets][0] == "OLD"                     # every row, in rank
    assert tag in _texts(page._view)
    assert _texts(page._view).count(tag) == 1


def test_when_which_models_are_inactive_cannot_be_read_the_page_says_so(make_app, monkeypatch):
    import laser_trim_analyzer.gui.v6.pages.findings_page as fp

    def boom(db):
        raise RuntimeError("invented activity crash")
    app = make_app()
    app.db.replace_process_findings("BIG", {"tracks": 1, "errors": {}}, [_finding("BIG", "big", 500.0)])
    monkeypatch.setattr(fp, "load_activity", boom)
    page = app.page_container.get_page("findings")
    page.reload_now()
    notices = " | ".join(_labels(page._notices))
    assert "Which models are inactive could not be worked out" in notices
    assert "RuntimeError: invented activity crash" in notices
    assert len(page._view.row_widgets) == 1                         # the list itself is still true
