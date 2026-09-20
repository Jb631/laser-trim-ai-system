"""M-1 — a failed Model-page load must never render as "no data", and one
model's verdict/pills must never linger under another model's name.

See .superpowers/sdd/prebuild-fixes/M1-brief.md. Seeding and the synchronous
open pattern come from test_spec3c_model.py / test_findings_tab.py (tests/ is
on sys.path, so `_seed` and `_status` are importable from a sibling module).
"""


def _open(app, model, focus=None):
    """Open the Model page on `model` for the first time in this app — the
    exact suppress-then-sync pattern test_findings_tab.py uses."""
    app.set_model_route(model, focus)
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None       # suppress on_show's BACKGROUND reload
    app.show_page("model")
    del page._reload
    page.reload_now()                      # the synchronous path
    return page


def _route(app, page, model, focus=None):
    """Route an ALREADY-OPEN Model page to a different model.

    PageContainer.show() is a no-op when the page is already current
    (page_container.py: `self._current == name` short-circuits), so this
    drives on_show() directly instead of app.show_page() — same
    suppress-then-sync pattern as _open.
    """
    app.set_model_route(model, focus)
    page._reload = lambda **kw: None
    page.on_show()
    del page._reload
    page.reload_now()


def test_a_crashed_loader_is_named_not_rendered_as_no_data(make_app, monkeypatch):
    from test_spec3c_model import _seed
    app = make_app()
    _seed(app.db, "HOT", fails_last=12)

    def _boom(*a, **kw):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(app.db, "get_model_trim_ft_agreement", _boom)

    page = _open(app, "HOT")
    assert page._load_banner.winfo_manager() != ""
    text = page._load_banner.cget("text")
    assert "Could not load" in text
    assert "trim vs final test" in text
    assert "unit list" not in text        # only the patched loader failed


def test_a_healthy_page_has_no_load_banner(make_app):
    """"No banner" has to hold whether or not the load-banner mechanism is
    even built yet — this pins the absence itself, not one implementation
    of it."""
    from test_spec3c_model import _seed
    app = make_app()
    _seed(app.db, "HOT", fails_last=12)
    page = _open(app, "HOT")
    banner = getattr(page, "_load_banner", None)
    assert banner is None or banner.winfo_manager() == ""


def test_the_banner_clears_when_the_next_load_succeeds(make_app, monkeypatch):
    from test_spec3c_model import _seed
    app = make_app()
    _seed(app.db, "HOT", fails_last=12)
    real = app.db.get_model_trim_ft_agreement
    state = {"fail": True}

    def _flaky(*a, **kw):
        if state["fail"]:
            raise RuntimeError("database is locked")
        return real(*a, **kw)
    monkeypatch.setattr(app.db, "get_model_trim_ft_agreement", _flaky)

    page = _open(app, "HOT")
    assert page._load_banner.winfo_manager() != ""
    state["fail"] = False
    page.reload_now()
    assert page._load_banner.winfo_manager() == ""


def test_model_b_never_shows_model_a_verdict(make_app, monkeypatch):
    from test_spec3c_model import _seed
    app = make_app()
    _seed(app.db, "AAA")
    _seed(app.db, "BBB")

    page = _open(app, "AAA")
    aaa_text = page._verdict.cget("text")
    assert aaa_text and aaa_text != "—"

    def _boom(*a, **kw):
        raise RuntimeError("verdict boom")
    monkeypatch.setattr(page, "_compute_verdict", _boom)

    _route(app, page, "BBB")
    assert page._verdict.cget("text") == "—"
    assert page._verdict.cget("text") != aaa_text
    assert "verdict" in page._load_banner.cget("text")


def test_model_b_never_shows_model_a_pills(make_app, monkeypatch):
    import laser_trim_analyzer.gui.v6.pages.model_page as mp
    from test_spec3c_model import _seed, _status
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
    app = make_app()
    _seed(app.db, "AAA")
    _seed(app.db, "BBB")

    page = _open(app, "AAA")
    page._pill_row.set_status(_status("AAA"))     # give AAA's pills real content

    def _boom(*a, **kw):
        raise RuntimeError("drift status boom")
    monkeypatch.setattr(mp, "get_model_drift_status", _boom)

    _route(app, page, "BBB")

    # A freshly constructed row's pills, never touched by set_status — the
    # target "just constructed" look (brief: read _Pill for what that is).
    fresh = MetricPillRow(app, theme=ThemeManager(), on_pill_click=lambda _: None)
    fresh_texts = {m: p._summary_label.cget("text") for m, p in fresh._pills.items()}
    actual_texts = {m: p._summary_label.cget("text")
                    for m, p in page._pill_row._pills.items()}
    assert actual_texts == fresh_texts
    assert "drift status" in page._load_banner.cget("text")
