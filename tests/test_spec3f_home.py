"""The Overview -- the landing page, key "home" (Graphite redesign, 2026-10-02).

Spec: docs/superpowers/specs/2026-10-02-graphite-redesign-design.md. James: "there is so much
going on its hard to see what is what". The numbers come from ONE loader,
gui/v6/overview_data.load_overview (tests/test_overview_data.py pins them); these tests pin what
the page DRAWS from what it is handed:

  * the landing page, titled Overview; every other route still reachable; Triage retired;
  * "N models need a look" and its window caption -- never a count it does not have;
  * the cards (model, units, big pass %, "was", twelve bars, the reason in the fail colour, the
    hand-trim tag), a click opening the model;
  * "Everything else" as plain rows with their trend words;
  * the inactive models on one collapsed line that expands in place;
  * the two quiet links, the quiet notices on top, and a banner naming every part that failed;
  * no blue button of its own (the top bar holds the one).

Data here is INVENTED.
"""
from datetime import datetime, timedelta

import pytest

from laser_trim_analyzer.gui.v6 import overview_data as od

ANCHOR = datetime(2026, 3, 20, 15, 30)
MONTHS = [None] * 8 + [50.0, None, 75.0, 100.0]


# ---- helpers ---------------------------------------------------------------------------------

def _home(app):
    return app.page_container.get_page("home")


def _walk(widget):
    import tkinter
    yield widget
    for child in tkinter.Misc.winfo_children(widget):
        yield from _walk(child)


def _labels(widget):
    import customtkinter as ctk
    return [w.cget("text") for w in _walk(widget) if isinstance(w, ctk.CTkLabel)]


def _buttons(widget):
    import customtkinter as ctk
    return [w for w in _walk(widget) if isinstance(w, ctk.CTkButton)]


def _card(model="7000", **kw):
    kw.setdefault("reason", "Fail rate 4% → 12%")
    kw.setdefault("units", 412)
    kw.setdefault("pass_pct", 71.6)
    kw.setdefault("was_pct", 84.0)
    kw.setdefault("months", list(MONTHS))
    return od.Card(model=model, **kw)


def _row(model="7100", units=120, pass_pct=96.0, was_pct=95.0, **kw):
    trend, tone = od.trend_words(pass_pct, was_pct)
    return od.Row(model=model, units=units, pass_pct=pass_pct, was_pct=was_pct, trend=trend,
                  tone=tone, months=list(MONTHS), **kw)


def _ov(cards=(), others=(), **kw):
    kw.setdefault("anchor", ANCHOR)
    kw.setdefault("inactive", {})
    return od.Overview(cards=list(cards), others=list(others), **kw)


def _show(app, monkeypatch, ov):
    """The page drawing exactly `ov` (the loader stood in for)."""
    monkeypatch.setattr(od, "load_overview", lambda db, **k: ov)
    page = _home(app)
    page.reload_now()
    return page


def _routes(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "set_model_route", lambda *a, **k: calls.append(("route", a, k)))
    monkeypatch.setattr(app, "show_page", lambda name: calls.append(("show", name)))
    return calls


# ---- the landing page --------------------------------------------------------------------------

def test_the_overview_is_the_landing_page(make_app):
    app = make_app()
    assert app.page_container.current_page == "home"
    assert app.topbar._active_name == "home"
    assert _home(app).page_title == "Overview"
    assert "Overview" in _labels(_home(app)._header)


def test_every_route_is_still_reachable_and_triage_is_retired(make_app):
    """Nothing reachable was lost but Triage, whose cards and list are the Overview's now."""
    app = make_app()
    for key in ("dashboard", "process", "model", "settings", "findings", "home"):
        app.show_page(key)
        assert app.page_container.current_page == key, key
    assert app.page_container.get_page("triage") is None


def test_before_the_first_load_it_says_loading_and_no_count(make_app):
    page = _home(make_app())                   # the start-up load is still on its worker
    assert page._need_heading.cget("text") == "Models that need a look"
    assert page._need_caption.cget("text") == "Loading…"
    assert page._card_widgets == [] and page._inactive_toggle.winfo_manager() == ""


# ---- "N models need a look" and the cards ------------------------------------------------------

def test_the_heading_counts_the_cards_and_the_caption_names_the_window(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(cards=[_card("A"), _card("B"), _card("C")]))
    assert page._need_heading.cget("text") == "3 models need a look"
    assert page._need_caption.cget("text") == "Last 90 days · newest file 20 Mar 2026"
    assert [w.card.model for w in page._card_widgets] == ["A", "B", "C"]


def test_a_card_draws_its_numbers_bars_and_reason(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(cards=[_card("7000")]))
    t = page.theme
    (card,) = page._card_widgets
    texts = _labels(card)
    assert texts[:2] == ["7000", "412 units"]
    assert card._pass.cget("text") == "72%" and card._was.cget("text") == "was 84%"
    assert card._pass.cget("font").cget("family") == t.resolved_mono_medium or \
        card._pass.cget("font").cget("family") == t.resolved_mono          # the mono font
    assert card._reason.cget("text") == "Fail rate 4% → 12%"
    assert card._reason.cget("text_color") == t.FAIL_FG
    assert (card.cget("fg_color"), card.cget("border_color")) == (t.CARD, t.BORDER)
    bars = card._bars
    fills = [bars.itemcget(i, "fill") for i in bars.find_withtag("bar")]
    assert fills == [t.CHART_HISTORY, t.CHART_HISTORY, t.CHART_HIGHLIGHT]   # Dec, Feb, newest Mar


def test_a_final_test_card_and_a_hand_trim_card_say_so(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(cards=[
        _card("8506", final_test=True, units=25, was_pct=None),
        _card("8232-1", hand_trim=True)]))
    ft, hand = page._card_widgets
    assert "25 units · final test" in _labels(ft) and ft._was.cget("text") == "new"
    assert "hand trim" in _labels(hand) and "hand trim" not in _labels(ft)


def test_a_click_anywhere_on_a_card_opens_its_model(make_app, monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(cards=[_card("7000"), _card("7001")]))
    calls = _routes(app, monkeypatch)
    try:
        app.attributes("-alpha", 0.0)
    except Exception:
        pass
    app.geometry("1280x720+20000+20000")
    app.deiconify()
    app.update_idletasks()
    app.update()
    try:
        page._card_widgets[1]._reason._label.event_generate("<Button-1>", x=2, y=2)
        page._card_widgets[0]._bars.event_generate("<Button-1>", x=2, y=2)
        app.update()
    finally:
        app.withdraw()
    assert calls == [("route", ("7001",), {}), ("show", "model"),
                     ("route", ("7000",), {}), ("show", "model")]


@pytest.mark.parametrize("size,columns", [((1400, 900), 4), ((1280, 720), 4), ((960, 640), 3)])
def test_the_cards_sit_four_across_from_about_1280_and_fewer_below(make_app, monkeypatch, size,
                                                                  columns):
    app = make_app()
    page = _show(app, monkeypatch, _ov(cards=[_card(f"M{i}") for i in range(9)]))
    try:
        app.attributes("-alpha", 0.0)
    except Exception:
        pass
    app.overrideredirect(True)
    app.geometry(f"{size[0]}x{size[1]}+20000+20000")
    app.deiconify()
    for _ in range(3):
        app.update_idletasks()
        app.update()
    try:
        cols = {int(w.grid_info()["column"]) for w in page._card_widgets}
        assert cols == set(range(columns)), (size, cols)
    finally:
        app.withdraw()


def test_a_failed_card_source_is_named_and_the_heading_has_no_count(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(
        cards=[_card("D1")], failed={od.PART_FOCUS: "RuntimeError: invented focus crash"}))
    t = page.theme
    assert page._need_heading.cget("text") == "Models that need a look"
    assert page._load_banner.winfo_manager() == "pack"
    said = page._load_banner.cget("text")
    assert "the drifting-now list (RuntimeError: invented focus crash)" in said
    assert "not an all-clear" in said
    assert (page._load_banner.cget("fg_color"), page._load_banner.cget("text_color")) == (
        t.CHECK_TINT, t.CHECK)
    assert len(page._card_widgets) == 1                    # what did load still shows


def test_with_both_card_sources_failed_the_cards_place_says_so(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(failed={od.PART_FOCUS: "RuntimeError: a",
                                                      od.PART_DRIFT: "RuntimeError: b"}))
    assert page._need_heading.cget("text") == "Models that need a look"
    assert page._cards_note.winfo_manager() == "grid"
    assert "Could not be worked out" in page._cards_note.cget("text")
    assert not any("0 models" in s or "No models" in s for s in _labels(page))


def test_with_nothing_drifting_the_heading_says_so_plainly(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(others=[_row()]))
    assert page._need_heading.cget("text") == "No models need a look"
    assert page._cards_note.winfo_manager() == "" and page._load_banner.winfo_manager() == ""


def test_a_good_load_after_a_failure_clears_the_banner(make_app, monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(failed={od.PART_DRIFT: "RuntimeError: x"}))
    assert page._load_banner.winfo_manager() == "pack"
    page = _show(app, monkeypatch, _ov(cards=[_card()]))
    assert page._load_banner.winfo_manager() == "" and page._load_banner.cget("text") == ""
    assert page._need_heading.cget("text") == "1 model needs a look"


# ---- "Everything else" --------------------------------------------------------------------------

def test_everything_else_is_plain_rows_with_their_trend_words(make_app, monkeypatch):
    rows = [_row("UP", units=300, pass_pct=90.0, was_pct=80.0),
            _row("DOWN", units=200, pass_pct=60.0, was_pct=70.0),
            _row("SAME", units=100, pass_pct=95.0, was_pct=93.0),
            _row("8340-1", units=50, pass_pct=40.0, was_pct=None, hand_trim=True)]
    page = _show(make_app(), monkeypatch, _ov(others=rows))
    t = page.theme
    drawn = page._others_frame.winfo_children()
    assert [w.row.model for w in drawn] == ["UP", "DOWN", "SAME", "8340-1"]
    assert _labels(drawn[0]) == ["UP", "300 units", "90%", "up 10 pts"]
    assert [w._trend.cget("text") for w in drawn] == ["up 10 pts", "down 10 pts", "steady", "new"]
    assert [w._trend.cget("text_color") for w in drawn] == [
        t.PASS_FG, t.FAIL_FG, t.TEXT_SECONDARY, t.TEXT_SECONDARY]
    assert "hand trim" in _labels(drawn[3]) and "hand trim" not in _labels(drawn[0])


def test_a_row_click_opens_its_model(make_app, monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(others=[_row("7100")]))
    calls = _routes(app, monkeypatch)
    (row,) = page._others_frame.winfo_children()
    try:
        app.attributes("-alpha", 0.0)
    except Exception:
        pass
    app.geometry("1280x720+20000+20000")
    app.deiconify()
    app.update_idletasks()
    app.update()
    try:
        row._trend._label.event_generate("<Button-1>", x=2, y=2)
        app.update()
    finally:
        app.withdraw()
    assert calls == [("route", ("7100",), {}), ("show", "model")]


def test_failed_pass_rates_never_read_as_an_empty_list(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(
        anchor=None, cards=[_card(pass_pct=None, units=0)],
        failed={od.PART_RATES: "RuntimeError: invented rates crash"}))
    assert page._others_frame.winfo_children() == []
    assert page._others_note.winfo_manager() == "pack"
    assert "Could not be worked out" in page._others_note.cget("text")
    assert page._need_caption.cget("text") == "Last 90 days"
    assert "the pass rates (RuntimeError: invented rates crash)" in page._load_banner.cget("text")


# ---- the inactive models: one line, expanding in place (F5) ------------------------------------

def test_the_inactive_line_is_collapsed_and_expands_in_place(make_app, monkeypatch):
    inactive = {"OLDER": datetime(2016, 3, 9), "NEVER": None, "NEWER": datetime(2019, 6, 2)}
    page = _show(make_app(), monkeypatch, _ov(others=[_row()], inactive=inactive))
    toggle = page._inactive_toggle
    assert toggle.winfo_manager() == "pack" and toggle.cget("text") == "Inactive models (3) ▸"
    assert page._inactive_list.winfo_manager() == ""
    toggle.invoke()
    assert toggle.cget("text") == "Inactive models (3) ▾"
    assert page._inactive_list.winfo_manager() == "pack"
    lines = "\n".join(_labels(page._inactive_list)).split("\n")
    assert lines == ["NEWER · last trimmed Jun 2019", "OLDER · last trimmed Mar 2016",
                     "NEVER · no trims on record"]
    order = page._body.pack_slaves()
    assert order.index(page._others_frame) < order.index(toggle) < order.index(page._inactive_list)
    assert order.index(page._inactive_list) < order.index(page._links)
    toggle.invoke()
    assert toggle.cget("text").endswith("▸") and page._inactive_list.winfo_manager() == ""


def test_when_activity_cannot_be_worked_out_there_is_no_line_and_the_banner_says_so(
        make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(
        inactive=None, failed={od.PART_ACTIVITY: "RuntimeError: invented activity crash"}))
    assert page._inactive_toggle.winfo_manager() == ""
    said = page._load_banner.cget("text")
    assert "Which models are inactive could not be worked out" in said
    assert "RuntimeError: invented activity crash" in said
    assert "Could not load" not in said                     # nothing else failed


# ---- the two quiet links, the notices, the one blue button --------------------------------------

def test_the_two_quiet_links_go_to_findings_and_company_trends(make_app):
    app = make_app()
    page = _home(app)
    t = page.theme
    assert (page._findings_link.cget("text"), page._trends_link.cget("text")) == (
        "All findings", "Company trends")
    assert page._findings_link.cget("fg_color") == "transparent"
    page._findings_link.invoke()
    assert app.page_container.current_page == "findings"
    page._trends_link.invoke()
    assert app.page_container.current_page == "dashboard"
    order = page._body.pack_slaves()
    assert order[-1] is page._links                         # at the foot
    assert t.ACCENT != page._trends_link.cget("fg_color")


def test_the_overview_draws_no_blue_button_of_its_own(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(cards=[_card()], others=[_row()],
                                              inactive={"OLD": datetime(2016, 1, 1)}))
    page._inactive_toggle.invoke()
    t = page.theme
    assert [b.cget("text") for b in _buttons(page) if b.cget("fg_color") == t.ACCENT] == []


def test_the_quiet_lines_sit_on_top_in_order(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(
        legacy_ft=12, unreadable=70, failed={od.PART_DRIFT: "RuntimeError: x"}))
    page.set_run_summary("3 folders · 214 new files · 2 min 40 s")
    t = page.theme
    order = page._body.pack_slaves()
    lines = [page._run_line, page._legacy_ft_label, page._unreadable_label, page._load_banner]
    assert [order.index(w) for w in lines] == sorted(order.index(w) for w in lines)
    assert order.index(page._load_banner) < order.index(page._need_heading)
    assert "12 final-test records" in page._legacy_ft_label.cget("text")
    assert "70 files are being skipped" in page._unreadable_label.cget("text")
    for quiet in (page._run_line, page._legacy_ft_label, page._unreadable_label):
        assert (quiet.cget("fg_color"), quiet.cget("text_color")) == (t.CARD, t.TEXT_SECONDARY)


def test_with_nothing_to_say_the_top_lines_are_not_drawn(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(others=[_row()]))
    for w in (page._run_line, page._legacy_ft_label, page._unreadable_label, page._load_banner):
        assert w.winfo_manager() == "" and w.cget("text") == ""


# ---- loads land in order; one draw failing never stops the rest ---------------------------------
# The helpers below are shared: tests/test_dashboard.py and tests/test_findings_page.py import them.

def _settle_workers(app, seconds=10.0):
    """Let the workers the app started itself (the Overview's start-up load) finish and apply, so
    the loads a test starts are the only ones in flight."""
    import threading
    import time
    main = threading.main_thread()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.update()
        if not any(t is not main and t.daemon and t.is_alive() for t in threading.enumerate()):
            break
        time.sleep(0.01)
    _pump_ui(app)


def _pump_ui(app, seconds=0.3):
    """Drain what workers posted (UiDispatcher empties its queue on an after() loop)."""
    import time
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.update()
        time.sleep(0.01)


def _pump_until(app, done, seconds=10.0):
    import time
    end = time.monotonic() + seconds
    while time.monotonic() < end and not done():
        app.update()
        time.sleep(0.01)
    return done()


def _older_then_newer(fake_older, fake_newer):
    """A loader whose FIRST call is the older load -- it blocks until released -- and whose later
    calls are the newer load, answered at once. It carries two helpers: `.started()` waits until
    the older call is inside it, and `.release()` lets that call finish and waits until its worker
    thread has posted its apply and exited."""
    import threading
    import time
    gate, state = threading.Event(), {}

    def loader(*a, **k):
        if "older" not in state:
            state["older"] = threading.current_thread()
            gate.wait(10)
            return fake_older()
        return fake_newer()

    def started():
        end = time.monotonic() + 10
        while "older" not in state and time.monotonic() < end:
            time.sleep(0.005)
        assert "older" in state, "the older load never started"

    def release():
        gate.set()
        state["older"].join(10)
        assert not state["older"].is_alive()

    loader.started, loader.release = started, release
    return loader


@pytest.mark.parametrize("newer", ("async", "sync"))
def test_an_older_load_never_overwrites_a_newer_one(make_app, monkeypatch, newer):
    app = make_app()
    page = _home(app)
    _settle_workers(app)
    load = _older_then_newer(
        lambda: _ov(failed={od.PART_FOCUS: "RuntimeError: an older crash"}),     # older: failed
        lambda: _ov(cards=[_card("NEWER")]))                                      # newer: healthy
    monkeypatch.setattr(od, "load_overview", load)
    page.on_show()                         # the older load, still in its query...
    load.started()
    if newer == "async":                   # ...when a newer one starts and finishes first
        page.on_show()
        assert _pump_until(app, lambda: [w.card.model for w in page._card_widgets] == ["NEWER"])
    else:
        page.reload_now()
    load.release()                         # the older load finishes LAST
    _pump_ui(app)
    assert [w.card.model for w in page._card_widgets] == ["NEWER"]
    assert page._load_banner.winfo_manager() == "", "an older load overwrote a newer one"


def test_one_part_failing_to_draw_never_stops_the_others(make_app, monkeypatch, caplog):
    import logging
    import laser_trim_analyzer.gui.v6.pages.home_page as home_mod

    def boom(*a, **k):
        raise RuntimeError("invented list render crash")
    monkeypatch.setattr(home_mod, "_ListRow", boom)
    with caplog.at_level(logging.ERROR):
        page = _show(make_app(), monkeypatch, _ov(cards=[_card("A")], others=[_row()], legacy_ft=3,
                                                  inactive={"OLD": datetime(2016, 1, 1)}))
    assert "invented list render crash" in caplog.text
    assert page._need_heading.cget("text") == "1 model needs a look"
    assert page._legacy_ft_label.winfo_manager() == "pack"
    assert page._inactive_toggle.winfo_manager() == "pack"


def test_a_load_with_nothing_new_redraws_nothing(make_app, monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(cards=[_card("A")], others=[_row()]))
    before = (list(page._card_widgets), page._others_frame.winfo_children())
    page = _show(app, monkeypatch, _ov(cards=[_card("A")], others=[_row()]))
    assert (page._card_widgets, page._others_frame.winfo_children()) == before


# ---- end to end, on a real (invented) database ---------------------------------------------------

def test_the_page_draws_what_the_loader_finds_in_a_real_database(make_app):
    from laser_trim_analyzer.database.models import AnalysisResult, StatusType, SystemType
    app = make_app()
    n = iter(range(1, 10 ** 6))

    def lot(model, day, passes, fails):
        with app.db.session() as s:
            for status, k in (("PASS", passes), ("FAIL", fails)):
                for _ in range(k):
                    i = next(n)
                    s.add(AnalysisResult(model=model, serial=f"{model}-{i}", system=SystemType.A,
                                         filename=f"{model}_{i}.xls", file_date=day,
                                         overall_status=StatusType[status]))
    start = ANCHOR - timedelta(days=77)
    for k in range(11):
        lot("HOT", start + timedelta(days=7 * k), 18, 2)
    lot("HOT", ANCHOR, 8, 12)
    lot("CALM", ANCHOR - timedelta(days=3), 50, 0)
    page = _home(app)
    page.reload_now()
    assert page._need_heading.cget("text") == "1 model needs a look"
    (card,) = page._card_widgets
    assert card.card.model == "HOT" and card._reason.cget("text") == "Fail rate 10% → 60%"
    assert [w.row.model for w in page._others_frame.winfo_children()] == ["CALM"]


# ---- nothing is cut at 1280x720 (the render audit's own detector) --------------------------------

_LONG = "Untrimmed resistance 21,270 → 23,113 · Fail rate 34% → 74% · still passing"


@pytest.mark.parametrize("scale", (1.0, 1.5))
def test_nothing_on_a_full_overview_is_cut(make_app, monkeypatch, scale):
    import customtkinter as ctk
    from test_blocks import failure_texts_cut
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    try:
        app = make_app()
        cards = [_card(f"MODEL-{i:03d}", reason=_LONG, hand_trim=i % 5 == 0, final_test=i % 7 == 0)
                 for i in range(16)]
        rows = [_row(f"ROW-{i:03d}", hand_trim=i == 2) for i in range(12)]
        page = _show(app, monkeypatch, _ov(
            cards=cards, others=rows, legacy_ft=1234, unreadable=56,
            inactive={f"OLD-{i}": datetime(2016, 1 + i % 12, 1) for i in range(30)},
            failed={od.PART_DRIFT: "RuntimeError: " + "a long invented explanation " * 6}))
        page.set_run_summary("8 folders · 12,345 new files · 1 h 2 min · " * 4)
        page._inactive_toggle.invoke()
        texts = [page._load_banner, page._run_line, page._legacy_ft_label, page._unreadable_label]
        cut = failure_texts_cut(app, page, texts)
        assert cut["1280x720"] == ([], []), cut["1280x720"]
        assert not cut["960x640"][0], cut["960x640"][0]
    finally:
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)
