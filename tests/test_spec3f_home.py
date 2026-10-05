"""The Overview -- the landing page, key "home" (Graphite redesign, 2026-10-02; option B, 2026-10-04).

Specs: docs/superpowers/specs/2026-10-02-graphite-redesign-design.md and
2026-10-04-option-b-design.md. James: "there is so much going on its hard to see what is what";
then, with Task Manager TMOG open, "i like B" -- a list of models with the selected model's detail
beside it. The numbers come from ONE loader, gui/v6/overview_data.load_overview
(tests/test_overview_data.py pins them); these tests pin what the page DRAWS from what it is
handed:

  * the landing page, titled Overview; every other route still reachable; Triage retired;
  * ONE header line -- "13 models need a look · $X lost at final test in the last 90 days · newest
    file 20 Mar 2026" -- never a count or a sum it does not have;
  * the yield-by-laser chart as a compact strip, always visible, above the list;
  * the LIST: "Needs a look (N)" then "Everything else (N)", each section naming its columns; each
    row the model over its units and tags, what changed (a card's reason, a row's year-on-year),
    twelve months as bars on one scale, the 90-day pass % -- every column lined up down the list
    (James, 2026-10-04, of the line and bare % before: "these little charts and % dont mean
    anything?"); the selected row ELEVATED with a border; "Other models on file" and "Inactive
    models" folded at its end;
  * the DETAIL of the selected row (the first card on load, a click selects): the model in the
    title face, its status word, a pass meter with "was", why it is here, twelve months, the facts,
    and "Open full page ›";
  * the quiet links, the quiet notices on top, and a banner naming every part that failed;
  * no blue button of its own (the top bar holds the one).

Data and prices here are INVENTED.
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
    if "reason" not in kw:                  # the loader's shape: the parts, and them on one line
        kw.setdefault("why", ["A recent run failed 12%", "usually 4%"])
        kw["reason"] = " · ".join(kw["why"])
    kw.setdefault("units", 412)
    kw.setdefault("pass_pct", 71.6)
    kw.setdefault("was_pct", 84.0)
    kw.setdefault("months", list(MONTHS))
    kw.setdefault("lasers", ["B", "A"])
    kw.setdefault("ft_fails", 0)
    kw.setdefault("newest", ANCHOR.date())
    return od.Card(model=model, **kw)


def _row(model="7100", units=120, pass_pct=96.0, was_pct=95.0, **kw):
    trend, tone = od.trend_words(pass_pct, was_pct)
    kw.setdefault("lasers", ["A"])
    kw.setdefault("ft_fails", 0)
    kw.setdefault("newest", ANCHOR.date())
    return od.Row(model=model, units=units, pass_pct=pass_pct, was_pct=was_pct, trend=trend,
                  tone=tone, months=list(MONTHS), **kw)


def _ov(cards=(), others=(), **kw):
    kw.setdefault("anchor", ANCHOR)
    kw.setdefault("inactive", {})
    kw.setdefault("money_total", 0.0)
    return od.Overview(cards=list(cards), others=list(others), **kw)


def _mapped(app, size="1280x720"):
    """The app laid out at `size`, off-screen and transparent, so clicks and geometry are real."""
    try:
        app.attributes("-alpha", 0.0)
    except Exception:
        pass
    app.overrideredirect(True)
    app.geometry(f"{size}+20000+20000")
    app.deiconify()
    for _ in range(3):
        app.update_idletasks()
        app.update()


def _click(app, widget, times=1):
    """A real click -- or `times` in a row: Tk refuses to generate a <Double-...> event, but two
    presses at one spot ARE a double click -- on `widget` (its own label for a CTk widget)."""
    target = getattr(widget, "_label", None) or getattr(widget, "_canvas", None) or widget
    for _ in range(times):
        target.event_generate("<ButtonPress-1>", x=2, y=2)
        target.event_generate("<ButtonRelease-1>", x=2, y=2)
    app.update()


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
    # The top bar names it -- once (option B, 2026-10-04: "Overview" twice was finish item 2).
    assert app.topbar._items["home"]._label.cget("text") == "Overview"
    assert "Overview" not in _labels(_home(app))


def test_every_route_is_still_reachable_and_triage_is_retired(make_app):
    """Nothing reachable was lost but Triage, whose cards and list are the Overview's now."""
    app = make_app()
    for key in ("dashboard", "process", "model", "settings", "findings", "home"):
        app.show_page(key)
        assert app.page_container.current_page == key, key
    assert app.page_container.get_page("triage") is None


def test_before_the_first_load_it_says_loading_and_no_count(make_app):
    page = _home(make_app())                   # the start-up load is still on its worker
    assert page._headline.cget("text") == "Loading…"
    assert not any(ch.isdigit() for ch in page._need_heading.cget("text"))
    assert page._need_heading.cget("text") == "Needs a look"
    assert page._card_rows == [] and page._other_rows == []
    assert page._inactive_toggle.winfo_manager() == ""
    # ...and the list says so under each heading, never a bare heading over nothing.
    for note in (page._cards_note, page._others_note):
        assert note.cget("text") == "Loading…" and note.winfo_manager() == "pack"
    assert page._detail.note.cget("text") == "Loading…"
    assert page._detail.head.winfo_manager() == ""            # no model drawn before a load
    # ...and the chart of each laser: "Loading…", never "No trim data" before it has looked.
    assert page._trend_note.cget("text") == "Loading…" and page._trend_note.winfo_manager() == "pack"
    assert page._trend_chart.winfo_manager() == ""


# ---- the header line ------------------------------------------------------------------------------

def test_the_header_line_counts_the_cards_the_dollars_and_names_the_window(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(cards=[_card("A"), _card("B"), _card("C")],
                                              money_total=12345.4, unpriced=2))
    assert page._headline.cget("text") == (
        "3 models need a look · $12,345 lost at final test in the last 90 days · 2 without a "
        "price · newest file 20 Mar 2026")
    t = page.theme
    assert page._headline.cget("text_color") == t.TEXT_PRIMARY
    order = page._body.pack_slaves()
    assert order.index(page._headline) < order.index(page._trend_box) < order.index(page._split)


def test_a_header_line_with_a_failed_card_source_or_money_read_says_so(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(
        cards=[_card("D1")], money_total=None,
        failed={od.PART_FOCUS: "RuntimeError: invented focus crash",
                od.PART_MONEY: "RuntimeError: invented money crash"}))
    said = page._headline.cget("text")
    assert said.startswith("Models that need a look · dollars lost at final test could not be")
    assert "$" not in said
    banner = page._load_banner.cget("text")
    assert "the dollars lost at final test (RuntimeError: invented money crash)" in banner


def test_with_no_price_loaded_the_header_asks_for_prices_never_zero(make_app, monkeypatch):
    """A missing input never reads as a zero (coordinator, 2026-10-04): with no price loaded at
    all the header says where prices come from, not "$0 lost"."""
    page = _show(make_app(), monkeypatch, _ov(cards=[_card("A")], money_total=0.0, unpriced=4,
                                              no_prices=True))
    said = page._headline.cget("text")
    assert "$" not in said and "without a price" not in said
    assert said == ("1 model needs a look · add prices in Settings → Backlog to see the dollars "
                    "lost at final test · newest file 20 Mar 2026")


# ---- the list ----------------------------------------------------------------------------------

def test_the_list_is_needs_a_look_then_everything_else_then_the_folded_lines(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(
        cards=[_card("C1"), _card("C2")], others=[_row("R1"), _row("R2"), _row("R3")],
        quiet={"Q1": datetime(2025, 6, 2)}, inactive={"OLD": datetime(2016, 3, 9)}))
    assert page._need_heading.cget("text") == "Needs a look (2)"
    assert page._others_heading.cget("text") == "Everything else (3)"
    assert [r.model for r in page._card_rows] == ["C1", "C2"]
    assert [r.model for r in page._other_rows] == ["R1", "R2", "R3"]
    order = page._list.pack_slaves()
    assert (order.index(page._need_heading) < order.index(page._cards_frame)
            < order.index(page._others_heading) < order.index(page._others_frame)
            < order.index(page._quiet_toggle) < order.index(page._inactive_toggle))
    # The list sits in the left pane, the detail in the right, under the chart strip.
    assert page._list.grid_info()["column"] == 0 and page._detail.grid_info()["column"] == 1


def test_a_row_is_its_model_units_what_changed_its_month_bars_and_its_pass(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(
        cards=[_card("7000", lasers=["A", "C"]),
               _card("8506", final_test=True, units=25, lasers=[]),
               _card("8232-1", hand_trim=True, lasers=["B"])],
        others=[_row("7100", lasers=["C"]), _row("7200", pass_pct=61.0, was_pct=70.0)]))
    t = page.theme
    first, ft, hand = page._card_rows
    assert _labels(first) == ["7000", "412 units", "A recent run failed 12%", "usually 4%", "72%"]
    assert first._why.cget("text_color") == t.FAIL_FG              # why it needs a look...
    assert first._why_more.cget("text_color") == t.TEXT_SECONDARY  # ...and what qualifies it
    assert first._pct.cget("font").cget("family") in (t.resolved_mono, t.resolved_mono_medium)
    assert first._bars.values == MONTHS
    assert first._bars.recent == od.window_months(ANCHOR) == 4     # 22 Dec - 20 Mar
    # The tags sit under the units: the numbers of a final-test card are final test's.
    assert _labels(ft)[:3] == ["8506", "25 units", "final test"]
    assert _labels(hand)[:3] == ["8232-1", "412 units", "hand trim"]
    assert "hand trim" not in _labels(first) and "final test" not in _labels(first)
    # Everything else: its 90 days against the year before, in that change's colour.
    steady, down = page._other_rows
    assert _labels(steady) == ["7100", "120 units", "steady", "96%"]
    assert steady._why.cget("text_color") == t.TEXT_SECONDARY and steady._why_more is None
    assert _labels(down)[2] == "down 9 pts" and down._why.cget("text_color") == t.FAIL_FG


def test_a_cards_signal_comes_first_and_each_qualifier_on_its_own_quiet_line(make_app, monkeypatch):
    """8889's shape on the work data: two signals. Each line whole -- "5,223 → 6,226" is never
    broken at its arrow (non-breaking spaces round it)."""
    why = ["A recent run failed 10%", "usually 0%", "Untrimmed resistance 5,223 → 6,226"]
    page = _show(make_app(), monkeypatch, _ov(cards=[_card("8889", why=why,
                                                               reason=" · ".join(why))]))
    (row,) = page._card_rows
    assert row._why.cget("text") == "A recent run failed 10%"
    assert row._why_more.cget("text") == "usually 0%\nUntrimmed resistance 5,223\u00a0→\u00a06,226"
    assert page._detail.reason.cget("text") == " · ".join(why)     # the detail: one line, as is


def test_each_section_names_its_columns_and_only_over_rows(make_app, monkeypatch):
    from laser_trim_analyzer.gui.v6.pages import home_page as hp
    page = _show(make_app(), monkeypatch, _ov(cards=[_card("A")], others=[_row("R")]))
    assert _labels(page._cards_head) == ["Model", "What changed", "Pass %", "12 months", "90 days"]
    assert _labels(page._others_head) == ["Model", "On the year before", "Pass %", "12 months",
                                          "90 days"]
    order = page._list.pack_slaves()
    assert (order.index(page._need_heading) < order.index(page._cards_head)
            < order.index(page._cards_frame) < order.index(page._others_heading)
            < order.index(page._others_head) < order.index(page._others_frame))
    assert hp.HEAD_WHY == "What changed" and hp.HEAD_YEAR == "On the year before"
    page = _show(page.app, monkeypatch, _ov(cards=[], others=[_row("R")]))
    assert page._cards_head.winfo_manager() == ""                  # "No model needs a look."
    assert page._others_head.winfo_manager() == "pack"


def test_every_column_lines_up_down_the_list_and_under_its_name(make_app, monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(
        cards=[_card("8232-1", hand_trim=True, pass_pct=41.0),
               _card("8504-2", pass_pct=100.0, why=_LONG.split(" · "), reason=_LONG),
               _card("1844205", final_test=True, pass_pct=65.0)],
        others=[_row("6126", pass_pct=94.5), _row("8877-4", pass_pct=100.0)]))
    _mapped(app)
    try:
        rows = page._card_rows + page._other_rows
        x = lambda w: w.winfo_rootx()                               # noqa: E731
        right = lambda w: w.winfo_rootx() + w.winfo_width()         # noqa: E731
        assert len({x(r._why_box) for r in rows}) == 1, [x(r._why_box) for r in rows]
        assert len({x(r._bars) for r in rows}) == 1, [x(r._bars) for r in rows]
        assert len({right(r._pct) for r in rows}) == 1, [right(r._pct) for r in rows]
        for head in (page._cards_head, page._others_head):
            assert abs(x(head.why) - x(rows[0]._why_box)) <= 1
            assert abs(x(head.bars) - x(rows[0]._bars)) <= 1
            assert abs(right(head.pct) - right(rows[0]._pct)) <= 1
            assert right(head.bars) < x(head.pct), "the two names overlap"
        # A reason's lines stay inside their column: never under the bars.
        long_row = page._card_rows[1]
        assert right(long_row._why) <= x(long_row._bars)
        assert right(long_row._why_more) <= x(long_row._bars)
        assert long_row._why_box.winfo_height() > page._card_rows[0]._why_box.winfo_height()
    finally:
        app.withdraw()


def test_the_first_card_is_selected_on_load_elevated_with_a_border(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(cards=[_card("A"), _card("B")], others=[_row("R")]))
    t = page.theme
    assert page._selected == "A" and page._detail.title.cget("text") == "A"
    a, b = page._card_rows
    assert (a.cget("fg_color"), a.cget("border_color")) == (t.ELEVATED, t.BORDER)
    assert b.cget("fg_color") != t.ELEVATED and b.cget("border_color") != t.BORDER
    assert a._bars.cget("bg") == t.ELEVATED and b._bars.cget("bg") == b.cget("fg_color")


def test_with_no_card_the_first_row_is_selected_and_a_reload_keeps_the_selection(make_app,
                                                                               monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(others=[_row("R1"), _row("R2")]))
    assert page._selected == "R1"
    page._select("R2")
    page = _show(app, monkeypatch, _ov(cards=[_card("NEW")], others=[_row("R1"), _row("R2")]))
    assert page._selected == "R2"                       # still there: still selected
    page = _show(app, monkeypatch, _ov(cards=[_card("NEW")], others=[_row("R1")]))
    assert page._selected == "NEW"                      # gone: the first card


def test_a_click_selects_a_row_and_the_detail_follows_without_leaving_the_page(make_app,
                                                                             monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(cards=[_card("7000"), _card("7001")], others=[_row("7100")]))
    calls = _routes(app, monkeypatch)
    _mapped(app)
    try:
        _click(app, page._card_rows[1]._pct)
        assert page._selected == "7001" and page._detail.title.cget("text") == "7001"
        _click(app, page._other_rows[0]._bars)
        assert page._selected == "7100" and page._detail.title.cget("text") == "7100"
    finally:
        app.withdraw()
    t = page.theme
    assert page._other_rows[0].cget("fg_color") == t.ELEVATED
    assert page._card_rows[1].cget("fg_color") != t.ELEVATED
    assert calls == []                                  # a click selects; it opens nothing


def test_a_double_click_opens_the_full_page(make_app, monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(cards=[_card("7000", metric="untrimmed_resistance")]))
    calls = _routes(app, monkeypatch)
    _mapped(app)
    try:
        _click(app, page._card_rows[0]._pct, times=2)
    finally:
        app.withdraw()
    assert calls == [("route", ("7000",), {"focus_metric": "untrimmed_resistance", "tab": "summary"}),
                     ("show", "model")]


# ---- the detail ----------------------------------------------------------------------------------

def test_the_detail_of_a_card(make_app, monkeypatch):
    from laser_trim_analyzer.ml.drift_types import metric_label
    page = _show(make_app(), monkeypatch, _ov(cards=[_card(
        "7000", lasers=["B", "A"], metric="untrimmed_resistance", money=1234.5, ft_fails=12,
        reason="Untrimmed resistance 4,693 → 5,897", newest=datetime(2026, 3, 18).date(),
        hand_trim=True)]))
    t = page.theme
    d = page._detail
    assert d.title.cget("text") == "7000" and d.title.cget("font") is t.title(t.SIZE_TITLE)
    assert d.status.cget("text") == "Drifting"
    assert (d.status.cget("text_color"), d.status.cget("fg_color")) == (t.CHECK, t.CHECK_TINT)
    assert d.hand.winfo_manager() == "pack" and d.hand.cget("text") == "hand trim"
    assert d.pct.cget("text") == "72%" and d.was.cget("text") == "was 84%"
    assert d.pct.cget("font").cget("family") in (t.resolved_mono, t.resolved_mono_medium)
    assert d.meter.pct == 71.6 and len(d.meter.find_withtag("on")) == 14
    assert d.reason.cget("text") == "Untrimmed resistance 4,693 → 5,897"
    assert d.reason.cget("text_color") == t.FAIL_FG
    assert d.chart.values == MONTHS and d.chart.color == t.SERIES_B
    assert d.chart.labels == ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                              "Jan", "Feb", "Mar"]
    assert d.chart_caption.cget("text") == "Pass % by month, Apr 2025 – Mar 2026"
    assert d.shown_facts() == {
        "Units, last 90 days": "412",
        "Lasers": "Laser 1 (LTS), Laser 2 (DLTS)",
        "Signal": metric_label("untrimmed_resistance"),
        "$ lost at final test, 90 days": "$1,234 · 12 failed final test",
        "Newest file": "18 Mar 2026"}
    assert d.open_link.cget("text") == "Open full page ›"


def test_the_detail_of_an_everything_else_row_is_steady_with_its_trend_and_no_signal(
        make_app, monkeypatch):
    rows = [_row("UP", pass_pct=90.0, was_pct=80.0), _row("DOWN", pass_pct=60.0, was_pct=70.0),
            _row("SAME", pass_pct=95.0, was_pct=93.0), _row("NEW", pass_pct=40.0, was_pct=None)]
    page = _show(make_app(), monkeypatch, _ov(others=rows))
    t = page.theme
    said = {}
    for r in rows:
        page._select(r.model)
        d = page._detail
        assert d.status.cget("text") == "Steady" and d.hand.winfo_manager() == ""
        assert "Signal" not in d.shown_facts()
        said[r.model] = (d.reason.cget("text"), d.reason.cget("text_color"))
    assert said == {
        "UP": ("up 10 pts on the year before", t.PASS_FG),
        "DOWN": ("down 10 pts on the year before", t.FAIL_FG),
        "SAME": ("steady on the year before", t.TEXT_SECONDARY),
        "NEW": ("new — nothing graded in the year before", t.TEXT_SECONDARY)}


def test_a_final_test_card_and_an_unpriced_model_say_so_in_the_detail(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(cards=[_card(
        "8506", final_test=True, units=25, lasers=[], money=None, ft_fails=3, was_pct=None)],
        unpriced=1))
    facts = page._detail.shown_facts()
    assert facts["Units, last 90 days"] == "25 · final test"
    assert facts["Lasers"] == "none in the last 90 days"
    assert facts["$ lost at final test, 90 days"] == "no price · 3 failed final test"
    assert page._detail.was.cget("text") == "new"
    assert page._detail.chart_caption.cget("text").startswith("Final-test pass % by month")
    assert page._detail.chart.color == page.theme.CHART_REFERENCE


def test_open_full_page_routes_a_card_with_its_signal_and_a_row_to_its_summary(make_app,
                                                                             monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(cards=[_card("7000", metric="linearity_fail_fraction")],
                                       others=[_row("7100")]))
    calls = _routes(app, monkeypatch)
    page._detail.open_link.invoke()
    page._select("7100")
    page._detail.open_link.invoke()
    # On Summary, charting the signal the card's reason names (final review, 2026-10-02).
    assert calls == [
        ("route", ("7000",), {"focus_metric": "linearity_fail_fraction", "tab": "summary"}),
        ("show", "model"),
        ("route", ("7100",), {"tab": "summary"}), ("show", "model")]


def test_with_nothing_to_show_the_detail_says_why(make_app, monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(anchor=None, money_total=None))
    assert page._detail.note.cget("text") == "No trim files on record yet."
    assert page._detail.head.winfo_manager() == ""
    page = _show(app, monkeypatch, _ov(anchor=None, money_total=None,
                                       failed={od.PART_RATES: "RuntimeError: x",
                                               od.PART_FOCUS: "RuntimeError: y"}))
    assert page._detail.note.cget("text") == "Could not be worked out — the notice above says what failed."


# ---- failures, named ---------------------------------------------------------------------------

def test_a_failed_card_source_is_named_and_nothing_claims_a_count_or_a_steady_model(make_app,
                                                                                  monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(
        cards=[_card("D1")], others=[_row("R1")],
        failed={od.PART_FOCUS: "RuntimeError: invented focus crash"}))
    t = page.theme
    assert page._headline.cget("text").startswith("Models that need a look · ")
    assert page._need_heading.cget("text") == "Needs a look"
    assert page._load_banner.winfo_manager() == "pack"
    said = page._load_banner.cget("text")
    assert "the drifting-now list (RuntimeError: invented focus crash)" in said
    assert "not an all-clear" in said
    assert (page._load_banner.cget("fg_color"), page._load_banner.cget("text_color")) == (
        t.CHECK_TINT, t.CHECK)
    assert len(page._card_rows) == 1                       # what did load still shows
    page._select("R1")                     # with a card source down, "Steady" would be a guess
    assert page._detail.status.winfo_manager() == ""


def test_with_both_card_sources_failed_the_list_says_so(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(failed={od.PART_FOCUS: "RuntimeError: a",
                                                      od.PART_DRIFT: "RuntimeError: b"}))
    assert page._need_heading.cget("text") == "Needs a look"
    assert page._cards_note.winfo_manager() == "pack"
    assert "Could not be worked out" in page._cards_note.cget("text")
    assert not any("0 models" in s or "No models" in s for s in _labels(page))
    assert "(" not in page._need_heading.cget("text")       # no count it does not have


def test_with_nothing_drifting_the_header_says_so_plainly(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(others=[_row()]))
    assert page._headline.cget("text").startswith("No models need a look · ")
    assert page._need_heading.cget("text") == "Needs a look (0)"
    assert page._cards_note.cget("text") == "No model needs a look."
    assert page._load_banner.winfo_manager() == ""


def test_a_good_load_after_a_failure_clears_the_banner(make_app, monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(failed={od.PART_DRIFT: "RuntimeError: x"}))
    assert page._load_banner.winfo_manager() == "pack"
    page = _show(app, monkeypatch, _ov(cards=[_card()]))
    assert page._load_banner.winfo_manager() == "" and page._load_banner.cget("text") == ""
    assert page._headline.cget("text").startswith("1 model needs a look · ")


def test_failed_pass_rates_never_read_as_an_empty_list(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(
        anchor=None, cards=[_card(pass_pct=None, units=None, lasers=None, months=[])],
        money_total=None, failed={od.PART_RATES: "RuntimeError: invented rates crash"}))
    assert page._other_rows == []
    assert page._others_note.winfo_manager() == "pack"
    assert "Could not be worked out" in page._others_note.cget("text")
    assert page._others_heading.cget("text") == "Everything else"
    assert "newest file" not in page._headline.cget("text")
    assert "the pass rates (RuntimeError: invented rates crash)" in page._load_banner.cget("text")
    facts = page._detail.shown_facts()
    assert facts["Units, last 90 days"] == "—" and facts["Lasers"] == "—"
    assert facts["$ lost at final test, 90 days"] == "could not be worked out"
    # The detail's chart and meter name the failure too -- never "No graded units", never an
    # empty meter that reads as 0% (the review of option B, 2026-10-04).
    from laser_trim_analyzer.gui.v6.pages.home_page import FAILED_NOTE
    chart = page._detail.chart
    assert [chart.itemcget(i, "text") for i in chart.find_withtag("note")] == [FAILED_NOTE]
    assert page._detail.meter.find_all() == () and page._detail.pct.cget("text") == "—"
    (row,) = page._card_rows
    assert row._bars.cget("text") == "—"                           # the row's bars: a dash too


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
    order = page._list.pack_slaves()                     # at the end of the list (option B)
    assert order.index(page._others_frame) < order.index(toggle) < order.index(page._inactive_list)
    assert order[-1] is page._inactive_list
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
    assert order.index(page._load_banner) < order.index(page._headline)
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
        assert _pump_until(app, lambda: [r.model for r in page._card_rows] == ["NEWER"])
    else:
        page.reload_now()
    load.release()                         # the older load finishes LAST
    _pump_ui(app)
    assert [r.model for r in page._card_rows] == ["NEWER"]
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
    assert page._headline.cget("text").startswith("1 model needs a look · ")
    assert page._detail.title.cget("text") == "A"            # the detail draws on its own
    assert page._legacy_ft_label.winfo_manager() == "pack"
    assert page._inactive_toggle.winfo_manager() == "pack"


def test_a_load_with_nothing_new_redraws_nothing(make_app, monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(cards=[_card("A")], others=[_row()]))
    before = (list(page._card_rows), list(page._other_rows))
    page = _show(app, monkeypatch, _ov(cards=[_card("A")], others=[_row()]))
    assert (page._card_rows, page._other_rows) == before


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
    assert page._headline.cget("text").startswith("1 model needs a look · ")
    (row,) = page._card_rows
    assert row.model == "HOT"
    assert page._detail.reason.cget("text") == "A recent run failed 60% · usually 10%"
    assert (row._why.cget("text"), row._why_more.cget("text")) == ("A recent run failed 60%",
                                                                   "usually 10%")
    assert [r.model for r in page._other_rows] == ["CALM"]


# ---- nothing is cut at 1280x720 (the render audit's own detector) --------------------------------

_LONG = ("3 recent runs failed 74% · usually 34% · Untrimmed resistance 21,270 → 23,113 · "
         "still passing")


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


# ---- the final review of the redesign (2026-10-02) ---------------------------------------------

def test_a_card_whose_count_could_not_be_read_prints_none(make_app, monkeypatch):
    """Never "0 units" over a read that failed: the banner names the failure; the card says
    nothing it does not know."""
    page = _show(make_app(), monkeypatch, _ov(
        cards=[_card("7000", units=None, pass_pct=None, was_pct=None)],
        failed={od.PART_RATES: "RuntimeError: invented rates crash"}))
    (row,) = page._card_rows
    texts = _labels(row)
    assert texts[:2] == ["7000", "— units"]
    assert not any(t[:1].isdigit() and "unit" in t for t in texts)


def test_the_caption_says_what_puts_a_model_on_a_card(make_app, monkeypatch):
    """The retired FOCUS list said why a model was on it and when it left; the cards did not."""
    from laser_trim_analyzer.ml.spc import RECENT_K
    page = _show(make_app(), monkeypatch, _ov(cards=[_card()]))
    rule = page._need_rule.cget("text")
    assert rule == od.CARD_RULE
    assert f"last {RECENT_K} runs" in rule and "watched signal" in rule
    order = page._body.pack_slaves()
    assert order.index(page._trend_box) < order.index(page._need_rule) < order.index(page._split)


def test_other_models_on_file_are_one_collapsed_line_above_the_inactive_one(make_app, monkeypatch):
    quiet = {"Q1": datetime(2025, 6, 2), "Q2": datetime(2025, 11, 3)}
    page = _show(make_app(), monkeypatch, _ov(others=[_row()], quiet=quiet,
                                              inactive={"OLD": datetime(2016, 3, 9)}))
    toggle = page._quiet_toggle
    assert toggle.winfo_manager() == "pack" and toggle.cget("text") == "Other models on file (2) ▸"
    assert page._quiet_list.winfo_manager() == ""
    toggle.invoke()
    assert toggle.cget("text") == "Other models on file (2) ▾"
    assert page._quiet_list.winfo_manager() == "pack"
    lines = "\n".join(_labels(page._quiet_list)).split("\n")
    assert lines == ["Q2 · last trimmed Nov 2025", "Q1 · last trimmed Jun 2025"]
    order = page._list.pack_slaves()
    assert (order.index(page._others_frame) < order.index(toggle) < order.index(page._quiet_list)
            < order.index(page._inactive_toggle))
    toggle.invoke()
    assert toggle.cget("text").endswith("▸") and page._quiet_list.winfo_manager() == ""


@pytest.mark.parametrize("quiet", [None, {}])
def test_with_no_other_models_or_none_known_there_is_no_line(make_app, monkeypatch, quiet):
    page = _show(make_app(), monkeypatch, _ov(others=[_row()], quiet=quiet))
    assert page._quiet_toggle.winfo_manager() == ""


def test_a_specific_folder_is_one_quiet_link_away_never_behind_a_run(make_app, monkeypatch):
    """The blue button starts the remembered run. Reaching the one-off folder picker through it
    started a scan of every remembered folder first (final review, 2026-10-02)."""
    app = make_app()
    page = _home(app)
    started = []
    monkeypatch.setattr(app.page_container.get_page("process"), "start_new_files",
                        lambda: started.append(1))
    assert page._process_link.cget("text") == "Process a specific folder"
    assert page._process_link.master is page._links
    page._process_link.invoke()
    assert app.page_container.current_page == "process" and started == []


def test_the_last_runs_line_opens_its_tally_without_starting_a_run(make_app, monkeypatch):
    """A run that ended with failed files lands here; its tally is on the Process page. The blue
    button would start a new run and wipe it -- the line itself opens the page."""
    app = make_app()
    page = _home(app)
    started = []
    monkeypatch.setattr(app.page_container.get_page("process"), "start_new_files",
                        lambda: started.append(1))
    assert page._run_link.winfo_manager() == ""            # no run yet: nothing to open
    page.set_run_summary("2 folders · 14 new files · 1 folder failed", ok=False)
    assert page._run_link.cget("text") == "See the run ›"
    order = page._body.pack_slaves()
    assert order.index(page._run_line) + 1 == order.index(page._run_link)    # right under it
    page._run_link.invoke()
    assert app.page_container.current_page == "process" and started == []



def test_a_model_with_no_trim_file_is_listed_with_the_other_models_and_says_so(make_app, monkeypatch):
    page = _show(make_app(), monkeypatch, _ov(others=[_row()],
                                              quiet={"Q1": datetime(2025, 6, 2), "FT1": None}))
    page._quiet_toggle.invoke()
    lines = "\n".join(_labels(page._quiet_list)).split("\n")
    assert lines == ["Q1 · last trimmed Jun 2025", "FT1 · no trim file on record"]



# ---- each laser, charted (James, 2026-10-04) ----------------------------------------------------

def _spy_trend(monkeypatch):
    from laser_trim_analyzer.gui.v6.widgets.company_trend_chart import CompanyTrendChart
    seen = []
    monkeypatch.setattr(CompanyTrendChart, "set_data",
                        lambda self, trend, period_label="week", note=None: seen.append(
                            (trend, period_label)))
    return seen


def test_the_overview_charts_each_laser_over_the_last_twelve_months(make_app, monkeypatch):
    """James, 2026-10-04: "on the overveiw screen i no longer have each laser charted overall?" --
    the Company trends chart, on top of the Overview: each laser and the company, by month."""
    from laser_trim_analyzer.gui.v6.widgets.company_trend_chart import CompanyTrendChart
    seen = _spy_trend(monkeypatch)
    trend = {"periods": ["2026-02", "2026-03"], "partial_last": True, "data_through": ANCHOR,
             "company": [{"period": "2026-02", "total": 10, "accepted": 8, "linearity_yield": 80.0},
                         {"period": "2026-03", "total": 5, "accepted": 5, "linearity_yield": 100.0}],
             "by_system": {"A": [{"period": "2026-02", "total": 10, "accepted": 8,
                                  "linearity_yield": 80.0},
                                 {"period": "2026-03", "total": 5, "accepted": 5,
                                  "linearity_yield": 100.0}]}}
    page = _show(make_app(), monkeypatch, _ov(cards=[_card()], trend=trend))
    assert isinstance(page._trend_chart, CompanyTrendChart)
    assert seen[-1] == (trend, "month")
    assert page._trend_heading.cget("text") == "Yield by laser — last 12 months"
    assert page._trend_chart.winfo_manager() == "pack" and page._trend_note.winfo_manager() == ""
    order = page._body.pack_slaves()
    assert order.index(page._trend_box) < order.index(page._split)       # always above the list
    # A compact strip (option B): the Company trends chart as it is, only shorter.
    from laser_trim_analyzer.gui.v6.pages.home_page import STRIP_INCHES
    fig = page._trend_chart._fig
    assert int(page._trend_chart.canvas.get_tk_widget().cget("height")) == round(STRIP_INCHES * fig.dpi)


def test_a_chart_that_cannot_load_says_so_and_the_banner_names_it(make_app, monkeypatch):
    seen = _spy_trend(monkeypatch)
    page = _show(make_app(), monkeypatch, _ov(
        cards=[_card()], trend=None, failed={od.PART_TREND: "RuntimeError: invented chart crash"}))
    assert seen[-1] == (None, "month")         # the chart: "Unavailable — the notice above says why."
    assert ("the yield chart by laser (RuntimeError: invented chart crash)"
            in page._load_banner.cget("text"))
    order = page._body.pack_slaves()
    assert order.index(page._load_banner) < order.index(page._trend_box)    # the notice is above it


# ---- option B (2026-10-04): the prices it is handed, and the two panes ---------------------------

def test_the_page_hands_the_configured_prices_and_cost_ratio_to_the_loader(make_app, monkeypatch):
    app = make_app()
    page = _home(app)
    _settle_workers(app)
    app.config.active_models.model_prices = {"INVENTED": 12.5}
    app.config.active_models.cost_ratio = 0.3
    seen = []
    monkeypatch.setattr(od, "load_overview", lambda db, **k: seen.append(k) or _ov())
    page.reload_now()
    page.on_show()
    assert _pump_until(app, lambda: len(seen) == 2)
    for k in seen:
        assert k == {"prices": {"INVENTED": 12.5}, "cost_ratio": 0.3}
    assert seen[0]["prices"] is not app.config.active_models.model_prices   # a copy, not the live dict


def test_the_list_and_the_detail_fill_the_window_below_the_strip(make_app, monkeypatch):
    """TMOG's two panes: the list and the detail reach the foot of the window -- and the page
    itself has nothing to scroll -- not a fixed box with empty window under it."""
    from laser_trim_analyzer.gui.v6.pages.home_page import SPLIT_MIN
    app = make_app()
    page = _show(app, monkeypatch, _ov(cards=[_card(f"C{i}") for i in range(3)],
                                       others=[_row(f"R{i}") for i in range(30)]))
    _mapped(app, "1400x900")
    try:
        for _ in range(3):
            app.update_idletasks()
            app.update()
        view = page._body._parent_canvas
        assert view.yview() == (0.0, 1.0), "the page scrolls: the panes overran the window"
        foot = page._links.winfo_rooty() + page._links.winfo_height()
        assert view.winfo_rooty() + view.winfo_height() - foot < 40, "empty window under the panes"
        assert page._list.cget("height") > SPLIT_MIN
        assert page._list._parent_canvas.yview() != (0.0, 1.0)        # 33 rows: the list scrolls
    finally:
        app.withdraw()


def test_the_wheel_over_a_pane_that_can_scroll_scrolls_that_pane_alone(make_app, monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(
        cards=[_card(f"C{i}") for i in range(3)], others=[_row(f"R{i}") for i in range(30)],
        legacy_ft=1234, unreadable=56,
        failed={od.PART_DRIFT: "RuntimeError: " + "a long invented explanation " * 40}))
    _mapped(app, "960x640")
    try:
        for _ in range(3):
            app.update_idletasks()
            app.update()
        row_label = page._other_rows[0]._pct._label
        assert page._list._parent_canvas.yview() != (0.0, 1.0)
        assert page._body.check_if_master_is_canvas(row_label) is False     # the list's own
        assert page._list.check_if_master_is_canvas(row_label) is True
        assert page._body.check_if_master_is_canvas(page._headline._label) is True
    finally:
        app.withdraw()


def test_a_newly_selected_model_is_read_from_the_top_of_its_detail(make_app, monkeypatch):
    app = make_app()
    page = _show(app, monkeypatch, _ov(cards=[_card("A", reason=_LONG), _card("B", reason=_LONG)]))
    _mapped(app, "960x640")
    try:
        canvas = page._detail._parent_canvas
        assert canvas.yview() != (0.0, 1.0), "the detail fits: nothing to scroll in this test"
        canvas.yview_moveto(1.0)
        app.update()
        assert canvas.yview()[0] > 0.0
        page._select("B")
        app.update()
        assert canvas.yview()[0] == 0.0
    finally:
        app.withdraw()
