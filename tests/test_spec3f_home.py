"""Spec 3f — HOME + the shell consolidation.

Spec: docs/superpowers/specs/2026-08-29-app-shape-investigate-design.md §1
(HOME: one-click ingest, the FOCUS list below it, the specific-folder picker
still reachable) and the build order's step 3 (nav becomes HOME · INVESTIGATE ·
SETTINGS, Dashboard de-emphasized pending James's retirement call).

Two decisions these tests exist to defend:

  * Route KEYS did not change. FOCUS rows and every deep link navigate to the
    key "model"; only its label became "Investigate". A rename would silently
    break click-through for a cosmetic win.
  * Nothing reachable was lost (spec line 65). Dashboard, Triage and Process
    are de-emphasized in the sidebar, not deleted, and each is still one
    show_page away.
"""
from datetime import datetime, timedelta

import pytest

from laser_trim_analyzer.core.ingest_run import FolderResult, IngestReport
from laser_trim_analyzer.ml.spc import FocusEntry, FocusResult, build_fraction_series


# ---- fixtures --------------------------------------------------------------

D0 = datetime(2026, 1, 5)


def _entry(model="8340-1"):
    """A FocusEntry around a REAL series — same construction as the zone tests."""
    hist = []
    for k in range(11):
        hist += [(D0 + timedelta(days=7 * k), 1.0 if i < 2 else 0.0)
                 for i in range(20)]
    last = D0 + timedelta(days=7 * 11)
    hist += [(last, 1.0 if i < 12 else 0.0) for i in range(20)]
    series = build_fraction_series(model, "linearity_fail_fraction", hist,
                                   anchor=last + timedelta(days=3))
    return FocusEntry(model=model, series=series, excess_per_week=12.0,
                      units_per_week=140.0, p_base=series.p_base, p_recent=0.6,
                      n_flagged_recent=1, last_lot_end=series.points[-1].end,
                      verdict="failing ~12 more units/week than its own baseline",
                      sub_line="1 of last 5 lots out of control")


def _labels(widget):
    import customtkinter as ctk
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
            out.append(c.cget("text"))
        out.extend(_labels(c))
    return out


def _home(app):
    return app.page_container.get_page("home")


def _with_folders(app, *folders):
    for f in folders:
        app.config.ingest.add(f)
    return _home(app)


class _NoThread:
    """Stand-in for threading.Thread: records, never starts."""
    started = []

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target, self.args = target, args or ()

    def start(self):
        _NoThread.started.append((self.target, self.args))


def _no_threads(monkeypatch):
    """Freeze thread spawning INSIDE home_page only.

    Patching `home_mod.threading.Thread` would reach through to the real
    threading module and silently freeze every other thread in the process
    (the Settings page's folder probe landed in this recorder that way).
    Replacing the module reference in home_page's namespace does not."""
    import types
    import laser_trim_analyzer.gui.v6.pages.home_page as home_mod
    _NoThread.started = []
    monkeypatch.setattr(home_mod, "threading",
                        types.SimpleNamespace(Thread=_NoThread))
    return _NoThread


# ---- Task 1: the shell -----------------------------------------------------

def test_sidebar_order_and_labels():
    """Home · Investigate · Settings, then the de-emphasized three.
    "model" keeps its key — only the label reads "Investigate".
    Findings added 2026-09-20 (process-findings engine), deliberately, after Investigate."""
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar
    assert Sidebar.ITEMS == [
        ("home", "Home"), ("model", "Investigate"), ("findings", "Findings"), ("settings", "Settings"),
        ("dashboard", "Dashboard"), ("triage", "Triage"), ("process", "Process"),
    ]


def test_sidebar_marks_the_de_emphasized_group():
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar
    assert Sidebar.MUTED == {"dashboard", "triage", "process"}
    assert Sidebar.SEPARATOR_AFTER == "settings"


def test_sidebar_still_emits_the_model_key_not_the_label(tk_root):
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    got = []
    sb = Sidebar(tk_root, on_select=got.append, theme=ThemeManager())
    sb._row_frames["model"]._on_click()
    assert got == ["model"]


def test_home_is_the_landing_page(make_app):
    app = make_app()
    assert app.page_container.current_page == "home"
    assert app.sidebar._active_name == "home"


def test_every_pre_existing_route_is_still_reachable(make_app):
    """Spec line 65: nothing currently reachable is lost."""
    app = make_app()
    for key in ("dashboard", "triage", "process", "model", "settings", "home"):
        app.show_page(key)
        assert app.page_container.current_page == key, key
        assert app.sidebar._active_name == key, key


def test_model_route_still_lands_on_the_model_page(make_app):
    """FOCUS click-through: set_model_route + show_page('model') unchanged.

    Asserted through the page that consumes it — on_show() pops the route, so
    a test that re-reads the hint afterwards would prove only that something
    ate it."""
    app = make_app()
    app.set_model_route("8340-1", "linearity_fail_fraction")
    app.show_page("model")
    assert app.page_container.current_page == "model"
    assert app.page_container.get_page("model")._current_model == "8340-1"


# ---- Task 2: the empty state ----------------------------------------------

def test_home_empty_state_points_at_settings(make_app):
    app = make_app()                       # fresh Config → no folders
    page = _home(app)
    assert str(page._run_button.cget("state")) == "disabled"
    text = " ".join(_labels(page))
    assert "Settings" in text
    assert "folder" in text.lower()


def test_home_empty_state_is_not_a_modal(make_app):
    """A blocking dialog on every cold start would be intolerable; the empty
    state is a line of text and a button."""
    app = make_app()
    page = _home(app)
    assert page.winfo_exists()
    assert app.page_container.current_page == "home"


def test_home_empty_state_button_opens_settings(make_app):
    app = make_app()
    page = _home(app)
    page._open_settings()
    assert app.page_container.current_page == "settings"


def test_home_enables_the_button_once_folders_exist(make_app, tmp_path):
    app = make_app()
    page = _with_folders(app, str(tmp_path))
    page.refresh_folders()
    assert str(page._run_button.cget("state")) == "normal"
    assert str(tmp_path) in " ".join(_labels(page))


# ---- Task 3: the run -------------------------------------------------------

def test_home_run_drives_the_shared_multi_folder_runner(make_app, monkeypatch):
    """No second pipeline: Home calls run_folders, in configured order."""
    from laser_trim_analyzer.core import ingest_run
    seen = {}

    def fake(folders, **kw):
        seen["folders"] = list(folders)
        seen["incremental"] = kw.get("incremental")
        seen["db"] = kw.get("db")
        return IngestReport(results=[], seconds=1.0)

    monkeypatch.setattr(ingest_run, "run_folders", fake)
    app = make_app()
    page = _with_folders(app, "/laser/a", "/laser/b", "/final_test")
    page._run(["/laser/a", "/laser/b", "/final_test"], True)
    assert seen["folders"] == ["/laser/a", "/laser/b", "/final_test"]
    assert seen["incremental"] is True
    assert seen["db"] is app.db


def test_button_is_disabled_while_a_run_is_in_flight(make_app, monkeypatch,
                                                     tmp_path):
    _no_threads(monkeypatch)
    app = make_app()
    page = _with_folders(app, str(tmp_path))
    page.refresh_folders()
    assert str(page._run_button.cget("state")) == "normal"
    page._start()
    assert str(page._run_button.cget("state")) == "disabled"
    assert _NoThread.started, "a worker should have been spawned"
    # ...and comes back when the run reports in.
    page._on_run_done(IngestReport(results=[], seconds=1.0))
    assert str(page._run_button.cget("state")) == "normal"


def test_start_is_a_no_op_with_no_folders(make_app, monkeypatch):
    _no_threads(monkeypatch)
    app = make_app()
    page = _home(app)
    _NoThread.started = []          # drop the FOCUS load the landing show kicks off
    page._start()
    assert _NoThread.started == []


def test_home_shows_the_combined_summary(make_app):
    from laser_trim_analyzer.core.ingest_run import format_ingest_summary
    app = make_app()
    page = _home(app)
    report = IngestReport(
        results=[FolderResult(folder=f"/f{i}", ok=True, files_found=100,
                              new_files=n, seconds=1.0)
                 for i, n in enumerate((100, 100, 14))],
        seconds=160.0)
    page._on_run_done(report)
    assert page._summary.cget("text") == format_ingest_summary(report)
    assert "3 folders · 214 new files · 2 min 40 s" in page._summary.cget("text")


def test_home_summary_names_a_folder_that_failed(make_app):
    app = make_app()
    page = _home(app)
    report = IngestReport(results=[
        FolderResult(folder="/laser/a", ok=True, files_found=10, new_files=4),
        FolderResult(folder="\\\\192.168.66.9\\Laser", ok=False,
                     error="not found — offline share?")], seconds=61.0)
    page._on_run_done(report)
    text = page._summary.cget("text")
    assert "\\\\192.168.66.9\\Laser" in text and "1 of 2 folders failed" in text


# ---- Task 4: the FOCUS list ------------------------------------------------

def test_home_uses_the_shared_focus_zone_and_computation(make_app):
    """Same widget, same loader as Triage — not a copy of either."""
    from laser_trim_analyzer.gui.v6 import focus_data
    from laser_trim_analyzer.gui.v6.pages import home_page, triage_page
    from laser_trim_analyzer.gui.v6.widgets.focus_list_zone import FocusListZone
    app = make_app()
    assert isinstance(_home(app)._focus, FocusListZone)
    assert home_page.load_focus is focus_data.load_focus
    assert triage_page.load_focus is focus_data.load_focus


def test_home_renders_the_focus_result_it_is_given(make_app):
    app = make_app()
    page = _home(app)
    entry = _entry()
    page._apply_focus(FocusResult(focus=[entry], chronic=[],
                                  anchor=entry.last_lot_end), None)
    text = " ".join(_labels(page))
    assert "8340-1" in text
    assert entry.verdict in text


def test_focus_row_click_routes_to_the_model_page(make_app):
    app = make_app()
    page = _home(app)
    page._on_focus_click("8340-1", "linearity_fail_fraction")
    assert app.page_container.current_page == "model"
    assert app.page_container.get_page("model")._current_model == "8340-1"


def test_load_focus_never_raises_but_a_crash_is_distinguishable_from_empty(
        monkeypatch, make_app):
    """Still non-raising for its callers (the lists are empty, so iterating works), but a crash
    is marked -- it used to come back as the very EMPTY a clean shop floor returns, so Home said
    "0 drifting now" and Triage "Needs a look · 0" over a crash (final review, 2026-09-24)."""
    from laser_trim_analyzer.gui.v6 import focus_data

    def boom(_db):
        raise RuntimeError("db gone")

    monkeypatch.setattr(focus_data, "compute_focus_list", boom)
    app = make_app()
    result, last = focus_data.load_focus(app.db)
    assert result.focus == [] and result.chronic == []
    assert focus_data.focus_failed(result) == "RuntimeError: db gone"
    monkeypatch.undo()
    healthy, _ = focus_data.load_focus(app.db)
    assert focus_data.focus_failed(healthy) is None


def test_load_focus_on_an_empty_database_is_empty(make_app):
    from laser_trim_analyzer.gui.v6 import focus_data
    app = make_app()
    result, last = focus_data.load_focus(app.db)
    assert result.focus == [] and last is None


# ---- Task 5: the specific-folder escape hatch ------------------------------

def test_specific_folder_link_routes_to_the_process_page(make_app):
    app = make_app()
    _home(app)._open_process()
    assert app.page_container.current_page == "process"


def test_home_offers_the_specific_folder_wording(make_app):
    app = make_app()
    text = " ".join(_labels(_home(app))
                    + [b.cget("text") for b in _buttons(_home(app))])
    assert "specific folder" in text.lower()


def _buttons(widget):
    import customtkinter as ctk
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkButton):
            out.append(c)
        out.extend(_buttons(c))
    return out


# ---- the full-width lines fit at 1280 wide (final review, 2026-09-24) -------

def test_the_full_width_home_lines_fit_at_1280_by_720(make_app):
    """Four Home lines wrapped at 1100 px inside a card ~1,064 px wide at 1280x720 -- clipped as soon
    as they held a real folder list. Measured with the audit's own detector, on a real mapped window
    (invisible: alpha 0), with invented folder names long enough to wrap."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from render_pages import find_clipped_text_widgets

    app = make_app()
    page = _home(app)
    folders = "  →  ".join(f"/invented/share/laser-{i}/Trim Data/Production line {i}" for i in range(1, 9))
    page._folders_label.configure(text=f"8 folders, in this order:  {folders}")
    page._summary.configure(text=f"Summary line {folders}")
    for label, lead in ((page._legacy_ft_label, "Legacy line"), (page._unreadable_label, "Unreadable line")):
        label.configure(text=f"{lead} {folders}")
        label.pack(side="top", fill="x")
    try:
        app.attributes("-alpha", 0.0)
    except Exception:
        pass
    app.geometry("1280x720+20000+20000")
    app.deiconify()
    app.update_idletasks()
    app.update()
    try:
        leads = ("8 folders", "Summary line", "Legacy line", "Unreadable line")
        ours = [c for c in find_clipped_text_widgets(page, page="home", window_size="1280x720")
                if c.text.startswith(leads)]
        assert not ours, [c.line() for c in ours]
    finally:
        app.withdraw()


def test_the_home_notices_sit_above_the_focus_list_never_below_it(make_app):
    """The focus list expands to fill the page. A notice packed AFTER it got no height at all on a
    720-px-tall window -- the audit found "70 files are being skipped..." squeezed out entirely."""
    app = make_app()
    page = _home(app)
    page._apply_legacy_ft(12)
    page._apply_unreadable(70)
    order = page._focus_header.master.pack_slaves()
    assert page._legacy_ft_label in order and page._unreadable_label in order
    assert order.index(page._legacy_ft_label) < order.index(page._focus_header)
    assert order.index(page._unreadable_label) < order.index(page._focus_header)
    assert order.index(page._focus_header) < order.index(page._focus)


# ---- Task 4 (facelift step 2): "Worth changing" (design doc section 2, ruling 2) ------------
#
# Caption = "Last processed {date} · {N} worth changing · {M} drifting now"; N is the yield
# group's row count (the same group FindingsView draws when it is given groups=("yield",));
# M is len(FocusResult.focus) -- both already known to Home via load_focus, no second query.

def _finding(model, title, tracks_per_year, size=100, analyzer="ink_target", category="Ink target"):
    # Same shape test_findings_page.py's own helper uses -- "lever" is a required column on
    # ProcessFinding (database/manager.replace_process_findings reads it as d["lever"], not
    # .get). analyzer="ink_target" (the default here) lands in the "yield" group.
    return {"model": model, "analyzer": analyzer, "category": category, "lever": "ink",
            "title": title, "summary": f"summary for {model}", "systems": ["B"], "n_units": size,
            "tracks_per_year": tracks_per_year, "evidence": {}}


def _seed_one_file(db, model="8340-1", day=None):
    """One processed file -- exactly what list_known_models (load_focus's own "last processed"
    lookup) reads. Required columns per test_focus_list_zone.py's own DB-seeding helper."""
    from laser_trim_analyzer.database.models import AnalysisResult, StatusType, SystemType
    day = day or datetime(2026, 9, 20, 14, 30)
    with db.session() as s:
        s.add(AnalysisResult(model=model, serial=f"{model}-seed-1", system=SystemType.A,
                             filename=f"{model}_seed.xls", file_date=day,
                             overall_status=StatusType.PASS))
    return day


def test_home_caption_reports_worth_changing_and_drifting_now(make_app):
    """N = 2: BIG and SMALL are both analyzer="ink_target" (group "yield") and different
    models, so they never merge into one row. M = 0: a single seeded file has no lot history
    to drift against. Date via f"{dt.day} {dt:%b}" -- never %-d (raises on Windows)."""
    app = make_app()
    day = _seed_one_file(app.db, "BIG")
    app.db.replace_process_findings("BIG", {"tracks": 1}, [_finding("BIG", "big one", 500.0)])
    app.db.replace_process_findings(
        "SMALL", {"tracks": 1}, [_finding("SMALL", "no rate here", None, size=9000)])
    page = _home(app)
    page.reload_now()
    assert page._caption.cget("text") == (
        f"Last processed {day.day} {day:%b} · 2 worth changing · 0 drifting now")


def test_worth_changing_section_shows_yield_rows_and_an_open_findings_link(make_app):
    app = make_app()
    _seed_one_file(app.db, "BIG")
    app.db.replace_process_findings("BIG", {"tracks": 1}, [_finding("BIG", "big one", 500.0)])
    page = _home(app)
    page.reload_now()
    assert page._worth_view is not None
    assert len(page._worth_view.row_widgets) == 1
    assert "BIG" in " ".join(_labels(page._worth_section))
    assert "Open Findings" in [b.cget("text") for b in _buttons(page._worth_section)]


def test_worth_changing_row_open_routes_exactly_like_the_findings_page(make_app, monkeypatch):
    """Context: route exactly as findings_page.py::_open does -- set_model_route(model,
    tab="findings") then show_page("model") -- never the FOCUS list's (model, metric) form."""
    app = make_app()
    page = _home(app)
    shown = []
    monkeypatch.setattr(app, "show_page", lambda name: shown.append(name))
    page._open_finding("BIG")
    assert shown == ["model"]
    assert app.consume_model_route() == "BIG"
    assert app.consume_model_tab() == "findings"


def test_worth_changing_is_a_quiet_line_not_a_blank_gap_when_the_yield_group_is_empty(make_app):
    app = make_app()
    _seed_one_file(app.db, "BIG")
    # A real finding, but not in the "yield" group -- the section must still say something,
    # never leave a blank gap under the "Worth changing" heading.
    app.db.replace_process_findings(
        "BIG", {"tracks": 1},
        [_finding("BIG", "Laser 1 (LTS): recipe changed from 4000 to 4100", None,
                  analyzer="recipe_change", category="Recipe")])
    page = _home(app)
    page.reload_now()
    assert page._worth_view is None
    assert page._worth_section.winfo_children(), "a blank gap, not a quiet line"
    assert "Nothing here yet" in " ".join(_labels(page._worth_section))


def test_worth_changing_is_a_quiet_line_with_no_findings_cached_at_all(make_app):
    app = make_app()
    _seed_one_file(app.db, "BIG")
    page = _home(app)
    page.reload_now()
    assert page._worth_view is None
    assert "Nothing here yet" in " ".join(_labels(page._worth_section))


def test_a_failed_findings_load_is_a_banner_never_a_quiet_nothing_line(make_app, monkeypatch):
    """CLAUDE.md: a failure must never look like a result -- not "nothing worth changing",
    and not a caption that states an N it could not actually compute."""
    app = make_app()
    _seed_one_file(app.db, "BIG")

    def boom():
        raise RuntimeError("database is locked")
    monkeypatch.setattr(app.db, "get_process_findings", boom)
    page = _home(app)
    page.reload_now()
    assert page._worth_view is None
    text = " ".join(_labels(page._worth_section))
    assert "Nothing here yet" not in text
    banner_text = page._worth_banner.cget("text")
    assert "findings" in banner_text.lower() and "RuntimeError: database is locked" in banner_text
    assert page._worth_banner.winfo_manager() == "pack"
    assert "worth changing" not in page._caption.cget("text")


def test_the_two_notices_render_as_quiet_banners(make_app):
    """Step 1: "the two notices (legacy final tests, unreadable files) are quiet banners" --
    blocks.banner(tone="quiet"), not the old bare CTkLabel with ad-hoc TIER_WARNING colour."""
    app = make_app()
    page = _home(app)
    t = page.theme
    page._apply_legacy_ft(12)
    page._apply_unreadable(70)
    assert page._legacy_ft_label.cget("fg_color") == t.CARD
    assert page._legacy_ft_label.cget("text_color") == t.TEXT_SECONDARY
    assert page._unreadable_label.cget("fg_color") == t.CARD
    assert page._unreadable_label.cget("text_color") == t.TEXT_SECONDARY


def test_the_focus_zone_is_titled_drifting_now(make_app):
    """Design doc: "two 'what the app is telling you' headings on one page would say
    nothing" -- the new findings section takes the generic wording; this one gets specific."""
    app = make_app()
    page = _home(app)
    text = " ".join(_labels(page))
    assert "Drifting now" in text
    assert "What the app is telling you" not in text


@pytest.mark.parametrize("scale", (1.0, 1.5))
def test_the_folders_and_summary_lines_wrap_to_their_container(make_app, scale):
    """global-constraints.md: no fixed pixel wraplength on page-width text -- blocks.wrap_to_width,
    not the old fixed _WRAP=950 constant. Same off-screen-mapped technique as
    test_the_full_width_home_lines_fit_at_1280_by_720, below.

    At 150% too (final review, 2026-09-24): this used to pin `wraplength == container width`,
    which is only true at 100% -- wraplength is in CustomTkinter's unscaled units, the width is
    real pixels, and on the Windows laptop the widget scaling is the monitor's DPI factor."""
    import customtkinter as ctk
    ctk.set_widget_scaling(scale)
    try:
        app = make_app()
        page = _home(app)
        folders = "  →  ".join(f"/invented/share/laser-{i}/Trim Data" for i in range(1, 9))
        page._folders_label.configure(text=f"8 folders, in this order:  {folders}")
        page._summary.configure(text=f"Summary line {folders}")
        try:
            app.attributes("-alpha", 0.0)
        except Exception:
            pass
        app.geometry("1280x720+20000+20000")
        app.deiconify()
        app.update_idletasks()
        app.update()
        try:
            container_width = page._folders_label.master.winfo_width()
            for label in (page._folders_label, page._summary):
                assert label.cget("wraplength") == int(container_width / scale)
                assert label._label.winfo_reqwidth() <= container_width, (
                    f"{label.cget('text')[:20]!r} laid out {label._label.winfo_reqwidth()} px "
                    f"wide in a {container_width} px container at {scale:.0%}")
            assert page._folders_label.cget("wraplength") != 950
        finally:
            app.withdraw()
    finally:
        ctk.set_widget_scaling(1.0)


def test_still_exactly_one_teal_button(make_app):
    app = make_app()
    _seed_one_file(app.db, "BIG")
    app.db.replace_process_findings("BIG", {"tracks": 1}, [_finding("BIG", "big one", 500.0)])
    page = _home(app)
    page.reload_now()
    t = page.theme
    # CRITICAL (review, 7bc0743): expanding a "Worth changing" row draws findings_view.py's own
    # "Open <model>" button -- before FindingsView grew open_as="link" for Home's use, that was
    # a SECOND blocks.primary_button next to "Process everything new". Red on the code before
    # that fix: the un-expanded check alone let it ship once already.
    page._worth_view.toggle(next(iter(page._worth_view.row_widgets)))
    teal = [b for b in _buttons(page) if b.cget("fg_color") == t.ACCENT]
    assert len(teal) == 1 and teal[0].cget("text") == "Process everything new"


# ---- final review (2026-09-24): "Worth changing" never says "nothing" when nothing was worked out,
# the ingest notices sit in "Bring in what's new", and the focus list has ONE heading -----------

def _check_banner_text(page):
    """The Worth-changing banner's text, when it is laid out (winfo_manager, never ismapped)."""
    return page._worth_banner.cget("text") if page._worth_banner.winfo_manager() else ""


def test_a_model_whose_analyzer_crashed_is_named_on_home(make_app):
    """The Findings page banners "N model(s) could not be fully worked out"; Home read the same
    cache and said nothing -- so a crashed analyzer's missing finding looked like no finding."""
    app = make_app()
    _seed_one_file(app.db, "BIG")
    app.db.replace_process_findings("BIG", {"tracks": 1}, [_finding("BIG", "big one", 500.0)])
    app.db.replace_process_findings("BAD", {"tracks": 1, "errors": {"cut_setting": "ValueError: x"}}, [])
    page = _home(app)
    page.reload_now()
    text = _check_banner_text(page)
    assert "1 model(s) could not be fully worked out" in text and "BAD" in text
    assert page._worth_banner.cget("fg_color") == page.theme.CHECK_TINT
    assert page._worth_view is not None                     # BIG's real row still shows


def test_when_the_failed_models_cannot_be_read_home_says_so(make_app, monkeypatch):
    app = make_app()
    _seed_one_file(app.db, "BIG")
    app.db.replace_process_findings("BIG", {"tracks": 1}, [_finding("BIG", "big one", 500.0)])

    def boom():
        raise RuntimeError("database is locked")
    monkeypatch.setattr(app.db, "get_process_errors", boom)
    page = _home(app)
    page.reload_now()
    text = _check_banner_text(page)
    assert "could not be checked" in text and "RuntimeError: database is locked" in text
    assert page._worth_view is not None


def test_with_nothing_worked_out_the_caption_claims_no_zero(make_app):
    """No cached findings at all -- never worked out, or nothing anywhere: unknown to this page, so
    the caption says nothing about it rather than "0 worth changing"."""
    app = make_app()
    _seed_one_file(app.db, "BIG")
    page = _home(app)
    page.reload_now()
    assert page._caption.cget("text").startswith("Last processed")
    assert "worth changing" not in page._caption.cget("text")
    assert "Nothing here yet" in " ".join(_labels(page._worth_section))


def test_the_ingest_notices_belong_to_bring_in_whats_new(make_app):
    """M2: the two notices are about ingest; packed before "Drifting now" they sat under the
    "Worth changing" findings, which they have nothing to do with."""
    app = make_app()
    page = _home(app)
    page._apply_legacy_ft(12)
    page._apply_unreadable(70)
    order = page._body.pack_slaves()
    for notice in (page._legacy_ft_label, page._unreadable_label):
        assert order.index(notice) < order.index(page._worth_header)
    assert order.index(page._worth_header) < order.index(page._worth_section)


def test_the_drifting_now_list_has_one_heading(make_app):
    """M3: the zone header already says "Drifting now"; the list's own "FOCUS — drifting now"
    heading under it said it twice."""
    app = make_app()
    page = _home(app)
    assert page._focus._heading is None
    assert not any(t.startswith("FOCUS —") for t in _labels(page))



def _crash_focus(monkeypatch):
    import laser_trim_analyzer.gui.v6.focus_data as fd

    def boom(_db):
        raise RuntimeError("invented focus crash")
    monkeypatch.setattr(fd, "compute_focus_list", boom)


def test_a_focus_crash_is_a_banner_and_never_zero_drifting_now(make_app, monkeypatch):
    """I2 (final review, 2026-09-24): the crash read as "0 drifting now" in the caption and
    "All models within tolerance" in the list."""
    app = make_app()
    _seed_one_file(app.db, "BIG")
    _crash_focus(monkeypatch)
    page = _home(app)
    page.reload_now()
    assert page._focus_banner.winfo_manager() == "pack"
    assert "RuntimeError: invented focus crash" in page._focus_banner.cget("text")
    assert page._focus_banner.cget("fg_color") == page.theme.CHECK_TINT
    caption = page._caption.cget("text")
    assert caption.startswith("Last processed") and "drifting now" not in caption
    zone = " ".join(_labels(page._focus))
    assert "within tolerance" not in zone and "Unavailable" in zone


def test_a_good_focus_load_after_a_crash_clears_the_banner(make_app, monkeypatch):
    app = make_app()
    _seed_one_file(app.db, "BIG")
    _crash_focus(monkeypatch)
    page = _home(app)
    page.reload_now()
    assert page._focus_banner.winfo_manager() == "pack"
    monkeypatch.undo()
    page.reload_now()
    assert page._focus_banner.winfo_manager() == ""
    assert page._caption.cget("text").endswith("0 drifting now")


# ---- facelift F4 (2026-09-25): an older load never overwrites a newer one -----------------------
# Home's loads run on worker threads and apply through safe_after, in whatever order they FINISH.
# The re-review watched the app's own start-up load land after a newer one and wipe it (both
# banners gone, the caption blanked). The Model page has guarded against this with a reload
# generation since I3; Home and Triage now do too.

def _settle_workers(app, seconds=10.0):
    """Let the workers the app started itself (Home's start-up loads) finish and apply, so the
    loads a test starts are the only ones in flight."""
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


@pytest.mark.parametrize("path", ("async", "sync"))
@pytest.mark.parametrize("broken", ("focus", "legacy", "unreadable"))
def test_one_render_error_in_the_focus_apply_never_stops_the_other_two(
        make_app, monkeypatch, caplog, broken, path):
    """F4 review (Minor 4): since F4, FOCUS and the two ingest notices apply as ONE closure, so a
    render error in the FOCUS list stopped both notices -- and safe_after swallowed it. Each of the
    three is guarded on its own now (the Model page's _try: log, go on), under the one counter
    check."""
    import logging
    import laser_trim_analyzer.gui.v6.pages.home_page as home_mod
    app = make_app()
    _seed_one_file(app.db, "BIG")
    page = _home(app)
    _settle_workers(app)
    monkeypatch.setattr(home_mod, "legacy_ft_count", lambda db: 5)
    monkeypatch.setattr(home_mod, "unreadable_count", lambda db: 7)

    def boom(*a, **k):
        raise RuntimeError(f"invented {broken} render crash")
    if broken == "focus":
        monkeypatch.setattr(page._focus, "set_result", boom)
    else:
        monkeypatch.setattr(home_mod, f"{broken}_notice" if broken == "unreadable"
                            else "legacy_ft_notice", boom)
    with caplog.at_level(logging.ERROR):
        if path == "async":
            page._reload_focus()
            _pump_until(app, lambda: f"invented {broken} render crash" in caplog.text)
            _pump_ui(app)
        else:
            page.reload_now()
    assert f"invented {broken} render crash" in caplog.text          # logged, never silent
    if broken != "legacy":
        assert page._legacy_ft_label.winfo_manager() == "pack"
    if broken != "unreadable":
        assert page._unreadable_label.winfo_manager() == "pack"
    if broken != "focus":
        assert page._focus_count == 0 and "0 drifting now" in page._caption.cget("text")


@pytest.mark.parametrize("newer", ("async", "sync"))
def test_an_older_focus_load_never_overwrites_a_newer_one(make_app, monkeypatch, newer):
    import laser_trim_analyzer.gui.v6.pages.home_page as home_mod
    from laser_trim_analyzer.gui.v6.focus_data import FocusLoadFailed
    app = make_app()
    _seed_one_file(app.db, "BIG")
    page = _home(app)
    _settle_workers(app)
    load = _older_then_newer(
        lambda: (FocusResult(focus=[], chronic=[], anchor=None), D0),               # older: healthy
        lambda: (FocusLoadFailed(focus=[], chronic=[], anchor=None,
                                 error="RuntimeError: invented focus crash"), D0))  # newer: crashed
    monkeypatch.setattr(home_mod, "load_focus", load)
    page._reload_focus()                       # the older load, still in its query...
    load.started()
    if newer == "async":                       # ...when a newer one starts and finishes first
        page._reload_focus()
        assert _pump_until(app, lambda: page._focus_banner.winfo_manager() == "pack")
    else:
        page.reload_now()
    assert page._focus_banner.winfo_manager() == "pack"
    load.release()                             # the older load finishes LAST
    _pump_ui(app)
    assert page._focus_banner.winfo_manager() == "pack", "an older load overwrote a newer one"
    assert "invented focus crash" in page._focus_banner.cget("text")
    assert "drifting now" not in page._caption.cget("text")


def test_an_older_findings_load_never_overwrites_a_newer_one(make_app, monkeypatch):
    app = make_app()
    _seed_one_file(app.db, "BIG")
    page = _home(app)
    _settle_workers(app)
    load = _older_then_newer(
        lambda: {"rows": [], "failed": "RuntimeError: invented findings crash", "errors": {},
                 "errors_failed": None},                                            # older: crashed
        lambda: {"rows": [], "failed": None, "errors": {}, "errors_failed": None})  # newer: healthy
    monkeypatch.setattr(page, "_query_findings", load)
    page._reload_findings()
    load.started()
    page._reload_findings()
    assert _pump_until(app, lambda: "Nothing here yet" in " ".join(_labels(page._worth_section)))
    load.release()
    _pump_ui(app)
    assert page._worth_banner.winfo_manager() == "", "an older failure came back over a newer load"
    assert "Nothing here yet" in " ".join(_labels(page._worth_section))


# ---- facelift F4 (2026-09-25): the failure banners wrap to the page ---------------------------

_LONG = ("invented crash with a long explanation that goes on for a while " * 3).strip()


@pytest.mark.parametrize("scale", (1.0, 1.5))
def test_the_failure_banners_and_notices_wrap_to_the_page(make_app, monkeypatch, scale):
    """Re-review Minor 2: each banner wrapped at a fixed 1000 units, and at the app's minimum
    960x640 every failure text on Home was cut, at 100% and at 150%. Checked with the audit's own
    detector: nothing on the page cut at 1280x720, and none of the four texts cut at 960x640 (the
    rest of that size is TRACKER D7's question)."""
    import customtkinter as ctk
    import laser_trim_analyzer.gui.v6.focus_data as fd
    import laser_trim_analyzer.gui.v6.pages.home_page as home_mod
    from test_blocks import failure_texts_cut

    def boom(_db):
        raise RuntimeError(_LONG)
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    try:
        monkeypatch.setattr(fd, "compute_focus_list", boom)
        monkeypatch.setattr(home_mod, "legacy_ft_count", lambda db: 1234)
        monkeypatch.setattr(home_mod, "unreadable_count", lambda db: 56)
        app = make_app()
        _seed_one_file(app.db, "BIG")
        page = _home(app)
        monkeypatch.setattr(page, "_query_findings", lambda: {
            "rows": [], "failed": f"RuntimeError: {_LONG}", "errors": {}, "errors_failed": None})
        page.reload_now()
        texts = [page._focus_banner, page._worth_banner, page._legacy_ft_label,
                 page._unreadable_label]
        assert all(w.winfo_manager() == "pack" and w.cget("text") for w in texts)
        cut = failure_texts_cut(app, page, texts)
        assert cut["1280x720"] == ([], []), cut["1280x720"]
        assert not cut["960x640"][0], cut["960x640"][0]
    finally:
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)


# ---- F5 (2026-09-25): "Inactive" models -- labelled, never hidden ------------------------------

def test_worth_changing_lists_active_models_first_and_still_counts_the_inactive_one(make_app):
    from test_findings_page import _seed_inactive
    app = make_app()
    tag = _seed_inactive(app)
    app.db.replace_process_findings("OLD", {"tracks": 1}, [_finding("OLD", "old one", 900.0)])
    for i in range(3):
        app.db.replace_process_findings(f"LIVE{i}", {"tracks": 1},
                                        [_finding(f"LIVE{i}", f"live {i}", 800.0 - i)])
    page = _home(app)
    page.reload_now()
    assert [k[1] for k in page._worth_view.row_widgets] == ["LIVE0", "LIVE1", "LIVE2"]
    assert "4 worth changing" in page._caption.cget("text")                     # still counted
    page._worth_view.show_all("yield")
    assert [k[1] for k in page._worth_view.row_widgets][0] == "OLD"
    assert tag in _labels(page._worth_section)


def test_home_says_so_when_which_models_are_inactive_cannot_be_read(make_app, monkeypatch):
    import laser_trim_analyzer.gui.v6.pages.home_page as home_mod

    def boom(db):
        raise RuntimeError("invented activity crash")
    app = make_app()
    _seed_one_file(app.db, "BIG")
    app.db.replace_process_findings("BIG", {"tracks": 1}, [_finding("BIG", "big one", 500.0)])
    monkeypatch.setattr(home_mod, "load_activity", boom)
    page = _home(app)
    page.reload_now()
    assert page._worth_banner.winfo_manager() == "pack"
    assert "Which models are inactive could not be worked out" in page._worth_banner.cget("text")
    assert len(page._worth_view.row_widgets) == 1
