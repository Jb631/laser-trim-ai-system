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


def test_load_focus_returns_an_empty_result_when_the_computation_blows_up(
        monkeypatch, make_app):
    """A crash must read as 'no data', never as a clean shop floor with no log."""
    from laser_trim_analyzer.gui.v6 import focus_data

    def boom(_db):
        raise RuntimeError("db gone")

    monkeypatch.setattr(focus_data, "compute_focus_list", boom)
    app = make_app()
    result, last = focus_data.load_focus(app.db)
    assert result.focus == [] and result.chronic == []


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


def test_the_folders_and_summary_lines_wrap_to_their_container(make_app):
    """global-constraints.md: no fixed pixel wraplength on page-width text -- blocks.wrap_to_width,
    not the old fixed _WRAP=950 constant. Same off-screen-mapped technique as
    test_the_full_width_home_lines_fit_at_1280_by_720, below."""
    app = make_app()
    page = _home(app)
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
        assert page._folders_label.cget("wraplength") == container_width
        assert page._summary.cget("wraplength") == container_width
        assert page._folders_label.cget("wraplength") != 950
    finally:
        app.withdraw()


def test_still_exactly_one_teal_button(make_app):
    app = make_app()
    _seed_one_file(app.db, "BIG")
    app.db.replace_process_findings("BIG", {"tracks": 1}, [_finding("BIG", "big one", 500.0)])
    page = _home(app)
    page.reload_now()
    t = page.theme
    teal = [b for b in _buttons(page) if b.cget("fg_color") == t.ACCENT]
    assert len(teal) == 1 and teal[0].cget("text") == "Process everything new"
