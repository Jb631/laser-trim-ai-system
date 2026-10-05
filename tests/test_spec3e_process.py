"""Spec 3e — Process page. Foundations §1.5. Fixtures in tests/conftest.py."""
from pathlib import Path

import pytest

# ---- Task 1: FolderPicker -------------------------------------------------

def test_folder_picker_initial_none(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.folder_picker import FolderPicker
    assert FolderPicker(tk_root, theme=ThemeManager(), on_change=lambda p: None).value() is None


def test_folder_picker_set_value(tk_root, tmp_path):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.folder_picker import FolderPicker
    got = []
    p = FolderPicker(tk_root, theme=ThemeManager(), on_change=got.append)
    p.set_value(str(tmp_path))
    assert p.value() == str(tmp_path) and got == [str(tmp_path)]


# ---- Task 2: ProcessProgressSection ---------------------------------------

def test_progress_section_initial(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.process_progress_section import ProcessProgressSection
    s = ProcessProgressSection(tk_root, theme=ThemeManager())
    assert s._counters == {"passed": 0, "warnings": 0, "failed": 0, "skipped": 0, "errors": 0}


def test_progress_section_increment_and_progress(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.process_progress_section import ProcessProgressSection
    s = ProcessProgressSection(tk_root, theme=ThemeManager())
    s.increment("passed"); s.increment("passed"); s.increment("failed", reason="bad")
    assert s._counters["passed"] == 2 and s._counters["failed"] == 1
    s.set_progress(15, 100, "x.xls")          # uses current/total, NOT current_file_index (C2)


def test_progress_section_set_final_from_summary(tk_root):
    from laser_trim_analyzer.core.models import BatchSummary
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.process_progress_section import ProcessProgressSection
    s = ProcessProgressSection(tk_root, theme=ThemeManager())
    s.set_final(BatchSummary(total_files=10, processed=8, passed=5, warnings=1, failed=2,
                             skipped=2, errors=0))
    assert s._counters == {"passed": 5, "warnings": 1, "failed": 2, "skipped": 2, "errors": 0}


# ---- Task 3: ProcessPage --------------------------------------------------

def test_bucket_mapping_covers_all_statuses_without_skipped():
    """C1: there is no AnalysisStatus.SKIPPED; UNTRIMMED counts as processed, not failed.

    Lives in core/ingest_run.py since 2026-08-31 — the page shares one
    pipeline with Home rather than owning a private copy of this mapping."""
    from laser_trim_analyzer.core.ingest_run import bucket_for_status
    from laser_trim_analyzer.core.models import AnalysisStatus
    assert bucket_for_status(AnalysisStatus.PASS) == "passed"
    assert bucket_for_status(AnalysisStatus.WARNING) == "warnings"
    assert bucket_for_status(AnalysisStatus.FAIL) == "failed"
    assert bucket_for_status(AnalysisStatus.ERROR) == "errors"
    assert bucket_for_status(AnalysisStatus.UNTRIMMED) == "passed"


def test_process_page_initial_state(make_app):
    app = make_app()
    page = app.page_container.get_page("process")
    assert page._folder_picker.value() is None
    assert str(page._start_button.cget("state")) == "disabled"


# ---- the Graphite redesign (2026-10-02): no blue button of its own ---------------------------

def _buttons(widget):
    import customtkinter as ctk
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkButton):
            out.append(c)
        out.extend(_buttons(c))
    return out


def _labels(widget):
    import customtkinter as ctk
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
            out.append(c.cget("text"))
        out.extend(_labels(c))
    return out


def test_the_page_draws_no_blue_button_the_top_bar_holds_the_one(make_app, tmp_path):
    """The spec: "accent #3b82f6 (the single primary button)" -- the top bar's "Process new
    files", on screen above every page. Both runs here start from plain buttons."""
    app = make_app()
    app.config.ingest.add(str(tmp_path))
    page = app.page_container.get_page("process")
    page.on_show()
    page._folder_picker.set_value(str(tmp_path))
    t = page.theme
    assert page._start_button.cget("text") == "Start processing"
    # The remembered run has no button of its own since the finish pass (2026-10-04): two
    # "Process new files" on one screen; the top bar's is the one.
    assert "Process new files" not in [b.cget("text") for b in _buttons(page)]
    blue = [b.cget("text") for b in _buttons(page) if b.cget("fg_color") == t.ACCENT]
    assert blue == []
    assert [b.cget("text") for b in _buttons(app.topbar) if b.cget("fg_color") == t.ACCENT] == [
        "Process new files"]


def test_the_page_holds_both_runs_new_files_first(make_app):
    app = make_app()
    page = app.page_container.get_page("process")
    order = page._body.pack_slaves()
    assert order.index(page._new_files) < order.index(page._folder_picker)
    text = " ".join(_labels(page))
    assert "New files from your folders" in text and "A specific folder" in text
    assert not hasattr(page, "_see_changed")          # a finished run lands on Overview instead


def test_each_progress_shows_only_once_its_run_has_started(make_app, monkeypatch, tmp_path):
    """An idle "Ready" bar and five zero counters, twice over, say nothing (James: "there is so
    much going on"). Each run's progress appears when it starts, and stays to show the tally."""
    _no_threads(monkeypatch)
    app = make_app()
    page = app.page_container.get_page("process")
    assert page._progress.winfo_manager() == "" and page._new_files._progress.winfo_manager() == ""
    page._folder_picker.set_value(str(tmp_path))
    page._start()
    assert page._progress.winfo_manager() == "pack"
    page._on_done()
    assert page._progress.winfo_manager() == "pack"
    app.config.ingest.add(str(tmp_path))
    page._new_files.refresh_folders()
    page._new_files._start()
    assert page._new_files._progress.winfo_manager() == "pack"


@pytest.mark.parametrize("scale", (1.0, 1.5))
def test_db_info_wraps_to_its_container(make_app, scale):
    """global-constraints.md: no fixed pixel wraplength on page-width text -- was a fixed
    wraplength=1200 (same class of bug the Home/Triage wrap tests guard).

    At 150% too (final review, 2026-09-24): this used to pin `wraplength == container width`,
    true only at 100% -- wraplength is CustomTkinter's unscaled units, the width real pixels."""
    import customtkinter as ctk
    ctk.set_widget_scaling(scale)
    try:
        app = make_app()
        page = app.page_container.get_page("process")
        page._db_info.configure(text="Database: /some/invented/path/analysis.db — 0 trim units on "
                                     "record · " * 6)
        try:
            app.attributes("-alpha", 0.0)
        except Exception:
            pass
        app.geometry("1280x720+20000+20000")
        app.deiconify()
        app.update_idletasks()
        app.update()
        try:
            width = page._db_info.master.winfo_width()
            assert page._db_info.cget("wraplength") == int(width / scale)
            assert page._db_info._label.winfo_reqwidth() <= width
            assert page._db_info.cget("wraplength") != 1200
        finally:
            app.withdraw()
    finally:
        ctk.set_widget_scaling(1.0)


def test_apply_progress_counts_skipped_from_processing_status(make_app):
    """C2: progress driven by ProcessingStatus (filename + a local done counter), not an index;
    skipped comes from status.status=='skipped', not a result status."""
    from laser_trim_analyzer.core.models import ProcessingStatus
    app = make_app()
    page = app.page_container.get_page("process")
    page._done = 0
    page._apply_progress(ProcessingStatus(filename="a.xls", status="skipped", progress_percent=10.0), total=100)
    assert page._progress._counters["skipped"] == 1
    assert page._done == 1


# ---- 2026-08-29: parallel folder walk --------------------------------------
# The walk moved to core/ingest_run.discover_excel_files (2026-08-31, one
# shared pipeline); its tests live in tests/test_ingest_run.py.


# ---- "Process new files": the remembered folder list (moved from Home, 2026-10-02) -------------
# Home's "Bring in what's new" card -- run, stop, progress, the one-line summary, and the way to
# the folder list in Settings -- lives at the top of this page now, above the one-off picker. The
# top bar's blue button is the same run: V6App.process_new_files shows this page and starts it.

from laser_trim_analyzer.core.ingest_run import FolderResult, IngestReport  # noqa: E402


class _NoThread:
    """Stand-in for threading.Thread: records, never starts."""
    started = []

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target, self.args = target, args or ()

    def start(self):
        _NoThread.started.append((self.target, self.args))

    def is_alive(self):
        return True

    def join(self, timeout=None):
        pass


def _no_threads(monkeypatch):
    """Freeze thread spawning INSIDE process_page only -- replacing the module reference in its
    namespace, never threading.Thread itself (that would freeze every other thread in the
    process: the Settings page's folder probe landed in the recorder that way once)."""
    import types
    import laser_trim_analyzer.gui.v6.pages.process_page as proc_mod
    _NoThread.started = []
    monkeypatch.setattr(proc_mod, "threading", types.SimpleNamespace(Thread=_NoThread))
    return _NoThread


def _process(app):
    return app.page_container.get_page("process")


def _runs_started(run):
    """The remembered-folder runs the fake recorded -- not the page's own database check, which
    starts a thread of its own every time the page is shown."""
    return [args for target, args in _NoThread.started if target == run._run]


def _new_files(app):
    return _process(app)._new_files


def _with_folders(app, *folders):
    for f in folders:
        app.config.ingest.add(f)
    run = _new_files(app)
    run.refresh_folders()
    return run


def test_with_no_folders_the_run_says_where_to_add_them(make_app):
    app = make_app()                       # fresh Config: no folders
    run = _new_files(app)
    assert run._folder_rows == []
    said = run._folders_label.cget("text")
    assert "Settings" in said and "folder" in said.lower()
    run._settings_link.invoke()
    assert app.page_container.current_page == "settings"


def test_with_folders_the_run_names_them_in_order(make_app, tmp_path):
    """One row per folder since the finish pass (2026-10-04) -- it was one run-on line."""
    app = make_app()
    run = _with_folders(app, str(tmp_path / "a"), str(tmp_path / "b"))
    said = run._folders_label.cget("text")
    assert said.startswith("2 folders, in this order:")
    assert [r.path_label.cget("text") for r in run._folder_rows] == [
        str(tmp_path / "a"), str(tmp_path / "b")]


def test_the_run_drives_the_shared_multi_folder_runner(make_app, monkeypatch):
    """No second pipeline: core/ingest_run.run_folders, in the configured order, incremental."""
    from laser_trim_analyzer.core import ingest_run
    seen = {}

    def fake(folders, **kw):
        seen.update(folders=list(folders), incremental=kw.get("incremental"), db=kw.get("db"))
        return IngestReport(results=[], seconds=1.0)

    monkeypatch.setattr(ingest_run, "run_folders", fake)
    app = make_app()
    run = _with_folders(app, "/laser/a", "/laser/b", "/final_test")
    run._run(["/laser/a", "/laser/b", "/final_test"], True)
    assert seen == {"folders": ["/laser/a", "/laser/b", "/final_test"], "incremental": True,
                    "db": app.db}


def test_stop_shows_while_the_run_is_in_flight_and_only_then(make_app, monkeypatch, tmp_path):
    """The run's own button went with the finish pass (2026-10-04); what the run still needs on
    the page is Stop, there for as long as the run is -- and a second press starts nothing
    (test_pressing_the_button_again_during_the_run_starts_no_second_one)."""
    _no_threads(monkeypatch)
    app = make_app()
    run = _with_folders(app, str(tmp_path))
    assert run._stop_button.winfo_manager() == ""
    run._start()
    assert run._running and run._stop_button.winfo_manager() == "pack"
    assert _NoThread.started, "a worker should have been spawned"
    run._on_run_done(IngestReport(results=[], seconds=1.0))
    assert not run._running and run._stop_button.winfo_manager() == ""


def test_start_is_a_no_op_with_no_folders(make_app, monkeypatch):
    app = make_app()
    _no_threads(monkeypatch)
    _new_files(app)._start()
    assert _NoThread.started == []


def test_the_run_shows_the_combined_summary(make_app):
    from laser_trim_analyzer.core.ingest_run import format_ingest_summary
    app = make_app()
    run = _new_files(app)
    report = IngestReport(
        results=[FolderResult(folder=f"/f{i}", ok=True, files_found=100, new_files=n, seconds=1.0)
                 for i, n in enumerate((100, 100, 14))], seconds=160.0)
    run._on_run_done(report)
    assert run._summary.cget("text") == format_ingest_summary(report)
    assert "3 folders · 214 new files · 2 min 40 s" in run._summary.cget("text")


def test_the_summary_names_a_folder_that_failed(make_app):
    app = make_app()
    run = _new_files(app)
    run._on_run_done(IngestReport(results=[
        FolderResult(folder="/laser/a", ok=True, files_found=10, new_files=4),
        FolderResult(folder="\\\\192.0.2.9\\Laser", ok=False, error="not found — offline share?")],
        seconds=61.0))
    text = run._summary.cget("text")
    assert "\\\\192.0.2.9\\Laser" in text and "1 of 2 folders failed" in text


# ---- the top bar's button --------------------------------------------------------------------

def test_the_top_bar_button_shows_the_page_and_starts_the_remembered_run(make_app, monkeypatch,
                                                                       tmp_path):
    _no_threads(monkeypatch)
    app = make_app()
    _with_folders(app, str(tmp_path / "laser"), str(tmp_path / "final test"))
    app.topbar._process_button.invoke()
    assert app.page_container.current_page == "process"
    assert app.topbar._active_name is None                  # Process has no item on the bar
    run = _new_files(app)
    assert run._running is True
    (args,) = _runs_started(run)
    assert args[0] == [str(tmp_path / "laser"), str(tmp_path / "final test")]


def test_with_no_folders_the_button_shows_the_page_saying_where_to_add_them(make_app, monkeypatch):
    _no_threads(monkeypatch)
    app = make_app()
    app.process_new_files()
    assert app.page_container.current_page == "process"
    assert _runs_started(_new_files(app)) == []
    assert "Settings" in _new_files(app)._folders_label.cget("text")


def test_pressing_the_button_again_during_the_run_starts_no_second_one(make_app, monkeypatch,
                                                                      tmp_path):
    _no_threads(monkeypatch)
    app = make_app()
    _with_folders(app, str(tmp_path))
    app.process_new_files()
    app.show_page("model")
    app.process_new_files()                   # back to the page, where the run is
    assert app.page_container.current_page == "process"
    assert len(_runs_started(_new_files(app))) == 1


# ---- a finished run lands on Overview, whose top shows its summary line ------------------------

def _report(new=214, failed=False, cancelled=False):
    results = [FolderResult(folder="/laser/a", ok=True, files_found=300, new_files=new, seconds=1.0)]
    if failed:
        results.append(FolderResult(folder="/laser/b", ok=False, error="not found — offline share?"))
    return IngestReport(results=results, seconds=160.0, cancelled=cancelled)


def test_a_finished_run_lands_on_overview_with_its_summary_on_top(make_app):
    from laser_trim_analyzer.core.ingest_run import format_ingest_summary
    app = make_app()
    app.show_page("process")
    report = _report()
    _new_files(app)._on_run_done(report)
    assert app.page_container.current_page == "home"
    overview = app.page_container.get_page("home")
    line = overview._run_line
    assert line.winfo_manager() == "pack" and line.cget("text") == format_ingest_summary(report)
    t = overview.theme
    assert (line.cget("fg_color"), line.cget("text_color")) == (t.CARD, t.TEXT_SECONDARY)  # quietly


def test_a_run_with_a_folder_that_failed_lands_with_its_line_in_the_check_colour(make_app):
    app = make_app()
    app.show_page("process")
    _new_files(app)._on_run_done(_report(failed=True))
    overview = app.page_container.get_page("home")
    t = overview.theme
    assert app.page_container.current_page == "home"
    assert "1 of 2 folders failed" in overview._run_line.cget("text")
    assert (overview._run_line.cget("fg_color"), overview._run_line.cget("text_color")) == (
        t.CHECK_TINT, t.CHECK)


def test_a_stopped_run_stays_on_the_process_page(make_app):
    app = make_app()
    app.show_page("process")
    run = _new_files(app)
    run._on_run_done(_report(new=52, cancelled=True))
    assert app.page_container.current_page == "process"
    assert run._summary.cget("text").startswith("Stopped after 52")
    assert "press Process new files again" in run._summary.cget("text")


def test_a_run_that_finishes_while_you_read_a_model_does_not_pull_you_away(make_app):
    app = make_app()
    app.show_page("settings")
    _new_files(app)._on_run_done(_report())
    assert app.page_container.current_page == "settings"
    overview = app.page_container.get_page("home")
    assert overview._run_line.winfo_manager() == "pack"           # waiting for you there


def test_a_finished_one_off_folder_lands_on_overview_too(make_app):
    app = make_app()
    page = _process(app)
    app.show_page("process")
    page._on_done(FolderResult(folder="/one/off", ok=True, files_found=60, new_files=52, seconds=40.0))
    assert app.page_container.current_page == "home"
    assert app.page_container.get_page("home")._run_line.cget("text") == "1 folder · 52 new files · 40 s"


def test_a_stopped_one_off_folder_stays(make_app):
    app = make_app()
    page = _process(app)
    app.show_page("process")
    page._on_done(FolderResult(folder="/one/off", ok=True, files_found=60, new_files=20,
                               cancelled=True))
    assert app.page_container.current_page == "process"


# ---- the full-width lines fit (moved from Home; final review, 2026-09-24) ----------------------
# Since the finish pass (2026-10-04) the folders are ONE ROW EACH: a name over its whole path. These
# fill those real rows with long network paths -- they used to fill the count line above them with a
# run-on string of paths that no screen shows any more (review of option B, #5).

def _map_offscreen(app, size="1280x720"):
    try:
        app.attributes("-alpha", 0.0)
    except Exception:
        pass
    app.geometry(f"{size}+20000+20000")
    app.deiconify()
    app.update_idletasks()
    app.update()


def _long_shares(n=8):
    """Invented network folders, each path too long for one line of a 1280-wide page: about 300
    characters, some 1,650 px in the caption face, where a row is under 1,250."""
    return [rf"\\server-invented\share\Laser trim archive {i}\Production line {i}\Trim data "
            rf"exports\Weekly folders kept for the invented audit\Second shift\Calibrated "
            rf"stations only\Exported by the invented line computer\Kept until the invented "
            rf"review is done\Invented station B\Folder {i} of a long invented chain"
            for i in range(1, n + 1)]


def _row_text_paths(run):
    """The real Tk label behind every piece of text in the folder rows -- each row's place, name
    and path -- by the name find_clipped_text_widgets reports a widget under."""
    import tkinter
    found = []

    def walk(widget):
        for child in tkinter.Misc.winfo_children(widget):
            if isinstance(child, tkinter.Label):
                found.append(str(child))
            walk(child)
    walk(run._folder_list)
    return found


def test_the_folder_rows_and_the_summary_fit_at_1280_by_720(make_app):
    """Measured with the audit's own detector, on a real mapped window (invisible): eight
    remembered folders on invented network paths, each long enough that it fits only by wrapping."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from render_pages import find_clipped_text_widgets
    app = make_app()
    app.show_page("process")
    shares = _long_shares()
    run = _with_folders(app, *shares)
    run._summary.configure(text="Summary line " + "  ·  ".join(shares))
    _map_offscreen(app)
    try:
        t = app.theme
        rows = run._folder_rows
        assert [r.path_label.cget("text") for r in rows] == shares
        assert all(t.font(t.SIZE_CAPTION).measure(r.path_label.cget("text"))
                   > r.path_label.master.winfo_width() > 1 for r in rows)      # it must wrap
        ours = set(_row_text_paths(run)) | {str(run._summary._label), str(run._folders_label._label)}
        assert all(str(r.path_label._label) in ours and str(r.name_label._label) in ours
                   for r in rows)
        cut = [c for c in find_clipped_text_widgets(_process(app), page="process",
                                                   window_size="1280x720")
               if c.path in ours]
        assert not cut, [c.line() for c in cut]
    finally:
        app.withdraw()


@pytest.mark.parametrize("scale", (1.0, 1.5))
def test_the_folder_paths_and_the_summary_wrap_to_their_container(make_app, scale):
    """global-constraints.md: no fixed pixel wraplength on page-width text. At 150% too:
    wraplength is CustomTkinter's unscaled units, the container's width real pixels. A path wraps
    to the frame built with its row (blocks.wrap_to_width's rule), the summary to the section."""
    import customtkinter as ctk
    ctk.set_widget_scaling(scale)
    try:
        app = make_app()
        app.show_page("process")
        shares = _long_shares()
        run = _with_folders(app, *shares)
        run._summary.configure(text="Summary line " + "  ·  ".join(shares))
        _map_offscreen(app)
        try:
            rows = run._folder_rows
            assert [r.path_label.cget("text") for r in rows] == shares
            for label in [r.path_label for r in rows] + [run._summary]:
                width = label.master.winfo_width()
                assert label.cget("wraplength") == int(width / scale), label.cget("text")[:40]
                assert label._label.winfo_reqwidth() <= width
        finally:
            app.withdraw()
    finally:
        ctk.set_widget_scaling(1.0)


# ---- one long job at a time, for the one-off folder too (final review, 2026-10-02) -------------

def test_a_specific_folder_never_starts_beside_another_run(make_app, monkeypatch, tmp_path):
    """The remembered run refuses to start beside another job (2026-09-14: two jobs over the plant
    share made each other slower than either alone); the one-off folder started anyway."""
    _no_threads(monkeypatch)
    app = make_app()
    page = _process(app)
    monkeypatch.setattr(app, "active_run_name", lambda: "A re-grade")
    page._folder_picker.set_value(str(tmp_path))
    page._start()
    assert [a for target, a in _NoThread.started if target == page._run] == []
    assert page._busy_note.winfo_manager() == "pack"
    assert page._busy_note.cget("text").startswith("A re-grade is running — stop it first")
    assert page._cancel is None                          # nothing was set going
    # The other job ends: the next look at the page no longer says it runs (re-review).
    monkeypatch.setattr(app, "active_run_name", lambda: None)
    page.on_show()
    assert page._busy_note.winfo_manager() == ""
    # ...and once nothing else runs, it starts, and the note stays gone.
    page._busy_note.configure(text="stale")
    page._busy_note.pack()
    page._start()
    assert len([a for target, a in _NoThread.started if target == page._run]) == 1
    assert page._busy_note.winfo_manager() == ""
