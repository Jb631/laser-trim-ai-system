"""Spec 3e — Process page. Foundations §1.5. Fixtures in tests/conftest.py."""
from pathlib import Path

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


# ---- facelift step 2, Task 7: one button, a way to what changed ------------

def test_start_processing_is_the_one_teal_button(make_app):
    """blocks.primary_button, and the only one on this page (global-constraints.md: at most
    ONE teal-filled button per screen)."""
    app = make_app()
    page = app.page_container.get_page("process")
    t = page.theme
    assert page._start_button.cget("text") == "Start processing"
    assert page._start_button.cget("fg_color") == t.ACCENT
    assert page._see_changed.cget("fg_color") != t.ACCENT


def test_see_what_changed_appears_after_a_run_and_routes_to_findings(make_app):
    """Replaces the old 'Go to Triage' teal button (ruling 6): a blocks.link_button that
    routes to Findings, not Triage -- 'what changed' is what the findings engine says. It used
    to make a SECOND teal button appear next to Start processing the moment a run finished."""
    app = make_app()
    page = app.page_container.get_page("process")
    t = page.theme

    def _packed(w):
        return w.winfo_manager() == "pack"

    assert _packed(page._see_changed) is False        # not shown before any run
    page._on_done()
    assert _packed(page._see_changed) is True
    assert page._see_changed.cget("fg_color") != t.ACCENT   # never a second teal button
    page._see_changed.invoke()
    assert app.page_container.current_page == "findings"


def test_db_info_wraps_to_its_container(make_app):
    """global-constraints.md: no fixed pixel wraplength on page-width text -- was a fixed
    wraplength=1200 (same class of bug the Home/Triage wrap tests guard)."""
    app = make_app()
    page = app.page_container.get_page("process")
    page._db_info.configure(text="Database: /some/path — 0 trim units on record")
    try:
        app.attributes("-alpha", 0.0)
    except Exception:
        pass
    app.geometry("1280x720+20000+20000")
    app.deiconify()
    app.update_idletasks()
    app.update()
    try:
        assert page._db_info.cget("wraplength") == page._db_info.master.winfo_width()
        assert page._db_info.cget("wraplength") != 1200
    finally:
        app.withdraw()


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
