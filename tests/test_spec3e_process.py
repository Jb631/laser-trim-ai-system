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
