"""Spec 3e — Process page. Foundations §1.5. Fixtures in tests/conftest.py."""

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
