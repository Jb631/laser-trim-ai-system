# Spec 3e — Process Page Rewrite + V5 Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the Process page using V6 widget primitives (full rewrite, not a port). Then retire all V5 GUI code: delete the 8 old pages + V5 root, promote `gui/v6/*` up to `gui/*`, remove the `--v6` CLI flag, update CLAUDE.md page count.

**Architecture:** ProcessPage composes FolderPicker + IncrementalToggle + ProgressSection. Uses existing `core.processor.Processor` API (same calls V5 made). After processing completes, calls `train_drift_detector` to refresh Triage data (deferred to user click for now — Process page doesn't autotrain). Retirement is a single atomic commit at the end.

**Tech Stack:** Python 3.x, customtkinter, pytest. Depends on Spec 3a (shell) and Specs 3b–3d (proves V6 shell is feature-complete).

**Target branch:** `V6` only. Latest commit before starting: the Spec 3d final commit.

**Spec reference:** `docs/superpowers/specs/2026-05-30-spec3-ui-shell-design.md` (Sub-spec 3e section).

---

## File Structure

**Files created:**
- `src/laser_trim_analyzer/gui/v6/widgets/folder_picker.py`
- `src/laser_trim_analyzer/gui/v6/widgets/process_progress_section.py`
- `src/laser_trim_analyzer/gui/v6/pages/process_page.py`
- `tests/test_spec3e_process_retirement.py`

**Files modified (Task 5 — the retirement commit):**
- Delete: `src/laser_trim_analyzer/gui/app.py`
- Delete: all 10 files in `src/laser_trim_analyzer/gui/pages/`
- Delete: `src/laser_trim_analyzer/gui/widgets/scrollable_combobox.py` (V5-only)
- Move: `src/laser_trim_analyzer/gui/v6/*` → `src/laser_trim_analyzer/gui/*`
- Modify: `src/laser_trim_analyzer/__main__.py` — remove `--v6` flag handling
- Modify: `src/laser_trim_analyzer/gui/__init__.py` — re-export from new locations
- Modify: `CLAUDE.md` — page count + name list

---

## Task 1: FolderPicker widget

Read-only display of selected folder + "Browse" button that opens a native folder dialog.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/folder_picker.py`
- Test: `tests/test_spec3e_process_retirement.py` (CREATE)

- [ ] **Step 1: Create test file**

Create `tests/test_spec3e_process_retirement.py`:

```python
"""Spec 3e — Process page rewrite + V5 retirement."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def tk_root():
    import customtkinter as ctk
    root = ctk.CTk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Task 1: FolderPicker
# ---------------------------------------------------------------------------


def test_folder_picker_initial_value(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.folder_picker import FolderPicker

    picker = FolderPicker(
        tk_root, theme=ThemeManager(),
        on_change=lambda p: None,
    )
    assert picker.value() is None


def test_folder_picker_set_value(tk_root, tmp_path):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.folder_picker import FolderPicker

    received: list[str] = []
    picker = FolderPicker(
        tk_root, theme=ThemeManager(),
        on_change=lambda p: received.append(p),
    )
    picker.set_value(str(tmp_path))
    assert picker.value() == str(tmp_path)
    assert received == [str(tmp_path)]
```

- [ ] **Step 2: Create the widget**

Create `src/laser_trim_analyzer/gui/v6/widgets/folder_picker.py`:

```python
"""Spec 3e — FolderPicker.

Display selected folder path + Browse button.  Emits on_change(path)
when the selection changes.
"""
from tkinter import filedialog
from typing import Callable, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class FolderPicker(ctk.CTkFrame):
    """Folder selection widget."""

    def __init__(
        self,
        master,
        theme: ThemeManager,
        on_change: Callable[[str], None],
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_change = on_change
        self._value: Optional[str] = None

        self._label = ctk.CTkLabel(
            self, text="No folder selected",
            font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
            text_color=theme.TEXT_SECONDARY,
            anchor="w",
        )
        self._label.pack(
            side="left", fill="x", expand=True,
            padx=(0, theme.SPACE_SM),
        )
        browse_btn = ctk.CTkButton(
            self, text="Browse...", width=100,
            fg_color=theme.CARD, hover_color=theme.ELEVATED,
            text_color=theme.TEXT_PRIMARY,
            command=self._on_browse,
            corner_radius=theme.RADIUS_SM,
        )
        browse_btn.pack(side="right")

    def value(self) -> Optional[str]:
        return self._value

    def set_value(self, path: str) -> None:
        self._value = path
        self._label.configure(
            text=path,
            text_color=self.theme.TEXT_PRIMARY,
        )
        self._on_change(path)

    def _on_browse(self) -> None:
        path = filedialog.askdirectory(title="Select folder")
        if path:
            self.set_value(path)
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_spec3e_process_retirement.py -v -k folder_picker
```

Expected: 2 PASS.

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/folder_picker.py tests/test_spec3e_process_retirement.py
git commit -m "feat(spec3e): FolderPicker widget"
```

---

## Task 2: ProcessProgressSection

Holds the progress bar, status label, success/skipped/failure tally counters, and a collapsible failure list.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/process_progress_section.py`
- Test: `tests/test_spec3e_process_retirement.py` (APPEND)

- [ ] **Step 1: Append test**

```python
# ---------------------------------------------------------------------------
# Task 2: ProcessProgressSection
# ---------------------------------------------------------------------------


def test_progress_section_initial_state(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.process_progress_section import (
        ProcessProgressSection,
    )

    section = ProcessProgressSection(tk_root, theme=ThemeManager())
    assert section._counters == {"passed": 0, "skipped": 0, "failed": 0}


def test_progress_section_increment(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.process_progress_section import (
        ProcessProgressSection,
    )

    section = ProcessProgressSection(tk_root, theme=ThemeManager())
    section.increment("passed")
    section.increment("passed")
    section.increment("failed", reason="Bad sigma")
    assert section._counters == {"passed": 2, "skipped": 0, "failed": 1}


def test_progress_section_set_progress(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.process_progress_section import (
        ProcessProgressSection,
    )

    section = ProcessProgressSection(tk_root, theme=ThemeManager())
    section.set_progress(current=15, total=100, current_filename="x.xls")
    # No assertion-on-text; widget update is the win condition (no exception)
```

- [ ] **Step 2: Create widget**

Create `src/laser_trim_analyzer/gui/v6/widgets/process_progress_section.py`:

```python
"""Spec 3e — ProcessProgressSection."""
from typing import Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class ProcessProgressSection(ctk.CTkFrame):
    """Progress bar + counters + failure list."""

    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._counters: Dict[str, int] = {"passed": 0, "skipped": 0, "failed": 0}
        self._failures: List[str] = []

        # Status label
        self._status_label = ctk.CTkLabel(
            self, text="Ready",
            font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
            text_color=theme.TEXT_PRIMARY,
            anchor="w",
        )
        self._status_label.pack(
            side="top", fill="x", pady=(0, theme.SPACE_SM),
        )

        # Progress bar
        self._progress = ctk.CTkProgressBar(
            self,
            progress_color=theme.ACCENT,
            fg_color=theme.CARD,
        )
        self._progress.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        self._progress.set(0)

        # Tally row
        tally_row = ctk.CTkFrame(self, fg_color="transparent")
        tally_row.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        self._counter_labels: Dict[str, ctk.CTkLabel] = {}
        for key, label_text, color in (
            ("passed", "Passed", theme.TEXT_PRIMARY),
            ("skipped", "Skipped", theme.TEXT_SECONDARY),
            ("failed", "Failed", theme.TIER_OOC),
        ):
            label = ctk.CTkLabel(
                tally_row, text=f"{label_text}: 0",
                font=(theme.FONT_FAMILY[0], theme.SIZE_BODY, "bold"),
                text_color=color,
            )
            label.pack(side="left", padx=(0, theme.SPACE_LG))
            self._counter_labels[key] = label

        # Failure list (small textbox)
        self._failure_label = ctk.CTkLabel(
            self, text="Recent failures:",
            font=(theme.FONT_FAMILY[0], theme.SIZE_CAPTION, "bold"),
            text_color=theme.TEXT_SECONDARY,
            anchor="w",
        )
        self._failure_box = ctk.CTkTextbox(
            self, height=80, fg_color=theme.CARD,
            text_color=theme.TEXT_SECONDARY,
            font=(theme.FONT_FAMILY[0], theme.SIZE_CAPTION),
        )
        # Don't pack the failure widgets until there's a failure to show

    def set_progress(
        self, current: int, total: int, current_filename: str = "",
    ) -> None:
        ratio = current / max(total, 1)
        self._progress.set(ratio)
        text = f"Processing {current} / {total}"
        if current_filename:
            text += f": {current_filename}"
        self._status_label.configure(text=text)

    def increment(self, key: str, reason: str = "") -> None:
        if key not in self._counters:
            return
        self._counters[key] += 1
        label_text = key.title()
        self._counter_labels[key].configure(
            text=f"{label_text}: {self._counters[key]}",
        )
        if key == "failed" and reason:
            self._failures.append(reason)
            if not self._failure_label.winfo_ismapped():
                self._failure_label.pack(
                    side="top", fill="x", pady=(self.theme.SPACE_SM, 0),
                )
                self._failure_box.pack(side="top", fill="x")
            # Keep last 10
            self._failure_box.configure(state="normal")
            self._failure_box.delete("1.0", "end")
            for line in self._failures[-10:]:
                self._failure_box.insert("end", line + "\n")
            self._failure_box.configure(state="disabled")

    def reset(self) -> None:
        self._counters = {"passed": 0, "skipped": 0, "failed": 0}
        self._failures = []
        for key, label in self._counter_labels.items():
            label.configure(text=f"{key.title()}: 0")
        self._progress.set(0)
        self._status_label.configure(text="Ready")
        if self._failure_label.winfo_ismapped():
            self._failure_label.pack_forget()
            self._failure_box.pack_forget()

    def set_complete(self) -> None:
        passed = self._counters["passed"]
        skipped = self._counters["skipped"]
        failed = self._counters["failed"]
        self._status_label.configure(
            text=(
                f"Processing complete: {passed} passed, "
                f"{skipped} skipped, {failed} failed."
            )
        )
        self._progress.set(1.0)
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_spec3e_process_retirement.py -v -k progress_section
```

Expected: 3 PASS.

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/process_progress_section.py tests/test_spec3e_process_retirement.py
git commit -m "feat(spec3e): ProcessProgressSection"
```

---

## Task 3: ProcessPage composition

Replaces the Process placeholder. Wires up file processing using the existing `core.processor.Processor`.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/pages/process_page.py`
- Modify: `src/laser_trim_analyzer/gui/v6/app.py` — register real ProcessPage
- Test: `tests/test_spec3e_process_retirement.py` (APPEND)

- [ ] **Step 1: Append test**

```python
# ---------------------------------------------------------------------------
# Task 3: ProcessPage composition
# ---------------------------------------------------------------------------


def test_process_page_initial_state(tmp_path):
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "p.db"
    app = V6App(cfg)
    try:
        app.withdraw()
        page = app.page_container.get_page("process")
        # The Start button exists and is initially disabled (no folder)
        assert page._start_button is not None
        assert page._folder_picker.value() is None
    finally:
        app.destroy()
```

- [ ] **Step 2: Create ProcessPage**

Create `src/laser_trim_analyzer/gui/v6/pages/process_page.py`:

```python
"""Spec 3e — ProcessPage.

Rewritten from scratch using V6 primitives.  Uses the existing core
Processor for the actual analysis.  After completion, gives the user
a "Go to Triage" button so they can immediately see the updated state.
"""
import os
import threading
from pathlib import Path
from typing import Optional

import customtkinter as ctk

from laser_trim_analyzer.core.models import AnalysisStatus, ProcessingStatus
from laser_trim_analyzer.core.processor import Processor
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.folder_picker import FolderPicker
from laser_trim_analyzer.gui.v6.widgets.process_progress_section import (
    ProcessProgressSection,
)


class ProcessPage(PageBase):
    page_title = "Process"

    def __init__(self, master, theme, app):
        self._app = app
        self._processing_thread: Optional[threading.Thread] = None
        super().__init__(master, theme=theme)

    def build_content(self, parent):
        # Folder picker
        picker_label = ctk.CTkLabel(
            parent, text="Folder to process:",
            font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_BODY, "bold"),
            text_color=self.theme.TEXT_PRIMARY,
            anchor="w",
        )
        picker_label.pack(side="top", fill="x", pady=(0, self.theme.SPACE_XS))
        self._folder_picker = FolderPicker(
            parent, theme=self.theme,
            on_change=lambda path: self._update_start_state(),
        )
        self._folder_picker.pack(side="top", fill="x", pady=(0, self.theme.SPACE_MD))

        # Incremental toggle
        self._incremental_var = ctk.BooleanVar(value=True)
        incremental = ctk.CTkCheckBox(
            parent,
            text="Incremental mode (skip already-processed files)",
            variable=self._incremental_var,
            font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_BODY),
            text_color=self.theme.TEXT_PRIMARY,
            fg_color=self.theme.ACCENT,
            hover_color=self.theme.ACCENT_HOVER,
        )
        incremental.pack(side="top", anchor="w", pady=(0, self.theme.SPACE_MD))

        # Start button
        self._start_button = ctk.CTkButton(
            parent, text="Start processing",
            fg_color=self.theme.ACCENT, hover_color=self.theme.ACCENT_HOVER,
            text_color=self.theme.TEXT_INVERSE,
            command=self._start_processing,
            corner_radius=self.theme.RADIUS_SM,
            state="disabled",
        )
        self._start_button.pack(side="top", anchor="w", pady=(0, self.theme.SPACE_MD))

        # Progress section
        self._progress_section = ProcessProgressSection(parent, theme=self.theme)
        self._progress_section.pack(side="top", fill="x", pady=(0, self.theme.SPACE_MD))

        # "Go to Triage" button (only shown after completion)
        self._goto_triage_btn = ctk.CTkButton(
            parent, text="Go to Triage",
            fg_color=self.theme.ACCENT, hover_color=self.theme.ACCENT_HOVER,
            text_color=self.theme.TEXT_INVERSE,
            command=lambda: self._app.show_page("triage"),
            corner_radius=self.theme.RADIUS_SM,
        )
        # Not packed until processing completes

    def _update_start_state(self) -> None:
        if self._folder_picker.value():
            self._start_button.configure(state="normal")
        else:
            self._start_button.configure(state="disabled")

    def _start_processing(self) -> None:
        folder = self._folder_picker.value()
        if not folder:
            return
        self._start_button.configure(state="disabled")
        self._goto_triage_btn.pack_forget()
        self._progress_section.reset()
        self._processing_thread = threading.Thread(
            target=self._run_processing, args=(folder,), daemon=True,
        )
        self._processing_thread.start()

    def _run_processing(self, folder: str) -> None:
        files = self._discover_files(folder)
        if not files:
            self.after(0, lambda: self._progress_section.set_progress(
                0, 0, "No files found"
            ))
            self.after(0, lambda: self._start_button.configure(state="normal"))
            return

        processor = Processor(config=self._app.config, db=self._app.db)
        incremental = self._incremental_var.get()
        total = len(files)

        def progress_callback(status: ProcessingStatus) -> None:
            self.after(0, lambda: self._progress_section.set_progress(
                current=status.current_file_index + 1,
                total=total,
                current_filename=status.filename or "",
            ))

        gen = processor.process_batch(
            file_paths=[Path(p) for p in files],
            progress_callback=progress_callback,
            incremental=incremental,
        )
        for result in gen:
            if result.overall_status == AnalysisStatus.ERROR:
                self.after(0, lambda r=result: self._progress_section.increment(
                    "failed",
                    reason=f"{r.metadata.filename}: {r.metadata.error or 'error'}",
                ))
            elif result.overall_status == AnalysisStatus.SKIPPED:
                self.after(0, lambda: self._progress_section.increment("skipped"))
            else:
                self.after(0, lambda: self._progress_section.increment("passed"))

        self.after(0, self._on_processing_done)

    def _discover_files(self, folder: str) -> list:
        """Walk the folder and return all .xls/.xlsx files."""
        out = []
        for root, _, names in os.walk(folder):
            for name in names:
                if name.lower().endswith((".xls", ".xlsx")):
                    out.append(os.path.join(root, name))
        return out

    def _on_processing_done(self) -> None:
        self._progress_section.set_complete()
        self._start_button.configure(state="normal")
        self._goto_triage_btn.pack(side="top", anchor="w", pady=(self.theme.SPACE_SM, 0))
```

- [ ] **Step 3: Register in V6App**

In `gui/v6/app.py`, in `_build_pages()`:

```python
        from laser_trim_analyzer.gui.v6.pages.process_page import ProcessPage
        process_page = ProcessPage(
            self.page_container, theme=self.theme, app=self,
        )
        self.page_container.add_page("process", process_page)
```

Remove `("process", "Process", "3e")` from the placeholder loop (which should be empty at this point — Settings was the last one, removed in Spec 3d). Verify the placeholder loop is now empty; if it is, remove the entire loop block.

- [ ] **Step 4: Run + commit**

```bash
pytest tests/test_spec3e_process_retirement.py -v -k process_page
```

Expected: 1 PASS.

```bash
git add src/laser_trim_analyzer/gui/v6/app.py src/laser_trim_analyzer/gui/v6/pages/process_page.py tests/test_spec3e_process_retirement.py
git commit -m "feat(spec3e): rewritten ProcessPage using V6 primitives

FolderPicker + Incremental toggle + Start button + ProcessProgressSection
+ Go to Triage CTA on completion.  Wraps the existing core.processor.Processor
unchanged.  Background thread runs processing; UI updates via after(0, ...)."
```

---

## Task 4: V6 manual smoke test (record + verify)

Before retiring V5, verify V6 launches and the basic navigation flow works end-to-end. This is a manual checkpoint, not an automated test.

- [ ] **Step 1: Launch V6 and click through every page**

Run: `python -m laser_trim_analyzer --v6`

Verify:
- Sidebar shows Triage / Process / Model / Settings
- Click each one; each renders without crash
- Triage shows the empty state if no flagged models
- Settings → Alert Thresholds card is expanded; sensitivity slider responds; preview updates
- Process → folder picker opens; selecting a folder enables Start

If anything crashes, fix it before proceeding to retirement.

- [ ] **Step 2: Verify V5 still works**

Run: `python -m laser_trim_analyzer`

V5 should still launch identically to before any V6 work landed.

- [ ] **Step 3: Note the verification in a commit**

```bash
git commit --allow-empty -m "chore(spec3e): manual V6 smoke verified

Pre-retirement checkpoint.  V6 shell launches, all 4 pages render
without crashing, V5 also still works."
```

---

## Task 5: V5 retirement — single atomic commit

Promote `gui/v6/` to `gui/`, delete the V5 GUI code, remove the `--v6` flag, update CLAUDE.md.

**Files:**
- Delete: `src/laser_trim_analyzer/gui/app.py`
- Delete: `src/laser_trim_analyzer/gui/pages/dashboard.py`, `analyze.py`, `compare.py`, `trends.py`, `quality_health.py`, `scorecard.py`, `smoothness.py`, `specs.py`, `process.py`, `settings.py`, `export.py`, `__init__.py` (if standalone)
- Delete: `src/laser_trim_analyzer/gui/widgets/scrollable_combobox.py`
- Move: `src/laser_trim_analyzer/gui/v6/*` → `src/laser_trim_analyzer/gui/*`
- Modify: `src/laser_trim_analyzer/__main__.py`
- Modify: `src/laser_trim_analyzer/gui/__init__.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Delete the V5 GUI files**

```bash
rm src/laser_trim_analyzer/gui/app.py
rm src/laser_trim_analyzer/gui/pages/dashboard.py
rm src/laser_trim_analyzer/gui/pages/analyze.py
rm src/laser_trim_analyzer/gui/pages/compare.py
rm src/laser_trim_analyzer/gui/pages/trends.py
rm src/laser_trim_analyzer/gui/pages/quality_health.py
rm src/laser_trim_analyzer/gui/pages/scorecard.py
rm src/laser_trim_analyzer/gui/pages/smoothness.py
rm src/laser_trim_analyzer/gui/pages/specs.py
rm src/laser_trim_analyzer/gui/pages/process.py
rm src/laser_trim_analyzer/gui/pages/settings.py
rm src/laser_trim_analyzer/gui/pages/export.py
rm -rf src/laser_trim_analyzer/gui/pages
rm src/laser_trim_analyzer/gui/widgets/scrollable_combobox.py
```

Verify the directory structure:

```bash
find src/laser_trim_analyzer/gui -type f -name '*.py'
```

Expected: only files inside `gui/v6/` (plus `gui/widgets/chart.py` and `gui/widgets/__init__.py`, which are V5 widgets still used by V6's FocusChart — keep those).

- [ ] **Step 2: Promote `gui/v6/` to `gui/`**

```bash
# Move files up one level, preserving subdirectory structure
mv src/laser_trim_analyzer/gui/v6/__init__.py src/laser_trim_analyzer/gui/__init__.py.new
mv src/laser_trim_analyzer/gui/v6/theme.py src/laser_trim_analyzer/gui/theme.py
mv src/laser_trim_analyzer/gui/v6/sidebar.py src/laser_trim_analyzer/gui/sidebar.py
mv src/laser_trim_analyzer/gui/v6/page_base.py src/laser_trim_analyzer/gui/page_base.py
mv src/laser_trim_analyzer/gui/v6/page_container.py src/laser_trim_analyzer/gui/page_container.py
mv src/laser_trim_analyzer/gui/v6/app.py src/laser_trim_analyzer/gui/app.py

# Move pages subdirectory
mv src/laser_trim_analyzer/gui/v6/pages src/laser_trim_analyzer/gui/pages

# Move sections subdirectory (created in 3d)
mv src/laser_trim_analyzer/gui/v6/sections src/laser_trim_analyzer/gui/sections

# Move v6 widgets into the existing widgets dir
for f in src/laser_trim_analyzer/gui/v6/widgets/*.py; do
    if [ "$(basename $f)" != "__init__.py" ]; then
        mv "$f" "src/laser_trim_analyzer/gui/widgets/"
    fi
done

# Remove the now-empty v6 directories
rm -rf src/laser_trim_analyzer/gui/v6

# Replace the gui __init__ with the moved v6 __init__ (if any content)
mv src/laser_trim_analyzer/gui/__init__.py.new src/laser_trim_analyzer/gui/__init__.py
```

- [ ] **Step 3: Update import paths**

After the file moves, every `from laser_trim_analyzer.gui.v6.X import Y` must become `from laser_trim_analyzer.gui.X import Y`. Run:

```bash
# Show all references to gui.v6
grep -rn "gui\.v6" src/ tests/
```

For each line found, edit the file and replace `gui.v6.` with `gui.`. Test files included.

Then specifically the imports inside the moved files reference each other — they should all be valid because the `v6` segment is just being dropped.

- [ ] **Step 4: Update `__main__.py` to drop the `--v6` flag**

In `src/laser_trim_analyzer/__main__.py`, find the `main()` function. Replace:

```python
def main():
    """Main entry point.

    By default, launches the legacy V5 UI (LaserTrimApp).
    Pass --v6 to launch the V6 UI shell (Spec 3a+).
    """
    use_v6 = "--v6" in sys.argv
    logger.info(
        f"Starting Laser Trim Analyzer v5... (UI: {'V6' if use_v6 else 'V5'})"
    )

    try:
        from laser_trim_analyzer.config import get_config

        config = get_config()
        logger.info(f"Config loaded - Database: {config.database.path}")
        config.database.ensure_directory()

        if use_v6:
            from laser_trim_analyzer.gui.v6.app import V6App
            app = V6App(config)
        else:
            from laser_trim_analyzer.app import LaserTrimApp
            app = LaserTrimApp(config)
        app.run()
    except ...
```

with:

```python
def main():
    """Main entry point."""
    logger.info("Starting Laser Trim Analyzer V6...")

    try:
        from laser_trim_analyzer.config import get_config
        from laser_trim_analyzer.gui.app import V6App

        config = get_config()
        logger.info(f"Config loaded - Database: {config.database.path}")
        config.database.ensure_directory()

        app = V6App(config)
        app.run()
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -e .")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
```

Also delete `src/laser_trim_analyzer/app.py` (the V5 LaserTrimApp). Use:

```bash
rm src/laser_trim_analyzer/app.py
```

- [ ] **Step 5: Update CLAUDE.md**

In `CLAUDE.md`, find the line that lists "10 pages: Dashboard, Process, Analyze, Compare, Trends, Quality Health, Scorecard, Smoothness, Specs, Settings". Replace with:

```
- **4 pages**: Triage, Process, Model, Settings (V6 mission-aligned redesign)
```

Find any other V5-specific page references and update.

- [ ] **Step 6: Run full regression sweep**

```bash
pytest tests/ 2>&1 | tail -10
```

Expected: 0 fail. If anything fails, it's likely an import path missed in Step 3 — find the stale `gui.v6.` reference and fix it.

- [ ] **Step 7: Launch V6 and verify it still works**

```bash
python -m laser_trim_analyzer
```

Verify: V6 launches (no `--v6` flag needed). All 4 pages render. Close the app.

- [ ] **Step 8: The retirement commit (single atomic commit)**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(spec3e): retire V5 GUI, promote V6 to default

Atomic cleanup commit completing the V6 redesign:

1. Delete the 10 V5 pages (dashboard / analyze / compare / trends /
   quality_health / scorecard / smoothness / specs / process / settings).
2. Delete the V5 LaserTrimApp root and gui/app.py V5 contents.
3. Delete V5-only widgets (scrollable_combobox).
4. Promote gui/v6/* up to gui/* (theme, sidebar, page_base,
   page_container, app + pages/ + sections/ + widgets/).
5. Remove the --v6 CLI flag from __main__.py.  V6 is now the only UI.
6. Update CLAUDE.md page count and name list.

This is the end of the V6 build sequence (Specs 3a-3e).  Spec 1
(untrimmed_sigma_gradient) and Spec 2 (multi-metric drift detector)
already shipped to main; V6 inherits both.
EOF
)"
```

- [ ] **Step 9: Final regression sweep + manual smoke**

```bash
pytest tests/ -v 2>&1 | tail -3
python -m laser_trim_analyzer
```

V6 should launch as the default and all tests should pass.

```bash
# Final V6 push
git push origin V6
```

---

## Out-of-scope reminders

- **Do not** add new features beyond what V5 already had on the Process page.
- **Do not** delete `gui/widgets/chart.py` — V6's FocusChart depends on its matplotlib infrastructure.
- **Do not** delete tests that exercise the old V5 pages — those should already be gone (we never wrote any in V5).
- **Do not** merge V6 → main yet. Graduating V6 to production is a separate explicit decision after V6 has been used at work.
