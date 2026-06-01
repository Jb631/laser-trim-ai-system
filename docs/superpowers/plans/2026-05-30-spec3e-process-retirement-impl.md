# Spec 3e — Process Page Rewrite + Gated V5 Graduation (rewritten 2026-06-01)

> **READ FIRST:** `docs/superpowers/plans/2026-06-01-spec3-rewrite-foundations.md`.
> This plan splits into two independent parts:
> **Part A — Process page rewrite** (ships now). **Part B — Graduation** (V5 deletion + promotion; runs
> ONLY after the foundations §7 gate passes — decision **D1**). The original plan bundled deletion of 13
> files + 20 moves + an import sweep into one atomic commit before parity was proven; that is reversed
> here. Shared fixtures in `tests/conftest.py`.

**Goal (Part A):** Replace the Process placeholder with a V6 Process page that actually ingests files
correctly — **persists trim results**, reports progress/skips/failures faithfully, and lands the user on
Triage afterward. **Goal (Part B):** when V6 has proven itself, promote `gui/v6/* → gui/*`, delete the V5
GUI, and drop `--v6` — in small, reversible commits.

**Target branch:** `V6`. Start at the Spec 3d final commit.

**Critical fixes applied (foundations §6):**
- **C1** — `AnalysisStatus` has no `SKIPPED`. Skips come from `progress_callback` (`status=="skipped"`)
  and `BatchSummary.skipped`, never from a result status.
- **C2** — `ProcessingStatus` has no `current_file_index`. Progress uses a local done-counter +
  `progress_percent` + `filename`.
- **Trim results were never saved** (a silent functional regression) → `db.save_analysis(result)` for
  `file_type == "trim"`; Final-Test/Smoothness are saved inside the processor (don't double-save).
- **I7** — `Processor` has no `db` param → `Processor(config=app.config)`.
- **I10/D1** — atomic mega-commit → gated, decomposed Graduation.

---

## Part A — Process page rewrite

### File Structure (Part A)
**Created:** `gui/v6/widgets/{folder_picker,process_progress_section}.py`,
`gui/v6/pages/process_page.py`, `tests/test_spec3e_process.py`.
**Modified:** `gui/v6/app.py` (register real ProcessPage).

---

### Task 1: FolderPicker

- [ ] **Step 1:** Create `tests/test_spec3e_process.py`:

```python
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
```

- [ ] **Step 2:** Create `widgets/folder_picker.py` (same as original draft; fonts via `theme.font`):

```python
"""Spec 3e — FolderPicker: path display + Browse button. Emits on_change(path)."""
from tkinter import filedialog
from typing import Callable, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class FolderPicker(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_change: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme; self._on_change = on_change; self._value: Optional[str] = None
        self._label = ctk.CTkLabel(self, text="No folder selected", font=theme.font(theme.SIZE_BODY),
                                   text_color=theme.TEXT_SECONDARY, anchor="w")
        self._label.pack(side="left", fill="x", expand=True, padx=(0, theme.SPACE_SM))
        ctk.CTkButton(self, text="Browse…", width=100, fg_color=theme.CARD, hover_color=theme.ELEVATED,
                      text_color=theme.TEXT_PRIMARY, command=self._browse,
                      corner_radius=theme.RADIUS_SM).pack(side="right")

    def value(self) -> Optional[str]:
        return self._value

    def set_value(self, path: str) -> None:
        self._value = path
        self._label.configure(text=path, text_color=self.theme.TEXT_PRIMARY)
        self._on_change(path)

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="Select folder to process")
        if path:
            self.set_value(path)
```

- [ ] **Step 3:** Run `-k folder_picker` → PASS. Commit `feat(spec3e): FolderPicker`.

---

### Task 2: ProcessProgressSection (5 QA-meaningful buckets)

Counters: `passed / warnings / failed / skipped / errors` (more honest than passed/skipped/failed — a
WARNING is "passed linearity, failed sigma or vice-versa" and must be visible for QA).

- [ ] **Step 1:** Append:

```python
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
```

- [ ] **Step 2:** Create `widgets/process_progress_section.py`:

```python
"""Spec 3e — ProcessProgressSection: bar + 5 counters + failure list."""
from typing import Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_BUCKETS = [("passed", "Passed"), ("warnings", "Warnings"), ("failed", "Failed"),
            ("skipped", "Skipped"), ("errors", "Errors")]


class ProcessProgressSection(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._counters: Dict[str, int] = {k: 0 for k, _ in _BUCKETS}
        self._failures: List[str] = []
        self._status = ctk.CTkLabel(self, text="Ready", font=theme.font(theme.SIZE_BODY),
                                    text_color=theme.TEXT_PRIMARY, anchor="w")
        self._status.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        self._bar = ctk.CTkProgressBar(self, progress_color=theme.ACCENT, fg_color=theme.CARD)
        self._bar.pack(side="top", fill="x", pady=(0, theme.SPACE_SM)); self._bar.set(0)
        row = ctk.CTkFrame(self, fg_color="transparent"); row.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        self._labels: Dict[str, ctk.CTkLabel] = {}
        colors = {"passed": theme.TEXT_PRIMARY, "warnings": theme.TIER_WARNING,
                  "failed": theme.TIER_OOC, "skipped": theme.TEXT_SECONDARY, "errors": theme.TIER_OOC}
        for key, label in _BUCKETS:
            lbl = ctk.CTkLabel(row, text=f"{label}: 0", font=theme.font(theme.SIZE_BODY, "bold"),
                               text_color=colors[key])
            lbl.pack(side="left", padx=(0, theme.SPACE_LG)); self._labels[key] = lbl
        self._fail_label = ctk.CTkLabel(self, text="Recent failures:", font=theme.font(theme.SIZE_CAPTION, "bold"),
                                        text_color=theme.TEXT_SECONDARY, anchor="w")
        self._fail_box = ctk.CTkTextbox(self, height=90, fg_color=theme.CARD,
                                        text_color=theme.TEXT_SECONDARY, font=theme.font(theme.SIZE_CAPTION))

    def set_progress(self, current: int, total: int, current_filename: str = "") -> None:
        self._bar.set(current / max(total, 1))
        txt = f"Processing {current} / {total}"
        if current_filename:
            txt += f": {current_filename}"
        self._status.configure(text=txt)

    def increment(self, key: str, reason: str = "") -> None:
        if key not in self._counters:
            return
        self._counters[key] += 1
        self._labels[key].configure(text=f"{dict(_BUCKETS)[key]}: {self._counters[key]}")
        if reason and key in ("failed", "errors"):
            self._failures.append(reason); self._render_failures()

    def set_final(self, summary) -> None:
        """Authoritative counts from BatchSummary (reconciles the live tally)."""
        self._counters = {"passed": summary.passed, "warnings": summary.warnings,
                          "failed": summary.failed, "skipped": summary.skipped, "errors": summary.errors}
        for key, label in _BUCKETS:
            self._labels[key].configure(text=f"{label}: {self._counters[key]}")
        self._bar.set(1.0)
        self._status.configure(text=(f"Complete: {summary.passed} passed, {summary.warnings} warnings, "
                                     f"{summary.failed} failed, {summary.skipped} skipped, "
                                     f"{summary.errors} errors."))

    def set_idle(self, message: str) -> None:
        self._status.configure(text=message); self._bar.set(0)

    def reset(self) -> None:
        self._counters = {k: 0 for k, _ in _BUCKETS}; self._failures = []
        for key, label in _BUCKETS:
            self._labels[key].configure(text=f"{label}: 0")
        self._bar.set(0); self._status.configure(text="Ready")
        if self._fail_label.winfo_ismapped():
            self._fail_label.pack_forget(); self._fail_box.pack_forget()

    def _render_failures(self) -> None:
        if not self._fail_label.winfo_ismapped():
            self._fail_label.pack(side="top", fill="x", pady=(self.theme.SPACE_SM, 0))
            self._fail_box.pack(side="top", fill="x")
        self._fail_box.configure(state="normal"); self._fail_box.delete("1.0", "end")
        for line in self._failures[-10:]:
            self._fail_box.insert("end", line + "\n")
        self._fail_box.configure(state="disabled")
```

- [ ] **Step 3:** Run `-k progress_section` → PASS. Commit `feat(spec3e): ProcessProgressSection (5
  buckets, authoritative final tally)`.

---

### Task 3: ProcessPage (correct ingest — saves trim, no SKIPPED, no current_file_index)

- [ ] **Step 1:** Append tests (pure-helper coverage proves C1/C2 fixed without running a real batch):

```python
# ---- Task 3: ProcessPage --------------------------------------------------

def test_bucket_mapping_covers_all_statuses_without_skipped():
    """C1: there is no AnalysisStatus.SKIPPED; UNTRIMMED counts as processed, not failed."""
    from laser_trim_analyzer.core.models import AnalysisStatus
    from laser_trim_analyzer.gui.v6.pages.process_page import ProcessPage
    assert ProcessPage._bucket_for_status(AnalysisStatus.PASS) == "passed"
    assert ProcessPage._bucket_for_status(AnalysisStatus.WARNING) == "warnings"
    assert ProcessPage._bucket_for_status(AnalysisStatus.FAIL) == "failed"
    assert ProcessPage._bucket_for_status(AnalysisStatus.ERROR) == "errors"
    assert ProcessPage._bucket_for_status(AnalysisStatus.UNTRIMMED) == "passed"


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
```

- [ ] **Step 2:** Create `pages/process_page.py`:

```python
"""Spec 3e — ProcessPage: folder pick → batch process → land on Triage.
Persists trim results (GUI owns trim save); progress from ProcessingStatus + a local
done-counter; final tally from BatchSummary."""
import os
import threading
from pathlib import Path
from typing import List, Optional

import customtkinter as ctk

from laser_trim_analyzer.core.models import AnalysisStatus, ProcessingStatus
from laser_trim_analyzer.core.processor import Processor
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.folder_picker import FolderPicker
from laser_trim_analyzer.gui.v6.widgets.process_progress_section import ProcessProgressSection


class ProcessPage(PageBase):
    page_title = "Process"

    def __init__(self, master, *, theme, app, page_title="Process"):
        self._done = 0
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    # ---- pure helper (C1: no SKIPPED status exists) ----
    @staticmethod
    def _bucket_for_status(status: AnalysisStatus) -> str:
        return {AnalysisStatus.PASS: "passed", AnalysisStatus.WARNING: "warnings",
                AnalysisStatus.FAIL: "failed", AnalysisStatus.ERROR: "errors",
                AnalysisStatus.UNTRIMMED: "passed"}.get(status, "passed")

    def build_content(self, parent):
        t = self.theme
        ctk.CTkLabel(parent, text="Folder to process:", font=t.font(t.SIZE_BODY, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(side="top", fill="x", pady=(0, t.SPACE_XS))
        self._folder_picker = FolderPicker(parent, theme=t, on_change=lambda _p: self._update_start())
        self._folder_picker.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        self._incremental = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(parent, text="Incremental mode (skip already-processed files)",
                        variable=self._incremental, font=t.font(t.SIZE_BODY), text_color=t.TEXT_PRIMARY,
                        fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER)\
            .pack(side="top", anchor="w", pady=(0, t.SPACE_MD))
        self._start_button = ctk.CTkButton(parent, text="Start processing", state="disabled",
                                           fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER,
                                           text_color=t.TEXT_INVERSE, command=self._start,
                                           corner_radius=t.RADIUS_SM)
        self._start_button.pack(side="top", anchor="w", pady=(0, t.SPACE_MD))
        self._progress = ProcessProgressSection(parent, theme=t)
        self._progress.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        self._goto_triage = ctk.CTkButton(parent, text="Go to Triage", fg_color=t.ACCENT,
                                          hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE,
                                          command=lambda: self.app.show_page("triage"),
                                          corner_radius=t.RADIUS_SM)  # packed on completion

    def _update_start(self):
        self._start_button.configure(state="normal" if self._folder_picker.value() else "disabled")

    def _start(self):
        folder = self._folder_picker.value()
        if not folder:
            return
        self._start_button.configure(state="disabled")
        self._goto_triage.pack_forget()
        self._progress.reset()
        self._done = 0
        threading.Thread(target=self._run, args=(folder,), daemon=True).start()

    # ---- progress (runs on Tk thread; called via safe_after, or directly in tests) ----
    def _apply_progress(self, status: ProcessingStatus, total: int) -> None:
        if status.status in ("completed", "skipped", "failed"):
            self._done += 1
        self._progress.set_progress(self._done, total, status.filename or "")
        if status.status == "skipped":
            self._progress.increment("skipped")

    def _run(self, folder: str) -> None:
        files = self._discover(folder)
        if not files:
            self.safe_after(lambda: self._progress.set_idle("No .xls/.xlsx files found."))
            self.safe_after(lambda: self._start_button.configure(state="normal"))
            return
        processor = Processor(config=self.app.config)        # I7: no db= param
        total = len(files)

        def progress_callback(status: ProcessingStatus) -> None:
            self.safe_after(lambda s=status: self._apply_progress(s, total))

        gen = processor.process_batch([Path(p) for p in files],
                                      progress_callback=progress_callback,
                                      incremental=self._incremental.get())
        summary = None
        try:
            while True:
                result = next(gen)
                # Persist trim results (GUI owns trim save; FT/smoothness already saved internally).
                if getattr(result, "file_type", "trim") == "trim":
                    try:
                        self.app.db.save_analysis(result)
                    except Exception as exc:
                        self.safe_after(lambda e=exc, r=result: self._progress.increment(
                            "errors", reason=f"{r.metadata.filename}: save failed: {e}"))
                bucket = self._bucket_for_status(result.overall_status)
                reason = (f"{result.metadata.filename}: {result.overall_status.value}"
                          if bucket in ("failed", "errors") else "")
                self.safe_after(lambda b=bucket, rs=reason: self._progress.increment(b, reason=rs))
        except StopIteration as stop:
            summary = stop.value
        # Authoritative final tally from BatchSummary (reconciles live counts incl. skips).
        if summary is not None:
            self.safe_after(lambda sm=summary: self._progress.set_final(sm))
        self.safe_after(self._on_done)

    def _discover(self, folder: str) -> List[str]:
        out = []
        for root, _dirs, names in os.walk(folder):
            for name in names:
                if name.lower().endswith((".xls", ".xlsx")):
                    out.append(os.path.join(root, name))
        return out

    def _on_done(self):
        self._start_button.configure(state="normal")
        self._goto_triage.pack(side="top", anchor="w", pady=(self.theme.SPACE_SM, 0))
```

- [ ] **Step 3:** Register the real ProcessPage in `gui/v6/app.py` `_build_pages()` (the placeholder loop
  should now be empty — verify and remove it):

```python
        from laser_trim_analyzer.gui.v6.pages.process_page import ProcessPage
        self.page_container.add_page(
            "process", ProcessPage(self.page_container, theme=self.theme, app=self, page_title="Process"))
```

- [ ] **Step 4:** Run `pytest tests/test_spec3e_process.py -v` → PASS. Regression sweep → 0 fail.
  Commit `feat(spec3e): ProcessPage (saves trim, ProcessingStatus-driven progress, BatchSummary tally)`.

---

### Task 4: Manual functional verification of Process (real files)

- [ ] Run `python -m laser_trim_analyzer --v6` → Process → pick a folder with known trim files →
  Start. Verify: progress advances; Passed/Warnings/Failed/Skipped/Errors update; **records actually land
  in the DB** (open Triage/Model and confirm the model appears / counts grew); "Go to Triage" works.
  Re-run with incremental ON and confirm everything is Skipped the second time. Commit an empty
  checkpoint: `chore(spec3e): Process verified against real files (records persisted)`.

---

## Part B — Graduation (gated; run ONLY after the foundations §7 gate passes)

> **Do not start Part B until:** all five spec3 test files + full `pytest tests/` are green; the manual
> smoke (every page, Model selector reaches an arbitrary model, per-unit modal, Copy summary + evidence
> Excel, all five Settings sections functional) passes; and **James has used V6 against a real DB copy and
> confirmed it answers the mission.** This is decision D1 — demotion before deletion. Until then `--v6`
> selects V6 and `main`/no-flag stays V5.

Each step below is its own commit so any single step is independently revertable.

### Task 5 (Graduation): promote `gui/v6/* → gui/*` with history, fix imports

- [ ] **Step 1:** Move with `git mv` (preserves history), not `mv`:

```bash
git mv src/laser_trim_analyzer/gui/v6/theme.py          src/laser_trim_analyzer/gui/theme.py
git mv src/laser_trim_analyzer/gui/v6/sidebar.py        src/laser_trim_analyzer/gui/sidebar.py
git mv src/laser_trim_analyzer/gui/v6/page_base.py      src/laser_trim_analyzer/gui/page_base.py
git mv src/laser_trim_analyzer/gui/v6/page_container.py src/laser_trim_analyzer/gui/page_container.py
git mv src/laser_trim_analyzer/gui/v6/pages            src/laser_trim_analyzer/gui/pages_v6_tmp   # avoid clash with V5 pages/
git mv src/laser_trim_analyzer/gui/v6/sections         src/laser_trim_analyzer/gui/sections
for f in src/laser_trim_analyzer/gui/v6/widgets/*.py; do
  base=$(basename "$f"); [ "$base" = "__init__.py" ] && continue
  git mv "$f" "src/laser_trim_analyzer/gui/widgets/$base"
done
git mv src/laser_trim_analyzer/gui/v6/app.py            src/laser_trim_analyzer/gui/app_v6.py        # temporary; V5 app.py still present this step
```

> The V5 `gui/pages/` still exists this step, so V6 pages land in `gui/pages_v6_tmp/` and V6 app as
> `gui/app_v6.py` to avoid clobbering. Task 6 deletes V5 and Task 7 finalizes the names.

- [ ] **Step 2:** Rewrite imports `laser_trim_analyzer.gui.v6.X` → the new locations. Find them:
```bash
grep -rn "gui\.v6" src/ tests/
```
Update every hit. The `pages` → `pages_v6_tmp` and `app` → `app_v6` temporary names are reflected in
imports and in `tests/conftest.py` `make_app` (the single import line). Run `pytest tests/` → green.

- [ ] **Step 3:** Commit `refactor(spec3e/graduation): move gui/v6 → gui (history-preserving), fix imports`.

### Task 6 (Graduation): delete V5 GUI

- [ ] **Step 1:** Confirm V5-only widgets are truly unused by V6 before deleting:
```bash
grep -rn "scrollable_combobox" src/laser_trim_analyzer/gui | grep -v "/pages/"   # expect: no V6 hits
```
- [ ] **Step 2:** Delete V5: `git rm src/laser_trim_analyzer/app.py src/laser_trim_analyzer/gui/app.py
  src/laser_trim_analyzer/gui/pages/*.py` and `git rm src/laser_trim_analyzer/gui/widgets/scrollable_combobox.py`
  (only if Step 1 showed no V6 usage; keep `gui/widgets/chart.py` — V6's per-unit modal uses it).
  Remove the empty V5 `gui/pages/` dir.
- [ ] **Step 3:** `pytest tests/` → green (V6 no longer references anything deleted). Commit
  `refactor(spec3e/graduation): delete V5 GUI (pages, root, V5-only widgets)`.

### Task 7 (Graduation): finalize names + drop `--v6` + docs

- [ ] **Step 1:** Rename the temporaries into place:
```bash
git mv src/laser_trim_analyzer/gui/app_v6.py     src/laser_trim_analyzer/gui/app.py
git mv src/laser_trim_analyzer/gui/pages_v6_tmp  src/laser_trim_analyzer/gui/pages
```
Update imports referencing `app_v6` / `pages_v6_tmp` (grep + edit), including `tests/conftest.py`.
- [ ] **Step 2:** `__main__.py` `main()` → drop the flag; V6 is the only UI:
```python
def main():
    logger.info("Starting Laser Trim Analyzer...")
    try:
        from laser_trim_analyzer.config import get_config
        from laser_trim_analyzer.gui.app import V6App
        config = get_config(); config.database.ensure_directory()
        V6App(config).run()
    except ImportError as e:
        logger.error(f"Import error: {e}"); sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}"); sys.exit(1)
```
- [ ] **Step 3:** Update `gui/__init__.py` to re-export `V6App` from the new `gui.app`. Update `CLAUDE.md`:
  "10 pages …" → "**4 pages**: Triage, Process, Model, Settings (V6 mission-aligned redesign)"; fix any
  other V5-page references.
- [ ] **Step 4:** `pytest tests/` → green. `python -m laser_trim_analyzer` (no flag) launches V6; click
  every page.
- [ ] **Step 5:** Commit `refactor(spec3e/graduation): V6 is the only UI (drop --v6, finalize names, docs)`.
- [ ] **Step 6:** `git push origin V6`. **Do not** merge V6 → main; graduating to the deployed build is a
  separate explicit decision after V6 has been used at work (anchor/redesign).

---

## Out of scope
- New Process features beyond V5 parity. Deleting `gui/widgets/chart.py` (V6 depends on it). Merging V6 →
  main. Any V5 deletion before the §7 gate passes.
</content>
