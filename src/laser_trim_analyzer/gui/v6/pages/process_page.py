"""Spec 3e — ProcessPage: folder pick → batch process → land on Triage.

The one-off folder path. The batch itself lives in `core/ingest_run.py` and is
shared with Home's "Process everything new" (spec 2026-08-29: "same worker, no
duplicate pipeline") — this page is now just a picker, a progress section and
the marshalling between the worker and Tk.
"""
import logging
import threading

import customtkinter as ctk

logger = logging.getLogger(__name__)

from laser_trim_analyzer.core import ingest_run
from laser_trim_analyzer.core.ingest_run import ProgressCoalescer, ProgressTicker
from laser_trim_analyzer.core.models import ProcessingStatus
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.folder_picker import FolderPicker
from laser_trim_analyzer.gui.v6.widgets.process_progress_section import ProcessProgressSection


class ProcessPage(PageBase):
    page_title = "Process"

    def __init__(self, master, *, theme, app, page_title="Process"):
        self._done = 0
        super().__init__(master, theme=theme, app=app, page_title=page_title)

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
        # Which database + how much it knows — BEFORE processing starts. The
        # work incident (2026-07-09) would have been obvious in one glance if
        # this had said "0 units": wrong/empty database, don't hit Start.
        self._db_info = ctk.CTkLabel(parent, text="", font=t.font(t.SIZE_CAPTION),
                                     text_color=t.TEXT_SECONDARY, anchor="w",
                                     justify="left", wraplength=1200)
        self._db_info.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))

    def on_show(self):
        def work():
            try:
                from laser_trim_analyzer.database.models import AnalysisResult as DBAR
                with self.app.db.session() as s:
                    n = s.query(DBAR.id).count()
                path = getattr(getattr(self.app.config, "database", None), "path", "?")
                txt = (f"Database: {path} — {n:,} trim units on record"
                       + ("   ⚠ EMPTY database — a first run processes everything as new"
                          if n == 0 else ""))
            except Exception as e:
                txt = f"Database check failed: {e}"
            self.safe_after(lambda: self._db_info.configure(text=txt))
        threading.Thread(target=work, daemon=True).start()

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
        # Read the Tk variable HERE, on the UI thread — the worker previously
        # called self._incremental.get() (a Tcl call) off-thread, violating
        # the "workers never call Tk" rule (code-review finding #8).
        incremental = bool(self._incremental.get())
        threading.Thread(target=self._run, args=(folder, incremental),
                         daemon=True).start()

    # ---- progress (kept for tests: single-event path on the Tk thread) ----
    def _apply_progress(self, status: ProcessingStatus, total: int) -> None:
        if status.status == "scanning":
            # "Found N new files (M already in database)" — the headline the
            # user waits for (work finding #11).
            self._progress.set_idle(status.message or "Scanning…")
            return
        if status.status in ("completed", "skipped", "failed"):
            self._done += 1
        self._progress.set_progress(self._done, total, status.filename or "")
        if status.status == "skipped":
            self._progress.increment("skipped")

    def _paint(self, coalescer: ProgressCoalescer, total: int) -> None:
        """Paint ONE coalesced snapshot. Tk thread only.

        One post per file made the whole app sluggish for the length of a
        170k-file batch (2026-07-13); the counters accumulate in memory and
        this runs a few times a second.
        """
        snap = coalescer.drain()
        if snap["scan_msg"] and not snap["moved"]:
            self._progress.set_idle(snap["scan_msg"])
            return
        if snap["moved"] or snap["done"]:
            self._progress.set_progress(snap["done"], total, snap["file"])
            if snap["counts"] or snap["reasons"]:
                self._progress.add_counts(snap["counts"], snap["reasons"])

    def _run(self, folder: str, incremental: bool = True) -> None:
        """Worker: drive the shared pipeline for one folder. Never calls Tk."""
        coalescer = ProgressCoalescer()
        total = {"n": 0}          # denominator, learned once the walk is done

        ticker = ProgressTicker(
            lambda: self.safe_after(lambda: self._paint(coalescer, total["n"]))
        ).start()
        try:
            result = ingest_run.run_folder(
                folder, db=self.app.db, config=self.app.config,
                incremental=incremental, progress=coalescer,
                on_phase=lambda msg: self.safe_after(
                    lambda m=msg: self._progress.set_idle(m)),
                on_total=lambda n: total.__setitem__("n", n))
        finally:
            ticker.stop()
        self.safe_after(lambda: self._paint(coalescer, total["n"]))

        if not result.ok:
            self.safe_after(lambda e=result.error: self._progress.set_idle(f"Stopped: {e}"))
        elif result.summary is not None:
            # Authoritative final tally from BatchSummary (reconciles the live
            # counts, including skips).
            self.safe_after(lambda sm=result.summary: self._progress.set_final(sm))
        self.safe_after(self._on_done)

    def _on_done(self):
        self._start_button.configure(state="normal")
        self._goto_triage.pack(side="top", anchor="w", pady=(self.theme.SPACE_SM, 0))
