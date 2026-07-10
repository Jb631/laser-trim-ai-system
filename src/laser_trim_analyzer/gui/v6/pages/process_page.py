"""Spec 3e — ProcessPage: folder pick → batch process → land on Triage.
Persists trim results (GUI owns trim save); progress from ProcessingStatus + a local
done-counter; final tally from BatchSummary."""
import logging
import os
import threading
from pathlib import Path
from typing import List

import customtkinter as ctk

logger = logging.getLogger(__name__)

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
        models_in_batch = set()
        try:
            while True:
                result = next(gen)
                # Persist trim results (GUI owns trim save; FT/smoothness already saved internally).
                if getattr(result, "file_type", "trim") == "trim":
                    try:
                        self.app.db.save_analysis(result)
                    except Exception as exc:
                        # A duplicate hitting the unique constraint means the
                        # unit is ALREADY in the database (e.g. same file under
                        # a second path form) — that's a skip, not an error.
                        if "UNIQUE constraint" in str(exc) or "IntegrityError" in type(exc).__name__:
                            self.safe_after(lambda r=result: self._progress.increment(
                                "skipped", reason=f"{r.metadata.filename}: already in database"))
                        else:
                            self.safe_after(lambda e=exc, r=result: self._progress.increment(
                                "errors", reason=f"{r.metadata.filename}: save failed: {e}"))
                model = getattr(result.metadata, "model", None)
                if model and model != "Unknown":
                    models_in_batch.add(model)
                bucket = self._bucket_for_status(result.overall_status)
                reason = (f"{result.metadata.filename}: {result.overall_status.value}"
                          if bucket in ("failed", "errors") else "")
                self.safe_after(lambda b=bucket, rs=reason: self._progress.increment(b, reason=rs))
        except StopIteration as stop:
            summary = stop.value
        except Exception as exc:
            # Work incident 2026-07-09: an exception here previously KILLED the
            # worker thread silently — Start stayed disabled, the app looked
            # locked, and the reason never reached the screen. Surface it.
            logger.exception("Batch processing aborted")
            msg = str(exc)
            self.safe_after(lambda m=msg: self._progress.set_idle(f"Stopped: {m}"))
            self.safe_after(self._on_done)
            return
        # Advance drift detectors over the just-saved data so Triage reflects
        # THIS batch (P0 fix: advance_drift_state previously had no callers —
        # drift state stayed frozen at training time until a full retrain).
        if models_in_batch:
            try:
                from laser_trim_analyzer.ml.drift_training import advance_drift_state
                advanced = 0
                for m in sorted(models_in_batch):
                    advanced += advance_drift_state(self.app.db, model=m)
                logger.info("Drift state advanced for %d (model, metric) rows "
                            "across %d models", advanced, len(models_in_batch))
            except Exception:
                logger.exception("Drift advance after batch failed")
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
