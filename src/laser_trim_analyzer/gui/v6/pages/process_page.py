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

    # ---- coalesced progress (2026-07-13: one safe_after per FILE made the
    # whole app sluggish for the length of a 170k-file batch — the Tk thread
    # was permanently busy repainting counters). Workers only bump in-memory
    # counters; a 4 Hz ticker paints ONE snapshot. ----
    def _pending_note(self, status: ProcessingStatus, lock, pend) -> None:
        with lock:
            if status.status == "scanning":
                pend["scan_msg"] = status.message or "Scanning…"
                return
            if status.status in ("completed", "skipped", "failed"):
                pend["done"] += 1
                pend["file"] = status.filename or pend["file"]
            if status.status == "skipped":
                pend["counts"]["skipped"] += 1

    def _pending_flush(self, lock, pend, total) -> None:
        with lock:
            scan_msg = pend.pop("scan_msg", None)
            done, fname = pend["done"], pend["file"]
            counts = {k: v for k, v in pend["counts"].items() if v}
            reasons = list(pend["reasons"][-10:])
            moved = pend["moved"] or bool(counts) or bool(reasons)
            pend["counts"] = {k: 0 for k in pend["counts"]}
            pend["reasons"] = []
            pend["moved"] = False
        if scan_msg and not moved:
            self._progress.set_idle(scan_msg)
            return
        if moved or done:
            self._progress.set_progress(done, total, fname)
            if counts or reasons:
                self._progress.add_counts(counts, reasons)

    def _run(self, folder: str) -> None:
        self.safe_after(lambda: self._progress.set_idle(
            "Scanning folder for Excel files… (network folders can take a minute)"))
        files, disk_stats = self._discover(folder)
        if not files:
            self.safe_after(lambda: self._progress.set_idle("No .xls/.xlsx files found."))
            self.safe_after(lambda: self._start_button.configure(state="normal"))
            return
        processor = Processor(config=self.app.config)        # I7: no db= param
        total = len(files)
        self.safe_after(lambda t_=total: self._progress.set_idle(
            f"Checking {t_:,} files against the database…"))

        import time as _time
        lock = threading.Lock()
        pend = {"done": 0, "file": "", "scan_msg": None, "moved": False,
                "counts": {"passed": 0, "warnings": 0, "failed": 0,
                           "skipped": 0, "errors": 0},
                "reasons": []}
        ticker_stop = threading.Event()

        def progress_callback(status: ProcessingStatus) -> None:
            self._pending_note(status, lock, pend)   # worker-side, no Tk

        def _tick():
            while not ticker_stop.wait(0.25):
                self.safe_after(lambda: self._pending_flush(lock, pend, total))
        threading.Thread(target=_tick, daemon=True).start()

        def _note_bucket(bucket, reason=""):
            with lock:
                pend["counts"][bucket] = pend["counts"].get(bucket, 0) + 1
                pend["moved"] = True
                if reason and bucket in ("failed", "errors"):
                    pend["reasons"].append(reason)

        gen = processor.process_batch([Path(p) for p in files],
                                      progress_callback=progress_callback,
                                      incremental=self._incremental.get(),
                                      disk_stats=disk_stats)
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
                            _note_bucket("skipped")
                        else:
                            _note_bucket("errors",
                                         f"{result.metadata.filename}: save failed: {exc}")
                model = getattr(result.metadata, "model", None)
                if model and model != "Unknown":
                    models_in_batch.add(model)
                bucket = self._bucket_for_status(result.overall_status)
                reason = (f"{result.metadata.filename}: {result.overall_status.value}"
                          if bucket in ("failed", "errors") else "")
                _note_bucket(bucket, reason)
        except StopIteration as stop:
            summary = stop.value
            ticker_stop.set()
            self.safe_after(lambda: self._pending_flush(lock, pend, total))
        except Exception as exc:
            ticker_stop.set()
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
            # Re-link final tests FIRST: FT files that arrived before their
            # trim files matched nothing at save time (2026-07-13 finding —
            # a third of recent unmatched FT records had an in-window trim
            # that simply wasn't in the DB yet). Must run before the drift
            # advance so escape/FT metrics see the fresh links.
            try:
                rl = self.app.db.rematch_unlinked_final_tests()
                if rl.get("new_matches"):
                    self.safe_after(lambda n=rl["new_matches"]: self._progress.set_idle(
                        f"Re-linked {n:,} final-test records to their trim data…"))
            except Exception:
                logger.exception("Post-batch FT rematch failed")
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

    def _discover(self, folder: str):
        """Walk the tree AND capture (size, mtime) from the directory listings.

        On Windows/SMB, scandir returns each entry's stat data with the
        listing itself — the same network round trip. Passing it to the
        processor makes the incremental check pure in-memory comparison
        (V4-era seconds) instead of one stat() round trip per file
        (Friday 2026-07-10: 73 minutes for 170k files).
        """
        out: List[str] = []
        stats: dict = {}
        stack = [folder]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                            elif entry.name.lower().endswith((".xls", ".xlsx")):
                                out.append(entry.path)
                                st = entry.stat()
                                stats[entry.path] = (st.st_size, st.st_mtime)
                        except OSError:
                            continue   # unreadable entry: skip, don't die
            except OSError:
                logger.warning("Could not list folder: %s", d)
        return out, stats

    def _on_done(self):
        self._start_button.configure(state="normal")
        self._goto_triage.pack(side="top", anchor="w", pady=(self.theme.SPACE_SM, 0))
