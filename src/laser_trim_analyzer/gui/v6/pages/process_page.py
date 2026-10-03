"""Process -- both runs, one pipeline (Graphite redesign, 2026-10-02).

Spec: docs/superpowers/specs/2026-10-02-graphite-redesign-design.md ("Process new files").

  * "New files from your folders", at the top: the remembered folder list (Settings -> Ingest
    folders), in order, through `core/ingest_run.run_folders` -- what Home's "Bring in what's new"
    card did, moved here whole: run, stop, progress, the one-line summary, and the way to the
    folder list in Settings (`NewFilesRun`). The top bar's blue "Process new files" is the same
    run: `V6App.process_new_files` shows this page and calls `start_new_files()`.
  * "A specific folder", below it: today's one-off picker, through `core/ingest_run.run_folder`.

ONE pipeline (spec 2026-08-29: "same worker, no duplicate pipeline"): core/ingest_run is the only
place a run is driven; this page is the pickers, the progress and the marshalling between the
worker and Tk.

A run that FINISHES lands on the Overview, whose top then shows the run's summary line
(`land_on_overview`; it used to land on Triage). A run stopped part-way stays here, where its own
line says how far it got and that pressing again continues.

The top bar holds the app's ONE blue button; nothing on this page is blue-filled.

Thread discipline (CLAUDE.md rule 5): every Tk read happens on the Tk thread and is handed INTO
the worker as plain values; the worker only posts back through `safe_after`.
"""
import logging
import threading
from threading import Event

import customtkinter as ctk

logger = logging.getLogger(__name__)

from laser_trim_analyzer.core import ingest_run
from laser_trim_analyzer.core.ingest_run import (
    EtaEstimator, IngestReport, ProgressCoalescer, ProgressTicker, format_ingest_summary,
    format_progress_line)
from laser_trim_analyzer.core.models import ProcessingStatus
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets import blocks
from laser_trim_analyzer.gui.v6.widgets.folder_picker import FolderPicker
from laser_trim_analyzer.gui.v6.widgets.process_progress_section import ProcessProgressSection

RUN_LABEL = "Process new files"        # the top bar's button says the same: it is the same run


def _plain_button(parent, t, text: str, command) -> ctk.CTkButton:
    """A run's own button: a card-coloured one, never the blue -- the top bar holds the one."""
    return ctk.CTkButton(parent, text=text, command=command, fg_color=t.CARD,
                         hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY, border_width=1,
                         border_color=t.BORDER, corner_radius=t.RADIUS_MD,
                         font=t.font(t.SIZE_BODY, "bold"), height=32)


class ProcessPage(PageBase):
    page_title = "Process"

    def __init__(self, master, *, theme, app, page_title="Process"):
        self._done = 0
        self._cancel = None            # threading.Event while the one-off run is in flight
        super().__init__(master, theme=theme, app=app, page_title=page_title)
        self._new_files.refresh_folders()

    def build_content(self, parent):
        t = self.theme
        # Scrollable: two runs, each with its progress once started, do not fit 1280x720 at once.
        self._body = body = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        body.pack(side="top", fill="both", expand=True)

        self._new_files = NewFilesRun(body, page=self)
        self._new_files.pack(side="top", fill="x", pady=(0, t.SPACE_XL))

        # ---- a specific folder: the one-off run, as before -------------------------------------
        self._zone_header(body, "A specific folder",
                          "A folder that is not on the list, processed once.")
        self._folder_picker = FolderPicker(body, theme=t, on_change=lambda _p: self._update_start())
        self._folder_picker.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        self._incremental = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(body, text="Incremental mode (skip already-processed files)",
                        variable=self._incremental, font=t.font(t.SIZE_BODY), text_color=t.TEXT_PRIMARY,
                        fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER,
                        checkmark_color=t.TEXT_INVERSE)\
            .pack(side="top", anchor="w", pady=(0, t.SPACE_MD))
        self._start_button = _plain_button(body, t, "Start processing", self._start)
        self._start_button.configure(state="disabled")
        self._start_button.pack(side="top", anchor="w", pady=(0, t.SPACE_MD))
        # Packed only while a run is in flight (see _set_running) -- the same cooperative stop
        # the remembered run offers, on the same shared runner.
        self._stop_button = ctk.CTkButton(body, text="Stop", fg_color=t.CARD,
                                          hover_color=t.ELEVATED,
                                          text_color=t.TEXT_PRIMARY,
                                          command=self._stop,
                                          corner_radius=t.RADIUS_SM)
        # Packed when this run starts (see _start), never before: an idle "Ready" bar and five
        # zero counters say nothing (James, 2026-10-02: "there is so much going on").
        self._progress = ProcessProgressSection(body, theme=t)
        # Which database + how much it knows -- BEFORE processing starts. The work incident
        # (2026-07-09) would have been obvious in one glance if this had said "0 units": wrong or
        # empty database, don't hit Start.
        self._db_info = ctk.CTkLabel(body, text="", font=t.font(t.SIZE_CAPTION),
                                     text_color=t.TEXT_SECONDARY, anchor="w", justify="left")
        self._db_info.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
        # Bound ONCE: the page's body is never destroyed/rebuilt for the page's own lifetime, so
        # this never stacks a second <Configure> handler.
        blocks.wrap_to_width(self._db_info, body)

    def on_show(self):
        self._new_files.refresh_folders()

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

    # ---- the top bar's button ------------------------------------------------------------------
    def start_new_files(self) -> None:
        """Start the remembered-folder run (V6App.process_new_files, with this page on screen).
        No folders: nothing starts, and the section says where to add them. A run already in
        flight: nothing new starts -- the page is showing it."""
        self._new_files.refresh_folders()
        self._new_files._start()

    # ---- where a finished run lands ------------------------------------------------------------
    def land_on_overview(self, summary: str, *, ok: bool = True) -> None:
        """A finished run lands on the Overview, whose top shows its summary line. Only when this
        page is the one on screen: a run that finishes while you are reading a model never pulls
        you away -- the Overview simply has the line, and fresh numbers, when you go to it."""
        pages = getattr(self.app, "page_container", None)
        if pages is None:
            return
        overview = pages.get_page("home")
        if overview is not None:
            overview.set_run_summary(summary, ok=ok)
        if pages.current_page == "process":
            self.app.show_page("home")
        elif pages.current_page == "home" and overview is not None:
            overview.on_show()           # on screen already: its numbers changed under it

    # ---- the one-off folder run ------------------------------------------------------------------
    def _update_start(self):
        self._start_button.configure(state="normal" if self._folder_picker.value() else "disabled")

    def _start(self):
        folder = self._folder_picker.value()
        if not folder:
            return
        self._cancel = Event()          # fresh per run; never a reused event
        self._set_running(True)
        if self._progress.winfo_manager() == "":
            self._progress.pack(side="top", fill="x", pady=(0, self.theme.SPACE_MD),
                                before=self._db_info)
        self._progress.reset()
        self._done = 0
        # Read the Tk variable HERE, on the UI thread -- the worker previously called
        # self._incremental.get() (a Tcl call) off-thread, violating the "workers never call Tk"
        # rule (code-review finding #8).
        incremental = bool(self._incremental.get())
        thread = threading.Thread(target=self._run,
                                  args=(folder, incremental, self._cancel),
                                  daemon=True)
        thread.start()
        register = getattr(self.app, "register_ingest", None)
        if register is not None:
            register(self._cancel, thread, "An ingest")

    def _stop(self):
        """Ask the run to stop; it lands at the end of the current batch."""
        if self._cancel is None:
            return
        self._cancel.set()
        self._stop_button.configure(text="Stopping after this batch…",
                                    state="disabled")

    def _set_running(self, running: bool) -> None:
        self._start_button.configure(state="disabled" if running else "normal")
        if running:
            self._stop_button.configure(text="Stop", state="normal")
            self._stop_button.pack(side="top", anchor="w",
                                   pady=(0, self.theme.SPACE_MD),
                                   after=self._start_button)
        else:
            self._stop_button.pack_forget()

    # ---- progress (kept for tests: single-event path on the Tk thread) ----
    def _apply_progress(self, status: ProcessingStatus, total: int) -> None:
        if status.status == "scanning":
            # "Found N new files (M already in database)" -- the headline the user waits for
            # (work finding #11).
            self._progress.set_idle(status.message or "Scanning…")
            return
        if status.status == "known":
            # The scan's one-shot credit for files the database already has (see
            # ProgressCoalescer.note) -- progress, not work.
            self._done += int(getattr(status, "count", 0) or 0)
        elif status.status in ("completed", "skipped", "failed"):
            self._done += 1
        self._progress.set_progress(self._done, total, status.filename or "")
        if status.status == "skipped":
            self._progress.increment("skipped")

    def _paint(self, coalescer: ProgressCoalescer, state: dict,
               eta: EtaEstimator) -> None:
        """Paint ONE coalesced snapshot. Tk thread only.

        One post per file made the whole app sluggish for the length of a 170k-file batch
        (2026-07-13); the counters accumulate in memory and this runs a few times a second.
        Same words as the remembered run, from the same helper: one folder is just a run of one,
        so the rate and the ETA come out of core/ingest_run either way.
        """
        snap = coalescer.drain()
        eta.note(snap["processed"])
        if snap["scan_msg"] and not snap["moved"]:
            self._progress.set_phase(snap["scan_msg"])
            return
        if snap["moved"] or snap["done"]:
            total = state["n"]
            self._progress.set_overall(snap["done"], total, format_progress_line(
                snap["done"], total, rate=eta.rate(),
                eta=eta.eta_text(max(0, total - snap["done"])),
                elapsed=eta.elapsed(), filename=snap["file"]))
            if snap["counts"] or snap["reasons"]:
                self._progress.add_counts(snap["counts"], snap["reasons"])

    def _run(self, folder: str, incremental: bool = True, cancel=None) -> None:
        """Worker: drive the shared pipeline for one folder. Never calls Tk."""
        coalescer = ProgressCoalescer()
        state = {"n": 0}          # denominator, learned once the walk is done
        eta = EtaEstimator()

        ticker = ProgressTicker(
            lambda: self.safe_after(lambda: self._paint(coalescer, state, eta))
        ).start()
        try:
            result = ingest_run.run_folder(
                folder, db=self.app.db, config=self.app.config,
                incremental=incremental, progress=coalescer,
                on_phase=lambda msg: self.safe_after(
                    lambda m=msg: self._progress.set_idle(m)),
                on_total=lambda n: state.__setitem__("n", n), cancel=cancel)
        finally:
            ticker.stop()
        self.safe_after(lambda: self._paint(coalescer, state, eta))

        if not result.ok:
            self.safe_after(lambda e=result.error: self._progress.set_idle(f"Stopped: {e}"))
        elif result.cancelled:
            # NOT set_final(): a full-looking tally on a folder that stopped half-way is the one
            # thing this must never say.
            self.safe_after(lambda r=result: self._progress.set_phase(
                f"Stopped after {r.new_files:,} of {r.files_found:,} files — "
                "start again to continue where this left off."))
        elif result.summary is not None:
            # Authoritative final tally from BatchSummary (reconciles the live counts, including
            # skips).
            self.safe_after(lambda sm=result.summary: self._progress.set_final(sm))
        self.safe_after(lambda r=result: self._on_done(r))

    def _on_done(self, result=None):
        """Tk thread. `result` is the folder's FolderResult -- None from a caller that only
        wants the buttons back."""
        self._set_running(False)
        # Tell the app this run is over (see V6App.unregister_ingest): a worker that has just
        # posted its result is still alive for a moment, and the one-job-at-a-time check must not
        # read that as a run.
        drop = getattr(self.app, "unregister_ingest", None)
        if drop is not None and self._cancel is not None:
            drop(self._cancel)
        if result is not None and not result.cancelled:
            # One folder is a run of one: the Overview's line is the same sentence the remembered
            # run writes ("1 folder · 52 new files · 40 s"), and a failure is named in it.
            report = IngestReport(results=[result], seconds=result.seconds, folders_requested=1)
            self.land_on_overview(format_ingest_summary(report), ok=report.ok)


class NewFilesRun(ctk.CTkFrame):
    """"New files from your folders" -- the remembered folder list, in order, through the shared
    multi-folder run (`core/ingest_run.run_folders`). Moved here whole from Home's "Bring in
    what's new" card (2026-10-02); the top bar's blue button starts the same run.

    The page's helpers -- `safe_after`, the app -- are borrowed from `page`: this section is the
    page's, never a page of its own.
    """

    def __init__(self, master, *, page: ProcessPage):
        super().__init__(master, fg_color="transparent")
        self._page = page
        self.app = page.app
        self.theme = t = page.theme
        self._running = False
        self._cancel = None            # threading.Event while a run is in flight

        page._zone_header(self, "New files from your folders",
                          "Your remembered folders, in order — laser folders first, Final Test "
                          "last. Files already in the database are skipped.")
        # What the button will do, before it is pressed: which folders, in which order.
        self._folders_label = ctk.CTkLabel(self, text="", anchor="w", justify="left",
                                           font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY)
        self._folders_label.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        # Built once, never destroyed/rebuilt (only .configure(text=...) later): bound once.
        blocks.wrap_to_width(self._folders_label, self)

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(side="top", fill="x")
        self._run_button = _plain_button(row, t, RUN_LABEL, self._start)
        self._run_button.configure(state="disabled")
        self._run_button.pack(side="left")
        # Packed only while a run is in flight (see _set_running). A Stop button on an idle
        # screen is a question with no answer; a run with no Stop button is hours you cannot get
        # back -- the first full ingest is ~4 hours here and 6-8 on the laptop.
        self._stop_button = ctk.CTkButton(
            row, text="Stop", height=32, fg_color=t.CARD, hover_color=t.ELEVATED,
            text_color=t.TEXT_PRIMARY, corner_radius=t.RADIUS_SM,
            font=t.font(t.SIZE_BODY, "bold"), command=self._stop)
        self._settings_link = blocks.link_button(row, t, "Edit folders in Settings",
                                                 self._open_settings)
        self._settings_link.pack(side="left", padx=(t.SPACE_MD, 0))

        # Packed when a run starts (see _start); stays afterwards with the run's tally.
        self._progress = ProcessProgressSection(self, theme=t)
        # The one line the whole run reduces to. Stays on screen afterwards -- "did that do
        # anything?" is a question the app should not need to be asked twice.
        self._summary = ctk.CTkLabel(self, text="", anchor="w", justify="left",
                                     font=t.font(t.SIZE_BODY), text_color=t.TEXT_PRIMARY)
        self._summary.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
        blocks.wrap_to_width(self._summary, self)

    def safe_after(self, fn, delay: int = 0) -> None:
        self._page.safe_after(fn, delay)

    # ---- folder list -------------------------------------------------------
    def _folders(self):
        cfg = getattr(self.app.config, "ingest", None)
        return list(getattr(cfg, "folders", []) or [])

    def refresh_folders(self) -> None:
        """Re-read the configured folders and set the button's state. Tk thread."""
        folders = self._folders()
        if not folders:
            # Empty state, not a modal: a blocking dialog on every cold start of a single-user
            # app is a tax, and this one has somewhere to go.
            self._run_button.configure(state="disabled")
            self._folders_label.configure(
                text="No ingest folders yet — add the laser folders and the Final Test folder "
                     "in Settings → Ingest folders, and this runs them all.")
            return
        if not self._running:
            self._run_button.configure(state="normal")
        n = len(folders)
        noun = "folder" if n == 1 else "folders"
        self._folders_label.configure(
            text=f"{n} {noun}, in this order:  " + "  →  ".join(folders))

    # ---- the run -----------------------------------------------------------
    def _start(self) -> None:
        # Everything Tk (and everything config) is read HERE, on the UI thread, and handed to
        # the worker as plain values.
        folders = self._folders()
        if not folders or self._running:
            return
        # One long job at a time (2026-09-14). A re-grade and an ingest both want the database
        # write lock and both pull files over the same SMB share; run together, the index load
        # went 2.6 s -> 101 s per folder, the final-test verify pass 140 s -> 542 s, and the batch
        # was cancelled after 0 of 1,371 files. Refusing is not a limitation, it is the
        # difference between one job finishing and neither.
        busy = getattr(self.app, "active_run_name", lambda: None)()
        if busy:
            self._summary.configure(
                text=f"{busy} is running — stop it first, then press this "
                     f"again. Two jobs over the plant share make each other "
                     f"slower than either one alone.")
            return
        self._cancel = Event()         # a FRESH event: a stopped run must not
        self._set_running(True)        # leave the next one pre-cancelled
        if self._progress.winfo_manager() == "":
            self._progress.pack(side="top", fill="x", pady=(self.theme.SPACE_SM, 0),
                                before=self._summary)
        self._progress.reset()
        self._summary.configure(text="")
        # "New" is the whole promise of the button, so the run is always incremental; the
        # specific-folder run below keeps the checkbox for a re-run.
        thread = threading.Thread(target=self._run,
                                  args=(folders, True, self._cancel),
                                  daemon=True)
        thread.start()
        # So closing the window can stop this cleanly instead of destroying Tk out from under a
        # batch that is mid-write.
        register = getattr(self.app, "register_ingest", None)
        if register is not None:
            register(self._cancel, thread, "An ingest")

    def _stop(self) -> None:
        """Ask the run to stop. Tk thread; the worker sees a set() Event.

        Relabels IMMEDIATELY, because the stop lands at the end of the current 20-file batch and
        a button that looks unpressed gets pressed again.
        """
        if not self._running or self._cancel is None:
            return
        self._cancel.set()
        self._stop_button.configure(text="Stopping after this batch…",
                                    state="disabled")

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._run_button.configure(state="disabled" if running else "normal",
                                   text=("Processing…" if running else RUN_LABEL))
        if running:
            self._stop_button.configure(text="Stop", state="normal")
            self._stop_button.pack(side="left", after=self._run_button,
                                   padx=(self.theme.SPACE_SM, 0))
        else:
            self._stop_button.pack_forget()

    def _paint(self, coalescer: ProgressCoalescer, state: dict,
               eta: EtaEstimator) -> None:
        """Paint ONE coalesced snapshot (Tk thread). See core/ingest_run.py.

        The whole RUN's numbers, not the folder's: how far through every file the pre-scan
        found, how fast, and how much longer. All of it is arithmetic on ints -- cheap enough for
        4 Hz on the Tk thread, and the only thread allowed to hold these widgets.

        The estimator is fed on EVERY paint, including the ones where nothing moved: that is what
        turns a stalled share into a falling rate and a growing ETA instead of a number frozen at
        whatever it last was.
        """
        snap = coalescer.drain()
        eta.note(snap["processed"])
        if snap["scan_msg"] and not snap["moved"]:
            # set_phase, never set_idle: set_idle zeroes the bar, and a bar that drops to zero
            # mid-run reads as progress thrown away.
            self._progress.set_phase(snap["scan_msg"])
            return
        if snap["moved"] or snap["done"]:
            total = state["n"]
            self._progress.set_overall(snap["done"], total, format_progress_line(
                snap["done"], total, folder=state["folder"],
                folders=state["folders"], rate=eta.rate(),
                eta=eta.eta_text(max(0, total - snap["done"])),
                elapsed=eta.elapsed(), filename=snap["file"]))
            if snap["counts"] or snap["reasons"]:
                self._progress.add_counts(snap["counts"], snap["reasons"])

    def _run(self, folders, incremental: bool, cancel=None) -> None:
        """Worker: drive the shared multi-folder run. Never touches Tk."""
        coalescer = ProgressCoalescer()
        # The run's own numbers, shared with _paint: the overall denominator (announced once,
        # after run_folders has scanned every folder) and where in the folder list it is.
        # Written here and in folder_start, read in _paint -- plain ints, no Tk.
        state = {"n": 0, "folder": 0, "folders": len(folders)}
        eta = EtaEstimator()
        ticker = ProgressTicker(
            lambda: self.safe_after(
                lambda: self._paint(coalescer, state, eta))).start()

        def folder_start(i, n, folder):
            # The coalescer is NOT reset. The bar's denominator is the whole run now, so zeroing
            # the count at a folder boundary would throw away everything the previous folders
            # did -- which is exactly what made "how many files has it got to go?" unanswerable.
            state["folder"], state["folders"] = i, n
            self.safe_after(lambda: self._progress.set_phase(
                f"Folder {i} of {n}: {folder}"))

        try:
            report = ingest_run.run_folders(
                folders, db=self.app.db, config=self.app.config,
                incremental=incremental, progress=coalescer,
                on_phase=lambda msg: self.safe_after(
                    lambda m=msg: self._progress.set_phase(m)),
                on_total=lambda n: state.__setitem__("n", n),
                on_folder_start=folder_start, cancel=cancel)
        except Exception as exc:
            # run_folders returns failures as data; reaching here means something outside a
            # folder's own run broke. Say so rather than leaving the button greyed out forever
            # (2026-07-09).
            logger.exception("Ingest run failed")
            self.safe_after(lambda e=exc: self._summary.configure(
                text=f"Stopped: {e}"))
            self.safe_after(lambda: self._set_running(False))
            return
        finally:
            ticker.stop()
        self.safe_after(lambda: self._paint(coalescer, state, eta))
        self.safe_after(lambda r=report: self._on_run_done(r))

    def _on_run_done(self, report) -> None:
        """Tk thread: the combined summary, the button back -- and, for a run that finished
        rather than stopped, the landing on the Overview with that same line on top."""
        text = format_ingest_summary(report)
        self._summary.configure(text=text)
        self._set_running(False)
        # Say it is over, rather than leaving the app to infer it from a thread that is still
        # alive for a few more instructions -- otherwise pressing the button straight after a run
        # would be refused by the one-job-at-a-time check.
        drop = getattr(self.app, "unregister_ingest", None)
        if drop is not None and self._cancel is not None:
            drop(self._cancel)
        if not report.cancelled:
            self._page.land_on_overview(text, ok=report.ok)

    # ---- routing -----------------------------------------------------------
    def _open_settings(self) -> None:
        self.app.show_page("settings")
