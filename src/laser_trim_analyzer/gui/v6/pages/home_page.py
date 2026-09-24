"""Spec 3f — HomePage: the landing view. Ingest at the top, FOCUS below it.

Spec: docs/superpowers/specs/2026-08-29-app-shape-investigate-design.md §1.
The app's first screen answers the two questions James actually opens it with,
in the order he asks them: "pull in whatever is new" and then "is anything
drifting?"

  * ONE button runs the remembered folder list through the existing batch
    pipeline (`core/ingest_run.run_folders`) — the same worker the Process
    page drives, never a second copy of it. The per-folder phase line goes to
    the progress section as it goes, and one combined summary lands at the
    end: "3 folders · 214 new files · 2 min 40 s".
  * The Process page's picker stays one click away ("process a specific
    folder…") for the one-off folder that isn't on the list.
  * Below it, the FOCUS list — the same widget and the same loader Triage
    uses (`widgets/focus_list_zone.py`, `focus_data.load_focus`), because the
    two screens disagreeing about what is drifting would be worse than either
    of them being wrong.

Thread discipline (CLAUDE.md rule 5): every Tk read happens on the Tk thread
and is passed INTO the worker; the worker only posts back through
`safe_after`. The folder list is read in `_start`, before the thread exists.
"""
import logging
import threading
from threading import Event

import customtkinter as ctk

logger = logging.getLogger(__name__)

from laser_trim_analyzer.core import ingest_run
from laser_trim_analyzer.core.ft_regrade import legacy_ft_count, legacy_ft_notice
from laser_trim_analyzer.core.ingest_run import (
    EtaEstimator, ProgressCoalescer, ProgressTicker, format_ingest_summary,
    format_progress_line, unreadable_count, unreadable_notice)
from laser_trim_analyzer.gui.v6.focus_data import load_focus
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.focus_list_zone import FocusListZone
from laser_trim_analyzer.gui.v6.widgets.process_progress_section import (
    ProcessProgressSection)


# Every full-width Home line wraps here. 1100 was wider than the room these lines get at the
# 1280x720 the audit checks (the run card is ~1,064 px there, ~1,040 inside its padding), so a
# real folder list would have been cut off at its right edge -- invisible while no folders were
# listed (final review, 2026-09-24). 950 is the value Task 9 gave the Model page's full-width
# lines for the same width, with the same margin.
_WRAP = 950


class HomePage(PageBase):
    page_title = "Home"

    def __init__(self, master, *, theme, app, page_title="Home"):
        self._running = False
        self._cancel = None            # threading.Event while a run is in flight
        super().__init__(master, theme=theme, app=app, page_title=page_title)
        self.refresh_folders()

    # ---- construction ------------------------------------------------------
    def build_content(self, parent):
        t = self.theme
        self._zone_header(parent, "Bring in what's new",
                          "your remembered folders, in order, through the same "
                          "batch the Process page runs")

        card = ctk.CTkFrame(parent, fg_color=t.CARD, corner_radius=t.RADIUS_MD)
        card.pack(side="top", fill="x", pady=(0, t.SPACE_LG))
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(side="top", fill="x", padx=t.SPACE_MD, pady=t.SPACE_MD)

        top = ctk.CTkFrame(inner, fg_color="transparent")
        top.pack(side="top", fill="x")
        self._run_button = ctk.CTkButton(
            top, text="Process everything new", state="disabled", height=38,
            font=t.font(t.SIZE_HEADING, "bold"), fg_color=t.ACCENT,
            hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE,
            corner_radius=t.RADIUS_SM, command=self._start)
        self._run_button.pack(side="left")
        # Packed only while a run is in flight (see _set_running). A Stop
        # button on an idle screen is a question with no answer; a run with no
        # Stop button is hours you cannot get back — the first full ingest is
        # ~4 hours here and 6-8 on the laptop.
        self._stop_button = ctk.CTkButton(
            top, text="Stop", height=38, fg_color=t.CARD, hover_color=t.ELEVATED,
            text_color=t.TEXT_PRIMARY, corner_radius=t.RADIUS_SM,
            font=t.font(t.SIZE_BODY, "bold"), command=self._stop)
        # The escape hatch, right beside the button it is an alternative to.
        self._specific_button = ctk.CTkButton(
            top, text="process a specific folder…", fg_color="transparent",
            hover_color=t.ELEVATED, text_color=t.ACCENT, corner_radius=t.RADIUS_SM,
            font=t.font(t.SIZE_BODY), command=self._open_process)
        self._specific_button.pack(side="left", padx=(t.SPACE_SM, 0))
        self._settings_button = ctk.CTkButton(
            top, text="Set up folders in Settings", fg_color=t.CARD,
            hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY,
            corner_radius=t.RADIUS_SM, font=t.font(t.SIZE_BODY),
            command=self._open_settings)     # packed only in the empty state

        # What the button will do, before it is pressed: which folders, in
        # which order. "Process everything new" is otherwise a promise with no
        # visible terms.
        self._folders_label = ctk.CTkLabel(
            inner, text="", anchor="w", justify="left", wraplength=_WRAP,
            font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY)
        self._folders_label.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))

        self._progress = ProcessProgressSection(inner, theme=t)
        self._progress.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))

        # The one line the whole run reduces to. Stays on screen afterwards —
        # "did that do anything?" is a question the app should not need to be
        # asked twice.
        self._summary = ctk.CTkLabel(inner, text="", anchor="w", justify="left",
                                     wraplength=_WRAP, font=t.font(t.SIZE_BODY),
                                     text_color=t.TEXT_PRIMARY)
        self._summary.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))

        # One line, only when there is something to say: final-test records
        # graded before the ignore-window fix (2026-09-13) carry a verdict
        # that may include sweep points the station never graded. It is not an
        # error state and it does not block anything, so it is a caption, not
        # a banner — but it stays up until Settings clears it, because every
        # final-test number on the screens below is computed from those rows.
        self._legacy_ft_label = ctk.CTkLabel(
            parent, text="", anchor="w", justify="left", wraplength=_WRAP,
            font=t.font(t.SIZE_CAPTION), text_color=t.TIER_WARNING)
        self._legacy_ft_count = 0

        # The same shape, for the files the button is NOT processing: ones
        # that failed to read on an earlier run and are skipped while they are
        # unchanged on disk (2026-09-17). Silent at zero; while it says
        # anything it also says the way back.
        self._unreadable_label = ctk.CTkLabel(
            parent, text="", anchor="w", justify="left", wraplength=_WRAP,
            font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY)
        self._unreadable_count = 0

        self._zone_header(parent, "What the app is telling you",
                          "drifting now, biggest first — one verdict per lot, "
                          "self-clearing")
        self._focus = FocusListZone(parent, theme=t,
                                    on_row_click=self._on_focus_click)
        self._focus.pack(side="top", fill="both", expand=True)

    # ---- folder list -------------------------------------------------------
    def _folders(self):
        cfg = getattr(self.app.config, "ingest", None)
        return list(getattr(cfg, "folders", []) or [])

    def refresh_folders(self) -> None:
        """Re-read the configured folders and set the button's state. Tk thread."""
        folders = self._folders()
        if not folders:
            # Empty state, not a modal: a blocking dialog on every cold start
            # of a single-user app is a tax, and this one has somewhere to go.
            self._run_button.configure(state="disabled")
            self._folders_label.configure(
                text="No ingest folders configured yet — add the laser folders "
                     "and the Final Test folder in Settings, and this button "
                     "runs them all.")
            self._settings_button.pack(side="left", padx=(self.theme.SPACE_SM, 0))
            return
        self._settings_button.pack_forget()
        if not self._running:
            self._run_button.configure(state="normal")
        n = len(folders)
        noun = "folder" if n == 1 else "folders"
        self._folders_label.configure(
            text=f"{n} {noun}, in this order:  " + "  →  ".join(folders))

    def on_show(self):
        self.refresh_folders()
        self._reload_focus()

    # ---- the run -----------------------------------------------------------
    def _start(self) -> None:
        # Everything Tk (and everything config) is read HERE, on the UI thread,
        # and handed to the worker as plain values.
        folders = self._folders()
        if not folders or self._running:
            return
        # One long job at a time (2026-09-14). A re-grade and an ingest both
        # want the database write lock and both pull files over the same SMB
        # share; run together, the index load went 2.6 s -> 101 s per folder,
        # the final-test verify pass 140 s -> 542 s, and the batch was
        # cancelled after 0 of 1,371 files. Refusing is not a limitation, it
        # is the difference between one job finishing and neither.
        busy = getattr(self.app, "active_run_name", lambda: None)()
        if busy:
            self._summary.configure(
                text=f"{busy} is running — stop it first, then press this "
                     f"again. Two jobs over the plant share make each other "
                     f"slower than either one alone.")
            return
        self._cancel = Event()         # a FRESH event: a stopped run must not
        self._set_running(True)        # leave the next one pre-cancelled
        self._progress.reset()
        self._summary.configure(text="")
        # "New" is the whole promise of the button, so the run is always
        # incremental; the Process page keeps the checkbox for a re-run.
        thread = threading.Thread(target=self._run,
                                  args=(folders, True, self._cancel),
                                  daemon=True)
        thread.start()
        # So closing the window can stop this cleanly instead of destroying Tk
        # out from under a batch that is mid-write.
        register = getattr(self.app, "register_ingest", None)
        if register is not None:
            register(self._cancel, thread, "An ingest")

    def _stop(self) -> None:
        """Ask the run to stop. Tk thread; the worker sees a set() Event.

        Relabels IMMEDIATELY, because the stop lands at the end of the current
        20-file batch and a button that looks unpressed gets pressed again.
        """
        if not self._running or self._cancel is None:
            return
        self._cancel.set()
        self._stop_button.configure(text="Stopping after this batch…",
                                    state="disabled")

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._run_button.configure(state="disabled" if running else "normal",
                                   text=("Processing…" if running
                                         else "Process everything new"))
        if running:
            self._stop_button.configure(text="Stop", state="normal")
            self._stop_button.pack(side="left", after=self._run_button,
                                   padx=(self.theme.SPACE_SM, 0))
        else:
            self._stop_button.pack_forget()

    def _paint(self, coalescer: ProgressCoalescer, state: dict,
               eta: EtaEstimator) -> None:
        """Paint ONE coalesced snapshot (Tk thread). See core/ingest_run.py.

        The whole RUN's numbers, not the folder's: how far through every file
        the pre-scan found, how fast, and how much longer. All of it is
        arithmetic on ints — cheap enough for 4 Hz on the Tk thread, and the
        only thread allowed to hold these widgets.

        The estimator is fed on EVERY paint, including the ones where nothing
        moved: that is what turns a stalled share into a falling rate and a
        growing ETA instead of a number frozen at whatever it last was.
        """
        snap = coalescer.drain()
        eta.note(snap["processed"])
        if snap["scan_msg"] and not snap["moved"]:
            # set_phase, never set_idle: set_idle zeroes the bar, and a bar
            # that drops to zero mid-run reads as progress thrown away.
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
        # The run's own numbers, shared with _paint: the overall denominator
        # (announced once, after run_folders has scanned every folder) and
        # where in the folder list it is. Written here and in folder_start,
        # read in _paint — plain ints, no Tk.
        state = {"n": 0, "folder": 0, "folders": len(folders)}
        eta = EtaEstimator()
        ticker = ProgressTicker(
            lambda: self.safe_after(
                lambda: self._paint(coalescer, state, eta))).start()

        def folder_start(i, n, folder):
            # The coalescer is NOT reset. The bar's denominator is the whole
            # run now, so zeroing the count at a folder boundary would throw
            # away everything the previous folders did — which is exactly what
            # made "how many files has it got to go?" unanswerable.
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
            # run_folders returns failures as data; reaching here means
            # something outside a folder's own run broke. Say so rather than
            # leaving the button greyed out forever (2026-07-09).
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
        """Tk thread: the combined summary, the button back, fresh FOCUS."""
        self._summary.configure(text=format_ingest_summary(report))
        self._set_running(False)
        # Say it is over, rather than leaving the app to infer it from a
        # thread that is still alive for a few more instructions — otherwise
        # pressing the button straight after a run would be refused by the
        # one-job-at-a-time check.
        drop = getattr(self.app, "unregister_ingest", None)
        if drop is not None and self._cancel is not None:
            drop(self._cancel)
        # The list on this same screen is now stale by exactly the data we
        # just ingested — reloading it is the point of having pressed the
        # button.
        self._reload_focus()

    # ---- FOCUS -------------------------------------------------------------
    def reload_now(self) -> None:
        """Synchronous load + apply (test path, and the main-thread apply)."""
        self._apply_focus(*load_focus(self.app.db))
        self._apply_legacy_ft(legacy_ft_count(self.app.db))
        self._apply_unreadable(unreadable_count(self.app.db))

    def _reload_focus(self) -> None:
        def work():
            data = load_focus(self.app.db)
            legacy = legacy_ft_count(self.app.db)
            unreadable = unreadable_count(self.app.db)
            self.safe_after(lambda: self._apply_focus(*data))
            self.safe_after(lambda: self._apply_legacy_ft(legacy))
            self.safe_after(lambda: self._apply_unreadable(unreadable))
        threading.Thread(target=work, daemon=True).start()

    def _apply_legacy_ft(self, count: int) -> None:
        """Show or hide the legacy-verdict line. Tk thread."""
        self._legacy_ft_count = int(count or 0)
        text = legacy_ft_notice(self._legacy_ft_count)
        if not text:
            self._legacy_ft_label.pack_forget()
            self._legacy_ft_label.configure(text="")
            return
        self._legacy_ft_label.configure(text=text)
        self._legacy_ft_label.pack(side="top", fill="x",
                                   pady=(0, self.theme.SPACE_SM))

    def _apply_unreadable(self, count: int) -> None:
        """Show or hide the skipped-because-unreadable line. Tk thread."""
        self._unreadable_count = int(count or 0)
        text = unreadable_notice(self._unreadable_count)
        if not text:
            self._unreadable_label.pack_forget()
            self._unreadable_label.configure(text="")
            return
        self._unreadable_label.configure(text=text)
        self._unreadable_label.pack(side="top", fill="x",
                                    pady=(0, self.theme.SPACE_SM))

    def _apply_focus(self, result, last_processed) -> None:
        # Handed to the zone untouched: one computation owns membership,
        # ranking and wording (see widgets/focus_list_zone.py).
        self._focus.set_result(result, last_processed=last_processed)

    # ---- routing -----------------------------------------------------------
    def _on_focus_click(self, model, focus_metric):
        self.app.set_model_route(model, focus_metric)
        self.app.show_page("model")

    def _open_process(self):
        self.app.show_page("process")

    def _open_settings(self):
        self.app.show_page("settings")
