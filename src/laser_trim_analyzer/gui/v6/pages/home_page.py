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

import customtkinter as ctk

logger = logging.getLogger(__name__)

from laser_trim_analyzer.core import ingest_run
from laser_trim_analyzer.core.ingest_run import (
    ProgressCoalescer, ProgressTicker, format_ingest_summary)
from laser_trim_analyzer.gui.v6.focus_data import load_focus
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.focus_list_zone import FocusListZone
from laser_trim_analyzer.gui.v6.widgets.process_progress_section import (
    ProcessProgressSection)


class HomePage(PageBase):
    page_title = "Home"

    def __init__(self, master, *, theme, app, page_title="Home"):
        self._running = False
        super().__init__(master, theme=theme, app=app, page_title=page_title)
        self.refresh_folders()

    # ---- construction ------------------------------------------------------
    def build_content(self, parent):
        t = self.theme
        self._zone_header(parent, "BRING IN WHAT'S NEW",
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
            inner, text="", anchor="w", justify="left", wraplength=1100,
            font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY)
        self._folders_label.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))

        self._progress = ProcessProgressSection(inner, theme=t)
        self._progress.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))

        # The one line the whole run reduces to. Stays on screen afterwards —
        # "did that do anything?" is a question the app should not need to be
        # asked twice.
        self._summary = ctk.CTkLabel(inner, text="", anchor="w", justify="left",
                                     wraplength=1100, font=t.font(t.SIZE_BODY),
                                     text_color=t.TEXT_PRIMARY)
        self._summary.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))

        self._zone_header(parent, "WHAT THE APP IS TELLING YOU",
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
        self._set_running(True)
        self._progress.reset()
        self._summary.configure(text="")
        # "New" is the whole promise of the button, so the run is always
        # incremental; the Process page keeps the checkbox for a re-run.
        threading.Thread(target=self._run, args=(folders, True),
                         daemon=True).start()

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._run_button.configure(state="disabled" if running else "normal",
                                   text=("Processing…" if running
                                         else "Process everything new"))

    def _paint(self, coalescer: ProgressCoalescer, total: dict) -> None:
        """Paint ONE coalesced snapshot (Tk thread). See core/ingest_run.py."""
        snap = coalescer.drain()
        if snap["scan_msg"] and not snap["moved"]:
            self._progress.set_idle(snap["scan_msg"])
            return
        if snap["moved"] or snap["done"]:
            self._progress.set_progress(snap["done"], total["n"], snap["file"])
            if snap["counts"] or snap["reasons"]:
                self._progress.add_counts(snap["counts"], snap["reasons"])

    def _run(self, folders, incremental: bool) -> None:
        """Worker: drive the shared multi-folder run. Never touches Tk."""
        coalescer = ProgressCoalescer()
        total = {"n": 0}
        ticker = ProgressTicker(
            lambda: self.safe_after(lambda: self._paint(coalescer, total))).start()

        def folder_start(i, n, folder):
            # Each folder has its own denominator, so the bar restarts with it;
            # the bucket counters keep accumulating across the whole run.
            coalescer.reset()
            total["n"] = 0
            self.safe_after(lambda: self._progress.set_idle(
                f"Folder {i} of {n}: {folder}"))

        try:
            report = ingest_run.run_folders(
                folders, db=self.app.db, config=self.app.config,
                incremental=incremental, progress=coalescer,
                on_phase=lambda msg: self.safe_after(
                    lambda m=msg: self._progress.set_idle(m)),
                on_total=lambda n: total.__setitem__("n", n),
                on_folder_start=folder_start)
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
        self.safe_after(lambda: self._paint(coalescer, total))
        self.safe_after(lambda r=report: self._on_run_done(r))

    def _on_run_done(self, report) -> None:
        """Tk thread: the combined summary, the button back, fresh FOCUS."""
        self._summary.configure(text=format_ingest_summary(report))
        self._set_running(False)
        # The list on this same screen is now stale by exactly the data we
        # just ingested — reloading it is the point of having pressed the
        # button.
        self._reload_focus()

    # ---- FOCUS -------------------------------------------------------------
    def reload_now(self) -> None:
        """Synchronous load + apply (test path, and the main-thread apply)."""
        self._apply_focus(*load_focus(self.app.db))

    def _reload_focus(self) -> None:
        def work():
            data = load_focus(self.app.db)
            self.safe_after(lambda: self._apply_focus(*data))
        threading.Thread(target=work, daemon=True).start()

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
