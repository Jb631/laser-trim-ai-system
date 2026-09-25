"""Spec 3f — HomePage: the landing view. Ingest at the top, then what needs attention.

Spec: docs/superpowers/specs/2026-08-29-app-shape-investigate-design.md §1;
relayout per docs/superpowers/specs/2026-09-24-facelift-step2-pages-design.md §2 (Task 4).
The app's first screen answers the questions James actually opens it with, in the order he
asks them: "pull in whatever is new", then "what needs attention" -- both "what's worth
changing" and "what's drifting" -- newest data first, verdicts after.

  * ONE button runs the remembered folder list through the existing batch
    pipeline (`core/ingest_run.run_folders`) — the same worker the Process
    page drives, never a second copy of it. The per-folder phase line goes to
    the progress section as it goes, and one combined summary lands at the
    end: "3 folders · 214 new files · 2 min 40 s".
  * The Process page's picker stays one click away ("process a specific
    folder…") for the one-off folder that isn't on the list.
  * "Worth changing" — the top of the Findings page's "yield" group, across every model:
    the same `FindingsView` and the same `findings/presentation.arrange()` the Findings
    page itself draws from, read via one `get_process_findings()` call so this section and
    the Findings page can never disagree.
  * "Drifting now" — the FOCUS list, the same widget and the same loader Triage uses
    (`widgets/focus_list_zone.py`, `focus_data.load_focus`), because the two screens
    disagreeing about what is drifting would be worse than either of them being wrong.
  * The page caption ties all three loads together in one line: "Last processed {date} ·
    {N} worth changing · {M} drifting now" (`_update_caption`) — date and M come from
    `load_focus`'s own return, N from the findings read; nothing here is a second query.

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
from laser_trim_analyzer.findings import presentation as P
from laser_trim_analyzer.gui.v6.focus_data import focus_failed, load_focus
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets import blocks
from laser_trim_analyzer.gui.v6.widgets.findings_view import FindingsView
from laser_trim_analyzer.gui.v6.widgets.focus_list_zone import FocusListZone
from laser_trim_analyzer.gui.v6.widgets.process_progress_section import (
    ProcessProgressSection)

# "Worth changing" (design doc 2026-09-24-facelift-step2-pages-design.md §2, ruling 2 item 3):
# the top of the Findings page's FIRST group, across every model -- P.GROUPS[0] is "yield"
# ("Change a setting to raise yield"). Looked up by key, not index, so a reorder of GROUPS in
# presentation.py can never silently point this at the wrong group.
_WORTH_CHANGING_GROUP = "yield"
_YIELD_SPEC = next(s for s in P.GROUPS if s.key == _WORTH_CHANGING_GROUP)


def _yield_findings_count(findings) -> int:
    """Rows the "Worth changing" section shows -- the SAME arrange() FindingsView itself calls
    (groups=(_WORTH_CHANGING_GROUP,)), so the caption's N and the rows under it can never
    disagree (same technique as model_page.py's _worth_changing_count)."""
    groups = P.arrange(findings or [], include_empty=False)
    return sum(len(g.rows) for g in groups if g.spec.key == _WORTH_CHANGING_GROUP)


class HomePage(PageBase):
    page_title = "Home"

    def __init__(self, master, *, theme, app, page_title="Home"):
        self._running = False
        self._cancel = None            # threading.Event while a run is in flight
        self._last_processed = None    # datetime | None -- the caption's date (from load_focus)
        self._focus_count = 0          # M -- len(FocusResult.focus) (from load_focus); None
                                        # while unknown (the FOCUS computation failed)
        self._worth_count = None       # N -- yield-group rows; None while unknown (not yet
                                        # loaded, or the findings read failed)
        self._worth_view = None        # the "Worth changing" FindingsView, when there is one
        # Reload generations, one per independent load (the Model page's _reload_gen pattern,
        # facelift F4): a load's apply is dropped unless it is still the newest of its kind.
        # Workers finish in any order, and the re-review watched the app's own start-up load land
        # after a newer one and wipe it. Bumped and read on the Tk thread only.
        self._focus_gen = 0
        self._findings_gen = 0
        super().__init__(master, theme=theme, app=app, page_title=page_title)
        self.refresh_folders()

    # ---- construction ------------------------------------------------------
    def build_content(self, parent):
        t = self.theme
        # SCROLLABLE body (facelift step 2, Task 4): three sections now compete for the same
        # fixed height "Bring in what's new" and the FOCUS list used to split between them.
        # render_pages.py --audit caught it immediately at 1280x720 -- "Worth changing"'s own
        # real content (a populated yield group plus its buttons) left the FOCUS list, and even
        # "Drifting now"'s own heading, squeezed to a sliver -- the same "page too tall for the
        # window" failure mode model_page.py already solved (James, 2026-07-13: "cant scroll
        # down on some of the pages"), now here too. `self._body` replaces `parent` as every
        # section's master; FocusListZone keeps fill="both", expand=True inside it, same as
        # ModelPage's own scrollable-tab content, so it still claims any leftover room instead
        # of being capped to its bare minimum.
        self._body = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self._body.pack(side="top", fill="both", expand=True)
        parent = self._body
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
            inner, text="", anchor="w", justify="left",
            font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY)
        self._folders_label.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
        # Built once, never destroyed/rebuilt for the page's whole lifetime (only
        # .configure(text=...) is called on it later) -- safe to bind wrap_to_width here,
        # once (global-constraints.md: no fixed pixel wraplength on page-width text; the old
        # fixed 950 was wider than the room this line gets at 1280x720 until the 2026-09-24
        # fix that introduced it, which is exactly the bug this helper exists to prevent from
        # coming back at some OTHER width).
        blocks.wrap_to_width(self._folders_label, inner)

        self._progress = ProcessProgressSection(inner, theme=t)
        self._progress.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))

        # The one line the whole run reduces to. Stays on screen afterwards —
        # "did that do anything?" is a question the app should not need to be
        # asked twice.
        self._summary = ctk.CTkLabel(inner, text="", anchor="w", justify="left",
                                     font=t.font(t.SIZE_BODY), text_color=t.TEXT_PRIMARY)
        self._summary.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
        blocks.wrap_to_width(self._summary, inner)

        # One line, only when there is something to say: final-test records
        # graded before the ignore-window fix (2026-09-13) carry a verdict
        # that may include sweep points the station never graded. It is not an
        # error state and it does not block anything, so it is a caption, not
        # a banner — but it stays up until Settings clears it, because every
        # final-test number on the screens below is computed from those rows.
        # blocks.banner(tone="quiet"): same quiet-notice treatment as the unreadable-files
        # line below it, not the old ad-hoc TIER_WARNING label (facelift step 2, Task 4).
        self._legacy_ft_label = blocks.banner(parent, t, "", tone="quiet", wrap_to=parent)
        self._legacy_ft_count = 0

        # The same shape, for the files the button is NOT processing: ones
        # that failed to read on an earlier run and are skipped while they are
        # unchanged on disk (2026-09-17). Silent at zero; while it says
        # anything it also says the way back.
        self._unreadable_label = blocks.banner(parent, t, "", tone="quiet", wrap_to=parent)
        self._unreadable_count = 0

        # ---- "Worth changing" (design doc §2, ruling 2 item 3): the top of the Findings
        # page's "yield" group, across every model -- the same FindingsView and the same
        # arrange() the Findings page itself draws, so this can never disagree with it.
        # Findings-loading FAILURE below is a check-tone banner, never "nothing worth
        # changing" (CLAUDE.md: a failure must never look like a result).
        self._worth_header = self._zone_header(
            parent, "Worth changing",
            "the top of the Findings page's yield group — a setting that did "
            "better on the same test, across every model")
        # Packed only when there is something to say: the findings load failed, or some models'
        # last refresh had an analyzer fail (or which ones could not be read) -- never both at
        # once (a failed load reads nothing else), so one banner holds whichever it is.
        self._worth_banner = blocks.banner(parent, t, "", wrap_to=parent)
        self._worth_section = ctk.CTkFrame(parent, fg_color="transparent")
        self._worth_section.pack(side="top", fill="x", pady=(0, t.SPACE_LG))

        # The two notices above are packed later, only when they have something to say -- and
        # always inside "Bring in what's new", just above the "Worth changing" header
        # (_show_notice): they are about ingest, not findings (final review, 2026-09-24), and
        # never after the focus list below, which expands to fill the page -- on a 720-px-tall
        # window a notice packed after it got no height at all and vanished (the audit found
        # "70 files are being skipped..." squeezed out, 2026-09-24). Renamed from "What the app
        # is telling you" (Task 4): with "Worth changing" now also on this page, two zones
        # sharing that generic title would say nothing -- this one names what it specifically is.
        self._focus_header = self._zone_header(parent, "Drifting now",
                                               "biggest first — one verdict per lot, "
                                               "self-clearing")
        # Packed (just under "Drifting now") only when the FOCUS computation failed: a crash is
        # named, never drawn as "0 drifting now" (final review, 2026-09-24).
        self._focus_banner = blocks.banner(parent, t, "", wrap_to=parent)
        # show_heading=False, like Triage: the zone header above already says "Drifting now";
        # the list's own "FOCUS — drifting now, biggest first (N)" under it said it twice.
        self._focus = FocusListZone(parent, theme=t,
                                    on_row_click=self._on_focus_click, show_heading=False)
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
        self._reload_findings()

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
        self._reload_findings()

    # ---- FOCUS -------------------------------------------------------------
    def reload_now(self) -> None:
        """Synchronous load + apply (test path, and the main-thread apply). The newest load of
        both kinds: anything still in flight is dropped when it lands."""
        self._focus_gen += 1
        self._findings_gen += 1
        self._apply_focus_load(load_focus(self.app.db), legacy_ft_count(self.app.db),
                               unreadable_count(self.app.db))
        self._apply_findings(self._query_findings())

    def _reload_focus(self) -> None:
        self._focus_gen += 1
        gen = self._focus_gen

        def work():
            data = load_focus(self.app.db)
            legacy = legacy_ft_count(self.app.db)
            unreadable = unreadable_count(self.app.db)

            def apply():
                if gen != self._focus_gen:
                    return              # a newer load superseded this one
                self._apply_focus_load(data, legacy, unreadable)
            self.safe_after(apply)
        threading.Thread(target=work, daemon=True).start()

    def _apply_focus_load(self, data, legacy, unreadable) -> None:
        """FOCUS and the two ingest notices, from ONE load -- each drawn under its own guard, the
        Model page's `_try`: a render error in one is logged and the other two still update (F4
        review, Minor 4: as one closure, a crash in the FOCUS list stopped both notices, and
        safe_after swallowed it). Tk thread; the caller has already checked the load is current."""
        for what, update in (("drifting-now list", lambda: self._apply_focus(*data)),
                             ("final-test notice", lambda: self._apply_legacy_ft(legacy)),
                             ("unreadable-files notice", lambda: self._apply_unreadable(unreadable))):
            try:
                update()
            except Exception:
                logger.exception("Home: the %s could not be drawn", what)

    def _apply_legacy_ft(self, count: int) -> None:
        """Show or hide the legacy-verdict line. Tk thread."""
        self._legacy_ft_count = int(count or 0)
        text = legacy_ft_notice(self._legacy_ft_count)
        if not text:
            self._legacy_ft_label.pack_forget()
            self._legacy_ft_label.configure(text="")
            return
        self._legacy_ft_label.configure(text=text)
        self._show_notice(self._legacy_ft_label)

    def _apply_unreadable(self, count: int) -> None:
        """Show or hide the skipped-because-unreadable line. Tk thread."""
        self._unreadable_count = int(count or 0)
        text = unreadable_notice(self._unreadable_count)
        if not text:
            self._unreadable_label.pack_forget()
            self._unreadable_label.configure(text="")
            return
        self._unreadable_label.configure(text=text)
        self._show_notice(self._unreadable_label)

    def _show_notice(self, label) -> None:
        """Pack an ingest notice at the foot of "Bring in what's new" -- just above the "Worth
        changing" header, where the expanding focus list can never take its room (see
        build_content)."""
        label.pack(side="top", fill="x", pady=(0, self.theme.SPACE_SM), before=self._worth_header)

    def _apply_focus(self, result, last_processed) -> None:
        # Handed to the zone untouched: one computation owns membership,
        # ranking and wording (see widgets/focus_list_zone.py).
        self._focus.set_result(result, last_processed=last_processed)
        # Same two values the caption reads (design doc §2 ruling 2 item 1) -- load_focus
        # already computed both, so this is not a second query, just a second consumer.
        self._last_processed = last_processed
        failed = focus_failed(result)
        self._focus_count = None if failed else len(result.focus)
        if failed:
            self._focus_banner.configure(
                text=f"What is drifting could not be worked out ({failed}). This is an error, "
                     f"not an all-clear — the log has the details.")
            self._focus_banner.pack(side="top", fill="x", pady=(0, self.theme.SPACE_SM),
                                    before=self._focus)
        else:
            self._focus_banner.pack_forget()
        self._update_caption()

    # ---- "Worth changing" ---------------------------------------------------
    def _query_findings(self) -> dict:
        """The same two reads the Findings page makes (findings_page.py::_query), guarded
        separately, worker-safe (no Tk):

        * every model's cached findings (get_process_findings) -- a failed read is `failed`,
          named in a check-tone banner, never drawn as "nothing worth changing" (CLAUDE.md: a
          failure must never look like a result);
        * which models' last refresh had an analyzer fail (get_process_errors) -- their
          findings may be MISSING from the list, so the banner says so (final review,
          2026-09-24: Home used to say nothing). If this read alone fails, the list is still
          true and still shown, and `errors_failed` says the other thing is unknown."""
        out = {"rows": [], "failed": None, "errors": {}, "errors_failed": None}
        try:
            out["rows"] = self.app.db.get_process_findings()
        except Exception as exc:
            logger.exception("Home: findings load failed")
            out["failed"] = f"{type(exc).__name__}: {exc}"
            return out
        try:
            out["errors"] = self.app.db.get_process_errors()
        except Exception as exc:
            logger.exception("Home: could not read which models failed")
            out["errors_failed"] = f"{type(exc).__name__}: {exc}"
        return out

    def _reload_findings(self) -> None:
        self._findings_gen += 1
        gen = self._findings_gen

        def work():
            data = self._query_findings()

            def apply():
                if gen == self._findings_gen:       # else a newer load superseded this one
                    self._apply_findings(data)
            self.safe_after(apply)
        threading.Thread(target=work, daemon=True).start()

    def _apply_findings(self, data: dict) -> None:
        """Rebuilt whole on every apply -- same as FindingsView._render() and the Model
        page's own "Worth changing on this model" section -- so the section's rows and the
        caption's N can never disagree. Tk thread."""
        t = self.theme
        for child in self._worth_section.winfo_children():
            child.destroy()
        self._worth_view = None
        if data.get("failed"):
            self._worth_count = None       # unknown, not zero -- never shown as a result
            self._worth_banner.configure(
                text=f"Findings could not be loaded ({data['failed']}). This is an error, "
                     f"not an empty list — the log has the details.")
            self._worth_banner.pack(side="top", fill="x", pady=(0, t.SPACE_SM),
                                    before=self._worth_section)
            self._update_caption()
            return
        notice = (P.errors_unknown_notice(data["errors_failed"]) if data.get("errors_failed")
                  else P.errors_notice(data["errors"]) if data.get("errors") else "")
        if notice:
            self._worth_banner.configure(text=notice)
            self._worth_banner.pack(side="top", fill="x", pady=(0, t.SPACE_SM),
                                    before=self._worth_section)
        else:
            self._worth_banner.pack_forget()
        rows = data.get("rows") or []
        # No cached findings at all is "never worked out" as much as "nothing found" -- this page
        # cannot tell them apart, so the caption states no N rather than a zero it cannot vouch
        # for (the Findings page drops its caption the same way); the section's own line below,
        # "Nothing here yet", is true of both.
        self._worth_count = _yield_findings_count(rows) if rows else None
        if not self._worth_count:
            # include_empty=False (below) means FindingsView draws nothing at all for an
            # empty group -- say it here instead, in the group's own words, so the section
            # is a quiet line, never a blank gap.
            ctk.CTkLabel(self._worth_section, text=_YIELD_SPEC.empty, font=t.font(t.SIZE_BODY),
                        text_color=t.TEXT_SECONDARY, anchor="w", justify="left", wraplength=1000
                        ).pack(fill="x", padx=t.SPACE_SM)
        else:
            # open_as="link": Home already carries its own primary_button ("Process everything
            # new"); a second teal-filled "Open <model>" the moment a row is expanded would
            # break "at most ONE teal-filled button per screen" (global-constraints.md).
            view = FindingsView(self._worth_section, t, on_open=self._open_finding,
                                include_empty=False, rows_per_group=3,
                                groups=(_WORTH_CHANGING_GROUP,), open_as="link")
            view.pack(fill="x")
            view.set_findings(rows)
            self._worth_view = view
        blocks.link_button(self._worth_section, t, "Open Findings", self._open_findings
                           ).pack(anchor="w", padx=t.SPACE_XS, pady=(t.SPACE_XS, 0))
        self._update_caption()

    def _update_caption(self) -> None:
        """"Last processed {date} · {N} worth changing · {M} drifting now" (design doc §2,
        ruling 2 item 1). No last-processed date at all -- nothing has ever been ingested --
        means no caption, the same posture the Findings page takes on its own empty/failed
        states: zeros are not a real reading of an app that has never run. N drops out of the
        sentence (never shown as a misleading "0") while the findings read failed or nothing
        is cached at all (_apply_findings); M drops out while the FOCUS computation failed
        (load_focus never raises, but marks the failure -- focus_data.focus_failed)."""
        if self._last_processed is None:
            self.set_caption("")
            return
        dt = self._last_processed
        parts = [f"Last processed {dt.day} {dt:%b}"]      # NOT %-d: it raises on Windows
        if self._worth_count is not None:
            parts.append(f"{self._worth_count:,} worth changing")
        if self._focus_count is not None:
            parts.append(f"{self._focus_count:,} drifting now")
        self.set_caption(" · ".join(parts))

    # ---- routing -----------------------------------------------------------
    def _on_focus_click(self, model, focus_metric):
        self.app.set_model_route(model, focus_metric)
        self.app.show_page("model")

    def _open_finding(self, model: str) -> None:
        """Same route the Findings page itself uses (findings_page.py::_open) -- straight
        onto the model's Findings tab, never the FOCUS list's (model, metric) form."""
        self.app.set_model_route(model, tab="findings")
        self.app.show_page("model")

    def _open_findings(self) -> None:
        self.app.show_page("findings")

    def _open_process(self):
        self.app.show_page("process")

    def _open_settings(self):
        self.app.show_page("settings")
