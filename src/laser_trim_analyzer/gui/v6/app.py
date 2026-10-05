"""V6App root — the top bar over the page container, the status bar under it, and the pages.
Foundations §2.2.

The top bar replaced the sidebar on 2026-10-02 (Graphite redesign,
docs/superpowers/specs/2026-10-02-graphite-redesign-design.md): Overview, Models and Settings
on the bar, and one blue "Process new files" that runs the remembered folders on the Process
page (`process_new_files`). Page KEYS never change -- every deep link navigates by key.

The status bar joined on 2026-10-04 (option B,
docs/superpowers/specs/2026-10-04-option-b-design.md): the database's health and counts, the drift
watch (`drift_watch`), and the run in flight. The run line is fed by a small observer here
(`add_run_listener` / `report_run_progress` / `run_progress`): the Process page reports the progress
its two runs already receive, and nothing about a run itself changed.
"""
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.config import Config
from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.gui.v6 import ctk_patches
from laser_trim_analyzer.gui.v6.page_container import PageContainer
from laser_trim_analyzer.gui.v6.status_bar import StatusBar
from laser_trim_analyzer.gui.v6.status_data import (
    DRIFT_CURRENT, DRIFT_FAILED, DRIFT_UPDATING, RunState)
from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.topbar import TopBar
from laser_trim_analyzer.gui.v6.ui_dispatch import UiDispatcher
from laser_trim_analyzer.utils.threads import guard_tk_font_finalizer

logger = logging.getLogger(__name__)

# How long closing the window waits for an ingest to finish the 20-file batch
# it is on. Long enough for a batch of network-share Excel files, short enough
# that a wedged worker cannot hold the window hostage.
CLOSE_GRACE_SECONDS = 2.0


class V6App(ctk.CTk):
    def __init__(self, config: Config, db=None, auto_train_on_first_run: bool = True):
        # FIRST, before any Tk object can exist: stop the garbage collector
        # running `tkinter.font.Font.__del__` (a `font delete` Tcl call) on an
        # ingest worker. Workers never call Tk. See the function's docstring
        # for the measured cost of leaving it unguarded.
        guard_tk_font_finalizer()
        # The bundled fonts BEFORE the window: Tk on the Mac reads its font list once, when the
        # first window is made, so a font registered after it is never seen -- the titles fell
        # back to the system font (measured 2026-10-04). Windows does not mind either order, and
        # the theme below resolves its families from what Tk can see (font_loader explains).
        from laser_trim_analyzer.gui.v6.font_loader import load_bundled_fonts
        load_bundled_fonts()
        super().__init__()
        # Appearance set HERE (not at import) so importing this module never mutates
        # global CTk state for V5 or test runs.
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        # Same reason, and the same place: patches on the pinned CustomTkinter
        # (see ctk_patches.py). Idempotent, so a second V6App is fine.
        ctk_patches.apply()

        self.config = config
        self.theme = ThemeManager()
        # Share ONE DatabaseManager with the rest of the app. Production: db is None ->
        # get_database() (same singleton Processor uses). Tests inject an isolated one.
        self.db = db if db is not None else get_database()
        self._model_route: Optional[Tuple[str, Optional[str]]] = None
        # Which tab to land on once there, e.g. "findings" from the Findings page's
        # opened-row button. Kept SEPARATE from _model_route (see set_model_route):
        # consuming one must never consume the other, and only one thing may
        # decide what "no tab requested" means -- None.
        self._model_tab_route: Optional[str] = None
        self._auto_train_on_first_run = auto_train_on_first_run
        # In-flight long runs: (cancel Event, worker thread, name). Closing
        # the window used to destroy Tk while a batch was mid-write — the
        # worker kept parsing and saving into a database whose app was already
        # gone. The name is what `active_run_name` reports when a second run
        # is refused.
        self._ingest_runs: List[Tuple[object, object, str]] = []
        # The run observer (the status bar's run line): what each registered run last reported,
        # keyed by id() of its cancel Event -- alive while registered, so never reused meanwhile
        # -- and who wants to hear when a run starts, moves or ends.
        self._run_reports: Dict[int, _RunReport] = {}
        self._run_listeners: List[Callable[[], None]] = []
        # The drift watch, for the status bar: (state, the exception's class when it failed). A
        # start-up catch-up is scheduled below when auto-train is on, and until it has run the
        # flags on screen may be behind -- "updating", not a "current" nothing has checked yet.
        self._drift_watch: Tuple[str, Optional[str]] = (
            DRIFT_UPDATING if auto_train_on_first_run else DRIFT_CURRENT, None)

        # Main-thread UI dispatcher: workers post callbacks here instead of
        # touching Tk from their own threads (see ui_dispatch.py).
        self.ui = UiDispatcher()
        self.ui.attach(self)

        self._setup_window()
        self._build_layout()
        self._build_pages()
        # Home is the landing page (app-shape spec §1, 2026-08-31). Dashboard
        # stays registered and reachable — its retirement is a later call.
        # Showing the Overview is also the status bar's first load (show_page).
        self.show_page("home")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        # Data-gated first-startup auto-train (Spec 3d / D3). Disabled in tests via
        # the flag; the method itself re-checks the flag and the data gate.
        if self._auto_train_on_first_run:
            self.after(500, self._maybe_run_first_startup_train)
            # Catch-up advance: consume any data ingested since the last session
            # (e.g. batches processed in V5 or by scripts) so the Overview isn't stale.
            # Delayed so it doesn't compete with the first page load for the DB.
            self.after(5000, self._start_drift_catchup)

    # ---- navigation ----
    def show_page(self, name: str) -> None:
        if self.page_container.get_page(name) is None:
            return
        arriving = self.page_container.current_page != name
        self.page_container.show(name)
        self.topbar.set_active(name)
        if name == "home" and arriving:
            # The Overview reloads whenever it is shown (its on_show); the status bar's counts
            # are read again with it -- the start-up load included.
            self.status_bar.refresh()

    def process_new_files(self) -> None:
        """The top bar's one blue button: show the Process page and start the remembered-folder
        run there. With no folders set, the page shows itself saying where to add them; with a run
        already in flight, it shows that run."""
        page = self.page_container.get_page("process")
        if page is None:
            return
        self.show_page("process")
        page.start_new_files()

    # ---- routing hint (3b adds consume_model_route, 3c adds consume_model_route_full) ----
    def set_model_route(self, model: str, focus_metric: Optional[str] = None,
                        tab: Optional[str] = None) -> None:
        self._model_route = (model, focus_metric)
        # Always set, even to None: a route call that does not ask for a tab must
        # CLEAR any tab requested by an earlier, unrelated navigation -- otherwise
        # a later plain set_model_route(model) could inherit a stale "findings"
        # from a previous visit to the Findings page (Task 7).
        self._model_tab_route = tab

    def consume_model_route(self) -> Optional[str]:
        """Pop the model name from the routing hint (focus consumed separately in 3c)."""
        if self._model_route is None:
            return None
        model, _focus = self._model_route
        self._model_route = None
        return model

    def consume_model_route_full(self) -> Tuple[Optional[str], Optional[str]]:
        """Pop (model, focus_metric). Either may be None. Used by the Model page."""
        if self._model_route is None:
            return (None, None)
        route = self._model_route
        self._model_route = None
        return route

    def consume_model_tab(self) -> Optional[str]:
        """Pop the tab hint set alongside set_model_route's `tab=` (Task 7). One-shot,
        like the other routing hints; None when no tab was requested."""
        tab = self._model_tab_route
        self._model_tab_route = None
        return tab

    # ---- first-startup auto-train (Spec 3d / decision D3) ----
    def _should_offer_first_startup_train(self) -> bool:
        """True only when there is data to train on AND model_metric_state is empty."""
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR, ModelMetricState
        try:
            with self.db.session() as s:
                has_data = s.query(DBAR.id).first() is not None
                trained = s.query(ModelMetricState.id).first() is not None
            return has_data and not trained
        except Exception:
            return False

    def _maybe_run_first_startup_train(self) -> None:
        if not self._auto_train_on_first_run:
            return

        # The data-gate check is a DB query — run it on a worker so startup
        # never blocks the UI behind the DB lock (e.g. during a batch save).
        def gate_check():
            if not self._should_offer_first_startup_train():
                return
            def open_modal():
                from laser_trim_analyzer.gui.v6.widgets.training_modal import TrainingModal
                preset = getattr(self.config.ml, "drift_sensitivity", "standard")
                TrainingModal(self, theme=self.theme, db=self.db, preset=preset).start()
            self.ui.post(open_modal)

        import threading
        threading.Thread(target=gate_check, daemon=True).start()

    def _start_drift_catchup(self) -> None:
        """The start-up rebuild and catch-up, with the status bar's drift line around it: "updating"
        while it runs, then "current" -- or what failed (Tk thread)."""
        self._set_drift_watch(DRIFT_UPDATING)
        self._advance_drift_catchup(on_done=lambda error: self.ui.post(
            lambda: self._set_drift_watch(DRIFT_FAILED if error else DRIFT_CURRENT, error)))

    def _advance_drift_catchup(self, on_done: Optional[Callable[[Optional[str]], None]] = None
                               ) -> None:
        """Advance all trained drift detectors over data that arrived since the
        last run. Worker thread; a no-op when nothing is new. `on_done(error)` is called on the
        worker when it has finished -- `error` the class name of the first step that failed, or
        None -- every time, even when the drift code itself will not load."""
        def work():
            import logging
            log = logging.getLogger(__name__)

            def report(error: Optional[str]) -> None:
                if on_done is not None:
                    try:
                        on_done(error)
                    except Exception:
                        log.exception("Could not report the drift catch-up's end")

            # Guarded like every step below (review of option B, 2026-10-04). This import used to
            # sit outside both guards: a drift module that would not load ended the thread before
            # it reported, and the status bar said "Drift watch updating…" for good -- a failure
            # reading as work in progress. Now it is named, the way a failed step is.
            try:
                from laser_trim_analyzer.ml.drift_training import (
                    advance_drift_state, ensure_drift_rules)
            except Exception as exc:
                log.exception("Startup drift catch-up: the drift code would not load")
                report(type(exc).__name__)
                return
            error = None
            # The drift rules changed since this state was built (2026-10-02: dirty readings,
            # small lots, old evidence, improvements): retrain once, about ten seconds, before
            # catching up. Its own try: a retrain that fails (a locked database during an
            # ingest) is retried at the next start, and must not cost this start its catch-up.
            try:
                if ensure_drift_rules(self.db, getattr(self.config.ml, "drift_sensitivity",
                                                       "standard")):
                    log.info("Startup: drift state retrained under the current rules")
                    # The page on screen loaded its flags before this finished.
                    self.ui.post(self._reload_visible_page)
            except Exception as exc:
                error = type(exc).__name__
                log.exception("Startup drift retrain under the current rules failed; "
                              "the next start tries again")
            try:
                n = advance_drift_state(self.db)
                if n:
                    log.info("Startup drift catch-up: advanced %d (model, metric) rows", n)
            except Exception as exc:
                error = error or type(exc).__name__
                log.exception("Startup drift catch-up failed")
            report(error)
        import threading
        threading.Thread(target=work, daemon=True).start()

    def drift_watch(self) -> Tuple[str, Optional[str]]:
        """(state, error) for the status bar: DRIFT_UPDATING while the start-up rebuild/catch-up
        runs (and from start-up until it has, when one is scheduled), DRIFT_CURRENT once it has,
        DRIFT_FAILED with the class name of what failed."""
        return self._drift_watch

    def _set_drift_watch(self, state: str, error: Optional[str] = None) -> None:
        """Tk thread. Once the catch-up has finished, either way, the bar reads the database again
        ("after the drift rebuild")."""
        self._drift_watch = (state, error)
        bar = getattr(self, "status_bar", None)
        if bar is None:
            return
        bar.show_drift()
        if state != DRIFT_UPDATING:
            bar.refresh()

    def _reload_visible_page(self) -> None:
        """Reload the page on screen (Tk thread only): its data changed underneath it. Through the
        page's own on_show(), which loads on a worker -- never reload_now(), which loads on this
        thread and would freeze the window for the page's whole query."""
        name = self.page_container.current_page          # a property, not a method
        page = self.page_container.get_page(name) if name else None
        if page is not None:
            page.on_show()

    # ---- setup ----
    def _setup_window(self) -> None:
        self.title("Laser Trim Analyzer")
        self.geometry(f"{self.config.gui.window_width}x{self.config.gui.window_height}")
        self.minsize(960, 640)
        self.configure(fg_color=self.theme.BG)
        self.grid_rowconfigure(0, weight=0)          # the top bar, its own height
        self.grid_rowconfigure(1, weight=1)          # the pages take the rest
        self.grid_rowconfigure(2, weight=0)          # the status bar, its own height
        self.grid_columnconfigure(0, weight=1)

    def _build_layout(self) -> None:
        self.topbar = TopBar(self, on_select=self.show_page, on_process=self.process_new_files,
                             theme=self.theme)
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.page_container = PageContainer(self, theme=self.theme)
        self.page_container.grid(row=1, column=0, sticky="nsew")
        self.status_bar = StatusBar(self, app=self, theme=self.theme)
        self.status_bar.grid(row=2, column=0, sticky="ew")

    def _build_pages(self) -> None:
        # All pages are real. The Overview (key "home") is the landing; Models and Settings are
        # on the top bar; Process, Findings and Dashboard are one click away (TopBar.OFF_BAR).
        # Triage was retired on 2026-10-02 (Graphite redesign): the Overview's cards and its
        # "Everything else" list are what it showed.
        from laser_trim_analyzer.gui.v6.pages.home_page import HomePage
        from laser_trim_analyzer.gui.v6.pages.dashboard_page import DashboardPage
        from laser_trim_analyzer.gui.v6.pages.model_page import ModelPage
        from laser_trim_analyzer.gui.v6.pages.findings_page import FindingsPage
        from laser_trim_analyzer.gui.v6.pages.settings_page import SettingsPage
        from laser_trim_analyzer.gui.v6.pages.process_page import ProcessPage
        self._add_page(
            "home",
            HomePage(self.page_container, theme=self.theme, app=self, page_title="Overview"),
        )
        self._add_page(
            "dashboard",
            # The key stays "dashboard"; the page reads "Company trends" everywhere (option B,
            # finish list 3) -- the Overview's link says so.
            DashboardPage(self.page_container, theme=self.theme, app=self,
                          page_title="Company trends"),
        )
        self._add_page(
            "model",
            # Route key stays "model" (FOCUS rows and set_model_route navigate to
            # it); only what the user reads says "Models", matching the top bar.
            ModelPage(self.page_container, theme=self.theme, app=self,
                      page_title="Models"),
        )
        self._add_page(
            "findings",
            FindingsPage(self.page_container, theme=self.theme, app=self, page_title="Findings"),
        )
        self._add_page(
            "settings",
            SettingsPage(self.page_container, theme=self.theme, app=self, page_title="Settings"),
        )
        self._add_page(
            "process",
            ProcessPage(self.page_container, theme=self.theme, app=self, page_title="Process"),
        )

    def _add_page(self, key: str, page) -> None:
        """Register `page` under `key`. A page the top bar has no item for -- Process, Findings,
        Company trends -- names itself at the left of its own row: with nothing lit on the bar,
        nothing on screen said where you were (review of option B, 2026-10-04). The bar names the
        other three, which never say it twice (finish item 2)."""
        self.page_container.add_page(key, page)
        if not self.topbar.lights(key):
            page.show_name()

    # ---- in-flight long runs ----
    def register_ingest(self, cancel, thread, name: str = "An ingest") -> None:
        """Remember a running long job so closing the window can stop it first.

        `name` is what the OTHER front end says when it refuses to start
        ("A re-grade is running — stop it first"), so it reads as a sentence
        with " is running" after it: "An ingest", "A re-grade".

        Called from the page's Tk thread right after the worker starts. Dead
        entries are swept here rather than by the worker, which must not touch
        app state from off-thread.
        """
        self._ingest_runs = [r for r in self._ingest_runs if _alive(r[1])]
        self._ingest_runs.append((cancel, thread, name))
        self._run_reports = {id(r[0]): self._run_reports.get(id(r[0])) or _RunReport()
                             for r in self._ingest_runs}
        self._tell_run_listeners()

    def unregister_ingest(self, cancel) -> None:
        """Forget a run that has finished. Tk thread only.

        `is_alive()` alone is not enough to decide that a run is over. A page
        learns its run has finished from the worker POSTING its result, and at
        that moment the worker thread is still alive for a few more
        instructions — so a second press landing in that window would be
        refused by `active_run_name` for a run that had already delivered its
        summary. The page says when it is done; liveness is the backstop for
        the case where nobody said.
        """
        self._ingest_runs = [r for r in self._ingest_runs
                             if r[0] is not cancel and _alive(r[1])]
        self._run_reports = {id(r[0]): self._run_reports.get(id(r[0])) or _RunReport()
                             for r in self._ingest_runs}
        self._tell_run_listeners()

    # ---- the run observer: each run's progress, app-wide (the status bar's run line) ----
    def add_run_listener(self, fn: Callable[[], None]) -> None:
        """Call `fn()` on the Tk thread whenever a run starts, reports progress, or ends."""
        self._run_listeners.append(fn)

    def report_run_progress(self, run, label: str, done: int, total: int) -> None:
        """A run's live progress, from the page that receives it (Tk thread): the numbers its own
        progress line was just painted with, so the rest of the app can show them too. `run` is
        the cancel Event the run registered with; a paint that lands after its run unregistered
        (the ticker's last tick) -- or from a page with no run started -- is dropped. `total` 0 =
        not known yet. Never raises: it is called from inside a run's own paint."""
        try:
            report = self._run_reports.get(id(run)) if run is not None else None
            if report is None:
                return
            now = (label, int(done or 0), int(total or 0))
            if (report.label, report.done, report.total) == now:
                return
            report.label, report.done, report.total = now
            self._tell_run_listeners()
        except Exception:
            logger.exception("Could not take a run's progress")

    def run_progress(self) -> Optional[RunState]:
        """The run in flight -- the one `active_run_name` names -- with what it last reported, or
        None when nothing is running. A run that reports no progress of its own (a re-grade, a
        findings refresh) comes back with no label."""
        for cancel, thread, name in self._ingest_runs:
            if _alive(thread):
                report = self._run_reports.get(id(cancel)) or _RunReport()
                return RunState(name=name, label=report.label, done=report.done,
                                total=report.total, started=report.started)
        return None

    def _tell_run_listeners(self) -> None:
        for fn in list(self._run_listeners):
            try:
                fn()
            except Exception:
                logger.exception("A run listener failed")

    def active_run_name(self) -> Optional[str]:
        """The name of a long job in flight, or None.

        ONE place that answers "is something already running?", so HOME and
        Settings cannot disagree about it — and so adding a third long job
        later means registering it, not editing two `if`s.

        Why this exists (2026-09-14). The re-grade was started at 14:19 and
        "Process everything new" was pressed at 14:45 with it still going.
        The two fought over the same database write lock and the same SMB
        share: the processed-file index load went from 2.6 s to 26, 32, 88 and
        101 s per folder, the final-test verify pass went from 140 s to 542,
        and the batch was cancelled after 0 of 1,371 files. Neither job was
        broken; they were simply sharing one lock and one network link.
        """
        for _cancel, thread, name in self._ingest_runs:
            if _alive(thread):
                return name
        return None

    def stop_ingests(self, timeout: float = CLOSE_GRACE_SECONDS) -> None:
        """Ask every in-flight run to stop and give it `timeout` to land.

        Cooperative, never forced: the worker finishes the batch it already
        handed to the thread pool and persists it. If it needs longer than the
        grace period we stop waiting — the window closes, the daemon thread
        dies with the process, and the batch boundary means the database is
        consistent either way.
        """
        runs = [r for r in self._ingest_runs if _alive(r[1])]
        self._ingest_runs = []
        if not runs:
            return
        logger.info("Closing with %d run(s) in flight (%s) — asking them to stop",
                    len(runs), ", ".join(sorted({r[2] for r in runs})))
        for cancel, _thread, _name in runs:
            try:
                cancel.set()
            except Exception:
                logger.exception("Could not signal a run to stop")
        deadline = time.monotonic() + timeout
        for _cancel, thread, _name in runs:
            try:
                thread.join(max(0.0, deadline - time.monotonic()))
            except Exception:
                logger.exception("Could not wait for a run's thread")
        # Then the ingest's worker PROCESSES (ingest-speed ruling 20), terminated at once -- a
        # worker holds no database handle, so nothing can be torn. Without this a worker stuck on
        # one file keeps the app from ever exiting: concurrent.futures joins every worker process
        # at interpreter exit. At once, not after a grace (final review, m-1): a closed pool hands
        # nothing over, so what a worker finished in the grace would be thrown away, and the grace
        # would hold this -- the Tk -- thread past Windows' 5 s "Not Responding".
        try:
            from laser_trim_analyzer.core import ingest_worker
            ingest_worker.close_worker_pools(grace=0.0)
        except Exception:
            logger.exception("Could not close the ingest's worker processes")

    def _on_closing(self) -> None:
        self.stop_ingests()
        self.destroy()

    def run(self) -> None:
        self.mainloop()



def _alive(thread) -> bool:
    """True only for a thread object that says it is still running."""
    try:
        return bool(thread.is_alive())
    except Exception:
        return False


class _RunReport:
    """What one registered run last reported (`report_run_progress`), and when it registered."""
    __slots__ = ("label", "done", "total", "started")

    def __init__(self) -> None:
        self.label: Optional[str] = None
        self.done = 0
        self.total = 0
        self.started = time.monotonic()
