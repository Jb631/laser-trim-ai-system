"""V6App root — sidebar + page container + the four real pages. Foundations §2.2."""
import logging
import time
from typing import List, Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.config import Config
from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.gui.v6 import ctk_patches
from laser_trim_analyzer.gui.v6.page_container import PageContainer
from laser_trim_analyzer.gui.v6.sidebar import Sidebar
from laser_trim_analyzer.gui.v6.theme import ThemeManager
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
        self._auto_train_on_first_run = auto_train_on_first_run
        # In-flight long runs: (cancel Event, worker thread, name). Closing
        # the window used to destroy Tk while a batch was mid-write — the
        # worker kept parsing and saving into a database whose app was already
        # gone. The name is what `active_run_name` reports when a second run
        # is refused.
        self._ingest_runs: List[Tuple[object, object, str]] = []

        # Main-thread UI dispatcher: workers post callbacks here instead of
        # touching Tk from their own threads (see ui_dispatch.py).
        self.ui = UiDispatcher()
        self.ui.attach(self)

        self._setup_window()
        self._build_layout()
        self._build_pages()
        # Home is the landing page (app-shape spec §1, 2026-08-31). Dashboard
        # stays registered and reachable — its retirement is a later call.
        self.show_page("home")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        # Data-gated first-startup auto-train (Spec 3d / D3). Disabled in tests via
        # the flag; the method itself re-checks the flag and the data gate.
        if self._auto_train_on_first_run:
            self.after(500, self._maybe_run_first_startup_train)
            # Catch-up advance: consume any data ingested since the last session
            # (e.g. batches processed in V5 or by scripts) so Triage isn't stale.
            # Delayed so it doesn't compete with the first page load for the DB.
            self.after(5000, self._advance_drift_catchup)

    # ---- navigation ----
    def show_page(self, name: str) -> None:
        if self.page_container.get_page(name) is None:
            return
        self.page_container.show(name)
        self.sidebar.set_active(name)

    # ---- routing hint (3b adds consume_model_route, 3c adds consume_model_route_full) ----
    def set_model_route(self, model: str, focus_metric: Optional[str] = None) -> None:
        self._model_route = (model, focus_metric)

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

    def _advance_drift_catchup(self) -> None:
        """Advance all trained drift detectors over data that arrived since the
        last run. Worker thread; a no-op when nothing is new."""
        def work():
            try:
                from laser_trim_analyzer.ml.drift_training import advance_drift_state
                import logging
                n = advance_drift_state(self.db)
                if n:
                    logging.getLogger(__name__).info(
                        "Startup drift catch-up: advanced %d (model, metric) rows", n)
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Startup drift catch-up failed")
        import threading
        threading.Thread(target=work, daemon=True).start()

    # ---- setup ----
    def _setup_window(self) -> None:
        self.title("Laser Trim Analyzer")
        self.geometry(f"{self.config.gui.window_width}x{self.config.gui.window_height}")
        self.minsize(960, 640)
        self.configure(fg_color=self.theme.BG)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def _build_layout(self) -> None:
        self.sidebar = Sidebar(self, on_select=self.show_page, theme=self.theme)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.page_container = PageContainer(self, theme=self.theme)
        self.page_container.grid(row=0, column=1, sticky="nsew")

    def _build_pages(self) -> None:
        # All pages are real. Home is the landing; the rest follow in the sidebar.
        from laser_trim_analyzer.gui.v6.pages.home_page import HomePage
        from laser_trim_analyzer.gui.v6.pages.dashboard_page import DashboardPage
        from laser_trim_analyzer.gui.v6.pages.triage_page import TriagePage
        from laser_trim_analyzer.gui.v6.pages.model_page import ModelPage
        from laser_trim_analyzer.gui.v6.pages.findings_page import FindingsPage
        from laser_trim_analyzer.gui.v6.pages.settings_page import SettingsPage
        from laser_trim_analyzer.gui.v6.pages.process_page import ProcessPage
        self.page_container.add_page(
            "home",
            HomePage(self.page_container, theme=self.theme, app=self, page_title="Home"),
        )
        self.page_container.add_page(
            "dashboard",
            DashboardPage(self.page_container, theme=self.theme, app=self, page_title="Dashboard"),
        )
        self.page_container.add_page(
            "triage",
            TriagePage(self.page_container, theme=self.theme, app=self, page_title="Triage"),
        )
        self.page_container.add_page(
            "model",
            # Route key stays "model" (FOCUS rows and set_model_route navigate to
            # it); only what the user reads says "Investigate", matching the nav.
            ModelPage(self.page_container, theme=self.theme, app=self,
                      page_title="Investigate"),
        )
        self.page_container.add_page(
            "findings",
            FindingsPage(self.page_container, theme=self.theme, app=self, page_title="Findings"),
        )
        self.page_container.add_page(
            "settings",
            SettingsPage(self.page_container, theme=self.theme, app=self, page_title="Settings"),
        )
        self.page_container.add_page(
            "process",
            ProcessPage(self.page_container, theme=self.theme, app=self, page_title="Process"),
        )

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
