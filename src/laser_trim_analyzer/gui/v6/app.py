"""Spec 3a — V6App root + placeholder pages. Foundations §2.2."""
from typing import Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.config import Config
from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.page_container import PageContainer
from laser_trim_analyzer.gui.v6.sidebar import Sidebar
from laser_trim_analyzer.gui.v6.theme import ThemeManager


class V6App(ctk.CTk):
    def __init__(self, config: Config, db=None, auto_train_on_first_run: bool = True):
        super().__init__()
        # Appearance set HERE (not at import) so importing this module never mutates
        # global CTk state for V5 or test runs.
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.config = config
        self.theme = ThemeManager()
        # Share ONE DatabaseManager with the rest of the app. Production: db is None ->
        # get_database() (same singleton Processor uses). Tests inject an isolated one.
        self.db = db if db is not None else get_database()
        self._model_route: Optional[Tuple[str, Optional[str]]] = None
        self._auto_train_on_first_run = auto_train_on_first_run

        self._setup_window()
        self._build_layout()
        self._build_pages()
        self.show_page("triage")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        # Data-gated first-startup auto-train (Spec 3d / D3). Disabled in tests via
        # the flag; the method itself re-checks the flag and the data gate.
        if self._auto_train_on_first_run:
            self.after(500, self._maybe_run_first_startup_train)

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
        if not self._should_offer_first_startup_train():
            return
        from laser_trim_analyzer.gui.v6.widgets.training_modal import TrainingModal
        preset = getattr(self.config.ml, "drift_sensitivity", "standard")
        TrainingModal(self, theme=self.theme, db=self.db, preset=preset).start()

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
        # Triage (3b), Model (3c), Settings (3d) are real; Process is a placeholder until 3e.
        from laser_trim_analyzer.gui.v6.pages.triage_page import TriagePage
        from laser_trim_analyzer.gui.v6.pages.model_page import ModelPage
        from laser_trim_analyzer.gui.v6.pages.settings_page import SettingsPage
        self.page_container.add_page(
            "triage",
            TriagePage(self.page_container, theme=self.theme, app=self, page_title="Triage"),
        )
        self.page_container.add_page(
            "model",
            ModelPage(self.page_container, theme=self.theme, app=self, page_title="Model"),
        )
        self.page_container.add_page(
            "settings",
            SettingsPage(self.page_container, theme=self.theme, app=self, page_title="Settings"),
        )
        self.page_container.add_page(
            "process",
            _PlaceholderPage(self.page_container, theme=self.theme, app=self,
                             page_title="Process", next_spec="3e"),
        )

    def _on_closing(self) -> None:
        self.destroy()

    def run(self) -> None:
        self.mainloop()


class _PlaceholderPage(PageBase):
    def __init__(self, master, *, theme, app, page_title, next_spec):
        self._next_spec = next_spec
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    def build_content(self, parent):
        ctk.CTkLabel(parent, text=f"{self.page_title} — coming in Spec {self._next_spec}.",
                     font=self.theme.font(self.theme.SIZE_HEADING),
                     text_color=self.theme.TEXT_SECONDARY).pack(expand=True)
