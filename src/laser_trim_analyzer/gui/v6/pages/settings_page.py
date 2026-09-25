"""Spec 3d — SettingsPage: scrollable list of 5 collapsible sections."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.sections.alert_thresholds import build_alert_thresholds_section
from laser_trim_analyzer.gui.v6.sections.backlog import build_backlog_section
from laser_trim_analyzer.gui.v6.sections.database_cleanup import build_database_cleanup_section
from laser_trim_analyzer.gui.v6.sections.ingest_folders import build_ingest_folders_section
from laser_trim_analyzer.gui.v6.sections.ml_training import build_ml_training_section
from laser_trim_analyzer.gui.v6.sections.per_model_specs import build_per_model_specs_section
from laser_trim_analyzer.gui.v6.widgets.settings_card import SettingsCard


class SettingsPage(PageBase):
    page_title = "Settings"

    def __init__(self, master, *, theme, app, page_title="Settings"):
        self._cards = []
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    def build_content(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        for title, expanded, build in (
            # First and open by default: on a fresh install this is the one
            # setting Home cannot work without, and Home's empty state sends
            # the user straight here. The Home connection is said in words
            # INSIDE the card (its body's own opening line already names
            # "Process everything new" -- see build_ingest_folders_section),
            # so the title itself stays short, sentence case (ruling 3).
            ("Ingest folders", True, build_ingest_folders_section),
            ("Backlog — active models and pricing", False, build_backlog_section),
            ("Alert thresholds", False, build_alert_thresholds_section),
            ("Per-model specs", False, build_per_model_specs_section),
            ("ML training", False, build_ml_training_section),
            ("Database", False, build_database_cleanup_section),
        ):
            card = SettingsCard(scroll, theme=self.theme, title=title, expanded=expanded)
            card.pack(side="top", fill="x", pady=(0, self.theme.SPACE_SM))
            build(card.body_frame(), theme=self.theme, app=self.app)
            self._cards.append(card)
