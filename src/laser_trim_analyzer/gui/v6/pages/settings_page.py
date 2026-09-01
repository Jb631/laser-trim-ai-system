"""Spec 3d — SettingsPage: scrollable list of 5 collapsible sections."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.sections.active_models import build_active_models_section
from laser_trim_analyzer.gui.v6.sections.alert_thresholds import build_alert_thresholds_section
from laser_trim_analyzer.gui.v6.sections.database_cleanup import build_database_cleanup_section
from laser_trim_analyzer.gui.v6.sections.ingest_folders import build_ingest_folders_section
from laser_trim_analyzer.gui.v6.sections.ml_training import build_ml_training_section
from laser_trim_analyzer.gui.v6.sections.per_model_specs import build_per_model_specs_section
from laser_trim_analyzer.gui.v6.sections.pricing import build_pricing_section
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
            # the user straight here.
            ("Ingest Folders (Home's “Process everything new”)", True,
             build_ingest_folders_section),
            ("Alert Thresholds", True, build_alert_thresholds_section),
            ("Active Models (MPS — your production schedule)", False, build_active_models_section),
            ("Per-model Specs", False, build_per_model_specs_section),
            ("ML Training", False, build_ml_training_section),
            ("Pricing", False, build_pricing_section),
            ("Database Cleanup", False, build_database_cleanup_section),
        ):
            card = SettingsCard(scroll, theme=self.theme, title=title, expanded=expanded)
            card.pack(side="top", fill="x", pady=(0, self.theme.SPACE_SM))
            build(card.body_frame(), theme=self.theme, app=self.app)
            self._cards.append(card)
