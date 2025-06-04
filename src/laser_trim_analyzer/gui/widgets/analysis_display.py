"""
Analysis Display Widget

Displays detailed analysis results for single file processing with
comprehensive validation information and visual feedback.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
import customtkinter as ctk
from tkinter import ttk
import tkinter as tk
from pathlib import Path

from laser_trim_analyzer.core.models import AnalysisResult, AnalysisStatus, ValidationStatus
from laser_trim_analyzer.gui.widgets.metric_card import MetricCard

logger = logging.getLogger(__name__)


class AnalysisDisplayWidget(ctk.CTkFrame):
    """Widget for displaying single file analysis results."""

    def __init__(self, parent, **kwargs):
        """Initialize analysis display widget."""
        super().__init__(parent, **kwargs)
        
        # State
        self.current_result: Optional[AnalysisResult] = None
        
        # Create widgets
        self._create_widgets()
        self._setup_layout()

    def _create_widgets(self):
        """Create analysis display widgets."""
        # Header frame
        self.header_frame = ctk.CTkFrame(self)
        
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        
        self.subtitle_label = ctk.CTkLabel(
            self.header_frame,
            text="No analysis completed",
            font=ctk.CTkFont(size=12)
        )
        
        # Status indicators frame
        self.status_frame = ctk.CTkFrame(self)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Status",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        # Status indicator cards
        self.overall_status_card = MetricCard(
            self.status_frame,
            title="Overall Status",
            value="Unknown",
            color_scheme="neutral"
        )
        
        self.validation_status_card = MetricCard(
            self.status_frame,
            title="Validation Status",
            value="Unknown",
            color_scheme="neutral"
        )
        
        self.validation_grade_card = MetricCard(
            self.status_frame,
            title="Validation Grade",
            value="N/A",
            color_scheme="neutral"
        )
        
        self.processing_time_card = MetricCard(
            self.status_frame,
            title="Processing Time",
            value="0s",
            color_scheme="info"
        )
        
        # Metadata frame
        self.metadata_frame = ctk.CTkFrame(self)
        
        self.metadata_label = ctk.CTkLabel(
            self.metadata_frame,
            text="File Information",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        # Metadata cards
        self.model_card = MetricCard(
            self.metadata_frame,
            title="Model",
            value="Unknown",
            color_scheme="info"
        )
        
        self.serial_card = MetricCard(
            self.metadata_frame,
            title="Serial",
            value="Unknown",
            color_scheme="info"
        )
        
        self.system_card = MetricCard(
            self.metadata_frame,
            title="System Type",
            value="Unknown",
            color_scheme="info"
        )
        
        self.date_card = MetricCard(
            self.metadata_frame,
            title="Trim Date",
            value="Unknown",
            color_scheme="info"
        )
        
        # Tracks frame
        self.tracks_frame = ctk.CTkFrame(self)
        
        self.tracks_label = ctk.CTkLabel(
            self.tracks_frame,
            text="Track Analysis",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        # Track selection
        self.track_selector_frame = ctk.CTkFrame(self.tracks_frame)
        
        self.track_selector_label = ctk.CTkLabel(
            self.track_selector_frame,
            text="Select Track:",
            font=ctk.CTkFont(size=12)
        )
        
        self.track_var = ctk.StringVar(value="No tracks")
        self.track_selector = ctk.CTkOptionMenu(
            self.track_selector_frame,
            variable=self.track_var,
            values=["No tracks"],
            command=self._on_track_selected
        )
        
        # Track details frame
        self.track_details_frame = ctk.CTkFrame(self.tracks_frame)
        
        # Track analysis cards
        self.sigma_gradient_card = MetricCard(
            self.track_details_frame,
            title="Sigma Gradient",
            value="N/A",
            color_scheme="neutral"
        )
        
        self.sigma_pass_card = MetricCard(
            self.track_details_frame,
            title="Sigma Pass",
            value="N/A",
            color_scheme="neutral"
        )
        
        self.linearity_error_card = MetricCard(
            self.track_details_frame,
            title="Linearity Error",
            value="N/A",
            color_scheme="neutral"
        )
        
        self.linearity_pass_card = MetricCard(
            self.track_details_frame,
            title="Linearity Pass",
            value="N/A",
            color_scheme="neutral"
        )
        
        # Validation details frame
        self.validation_frame = ctk.CTkFrame(self)
        
        self.validation_label = ctk.CTkLabel(
            self.validation_frame,