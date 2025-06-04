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
            text="Validation Details",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        # Validation text area
        self.validation_text = ctk.CTkTextbox(
            self.validation_frame,
            height=100,
            width=400
        )

    def _setup_layout(self):
        """Setup widget layout."""
        self.grid_rowconfigure([1, 2, 3, 4, 5], weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        self.title_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.subtitle_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))
        
        # Status indicators
        self.status_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.status_frame.grid_columnconfigure([0, 1, 2, 3], weight=1)
        
        self.status_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 5))
        
        self.overall_status_card.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.validation_status_card.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.validation_grade_card.grid(row=1, column=2, sticky="ew", padx=5, pady=5)
        self.processing_time_card.grid(row=1, column=3, sticky="ew", padx=5, pady=5)
        
        # Metadata
        self.metadata_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.metadata_frame.grid_columnconfigure([0, 1, 2, 3], weight=1)
        
        self.metadata_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 5))
        
        self.model_card.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.serial_card.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.system_card.grid(row=1, column=2, sticky="ew", padx=5, pady=5)
        self.date_card.grid(row=1, column=3, sticky="ew", padx=5, pady=5)
        
        # Tracks
        self.tracks_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        self.tracks_frame.grid_columnconfigure(0, weight=1)
        
        self.tracks_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        
        # Track selector
        self.track_selector_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.track_selector_frame.grid_columnconfigure(1, weight=1)
        
        self.track_selector_label.grid(row=0, column=0, sticky="w", padx=(10, 5), pady=5)
        self.track_selector.grid(row=0, column=1, sticky="ew", padx=(5, 10), pady=5)
        
        # Track details
        self.track_details_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))
        self.track_details_frame.grid_columnconfigure([0, 1, 2, 3], weight=1)
        
        self.sigma_gradient_card.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.sigma_pass_card.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.linearity_error_card.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
        self.linearity_pass_card.grid(row=0, column=3, sticky="ew", padx=5, pady=5)
        
        # Validation details
        self.validation_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.validation_frame.grid_rowconfigure(1, weight=1)
        self.validation_frame.grid_columnconfigure(0, weight=1)
        
        self.validation_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        self.validation_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def display_result(self, result: AnalysisResult):
        """Display analysis result."""
        self.current_result = result
        
        # Update header
        file_name = getattr(result.metadata, 'file_name', 
                           getattr(result.metadata, 'filename', 'Unknown'))
        self.subtitle_label.configure(text=f"Analysis of: {file_name}")
        
        # Update status cards
        self._update_status_cards(result)
        
        # Update metadata cards
        self._update_metadata_cards(result)
        
        # Update track information
        self._update_track_information(result)
        
        # Update validation details
        self._update_validation_details(result)
        
        logger.info(f"Displayed analysis result for: {file_name}")

    def _update_status_cards(self, result: AnalysisResult):
        """Update status indicator cards."""
        # Overall status
        status_value = getattr(result.overall_status, 'value', str(result.overall_status))
        status_color = self._get_status_color_scheme(result.overall_status)
        self.overall_status_card.update_value(status_value, status_color)
        
        # Validation status
        if hasattr(result, 'overall_validation_status') and result.overall_validation_status:
            validation_value = getattr(result.overall_validation_status, 'value', 
                                     str(result.overall_validation_status))
            validation_color = self._get_validation_color_scheme(result.overall_validation_status)
            self.validation_status_card.update_value(validation_value, validation_color)
        else:
            self.validation_status_card.update_value("Not Available", "neutral")
        
        # Validation grade
        grade = getattr(result, 'validation_grade', 'N/A')
        self.validation_grade_card.update_value(grade, "info")
        
        # Processing time
        processing_time = getattr(result, 'processing_time', 0)
        self.processing_time_card.update_value(f"{processing_time:.2f}s", "info")

    def _update_metadata_cards(self, result: AnalysisResult):
        """Update metadata cards."""
        if result.metadata:
            # Model
            model = getattr(result.metadata, 'model', 'Unknown')
            self.model_card.update_value(model, "info")
            
            # Serial
            serial = getattr(result.metadata, 'serial', 'Unknown')
            self.serial_card.update_value(serial, "info")
            
            # System type
            system_type = 'Unknown'
            if hasattr(result.metadata, 'system_type'):
                system_type = getattr(result.metadata.system_type, 'value', 
                                    str(result.metadata.system_type))
            elif hasattr(result.metadata, 'system'):
                system_type = getattr(result.metadata.system, 'value', 
                                    str(result.metadata.system))
            self.system_card.update_value(system_type, "info")
            
            # Date
            date_value = 'Unknown'
            if hasattr(result.metadata, 'file_date') and result.metadata.file_date:
                if hasattr(result.metadata.file_date, 'strftime'):
                    date_value = result.metadata.file_date.strftime('%Y-%m-%d')
                else:
                    date_value = str(result.metadata.file_date)
            self.date_card.update_value(date_value, "info")

    def _update_track_information(self, result: AnalysisResult):
        """Update track selection and details."""
        if hasattr(result, 'tracks') and result.tracks:
            # Update track selector
            track_ids = list(result.tracks.keys())
            self.track_selector.configure(values=track_ids)
            
            if track_ids:
                # Select first track by default
                self.track_var.set(track_ids[0])
                self._display_track_details(track_ids[0], result.tracks[track_ids[0]])
        else:
            # No tracks available
            self.track_selector.configure(values=["No tracks"])
            self.track_var.set("No tracks")
            self._clear_track_details()

    def _update_validation_details(self, result: AnalysisResult):
        """Update validation details text area."""
        validation_text = self._format_validation_details(result)
        
        self.validation_text.delete("1.0", ctk.END)
        self.validation_text.insert("1.0", validation_text)

    def _format_validation_details(self, result: Analysis