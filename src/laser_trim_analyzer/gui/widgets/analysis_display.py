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
from laser_trim_analyzer.gui.widgets.metric_card_ctk import MetricCard
from laser_trim_analyzer.gui.widgets.plot_viewer import PlotViewerWidget

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
        
        # Status reason label (for warnings/failures)
        self.status_reason_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=("orange", "orange"),
            wraplength=400
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
        
        # Validation text area (increased height for better visibility)
        self.validation_text = ctk.CTkTextbox(
            self.validation_frame,
            height=250,
            width=400
        )
        
        # Plot viewer frame
        self.plot_frame = ctk.CTkFrame(self)
        
        self.plot_label = ctk.CTkLabel(
            self.plot_frame,
            text="Analysis Plot",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        # Plot viewer widget
        self.plot_viewer = PlotViewerWidget(self.plot_frame)
        self.plot_viewer.configure(height=600)

    def _setup_layout(self):
        """Setup widget layout."""
        self.grid_rowconfigure([1, 2, 3, 4, 5, 6], weight=1)
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
        
        # Status reason label spans across columns
        self.status_reason_label.grid(row=2, column=0, columnspan=4, sticky="ew", padx=15, pady=(0, 5))
        
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
        
        # Plot viewer
        self.plot_frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.plot_frame.grid_rowconfigure(1, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        
        self.plot_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        self.plot_viewer.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

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
        
        # Update plot display
        self._update_plot_display(result)
        
        logger.info(f"Displayed analysis result for: {file_name}")

    def _update_status_cards(self, result: AnalysisResult):
        """Update status indicator cards."""
        # Overall status
        status_value = getattr(result.overall_status, 'value', str(result.overall_status))
        status_color = self._get_status_color_scheme(result.overall_status)
        self.overall_status_card.update_value(status_value, status_color)
        
        # Status reason - collect reasons from all tracks
        status_reasons = []
        if hasattr(result, 'tracks') and result.tracks:
            for track_id, track_data in result.tracks.items():
                if hasattr(track_data, 'status_reason') and track_data.status_reason:
                    # Only show reasons for warnings and failures
                    if track_data.status in [AnalysisStatus.WARNING, AnalysisStatus.FAIL]:
                        if len(result.tracks) > 1:
                            status_reasons.append(f"Track {track_id}: {track_data.status_reason}")
                        else:
                            status_reasons.append(track_data.status_reason)
        
        # Update status reason label
        if status_reasons:
            reason_text = " | ".join(status_reasons)
            self.status_reason_label.configure(text=reason_text)
            # Set color based on overall status
            if result.overall_status == AnalysisStatus.FAIL:
                self.status_reason_label.configure(text_color=("red", "red"))
            else:
                self.status_reason_label.configure(text_color=("orange", "orange"))
        else:
            self.status_reason_label.configure(text="")
        
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

    def _format_validation_details(self, result: AnalysisResult) -> str:
        """Format validation details for display."""
        details = []
        
        # Basic validation info
        details.append("=== VALIDATION SUMMARY ===")
        
        if hasattr(result, 'overall_validation_status'):
            details.append(f"Overall Status: {result.overall_validation_status.value}")
        
        if hasattr(result, 'validation_grade'):
            details.append(f"Validation Grade: {result.validation_grade}")
        
        details.append("")
        
        # Validation warnings
        if hasattr(result, 'validation_warnings') and result.validation_warnings:
            details.append("=== WARNINGS ===")
            for i, warning in enumerate(result.validation_warnings[:10], 1):
                details.append(f"{i}. {warning}")
            
            if len(result.validation_warnings) > 10:
                details.append(f"... and {len(result.validation_warnings) - 10} more warnings")
            
            details.append("")
        
        # Validation recommendations
        if hasattr(result, 'validation_recommendations') and result.validation_recommendations:
            details.append("=== RECOMMENDATIONS ===")
            for i, rec in enumerate(result.validation_recommendations[:5], 1):
                details.append(f"{i}. {rec}")
            
            if len(result.validation_recommendations) > 5:
                details.append(f"... and {len(result.validation_recommendations) - 5} more recommendations")
            
            details.append("")
        
        # Track validation details
        if hasattr(result, 'tracks') and result.tracks:
            details.append("=== TRACK VALIDATION ===")
            for track_id, track_data in result.tracks.items():
                details.append(f"Track {track_id}:")
                
                if hasattr(track_data, 'validation_status'):
                    details.append(f"  Status: {track_data.validation_status.value}")
                
                if hasattr(track_data, 'validation_warnings') and track_data.validation_warnings:
                    details.append(f"  Warnings: {len(track_data.validation_warnings)}")
                    for warning in track_data.validation_warnings[:3]:
                        details.append(f"    • {warning}")
                
                details.append("")
        
        # If no validation details available
        if len(details) <= 3:
            details = ["No detailed validation information available."]
        
        return "\n".join(details)

    def _on_track_selected(self, selected_track: str):
        """Handle track selection change."""
        # Prevent multiple simultaneous updates
        if hasattr(self, '_updating_plot') and self._updating_plot:
            return
            
        if self.current_result and hasattr(self.current_result, 'tracks'):
            if selected_track in self.current_result.tracks:
                track_data = self.current_result.tracks[selected_track]
                self._display_track_details(selected_track, track_data)
            else:
                self._clear_track_details()
                
            # Update plot display for the selected track
            self._update_plot_display(self.current_result)

    def _display_track_details(self, track_id: str, track_data):
        """Display details for selected track."""
        try:
            # Sigma analysis
            if hasattr(track_data, 'sigma_analysis') and track_data.sigma_analysis:
                sigma_gradient = getattr(track_data.sigma_analysis, 'sigma_gradient', None)
                sigma_threshold = getattr(track_data.sigma_analysis, 'sigma_threshold', None)
                
                if sigma_gradient is not None and sigma_threshold is not None:
                    # Show gradient with threshold label
                    self.sigma_gradient_card.update_value(
                        f"{sigma_gradient:.4f} (Limit: {sigma_threshold:.4f})", 
                        "info"
                    )
                elif sigma_gradient is not None:
                    self.sigma_gradient_card.update_value(f"{sigma_gradient:.4f}", "info")
                else:
                    self.sigma_gradient_card.update_value("N/A", "neutral")
                
                sigma_pass = getattr(track_data.sigma_analysis, 'sigma_pass', None)
                if sigma_pass is not None:
                    pass_text = "PASS" if sigma_pass else "FAIL"
                    color = "success" if sigma_pass else "danger"
                    self.sigma_pass_card.update_value(pass_text, color)
                else:
                    self.sigma_pass_card.update_value("N/A", "neutral")
            else:
                self.sigma_gradient_card.update_value("N/A", "neutral")
                self.sigma_pass_card.update_value("N/A", "neutral")
            
            # Linearity analysis
            if hasattr(track_data, 'linearity_analysis') and track_data.linearity_analysis:
                linearity_error = getattr(track_data.linearity_analysis, 'final_linearity_error_shifted', None)
                linearity_spec = getattr(track_data.linearity_analysis, 'linearity_spec', None)
                
                if linearity_error is not None and linearity_spec is not None:
                    # Show error with spec limit label
                    self.linearity_error_card.update_value(
                        f"{linearity_error:.3f} (Limit: {linearity_spec:.3f})", 
                        "info"
                    )
                elif linearity_error is not None:
                    self.linearity_error_card.update_value(f"{linearity_error:.3f}", "info")
                else:
                    self.linearity_error_card.update_value("N/A", "neutral")
                
                linearity_pass = getattr(track_data.linearity_analysis, 'linearity_pass', None)
                if linearity_pass is not None:
                    pass_text = "PASS" if linearity_pass else "FAIL"
                    color = "success" if linearity_pass else "danger"
                    self.linearity_pass_card.update_value(pass_text, color)
                else:
                    self.linearity_pass_card.update_value("N/A", "neutral")
            else:
                self.linearity_error_card.update_value("N/A", "neutral")
                self.linearity_pass_card.update_value("N/A", "neutral")
                
        except Exception as e:
            logger.error(f"Error displaying track details: {e}")
            self._clear_track_details()

    def _clear_track_details(self):
        """Clear track detail cards."""
        self.sigma_gradient_card.update_value("N/A", "neutral")
        self.sigma_pass_card.update_value("N/A", "neutral")
        self.linearity_error_card.update_value("N/A", "neutral")
        self.linearity_pass_card.update_value("N/A", "neutral")

    def _get_status_color_scheme(self, status) -> str:
        """Get color scheme for analysis status."""
        if hasattr(status, 'value'):
            status_value = status.value.lower()
        else:
            status_value = str(status).lower()
        
        if 'pass' in status_value:
            return "success"
        elif 'fail' in status_value:
            return "danger"
        elif 'warning' in status_value:
            return "warning"
        else:
            return "neutral"

    def _get_validation_color_scheme(self, validation_status) -> str:
        """Get color scheme for validation status."""
        if hasattr(validation_status, 'value'):
            status_value = validation_status.value.lower()
        else:
            status_value = str(validation_status).lower()
        
        if 'validated' in status_value:
            return "success"
        elif 'failed' in status_value:
            return "danger"
        elif 'warning' in status_value:
            return "warning"
        else:
            return "neutral"

    def clear(self):
        """Clear all displayed data."""
        self.current_result = None
        
        # Reset header
        self.subtitle_label.configure(text="No analysis completed")
        
        # Reset status cards
        self.overall_status_card.update_value("Unknown", "neutral")
        self.validation_status_card.update_value("Unknown", "neutral")
        self.validation_grade_card.update_value("N/A", "neutral")
        self.processing_time_card.update_value("0s", "info")
        
        # Reset metadata cards
        self.model_card.update_value("Unknown", "info")
        self.serial_card.update_value("Unknown", "info")
        self.system_card.update_value("Unknown", "info")
        self.date_card.update_value("Unknown", "info")
        
        # Reset track selector
        self.track_selector.configure(values=["No tracks"])
        self.track_var.set("No tracks")
        self._clear_track_details()
        
        # Clear validation text
        self.validation_text.delete("1.0", ctk.END)
        self.validation_text.insert("1.0", "No validation details available.")
        
        # Clear plot viewer
        self.plot_viewer.clear()
        
    def _update_plot_display(self, result: AnalysisResult):
        """Update plot display with current track's plot if available."""
        # Set flag to prevent concurrent updates
        self._updating_plot = True
        
        try:
            # Clear previous plot
            self.plot_viewer.clear()
            
            # Small delay to ensure Tkinter processes the clear operation
            self.plot_viewer.update_idletasks()
            
            # Check if we have tracks and a selected track
            if hasattr(result, 'tracks') and result.tracks:
                selected_track = self.track_var.get()
                
                # Find the selected track data
                track_data = None
                if isinstance(result.tracks, dict):
                    track_data = result.tracks.get(selected_track)
                elif isinstance(result.tracks, list) and selected_track.startswith("Track "):
                    # Extract index from "Track X" format
                    try:
                        track_index = int(selected_track.split()[1]) - 1
                        if 0 <= track_index < len(result.tracks):
                            track_data = result.tracks[track_index]
                    except (ValueError, IndexError):
                        pass
                
                # Load plot if available
                if track_data and hasattr(track_data, 'plot_path') and track_data.plot_path:
                    plot_path = Path(track_data.plot_path)
                    if plot_path.exists():
                        self.plot_viewer.load_plot(plot_path)
                        logger.info(f"Loaded plot for {selected_track}: {plot_path}")
                    else:
                        logger.warning(f"Plot file not found: {plot_path}")
                        self.plot_viewer._show_error(f"Plot file not found for {selected_track}")
                else:
                    # No plot available for this track
                    logger.debug(f"No plot path available for {selected_track}")
                    self.plot_viewer._show_error(f"No plot available for {selected_track}")
            else:
                logger.debug("No tracks available for plot display")
                self.plot_viewer._show_error("No analysis tracks available")
                
        except Exception as e:
            logger.error(f"Error updating plot display: {e}")
            self.plot_viewer._show_error(f"Error loading plot: {str(e)}")
        finally:
            # Clear the update flag
            self._updating_plot = False
