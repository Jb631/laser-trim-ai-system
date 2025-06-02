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
        
        # Validation text display
        self.validation_text = ctk.CTkTextbox(
            self.validation_frame,
            height=200,
            width=400
        )

    def _setup_layout(self):
        """Setup widget layout."""
        self.grid_rowconfigure([1, 2, 3, 4], weight=1)
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
        self.overall_status_card.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.validation_status_card.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.validation_grade_card.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        self.processing_time_card.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        
        # Metadata
        self.metadata_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.metadata_frame.grid_columnconfigure([0, 1, 2, 3], weight=1)
        
        self.metadata_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 5))
        self.model_card.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.serial_card.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.system_card.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        self.date_card.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        
        # Tracks
        self.tracks_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        self.tracks_frame.grid_rowconfigure(2, weight=1)
        self.tracks_frame.grid_columnconfigure(0, weight=1)
        
        self.tracks_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        
        # Track selector
        self.track_selector_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.track_selector_frame.grid_columnconfigure(1, weight=1)
        
        self.track_selector_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.track_selector.grid(row=0, column=1, sticky="ew")
        
        # Track details
        self.track_details_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.track_details_frame.grid_columnconfigure([0, 1, 2, 3], weight=1)
        
        self.sigma_gradient_card.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.sigma_pass_card.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.linearity_error_card.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        self.linearity_pass_card.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
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
        self.title_label.configure(text=f"Analysis Results: {result.metadata.filename}")
        self.subtitle_label.configure(text=f"Completed at {result.metadata.file_date}")
        
        # Update status cards
        self._update_status_cards(result)
        
        # Update metadata cards
        self._update_metadata_cards(result)
        
        # Update tracks
        self._update_tracks_display(result)
        
        # Update validation details
        self._update_validation_details(result)
        
        logger.info(f"Displayed analysis result for {result.metadata.filename}")

    def _update_status_cards(self, result: AnalysisResult):
        """Update status indicator cards."""
        # Overall status
        status_color = self._get_status_color(result.overall_status)
        self.overall_status_card.update_value(result.overall_status.value, status_color)
        
        # Validation status
        if hasattr(result, 'overall_validation_status'):
            validation_color = self._get_validation_color(result.overall_validation_status)
            self.validation_status_card.update_value(result.overall_validation_status.value, validation_color)
        else:
            self.validation_status_card.update_value("Unknown", "neutral")
        
        # Validation grade
        if hasattr(result, 'validation_grade') and result.validation_grade:
            grade_color = self._get_grade_color(result.validation_grade)
            self.validation_grade_card.update_value(result.validation_grade, grade_color)
        else:
            self.validation_grade_card.update_value("N/A", "neutral")
        
        # Processing time
        if result.processing_time:
            self.processing_time_card.update_value(f"{result.processing_time:.2f}s", "info")
        else:
            self.processing_time_card.update_value("N/A", "neutral")

    def _update_metadata_cards(self, result: AnalysisResult):
        """Update metadata cards."""
        metadata = result.metadata
        
        self.model_card.update_value(metadata.model or "Unknown", "info")
        self.serial_card.update_value(metadata.serial or "Unknown", "info")
        self.system_card.update_value(metadata.system.value if metadata.system else "Unknown", "info")
        
        if metadata.file_date:
            date_str = metadata.file_date.strftime("%Y-%m-%d %H:%M")
            self.date_card.update_value(date_str, "info")
        else:
            self.date_card.update_value("Unknown", "neutral")

    def _update_tracks_display(self, result: AnalysisResult):
        """Update tracks display."""
        if result.tracks:
            track_ids = list(result.tracks.keys())
            self.track_selector.configure(values=track_ids)
            
            # Select primary track or first track
            if result.primary_track and result.primary_track.track_id in track_ids:
                self.track_var.set(result.primary_track.track_id)
            else:
                self.track_var.set(track_ids[0])
            
            # Update track details
            self._update_track_details(result.tracks[self.track_var.get()])
        else:
            self.track_selector.configure(values=["No tracks"])
            self.track_var.set("No tracks")
            self._clear_track_details()

    def _update_track_details(self, track_data):
        """Update track details cards."""
        # Sigma analysis
        if track_data.sigma_analysis:
            sigma = track_data.sigma_analysis
            self.sigma_gradient_card.update_value(f"{sigma.sigma_gradient:.4f}", "info")
            
            if hasattr(sigma, 'sigma_pass'):
                pass_color = "success" if sigma.sigma_pass else "danger"
                pass_text = "✓ Pass" if sigma.sigma_pass else "✗ Fail"
                self.sigma_pass_card.update_value(pass_text, pass_color)
            else:
                self.sigma_pass_card.update_value("N/A", "neutral")
        else:
            self.sigma_gradient_card.update_value("N/A", "neutral")
            self.sigma_pass_card.update_value("N/A", "neutral")
        
        # Linearity analysis
        if track_data.linearity_analysis:
            linearity = track_data.linearity_analysis
            
            if hasattr(linearity, 'final_linearity_error_shifted'):
                self.linearity_error_card.update_value(f"{linearity.final_linearity_error_shifted:.3f}", "info")
            else:
                self.linearity_error_card.update_value("N/A", "neutral")
            
            if hasattr(linearity, 'linearity_pass'):
                pass_color = "success" if linearity.linearity_pass else "danger"
                pass_text = "✓ Pass" if linearity.linearity_pass else "✗ Fail"
                self.linearity_pass_card.update_value(pass_text, pass_color)
            else:
                self.linearity_pass_card.update_value("N/A", "neutral")
        else:
            self.linearity_error_card.update_value("N/A", "neutral")
            self.linearity_pass_card.update_value("N/A", "neutral")

    def _clear_track_details(self):
        """Clear track details cards."""
        self.sigma_gradient_card.update_value("N/A", "neutral")
        self.sigma_pass_card.update_value("N/A", "neutral")
        self.linearity_error_card.update_value("N/A", "neutral")
        self.linearity_pass_card.update_value("N/A", "neutral")

    def _update_validation_details(self, result: AnalysisResult):
        """Update validation details text."""
        details = []
        
        # Validation summary
        if hasattr(result, 'overall_validation_status'):
            details.append(f"Overall Validation Status: {result.overall_validation_status.value}")
        
        if hasattr(result, 'validation_grade') and result.validation_grade:
            details.append(f"Validation Grade: {result.validation_grade}")
        
        details.append("")
        
        # Validation warnings
        if hasattr(result, 'validation_warnings') and result.validation_warnings:
            details.append("Validation Warnings:")
            for warning in result.validation_warnings[:20]:  # Limit to first 20
                details.append(f"  • {warning}")
            
            if len(result.validation_warnings) > 20:
                details.append(f"  ... and {len(result.validation_warnings) - 20} more warnings")
            
            details.append("")
        
        # Validation recommendations
        if hasattr(result, 'validation_recommendations') and result.validation_recommendations:
            details.append("Validation Recommendations:")
            for recommendation in result.validation_recommendations[:15]:  # Limit to first 15
                details.append(f"  • {recommendation}")
            
            if len(result.validation_recommendations) > 15:
                details.append(f"  ... and {len(result.validation_recommendations) - 15} more recommendations")
            
            details.append("")
        
        # Track validation details
        if result.tracks:
            details.append("Track Validation Details:")
            for track_id, track_data in result.tracks.items():
                details.append(f"  {track_id}:")
                
                if hasattr(track_data, 'validation_status'):
                    details.append(f"    Status: {track_data.validation_status.value}")
                
                if hasattr(track_data, 'validation_warnings') and track_data.validation_warnings:
                    details.append(f"    Warnings: {len(track_data.validation_warnings)}")
                
                # Individual analysis validation
                for analysis_name, analysis in [
                    ('Sigma', track_data.sigma_analysis),
                    ('Linearity', track_data.linearity_analysis),
                    ('Resistance', track_data.resistance_analysis)
                ]:
                    if analysis and hasattr(analysis, 'validation_status'):
                        details.append(f"    {analysis_name}: {analysis.validation_status.value}")
        
        # Display details
        details_text = "\n".join(details)
        self.validation_text.delete("1.0", ctk.END)
        self.validation_text.insert("1.0", details_text)

    def _on_track_selected(self, track_id: str):
        """Handle track selection."""
        if self.current_result and track_id in self.current_result.tracks:
            self._update_track_details(self.current_result.tracks[track_id])

    def _get_status_color(self, status: AnalysisStatus) -> str:
        """Get color scheme for analysis status."""
        status_colors = {
            AnalysisStatus.PASS: "success",
            AnalysisStatus.WARNING: "warning",
            AnalysisStatus.FAIL: "danger",
            AnalysisStatus.ERROR: "danger"
        }
        return status_colors.get(status, "neutral")

    def _get_validation_color(self, status: ValidationStatus) -> str:
        """Get color scheme for validation status."""
        validation_colors = {
            ValidationStatus.VALIDATED: "success",
            ValidationStatus.WARNING: "warning",
            ValidationStatus.FAILED: "danger",
            ValidationStatus.NOT_VALIDATED: "neutral"
        }
        return validation_colors.get(status, "neutral")

    def _get_grade_color(self, grade: str) -> str:
        """Get color scheme for validation grade."""
        if grade.startswith('A'):
            return "success"
        elif grade.startswith('B'):
            return "info"
        elif grade.startswith('C'):
            return "warning"
        elif grade.startswith('D') or grade.startswith('F'):
            return "danger"
        else:
            return "neutral"

    def clear(self):
        """Clear the analysis display."""
        self.current_result = None
        
        # Reset header
        self.title_label.configure(text="Analysis Results")
        self.subtitle_label.configure(text="No analysis completed")
        
        # Reset status cards
        self.overall_status_card.update_value("Unknown", "neutral")
        self.validation_status_card.update_value("Unknown", "neutral")
        self.validation_grade_card.update_value("N/A", "neutral")
        self.processing_time_card.update_value("0s", "neutral")
        
        # Reset metadata cards
        self.model_card.update_value("Unknown", "neutral")
        self.serial_card.update_value("Unknown", "neutral")
        self.system_card.update_value("Unknown", "neutral")
        self.date_card.update_value("Unknown", "neutral")
        
        # Reset tracks
        self.track_selector.configure(values=["No tracks"])
        self.track_var.set("No tracks")
        self._clear_track_details()
        
        # Clear validation details
        self.validation_text.delete("1.0", ctk.END)

    def export_details(self) -> str:
        """Export current details as formatted text."""
        if not self.current_result:
            return "No analysis result to export"
        
        result = self.current_result
        export_lines = []
        
        # Header
        export_lines.append("="*60)
        export_lines.append(f"LASER TRIM ANALYSIS REPORT")
        export_lines.append("="*60)
        export_lines.append("")
        
        # File information
        export_lines.append("FILE INFORMATION:")
        export_lines.append(f"  Filename: {result.metadata.filename}")
        export_lines.append(f"  Model: {result.metadata.model}")
        export_lines.append(f"  Serial: {result.metadata.serial}")
        export_lines.append(f"  System: {result.metadata.system.value}")
        export_lines.append(f"  Date: {result.metadata.file_date}")
        export_lines.append("")
        
        # Overall results
        export_lines.append("OVERALL RESULTS:")
        export_lines.append(f"  Status: {result.overall_status.value}")
        
        if hasattr(result, 'overall_validation_status'):
            export_lines.append(f"  Validation: {result.overall_validation_status.value}")
        
        if hasattr(result, 'validation_grade'):
            export_lines.append(f"  Grade: {result.validation_grade}")
        
        export_lines.append(f"  Processing Time: {result.processing_time:.2f}s")
        export_lines.append("")
        
        # Track details
        if result.tracks:
            export_lines.append("TRACK ANALYSIS:")
            for track_id, track_data in result.tracks.items():
                export_lines.append(f"  {track_id}:")
                export_lines.append(f"    Status: {track_data.status.value}")
                
                if track_data.sigma_analysis:
                    sigma = track_data.sigma_analysis
                    export_lines.append(f"    Sigma Gradient: {sigma.sigma_gradient:.4f}")
                    if hasattr(sigma, 'sigma_pass'):
                        export_lines.append(f"    Sigma Pass: {'Yes' if sigma.sigma_pass else 'No'}")
                
                if track_data.linearity_analysis:
                    linearity = track_data.linearity_analysis
                    if hasattr(linearity, 'final_linearity_error_shifted'):
                        export_lines.append(f"    Linearity Error: {linearity.final_linearity_error_shifted:.3f}")
                    if hasattr(linearity, 'linearity_pass'):
                        export_lines.append(f"    Linearity Pass: {'Yes' if linearity.linearity_pass else 'No'}")
                
                export_lines.append("")
        
        # Validation details
        if hasattr(result, 'validation_warnings') and result.validation_warnings:
            export_lines.append("VALIDATION WARNINGS:")
            for warning in result.validation_warnings:
                export_lines.append(f"  • {warning}")
            export_lines.append("")
        
        if hasattr(result, 'validation_recommendations') and result.validation_recommendations:
            export_lines.append("RECOMMENDATIONS:")
            for recommendation in result.validation_recommendations:
                export_lines.append(f"  • {recommendation}")
            export_lines.append("")
        
        export_lines.append("="*60)
        export_lines.append(f"Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        export_lines.append("="*60)
        
        return "\n".join(export_lines) 