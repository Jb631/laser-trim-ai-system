"""
Batch Results Widget

Displays batch processing results in a structured table format with
comprehensive analysis information and validation status.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import customtkinter as ctk
from tkinter import ttk
import tkinter as tk
import pandas as pd

from laser_trim_analyzer.core.models import AnalysisResult, AnalysisStatus, ValidationStatus
from laser_trim_analyzer.gui.theme_helper import ThemeHelper

logger = logging.getLogger(__name__)


class BatchResultsWidget(ctk.CTkFrame):
    """Widget for displaying batch processing results."""

    def __init__(self, parent, **kwargs):
        """Initialize batch results widget."""
        super().__init__(parent, **kwargs)
        
        # State
        self.results: Dict[str, AnalysisResult] = {}
        
        # Create widgets
        self._create_widgets()
        self._setup_layout()

    def _create_widgets(self):
        """Create batch results widgets."""
        # Header frame
        self.header_frame = ctk.CTkFrame(self)
        
        self.header_label = ctk.CTkLabel(
            self.header_frame,
            text="Batch Processing Results",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        
        self.summary_label = ctk.CTkLabel(
            self.header_frame,
            text="No results to display",
            font=ctk.CTkFont(size=12)
        )
        
        # Results table frame
        self.table_frame = ctk.CTkFrame(self)
        
        # Create treeview for results table
        self.tree_frame = ctk.CTkFrame(self.table_frame)
        
        # Configure treeview style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Apply theme-aware styling
        ThemeHelper.apply_ttk_style(style)
        
        # Define columns
        self.columns = [
            "File",
            "Status",
            "Validation",
            "Grade",
            "Sigma Pass",
            "Linearity Pass",
            "Processing Time",
            "Warnings"
        ]
        
        self.tree = ttk.Treeview(
            self.tree_frame,
            columns=self.columns,
            show='headings',
            height=15
        )
        
        # Configure column headings and widths
        column_widths = {
            "File": 200,
            "Status": 80,
            "Validation": 100,
            "Grade": 80,
            "Sigma Pass": 80,
            "Linearity Pass": 100,
            "Processing Time": 120,
            "Warnings": 80
        }
        
        for col in self.columns:
            self.tree.heading(col, text=col, anchor='w')
            self.tree.column(col, width=column_widths.get(col, 100), anchor='w')
        
        # Scrollbars
        self.v_scrollbar = ttk.Scrollbar(self.tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.v_scrollbar.set)
        
        self.h_scrollbar = ttk.Scrollbar(self.tree_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(xscrollcommand=self.h_scrollbar.set)
        
        # Details frame for selected item
        self.details_frame = ctk.CTkFrame(self)
        
        self.details_label = ctk.CTkLabel(
            self.details_frame,
            text="File Details",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        self.details_text = ctk.CTkTextbox(
            self.details_frame,
            height=150,
            width=400
        )
        
        # Configure tags for row styling
        self.tree.tag_configure('pass', foreground='#00ff00')
        self.tree.tag_configure('fail', foreground='#ff0000')
        self.tree.tag_configure('warning', foreground='#ffaa00')
        self.tree.tag_configure('error', foreground='#ff0000')
        self.tree.tag_configure('validated', background='#1a3d1a')
        self.tree.tag_configure('validation_failed', background='#3d1a1a')
        self.tree.tag_configure('validation_warning', background='#3d3d1a')
        
        # Bind tree selection event
        self.tree.bind('<<TreeviewSelect>>', self._on_item_select)

    def _setup_layout(self):
        """Setup widget layout."""
        # Configure grid weights for proper resizing
        self.grid_rowconfigure(1, weight=2)  # Table gets most space
        self.grid_rowconfigure(2, weight=1)  # Details get less space
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        self.header_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.summary_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))
        
        # Table
        self.table_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.table_frame.grid_rowconfigure(0, weight=1)
        self.table_frame.grid_columnconfigure(0, weight=1)
        
        self.tree_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_columnconfigure(0, weight=1)
        
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Details
        self.details_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.details_frame.grid_rowconfigure(1, weight=1)
        self.details_frame.grid_columnconfigure(0, weight=1)
        
        self.details_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        self.details_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def display_results(self, results: Dict[str, AnalysisResult]):
        """Display batch processing results."""
        self.results = results
        
        # Clear existing items
        self.clear()
        
        if not results:
            self.summary_label.configure(text="No results to display")
            return
        
        # Update summary
        total_files = len(results)
        passed_files = sum(1 for r in results.values() if r.overall_status == AnalysisStatus.PASS)
        validated_files = sum(1 for r in results.values() if r.overall_validation_status == ValidationStatus.VALIDATED)
        
        summary_text = f"Total: {total_files} | Passed: {passed_files} | Validated: {validated_files}"
        self.summary_label.configure(text=summary_text)
        
        # Populate table
        for file_path, result in results.items():
            self._add_result_row(file_path, result)
        
        logger.info(f"Displayed {total_files} batch results")

    def _add_result_row(self, file_path: str, result: AnalysisResult):
        """Add a result row to the table."""
        # Extract key information
        file_name = Path(file_path).name
        status = result.overall_status.value
        validation_status = result.overall_validation_status.value if hasattr(result, 'overall_validation_status') else "Unknown"
        grade = getattr(result, 'validation_grade', 'N/A')
        
        # Get primary track info
        primary_track = result.primary_track
        sigma_pass = "N/A"
        linearity_pass = "N/A"
        
        if primary_track:
            if primary_track.sigma_analysis and hasattr(primary_track.sigma_analysis, 'sigma_pass'):
                sigma_pass = "✓" if primary_track.sigma_analysis.sigma_pass else "✗"
            
            if primary_track.linearity_analysis and hasattr(primary_track.linearity_analysis, 'linearity_pass'):
                linearity_pass = "✓" if primary_track.linearity_analysis.linearity_pass else "✗"
        
        processing_time = f"{result.processing_time:.2f}s" if result.processing_time else "N/A"
        
        # Count warnings
        warning_count = 0
        if hasattr(result, 'validation_warnings') and result.validation_warnings:
            warning_count = len(result.validation_warnings)
        
        warnings_text = str(warning_count) if warning_count > 0 else "-"
        
        # Determine row colors based on status
        tags = self._get_row_tags(result.overall_status, result.overall_validation_status if hasattr(result, 'overall_validation_status') else None)
        
        # Insert row
        item_id = self.tree.insert('', 'end', values=[
            file_name,
            status,
            validation_status,
            grade,
            sigma_pass,
            linearity_pass,
            processing_time,
            warnings_text
        ], tags=tags)
        
        # Store result reference for details view
        # Note: ttk.Treeview doesn't have a set() method, so we'll use the file path lookup approach

    def _get_row_tags(self, status: AnalysisStatus, validation_status: Optional[ValidationStatus] = None) -> tuple:
        """Get tags for row coloring based on status."""
        tags = []
        
        # Status-based tags
        if status == AnalysisStatus.PASS:
            tags.append('pass')
        elif status == AnalysisStatus.FAIL:
            tags.append('fail')
        elif status == AnalysisStatus.WARNING:
            tags.append('warning')
        else:
            tags.append('error')
        
        # Validation-based tags
        if validation_status:
            if validation_status == ValidationStatus.VALIDATED:
                tags.append('validated')
            elif validation_status == ValidationStatus.FAILED:
                tags.append('validation_failed')
            elif validation_status == ValidationStatus.WARNING:
                tags.append('validation_warning')
        
        return tuple(tags)

    def _on_item_select(self, event):
        """Handle tree item selection."""
        selection = self.tree.selection()
        if not selection:
            return
        
        # Get selected item
        item = selection[0]
        file_path = None
        
        # Find the file path (stored during row creation)
        for file_path_key, result in self.results.items():
            file_name = Path(file_path_key).name
            if self.tree.item(item)['values'][0] == file_name:
                file_path = file_path_key
                break
        
        if file_path and file_path in self.results:
            self._show_file_details(file_path, self.results[file_path])

    def _show_file_details(self, file_path: str, result: AnalysisResult):
        """Show detailed information for selected file."""
        details = self._format_file_details(file_path, result)
        
        self.details_text.delete("1.0", ctk.END)
        self.details_text.insert("1.0", details)

    def _format_file_details(self, file_path: str, result: AnalysisResult) -> str:
        """Format detailed information for a file."""
        details = []
        
        # File information
        details.append(f"File: {Path(file_path).name}")
        details.append(f"Path: {file_path}")
        details.append(f"Status: {result.overall_status.value}")
        
        if hasattr(result, 'overall_validation_status'):
            details.append(f"Validation: {result.overall_validation_status.value}")
        
        if hasattr(result, 'validation_grade'):
            details.append(f"Grade: {result.validation_grade}")
        
        details.append(f"Processing Time: {result.processing_time:.2f}s")
        details.append("")
        
        # Metadata
        if result.metadata:
            details.append("Metadata:")
            details.append(f"  Model: {result.metadata.model}")
            details.append(f"  Serial: {result.metadata.serial}")
            details.append(f"  Date: {result.metadata.file_date}")
            details.append(f"  System: {result.metadata.system.value}")
            details.append("")
        
        # Track information
        if result.tracks:
            details.append(f"Tracks ({len(result.tracks)}):")
            for track_id, track_data in result.tracks.items():
                details.append(f"  {track_id}:")
                details.append(f"    Status: {track_data.status.value}")
                
                if hasattr(track_data, 'validation_status'):
                    details.append(f"    Validation: {track_data.validation_status.value}")
                
                if track_data.sigma_analysis:
                    details.append(f"    Sigma: {track_data.sigma_analysis.sigma_gradient:.4f}")
                
                if track_data.linearity_analysis:
                    details.append(f"    Linearity: {track_data.linearity_analysis.final_linearity_error_shifted:.3f}")
            
            details.append("")
        
        # Validation warnings
        if hasattr(result, 'validation_warnings') and result.validation_warnings:
            details.append("Validation Warnings:")
            for warning in result.validation_warnings[:10]:  # Limit to first 10
                details.append(f"  • {warning}")
            
            if len(result.validation_warnings) > 10:
                details.append(f"  ... and {len(result.validation_warnings) - 10} more")
            
            details.append("")
        
        # Validation recommendations
        if hasattr(result, 'validation_recommendations') and result.validation_recommendations:
            details.append("Recommendations:")
            for recommendation in result.validation_recommendations[:5]:  # Limit to first 5
                details.append(f"  • {recommendation}")
            
            if len(result.validation_recommendations) > 5:
                details.append(f"  ... and {len(result.validation_recommendations) - 5} more")
        
        return "\n".join(details)

    def clear(self):
        """Clear all results from the widget."""
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Clear details
        self.details_text.delete("1.0", ctk.END)
        
        # Reset summary
        self.summary_label.configure(text="No results to display")
        
        # Clear stored results
        self.results = {}

    def export_to_csv(self, file_path: str):
        """Export results to CSV file."""
        if not self.results:
            return
        
        # Prepare data for export
        export_data = []
        for file_path_key, result in self.results.items():
            row_data = {
                'File': Path(file_path_key).name,
                'Full_Path': file_path_key,
                'Status': result.overall_status.value,
                'Processing_Time': result.processing_time
            }
            
            # Add validation info if available
            if hasattr(result, 'overall_validation_status'):
                row_data['Validation_Status'] = result.overall_validation_status.value
            
            if hasattr(result, 'validation_grade'):
                row_data['Validation_Grade'] = result.validation_grade
            
            # Add metadata
            if result.metadata:
                row_data.update({
                    'Model': result.metadata.model,
                    'Serial': result.metadata.serial,
                    'File_Date': result.metadata.file_date,
                    'System': result.metadata.system.value
                })
            
            # Add primary track info
            if result.primary_track:
                primary = result.primary_track
                row_data.update({
                    'Track_Status': primary.status.value,
                    'Sigma_Gradient': getattr(primary.sigma_analysis, 'sigma_gradient', None),
                    'Sigma_Pass': getattr(primary.sigma_analysis, 'sigma_pass', None),
                    'Linearity_Error': getattr(primary.linearity_analysis, 'final_linearity_error_shifted', None),
                    'Linearity_Pass': getattr(primary.linearity_analysis, 'linearity_pass', None)
                })
            
            export_data.append(row_data)
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        df.to_csv(file_path, index=False)
        
        logger.info(f"Exported {len(export_data)} batch results to {file_path}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the batch results."""
        if not self.results:
            return {}
        
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r.overall_status == AnalysisStatus.PASS)
        failed = sum(1 for r in self.results.values() if r.overall_status == AnalysisStatus.FAIL)
        warnings = sum(1 for r in self.results.values() if r.overall_status == AnalysisStatus.WARNING)
        
        # Validation stats
        validated = 0
        validation_failed = 0
        validation_warnings = 0
        
        for result in self.results.values():
            if hasattr(result, 'overall_validation_status'):
                if result.overall_validation_status == ValidationStatus.VALIDATED:
                    validated += 1
                elif result.overall_validation_status == ValidationStatus.FAILED:
                    validation_failed += 1
                elif result.overall_validation_status == ValidationStatus.WARNING:
                    validation_warnings += 1
        
        # Calculate rates
        pass_rate = (passed / total * 100) if total > 0 else 0
        validation_rate = (validated / total * 100) if total > 0 else 0
        
        # Average processing time
        avg_processing_time = sum(r.processing_time for r in self.results.values() if r.processing_time) / total
        
        return {
            'total_files': total,
            'passed': passed,
            'failed': failed,
            'warnings': warnings,
            'validated': validated,
            'validation_failed': validation_failed,
            'validation_warnings': validation_warnings,
            'pass_rate': pass_rate,
            'validation_rate': validation_rate,
            'avg_processing_time': avg_processing_time
        } 