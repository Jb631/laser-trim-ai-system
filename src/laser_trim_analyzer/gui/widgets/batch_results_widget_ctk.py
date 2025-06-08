"""
Batch Results Widget - CustomTkinter Version

Display results from batch processing in a table format.
"""

import customtkinter as ctk
from typing import Dict, List
from pathlib import Path
import logging

from laser_trim_analyzer.core.models import AnalysisResult, ValidationStatus


class BatchResultsWidget(ctk.CTkFrame):
    """Widget to display batch processing results in a scrollable table."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the widget UI."""
        # Define fixed column widths for alignment
        self.column_widths = {
            'File': 250,
            'Model': 100,
            'Serial': 100,
            'Status': 80,
            'Validation': 100,
            'Tracks': 60,
            'Pass/Fail': 80
        }
        
        # Header
        header_frame = ctk.CTkFrame(self)
        header_frame.pack(fill='x', padx=5, pady=(5, 0))
        
        headers = ['File', 'Model', 'Serial', 'Status', 'Validation', 'Tracks', 'Pass/Fail']
        self.header_labels = []
        
        for i, header in enumerate(headers):
            label = ctk.CTkLabel(
                header_frame,
                text=header,
                font=ctk.CTkFont(size=12, weight="bold"),
                width=self.column_widths[header],
                anchor='w'
            )
            label.grid(row=0, column=i, padx=5, pady=5, sticky='w')
            # Don't use weight for columns - use fixed widths instead
            header_frame.columnconfigure(i, weight=0, minsize=self.column_widths[header])
            self.header_labels.append(label)
            
        # Scrollable results area
        self.results_frame = ctk.CTkScrollableFrame(self, height=300)
        self.results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configure grid columns with same widths as header
        for i, header in enumerate(headers):
            self.results_frame.columnconfigure(i, weight=0, minsize=self.column_widths[header])
            
    def display_results(self, results: Dict[str, AnalysisResult]):
        """Display batch processing results."""
        self.results = results
        
        # Clear existing results
        self.clear()
        
        # Add results
        for row, (file_path, result) in enumerate(results.items()):
            self._add_result_row(row, file_path, result)
            
    def _add_result_row(self, row: int, file_path: str, result: AnalysisResult):
        """Add a single result row."""
        file_name = Path(file_path).name
        
        # Get values safely
        model = getattr(result.metadata, 'model', 'Unknown')
        serial = getattr(result.metadata, 'serial', 'Unknown')
        
        # Status
        status = 'Unknown'
        status_color = "gray"
        if hasattr(result, 'overall_status'):
            status_value = getattr(result.overall_status, 'value', str(result.overall_status))
            status = status_value
            status_color = "green" if status == "Pass" else "red"
            
        # Validation status
        validation = 'N/A'
        validation_color = "gray"
        if hasattr(result, 'overall_validation_status'):
            val_status = result.overall_validation_status
            if hasattr(val_status, 'value'):
                validation = val_status.value
            else:
                validation = str(val_status)
                
            if validation == 'VALIDATED':
                validation_color = "green"
            elif validation == 'WARNING':
                validation_color = "orange"
            elif validation == 'FAILED':
                validation_color = "red"
                
        # Track counts
        track_count = 0
        pass_count = 0
        fail_count = 0
        
        if hasattr(result, 'tracks') and result.tracks:
            # Handle both dict (from analysis) and list (from DB) formats
            if isinstance(result.tracks, dict):
                track_count = len(result.tracks)
                tracks_iter = result.tracks.values()
            else:
                track_count = len(result.tracks)
                tracks_iter = result.tracks
                
            for track in tracks_iter:
                if hasattr(track, 'status'):
                    # Track has 'status' field, not 'overall_status'
                    track_status = getattr(track.status, 'value', str(track.status))
                    if track_status in ['Pass', 'PASS']:
                        pass_count += 1
                    elif track_status in ['Fail', 'FAIL', 'Warning', 'WARNING', 'Error', 'ERROR']:
                        fail_count += 1
                    # If status is not recognized, log it for debugging
                    else:
                        self.logger.debug(f"Unknown track status: {track_status}")
                elif hasattr(track, 'overall_status'):
                    # Analysis tracks might have 'overall_status'
                    track_status = getattr(track.overall_status, 'value', str(track.overall_status))
                    if track_status in ['Pass', 'PASS']:
                        pass_count += 1
                    elif track_status in ['Fail', 'FAIL', 'Warning', 'WARNING', 'Error', 'ERROR']:
                        fail_count += 1
                        
        # Create row widgets with fixed widths matching headers
        headers = ['File', 'Model', 'Serial', 'Status', 'Validation', 'Tracks', 'Pass/Fail']
        values = [
            file_name[:35] + "..." if len(file_name) > 35 else file_name,
            model,
            serial,
            status,
            validation,
            str(track_count),
            f"{pass_count}/{fail_count}"
        ]
        
        widgets = []
        for header, value in zip(headers, values):
            label = ctk.CTkLabel(
                self.results_frame, 
                text=value,
                width=self.column_widths[header],
                anchor='w'
            )
            # Apply color to status and validation columns
            if header == 'Status':
                label.configure(text_color=status_color)
            elif header == 'Validation':
                label.configure(text_color=validation_color)
            widgets.append(label)
        
        # Add widgets to grid
        for col, widget in enumerate(widgets):
            widget.grid(row=row, column=col, padx=5, pady=2, sticky='w')
            
    def clear(self):
        """Clear all results."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.results = {}