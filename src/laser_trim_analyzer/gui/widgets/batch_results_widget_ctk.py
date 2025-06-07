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
        # Header
        header_frame = ctk.CTkFrame(self)
        header_frame.pack(fill='x', padx=5, pady=(5, 0))
        
        headers = ['File', 'Model', 'Serial', 'Status', 'Validation', 'Tracks', 'Pass/Fail']
        self.header_labels = []
        
        for i, header in enumerate(headers):
            label = ctk.CTkLabel(
                header_frame,
                text=header,
                font=ctk.CTkFont(size=12, weight="bold")
            )
            label.grid(row=0, column=i, padx=5, pady=5, sticky='w')
            header_frame.columnconfigure(i, weight=1 if i == 0 else 0)
            self.header_labels.append(label)
            
        # Scrollable results area
        self.results_frame = ctk.CTkScrollableFrame(self, height=300)
        self.results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configure grid columns
        for i in range(len(headers)):
            self.results_frame.columnconfigure(i, weight=1 if i == 0 else 0)
            
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
        
        if hasattr(result, 'tracks'):
            track_count = len(result.tracks)
            for track in result.tracks.values():
                if hasattr(track, 'overall_status'):
                    track_status = getattr(track.overall_status, 'value', str(track.overall_status))
                    if track_status == 'Pass':
                        pass_count += 1
                    else:
                        fail_count += 1
                        
        # Create row widgets
        widgets = [
            ctk.CTkLabel(self.results_frame, text=file_name[:40] + "..." if len(file_name) > 40 else file_name),
            ctk.CTkLabel(self.results_frame, text=model),
            ctk.CTkLabel(self.results_frame, text=serial),
            ctk.CTkLabel(self.results_frame, text=status, text_color=status_color),
            ctk.CTkLabel(self.results_frame, text=validation, text_color=validation_color),
            ctk.CTkLabel(self.results_frame, text=str(track_count)),
            ctk.CTkLabel(self.results_frame, text=f"{pass_count}/{fail_count}")
        ]
        
        # Add widgets to grid
        for col, widget in enumerate(widgets):
            widget.grid(row=row, column=col, padx=5, pady=2, sticky='w')
            
    def clear(self):
        """Clear all results."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.results = {}