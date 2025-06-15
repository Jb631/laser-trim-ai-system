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
        
        # Handle empty results
        if not results:
            empty_label = ctk.CTkLabel(
                self.results_frame,
                text="No results to display",
                font=ctk.CTkFont(size=14),
                text_color="gray"
            )
            empty_label.grid(row=0, column=0, columnspan=7, pady=50)
            return
        
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
        
        # Format values with null/error handling
        try:
            file_display = file_name[:35] + "..." if file_name and len(file_name) > 35 else (file_name or "Unknown")
        except:
            file_display = "Unknown"
            
        values = [
            file_display,
            str(model) if model else "--",
            str(serial) if serial else "--",
            str(status) if status else "--",
            str(validation) if validation else "--",
            str(track_count) if track_count is not None else "0",
            f"{pass_count}/{fail_count}" if pass_count is not None and fail_count is not None else "--/--"
        ]
        
        # Create row frame for alternating colors
        row_frame = ctk.CTkFrame(
            self.results_frame,
            fg_color=("gray90", "gray20") if row % 2 == 0 else ("gray95", "gray25"),
            corner_radius=0
        )
        row_frame.grid(row=row, column=0, columnspan=len(headers), sticky='ew', padx=0, pady=1)
        
        # Configure grid weights for row frame
        for i, header in enumerate(headers):
            row_frame.columnconfigure(i, weight=0, minsize=self.column_widths[header])
        
        widgets = []
        for col, (header, value) in enumerate(zip(headers, values)):
            # Create label with proper text truncation and tooltip
            label = ctk.CTkLabel(
                row_frame, 
                text=value,
                width=self.column_widths[header],
                anchor='w',
                font=ctk.CTkFont(size=12),
                fg_color="transparent"
            )
            
            # Add tooltip for long text
            if header == 'File' and file_name and len(file_name) > 35:
                # Store full path for potential tooltip implementation
                label._full_text = file_name
                
            # Apply color to status and validation columns
            if header == 'Status':
                label.configure(text_color=status_color)
            elif header == 'Validation':
                label.configure(text_color=validation_color)
                
            label.grid(row=0, column=col, padx=5, pady=4, sticky='w')
            widgets.append(label)
        
        # Add hover effect to row
        def on_enter(event):
            row_frame.configure(fg_color=("gray85", "gray15"))
            
        def on_leave(event):
            row_frame.configure(
                fg_color=("gray90", "gray20") if row % 2 == 0 else ("gray95", "gray25")
            )
            
        row_frame.bind("<Enter>", on_enter)
        row_frame.bind("<Leave>", on_leave)
        
        # Make row clickable for potential future functionality
        row_frame.bind("<Button-1>", lambda e: self._on_row_click(file_path, result))
            
    def clear(self):
        """Clear all results."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.results = {}
    
    def _on_row_click(self, file_path: str, result: AnalysisResult):
        """Handle row click event (for future functionality)."""
        # This could be extended to show detailed view, export individual result, etc.
        self.logger.debug(f"Row clicked: {file_path}")