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
        self._shutting_down = False
        
        # Virtual scrolling parameters
        self.visible_rows = 20  # Number of rows to display at once
        self.row_height = 35  # Estimated height of each row
        self.current_offset = 0  # Current scroll offset
        self.row_widgets = []  # Keep track of created widgets
        
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
            
        # Scrollable results area - increased height for better visibility
        self.results_frame = ctk.CTkScrollableFrame(self, height=500)
        self.results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configure grid columns with same widths as header
        for i, header in enumerate(headers):
            self.results_frame.columnconfigure(i, weight=0, minsize=self.column_widths[header])
        
    def display_results(self, results: Dict[str, AnalysisResult]):
        """Display batch processing results with lazy loading."""
        if self._shutting_down:
            return
            
        self.results = results
        self.current_offset = 0
        
        # Force UI setup to be synchronous
        self._ensure_ui_ready()
        
        # Now display results
        self._display_results_internal(results)
    
    def _ensure_ui_ready(self):
        """Ensure UI is properly set up and ready for use."""
        # Check if UI exists and is valid
        ui_ready = (hasattr(self, 'results_frame') and 
                   self.results_frame is not None and 
                   self.results_frame.winfo_exists())
        
        if not ui_ready:
            self.logger.info("Setting up UI for batch results display")
            
            # Clear any existing widgets first to avoid conflicts
            try:
                for widget in list(self.winfo_children()):
                    widget.destroy()
            except Exception as e:
                self.logger.warning(f"Error clearing existing widgets: {e}")
            
            # Set up UI
            self._setup_ui()
            self.update_idletasks()
            
            # Verify setup worked
            ui_ready_after = (hasattr(self, 'results_frame') and 
                             self.results_frame is not None and 
                             self.results_frame.winfo_exists())
            
            if not ui_ready_after:
                self.logger.error("UI setup failed - results_frame not properly created")
    
    def _check_ui_ready_and_display(self, results: Dict[str, AnalysisResult], retry_count: int):
        """Check if UI is ready and display results, with retry logic."""
        max_retries = 5
        
        if hasattr(self, 'results_frame') and self.results_frame and self.results_frame.winfo_exists():
            self.logger.info(f"UI ready after {retry_count} retries - displaying results")
            self._display_results_internal(results)
        elif retry_count < max_retries:
            self.logger.info(f"UI not ready, retry {retry_count + 1}/{max_retries}")
            try:
                self.after(50, lambda: self._check_ui_ready_and_display(results, retry_count + 1))
            except:
                # Widget destroyed, stop retrying
                pass
        else:
            self.logger.error(f"UI setup failed after {max_retries} retries - forcing setup")
            self._setup_ui()
            self.update_idletasks()
            self._display_results_internal(results)
    
    def _display_results_after_ui_setup(self, results: Dict[str, AnalysisResult]):
        """Display results after ensuring UI is set up."""
        self._display_results_internal(results)
    
    def _display_results_internal(self, results: Dict[str, AnalysisResult]):
        """Internal method to display results - assumes UI is ready."""
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
        
        # Convert results to list for easier slicing
        self.results_list = list(results.items())
        self.logger.info(f"Displaying {len(self.results_list)} batch results with pagination")
        
        # Update pagination info first
        self._update_pagination_info()
        
        # Load initial visible rows
        self._update_visible_rows()
    
    def _create_header_row(self):
        """Create the header row for the results table."""
        # Skip creating header in results_frame since we have a fixed header above
        # Just configure the columns to match the fixed header
        headers = ["File", "Model", "Serial", "Status", "Validation", "Tracks", "Pass/Fail"]
        
        # Configure column weights and minimum sizes to match fixed header
        for col, header in enumerate(headers):
            self.results_frame.grid_columnconfigure(col, weight=0, minsize=self.column_widths[header])

    def _add_result_row(self, row: int, file_path: str, result: AnalysisResult):
        """Add a single result row."""
        self.logger.debug(f"Adding result row {row} for file: {file_path}")
        
        try:
            file_name = Path(file_path).name
        except Exception as e:
            self.logger.error(f"Error getting file name from {file_path}: {e}")
            file_name = str(file_path)
        
        # Check if result is valid
        if result is None:
            self.logger.warning(f"Null result for {file_name} - skipping row")
            return
            
        if not hasattr(result, 'metadata') or result.metadata is None:
            self.logger.warning(f"No metadata for {file_name} - skipping row")
            return
        
        # Get values safely
        model = getattr(result.metadata, 'model', 'Unknown')
        serial = getattr(result.metadata, 'serial', 'Unknown')
        
        # Status
        status = 'Unknown'
        status_color = "gray"
        if hasattr(result, 'overall_status') and result.overall_status is not None:
            status_value = getattr(result.overall_status, 'value', str(result.overall_status))
            status = status_value
            # Proper color mapping for all status types
            if status == "Pass":
                status_color = "green"
            elif status == "Warning":
                status_color = "orange"
            elif status in ["Fail", "Error"]:
                status_color = "red"
            else:
                status_color = "gray"
            
        # Validation status
        validation = 'N/A'
        validation_color = "gray"
        if hasattr(result, 'overall_validation_status'):
            val_status = result.overall_validation_status
            if hasattr(val_status, 'value'):
                validation = val_status.value
            else:
                validation = str(val_status)
                
            # Handle the actual enum values
            if validation.upper() in ['VALIDATED', 'VALID']:
                validation_color = "green"
            elif validation.upper() in ['WARNING', 'WARN']:
                validation_color = "orange"  
            elif validation.upper() in ['FAILED', 'FAIL', 'NOT VALIDATED']:
                validation_color = "red"
                
        # Track counts
        track_count = 0
        pass_count = 0
        fail_count = 0
        warning_count = 0
        
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
                    elif track_status in ['Warning', 'WARNING']:
                        warning_count += 1
                        # Count warnings as pass for Pass/Fail display
                        pass_count += 1
                    elif track_status in ['Fail', 'FAIL', 'Error', 'ERROR']:
                        fail_count += 1
                    # If status is not recognized, count as fail for safety
                    else:
                        fail_count += 1
                else:
                    # Fallback - count missing status as fail for safety
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
        
        # Add hover effect to row with error handling
        def on_enter(event):
            try:
                if row_frame.winfo_exists():
                    row_frame.configure(fg_color=("gray85", "gray15"))
            except:
                pass  # Widget destroyed, ignore
            
        def on_leave(event):
            try:
                if row_frame.winfo_exists():
                    row_frame.configure(
                        fg_color=("gray90", "gray20") if row % 2 == 0 else ("gray95", "gray25")
                    )
            except:
                pass  # Widget destroyed, ignore
            
        def on_click(event):
            try:
                if row_frame.winfo_exists():
                    self._on_row_click(file_path, result)
            except:
                pass  # Widget destroyed, ignore
            
        row_frame.bind("<Enter>", on_enter)
        row_frame.bind("<Leave>", on_leave)
        row_frame.bind("<Button-1>", on_click)
            
    def clear(self):
        """Clear all results."""
        # Only clear the contents of results_frame, NOT the frame itself
        if hasattr(self, 'results_frame') and self.results_frame and self.results_frame.winfo_exists():
            try:
                # Clear only the children of results_frame, not the frame itself
                for widget in list(self.results_frame.winfo_children()):
                    widget.destroy()
                self.logger.debug("Cleared results_frame contents successfully")
            except Exception as e:
                self.logger.error(f"Error clearing results frame contents: {e}")
        
        # Clear pagination controls if they exist
        if hasattr(self, 'pagination_frame') and self.pagination_frame:
            try:
                self.pagination_frame.destroy()
                self.pagination_frame = None
                self.logger.debug("Cleared pagination controls successfully")
            except Exception as e:
                self.logger.error(f"Error clearing pagination controls: {e}")
        
        # Clear individual pagination control references  
        for attr in ['status_label', 'prev_button', 'next_button']:
            if hasattr(self, attr):
                setattr(self, attr, None)
        
        # Reset internal state
        self.results = {}
        self.current_offset = 0
        if hasattr(self, 'results_list'):
            self.results_list = []
    
    def _on_row_click(self, file_path: str, result: AnalysisResult):
        """Handle row click event (for future functionality)."""
        # This could be extended to show detailed view, export individual result, etc.
        self.logger.debug(f"Row clicked: {file_path}")
    
    def _update_pagination_info(self):
        """Update the pagination status label and controls."""
        if self._shutting_down:
            return
        if hasattr(self, 'results_list') and self.results_list:
            # Remove existing pagination controls if they exist
            if hasattr(self, 'pagination_frame') and self.pagination_frame:
                self.pagination_frame.destroy()
            
            # Create pagination frame
            self.pagination_frame = ctk.CTkFrame(self, fg_color="transparent")
            self.pagination_frame.pack(pady=(10, 0), fill='x')
            
            # Calculate pagination info
            total_results = len(self.results_list)
            start_showing = self.current_offset + 1
            end_showing = min(self.current_offset + self.visible_rows, total_results)
            total_pages = (total_results + self.visible_rows - 1) // self.visible_rows  # Ceiling division
            current_page = (self.current_offset // self.visible_rows) + 1
            
            # Previous button
            self.prev_button = ctk.CTkButton(
                self.pagination_frame,
                text="← Previous",
                width=100,
                height=30,
                command=self._previous_page,
                state="normal" if self.current_offset > 0 else "disabled"
            )
            self.prev_button.pack(side='left', padx=(10, 5))
            
            # Status label (centered)
            status_text = f"Showing {start_showing}-{end_showing} of {total_results} results (Page {current_page} of {total_pages})"
            self.status_label = ctk.CTkLabel(
                self.pagination_frame,
                text=status_text,
                font=ctk.CTkFont(size=11),
                text_color="gray"
            )
            self.status_label.pack(side='left', expand=True)
            
            # Next button  
            self.next_button = ctk.CTkButton(
                self.pagination_frame,
                text="Next →",
                width=100,
                height=30,
                command=self._next_page,
                state="normal" if end_showing < total_results else "disabled"
            )
            self.next_button.pack(side='right', padx=(5, 10))

    def _update_visible_rows(self):
        """Update visible rows based on current offset."""
        if self._shutting_down:
            return
        # Ensure we have results to display
        if not hasattr(self, 'results_list') or not self.results_list:
            self.logger.warning("No results list available for display")
            return
            
        # Detailed results_frame checking with specific error messages
        if not hasattr(self, 'results_frame'):
            self.logger.error("results_frame attribute missing completely")
            return
        elif self.results_frame is None:
            self.logger.error("results_frame is None")
            return
        elif not self.results_frame.winfo_exists():
            self.logger.error("results_frame doesn't exist (winfo_exists returned False)")
            return
            
        # Clear current row widgets
        try:
            widgets_to_clear = list(self.results_frame.winfo_children())
            for widget in widgets_to_clear:
                widget.destroy()
        except Exception as e:
            self.logger.error(f"Error clearing widgets: {e}")
            return
            
        self.row_widgets = []
        
        # Calculate visible range
        start_idx = self.current_offset
        end_idx = min(start_idx + self.visible_rows, len(self.results_list))
        
        # Add header row first
        try:
            self._create_header_row()
        except Exception as e:
            self.logger.error(f"Error creating header row: {e}")
            return
        
        # Add result rows
        try:
            for i in range(start_idx, end_idx):
                file_path, result = self.results_list[i]
                row_num = i - start_idx  # No +1 since no header row in results_frame
                self._add_result_row(row_num, file_path, result)
            self.logger.info(f"Successfully displayed {end_idx - start_idx} result rows (page {(start_idx // self.visible_rows) + 1})")
        except Exception as e:
            self.logger.error(f"Error adding result rows: {e}")
            return
        
        # Force update
        try:
            self.update_idletasks()
        except Exception as e:
            self.logger.error(f"Error updating UI: {e}")

    def _next_page(self):
        """Navigate to the next page of results."""
        if self._shutting_down:
            return
        if hasattr(self, 'results_list') and self.results_list:
            max_offset = len(self.results_list) - self.visible_rows
            if self.current_offset < max_offset:
                self.current_offset += self.visible_rows
                if self.current_offset > max_offset:
                    self.current_offset = max_offset
                self._update_display()
    
    def _previous_page(self):
        """Navigate to the previous page of results.""" 
        if self._shutting_down:
            return
        if self.current_offset > 0:
            self.current_offset -= self.visible_rows
            if self.current_offset < 0:
                self.current_offset = 0
            self._update_display()
    
    def _update_display(self):
        """Update both the pagination info and visible rows."""
        if self._shutting_down:
            return
        self._update_pagination_info()
        self._update_visible_rows()
    
    def cleanup(self):
        """Cleanup method to prevent callback errors during shutdown."""
        self._shutting_down = True
        # Clear results to stop any ongoing operations
        self.results = {}
        if hasattr(self, 'results_list'):
            self.results_list = []