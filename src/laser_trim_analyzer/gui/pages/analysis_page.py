"""
Analysis Page for Laser Trim Analyzer

Provides interface for file selection, processing options,
and analysis execution with results display.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import asyncio
import threading
import time as time_module
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from laser_trim_analyzer.core.models import AnalysisResult, ProcessingMode
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.core.exceptions import ProcessingError
from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.file_drop_zone import FileDropZone
from laser_trim_analyzer.gui.widgets.file_analysis_widget import FileAnalysisWidget
from laser_trim_analyzer.gui.widgets import add_mousewheel_support


class AnalysisPage(BasePage):
    """
    Analysis page for processing laser trim files.

    Features:
    - Drag-and-drop file selection
    - Real-time processing progress
    - ML insights display
    - Result visualization
    """

    def __init__(self, parent: ttk.Frame, main_window: Any):
        """Initialize analysis page."""
        # Initialize state with thread safety
        self.input_files: List[Path] = []
        self.file_widgets: Dict[str, FileAnalysisWidget] = {}
        self.processor: Optional[LaserTrimProcessor] = None
        self.is_processing = False
        self.current_task = None
        
        # File state management
        self._file_selection_lock = threading.Lock()
        self._file_metadata_cache = {}  # Cache file metadata to prevent loss
        self._processing_results = {}  # Store results by file path
        
        # Progress update throttling
        self.last_progress_update = 0
        self.progress_update_interval = 0.05  # Faster updates for responsiveness

        # Processing options with safe defaults
        self.processing_mode = tk.StringVar(value='detail')
        self.enable_plots = tk.BooleanVar(value=True)
        self.enable_ml = tk.BooleanVar(value=True)
        self.enable_database = tk.BooleanVar(value=True)

        # UI components
        self.drop_zone = None
        self.file_list_frame = None
        self.alert_stack = None
        self.progress_frame = None
        self.results_notebook = None
        
        # UI state management
        self._ui_update_lock = threading.Lock()
        self._last_ui_update = 0

        super().__init__(parent, main_window)

    def _create_page(self):
        """Create analysis page content with consistent theme (matching batch processing)."""
        # Main scrollable container (matching batch processing theme)
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create sections in order (matching batch processing pattern)
        self._create_header()
        self._create_file_section()
        self._create_options_section()
        self._create_results_section()

    def _create_header(self):
        """Create header section (matching batch processing theme)."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="File Analysis",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)

    def _create_file_section(self):
        """Create file selection section (matching batch processing theme)."""
        self.file_frame = ctk.CTkFrame(self.main_container)
        self.file_frame.pack(fill='x', pady=(0, 20))

        self.file_label = ctk.CTkLabel(
            self.file_frame,
            text="File Selection:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.file_label.pack(anchor='w', padx=15, pady=(15, 10))

        # File selection container
        self.file_container = ctk.CTkFrame(self.file_frame)
        self.file_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Create drop zone
        self.drop_zone = FileDropZone(
            self.file_container,
            accept_extensions=['.xlsx', '.xls'],
            on_files_dropped=self._handle_files_dropped,
            height=160
        )
        self.drop_zone.pack(fill='x', padx=10, pady=10)

        # File list display
        self.file_list_label = ctk.CTkLabel(
            self.file_container,
            text="Selected Files (0):",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.file_list_label.pack(anchor='w', padx=10, pady=(10, 5))

        self.file_listbox = ctk.CTkTextbox(
            self.file_container,
            height=150,
            state="disabled"
        )
        self.file_listbox.pack(fill='x', padx=10, pady=(0, 10))

    def _create_options_section(self):
        """Create processing options section (matching batch processing theme)."""
        self.options_frame = ctk.CTkFrame(self.main_container)
        self.options_frame.pack(fill='x', pady=(0, 20))

        self.options_label = ctk.CTkLabel(
            self.options_frame,
            text="Processing Options:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.options_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Options container
        self.options_container = ctk.CTkFrame(self.options_frame)
        self.options_container.pack(fill='x', padx=15, pady=(0, 15))

        # Processing mode
        mode_frame = ctk.CTkFrame(self.options_container)
        mode_frame.pack(fill='x', padx=10, pady=(10, 5))

        mode_label = ctk.CTkLabel(mode_frame, text="Processing Mode:")
        mode_label.pack(side='left', padx=10, pady=10)

        self.mode_combo = ctk.CTkComboBox(
            mode_frame,
            variable=self.processing_mode,
            values=["detail", "summary", "quick"],
            width=120,
            height=30
        )
        self.mode_combo.pack(side='left', padx=10, pady=10)

        # Checkboxes
        self.enable_plots_check = ctk.CTkCheckBox(
            self.options_container,
            text="Generate Plots",
            variable=self.enable_plots
        )
        self.enable_plots_check.pack(anchor='w', padx=10, pady=5)

        self.enable_ml_check = ctk.CTkCheckBox(
            self.options_container,
            text="Enable ML Analysis",
            variable=self.enable_ml
        )
        self.enable_ml_check.pack(anchor='w', padx=10, pady=5)

        self.enable_database_check = ctk.CTkCheckBox(
            self.options_container,
            text="Save to Database",
            variable=self.enable_database
        )
        self.enable_database_check.pack(anchor='w', padx=10, pady=5)

        # Control buttons
        controls_frame = ctk.CTkFrame(self.options_container)
        controls_frame.pack(fill='x', padx=10, pady=(10, 10))

        self.start_button = ctk.CTkButton(
            controls_frame,
            text="Start Analysis",
            command=self._start_analysis,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.start_button.pack(side='left', padx=(10, 10), pady=10)

        self.clear_button = ctk.CTkButton(
            controls_frame,
            text="Clear Files",
            command=self._clear_files,
            width=120,
            height=40
        )
        self.clear_button.pack(side='left', padx=(0, 10), pady=10)

    def _create_results_section(self):
        """Create results display section (matching batch processing theme)."""
        self.results_frame = ctk.CTkFrame(self.main_container)
        self.results_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="Analysis Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.results_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Results display area
        self.results_display = ctk.CTkTextbox(
            self.results_frame,
            height=200,
            state="disabled"
        )
        self.results_display.pack(fill='both', expand=True, padx=15, pady=(0, 15))

    def _handle_files_dropped(self, files: List[str]):
        """Handle files dropped in the drop zone."""
        self._add_files([Path(f) for f in files])

    def _add_files(self, files: List[Path]):
        """Add files to the analysis list with proper CTk widget updates."""
        with self._file_selection_lock:
            # Filter valid Excel files
            valid_files = []
            for file_path in files:
                if file_path.exists() and file_path.suffix.lower() in ['.xlsx', '.xls']:
                    if file_path not in self.input_files:
                        valid_files.append(file_path)
                        self.input_files.append(file_path)
            
            if valid_files:
                # Update file listbox display
                self._update_file_display()
                
                # Update UI state
                self._update_ui_state()
                self._update_stats()
                
                logger.info(f"Added {len(valid_files)} files to analysis")

    def _update_file_display(self):
        """Update the file display in the CTk textbox."""
        try:
            self.file_listbox.configure(state="normal")
            self.file_listbox.delete('1.0', 'end')
            
            for i, file_path in enumerate(self.input_files, 1):
                self.file_listbox.insert('end', f"{i}. {file_path.name}\n")
            
            if not self.input_files:
                self.file_listbox.insert('end', "No files loaded")
                
            self.file_listbox.configure(state="disabled")
        except Exception as e:
            logger.error(f"Error updating file display: {e}")

    def _clear_files(self):
        """Clear all selected files."""
        with self._file_selection_lock:
            self.input_files.clear()
            self.file_widgets.clear()
            self._file_metadata_cache.clear()
            self._processing_results.clear()
            
            # Update displays
            self._update_file_display()
            self._update_ui_state()
            self._update_stats()
            
            # Clear results
            self.results_display.configure(state="normal")
            self.results_display.delete('1.0', 'end')
            self.results_display.insert('end', "No analysis results yet")
            self.results_display.configure(state="disabled")
            
            logger.info("Cleared all files and results")

    def _update_ui_state(self):
        """Update UI state based on current files and processing status."""
        has_files = len(self.input_files) > 0
        
        # Update buttons
        if hasattr(self, 'start_button'):
            self.start_button.configure(
                state="normal" if has_files and not self.is_processing else "disabled"
            )
        
        if hasattr(self, 'clear_button'):
            self.clear_button.configure(
                state="normal" if has_files else "disabled"
            )

    def _update_stats(self):
        """Update file statistics display."""
        num_files = len(self.input_files)
        if hasattr(self, 'file_list_label'):
            self.file_list_label.configure(text=f"Selected Files ({num_files}):")

    def _show_results_responsive(self, results: List[AnalysisResult]):
        """Display analysis results in the results textbox."""
        try:
            self.results_display.configure(state="normal")
            self.results_display.delete('1.0', 'end')
            
            # Display summary
            total_files = len(results)
            passed_files = sum(1 for r in results if r.overall_status.value == "Pass")
            
            summary = f"Analysis Complete!\n\n"
            summary += f"Total Files: {total_files}\n"
            summary += f"Passed: {passed_files}\n"
            summary += f"Failed: {total_files - passed_files}\n"
            summary += f"Pass Rate: {(passed_files/total_files*100):.1f}%\n\n"
            
            # Display individual results
            summary += "Individual Results:\n"
            summary += "-" * 50 + "\n"
            
            for i, result in enumerate(results, 1):
                summary += f"{i}. {result.filename}\n"
                summary += f"   Status: {result.overall_status.value}\n"
                if result.tracks:
                    track = result.tracks[0]
                    summary += f"   Sigma: {track.sigma_gradient:.4f}\n"
                    summary += f"   Pass: {'✓' if track.sigma_pass else '✗'}\n"
                summary += "\n"
            
            self.results_display.insert('1.0', summary)
            self.results_display.configure(state="disabled")
            
        except Exception as e:
            logger.error(f"Error displaying results: {e}")

    def _processing_complete_responsive(self, results: List[AnalysisResult]):
        """Handle processing completion with CTk updates."""
        self.is_processing = False
        
        # Update UI state
        self._update_ui_state()
        
        # Show results
        self._show_results_responsive(results)
        
        # Show completion message
        messagebox.showinfo(
            "Analysis Complete",
            f"Successfully analyzed {len(results)} files!"
        )
        
        logger.info(f"Analysis completed for {len(results)} files")

    def _processing_error_responsive(self, error: str):
        """Handle processing error with CTk updates."""
        self.is_processing = False
        
        # Update UI state
        self._update_ui_state()
        
        # Show error in results
        self.results_display.configure(state="normal")
        self.results_display.delete('1.0', 'end')
        self.results_display.insert('1.0', f"Analysis Failed:\n\n{error}")
        self.results_display.configure(state="disabled")
        
        # Show error dialog
        messagebox.showerror("Analysis Failed", f"Processing failed:\n{error}")
        
        logger.error(f"Analysis failed: {error}")

    def _start_analysis(self):
        """Start processing selected files with enhanced responsiveness."""
        if not self.input_files or self.is_processing:
            return

        self.is_processing = True
        self._update_ui_state()

        # Show immediate responsive feedback
        self.alert_stack.add_alert(
            alert_type='info',
            title='Starting Analysis',
            message=f'Initializing analysis for {len(self.input_files)} files...',
            dismissible=False,
            allow_scroll=True  # Keep app scrollable during processing
        )

        # Show progress with responsive design
        self.progress_frame.pack(fill='x', pady=(10, 0))
        self.progress_var.set(0)
        self.progress_label.config(text="Preparing analysis environment...")

        # Clear previous alerts but keep critical ones
        self._clear_non_critical_alerts()
        
        # Ensure files remain visible during processing with immediate feedback
        self._ensure_files_visible_responsive()

        # Start processing in background thread with responsive UI updates
        thread = threading.Thread(target=self._process_files_thread_responsive, daemon=True)
        thread.start()

        # Schedule UI responsiveness checks
        self._schedule_responsiveness_checks()

    def _clear_non_critical_alerts(self):
        """Clear non-critical alerts while preserving important ones."""
        try:
            # With simplified alerts, just clear all alerts for a clean slate
            # The simplified system doesn't distinguish between alert types
            self.alert_stack.clear_all()
        except Exception as e:
            self.logger.warning(f"Error clearing alerts: {e}")

    def _ensure_files_visible_responsive(self):
        """Ensure selected files remain visible during processing with responsive updates."""
        try:
            total_files = len(self.file_widgets)
            processed_count = 0
            
            # Update files in batches to maintain responsiveness
            for file_path, widget_data in self.file_widgets.items():
                if isinstance(widget_data, dict) and widget_data.get('tree_mode'):
                    # Tree view mode - ensure item is visible
                    item_id = widget_data['tree_item']
                    if hasattr(self, 'file_tree') and self.file_tree.exists(item_id):
                        current_values = list(self.file_tree.item(item_id, 'values'))
                        current_values[2] = 'Ready'  # Status column
                        self.file_tree.item(item_id, values=current_values)
                        self.file_tree.item(item_id, tags=('ready',))
                else:
                    # Individual widget mode
                    if hasattr(widget_data, 'update_data'):
                        widget_data.update_data({'status': 'Ready'})
                
                processed_count += 1
                
                # Update UI every 10 files to maintain responsiveness
                if processed_count % 10 == 0:
                    self.update_idletasks()
                        
            # Configure tree view colors for processing states
            if hasattr(self, 'file_tree'):
                self.file_tree.tag_configure('ready', foreground='blue')
                self.file_tree.tag_configure('processing', foreground='orange')
                self.file_tree.tag_configure('completed', foreground='green')
                self.file_tree.tag_configure('error', foreground='red')
                
        except Exception as e:
            self.logger.warning(f"Error ensuring files visible: {e}")

    def _schedule_responsiveness_checks(self):
        """Schedule periodic UI responsiveness checks during processing."""
        def check_responsiveness():
            if self.is_processing:
                # Update UI to ensure responsiveness
                try:
                    self.update_idletasks()
                    # Schedule next check
                    self.after(500, check_responsiveness)  # Check every 500ms
                except:
                    pass  # If UI is being destroyed, stop checks
                    
        # Start responsiveness checks
        self.after(500, check_responsiveness)

    def _process_files_thread_responsive(self):
        """Background thread for file processing with responsive UI updates."""
        try:
            # Create event loop for async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run async processing with responsive updates
            results = loop.run_until_complete(self._process_files_async_responsive())

            # Update UI in main thread
            self.after(0, self._processing_complete_responsive, results)

        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            self.after(0, self._processing_error_responsive, str(e))

        finally:
            loop.close()

    async def _process_files_async_responsive(self) -> List[AnalysisResult]:
        """Async method to process files with responsive UI updates."""
        # Initialize processor with progress feedback
        self.after(0, self._update_progress_responsive, 5, "Initializing processor...")
        
        try:
            self.processor = LaserTrimProcessor(
                config=self.config,
                db_manager=self.db_manager if self.enable_database.get() else None,
                ml_predictor=self.main_window.ml_predictor if self.enable_ml.get() else None,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize processor: {e}")
            raise ProcessingError(f"Processor initialization failed: {e}")

        # Configure based on options
        self.config.processing.generate_plots = (
                self.enable_plots.get() and self.processing_mode.get() == 'detail'
        )

        # Create output directory with feedback
        self.after(0, self._update_progress_responsive, 10, "Creating output directory...")
        try:
            output_dir = self.config.data_directory / datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise ProcessingError(f"Output directory creation failed: {e}")

        results = []
        total_files = len(self.input_files)
        
        # Validate we have files to process
        if total_files == 0:
            raise ProcessingError("No files selected for processing")
        
        # Update initial progress
        self.after(0, self._update_progress_responsive, 15, f"Starting analysis of {total_files} files...")

        # Process files with enhanced error handling and state preservation
        for i, file_path in enumerate(self.input_files):
            # Calculate progress with proper scaling
            base_progress = 15 + (i / total_files) * 80  # Scale to 15-95%
            
            # Update progress with responsive feedback
            self.after(0, self._update_progress_responsive, 
                      base_progress, f"Processing {file_path.name} ({i+1}/{total_files})...")

            # Update file widget status immediately with state preservation
            self.after(0, self._update_file_status_responsive, str(file_path), 'Processing')

            try:
                # Process file with responsive progress callbacks
                result = await self.processor.process_file(
                    file_path,
                    output_dir / file_path.stem,
                    progress_callback=lambda msg, prog: self.after(
                        0, self._update_progress_responsive,
                        base_progress + (prog * 0.8),  # Scale sub-progress within file progress
                        f"{file_path.name}: {msg}"
                    )
                )

                # Store result to prevent loss
                self._processing_results[str(file_path)] = result

                # Save to database if enabled with non-blocking feedback
                if self.enable_database.get() and self.db_manager:
                    try:
                        self.after(0, self._update_progress_responsive, 
                                  base_progress + 70, f"Saving {file_path.name} to database...")
                        
                        # Check for duplicates first
                        existing_id = self.db_manager.check_duplicate_analysis(
                            result.metadata.model,
                            result.metadata.serial,
                            result.metadata.file_date
                        )
                        
                        if existing_id:
                            self.logger.info(f"Duplicate analysis found for {file_path.name} (ID: {existing_id})")
                            result.db_id = existing_id
                        else:
                            # Try normal save first
                            try:
                                result.db_id = self.db_manager.save_analysis(result)
                                self.logger.info(f"Saved analysis to database with ID: {result.db_id}")
                                
                                # Validate the save
                                if not self.db_manager.validate_saved_analysis(result.db_id):
                                    raise RuntimeError("Database validation failed")
                                    
                            except Exception as save_error:
                                self.logger.warning(f"Normal save failed, trying force save: {save_error}")
                                # Try force save as fallback
                                result.db_id = self.db_manager.force_save_analysis(result)
                                self.logger.info(f"Force saved analysis to database with ID: {result.db_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Database save failed for {file_path.name}: {e}")
                        # Show non-blocking warning
                        self.after(0, self._show_non_blocking_warning, file_path.name, "Database save failed")

                results.append(result)

                # Update file widget with results (responsive)
                self.after(0, self._update_file_widget_responsive, str(file_path), result)

                # Allow UI breathing room between files
                await asyncio.sleep(0.1)  # 100ms pause between files

            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {e}")
                self.after(0, self._update_file_status_responsive, str(file_path), 'Error')
                self.after(0, self._show_file_error_responsive, file_path.name, str(e))
                
                # Store error state to preserve file information
                self._processing_results[str(file_path)] = {
                    'error': str(e),
                    'status': 'Error',
                    'file_path': str(file_path)
                }

        # Final progress update
        self.after(0, self._update_progress_responsive, 95, "Finalizing results...")
        
        return results

    def _update_progress_responsive(self, value: float, text: str):
        """Update progress display with enhanced responsiveness."""
        current_time = time_module.time()
        
        # More responsive throttling for better user feedback
        if current_time - self.last_progress_update < 0.05:  # 50ms instead of 100ms
            return
            
        self.last_progress_update = current_time
        
        try:
            self.progress_var.set(min(100, max(0, value)))  # Clamp between 0-100
            self.progress_label.config(text=text)
            
            # Force immediate UI update for responsiveness
            self.progress_bar.update_idletasks()
            self.progress_label.update_idletasks()
            
        except Exception as e:
            self.logger.warning(f"Progress update error: {e}")

    def _update_file_status_responsive(self, file_path: str, status: str):
        """Update file widget status with responsive feedback and state preservation."""
        try:
            # Update cached metadata to preserve state
            if file_path in self._file_metadata_cache:
                self._file_metadata_cache[file_path]['status'] = status
                self._file_metadata_cache[file_path]['last_updated'] = time_module.time()
            
            if file_path in self.file_widgets:
                widget_data = self.file_widgets[file_path]
                if isinstance(widget_data, dict) and widget_data.get('tree_mode'):
                    # Tree view mode
                    item_id = widget_data['tree_item']
                    if hasattr(self, 'file_tree') and self.file_tree.exists(item_id):
                        current_values = list(self.file_tree.item(item_id, 'values'))
                        current_values[2] = status  # Status is the 3rd column
                        self.file_tree.item(item_id, values=current_values)
                        
                        # Update tag for color coding with immediate visual feedback
                        status_tag = status.lower()
                        self.file_tree.item(item_id, tags=(status_tag,))
                        
                        # Ensure tree view updates immediately
                        self.file_tree.update_idletasks()
                else:
                    # Individual widget mode
                    if hasattr(widget_data, 'update_data'):
                        widget_data.update_data({'status': status})
                        # Force widget update
                        widget_data.update_idletasks()
                        
        except Exception as e:
            self.logger.warning(f"File status update error: {e}")

    def _update_file_widget_responsive(self, file_path: str, result: AnalysisResult):
        """Update file widget with analysis results using responsive updates and state preservation."""
        try:
            # Store result in cache for state preservation
            self._processing_results[file_path] = result
            
            # Update cached metadata
            if file_path in self._file_metadata_cache:
                self._file_metadata_cache[file_path].update({
                    'status': 'Completed',
                    'result': result,
                    'completed_time': time_module.time()
                })
            
            if file_path in self.file_widgets:
                widget_data = self.file_widgets[file_path]
                
                if isinstance(widget_data, dict) and widget_data.get('tree_mode'):
                    # Tree view mode - update tree item responsively
                    item_id = widget_data['tree_item']
                    primary_track = result.primary_track
                    
                    # Update tree item values
                    self.file_tree.item(item_id, values=(
                        result.metadata.model,
                        result.metadata.serial,
                        'Completed'  # Mark as completed after processing
                    ))
                    
                    # Update tag for color coding
                    status_tag = 'completed'
                    self.file_tree.item(item_id, tags=(status_tag,))
                    
                    # Store result data for context menu access
                    widget_data['result'] = result
                    
                    # Force immediate tree update
                    self.file_tree.update_idletasks()
                    
                else:
                    # Individual widget mode - use existing logic with responsiveness
                    primary_track = result.primary_track

                    widget_update_data = {
                        'filename': result.metadata.filename,
                        'model': result.metadata.model,
                        'serial': result.metadata.serial,
                        'status': 'Completed',  # Mark as completed
                        'timestamp': datetime.now(),
                        'has_multi_tracks': result.metadata.has_multi_tracks,
                        'sigma_gradient': primary_track.sigma_analysis.sigma_gradient,
                        'sigma_pass': primary_track.sigma_analysis.sigma_pass,
                        'linearity_pass': primary_track.linearity_analysis.linearity_pass,
                        'risk_category': primary_track.failure_prediction.risk_category.value if primary_track.failure_prediction else 'Unknown',
                        'plot_path': primary_track.plot_path
                    }

                    # Add tracks for multi-track files
                    if result.metadata.has_multi_tracks:
                        widget_update_data['tracks'] = {}
                        for track_id, track in result.tracks.items():
                            widget_update_data['tracks'][track_id] = {
                                'status': self._determine_track_status(track),
                                'sigma_gradient': track.sigma_analysis.sigma_gradient,
                                'sigma_pass': track.sigma_analysis.sigma_pass,
                                'linearity_pass': track.linearity_analysis.linearity_pass,
                                'risk_category': track.failure_prediction.risk_category.value if track.failure_prediction else 'Unknown'
                            }

                    widget_data.update_data(widget_update_data)
                    
                    # Force widget update
                    widget_data.update_idletasks()
                    
        except Exception as e:
            self.logger.warning(f"File widget update error: {e}")

    def _determine_track_status(self, track) -> str:
        """Determine track status for display."""
        if not track.sigma_analysis.sigma_pass or not track.linearity_analysis.linearity_pass:
            return 'Fail'
        elif track.failure_prediction and track.failure_prediction.risk_category.value == 'High':
            return 'Warning'
        else:
            return 'Pass'

    def _show_non_blocking_warning(self, filename: str, message: str):
        """Show a non-blocking warning that doesn't interrupt processing."""
        try:
            self.alert_stack.add_alert(
                alert_type='warning',
                title=f'Warning: {filename}',
                message=message,
                auto_dismiss=8,  # Auto-dismiss to avoid UI clutter
                allow_scroll=True,  # Keep app responsive
                dismissible=True
            )
        except Exception as e:
            self.logger.warning(f"Error showing warning: {e}")

    def _show_file_error_responsive(self, filename: str, error: str):
        """Show error alert for file with responsive design."""
        try:
            # Truncate very long error messages for better UX
            display_error = error[:150] + "..." if len(error) > 150 else error
            
            self.alert_stack.add_alert(
                alert_type='error',
                title=f'Error: {filename}',
                message=display_error,
                dismissible=True,
                allow_scroll=True,  # Keep app scrollable
                actions=[
                    {'text': 'View Details', 'command': lambda: self._show_detailed_error(filename, error)},
                    {'text': 'Continue', 'command': lambda: None}  # Just dismiss
                ]
            )
        except Exception as e:
            self.logger.warning(f"Error showing file error: {e}")

    def _show_detailed_error(self, filename: str, error: str):
        """Show detailed error information in a separate dialog."""
        try:
            dialog = tk.Toplevel(self.winfo_toplevel())
            dialog.title(f"Error Details - {filename}")
            dialog.geometry("600x400")
            dialog.configure(bg=self.colors['bg'])
            
            # Error details text
            text_frame = ttk.Frame(dialog)
            text_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            error_text = tk.Text(text_frame, wrap='word', font=('Consolas', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=error_text.yview)
            error_text.configure(yscrollcommand=scrollbar.set)
            
            error_text.insert('1.0', f"File: {filename}\n\nError Details:\n{error}")
            error_text.config(state='disabled')
            
            error_text.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
            
            # Close button
            ttk.Button(
                dialog,
                text="Close",
                command=dialog.destroy
            ).pack(pady=(0, 20))
            
        except Exception as e:
            self.logger.error(f"Error showing detailed error dialog: {e}")

    def _prepare_new_analysis(self):
        """Prepare for a new analysis with responsive cleanup and state preservation."""
        try:
            # Clear current results
            self._show_empty_results()
            
            # Clear alerts except critical ones
            self._clear_non_critical_alerts()
            
            # Reset file statuses to ready while preserving file selection
            for file_path, widget_data in self.file_widgets.items():
                # Update cached metadata
                if file_path in self._file_metadata_cache:
                    self._file_metadata_cache[file_path]['status'] = 'Ready'
                    self._file_metadata_cache[file_path]['last_reset'] = time_module.time()
                
                if isinstance(widget_data, dict) and widget_data.get('tree_mode'):
                    item_id = widget_data['tree_item']
                    if hasattr(self, 'file_tree') and self.file_tree.exists(item_id):
                        current_values = list(self.file_tree.item(item_id, 'values'))
                        current_values[2] = 'Ready'  # Reset status
                        self.file_tree.item(item_id, values=current_values)
                        self.file_tree.item(item_id, tags=('ready',))
                else:
                    if hasattr(widget_data, 'update_data'):
                        widget_data.update_data({'status': 'Ready'})
            
            # Clear processing results but keep file metadata
            self._processing_results.clear()
            
            # Show ready message
            self.alert_stack.add_alert(
                alert_type='info',
                title='Ready for New Analysis',
                message='Files have been reset and are ready for analysis.',
                auto_dismiss=3,
                allow_scroll=True
            )
            
        except Exception as e:
            self.logger.error(f"Error preparing new analysis: {e}")

    def _retry_analysis(self):
        """Retry the analysis with current settings."""
        try:
            if self.input_files:
                self._start_analysis()
            else:
                self.alert_stack.add_alert(
                    alert_type='warning',
                    title='No Files Selected',
                    message='Please select files before starting analysis.',
                    auto_dismiss=5
                )
        except Exception as e:
            self.logger.error(f"Error retrying analysis: {e}")

    def _show_help_dialog(self):
        """Show help dialog for troubleshooting."""
        try:
            dialog = tk.Toplevel(self.winfo_toplevel())
            dialog.title("Analysis Help")
            dialog.geometry("500x400")
            dialog.configure(bg=self.colors['bg'])
            
            # Help content
            help_text = """
Analysis Troubleshooting Guide:

1. File Format Issues:
   • Ensure files are valid Excel (.xlsx, .xls) format
   • Check that files contain the expected data structure
   • Verify file permissions allow reading

2. Processing Errors:
   • Check available disk space for output files
   • Ensure database connection (if enabled)
   • Verify ML predictor is loaded correctly

3. Performance Issues:
   • Reduce batch size for large file sets
   • Disable plot generation for faster processing
   • Close other applications to free memory

4. Common Solutions:
   • Restart the application
   • Check log files for detailed error information
   • Update file paths if files have moved

Contact support if issues persist.
            """
            
            text_widget = tk.Text(dialog, wrap='word', font=('Segoe UI', 10), padx=20, pady=20)
            text_widget.insert('1.0', help_text.strip())
            text_widget.config(state='disabled')
            text_widget.pack(fill='both', expand=True)
            
            # Close button
            ttk.Button(
                dialog,
                text="Close",
                command=dialog.destroy
            ).pack(pady=(0, 20))
            
        except Exception as e:
            self.logger.error(f"Error showing help dialog: {e}")

    def _cancel_analysis(self):
        """Cancel ongoing analysis with responsive feedback."""
        if self.current_task:
            # TODO: Implement proper cancellation
            pass

        self.is_processing = False
        self._update_ui_state()
        
        # Hide progress smoothly
        self.after(100, lambda: self.progress_frame.pack_forget())

        # Show responsive cancellation feedback
        self.alert_stack.add_alert(
            alert_type='warning',
            title='Analysis Cancelled',
            message='Processing was cancelled by user. Files remain loaded for retry.',
            auto_dismiss=5,
            allow_scroll=True,
            actions=[
                {'text': 'Restart Analysis', 'command': lambda: self._start_analysis()},
                {'text': 'Clear Files', 'command': lambda: self._clear_files()}
            ]
        )

    def _dismiss_large_batch_alert(self):
        """Dismiss the large batch mode alert."""
        try:
            # Find and dismiss any alerts with "Large Batch Mode" in the title
            # Since we simplified the alerts, we'll just clear all info alerts
            self.alert_stack.clear_all()
        except Exception as e:
            self.logger.error(f"Error dismissing large batch alert: {e}")

class SimpleAlertStack(ttk.Frame):
    """Simplified alert stack without complex animations that cause glitching."""

    def __init__(self, parent, max_alerts: int = 3, **kwargs):
        """Initialize simplified alert stack."""
        super().__init__(parent, **kwargs)
        
        self.max_alerts = max_alerts
        self.alerts: List[tk.Frame] = []
        self.configure(relief='flat', borderwidth=0)

    def add_alert(self, alert_type: str = 'info',
                  title: str = "", message: str = "",
                  dismissible: bool = True,
                  auto_dismiss: Optional[int] = None,
                  actions: Optional[List[dict]] = None,
                  allow_scroll: bool = True) -> Optional[tk.Frame]:
        """Add a simple alert without complex animations."""
        try:
            # Remove oldest alert if at max capacity
            if len(self.alerts) >= self.max_alerts:
                oldest_alert = self.alerts[0]
                self._remove_alert(oldest_alert)

            # Create simple alert frame
            alert = self._create_simple_alert(
                alert_type, title, message, dismissible, auto_dismiss, actions
            )
            
            if alert:
                self.alerts.append(alert)
                alert.pack(fill='x', pady=(0, 5))
                
                # Auto-dismiss if requested
                if auto_dismiss:
                    self.after(auto_dismiss * 1000, lambda: self._remove_alert(alert))
            
            return alert
            
        except Exception as e:
            print(f"Error adding alert: {e}")
            return None

    def _create_simple_alert(self, alert_type: str, title: str, message: str, 
                           dismissible: bool, auto_dismiss: Optional[int], 
                           actions: Optional[List[dict]]) -> Optional[tk.Frame]:
        """Create a simple alert frame without animations."""
        try:
            # Color scheme
            colors = {
                'info': {'bg': '#d1ecf1', 'fg': '#0c5460', 'border': '#bee5eb'},
                'warning': {'bg': '#fff3cd', 'fg': '#856404', 'border': '#ffeaa7'},
                'error': {'bg': '#f8d7da', 'fg': '#721c24', 'border': '#f5c6cb'},
                'success': {'bg': '#d4edda', 'fg': '#155724', 'border': '#c3e6cb'}
            }
            
            color = colors.get(alert_type, colors['info'])
            
            # Main alert frame
            alert_frame = tk.Frame(
                self, 
                bg=color['bg'], 
                relief='solid', 
                bd=1,
                highlightbackground=color['border'],
                highlightthickness=1
            )
            
            # Content frame
            content_frame = tk.Frame(alert_frame, bg=color['bg'])
            content_frame.pack(fill='both', expand=True, padx=15, pady=10)
            
            # Left side - text
            text_frame = tk.Frame(content_frame, bg=color['bg'])
            text_frame.pack(side='left', fill='both', expand=True)
            
            if title:
                title_label = tk.Label(
                    text_frame, 
                    text=title,
                    font=('Segoe UI', 10, 'bold'),
                    bg=color['bg'],
                    fg=color['fg']
                )
                title_label.pack(anchor='w')
            
            if message:
                msg_label = tk.Label(
                    text_frame,
                    text=message,
                    font=('Segoe UI', 9),
                    bg=color['bg'],
                    fg=color['fg'],
                    wraplength=400,
                    justify='left'
                )
                msg_label.pack(anchor='w', pady=(2, 0))
            
            # Right side - actions and dismiss
            right_frame = tk.Frame(content_frame, bg=color['bg'])
            right_frame.pack(side='right')
            
            # Action buttons
            if actions:
                for action in actions:
                    btn = tk.Button(
                        right_frame,
                        text=action['text'],
                        command=action['command'],
                        font=('Segoe UI', 8),
                        bg='white',
                        fg=color['fg'],
                        relief='flat',
                        padx=10, pady=4,
                        cursor='hand2'
                    )
                    btn.pack(side='left', padx=(0, 5))
            
            # Dismiss button
            if dismissible:
                dismiss_btn = tk.Label(
                    right_frame,
                    text='✕',
                    font=('Segoe UI', 12, 'bold'),
                    bg=color['bg'],
                    fg=color['fg'],
                    cursor='hand2',
                    padx=8, pady=4
                )
                dismiss_btn.pack(side='right', padx=(10, 0))
                dismiss_btn.bind('<Button-1>', lambda e: self._remove_alert(alert_frame))
            
            return alert_frame
            
        except Exception as e:
            print(f"Error creating alert: {e}")
            return None

    def _remove_alert(self, alert: tk.Frame):
        """Remove alert without animations."""
        try:
            if alert in self.alerts:
                self.alerts.remove(alert)
            alert.destroy()
        except Exception as e:
            print(f"Error removing alert: {e}")

    def clear_all(self):
        """Clear all alerts."""
        try:
            for alert in self.alerts.copy():
                self._remove_alert(alert)
        except Exception as e:
            print(f"Error clearing alerts: {e}")

    def dismiss_alert(self):
        """Method for compatibility with old alert system."""
        pass

    def update_alert(self, message: str):
        """Method for compatibility with old alert system."""
        pass