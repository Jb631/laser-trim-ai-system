"""
Multi-Track Analysis Mixin

Contains file and folder analysis logic for MultiTrackPage.
Handles track file discovery, parsing, and unit grouping.
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Any

import numpy as np

logger = logging.getLogger(__name__)


class AnalysisMixin:
    """Mixin containing file/folder analysis logic for MultiTrackPage.

    Required attributes from main class:
    - main_window: Reference to main application window
    - unit_info_label: Label widget for status display
    - current_unit_data: Dict to store analysis results
    - logger: Logger instance
    - progress_dialog: Progress dialog widget (optional)

    Required methods from main class:
    - after(): Schedule callback
    - update(): Update display
    - _update_multi_track_display(): Update UI with results
    - _extract_track_id(): Extract track ID from filename
    - _calculate_consistency_rating(): Calculate consistency rating
    """

    def _select_track_file(self):
        """Select a single track file to analyze."""
        filename = filedialog.askopenfilename(
            title="Select Track File",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )

        if filename:
            self._analyze_track_file(Path(filename))

    def _analyze_folder(self):
        """Analyze all track files in a folder."""
        folder = filedialog.askdirectory(title="Select Folder with Track Files")

        if folder:
            self._analyze_folder_tracks(Path(folder))

    def _analyze_track_file(self, file_path: Path):
        """Analyze a single multi-track file."""
        self.unit_info_label.configure(text=f"Analyzing: {file_path.name}...")
        self.update()

        # Run analysis in background thread
        threading.Thread(
            target=self._run_file_analysis,
            args=(file_path,),
            daemon=True
        ).start()

    def _run_file_analysis(self, file_path: Path):
        """Run analysis on a single track file in background thread."""
        try:
            # Parse filename to extract model and serial
            file_parts = file_path.stem.split('_')
            if len(file_parts) < 2:
                self.after(0, lambda: messagebox.showerror(
                    "Error", "Invalid filename format. Expected: Model_Serial_Track.xlsx"
                ))
                return

            model = file_parts[0]
            serial = file_parts[1]

            # Look for related track files
            parent_dir = file_path.parent
            related_files = []

            # Search for files with same model and serial
            for f in parent_dir.glob(f"{model}_{serial}_*.xlsx"):
                if f != file_path:
                    related_files.append(f)

            # Add the current file
            related_files.insert(0, file_path)

            # Process the Excel files to extract actual data
            from laser_trim_analyzer.core.processor import LaserTrimProcessor
            from laser_trim_analyzer.core.config import Config

            # Initialize processor with config
            try:
                # Get config from main window or create default
                config = None
                if hasattr(self.main_window, 'config') and self.main_window.config is not None:
                    config = self.main_window.config
                    # Verify it's an instance, not a class
                    if isinstance(config, type):
                        self.logger.warning("Config is a class, not an instance. Creating instance.")
                        config = config()
                else:
                    config = Config()

                # Get database manager
                db_manager = None
                if hasattr(self.main_window, 'db_manager'):
                    db_manager = self.main_window.db_manager

                # Try to get ML predictor from main window if available
                ml_predictor = None
                if hasattr(self.main_window, 'ml_manager') and self.main_window.ml_manager is not None:
                    ml_predictor = self.main_window.ml_manager
                elif hasattr(self.main_window, 'ml_predictor') and self.main_window.ml_predictor is not None:
                    ml_predictor = self.main_window.ml_predictor

                processor = LaserTrimProcessor(
                    config=config,
                    db_manager=db_manager,
                    ml_predictor=ml_predictor,
                    logger=self.logger
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize processor with full config: {e}")
                # Create minimal processor with just config
                config = Config()
                processor = LaserTrimProcessor(config=config, logger=self.logger)

            # Process files and extract track data
            tracks_data = {}
            overall_status = 'PASS'

            for track_file in related_files:
                try:
                    # Process the Excel file (use sync version)
                    self.logger.info(f"Processing file: {track_file}")
                    result = processor.process_file_sync(str(track_file))

                    if result:
                        self.logger.info(f"Result type: {type(result)}, has tracks: {hasattr(result, 'tracks')}")
                        if hasattr(result, 'tracks'):
                            self.logger.info(f"Number of tracks: {len(result.tracks)}")

                        # Handle different result structures
                        if hasattr(result, 'tracks') and result.tracks:
                            # Process all tracks in the result
                            self.logger.info(f"Processing {len(result.tracks)} tracks from {track_file}")
                            for track_id, primary in result.tracks.items():
                                self.logger.info(f"Processing track {track_id}")

                                # Debug log to check what attributes primary has
                                self.logger.debug(f"Primary object type: {type(primary)}")
                                self.logger.debug(f"Primary has position_data: {hasattr(primary, 'position_data')}")
                                self.logger.debug(f"Primary has error_data: {hasattr(primary, 'error_data')}")

                                # Create comprehensive track data
                                track_data = {
                                    'track_id': track_id,
                                    'position': track_id,
                                    'serial': serial,
                                    'timestamp': datetime.now().isoformat(),
                                    'overall_status': primary.status.value if hasattr(primary.status, 'value') else str(primary.status),
                                    'validation_status': 'Valid' if primary.linearity_analysis.linearity_pass and primary.sigma_analysis.sigma_pass else 'Invalid',

                                    # Sigma analysis
                                    'sigma_gradient': primary.sigma_analysis.sigma_gradient,
                                    'sigma_spec': primary.sigma_analysis.sigma_threshold,
                                    'sigma_margin': primary.sigma_analysis.gradient_margin,
                                    'sigma_pass': primary.sigma_analysis.sigma_pass,
                                    'sigma_analysis': {
                                        'sigma_gradient': primary.sigma_analysis.sigma_gradient
                                    },

                                    # Linearity analysis
                                    'linearity_error': primary.linearity_analysis.final_linearity_error_shifted,
                                    'linearity_spec': primary.linearity_analysis.linearity_spec,
                                    'linearity_pass': primary.linearity_analysis.linearity_pass,
                                    'linearity_offset': primary.linearity_analysis.optimal_offset,
                                    'linearity_analysis': {
                                        'final_linearity_error_shifted': primary.linearity_analysis.final_linearity_error_shifted,
                                        'optimal_offset': primary.linearity_analysis.optimal_offset
                                    },

                                    # Unit properties
                                    'resistance_change': primary.unit_properties.resistance_change_percent,
                                    'resistance_change_percent': primary.unit_properties.resistance_change_percent,
                                    'unit_properties': {
                                        'resistance_change_percent': primary.unit_properties.resistance_change_percent
                                    },

                                    # Position and error data for plotting
                                    'position_data': primary.position_data if hasattr(primary, 'position_data') and primary.position_data else [],
                                    'error_data': primary.error_data if hasattr(primary, 'error_data') and primary.error_data else [],
                                    # Include untrimmed data if available
                                    'untrimmed_positions': primary.untrimmed_positions if hasattr(primary, 'untrimmed_positions') else [],
                                    'untrimmed_errors': primary.untrimmed_errors if hasattr(primary, 'untrimmed_errors') else [],

                                    # Log data ranges for debugging
                                    'position_range': [min(primary.position_data), max(primary.position_data)] if hasattr(primary, 'position_data') and primary.position_data else [0, 0],
                                    'travel_length': primary.travel_length if hasattr(primary, 'travel_length') else 0,

                                    # Additional metrics
                                    'trim_stability': 0.95,  # Calculate if available
                                    'industry_grade': primary.industry_grade if hasattr(primary, 'industry_grade') else 'B',
                                    'file_path': str(track_file)
                                }

                                # Log the data we extracted
                                pos_data = track_data.get('position_data', [])
                                err_data = track_data.get('error_data', [])
                                self.logger.info(f"Track {track_id} data extraction: {len(pos_data)} positions, {len(err_data)} errors")
                                if pos_data:
                                    self.logger.info(f"Track {track_id} position range: [{min(pos_data):.1f}, {max(pos_data):.1f}]")
                                if err_data:
                                    self.logger.info(f"Track {track_id} error range: [{min(err_data):.6f}, {max(err_data):.6f}]")
                                self.logger.info(f"Track {track_id} linearity offset: {track_data.get('linearity_offset', 0)}")

                                tracks_data[track_id] = track_data

                                # Update overall status
                                if primary.status.value != 'PASS':
                                    overall_status = 'FAIL' if primary.status.value == 'FAIL' else 'WARNING'
                        else:
                            self.logger.warning(f"No tracks found in result for {track_file}")
                    else:
                        self.logger.warning(f"No result returned for {track_file}")

                except Exception as e:
                    self.logger.error(f"Error processing file {track_file}: {e}")
                    # Add placeholder data for failed file
                    track_id = self._extract_track_id(track_file)
                    tracks_data[track_id] = {
                        'track_id': track_id,
                        'file_path': str(track_file),
                        'filename': track_file.name,
                        'status': 'ERROR',
                        'error': str(e)
                    }

            # Calculate CV metrics if multiple tracks
            sigma_cv = 0
            linearity_cv = 0
            resistance_cv = 0

            if len(tracks_data) > 1:
                sigma_values = [t.get('sigma_gradient', 0) for t in tracks_data.values() if 'sigma_gradient' in t]
                linearity_values = [t.get('linearity_error', 0) for t in tracks_data.values() if 'linearity_error' in t]
                resistance_values = [t.get('resistance_change_percent', 0) for t in tracks_data.values() if 'resistance_change_percent' in t]

                if len(sigma_values) > 1:
                    sigma_cv = (np.std(sigma_values) / np.mean(sigma_values)) * 100 if np.mean(sigma_values) != 0 else 0
                if len(linearity_values) > 1:
                    linearity_cv = (np.std(linearity_values) / np.mean(linearity_values)) * 100 if np.mean(linearity_values) != 0 else 0
                if len(resistance_values) > 1:
                    resistance_cv = (np.std(resistance_values) / np.mean(resistance_values)) * 100 if np.mean(resistance_values) != 0 else 0

            # Create unit data structure with actual data
            self.current_unit_data = {
                'model': model,
                'serial': serial,
                'total_files': len(related_files),
                'track_count': len(tracks_data),
                'overall_status': overall_status,
                'files': [{
                    'filename': f"{model}_{serial}_MultiTrack",
                    'status': overall_status,
                    'track_count': len(tracks_data),
                    'tracks': tracks_data
                }],
                'sigma_cv': sigma_cv,
                'linearity_cv': linearity_cv,
                'resistance_cv': resistance_cv,
                'consistency': self._calculate_consistency_rating(sigma_cv, linearity_cv, resistance_cv)
            }


            # Update UI
            self.after(0, self._update_multi_track_display)

            # Log summary
            self.logger.info(f"Analysis complete: {len(tracks_data)} tracks found in {len(related_files)} files")
            self.logger.info(f"Tracks: {list(tracks_data.keys())}")
            for tid, tdata in tracks_data.items():
                pos_data = tdata.get('position_data', [])
                if pos_data:
                    self.logger.info(f"Track {tid}: {len(pos_data)} points, range [{min(pos_data):.1f}, {max(pos_data):.1f}], travel length: {tdata.get('travel_length', 'N/A')}")
                else:
                    self.logger.warning(f"Track {tid}: No position data available")

            # Show confirmation
            if len(tracks_data) > 0:
                track_list = ', '.join([track_data['track_id'] for track_data in tracks_data.values()])
                message = f"Found {len(tracks_data)} tracks in {len(related_files)} files for {model}/{serial}:\n{track_list}\n\n"
                if len(tracks_data) == 1:
                    message += "Note: Only one track found. This appears to be a single-track unit."
                else:
                    message += "Track files have been grouped for comparison analysis."

                self.after(0, lambda msg=message: messagebox.showinfo(
                    "Analysis Complete",
                    msg
                ))
            else:
                self.logger.warning("No tracks found in analysis")

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"File analysis failed: {error_msg}")
            self.after(0, lambda: messagebox.showerror(
                "Error", f"File analysis failed:\n{error_msg}"
            ))

    def _analyze_folder_tracks(self, folder_path: Path):
        """Analyze all track files in a folder and group by units."""
        self.unit_info_label.configure(text=f"Analyzing folder: {folder_path.name}...")
        self.update()

        # Show progress dialog
        from laser_trim_analyzer.gui.widgets.progress_widgets_ctk import ProgressDialog
        self.progress_dialog = ProgressDialog(
            self,
            title="Analyzing Folder",
            message="Scanning for track files..."
        )
        self.progress_dialog.show()

        # Run analysis in background thread
        threading.Thread(
            target=self._run_folder_analysis,
            args=(folder_path,),
            daemon=True
        ).start()

    def _run_folder_analysis(self, folder_path: Path):
        """Run folder analysis to find and group multi-track units."""
        try:
            # Update progress
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.after(0, lambda: self.progress_dialog.update_progress(
                    "Finding Excel files...", 0.1
                ))

            # Find all Excel files in folder
            excel_files = list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls"))

            if not excel_files:
                # Hide progress dialog
                if hasattr(self, 'progress_dialog') and self.progress_dialog:
                    self.after(0, lambda: self.progress_dialog.hide())
                    self.progress_dialog = None

                self.after(0, lambda: messagebox.showwarning(
                    "No Files", "No Excel files found in selected folder"
                ))
                return

            # Update progress
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.after(0, lambda: self.progress_dialog.update_progress(
                    f"Analyzing {len(excel_files)} files...", 0.3
                ))

            # Group files by unit (model_serial)
            unit_groups = {}

            for idx, file_path in enumerate(excel_files):
                # Update progress for each file
                progress = 0.3 + (0.5 * idx / len(excel_files))
                if hasattr(self, 'progress_dialog') and self.progress_dialog:
                    self.after(0, lambda p=progress, f=file_path: self.progress_dialog.update_progress(
                        f"Processing: {f.name}", p
                    ))
                filename = file_path.stem
                parts = filename.split('_')

                if len(parts) >= 2:
                    model = parts[0]
                    serial = parts[1]
                    unit_key = f"{model}_{serial}"

                    # Add all files to unit groups - don't require track ID in filename
                    # Files might contain multiple tracks internally
                    if unit_key not in unit_groups:
                        unit_groups[unit_key] = []
                    unit_groups[unit_key].append(file_path)

            # Include all units - even single files might contain multiple tracks
            # But prioritize units with multiple files
            if unit_groups:
                # Show all units, not just ones with multiple files
                multi_track_units = unit_groups
            else:
                # Hide progress dialog
                if hasattr(self, 'progress_dialog') and self.progress_dialog:
                    self.after(0, lambda: self.progress_dialog.hide())
                    self.progress_dialog = None

                self.after(0, lambda: messagebox.showinfo(
                    "No Units Found",
                    "No units found in folder.\n"
                    "Expected filename format: Model_Serial_*.xlsx"
                ))
                return

            # Update progress
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.after(0, lambda: self.progress_dialog.update_progress(
                    "Analysis complete!", 1.0
                ))

            # Show selection dialog
            self.after(0, lambda: self._show_unit_selection_dialog(multi_track_units))

            # Hide progress dialog after a short delay
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.after(500, lambda: self.progress_dialog.hide() if self.progress_dialog else None)
                self.after(600, lambda: setattr(self, 'progress_dialog', None))

        except Exception as e:
            self.logger.error(f"Folder analysis failed: {e}")

            # Hide progress dialog on error
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.after(0, lambda: self.progress_dialog.hide())
                self.progress_dialog = None

            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror(
                "Error", f"Folder analysis failed:\n{error_msg}"
            ))

    def _show_unit_selection_dialog(self, unit_groups: Dict[str, List[Path]]):
        """Show dialog to select which unit to analyze."""
        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title("Select Unit to Analyze")
        dialog.geometry("500x400")
        dialog.grab_set()

        # Title
        ttk.Label(
            dialog,
            text="Multi-Track Units Found:",
            font=('Segoe UI', 14, 'bold')
        ).pack(pady=(10, 20))

        # Create listbox with scrollbar
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        listbox = tk.Listbox(list_frame, font=('Segoe UI', 10))
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)

        # Populate listbox
        unit_list = []
        for unit_id, files in unit_groups.items():
            track_ids = []
            for file_path in files:
                parts = file_path.stem.split('_')
                for part in parts:
                    if len(part) == 2 and part[0] == 'T' and part[1].isalpha():
                        track_ids.append(part)
                        break

            display_text = f"{unit_id} ({len(files)} tracks: {', '.join(sorted(track_ids))})"
            listbox.insert(tk.END, display_text)
            unit_list.append((unit_id, files))

        listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill='x', padx=20, pady=(0, 10))

        def analyze_selected():
            selection = listbox.curselection()
            if selection:
                unit_id, files = unit_list[selection[0]]
                dialog.destroy()
                # Analyze the first file (others will be found automatically)
                self._analyze_track_file(files[0])

        ttk.Button(
            btn_frame,
            text="Analyze Selected Unit",
            command=analyze_selected,
            style='Primary.TButton'
        ).pack(side='left', padx=(0, 10))

        ttk.Button(
            btn_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side='left')
