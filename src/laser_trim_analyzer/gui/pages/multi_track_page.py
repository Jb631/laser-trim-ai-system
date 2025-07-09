"""
Multi-Track Analysis Page for Laser Trim Analyzer

Provides interface for analyzing and comparing multi-track units,
particularly for System B multi-track files with TA, TB identifiers.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
import threading
import os
from pathlib import Path
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import SQLAlchemy func for database queries
from sqlalchemy import func

from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.gui.widgets.stat_card import StatCard
from laser_trim_analyzer.gui.widgets.metric_card_ctk import MetricCard
# from laser_trim_analyzer.gui.widgets import add_mousewheel_support  # Not used
from laser_trim_analyzer.gui.widgets.track_viewer import IndividualTrackViewer
from laser_trim_analyzer.analysis.consistency_analyzer import ConsistencyAnalyzer

# Get logger
logger = logging.getLogger(__name__)

class MultiTrackPage(ctk.CTkFrame):
    """Multi-track analysis and comparison page."""

    def __init__(self, parent, main_window, **kwargs):
        # Initialize as CTkFrame to avoid widget hierarchy issues
        super().__init__(parent, **kwargs)
        self.main_window = main_window
        
        # Add missing BasePage functionality
        self.is_visible = False
        self.needs_refresh = True
        self._stop_requested = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.current_unit_data = None
        self.comparison_data = None
        self.consistency_analyzer = ConsistencyAnalyzer()
        
        # Initialize the page
        try:
            self._create_page()
        except Exception as e:
            self.logger.error(f"Error creating page: {e}", exc_info=True)
            self._create_error_page(str(e))

    @property
    def db_manager(self):
        """Get database manager from main window."""
        return getattr(self.main_window, 'db_manager', None)

    def _create_page(self):
        """Create multi-track page content with consistent theme (matching batch processing)."""
        # Main scrollable container (matching batch processing theme)
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create sections in order (matching batch processing pattern)
        self._create_header()
        self._create_file_selection()
        self._create_overview_section()
        self._create_comparison_section()
        self._create_individual_track_viewer()
        self._create_consistency_section()
        self._create_actions_section()

    def _create_header(self):
        """Create header section (matching batch processing theme)."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Multi-Track Unit Analysis",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)

    def _create_file_selection(self):
        """Create file selection section (matching batch processing theme)."""
        self.selection_frame = ctk.CTkFrame(self.main_container)
        self.selection_frame.pack(fill='x', pady=(0, 20))

        self.selection_label = ctk.CTkLabel(
            self.selection_frame,
            text="File Selection:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.selection_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Selection container
        self.selection_container = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        self.selection_container.pack(fill='x', padx=15, pady=(0, 15))

        # Selection buttons row
        button_frame = ctk.CTkFrame(self.selection_container, fg_color="transparent")
        button_frame.pack(fill='x', padx=10, pady=(10, 10))

        self.select_file_btn = ctk.CTkButton(
            button_frame,
            text="📁 Select Track File",
            command=self._select_track_file,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="blue",
            hover_color="darkblue"
        )
        self.select_file_btn.pack(side='left', padx=(10, 10), pady=10)

        # Analyze Folder button removed per user request
        # self.analyze_folder_btn = ctk.CTkButton(
        #     button_frame,
        #     text="📂 Analyze Folder",
        #     command=self._analyze_folder,
        #     width=150,
        #     height=40
        # )
        # self.analyze_folder_btn.pack(side='left', padx=(0, 10), pady=10)

        self.from_database_btn = ctk.CTkButton(
            button_frame,
            text="🗄️ From Database",
            command=self._select_unit_from_database,
            width=150,
            height=40
        )
        self.from_database_btn.pack(side='left', padx=(0, 10), pady=10)

        # Unit info display
        self.unit_info_label = ctk.CTkLabel(
            self.selection_container,
            text="Select a track file to begin multi-track analysis",
            font=ctk.CTkFont(size=12)
        )
        self.unit_info_label.pack(padx=10, pady=(0, 10))

    def _create_overview_section(self):
        """Create unit overview metrics section (matching batch processing theme)."""
        self.overview_frame = ctk.CTkFrame(self.main_container)
        self.overview_frame.pack(fill='x', pady=(0, 20))

        self.overview_label = ctk.CTkLabel(
            self.overview_frame,
            text="Unit Overview:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.overview_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Overview container
        self.overview_container = ctk.CTkFrame(self.overview_frame, fg_color="transparent")
        self.overview_container.pack(fill='x', padx=15, pady=(0, 15))

        # Row 1 of overview metrics
        overview_row1 = ctk.CTkFrame(self.overview_container, fg_color="transparent")
        overview_row1.pack(fill='x', padx=10, pady=(10, 5))

        self.overview_cards = {}
        
        self.overview_cards['unit_id'] = MetricCard(
            overview_row1,
            title="Unit ID",
            value="--",
            color_scheme="neutral"
        )
        self.overview_cards['unit_id'].pack(side='left', fill='x', expand=True, padx=(10, 5), pady=10)

        self.overview_cards['track_count'] = MetricCard(
            overview_row1,
            title="Track Count",
            value="-- tracks",
            color_scheme="info"
        )
        self.overview_cards['track_count'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.overview_cards['overall_status'] = MetricCard(
            overview_row1,
            title="Overall Status",
            value="--",
            color_scheme="neutral"
        )
        self.overview_cards['overall_status'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.overview_cards['consistency'] = MetricCard(
            overview_row1,
            title="Track Consistency",
            value="--",
            color_scheme="neutral"
        )
        self.overview_cards['consistency'].pack(side='left', fill='x', expand=True, padx=(5, 10), pady=10)

        # Row 2 of overview metrics
        overview_row2 = ctk.CTkFrame(self.overview_container, fg_color="transparent")
        overview_row2.pack(fill='x', padx=10, pady=(5, 10))

        self.overview_cards['sigma_cv'] = MetricCard(
            overview_row2,
            title="Sigma Variation (CV)",
            value="--%",
            color_scheme="warning"
        )
        self.overview_cards['sigma_cv'].pack(side='left', fill='x', expand=True, padx=(10, 5), pady=10)

        self.overview_cards['linearity_cv'] = MetricCard(
            overview_row2,
            title="Linearity Variation (CV)",
            value="--%",
            color_scheme="warning"
        )
        self.overview_cards['linearity_cv'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.overview_cards['resistance_cv'] = MetricCard(
            overview_row2,
            title="Resistance Variation (CV)",
            value="--%",
            color_scheme="warning"
        )
        self.overview_cards['resistance_cv'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.overview_cards['risk_level'] = MetricCard(
            overview_row2,
            title="Risk Level",
            value="--",
            color_scheme="neutral"
        )
        self.overview_cards['risk_level'].pack(side='left', fill='x', expand=True, padx=(5, 10), pady=10)

    def _create_comparison_section(self):
        """Create track linearity error overlay plot section."""
        self.comparison_frame = ctk.CTkFrame(self.main_container)
        self.comparison_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.comparison_label = ctk.CTkLabel(
            self.comparison_frame,
            text="Track Linearity Error Overlay:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.comparison_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Comparison container
        self.comparison_container = ctk.CTkFrame(self.comparison_frame, fg_color="transparent")
        self.comparison_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Control frame for scaling and export
        self.control_frame = ctk.CTkFrame(self.comparison_container)
        self.control_frame.pack(fill='x', padx=5, pady=(0, 5))
        
        # Y-axis scaling controls
        scale_label = ctk.CTkLabel(self.control_frame, text="Y-Axis Scale:")
        scale_label.pack(side='left', padx=(10, 5))
        
        self.y_scale_var = tk.DoubleVar(value=1.5)  # Default 1.5x spec limits
        self.y_scale_slider = ctk.CTkSlider(
            self.control_frame,
            from_=0.5,
            to=3.0,
            variable=self.y_scale_var,
            command=self._on_scale_changed,
            width=150
        )
        self.y_scale_slider.pack(side='left', padx=5)
        
        self.scale_value_label = ctk.CTkLabel(self.control_frame, text="1.5x")
        self.scale_value_label.pack(side='left', padx=(0, 10))
        
        # Export button
        self.export_chart_btn = ctk.CTkButton(
            self.control_frame,
            text="📊 Export Chart",
            command=self._export_linearity_chart,
            width=120,
            height=32
        )
        self.export_chart_btn.pack(side='right', padx=10)

        # Create matplotlib figure for linearity overlay
        self.linearity_fig = Figure(figsize=(12, 6), dpi=100)
        self.linearity_ax = self.linearity_fig.add_subplot(111)
        
        # Create canvas
        self.linearity_canvas = FigureCanvasTkAgg(self.linearity_fig, self.comparison_container)
        self.linearity_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize the plot
        self._initialize_linearity_plot()
        
        # Set comparison charts to None for compatibility
        self.comparison_charts = {}
        self.summary_chart = None
        self.sigma_comparison_chart = None
        self.linearity_comparison_chart = None
        self.profile_comparison_chart = None
    
    def _create_individual_track_viewer(self):
        """Create individual track viewer section."""
        self.track_viewer_frame = ctk.CTkFrame(self.main_container)
        self.track_viewer_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        self.track_viewer_label = ctk.CTkLabel(
            self.track_viewer_frame,
            text="Individual Track Details:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.track_viewer_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Track viewer container
        self.track_viewer_container = ctk.CTkFrame(self.track_viewer_frame, fg_color="transparent")
        self.track_viewer_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        # Create the individual track viewer widget
        self.individual_track_viewer = IndividualTrackViewer(self.track_viewer_container)
        self.individual_track_viewer.pack(fill='both', expand=True, padx=5, pady=5)

    def _create_consistency_section(self):
        """Create consistency analysis section (matching batch processing theme)."""
        self.consistency_frame = ctk.CTkFrame(self.main_container)
        self.consistency_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.consistency_label = ctk.CTkLabel(
            self.consistency_frame,
            text="Consistency Analysis:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.consistency_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Consistency container
        self.consistency_container = ctk.CTkFrame(self.consistency_frame, fg_color="transparent")
        self.consistency_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Consistency display
        self.consistency_display = ctk.CTkTextbox(
            self.consistency_container,
            height=200,
            state="disabled"
        )
        self.consistency_display.pack(fill='both', expand=True, padx=10, pady=10)

    def _create_actions_section(self):
        """Create export and actions section (matching batch processing theme)."""
        self.actions_frame = ctk.CTkFrame(self.main_container)
        self.actions_frame.pack(fill='x', pady=(0, 20))

        self.actions_label = ctk.CTkLabel(
            self.actions_frame,
            text="Export Options:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.actions_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Actions container
        self.actions_container = ctk.CTkFrame(self.actions_frame, fg_color="transparent")
        self.actions_container.pack(fill='x', padx=15, pady=(0, 15))

        # Export buttons only - no file selection buttons here
        button_frame = ctk.CTkFrame(self.actions_container, fg_color="transparent")
        button_frame.pack(fill='x', padx=10, pady=(10, 10))

        self.export_report_btn = ctk.CTkButton(
            button_frame,
            text="📊 Export Report",
            command=self._export_comparison_report,
            width=140,
            height=40,
            state="disabled"
        )
        self.export_report_btn.pack(side='left', padx=(10, 10), pady=10)

        self.generate_pdf_btn = ctk.CTkButton(
            button_frame,
            text="📄 Generate PDF",
            command=self._generate_pdf_report,
            width=140,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="green",
            hover_color="darkgreen",
            state="disabled"
        )
        self.generate_pdf_btn.pack(side='left', padx=(0, 10), pady=10)
        
        # Add test data button for debugging
        self.test_data_button = ctk.CTkButton(
            button_frame,
            text="🧪 Test Data",
            command=self._load_test_data,
            width=120,
            height=40
        )
        self.test_data_button.pack(side='left', padx=(0, 10), pady=10)

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

    def _update_multi_track_display(self):
        """Update the display with current unit data with enhanced safety checks."""
        if not self.current_unit_data:
            # Show empty state
            self.unit_info_label.configure(text="No multi-track data loaded. Select a file, folder, or unit from database to begin analysis.")
            
            # Reset all overview cards to empty state
            for card in self.overview_cards.values():
                if card is not None:  # Safety check for None cards
                    card.update_value("--")
                    
            # Clear charts
            for chart in self.comparison_charts.values():
                if chart is not None:  # Safety check for None charts
                    chart.clear()
                    
            return

        try:
            unit_data = self.current_unit_data
            
            # Update unit info label
            if 'model' in unit_data and 'serial' in unit_data:
                # Database data format
                model = unit_data['model']
                serial = unit_data['serial']
                file_count = unit_data.get('total_files', 0)
                track_count = unit_data.get('track_count', 0)
                status = unit_data.get('overall_status', 'UNKNOWN')
                
                self.unit_info_label.configure(
                    text=f"Unit: {model}/{serial} | {file_count} files | {track_count} tracks | Status: {status}"
                )
                
                # Update overview cards with database data
                self.overview_cards['unit_id'].update_value(f"{model}/{serial}")
                self.overview_cards['track_count'].update_value(str(track_count))
                self.overview_cards['overall_status'].update_value(status)
                
                # Set color scheme based on status
                status_color = {
                    'PASS': 'success',
                    'FAIL': 'danger',
                    'WARNING': 'warning',
                    'MIXED': 'info'
                }.get(status, 'default')
                self.overview_cards['overall_status'].set_color_scheme(status_color)
                
                # Update consistency metrics
                consistency = unit_data.get('consistency', 'UNKNOWN')
                self.overview_cards['consistency'].update_value(consistency)
                
                sigma_cv = unit_data.get('sigma_cv', 0)
                linearity_cv = unit_data.get('linearity_cv', 0)
                resistance_cv = unit_data.get('resistance_cv', 0)
                
                self.overview_cards['sigma_cv'].update_value(f"{sigma_cv:.1f}%")
                self.overview_cards['linearity_cv'].update_value(f"{linearity_cv:.1f}%")
                self.overview_cards['resistance_cv'].update_value(f"{resistance_cv:.1f}%")
                
                # Set color schemes based on variation
                sigma_color = 'success' if sigma_cv < 5 else 'warning' if sigma_cv < 10 else 'danger'
                linearity_color = 'success' if linearity_cv < 10 else 'warning' if linearity_cv < 20 else 'danger'
                resistance_color = 'success' if resistance_cv < 2 else 'warning' if resistance_cv < 5 else 'danger'
                
                self.overview_cards['sigma_cv'].set_color_scheme(sigma_color)
                self.overview_cards['linearity_cv'].set_color_scheme(linearity_color)
                self.overview_cards['resistance_cv'].set_color_scheme(resistance_color)
                
                # Calculate and update risk level
                risk_score = 0
                if sigma_cv > 10: risk_score += 1
                if linearity_cv > 20: risk_score += 1
                if resistance_cv > 5: risk_score += 1
                
                risk_level = 'Low' if risk_score == 0 else 'Medium' if risk_score == 1 else 'High'
                risk_color = 'success' if risk_score == 0 else 'warning' if risk_score == 1 else 'danger'
                self.overview_cards['risk_level'].update_value(risk_level)
                self.overview_cards['risk_level'].set_color_scheme(risk_color)
                
                # Update validation grade if available
                if 'validation_grade' in unit_data:
                    self.overview_cards['validation_grade'].update_value(unit_data['validation_grade'])
            
            else:
                # File-based data format (original processor output)
                unit_id = unit_data.get('unit_id', 'Unknown')
                track_count = len(unit_data.get('tracks', {}))
                overall_status = unit_data.get('overall_status', 'UNKNOWN')
                
                self.unit_info_label.configure(
                    text=f"Unit: {unit_id} | {track_count} tracks | Status: {overall_status}"
                )
                
                # Update overview cards
                self.overview_cards['unit_id'].update_value(unit_id)
                self.overview_cards['track_count'].update_value(str(track_count))
                self.overview_cards['overall_status'].update_value(overall_status)
                
                # Calculate metrics for file-based data
                tracks = unit_data.get('tracks', {})
                if tracks:
                    sigma_values = [t.get('sigma_gradient', 0) for t in tracks.values() if t.get('sigma_gradient')]
                    if sigma_values:
                        sigma_cv = (np.std(sigma_values) / np.mean(sigma_values)) * 100
                        self.overview_cards['sigma_cv'].update_value(f"{sigma_cv:.1f}")
                    
                    # Update other metrics as available
                    consistency = "GOOD" if len(tracks) > 1 else "N/A"
                    self.overview_cards['consistency'].update_value(consistency)

            # Update individual track viewer
            try:
                self._update_individual_track_viewer()
            except Exception as e:
                self.logger.warning(f"Failed to update track viewer: {e}")
            
            # Create comparison data for charts
            self._prepare_comparison_data()
            
            # Update charts with current data
            try:
                self._update_comparison_charts()
            except Exception as e:
                self.logger.error(f"Error updating comparison charts: {e}")
                
            try:
                self._update_consistency_analysis()
            except Exception as e:
                self.logger.error(f"Error updating consistency analysis: {e}")

            # Enable action buttons
            if hasattr(self, 'export_report_btn'):
                self.export_report_btn.configure(state='normal')
            if hasattr(self, 'generate_pdf_btn'):
                self.generate_pdf_btn.configure(state='normal')
            
            self.logger.info("Successfully updated multi-track display")
            
        except Exception as e:
            self.logger.error(f"Error updating multi-track display: {e}")
            self.unit_info_label.configure(text="Error displaying multi-track data - check logs")

    def _prepare_comparison_data(self):
        """Prepare comparison data from current unit data."""
        if not self.current_unit_data:
            return
            
        self.comparison_data = {
            'comparison_performed': True,
            'sigma_comparison': {'values': {}},
            'linearity_comparison': {'values': {}}
        }
        
        # Extract data based on format
        if 'tracks' in self.current_unit_data:
            # Direct tracks format (from file processing)
            tracks = self.current_unit_data['tracks']
            
            for track_id, track_data in tracks.items():
                # Handle different data structures
                if hasattr(track_data, 'primary_track'):
                    # Object format
                    primary = track_data.primary_track
                    if primary:
                        self.comparison_data['sigma_comparison']['values'][track_id] = getattr(primary, 'sigma_gradient', 0)
                        self.comparison_data['linearity_comparison']['values'][track_id] = getattr(primary, 'linearity_error', 0)
                elif isinstance(track_data, dict):
                    # Dictionary format
                    self.comparison_data['sigma_comparison']['values'][track_id] = track_data.get('sigma_gradient', 0)
                    self.comparison_data['linearity_comparison']['values'][track_id] = track_data.get('linearity_error', 0)
                    
        elif 'files' in self.current_unit_data:
            # Database format (tracks are within files)
            track_summary = {}  # Aggregate by track ID
            
            for file_info in self.current_unit_data['files']:
                for track_id, track_data in file_info['tracks'].items():
                    if track_id not in track_summary:
                        track_summary[track_id] = {
                            'sigma_values': [],
                            'linearity_values': [],
                            'count': 0
                        }
                    
                    # Collect values for averaging
                    sigma = track_data.get('sigma_gradient')
                    linearity = track_data.get('linearity_error')
                    
                    if sigma is not None:
                        track_summary[track_id]['sigma_values'].append(sigma)
                    if linearity is not None:
                        track_summary[track_id]['linearity_values'].append(linearity)
                    track_summary[track_id]['count'] += 1
            
            # Calculate averages for each track
            for track_id, summary in track_summary.items():
                # Average sigma gradient
                if summary['sigma_values']:
                    avg_sigma = np.mean(summary['sigma_values'])
                    self.comparison_data['sigma_comparison']['values'][track_id] = avg_sigma
                else:
                    self.comparison_data['sigma_comparison']['values'][track_id] = 0
                    
                # Average linearity error
                if summary['linearity_values']:
                    avg_linearity = np.mean(summary['linearity_values'])
                    self.comparison_data['linearity_comparison']['values'][track_id] = avg_linearity
                else:
                    self.comparison_data['linearity_comparison']['values'][track_id] = 0
    
    def _update_comparison_charts(self):
        """Update comparison charts with track data."""
        # Simply update the linearity overlay plot
        self._update_linearity_overlay()

    def _update_summary_chart(self):
        """Update the summary chart with track comparison data."""
        if not self.comparison_data or not self.current_unit_data:
            return
            
        try:
            # Prepare summary data
            track_ids = []
            pass_counts = []
            fail_counts = []
            warning_counts = []
            
            if 'files' in self.current_unit_data:
                # Database format - aggregate by track ID
                track_summary = {}
                
                for file_info in self.current_unit_data['files']:
                    for track_id, track_data in file_info['tracks'].items():
                        if track_id not in track_summary:
                            track_summary[track_id] = {'PASS': 0, 'FAIL': 0, 'WARNING': 0}
                        
                        status = track_data.get('status', 'UNKNOWN')
                        if status in track_summary[track_id]:
                            track_summary[track_id][status] += 1
                
                # Convert to lists for plotting
                for track_id in sorted(track_summary.keys()):
                    track_ids.append(track_id)
                    pass_counts.append(track_summary[track_id]['PASS'])
                    fail_counts.append(track_summary[track_id]['FAIL'])
                    warning_counts.append(track_summary[track_id]['WARNING'])
            
            # Create bar chart
            if track_ids:
                self.summary_chart.clear_chart()
                
                # Use matplotlib directly for stacked bar chart
                ax = self.summary_chart.figure.add_subplot(111)
                
                # Create the stacked bar chart
                x = range(len(track_ids))
                width = 0.6
                
                # Stack the bars
                ax.bar(x, pass_counts, width, label='Pass', color='green')
                ax.bar(x, warning_counts, width, bottom=pass_counts, label='Warning', color='orange')
                ax.bar(x, fail_counts, width, bottom=[p+w for p,w in zip(pass_counts, warning_counts)], label='Fail', color='red')
                
                # Set labels and title
                ax.set_xlabel("Track ID")
                ax.set_ylabel("File Count")
                ax.set_title("Track Status Summary")
                ax.set_xticks(x)
                ax.set_xticklabels(track_ids)
                ax.legend()
                
                # Apply theme
                self.summary_chart._apply_theme_to_axes(ax)
                self.summary_chart.figure.tight_layout()
                self.summary_chart.canvas.draw()
                
        except Exception as e:
            self.logger.error(f"Error updating summary chart: {e}")

    def _update_error_profiles(self):
        """Update error profile comparison chart."""
        if not self.current_unit_data or not hasattr(self, 'profile_comparison_chart') or not self.profile_comparison_chart:
            return

        try:
            self.profile_comparison_chart.clear_chart()
            
            # For now, create a placeholder chart showing track comparison metrics
            track_ids = []
            metrics_data = []
            
            # Extract track data based on format
            if 'files' in self.current_unit_data:
                # Database format
                for file_info in self.current_unit_data['files']:
                    for track_id, track_data in file_info['tracks'].items():
                        if track_id not in track_ids:
                            track_ids.append(track_id)
                            metrics_data.append({
                                'sigma': track_data.get('sigma_gradient', 0),
                                'linearity': abs(track_data.get('linearity_error', 0)),
                                'resistance': abs(track_data.get('resistance_change_percent', 0))
                            })
            elif 'tracks' in self.current_unit_data:
                # Direct tracks format
                for track_id, track_data in self.current_unit_data['tracks'].items():
                    track_ids.append(track_id)
                    metrics_data.append({
                        'sigma': track_data.get('sigma_gradient', 0),
                        'linearity': abs(track_data.get('linearity_error', 0)),
                        'resistance': abs(track_data.get('resistance_change', 0))
                    })
            
            if track_ids and metrics_data:
                # Create a multi-metric comparison chart
                ax = self.profile_comparison_chart.figure.add_subplot(111)
                
                x = np.arange(len(track_ids))
                width = 0.25
                
                sigma_values = [m['sigma'] for m in metrics_data]
                linearity_values = [m['linearity'] for m in metrics_data]
                resistance_values = [m['resistance'] for m in metrics_data]
                
                ax.bar(x - width, sigma_values, width, label='Sigma Gradient', color='blue')
                ax.bar(x, linearity_values, width, label='Linearity Error (V)', color='orange')
                ax.bar(x + width, resistance_values, width, label='Resistance Change %', color='green')
                
                ax.set_xlabel('Track ID')
                ax.set_ylabel('Value')
                ax.set_title('Track Metrics Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(track_ids)
                ax.legend()
                
                self.profile_comparison_chart.canvas.draw()
                
        except Exception as e:
            self.logger.error(f"Error updating error profiles: {e}")

    def _update_consistency_analysis(self):
        """Update consistency analysis using the ConsistencyAnalyzer."""
        self.consistency_display.configure(state='normal')
        self.consistency_display.delete("1.0", "end")
        
        if not self.current_unit_data:
            self.consistency_display.insert("1.0", "No unit data available for consistency analysis.")
            self.consistency_display.configure(state='disabled')
            return
            
        try:
            # Extract tracks data for analysis
            tracks_data = {}
            
            # Check different data formats
            if 'tracks' in self.current_unit_data:
                # Direct tracks format
                tracks_data = self.current_unit_data['tracks']
            elif 'files' in self.current_unit_data:
                # File-based format - extract all tracks
                for file_data in self.current_unit_data.get('files', []):
                    file_tracks = file_data.get('tracks', {})
                    tracks_data.update(file_tracks)
                    
            if not tracks_data:
                self.consistency_display.insert("1.0", "No track data found for consistency analysis.")
                self.consistency_display.configure(state='disabled')
                return
                
            # Log tracks data structure for debugging
            self.logger.info(f"Analyzing consistency for {len(tracks_data)} tracks")
            for track_id, track_data in tracks_data.items():
                if isinstance(track_data, dict):
                    self.logger.debug(f"Track {track_id} keys: {list(track_data.keys())[:10]}")
                    if 'sigma_analysis' in track_data:
                        self.logger.debug(f"Track {track_id} sigma_analysis: {track_data['sigma_analysis']}")
            
            # Log the tracks data structure before analysis
            self.logger.info(f"Sending {len(tracks_data)} tracks to consistency analyzer")
            for track_id, track_info in tracks_data.items():
                if isinstance(track_info, dict):
                    self.logger.debug(f"Track {track_id} has keys: {list(track_info.keys())}")
                    # Log the actual values we're interested in
                    if 'sigma_gradient' in track_info:
                        self.logger.debug(f"Track {track_id} sigma_gradient (direct): {track_info['sigma_gradient']}")
                    if 'sigma_analysis' in track_info and isinstance(track_info['sigma_analysis'], dict):
                        self.logger.debug(f"Track {track_id} sigma_analysis.sigma_gradient: {track_info['sigma_analysis'].get('sigma_gradient')}")
                    if 'linearity_error' in track_info:
                        self.logger.debug(f"Track {track_id} linearity_error (direct): {track_info['linearity_error']}")
                    if 'resistance_change_percent' in track_info:
                        self.logger.debug(f"Track {track_id} resistance_change_percent (direct): {track_info['resistance_change_percent']}")
            
            # Perform consistency analysis
            consistency_metrics = self.consistency_analyzer.analyze_tracks(tracks_data)
            
            # Generate and display report
            report = self.consistency_analyzer.generate_consistency_report(consistency_metrics)
            self.consistency_display.insert("1.0", report)
            
            # Update overview card with consistency rating
            if hasattr(self, 'overview_cards') and 'consistency' in self.overview_cards:
                self.overview_cards['consistency'].update_value(consistency_metrics.overall_consistency)
                
                # Set color based on consistency
                color_map = {
                    'EXCELLENT': 'success',
                    'GOOD': 'info', 
                    'FAIR': 'warning',
                    'POOR': 'danger',
                    'N/A - Single Track': 'neutral'
                }
                color = color_map.get(consistency_metrics.overall_consistency, 'neutral')
                self.overview_cards['consistency'].set_color_scheme(color)
                
            # Update CV cards if they exist
            if hasattr(self, 'overview_cards'):
                if 'sigma_cv' in self.overview_cards:
                    self.overview_cards['sigma_cv'].update_value(f"{consistency_metrics.sigma_cv:.1f}%")
                if 'linearity_cv' in self.overview_cards:
                    self.overview_cards['linearity_cv'].update_value(f"{consistency_metrics.linearity_cv:.1f}%")
                    
            # Store metrics in comparison_data for export
            if not self.comparison_data:
                self.comparison_data = {}
            self.comparison_data['consistency_metrics'] = consistency_metrics
            self.comparison_data['consistency_issues'] = consistency_metrics.issues
            
        except Exception as e:
            self.logger.error(f"Consistency analysis failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.consistency_display.insert("1.0", f"Error performing consistency analysis:\n{str(e)}")
            
        finally:
            self.consistency_display.configure(state='disabled')

    def _export_comparison_report(self):
        """Export multi-track comparison report to Excel."""
        if not self.current_unit_data:
            messagebox.showwarning("No Data", "No unit data available to export")
            return

        try:
            # Get save location
            unit_id = f"{self.current_unit_data.get('model', 'Unknown')}_{self.current_unit_data.get('serial', 'Unknown')}"
            initial_filename = f"multitrack_report_{unit_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            filename = filedialog.asksaveasfilename(
                title="Export Multi-Track Report",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile=initial_filename
            )
            
            if not filename:
                return

            # Create comprehensive Excel export
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Unit summary
                unit_summary = pd.DataFrame([{
                    'Model': self.current_unit_data.get('model', 'Unknown'),
                    'Serial': self.current_unit_data.get('serial', 'Unknown'),
                    'Track Count': self.current_unit_data.get('track_count', 0),
                    'File Count': self.current_unit_data.get('total_files', 0),
                    'Overall Status': self.current_unit_data.get('overall_status', 'Unknown'),
                    'Consistency': self.current_unit_data.get('consistency', 'Unknown'),
                    'Sigma CV': f"{self.current_unit_data.get('sigma_cv', 0):.1f}%",
                    'Linearity CV': f"{self.current_unit_data.get('linearity_cv', 0):.1f}%",
                    'Resistance CV': f"{self.current_unit_data.get('resistance_cv', 0):.1f}%",
                    'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }])
                unit_summary.to_excel(writer, sheet_name='Unit Summary', index=False)
                
                # Track comparison data
                if self.comparison_data and self.comparison_data.get('comparison_performed'):
                    comparison_df = pd.DataFrame([self.comparison_data])
                    comparison_df.to_excel(writer, sheet_name='Comparison Analysis', index=False)
                
                # Individual track data with validation information
                track_data = []
                # Extract tracks from files structure
                all_tracks = {}
                if 'files' in self.current_unit_data:
                    for file_data in self.current_unit_data.get('files', []):
                        file_tracks = file_data.get('tracks', {})
                        all_tracks.update(file_tracks)
                elif 'tracks' in self.current_unit_data:
                    all_tracks = self.current_unit_data['tracks']
                    
                for track_id, track_info in all_tracks.items():
                    # Handle dictionary format
                    if isinstance(track_info, dict):
                        track_record = {
                            'Track ID': track_id,
                            'Status': track_info.get('overall_status', 'Unknown'),
                            'Sigma Gradient': track_info.get('sigma_gradient', 0),
                            'Sigma Pass': track_info.get('sigma_pass', False),
                            'Sigma Margin': track_info.get('sigma_margin', 0),
                            'Linearity Error': track_info.get('linearity_error', 0),
                            'Linearity Pass': track_info.get('linearity_pass', False),
                            'Linearity Spec': track_info.get('linearity_spec', 0),
                            'Resistance Change %': track_info.get('resistance_change_percent', 0),
                            'Travel Length': track_info.get('travel_length', 0),
                            'File Path': track_info.get('file_path', 'N/A')
                        }
                        track_data.append(track_record)
                    else:
                        # Handle object format (legacy)
                        try:
                            primary_track = track_info.primary_track if hasattr(track_info, 'primary_track') else track_info
                            track_record = {
                                'Track ID': track_id,
                                'Status': primary_track.status.value if hasattr(primary_track.status, 'value') else str(primary_track.status),
                                'Sigma Gradient': primary_track.sigma_analysis.sigma_gradient if hasattr(primary_track, 'sigma_analysis') else 0,
                                'Sigma Pass': primary_track.sigma_analysis.sigma_pass if hasattr(primary_track, 'sigma_analysis') else False,
                                'Linearity Error': primary_track.linearity_analysis.final_linearity_error_shifted if hasattr(primary_track, 'linearity_analysis') else 0,
                                'Linearity Pass': primary_track.linearity_analysis.linearity_pass if hasattr(primary_track, 'linearity_analysis') else False,
                                'Resistance Change %': primary_track.unit_properties.resistance_change_percent if hasattr(primary_track, 'unit_properties') else 0
                            }
                            track_data.append(track_record)
                        except:
                            pass
                
                if track_data:
                    tracks_df = pd.DataFrame(track_data)
                    tracks_df.to_excel(writer, sheet_name='Track Details', index=False)
                
                # Validation summary sheet
                validation_summary_data = []
                for track_id, track_info in all_tracks.items():
                    if isinstance(track_info, dict):
                        # Create validation summary from available data
                        validation_record = {
                            'Track ID': track_id,
                            'Overall Status': track_info.get('overall_status', 'Unknown'),
                            'Sigma Test': 'PASS' if track_info.get('sigma_pass', False) else 'FAIL',
                            'Sigma Gradient': track_info.get('sigma_gradient', 0),
                            'Sigma Threshold': track_info.get('sigma_spec', 0),
                            'Linearity Test': 'PASS' if track_info.get('linearity_pass', False) else 'FAIL',
                            'Linearity Error': track_info.get('linearity_error', 0),
                            'Linearity Spec': track_info.get('linearity_spec', 0),
                            'Resistance Test': track_info.get('validation_status', 'Unknown'),
                            'Resistance Change %': track_info.get('resistance_change_percent', 0),
                            'Industry Grade': track_info.get('industry_grade', 'N/A')
                        }
                        validation_summary_data.append(validation_record)
                
                if validation_summary_data:
                    validation_df = pd.DataFrame(validation_summary_data)
                    validation_df.to_excel(writer, sheet_name='Validation Summary', index=False)

            messagebox.showinfo("Export Complete", f"Multi-track report exported to:\n{filename}")
            self.logger.info(f"Exported multi-track report to {filename}")

        except Exception as e:
            error_msg = f"Failed to export report: {str(e)}"
            messagebox.showerror("Export Error", error_msg)
            self.logger.error(f"Export failed: {e}")

    def _generate_pdf_report(self):
        """Generate PDF report for multi-track analysis."""
        if not self.current_unit_data:
            messagebox.showwarning("No Data", "No multi-track data available to generate report")
            return
            
        # Extract tracks from files structure
        all_tracks = {}
        if 'files' in self.current_unit_data:
            for file_data in self.current_unit_data.get('files', []):
                file_tracks = file_data.get('tracks', {})
                all_tracks.update(file_tracks)
        elif 'tracks' in self.current_unit_data:
            all_tracks = self.current_unit_data['tracks']
            
        if not all_tracks:
            messagebox.showwarning("No Data", "No multi-track data available to generate report")
            return
        
        # Ask for save location
        default_filename = f"{self.current_unit_data.get('model', 'unit')}_{self.current_unit_data.get('serial', 'unknown')}_multi_track_report.pdf"
        filename = filedialog.asksaveasfilename(
            title="Save PDF Report",
            defaultextension=".pdf",
            initialfile=default_filename,
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Import matplotlib backends for PDF
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # Create PDF with multiple pages
            with PdfPages(filename) as pdf:
                # Page 1: Summary and Overview
                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle(f'Multi-Track Analysis Report\n{self.current_unit_data.get("model", "N/A")} - {self.current_unit_data.get("serial", "N/A")}', 
                            fontsize=16, fontweight='bold')
                
                # Create grid layout
                gs = GridSpec(6, 2, figure=fig, hspace=0.4, wspace=0.3)
                
                # Summary text
                ax_summary = fig.add_subplot(gs[0:2, :])
                ax_summary.axis('off')
                
                summary_text = f"""
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Unit Information:
• Model: {self.current_unit_data.get('model', 'N/A')}
• Serial: {self.current_unit_data.get('serial', 'N/A')}
• Total Tracks: {self.current_unit_data.get('track_count', 0)}
• Overall Status: {self.current_unit_data.get('overall_status', 'N/A')}

Consistency Analysis:
• Consistency Grade: {self.current_unit_data.get('consistency', 'N/A')}
• Sigma CV: {self.current_unit_data.get('sigma_cv', 0):.2f}%
• Linearity CV: {self.current_unit_data.get('linearity_cv', 0):.2f}%
"""
                ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes, 
                               fontsize=10, verticalalignment='top', fontfamily='monospace')
                
                # Track summary table
                ax_table = fig.add_subplot(gs[2:4, :])
                ax_table.axis('off')
                
                # Prepare table data
                table_data = [['Track ID', 'Status', 'Sigma Gradient', 'Linearity Error (V)']]
                
                if all_tracks:
                    for track_id, result in all_tracks.items():
                        if isinstance(result, dict):
                            # Handle dictionary format
                            table_data.append([
                                track_id,
                                result.get('overall_status', 'Unknown'),
                                f"{result.get('sigma_gradient', 0):.6f}",
                                f"{result.get('linearity_error', 0):.4f}"
                            ])
                        elif hasattr(result, 'primary_track') and result.primary_track:
                            # Handle object format
                            primary_track = result.primary_track
                            table_data.append([
                                track_id,
                                primary_track.status.value,
                                f"{primary_track.sigma_analysis.sigma_gradient:.6f}",
                                f"{primary_track.linearity_analysis.final_linearity_error_shifted:.4f}"
                            ])
                
                if len(table_data) > 1:
                    table = ax_table.table(cellText=table_data[1:], 
                                         colLabels=table_data[0],
                                         cellLoc='center',
                                         loc='center',
                                         colWidths=[0.2, 0.2, 0.3, 0.3])
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1.2, 1.5)
                    
                    # Color code status cells
                    for i in range(1, len(table_data)):
                        status = table_data[i][1]
                        if status == 'PASS':
                            table[(i, 1)].set_facecolor('#90EE90')
                        elif status == 'FAIL':
                            table[(i, 1)].set_facecolor('#FFB6C1')
                        elif status == 'WARNING':
                            table[(i, 1)].set_facecolor('#FFFFE0')
                
                # Risk Assessment
                ax_risk = fig.add_subplot(gs[4:6, :])
                ax_risk.axis('off')
                
                risk_text = self._generate_risk_assessment_text()
                ax_risk.text(0.1, 0.9, risk_text, transform=ax_risk.transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # Page 2: Track Comparison Charts
                if all_tracks and len(all_tracks) > 0:
                    fig2 = plt.figure(figsize=(8.5, 11))
                    fig2.suptitle('Track Comparison Charts', fontsize=16, fontweight='bold')
                    
                    # Prepare data for charts
                    track_ids = []
                    sigma_values = []
                    linearity_values = []
                    
                    for track_id, track_data in all_tracks.items():
                        track_ids.append(track_id)
                        # Handle different data structures
                        if isinstance(track_data, dict):
                            sigma_values.append(track_data.get('sigma_gradient', 0))
                            linearity_values.append(abs(track_data.get('linearity_error', 0)))
                        elif hasattr(track_data, 'sigma_gradient'):
                            sigma_values.append(getattr(track_data, 'sigma_gradient', 0))
                            linearity_values.append(abs(getattr(track_data, 'linearity_error', 0)))
                    
                    if track_ids:
                        # Sigma comparison
                        ax1 = fig2.add_subplot(2, 1, 1)
                        bars1 = ax1.bar(track_ids, sigma_values, color='skyblue', edgecolor='navy')
                        ax1.set_xlabel('Track ID')
                        ax1.set_ylabel('Sigma Gradient')
                        ax1.set_title('Sigma Gradient by Track')
                        ax1.grid(True, alpha=0.3)
                        
                        # Add value labels on bars
                        for bar, value in zip(bars1, sigma_values):
                            height = bar.get_height()
                            ax1.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.6f}', ha='center', va='bottom', fontsize=8)
                        
                        # Linearity comparison
                        ax2 = fig2.add_subplot(2, 1, 2)
                        bars2 = ax2.bar(track_ids, linearity_values, color='lightcoral', edgecolor='darkred')
                        ax2.set_xlabel('Track ID')
                        ax2.set_ylabel('Linearity Error (V)')
                        ax2.set_title('Linearity Error by Track')
                        ax2.grid(True, alpha=0.3)
                        
                        # Add value labels on bars
                        for bar, value in zip(bars2, linearity_values):
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.4f}', ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    pdf.savefig(fig2, bbox_inches='tight')
                    plt.close(fig2)
                
                # Page 3: Individual Track Error Plots (if available)
                track_count = 0
                tracks_per_page = 4
                track_list = []
                
                if 'tracks' in self.current_unit_data:
                    for track_id, track_data in all_tracks.items():
                        # Handle different data structures
                        if isinstance(track_data, dict):
                            # Check for position and error data
                            if 'position_data' in track_data and 'error_data' in track_data:
                                if track_data['position_data'] and track_data['error_data']:
                                    track_list.append((track_id, track_data))
                        elif hasattr(track_data, 'position_data') and hasattr(track_data, 'error_data'):
                            if track_data.position_data and track_data.error_data:
                                track_list.append((track_id, track_data))
                
                if track_list:
                    # Create pages with 4 tracks each
                    for page_start in range(0, len(track_list), tracks_per_page):
                        fig3 = plt.figure(figsize=(8.5, 11))
                        fig3.suptitle('Track Error Plots', fontsize=16, fontweight='bold')
                        
                        page_tracks = track_list[page_start:page_start + tracks_per_page]
                        
                        for i, (track_id, track_data) in enumerate(page_tracks):
                            ax = fig3.add_subplot(2, 2, i + 1)
                            
                            # Get position and error data
                            if isinstance(track_data, dict):
                                positions = track_data.get('position_data', [])
                                errors = track_data.get('error_data', [])
                                spec_limit = track_data.get('linearity_spec', 0.01)
                            else:
                                positions = track_data.position_data
                                errors = track_data.error_data
                                spec_limit = getattr(track_data, 'linearity_spec', 0.01)
                            
                            # Plot error data
                            ax.plot(positions, errors, 'b-', linewidth=1.5, label='Error')
                            
                            # Add zero line
                            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                            
                            # Add spec limit lines
                            ax.axhline(y=spec_limit, color='r', linestyle='--', alpha=0.7, label=f'Spec: ±{spec_limit:.4f}V')
                            ax.axhline(y=-spec_limit, color='r', linestyle='--', alpha=0.7)
                            
                            ax.set_xlabel('Position (mm)')
                            ax.set_ylabel('Error (V)')
                            ax.set_title(f'Track {track_id}')
                            ax.grid(True, alpha=0.3)
                            ax.legend(fontsize=8)
                        
                        plt.tight_layout()
                        pdf.savefig(fig3, bbox_inches='tight')
                        plt.close(fig3)
            
            messagebox.showinfo("Success", f"PDF report saved to:\n{filename}")
            self.logger.info(f"PDF report generated: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {e}")
            messagebox.showerror("Error", f"Failed to generate PDF report:\n{str(e)}")
    
    def _generate_risk_assessment_text(self) -> str:
        """Generate risk assessment text based on consistency analysis."""
        consistency = self.current_unit_data.get('consistency', 'UNKNOWN')
        sigma_cv = self.current_unit_data.get('sigma_cv', 0)
        linearity_cv = self.current_unit_data.get('linearity_cv', 0)
        
        risk_level = "UNKNOWN"
        recommendations = []
        
        if consistency == 'EXCELLENT':
            risk_level = "LOW"
            recommendations = [
                "• Excellent track-to-track consistency",
                "• Continue current manufacturing process",
                "• Regular monitoring recommended"
            ]
        elif consistency == 'GOOD':
            risk_level = "LOW-MEDIUM"
            recommendations = [
                "• Good overall consistency",
                "• Minor variations detected",
                "• Review process parameters periodically"
            ]
        elif consistency == 'FAIR':
            risk_level = "MEDIUM"
            recommendations = [
                "• Moderate consistency issues detected",
                "• Review laser trimming parameters",
                "• Consider process optimization"
            ]
        elif consistency == 'POOR':
            risk_level = "HIGH"
            recommendations = [
                "• Significant track-to-track variations",
                "• Immediate process review recommended",
                "• Check equipment calibration",
                "• Consider re-trimming if possible"
            ]
        
        text = f"""
Risk Assessment:
• Risk Level: {risk_level}
• Consistency Grade: {consistency}

Key Metrics:
• Sigma Coefficient of Variation: {sigma_cv:.2f}%
• Linearity Coefficient of Variation: {linearity_cv:.2f}%

Recommendations:
{chr(10).join(recommendations)}
"""
        return text

    def _view_individual_tracks(self):
        """Open individual track analysis in separate windows."""
        if not self.current_unit_data or not self.current_unit_data.get('tracks'):
            messagebox.showwarning("No Data", "No track data available to view")
            return

        try:
            # Create track selection dialog
            dialog = ctk.CTkToplevel(self.winfo_toplevel())
            dialog.title("Individual Track Viewer")
            dialog.geometry("800x600")
            dialog.transient(self.winfo_toplevel())
            
            # Main container
            main_frame = ctk.CTkFrame(dialog)
            main_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Header
            header_label = ctk.CTkLabel(
                main_frame,
                text="Select Track to View Details",
                font=ctk.CTkFont(size=16, weight="bold")
            )
            header_label.pack(pady=(10, 20))
            
            # Track list
            track_list_frame = ctk.CTkFrame(main_frame)
            track_list_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
            
            # Create scrollable frame for tracks
            tracks_scroll = ctk.CTkScrollableFrame(track_list_frame)
            tracks_scroll.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Add track cards
            for track_id, result in self.current_unit_data['tracks'].items():
                primary_track = result.primary_track
                if primary_track:
                    # Track card
                    track_card = ctk.CTkFrame(tracks_scroll)
                    track_card.pack(fill='x', padx=5, pady=5)
                    
                    # Track info layout
                    info_frame = ctk.CTkFrame(track_card)
                    info_frame.pack(fill='x', padx=10, pady=10)
                    
                    # Left side - basic info
                    left_frame = ctk.CTkFrame(info_frame)
                    left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
                    
                    ctk.CTkLabel(
                        left_frame,
                        text=f"Track {track_id}",
                        font=ctk.CTkFont(size=14, weight="bold")
                    ).pack(anchor='w', padx=10, pady=(10, 5))
                    
                    ctk.CTkLabel(
                        left_frame,
                        text=f"Status: {primary_track.status.value}",
                        font=ctk.CTkFont(size=12)
                    ).pack(anchor='w', padx=10, pady=2)
                    
                    ctk.CTkLabel(
                        left_frame,
                        text=f"Sigma: {primary_track.sigma_analysis.sigma_gradient:.6f}",
                        font=ctk.CTkFont(size=12)
                    ).pack(anchor='w', padx=10, pady=2)
                    
                    ctk.CTkLabel(
                        left_frame,
                        text=f"Linearity: {primary_track.linearity_analysis.final_linearity_error_shifted:.4f}V",
                        font=ctk.CTkFont(size=12)
                    ).pack(anchor='w', padx=10, pady=(2, 10))
                    
                    # Right side - action button
                    right_frame = ctk.CTkFrame(info_frame)
                    right_frame.pack(side='right', padx=10, pady=10)
                    
                    view_btn = ctk.CTkButton(
                        right_frame,
                        text="View Details",
                        command=lambda tid=track_id, track=primary_track: self._show_track_details(tid, track),
                        width=100
                    )
                    view_btn.pack(pady=10)
            
            # Close button
            close_btn = ctk.CTkButton(
                main_frame,
                text="Close",
                command=dialog.destroy,
                width=100
            )
            close_btn.pack(pady=10)
            
        except Exception as e:
            self.logger.error(f"Error opening individual track viewer: {e}")
            messagebox.showerror("Error", f"Failed to open track viewer:\n{str(e)}")

    def _show_track_details(self, track_id: str, track_data):
        """Show detailed analysis for a specific track."""
        try:
            # Create detailed track window
            detail_window = ctk.CTkToplevel(self.winfo_toplevel())
            detail_window.title(f"Track {track_id} - Detailed Analysis")
            detail_window.geometry("1000x700")
            detail_window.transient(self.winfo_toplevel())
            
            # Create notebook for different sections
            notebook = ctk.CTkTabview(detail_window)
            notebook.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Overview tab
            overview_tab = notebook.add("Overview")
            self._create_track_overview_tab(overview_tab, track_id, track_data)
            
            # Sigma analysis tab
            sigma_tab = notebook.add("Sigma Analysis")
            self._create_track_sigma_tab(sigma_tab, track_data)
            
            # Linearity analysis tab
            linearity_tab = notebook.add("Linearity Analysis")
            self._create_track_linearity_tab(linearity_tab, track_data)
            
            # Resistance analysis tab
            resistance_tab = notebook.add("Resistance Analysis")
            self._create_track_resistance_tab(resistance_tab, track_data)
            
            # Data visualization tab
            viz_tab = notebook.add("Data Visualization")
            self._create_track_visualization_tab(viz_tab, track_data)
            
            # Set default tab
            notebook.set("Overview")
            
        except Exception as e:
            self.logger.error(f"Error showing track details: {e}")
            messagebox.showerror("Error", f"Failed to show track details:\n{str(e)}")

    def _create_track_overview_tab(self, parent, track_id: str, track_data):
        """Create overview tab for individual track."""
        # Scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Basic information section
        info_frame = ctk.CTkFrame(scroll_frame)
        info_frame.pack(fill='x', pady=(0, 10))
        
        ctk.CTkLabel(
            info_frame,
            text="Track Information",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        info_text = f"""Track ID: {track_id}
Overall Status: {track_data.overall_status.value}
Risk Category: {track_data.failure_prediction.risk_category.value if track_data.failure_prediction else 'Unknown'}

Unit Properties:
• Travel Length: {track_data.unit_properties.travel_length:.2f} mm
• Unit Length: {track_data.unit_properties.unit_length:.2f} mm  
• Resistance Before: {track_data.unit_properties.resistance_before:.2f} Ω
• Resistance After: {track_data.unit_properties.resistance_after:.2f} Ω
• Resistance Change: {track_data.unit_properties.resistance_change_percent:.2f}%
"""
        
        info_display = ctk.CTkTextbox(info_frame, height=200)
        info_display.pack(fill='x', padx=10, pady=(0, 10))
        info_display.insert('1.0', info_text)
        info_display.configure(state='disabled')
        
        # Quick metrics section
        metrics_frame = ctk.CTkFrame(scroll_frame)
        metrics_frame.pack(fill='x', pady=(0, 10))
        
        ctk.CTkLabel(
            metrics_frame,
            text="Quick Metrics",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        metrics_container = ctk.CTkFrame(metrics_frame)
        metrics_container.pack(fill='x', padx=10, pady=(0, 10))
        
        # Metric cards
        sigma_card = MetricCard(
            metrics_container,
            title="Sigma Gradient",
            value=f"{track_data.sigma_analysis.sigma_gradient:.6f}",
            status="success" if track_data.sigma_analysis.sigma_pass else "danger"
        )
        sigma_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)
        
        linearity_card = MetricCard(
            metrics_container,
            title="Linearity Error",
            value=f"{track_data.linearity_analysis.final_linearity_error_shifted:.4f}V",
            status="success" if track_data.linearity_analysis.linearity_pass else "danger"
        )
        linearity_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)
        
        # Analysis summary
        summary_frame = ctk.CTkFrame(scroll_frame)
        summary_frame.pack(fill='x', pady=(0, 10))
        
        ctk.CTkLabel(
            summary_frame,
            text="Analysis Summary",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        summary_text = f"""Sigma Analysis:
• Gradient: {track_data.sigma_analysis.sigma_gradient:.6f}
• Threshold: {track_data.sigma_analysis.sigma_threshold:.6f}
• Pass: {'✓' if track_data.sigma_analysis.sigma_pass else '✗'}

Linearity Analysis:
• Error (Shifted): {track_data.linearity_analysis.final_linearity_error_shifted:.4f}V
• Specification: {track_data.linearity_analysis.linearity_spec:.4f}V
• Pass: {'✓' if track_data.linearity_analysis.linearity_pass else '✗'}

Trim Effectiveness:
• Improvement: {track_data.trim_effectiveness.improvement_percent:.2f}%
• Effectiveness Grade: {track_data.trim_effectiveness.effectiveness_grade}
"""
        
        summary_display = ctk.CTkTextbox(summary_frame, height=200)
        summary_display.pack(fill='x', padx=10, pady=(0, 10))
        summary_display.insert('1.0', summary_text)
        summary_display.configure(state='disabled')

    def _create_track_sigma_tab(self, parent, track_data):
        """Create sigma analysis tab."""
        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        sigma = track_data.sigma_analysis
        
        # Sigma metrics
        metrics_frame = ctk.CTkFrame(scroll_frame)
        metrics_frame.pack(fill='x', pady=(0, 10))
        
        ctk.CTkLabel(
            metrics_frame,
            text="Sigma Analysis Details",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        metrics_text = f"""Primary Measurements:
• Sigma Gradient: {sigma.sigma_gradient:.6f}
• Sigma Threshold: {sigma.sigma_threshold:.6f}
• Pass Status: {'PASS' if sigma.sigma_pass else 'FAIL'}

Improvement Analysis:
• Improvement Percent: {sigma.improvement_percent:.2f}%
• Absolute Improvement: {sigma.absolute_improvement:.6f}

Statistical Analysis:
• Sigma Rating: {getattr(sigma, 'sigma_rating', 'Not calculated')}
• Process Capability: {getattr(sigma, 'process_capability', 'Not calculated')}
"""
        
        metrics_display = ctk.CTkTextbox(metrics_frame, height=200)
        metrics_display.pack(fill='x', padx=10, pady=(0, 10))
        metrics_display.insert('1.0', metrics_text)
        metrics_display.configure(state='disabled')

    def _create_track_linearity_tab(self, parent, track_data):
        """Create linearity analysis tab."""
        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        linearity = track_data.linearity_analysis
        
        # Linearity metrics
        metrics_frame = ctk.CTkFrame(scroll_frame)
        metrics_frame.pack(fill='x', pady=(0, 10))
        
        ctk.CTkLabel(
            metrics_frame,
            text="Linearity Analysis Details",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        metrics_text = f"""Primary Measurements:
• Final Linearity Error (Shifted): {linearity.final_linearity_error_shifted:.4f}V
• Linearity Specification: {linearity.linearity_spec:.4f}V
• Pass Status: {'PASS' if linearity.linearity_pass else 'FAIL'}

Analysis Details:
• Independent Linearity: {getattr(linearity, 'independent_linearity', 'Not calculated'):.4f}%
• Zero Based Linearity: {getattr(linearity, 'zero_based_linearity', 'Not calculated'):.4f}%
• End Point Linearity: {getattr(linearity, 'end_point_linearity', 'Not calculated'):.4f}%

Quality Metrics:
• Linearity Grade: {getattr(linearity, 'linearity_grade', 'Not assigned')}
• Analysis Quality: {getattr(linearity, 'analysis_quality', 'Not rated')}
"""
        
        metrics_display = ctk.CTkTextbox(metrics_frame, height=200)
        metrics_display.pack(fill='x', padx=10, pady=(0, 10))
        metrics_display.insert('1.0', metrics_text)
        metrics_display.configure(state='disabled')

    def _create_track_resistance_tab(self, parent, track_data):
        """Create resistance analysis tab."""
        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        resistance = track_data.resistance_analysis
        props = track_data.unit_properties
        
        # Resistance metrics
        metrics_frame = ctk.CTkFrame(scroll_frame)
        metrics_frame.pack(fill='x', pady=(0, 10))
        
        ctk.CTkLabel(
            metrics_frame,
            text="Resistance Analysis Details",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        metrics_text = f"""Resistance Values:
• Before Trim: {props.resistance_before:.2f} Ω
• After Trim: {props.resistance_after:.2f} Ω
• Change: {props.resistance_change_percent:.2f}%

Analysis Results:
• Resistance Stability: {getattr(resistance, 'resistance_stability', 'Not calculated')}
• Temperature Coefficient: {getattr(resistance, 'temperature_coefficient', 'Not measured')}
• Process Variation: {getattr(resistance, 'process_variation', 'Not calculated')}

Quality Assessment:
• Stability Grade: {getattr(resistance, 'stability_grade', 'Not assigned')}
• Reliability Score: {getattr(resistance, 'reliability_score', 'Not calculated')}
"""
        
        metrics_display = ctk.CTkTextbox(metrics_frame, height=200)
        metrics_display.pack(fill='x', padx=10, pady=(0, 10))
        metrics_display.insert('1.0', metrics_text)
        metrics_display.configure(state='disabled')

    def _create_track_visualization_tab(self, parent, track_data):
        """Create data visualization tab."""
        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Check if we have position and error data
        if hasattr(track_data, 'position_data') and hasattr(track_data, 'error_data'):
            # Error profile chart
            from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
            
            error_chart = ChartWidget(
                scroll_frame,
                chart_type='line',
                title='Error Profile vs Position',
                figsize=(10, 4)
            )
            error_chart.pack(fill='x', padx=10, pady=10)
            
            # Plot error data - use untrimmed if available for full view
            positions = None
            errors = None
            
            # Check for untrimmed data first
            if hasattr(track_data, 'untrimmed_positions') and hasattr(track_data, 'untrimmed_errors'):
                if track_data.untrimmed_positions and track_data.untrimmed_errors:
                    positions = track_data.untrimmed_positions
                    errors = track_data.untrimmed_errors
            
            # Fall back to trimmed data if untrimmed not available
            if positions is None and track_data.position_data and track_data.error_data:
                positions = track_data.position_data
                errors = track_data.error_data
            
            if positions and errors:
                error_chart.plot_line(
                    x_data=positions,
                    y_data=errors,
                    label="Error Profile",
                    color='primary',
                    xlabel="Position (mm)",
                    ylabel="Error (V)"
                )
        else:
            # No chart data available
            no_data_frame = ctk.CTkFrame(scroll_frame)
            no_data_frame.pack(fill='x', pady=10)
            
            ctk.CTkLabel(
                no_data_frame,
                text="No position/error data available for visualization",
                font=ctk.CTkFont(size=12)
            ).pack(pady=20)

    def _load_multi_track_from_database(self, model: str, serial: str):
        """Load multi-track data for a specific unit from database."""
        try:
            if not self.db_manager:
                raise ValueError("Database not connected")
            
            # Use the db_manager from main_window directly
            db_manager = self.main_window.db_manager if hasattr(self.main_window, 'db_manager') else None
            if not db_manager:
                raise ValueError("Database not connected")

            # Get all analyses for this model/serial combination
            historical_data = db_manager.get_historical_data(
                model=model,
                serial=serial,
                include_tracks=True,
                limit=None  # Get all data
            )

            if not historical_data:
                self.after(0, lambda: messagebox.showinfo(
                    "No Data",
                    f"No data found for Model: {model}, Serial: {serial}"
                ))
                return

            # Group by track files (different filenames might represent different tracks)
            track_data = {}
            unit_summary = {
                'model': model,
                'serial': serial,
                'total_files': len(historical_data),
                'track_count': 0,
                'overall_status': 'UNKNOWN',
                'files': []
            }

            for analysis in historical_data:
                file_info = {
                    'filename': analysis.filename,
                    'file_date': analysis.file_date,
                    'timestamp': analysis.timestamp,
                    'status': analysis.overall_status.value,
                    'track_count': len(analysis.tracks),
                    'tracks': {}
                }

                for track in analysis.tracks:
                    # Debug logging for raw data
                    self.logger.debug(f"Track {track.track_id} - position_data type: {type(track.position_data)}, length: {len(track.position_data) if track.position_data else 0}")
                    self.logger.debug(f"Track {track.track_id} - error_data type: {type(track.error_data)}, length: {len(track.error_data) if track.error_data else 0}")
                    
                    # Get all track data including analysis details
                    track_info = {
                        'track_id': track.track_id,
                        'status': track.status.value,
                        'overall_status': track.status.value,
                        'sigma_gradient': track.sigma_gradient,
                        'sigma_pass': track.sigma_pass,
                        'sigma_spec': track.sigma_threshold,
                        'sigma_margin': getattr(track, 'sigma_margin', None),
                        'linearity_error': track.final_linearity_error_shifted,
                        'linearity_spec': track.linearity_spec,
                        'linearity_pass': track.linearity_pass,
                        'resistance_change': track.resistance_change,
                        'resistance_change_percent': track.resistance_change_percent,
                        'failure_probability': track.failure_probability,
                        'risk_category': track.risk_category.value if track.risk_category else 'UNKNOWN',
                        'validation_status': 'PASS' if track.sigma_pass and track.linearity_pass else 'FAIL',
                        'position': track.track_id,  # For track viewer
                        'serial': serial,  # For track viewer
                        'timestamp': analysis.timestamp,
                        # Add additional fields for full feature support
                        'travel_length': track.travel_length,
                        'untrimmed_resistance': track.untrimmed_resistance,
                        'trimmed_resistance': track.trimmed_resistance,
                        'optimal_offset': track.optimal_offset,
                        'linearity_offset': track.optimal_offset,  # Legacy name
                        # Add nested structure for consistency analyzer
                        'sigma_analysis': {
                            'sigma_gradient': track.sigma_gradient
                        },
                        'linearity_analysis': {
                            'final_linearity_error_shifted': track.final_linearity_error_shifted,
                            'optimal_offset': track.optimal_offset,
                            'linearity_spec': track.linearity_spec
                        },
                        'unit_properties': {
                            'resistance_change_percent': track.resistance_change_percent
                        },
                        # Get raw position/error data from database
                        'position_data': track.position_data if track.position_data else [],
                        'error_data': track.error_data if track.error_data else [],
                        'error_profile': {
                            'positions': track.position_data if track.position_data else [],
                            'errors': track.error_data if track.error_data else [],
                            'spec_limit': track.linearity_spec,
                            'note': 'Database data' if track.position_data else 'Raw data not available'
                        }
                    }
                    
                    # More debug logging
                    self.logger.debug(f"Track {track.track_id} - position_data in track_info: {len(track_info['position_data'])} points")
                    self.logger.debug(f"Track {track.track_id} - error_data in track_info: {len(track_info['error_data'])} points")
                    file_info['tracks'][track.track_id] = track_info

                unit_summary['files'].append(file_info)
                unit_summary['track_count'] += len(analysis.tracks)

            # Determine overall unit status
            all_statuses = [f['status'] for f in unit_summary['files']]
            if 'FAIL' in all_statuses:
                unit_summary['overall_status'] = 'FAIL'
            elif 'WARNING' in all_statuses:
                unit_summary['overall_status'] = 'WARNING'
            elif all(s == 'PASS' for s in all_statuses):
                unit_summary['overall_status'] = 'PASS'
            else:
                unit_summary['overall_status'] = 'MIXED'

            # Calculate consistency metrics
            all_sigma_gradients = []
            all_linearity_errors = []
            all_resistance_changes = []
            
            for file_info in unit_summary['files']:
                for track_info in file_info['tracks'].values():
                    if track_info['sigma_gradient'] is not None:
                        all_sigma_gradients.append(track_info['sigma_gradient'])
                    if track_info['linearity_error'] is not None:
                        all_linearity_errors.append(abs(track_info['linearity_error']))
                    if track_info.get('resistance_change_percent') is not None:
                        all_resistance_changes.append(abs(track_info['resistance_change_percent']))

            if all_sigma_gradients and len(all_sigma_gradients) > 1:
                sigma_cv = (np.std(all_sigma_gradients) / np.mean(all_sigma_gradients)) * 100
                unit_summary['sigma_cv'] = sigma_cv
            else:
                unit_summary['sigma_cv'] = 0

            if all_linearity_errors and len(all_linearity_errors) > 1:
                linearity_cv = (np.std(all_linearity_errors) / abs(np.mean(all_linearity_errors))) * 100
                unit_summary['linearity_cv'] = linearity_cv
            else:
                unit_summary['linearity_cv'] = 0

            if all_resistance_changes and len(all_resistance_changes) > 1:
                resistance_cv = (np.std(all_resistance_changes) / abs(np.mean(all_resistance_changes))) * 100
                unit_summary['resistance_cv'] = resistance_cv
            else:
                unit_summary['resistance_cv'] = 0

            # Determine consistency grade
            if unit_summary['sigma_cv'] < 5 and unit_summary['linearity_cv'] < 10:
                unit_summary['consistency'] = 'EXCELLENT'
            elif unit_summary['sigma_cv'] < 10 and unit_summary['linearity_cv'] < 20:
                unit_summary['consistency'] = 'GOOD'
            elif unit_summary['sigma_cv'] < 20 and unit_summary['linearity_cv'] < 30:
                unit_summary['consistency'] = 'FAIR'
            else:
                unit_summary['consistency'] = 'POOR'

            self.current_unit_data = unit_summary
            
            # Debug logging for final unit summary
            self.logger.debug(f"Unit summary files: {len(unit_summary['files'])}")
            for i, file_info in enumerate(unit_summary['files']):
                self.logger.debug(f"File {i}: {file_info['filename']} - tracks: {list(file_info['tracks'].keys())}")
                for track_id, track_data in file_info['tracks'].items():
                    self.logger.debug(f"  Track {track_id}: position_data={len(track_data.get('position_data', []))} points, error_data={len(track_data.get('error_data', []))} points")
            
            # Check if we have raw data available
            has_raw_data = False
            for file_info in unit_summary['files']:
                for track_data in file_info['tracks'].values():
                    if track_data.get('position_data') and len(track_data['position_data']) > 0:
                        has_raw_data = True
                        break
                if has_raw_data:
                    break
            
            if not has_raw_data:
                self.logger.warning(f"No raw position/error data found for {model}/{serial}. This unit may have been processed before raw data storage was implemented.")
                # Show message to user
                self.after(0, lambda: messagebox.showinfo(
                    "Limited Data Available",
                    f"Unit {model}/{serial} was found in the database, but does not contain raw position/error data.\n\n"
                    "This typically means the unit was processed before raw data storage was implemented.\n\n"
                    "To see full charts and features, please re-process the original Excel file."
                ))
            
            # Update UI in main thread
            self.after(0, self._update_multi_track_display)

            self.logger.info(f"Loaded multi-track data from database: {model}/{serial} - {len(historical_data)} files, has_raw_data={has_raw_data}")

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Failed to load multi-track data from database: {error_msg}")
            self.after(0, lambda: messagebox.showerror(
                "Error", f"Failed to load data from database:\n{error_msg}"
            ))

    def _select_unit_from_database(self):
        """Show dialog to select a unit from the database for multi-track analysis."""
        if not self.db_manager:
            messagebox.showerror("Error", "Database not connected")
            return

        try:
            # Get all unique model/serial combinations that have multiple tracks
            with self.db_manager.get_session() as session:
                from laser_trim_analyzer.database.manager import DBAnalysisResult, DBTrackResult
                
                # Subquery to get model/serial combinations with their track counts
                subq = session.query(
                    DBAnalysisResult.model,
                    DBAnalysisResult.serial,
                    func.count(func.distinct(DBTrackResult.track_id)).label('track_count'),
                    func.count(func.distinct(DBAnalysisResult.id)).label('file_count')
                ).join(
                    DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id
                ).filter(
                    DBAnalysisResult.model.isnot(None),
                    DBAnalysisResult.serial.isnot(None)
                ).group_by(
                    DBAnalysisResult.model,
                    DBAnalysisResult.serial
                ).subquery()
                
                # Main query to get only units with multiple tracks
                results = session.query(
                    subq.c.model,
                    subq.c.serial,
                    subq.c.track_count,
                    subq.c.file_count
                ).filter(
                    subq.c.track_count > 1  # Only units with multiple tracks
                ).order_by(
                    subq.c.model,
                    subq.c.serial
                ).all()

            if not results:
                messagebox.showinfo(
                    "No Multi-Track Units",
                    "No units with multiple tracks found in database.\n\n"
                    "Multi-track units are models that have been analyzed with different track identifiers (TA, TB, TC, etc.)."
                )
                return

            # Show selection dialog
            dialog = tk.Toplevel(self.winfo_toplevel())
            dialog.title("Select Unit for Multi-Track Analysis")
            dialog.geometry("600x400")
            dialog.grab_set()

            # Title
            ttk.Label(
                dialog,
                text="Units with Multiple Tracks:",
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
            for model, serial, track_count, file_count in results:
                display_text = f"{model} / {serial} ({track_count} tracks, {file_count} files)"
                listbox.insert(tk.END, display_text)
                unit_list.append((model, serial))

            listbox.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')

            # Buttons
            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(fill='x', padx=20, pady=(0, 10))

            def analyze_selected():
                selection = listbox.curselection()
                if selection:
                    model, serial = unit_list[selection[0]]
                    dialog.destroy()
                    # Load data from database
                    self.unit_info_label.configure(text=f"Loading data for {model}/{serial}...")
                    threading.Thread(
                        target=self._load_multi_track_from_database,
                        args=(model, serial),
                        daemon=True
                    ).start()

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

        except Exception as e:
            self.logger.error(f"Failed to get units from database: {e}")
            messagebox.showerror("Error", f"Failed to load units:\n{str(e)}")

    def on_show(self):
        """Called when page is shown."""
        pass
    
    def _update_individual_track_viewer(self):
        """Update the individual track viewer with current unit data."""
        if not self.current_unit_data or not hasattr(self, 'individual_track_viewer'):
            return
            
        try:
            # Extract track data from current unit
            tracks_data = {}
            
            # Check if we have file-based data
            if 'files' in self.current_unit_data:
                for file_data in self.current_unit_data.get('files', []):
                    file_tracks = file_data.get('tracks', {})
                    for track_id, track_data in file_tracks.items():
                        # Format track data for viewer with all required fields
                        formatted_track = {
                            'track_id': track_id,
                            'position': track_data.get('position', track_id),
                            'serial': self.current_unit_data.get('serial', 'Unknown'),
                            'timestamp': track_data.get('timestamp'),
                            'overall_status': track_data.get('status', track_data.get('overall_status', 'Unknown')),
                            'validation_status': track_data.get('validation_status', 'Valid' if track_data.get('sigma_pass') and track_data.get('linearity_pass') else 'Invalid'),
                            'sigma_gradient': track_data.get('sigma_gradient'),
                            'sigma_spec': track_data.get('sigma_spec'),
                            'sigma_margin': track_data.get('sigma_margin'),
                            'linearity_error': track_data.get('linearity_error'),
                            'linearity_spec': track_data.get('linearity_spec'),
                            'resistance_change': track_data.get('resistance_change'),
                            'resistance_change_percent': track_data.get('resistance_change_percent'),
                            'trim_stability': track_data.get('trim_stability', 'N/A'),
                            'industry_grade': track_data.get('industry_grade', 'N/A'),
                            'failure_probability': track_data.get('failure_probability', 0),
                            'risk_category': track_data.get('risk_category', 'Unknown'),
                            'error_profile': self._format_error_profile(track_data),
                            'statistics': self._calculate_track_statistics(track_data),
                            'file_path': track_data.get('file_path')
                        }
                        tracks_data[track_id] = formatted_track
                        
            # Check if we have direct tracks data
            elif 'tracks' in self.current_unit_data:
                tracks = self.current_unit_data.get('tracks', {})
                for track_id, track_data in tracks.items():
                    formatted_track = {
                        'track_id': track_id,
                        'position': track_data.get('position', track_id),
                        'serial': track_data.get('serial', self.current_unit_data.get('serial', 'Unknown')),
                        'timestamp': track_data.get('timestamp'),
                        'overall_status': track_data.get('overall_status', 'Unknown'),
                        'validation_status': track_data.get('validation_status', 'Unknown'),
                        'sigma_gradient': track_data.get('sigma_gradient'),
                        'sigma_spec': track_data.get('sigma_spec'),
                        'sigma_margin': track_data.get('sigma_margin'),
                        'linearity_error': track_data.get('linearity_error'),
                        'linearity_spec': track_data.get('linearity_spec'),
                        'resistance_change': track_data.get('resistance_change'),
                        'trim_stability': track_data.get('trim_stability'),
                        'industry_grade': track_data.get('industry_grade', 'N/A'),
                        'error_profile': self._format_error_profile(track_data),
                        'statistics': self._calculate_track_statistics(track_data)
                    }
                    tracks_data[track_id] = formatted_track
                    
            # Load tracks into viewer
            if tracks_data:
                self.individual_track_viewer.load_tracks(tracks_data)
            else:
                # Clear viewer if no tracks
                self.individual_track_viewer.load_tracks({})
                
        except Exception as e:
            self.logger.error(f"Error updating individual track viewer: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _initialize_linearity_plot(self):
        """Initialize the linearity error overlay plot."""
        self.linearity_ax.clear()
        
        # Set white background for better visibility
        self.linearity_ax.set_facecolor('white')
        self.linearity_ax.tick_params(colors='black', labelcolor='black')
        for spine in self.linearity_ax.spines.values():
            spine.set_color('#cccccc')
        
        # Set labels and title
        self.linearity_ax.set_xlabel('Position (mm)', fontsize=12, color='black')
        self.linearity_ax.set_ylabel('Linearity Error (Volts)', fontsize=12, color='black')
        self.linearity_ax.set_title('Track Linearity Error Overlay', fontsize=14, fontweight='bold', color='black')
        
        # Add grid
        self.linearity_ax.grid(True, alpha=0.3, color='#cccccc')
        
        # Add a message if no data
        self.linearity_ax.text(0.5, 0.5, 'No track data loaded', 
                              transform=self.linearity_ax.transAxes,
                              horizontalalignment='center',
                              verticalalignment='center',
                              fontsize=12, color='gray')
        
        self.linearity_fig.tight_layout()
        self.linearity_canvas.draw()
    
    def _update_linearity_overlay(self):
        """Update the linearity error overlay plot with track data."""
        self.linearity_ax.clear()
        
        # Set white background
        self.linearity_ax.set_facecolor('white')
        self.linearity_ax.tick_params(colors='black', labelcolor='black')
        for spine in self.linearity_ax.spines.values():
            spine.set_color('#cccccc')
        
        if not self.current_unit_data:
            self._initialize_linearity_plot()
            return
            
        # Color palette for different tracks
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        track_count = 0
        spec_limit = 5.0  # Default spec
        tracks_plotted = {}  # Keep track of unique tracks
        
        # Collect unique tracks (avoid duplicates)
        all_tracks = {}
        
        # Check different data formats
        if 'tracks' in self.current_unit_data:
            # Direct tracks format
            all_tracks = self.current_unit_data['tracks']
        elif 'files' in self.current_unit_data:
            # Files format - collect unique tracks
            for file_data in self.current_unit_data.get('files', []):
                tracks = file_data.get('tracks', {})
                for track_id, track_data in tracks.items():
                    if track_id not in all_tracks:
                        all_tracks[track_id] = track_data
        
        # Plot each unique track's error profile
        for track_id, track_data in all_tracks.items():
            if track_id in tracks_plotted:
                continue  # Skip duplicates
                
            color = colors[track_count % len(colors)]
            
            # Get error profile data - only use real data, not synthetic
            error_profile = track_data.get('error_profile', {})
            
            # Check if we have position_data and error_data
            positions = []
            errors = []
            untrimmed_positions = []
            untrimmed_errors = []
            
            # First try to get untrimmed data (full dataset)
            if 'untrimmed_positions' in track_data and 'untrimmed_errors' in track_data:
                untrimmed_positions = track_data.get('untrimmed_positions', [])
                untrimmed_errors = track_data.get('untrimmed_errors', [])
                if untrimmed_positions:
                    self.logger.info(f"Track {track_id}: Found untrimmed data ({len(untrimmed_positions)} points)")
                    self.logger.info(f"Track {track_id}: Untrimmed position range: [{min(untrimmed_positions):.1f}, {max(untrimmed_positions):.1f}]")
            
            # Get trimmed data (final trim data only)
            if 'position_data' in track_data and 'error_data' in track_data:
                positions = track_data.get('position_data', [])
                errors = track_data.get('error_data', [])
                if positions:
                    self.logger.info(f"Track {track_id}: Using trimmed position_data ({len(positions)} points)")
                    self.logger.info(f"Track {track_id}: Position range from trimmed: [{min(positions):.1f}, {max(positions):.1f}]")
            
            # Last resort: check error_profile (but don't filter synthetic data)
            if not positions and error_profile and 'positions' in error_profile and 'errors' in error_profile:
                positions = error_profile.get('positions', [])
                errors = error_profile.get('errors', [])
                self.logger.info(f"Track {track_id}: Using error_profile data ({len(positions)} points)")
                if positions:
                    self.logger.info(f"Track {track_id}: Position range from error_profile: [{min(positions):.1f}, {max(positions):.1f}]")
            
            # Only use trimmed data for multi-track page (final trim data)
            if not positions:
                self.logger.debug(f"Track {track_id}: No trimmed position/error data found")
                # Check if this is from database without raw data
                if 'files' in self.current_unit_data:
                    self.logger.info(f"Track {track_id}: This appears to be database data without raw position/error arrays")
            
            if positions and errors and len(positions) == len(errors) and len(positions) > 0:
                pos_min, pos_max = min(positions), max(positions)
                self.logger.info(f"Plotting track {track_id} with {len(positions)} points, position range: [{pos_min:.1f}, {pos_max:.1f}]")
                
                # Apply offset if available (to get shifted linearity error)
                # Try to get offset from linearity analysis first
                offset = 0.0
                if 'linearity_analysis' in track_data and track_data['linearity_analysis']:
                    linearity_analysis = track_data['linearity_analysis']
                    if isinstance(linearity_analysis, dict):
                        offset = linearity_analysis.get('optimal_offset', 0.0)
                    else:
                        # Handle object attributes
                        offset = getattr(linearity_analysis, 'optimal_offset', 0.0)
                
                # Fallback to legacy field name
                if offset == 0.0:
                    offset = track_data.get('linearity_offset', 0.0)
                
                if offset != 0.0:
                    self.logger.info(f"Track {track_id}: Applying offset {offset:.6f}")
                    # Log some statistics before and after offset
                    errors_min, errors_max = min(errors), max(errors)
                    errors_mean = sum(errors) / len(errors)
                    self.logger.info(f"Track {track_id}: Before offset - min={errors_min:.4f}, max={errors_max:.4f}, mean={errors_mean:.4f}")
                    
                    # Note: The analyzer ADDS the offset to errors, so we do the same here
                    shifted_errors = [e + offset for e in errors]
                    
                    shifted_min, shifted_max = min(shifted_errors), max(shifted_errors)
                    shifted_mean = sum(shifted_errors) / len(shifted_errors)
                    self.logger.info(f"Track {track_id}: After offset - min={shifted_min:.4f}, max={shifted_max:.4f}, mean={shifted_mean:.4f}")
                else:
                    self.logger.warning(f"Track {track_id}: No offset available (offset=0.0)")
                    shifted_errors = errors
                
                # Plot the error profile with shifted values
                self.linearity_ax.plot(positions, shifted_errors, 
                                     label=f'Track {track_id}', 
                                     color=color, 
                                     linewidth=2,
                                     alpha=0.8)
                tracks_plotted[track_id] = True
                track_count += 1
                
                # Get spec limit from track data
                if 'linearity_spec' in track_data:
                    spec_limit = track_data['linearity_spec']
                elif error_profile and 'spec_limit' in error_profile:
                    spec_limit = error_profile['spec_limit']
            else:
                self.logger.warning(f"Track {track_id}: Cannot plot - positions:{len(positions) if positions else 0}, errors:{len(errors) if errors else 0}")
        
        # Add spec limit lines
        if track_count > 0:
            self.linearity_ax.axhline(y=spec_limit, color='red', linestyle='--', 
                                    linewidth=2, alpha=0.7, label=f'Spec Limit: ±{spec_limit:.4f}V')
            self.linearity_ax.axhline(y=-spec_limit, color='red', linestyle='--', 
                                    linewidth=2, alpha=0.7)
            
            # Add zero line
            self.linearity_ax.axhline(y=0, color='black', linestyle='-', 
                                    linewidth=1, alpha=0.5)
        
        # Set labels and title
        self.linearity_ax.set_xlabel('Position (mm)', fontsize=12, color='black')
        self.linearity_ax.set_ylabel('Linearity Error (Volts)', fontsize=12, color='black')
        self.linearity_ax.set_title(f'Track Linearity Error Overlay - {self.current_unit_data.get("model", "Unknown")} {self.current_unit_data.get("serial", "Unknown")}', 
                                  fontsize=14, fontweight='bold', color='black')
        
        # Add grid
        self.linearity_ax.grid(True, alpha=0.3, color='#cccccc')
        
        # Add legend if tracks were plotted
        if track_count > 0:
            self.linearity_ax.legend(loc='best', framealpha=0.9)
            
            # Add note about offset application
            self.linearity_ax.text(0.02, 0.98, 'Note: Optimal offset applied', 
                                  transform=self.linearity_ax.transAxes,
                                  verticalalignment='top',
                                  fontsize=10, 
                                  style='italic',
                                  color='gray',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
        else:
            # Check if this is database data
            from_database = False
            if all_tracks:
                # Check if any track has the database note
                for track_data in all_tracks.values():
                    if isinstance(track_data.get('error_profile'), dict):
                        if track_data['error_profile'].get('note') == 'Raw data not available from database':
                            from_database = True
                            break
            
            if from_database:
                message = 'Error profile data not available from database\n\nDatabase only stores calculated metrics.\nLoad files directly to see error profiles.'
            else:
                message = 'No error profile data available'
                
            self.linearity_ax.text(0.5, 0.5, message, 
                                  transform=self.linearity_ax.transAxes,
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  fontsize=12, color='gray',
                                  multialignment='center')
        
        # Set scaling based on slider value
        if track_count > 0:
            # Get scale factor (default to 1.5 if not initialized yet)
            scale_factor = self.y_scale_var.get() if hasattr(self, 'y_scale_var') else 1.5
            
            # Calculate actual x-axis range from plotted data
            all_positions = []
            for track_id, track_data in all_tracks.items():
                if track_id in tracks_plotted:
                    # Get the positions that were actually plotted
                    if 'untrimmed_positions' in track_data and track_data['untrimmed_positions']:
                        all_positions.extend(track_data['untrimmed_positions'])
                    elif 'position_data' in track_data and track_data['position_data']:
                        all_positions.extend(track_data['position_data'])
            
            if all_positions:
                x_min = min(all_positions)
                x_max = max(all_positions)
                x_range = x_max - x_min
                x_padding = x_range * 0.05  # 5% padding
                self.linearity_ax.set_xlim(x_min - x_padding, x_max + x_padding)
                self.logger.info(f"X-axis range: [{x_min - x_padding:.1f}, {x_max + x_padding:.1f}] mm")
            
            # Set y-axis based on scale factor and spec limits with additional padding
            y_limit = spec_limit * scale_factor
            y_padding = y_limit * 0.1  # Add 10% padding
            self.linearity_ax.set_ylim(-y_limit - y_padding, y_limit + y_padding)
            
            self.logger.info(f"Chart scaled to ±{y_limit:.1f} (spec: {spec_limit}, scale: {scale_factor}x)")
        
        self.linearity_fig.tight_layout()
        self.linearity_canvas.draw()
    
    def _on_scale_changed(self, value):
        """Handle scale slider change."""
        scale_value = self.y_scale_var.get()
        self.scale_value_label.configure(text=f"{scale_value:.1f}x")
        # Update the plot with new scaling
        self._update_linearity_overlay()
    
    def _export_linearity_chart(self):
        """Export the linearity overlay chart."""
        if not self.current_unit_data:
            messagebox.showwarning("No Data", "No data to export")
            return
            
        # Get filename
        default_filename = f"linearity_overlay_{self.current_unit_data.get('model', 'Unknown')}_{self.current_unit_data.get('serial', 'Unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        filename = filedialog.asksaveasfilename(
            title="Export Linearity Overlay Chart",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ],
            initialfile=default_filename
        )
        
        if filename:
            try:
                # Save the figure
                self.linearity_fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Export Successful", f"Chart exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export chart: {str(e)}")
    
    def _extract_track_id(self, file_path: Path) -> str:
        """Extract track ID from filename."""
        file_parts = file_path.stem.split('_')
        
        # Look for track identifier like TA, TB, TC
        for part in file_parts:
            if len(part) == 2 and part[0] == 'T' and part[1].isalpha():
                return part
                
        # Default to TA if no track identifier found
        return 'TA'
    
    def _calculate_consistency_rating(self, sigma_cv: float, linearity_cv: float, resistance_cv: float) -> str:
        """Calculate overall consistency rating based on CV values."""
        if sigma_cv == 0 and linearity_cv == 0 and resistance_cv == 0:
            return 'SINGLE_TRACK'
            
        # Use similar thresholds as ConsistencyAnalyzer
        if sigma_cv > 15 or linearity_cv > 30 or resistance_cv > 10:
            return 'POOR'
        elif sigma_cv > 10 or linearity_cv > 20 or resistance_cv > 5:
            return 'FAIR'
        elif sigma_cv > 5 or linearity_cv > 10 or resistance_cv > 2:
            return 'GOOD'
        else:
            return 'EXCELLENT'
    
    def _format_error_profile(self, track_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format error profile data for track viewer."""
        error_profile = track_data.get('error_profile', {})
        
        # If error_profile already has the correct structure, return it
        if isinstance(error_profile, dict) and 'positions' in error_profile and 'errors' in error_profile:
            return error_profile
            
        # Try to create error profile from other data
        profile = {
            'positions': [],
            'errors': [],
            'spec_limit': track_data.get('linearity_spec', 5.0)
        }
        
        # Check if we have position and error data in other fields
        if 'position_data' in track_data and 'error_data' in track_data:
            profile['positions'] = track_data['position_data']
            profile['errors'] = track_data['error_data']
        elif 'measurements' in track_data:
            # Extract from measurements if available
            measurements = track_data['measurements']
            if isinstance(measurements, list):
                positions = []
                errors = []
                for m in measurements:
                    if isinstance(m, dict):
                        if 'position' in m:
                            positions.append(m['position'])
                        if 'error' in m:
                            errors.append(m['error'])
                if positions and errors and len(positions) == len(errors):
                    profile['positions'] = positions
                    profile['errors'] = errors
        
        # Don't generate synthetic data - only use real data
            
        return profile
    
    def _calculate_track_statistics(self, track_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for track data."""
        stats = {
            'mean_error': 0,
            'std_error': 0,
            'max_error': 0,
            'min_error': 0,
            'data_points': 0
        }
        
        # Try to get error data from nested structure
        if 'linearity_analysis' in track_data and isinstance(track_data['linearity_analysis'], dict):
            lin_analysis = track_data['linearity_analysis']
            if 'error_data' in lin_analysis and lin_analysis['error_data']:
                errors = lin_analysis['error_data']
                stats['mean_error'] = np.mean(errors)
                stats['std_error'] = np.std(errors)
                stats['max_error'] = np.max(errors)
                stats['min_error'] = np.min(errors)
                stats['data_points'] = len(errors)
        
        return stats
    
    def _load_test_data(self):
        """Load test data for debugging."""
        # Create sample multi-track data
        test_data = {
            'model': 'TEST_MODEL',
            'serial': 'TEST_001',
            'total_files': 3,
            'track_count': 3,
            'overall_status': 'PASS',
            'consistency': 'GOOD',
            'sigma_cv': 8.5,
            'linearity_cv': 15.2,
            'resistance_cv': 3.7,
            'files': [
                {
                    'filename': 'test_file_1.xlsx',
                    'status': 'PASS',
                    'track_count': 3,
                    'tracks': {
                        'TA': {
                            'track_id': 'TA',
                            'status': 'PASS',
                            'overall_status': 'PASS',
                            'sigma_gradient': 0.00045,
                            'sigma_spec': 0.0006,
                            'sigma_margin': 0.25,
                            'linearity_error': 0.25,
                            'linearity_spec': 0.5,
                            'resistance_change': 2.5,
                            'resistance_change_percent': 2.5,
                            'trim_stability': 0.95,
                            'industry_grade': 'A',
                            'sigma_pass': True,
                            'linearity_pass': True,
                            'validation_status': 'Valid',
                            'timestamp': datetime.now().isoformat(),
                            'position': 'A',
                            'sigma_analysis': {'sigma_gradient': 0.00045},
                            'linearity_analysis': {'final_linearity_error_shifted': 0.25},
                            'unit_properties': {'resistance_change_percent': 2.5},
                            # Add position and error data for plotting
                            'position_data': list(range(0, 101, 2)),  # 0, 2, 4, ... 100
                            'error_data': [0.1 * (i % 10 - 5) + 0.05 * np.sin(i/10) for i in range(0, 101, 2)]
                        },
                        'TB': {
                            'track_id': 'TB',
                            'status': 'PASS',
                            'overall_status': 'PASS',
                            'sigma_gradient': 0.00048,
                            'sigma_spec': 0.0006,
                            'sigma_margin': 0.20,
                            'linearity_error': 0.28,
                            'linearity_spec': 0.5,
                            'resistance_change': 2.8,
                            'resistance_change_percent': 2.8,
                            'trim_stability': 0.92,
                            'industry_grade': 'B',
                            'sigma_pass': True,
                            'linearity_pass': True,
                            'validation_status': 'Valid',
                            'timestamp': datetime.now().isoformat(),
                            'position': 'B',
                            'sigma_analysis': {'sigma_gradient': 0.00048},
                            'linearity_analysis': {'final_linearity_error_shifted': 0.28},
                            'unit_properties': {'resistance_change_percent': 2.8},
                            # Add position and error data for plotting
                            'position_data': list(range(0, 101, 2)),  # 0, 2, 4, ... 100
                            'error_data': [0.15 * (i % 10 - 5) + 0.08 * np.sin(i/8) for i in range(0, 101, 2)]
                        },
                        'TC': {
                            'track_id': 'TC',
                            'status': 'WARNING',
                            'overall_status': 'WARNING',
                            'sigma_gradient': 0.00052,
                            'sigma_spec': 0.0006,
                            'sigma_margin': 0.13,
                            'linearity_error': 0.35,
                            'linearity_spec': 0.5,
                            'resistance_change': 3.2,
                            'resistance_change_percent': 3.2,
                            'trim_stability': 0.88,
                            'industry_grade': 'C',
                            'sigma_pass': False,
                            'linearity_pass': True,
                            'validation_status': 'Warning',
                            'timestamp': datetime.now().isoformat(),
                            'position': 'C',
                            'sigma_analysis': {'sigma_gradient': 0.00052},
                            'linearity_analysis': {'final_linearity_error_shifted': 0.35},
                            'unit_properties': {'resistance_change_percent': 3.2},
                            # Add position and error data for plotting
                            'position_data': list(range(0, 101, 2)),  # 0, 2, 4, ... 100
                            'error_data': [0.2 * (i % 10 - 5) + 0.1 * np.sin(i/6) for i in range(0, 101, 2)]
                        }
                    }
                }
            ]
        }
        
        self.current_unit_data = test_data
        self._update_multi_track_display()
        self.logger.info("Test data loaded successfully")
            
    def show(self):
        """Show the page using grid layout."""
        self.grid(row=0, column=0, sticky="nsew")
        self.is_visible = True
        
        # Refresh if needed
        if self.needs_refresh:
            self.refresh()
            self.needs_refresh = False
            
        self.on_show()
        
    def hide(self):
        """Hide the page."""
        self.grid_remove()
        self.is_visible = False
        
    def refresh(self):
        """Refresh page content."""
        pass
        
    def _create_error_page(self, error_msg: str):
        """Create an error display page."""
        error_frame = ctk.CTkFrame(self)
        error_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        error_label = ctk.CTkLabel(
            error_frame,
            text=f"Error initializing {self.__class__.__name__}",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="red"
        )
        error_label.pack(pady=(20, 10))
        
        detail_label = ctk.CTkLabel(
            error_frame,
            text=error_msg,
            font=ctk.CTkFont(size=14),
            wraplength=600
        )
        detail_label.pack(pady=10)
    
    def _extract_track_id(self, file_path: Path) -> str:
        """Extract track ID from filename."""
        # Try to extract track ID from filename pattern Model_Serial_TrackID.xlsx
        parts = file_path.stem.split('_')
        if len(parts) >= 3:
            # Last part might be track ID
            potential_track = parts[-1]
            if len(potential_track) == 2 and potential_track[0] == 'T' and potential_track[1].isalpha():
                return potential_track
        
        # Fallback: use last two characters if they look like a track ID
        stem = file_path.stem
        if len(stem) >= 2:
            last_two = stem[-2:]
            if last_two[0] == 'T' and last_two[1].isalpha():
                return last_two
        
        # Final fallback: generate from filename
        return f"T{file_path.stem[-1].upper()}" if file_path.stem else "TX"
    
    def _calculate_consistency_rating(self, sigma_cv: float, linearity_cv: float, resistance_cv: float) -> str:
        """Calculate overall consistency rating based on CV values."""
        # Define thresholds
        excellent_threshold = {'sigma': 5, 'linearity': 10, 'resistance': 2}
        good_threshold = {'sigma': 10, 'linearity': 20, 'resistance': 5}
        acceptable_threshold = {'sigma': 15, 'linearity': 30, 'resistance': 10}
        
        # Check each metric
        if (sigma_cv <= excellent_threshold['sigma'] and 
            linearity_cv <= excellent_threshold['linearity'] and 
            resistance_cv <= excellent_threshold['resistance']):
            return 'EXCELLENT'
        elif (sigma_cv <= good_threshold['sigma'] and 
              linearity_cv <= good_threshold['linearity'] and 
              resistance_cv <= good_threshold['resistance']):
            return 'GOOD'
        elif (sigma_cv <= acceptable_threshold['sigma'] and 
              linearity_cv <= acceptable_threshold['linearity'] and 
              resistance_cv <= acceptable_threshold['resistance']):
            return 'ACCEPTABLE'
        else:
            return 'POOR' 