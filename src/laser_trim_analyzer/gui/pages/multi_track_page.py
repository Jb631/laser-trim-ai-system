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

from laser_trim_analyzer.gui.pages.base_page_ctk import BasePage
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.gui.widgets.stat_card import StatCard
from laser_trim_analyzer.gui.widgets.metric_card_ctk import MetricCard
from laser_trim_analyzer.gui.widgets import add_mousewheel_support
from laser_trim_analyzer.gui.widgets.track_viewer import IndividualTrackViewer
from laser_trim_analyzer.analysis.consistency_analyzer import ConsistencyAnalyzer

# Get logger
logger = logging.getLogger(__name__)

class MultiTrackPage(BasePage):
    """Multi-track analysis and comparison page."""

    def __init__(self, parent, main_window):
        self.current_unit_data = None
        self.comparison_data = None
        self.consistency_analyzer = ConsistencyAnalyzer()
        super().__init__(parent, main_window)

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

        self.analyze_folder_btn = ctk.CTkButton(
            button_frame,
            text="📂 Analyze Folder",
            command=self._analyze_folder,
            width=150,
            height=40
        )
        self.analyze_folder_btn.pack(side='left', padx=(0, 10), pady=10)

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
        """Create track comparison charts section (matching batch processing theme)."""
        self.comparison_frame = ctk.CTkFrame(self.main_container)
        self.comparison_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.comparison_label = ctk.CTkLabel(
            self.comparison_frame,
            text="Track Comparison:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.comparison_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Comparison container with tabs
        self.comparison_container = ctk.CTkFrame(self.comparison_frame, fg_color="transparent")
        self.comparison_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Comparison tabs
        self.comparison_tabview = ctk.CTkTabview(self.comparison_container)
        self.comparison_tabview.pack(fill='both', expand=True, padx=10, pady=10)

        # Add tabs
        self.comparison_tabview.add("Summary")
        self.comparison_tabview.add("Detailed")
        self.comparison_tabview.add("Charts")

        # Create actual chart widgets instead of placeholders
        try:
            from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
            
            # Summary chart for overview metrics
            self.summary_chart = ChartWidget(
                self.comparison_tabview.tab("Summary"),
                chart_type='bar',
                title="Track Summary Comparison",
                figsize=(10, 4)
            )
            self.summary_chart.pack(fill='both', expand=True)
            
            # Detailed comparison charts
            detailed_frame = ctk.CTkFrame(self.comparison_tabview.tab("Detailed"), fg_color="transparent")
            detailed_frame.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Sigma comparison chart
            self.sigma_comparison_chart = ChartWidget(
                detailed_frame,
                chart_type='bar',
                title="Sigma Gradient Comparison",
                figsize=(10, 3)
            )
            self.sigma_comparison_chart.pack(fill='x', padx=5, pady=(5, 2))
            
            # Linearity comparison chart
            self.linearity_comparison_chart = ChartWidget(
                detailed_frame,
                chart_type='bar',
                title="Linearity Error Comparison",
                figsize=(10, 3)
            )
            self.linearity_comparison_chart.pack(fill='x', padx=5, pady=(2, 5))
            
            # Charts tab for profile comparisons
            self.profile_comparison_chart = ChartWidget(
                self.comparison_tabview.tab("Charts"),
                chart_type='line',
                title="Error Profile Comparison",
                figsize=(10, 4)
            )
            self.profile_comparison_chart.pack(fill='both', expand=True)
            
            # Initialize comparison charts dictionary for easy access
            self.comparison_charts = {
                'summary': self.summary_chart,
                'sigma': self.sigma_comparison_chart,
                'linearity': self.linearity_comparison_chart,
                'profile': self.profile_comparison_chart
            }
            
        except ImportError as e:
            logger.warning(f"Could not create chart widgets: {e}")
            # Fallback to placeholder labels
            self.summary_chart_label = ctk.CTkLabel(
                self.comparison_tabview.tab("Summary"),
                text="Chart widgets not available",
                font=ctk.CTkFont(size=12)
            )
            self.summary_chart_label.pack(expand=True)

            self.detailed_chart_label = ctk.CTkLabel(
                self.comparison_tabview.tab("Detailed"),
                text="Chart widgets not available",
                font=ctk.CTkFont(size=12)
            )
            self.detailed_chart_label.pack(expand=True)

            self.charts_chart_label = ctk.CTkLabel(
                self.comparison_tabview.tab("Charts"),
                text="Chart widgets not available",
                font=ctk.CTkFont(size=12)
            )
            self.charts_chart_label.pack(expand=True)
            
            # Set chart objects to None for safe access
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
            text="Export & Actions:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.actions_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Actions container
        self.actions_container = ctk.CTkFrame(self.actions_frame, fg_color="transparent")
        self.actions_container.pack(fill='x', padx=15, pady=(0, 15))

        # Action buttons
        button_frame = ctk.CTkFrame(self.actions_container, fg_color="transparent")
        button_frame.pack(fill='x', padx=10, pady=(10, 10))

        self.export_report_btn = ctk.CTkButton(
            button_frame,
            text="📊 Export Comparison Report",
            command=self._export_comparison_report,
            width=200,
            height=40
        )
        self.export_report_btn.pack(side='left', padx=(10, 10), pady=10)

        self.generate_pdf_btn = ctk.CTkButton(
            button_frame,
            text="📄 Generate PDF Report",
            command=self._generate_pdf_report,
            width=180,
            height=40
        )
        self.generate_pdf_btn.pack(side='left', padx=(0, 10), pady=10)

        self.view_tracks_btn = ctk.CTkButton(
            button_frame,
            text="👁️ View Individual Tracks",
            command=self._view_individual_tracks,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.view_tracks_btn.pack(side='left', padx=(0, 10), pady=10)

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

    def _analyze_folder_tracks(self, folder_path: Path):
        """Analyze all track files in a folder and group by units."""
        self.unit_info_label.configure(text=f"Analyzing folder: {folder_path.name}...")
        self.update()
        
        # Run analysis in background thread
        threading.Thread(
            target=self._run_folder_analysis,
            args=(folder_path,),
            daemon=True
        ).start()

    def _run_file_analysis(self, file_path: Path):
        """Run analysis for a single track file and find related tracks."""
        try:
            self.after(0, lambda: self.unit_info_label.configure(text=f"Analyzing {file_path.name}..."))
            
            # Extract unit information from filename
            filename = file_path.stem
            parts = filename.split('_')
            
            if len(parts) < 2:
                self.after(0, lambda: messagebox.showerror(
                    "Invalid File", 
                    "Filename must contain model and serial (e.g., 8340_12345_TA.xlsx)"
                ))
                return
            
            model = parts[0]
            serial = parts[1]
            
            # Extract track ID if present
            current_track_id = None
            for part in parts:
                if len(part) == 2 and part[0] == 'T' and part[1].isalpha():
                    current_track_id = part
                    break
            
            # Search for related track files in the same directory
            folder_path = file_path.parent
            excel_files = list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls"))
            
            # Find files with same model/serial
            related_files = []
            for excel_file in excel_files:
                file_parts = excel_file.stem.split('_')
                if len(file_parts) >= 2 and file_parts[0] == model and file_parts[1] == serial:
                    # Check if it has a track identifier
                    has_track_id = any(len(part) == 2 and part[0] == 'T' and part[1].isalpha() 
                                     for part in file_parts)
                    if has_track_id:
                        related_files.append(excel_file)
            
            if len(related_files) < 2:
                # Single track file - still analyze but note it's not multi-track
                self.after(0, lambda: messagebox.showinfo(
                    "Single Track", 
                    f"Only one track file found for {model}/{serial}.\n"
                    "Analyzing as single track. Use 'From Database' to find historical tracks."
                ))
                
                # Create single track unit data
                self.current_unit_data = {
                    'model': model,
                    'serial': serial,
                    'total_files': 1,
                    'track_count': 1,
                    'overall_status': 'UNKNOWN',
                    'files': [{
                        'filename': file_path.name,
                        'file_path': str(file_path),
                        'status': 'PENDING',
                        'track_count': 1,
                        'tracks': {current_track_id or 'T1': {
                            'track_id': current_track_id or 'T1',
                            'status': 'PENDING',
                            'file_path': str(file_path)
                        }}
                    }],
                    'sigma_cv': 0,
                    'linearity_cv': 0,
                    'consistency': 'SINGLE_TRACK'
                }
                
                self.after(0, self._update_multi_track_display)
                return
            
            # Multiple track files found
            track_data = []
            for track_file in related_files:
                file_parts = track_file.stem.split('_')
                track_id = None
                for part in file_parts:
                    if len(part) == 2 and part[0] == 'T' and part[1].isalpha():
                        track_id = part
                        break
                
                track_data.append({
                    'track_id': track_id or f'T{len(track_data)+1}',
                    'file_path': str(track_file),
                    'filename': track_file.name,
                    'status': 'FOUND'
                })
            
            # Sort tracks by track ID
            track_data.sort(key=lambda x: x['track_id'])
            
            # Create multi-track unit data
            tracks_dict = {track['track_id']: track for track in track_data}
            
            self.current_unit_data = {
                'model': model,
                'serial': serial,
                'total_files': len(related_files),
                'track_count': len(related_files),
                'overall_status': 'FOUND',
                'files': [{
                    'filename': f"{model}_{serial}_MultiTrack.xlsx",
                    'status': 'FOUND',
                    'track_count': len(related_files),
                    'tracks': tracks_dict
                }],
                'sigma_cv': 0,  # Will be calculated if analysis is run
                'linearity_cv': 0,  # Will be calculated if analysis is run
                'consistency': 'PENDING_ANALYSIS'
            }
            
            # Update UI
            self.after(0, self._update_multi_track_display)
            
            # Show confirmation dialog
            track_list = ', '.join([track['track_id'] for track in track_data])
            self.after(0, lambda: messagebox.showinfo(
                "Multi-Track Unit Found",
                f"Found {len(related_files)} track files for {model}/{serial}:\n{track_list}\n\n"
                "Track files have been grouped for comparison analysis."
            ))
            
        except Exception as e:
            self.logger.error(f"File analysis failed: {e}")
            self.after(0, lambda: messagebox.showerror(
                "Error", f"File analysis failed:\n{str(e)}"
            ))

    def _run_folder_analysis(self, folder_path: Path):
        """Run folder analysis to find and group multi-track units."""
        try:
            # Find all Excel files in folder
            excel_files = list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls"))
            
            if not excel_files:
                self.after(0, lambda: messagebox.showwarning(
                    "No Files", "No Excel files found in selected folder"
                ))
                return
            
            # Group files by unit (model_serial)
            unit_groups = {}
            
            for file_path in excel_files:
                filename = file_path.stem
                parts = filename.split('_')
                
                if len(parts) >= 2:
                    model = parts[0]
                    serial = parts[1]
                    unit_key = f"{model}_{serial}"
                    
                    # Check if this is a track file (has TA, TB, etc.)
                    track_id = None
                    for part in parts:
                        if len(part) == 2 and part[0] == 'T' and part[1].isalpha():
                            track_id = part
                            break
                    
                    if track_id:
                        if unit_key not in unit_groups:
                            unit_groups[unit_key] = []
                        unit_groups[unit_key].append(file_path)
            
            # Find units with multiple tracks
            multi_track_units = {k: v for k, v in unit_groups.items() if len(v) > 1}
            
            if not multi_track_units:
                self.after(0, lambda: messagebox.showinfo(
                    "No Multi-Track Units", 
                    "No multi-track units found in folder.\n"
                    "Looking for files with TA, TB, etc. identifiers."
                ))
                return
            
            # Show selection dialog
            self.after(0, lambda: self._show_unit_selection_dialog(multi_track_units))

        except Exception as e:
            self.logger.error(f"Folder analysis failed: {e}")
            self.after(0, lambda: messagebox.showerror(
                "Error", f"Folder analysis failed:\n{str(e)}"
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
                
                self.overview_cards['sigma_cv'].update_value(f"{sigma_cv:.1f}")
                self.overview_cards['linearity_cv'].update_value(f"{linearity_cv:.1f}")
                
                # Set color schemes based on variation
                sigma_color = 'success' if sigma_cv < 5 else 'warning' if sigma_cv < 10 else 'danger'
                linearity_color = 'success' if linearity_cv < 10 else 'warning' if linearity_cv < 20 else 'danger'
                
                self.overview_cards['sigma_cv'].set_color_scheme(sigma_color)
                self.overview_cards['linearity_cv'].set_color_scheme(linearity_color)
                
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
            for file_info in self.current_unit_data['files']:
                for track_id, track_data in file_info['tracks'].items():
                    # Use file+track as unique key for comparison
                    display_id = f"{file_info['filename'].split('_')[0]}_{track_id}"
                    self.comparison_data['sigma_comparison']['values'][display_id] = track_data.get('sigma_gradient', 0)
                    self.comparison_data['linearity_comparison']['values'][display_id] = track_data.get('linearity_error', 0)
    
    def _update_comparison_charts(self):
        """Update comparison charts with track data."""
        if not self.comparison_data or not self.comparison_data.get('comparison_performed'):
            return

        comparison = self.comparison_data

        # Sigma comparison chart
        if 'sigma_comparison' in comparison and comparison['sigma_comparison']:
            sigma_data = comparison['sigma_comparison']['values']
            
            self.sigma_comparison_chart.clear_chart()
            self.sigma_comparison_chart.plot_bar(
                categories=list(sigma_data.keys()),
                values=list(sigma_data.values()),
                colors=['primary'] * len(sigma_data),
                xlabel="Track ID",
                ylabel="Sigma Gradient"
            )

        # Linearity comparison chart
        if 'linearity_comparison' in comparison and comparison['linearity_comparison']:
            linearity_data = comparison['linearity_comparison']['values']
            
            self.linearity_comparison_chart.clear_chart()
            self.linearity_comparison_chart.plot_bar(
                categories=list(linearity_data.keys()),
                values=list(linearity_data.values()),
                colors=['warning'] * len(linearity_data),
                xlabel="Track ID",
                ylabel="Linearity Error"
            )

        # Error profile comparison (if position data available)
        self._update_error_profiles()

    def _update_error_profiles(self):
        """Update error profile comparison chart."""
        if not self.current_unit_data or not self.current_unit_data.get('tracks'):
            return

        self.profile_comparison_chart.clear_chart()
        
        # Plot error profiles for each track
        colors = ['primary', 'success', 'warning', 'danger', 'info']
        color_idx = 0
        
        for track_id, result in self.current_unit_data['tracks'].items():
            primary_track = result.primary_track
            if primary_track and primary_track.position_data and primary_track.error_data:
                
                color = colors[color_idx % len(colors)]
                
                self.profile_comparison_chart.plot_line(
                    x_data=primary_track.position_data,
                    y_data=primary_track.error_data,
                    label=f"Track {track_id}",
                    color=color,
                    alpha=0.7
                )
                
                color_idx += 1

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
            initial_filename = f"multitrack_report_{self.current_unit_data['unit_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
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
                    'Unit ID': self.current_unit_data['unit_id'],
                    'Track Count': len(self.current_unit_data['tracks']),
                    'Overall Status': self.current_unit_data['overall_status'].value,
                    'Overall Validation Status': getattr(self.current_unit_data.get('tracks', {}).get(list(self.current_unit_data['tracks'].keys())[0]), 'overall_validation_status', 'Not Available'),
                    'Validation Grade': getattr(self.current_unit_data.get('tracks', {}).get(list(self.current_unit_data['tracks'].keys())[0]), 'overall_validation_grade', 'N/A'),
                    'Has Issues': self.current_unit_data.get('has_multi_track_issues', False),
                    'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }])
                unit_summary.to_excel(writer, sheet_name='Unit Summary', index=False)
                
                # Track comparison data
                if self.comparison_data and self.comparison_data.get('comparison_performed'):
                    comparison_df = pd.DataFrame([self.comparison_data])
                    comparison_df.to_excel(writer, sheet_name='Comparison Analysis', index=False)
                
                # Individual track data with validation information
                track_data = []
                for track_id, result in self.current_unit_data['tracks'].items():
                    primary_track = result.primary_track
                    if primary_track:
                        track_info = {
                            'Track ID': track_id,
                            'Status': primary_track.status.value,
                            'Sigma Gradient': primary_track.sigma_analysis.sigma_gradient,
                            'Sigma Pass': primary_track.sigma_analysis.sigma_pass,
                            'Sigma Validation Status': primary_track.sigma_analysis.validation_status.value if hasattr(primary_track.sigma_analysis, 'validation_status') else 'Not Validated',
                            'Sigma Industry Compliance': primary_track.sigma_analysis.industry_compliance if hasattr(primary_track.sigma_analysis, 'industry_compliance') else 'Not Available',
                            'Linearity Error': primary_track.linearity_analysis.final_linearity_error_shifted,
                            'Linearity Pass': primary_track.linearity_analysis.linearity_pass,
                            'Linearity Validation Status': primary_track.linearity_analysis.validation_status.value if hasattr(primary_track.linearity_analysis, 'validation_status') else 'Not Validated',
                            'Linearity Industry Grade': primary_track.linearity_analysis.industry_grade if hasattr(primary_track.linearity_analysis, 'industry_grade') else 'Not Available',
                            'Resistance Change %': primary_track.unit_properties.resistance_change_percent,
                            'Resistance Validation Status': primary_track.resistance_analysis.validation_status.value if hasattr(primary_track.resistance_analysis, 'validation_status') else 'Not Validated',
                            'Resistance Stability Grade': primary_track.resistance_analysis.resistance_stability_grade if hasattr(primary_track.resistance_analysis, 'resistance_stability_grade') else 'Not Available',
                            'Risk Category': primary_track.failure_prediction.risk_category.value if primary_track.failure_prediction else 'Unknown',
                            'Overall Validation Status': primary_track.overall_validation_status.value if hasattr(primary_track, 'overall_validation_status') else 'Not Validated',
                            'Validation Warnings Count': len(primary_track.validation_warnings) if hasattr(primary_track, 'validation_warnings') else 0,
                            'Validation Recommendations Count': len(primary_track.validation_recommendations) if hasattr(primary_track, 'validation_recommendations') else 0
                        }
                        track_data.append(track_info)
                
                if track_data:
                    tracks_df = pd.DataFrame(track_data)
                    tracks_df.to_excel(writer, sheet_name='Track Details', index=False)
                
                # Validation summary sheet
                validation_summary_data = []
                for track_id, result in self.current_unit_data['tracks'].items():
                    primary_track = result.primary_track
                    if primary_track and hasattr(primary_track, 'validation_summary'):
                        validation_info = primary_track.validation_summary
                        validation_info['Track ID'] = track_id
                        validation_summary_data.append(validation_info)
                
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
        if not self.current_unit_data or not self.current_unit_data.get('tracks'):
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
                table_data = [['Track ID', 'Status', 'Sigma Gradient', 'Linearity Error (%)']]
                
                if 'tracks' in self.current_unit_data:
                    for track_id, result in self.current_unit_data['tracks'].items():
                        if hasattr(result, 'primary_track') and result.primary_track:
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
                if 'tracks' in self.current_unit_data and len(self.current_unit_data['tracks']) > 0:
                    fig2 = plt.figure(figsize=(8.5, 11))
                    fig2.suptitle('Track Comparison Charts', fontsize=16, fontweight='bold')
                    
                    # Prepare data for charts
                    track_ids = []
                    sigma_values = []
                    linearity_values = []
                    
                    for track_id, result in self.current_unit_data['tracks'].items():
                        if hasattr(result, 'primary_track') and result.primary_track:
                            track_ids.append(track_id)
                            sigma_values.append(result.primary_track.sigma_analysis.sigma_gradient)
                            linearity_values.append(result.primary_track.linearity_analysis.final_linearity_error_shifted)
                    
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
                        ax2.set_ylabel('Linearity Error (%)')
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
                    for track_id, result in self.current_unit_data['tracks'].items():
                        if hasattr(result, 'primary_track') and result.primary_track:
                            primary_track = result.primary_track
                            if hasattr(primary_track, 'position_data') and hasattr(primary_track, 'error_data'):
                                if primary_track.position_data is not None and primary_track.error_data is not None:
                                    track_list.append((track_id, primary_track))
                
                if track_list:
                    # Create pages with 4 tracks each
                    for page_start in range(0, len(track_list), tracks_per_page):
                        fig3 = plt.figure(figsize=(8.5, 11))
                        fig3.suptitle('Track Error Plots', fontsize=16, fontweight='bold')
                        
                        page_tracks = track_list[page_start:page_start + tracks_per_page]
                        
                        for i, (track_id, primary_track) in enumerate(page_tracks):
                            ax = fig3.add_subplot(2, 2, i + 1)
                            
                            # Plot error data
                            ax.plot(primary_track.position_data, primary_track.error_data, 
                                   'b-', linewidth=1.5, label='Error')
                            
                            # Add zero line
                            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                            
                            # Add limit lines if available
                            if hasattr(primary_track, 'limits') and primary_track.limits:
                                if hasattr(primary_track.limits, 'upper_limit'):
                                    ax.axhline(y=primary_track.limits.upper_limit, 
                                             color='r', linestyle='--', alpha=0.7, label='Upper Limit')
                                if hasattr(primary_track.limits, 'lower_limit'):
                                    ax.axhline(y=primary_track.limits.lower_limit, 
                                             color='r', linestyle='--', alpha=0.7, label='Lower Limit')
                            
                            ax.set_xlabel('Position')
                            ax.set_ylabel('Error (%)')
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
                        text=f"Status: {primary_track.overall_status.value}",
                        font=ctk.CTkFont(size=12)
                    ).pack(anchor='w', padx=10, pady=2)
                    
                    ctk.CTkLabel(
                        left_frame,
                        text=f"Sigma: {primary_track.sigma_analysis.sigma_gradient:.6f}",
                        font=ctk.CTkFont(size=12)
                    ).pack(anchor='w', padx=10, pady=2)
                    
                    ctk.CTkLabel(
                        left_frame,
                        text=f"Linearity: {primary_track.linearity_analysis.final_linearity_error_shifted:.4f}%",
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
            value=f"{track_data.linearity_analysis.final_linearity_error_shifted:.4f}%",
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
• Error (Shifted): {track_data.linearity_analysis.final_linearity_error_shifted:.4f}%
• Specification: {track_data.linearity_analysis.linearity_spec:.4f}%
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
• Final Linearity Error (Shifted): {linearity.final_linearity_error_shifted:.4f}%
• Linearity Specification: {linearity.linearity_spec:.4f}%
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
            
            # Plot error data
            if track_data.position_data and track_data.error_data:
                error_chart.plot_line(
                    x_data=track_data.position_data,
                    y_data=track_data.error_data,
                    label="Error Profile",
                    color='primary',
                    xlabel="Position (mm)",
                    ylabel="Error (%)"
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
                    track_info = {
                        'track_id': track.track_id,
                        'status': track.status.value,
                        'sigma_gradient': track.sigma_gradient,
                        'sigma_pass': track.sigma_pass,
                        'linearity_error': track.final_linearity_error_shifted,
                        'linearity_pass': track.linearity_pass,
                        'failure_probability': track.failure_probability,
                        'risk_category': track.risk_category.value if track.risk_category else 'UNKNOWN'
                    }
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
            
            for file_info in unit_summary['files']:
                for track_info in file_info['tracks'].values():
                    if track_info['sigma_gradient'] is not None:
                        all_sigma_gradients.append(track_info['sigma_gradient'])
                    if track_info['linearity_error'] is not None:
                        all_linearity_errors.append(track_info['linearity_error'])

            if all_sigma_gradients:
                sigma_cv = (np.std(all_sigma_gradients) / np.mean(all_sigma_gradients)) * 100
                unit_summary['sigma_cv'] = sigma_cv
            else:
                unit_summary['sigma_cv'] = 0

            if all_linearity_errors:
                linearity_cv = (np.std(all_linearity_errors) / np.mean(all_linearity_errors)) * 100
                unit_summary['linearity_cv'] = linearity_cv
            else:
                unit_summary['linearity_cv'] = 0

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
            
            # Update UI in main thread
            self.after(0, self._update_multi_track_display)

            self.logger.info(f"Loaded multi-track data from database: {model}/{serial} - {len(historical_data)} files")

        except Exception as e:
            self.logger.error(f"Failed to load multi-track data from database: {e}")
            self.after(0, lambda: messagebox.showerror(
                "Error", f"Failed to load data from database:\n{str(e)}"
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
                        # Format track data for viewer
                        formatted_track = {
                            'track_id': track_id,
                            'position': track_data.get('position', track_id),
                            'serial': self.current_unit_data.get('serial', 'Unknown'),
                            'timestamp': track_data.get('timestamp'),
                            'overall_status': track_data.get('status', 'Unknown'),
                            'validation_status': track_data.get('validation_status', 'Unknown'),
                            'sigma_gradient': track_data.get('sigma_gradient'),
                            'sigma_spec': track_data.get('sigma_spec'),
                            'sigma_margin': track_data.get('sigma_margin'),
                            'linearity_error': track_data.get('linearity_error'),
                            'linearity_spec': track_data.get('linearity_spec'),
                            'resistance_change': track_data.get('resistance_change'),
                            'trim_stability': track_data.get('trim_stability'),
                            'industry_grade': track_data.get('industry_grade', 'N/A'),
                            'error_profile': track_data.get('error_profile', {}),
                            'statistics': track_data.get('statistics', {}),
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
                        'error_profile': track_data.get('error_profile', {}),
                        'statistics': track_data.get('statistics', {})
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