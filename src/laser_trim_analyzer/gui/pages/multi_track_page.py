"""
Multi-Track Analysis Page for Laser Trim Analyzer

Provides interface for analyzing and comparing multi-track units,
particularly for System B multi-track files with TA, TB identifiers.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
import threading
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.gui.widgets.stat_card import StatCard
from laser_trim_analyzer.gui.widgets import add_mousewheel_support


class MultiTrackPage(BasePage):
    """Multi-track analysis and comparison page."""

    def __init__(self, parent, main_window):
        self.current_unit_data = None
        self.comparison_data = None
        super().__init__(parent, main_window)

    def _create_page(self):
        """Set up the multi-track analysis page."""
        # Create scrollable frame
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add mouse wheel scrolling support
        add_mousewheel_support(scrollable_frame, canvas)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create content in scrollable frame
        content_frame = scrollable_frame
        
        # Title and file selection
        self._create_header_section(content_frame)
        
        # Unit overview metrics
        self._create_overview_section(content_frame)
        
        # Track comparison charts
        self._create_comparison_section(content_frame)
        
        # Consistency analysis
        self._create_consistency_section(content_frame)
        
        # Export controls
        self._create_actions_section(content_frame)

    def _create_header_section(self, parent):
        """Create header with title and file selection."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', padx=20, pady=(20, 10))

        # Configure grid for responsive layout
        header_frame.columnconfigure(0, weight=1)
        header_frame.columnconfigure(1, weight=0)

        # Title on the left
        title_label = ttk.Label(
            header_frame,
            text="Multi-Track Unit Analysis",
            font=('Segoe UI', 24, 'bold')
        )
        title_label.grid(row=0, column=0, sticky='w')

        # File selection on the right
        selection_frame = ttk.Frame(header_frame)
        selection_frame.grid(row=0, column=1, sticky='e', padx=(10, 0))

        ttk.Button(
            selection_frame,
            text="📁 Select Track File",
            command=self._select_track_file,
            style='Primary.TButton'
        ).pack(side='left', padx=(0, 10))

        ttk.Button(
            selection_frame,
            text="📂 Analyze Folder",
            command=self._analyze_folder,
        ).pack(side='left', padx=(0, 10))

        ttk.Button(
            selection_frame,
            text="🗄️ From Database",
            command=self._select_unit_from_database,
        ).pack(side='left')

        # Selected unit info (full width below header)
        self.unit_info_label = ttk.Label(
            parent,
            text="Select a track file to begin multi-track analysis",
            font=('Segoe UI', 11),
            foreground=self.colors['text_secondary']
        )
        self.unit_info_label.pack(fill='x', padx=20, pady=(0, 10))

    def _create_overview_section(self, parent):
        """Create unit overview metrics."""
        overview_frame = ttk.LabelFrame(
            parent,
            text="Unit Overview",
            padding=15
        )
        overview_frame.pack(fill='x', padx=20, pady=10)

        # Create 2x4 grid of metric cards
        self.overview_grid = ttk.Frame(overview_frame)
        self.overview_grid.pack(fill='x')

        # Configure grid
        for i in range(4):
            self.overview_grid.columnconfigure(i, weight=1, minsize=180)

        # Initialize overview cards
        self.overview_cards = {}
        
        # Row 1: Basic info
        self.overview_cards['unit_id'] = StatCard(
            self.overview_grid,
            title="Unit ID",
            value="--",
            unit="",
            color_scheme="default"
        )
        self.overview_cards['unit_id'].grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.overview_cards['track_count'] = StatCard(
            self.overview_grid,
            title="Track Count",
            value="--",
            unit="tracks",
            color_scheme="info"
        )
        self.overview_cards['track_count'].grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.overview_cards['overall_status'] = StatCard(
            self.overview_grid,
            title="Overall Status",
            value="--",
            unit="",
            color_scheme="default"
        )
        self.overview_cards['overall_status'].grid(row=0, column=2, padx=5, pady=5, sticky='ew')

        self.overview_cards['consistency'] = StatCard(
            self.overview_grid,
            title="Track Consistency",
            value="--",
            unit="",
            color_scheme="default"
        )
        self.overview_cards['consistency'].grid(row=0, column=3, padx=5, pady=5, sticky='ew')

        # Row 2: Performance metrics
        self.overview_cards['sigma_cv'] = StatCard(
            self.overview_grid,
            title="Sigma Variation (CV)",
            value="--",
            unit="%",
            color_scheme="warning"
        )
        self.overview_cards['sigma_cv'].grid(row=1, column=0, padx=5, pady=5, sticky='ew')

        self.overview_cards['linearity_cv'] = StatCard(
            self.overview_grid,
            title="Linearity Variation (CV)",
            value="--",
            unit="%",
            color_scheme="warning"
        )
        self.overview_cards['linearity_cv'].grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        self.overview_cards['validation_grade'] = StatCard(
            self.overview_grid,
            title="Validation Grade",
            value="--",
            unit="",
            color_scheme="info"
        )
        self.overview_cards['validation_grade'].grid(row=1, column=2, padx=5, pady=5, sticky='ew')

        self.overview_cards['issues_found'] = StatCard(
            self.overview_grid,
            title="Issues Found",
            value="--",
            unit="",
            color_scheme="danger"
        )
        self.overview_cards['issues_found'].grid(row=1, column=3, padx=5, pady=5, sticky='ew')

    def _create_comparison_section(self, parent):
        """Create track comparison charts section."""
        comparison_frame = ttk.LabelFrame(
            parent,
            text="Track Comparison",
            padding=15
        )
        comparison_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Create notebook for different comparison views
        self.comparison_notebook = ttk.Notebook(comparison_frame)
        self.comparison_notebook.pack(fill='both', expand=True)

        # Sigma comparison tab
        sigma_frame = ttk.Frame(self.comparison_notebook)
        self.comparison_notebook.add(sigma_frame, text="Sigma Comparison")

        self.sigma_comparison_chart = ChartWidget(
            sigma_frame,
            chart_type='bar',
            title="Sigma Gradient by Track",
            figsize=(10, 6)
        )
        self.sigma_comparison_chart.pack(fill='both', expand=True, padx=10, pady=10)

        # Linearity comparison tab
        linearity_frame = ttk.Frame(self.comparison_notebook)
        self.comparison_notebook.add(linearity_frame, text="Linearity Comparison")

        self.linearity_comparison_chart = ChartWidget(
            linearity_frame,
            chart_type='bar',
            title="Linearity Error by Track",
            figsize=(10, 6)
        )
        self.linearity_comparison_chart.pack(fill='both', expand=True, padx=10, pady=10)

        # Error profile comparison tab
        profile_frame = ttk.Frame(self.comparison_notebook)
        self.comparison_notebook.add(profile_frame, text="Error Profiles")

        self.profile_comparison_chart = ChartWidget(
            profile_frame,
            chart_type='line',
            title="Error Profile Comparison",
            figsize=(10, 6)
        )
        self.profile_comparison_chart.pack(fill='both', expand=True, padx=10, pady=10)

    def _create_consistency_section(self, parent):
        """Create consistency analysis section."""
        consistency_frame = ttk.LabelFrame(
            parent,
            text="Consistency Analysis",
            padding=15
        )
        consistency_frame.pack(fill='x', padx=20, pady=10)

        # Issues list
        issues_label = ttk.Label(
            consistency_frame,
            text="Consistency Issues:",
            font=('Segoe UI', 12, 'bold')
        )
        issues_label.pack(anchor='w', pady=(0, 10))

        # Create frame for issues list with scrollbar
        issues_container = ttk.Frame(consistency_frame)
        issues_container.pack(fill='x', pady=(0, 10))

        self.issues_text = tk.Text(
            issues_container,
            height=6,
            wrap='word',
            state='disabled',
            font=('Segoe UI', 10)
        )
        
        issues_scrollbar = ttk.Scrollbar(
            issues_container,
            orient='vertical',
            command=self.issues_text.yview
        )
        self.issues_text.config(yscrollcommand=issues_scrollbar.set)

        self.issues_text.pack(side='left', fill='both', expand=True)
        issues_scrollbar.pack(side='right', fill='y')

        # Recommendations
        recommendations_label = ttk.Label(
            consistency_frame,
            text="Recommendations:",
            font=('Segoe UI', 12, 'bold')
        )
        recommendations_label.pack(anchor='w', pady=(10, 5))

        self.recommendations_text = tk.Text(
            consistency_frame,
            height=4,
            wrap='word',
            state='disabled',
            font=('Segoe UI', 10)
        )
        self.recommendations_text.pack(fill='x')

    def _create_actions_section(self, parent):
        """Create export and action buttons."""
        actions_frame = ttk.LabelFrame(
            parent,
            text="Actions",
            padding=15
        )
        actions_frame.pack(fill='x', padx=20, pady=(10, 20))

        # Button container
        btn_container = ttk.Frame(actions_frame)
        btn_container.pack(fill='x')

        # Export comparison report button
        self.export_report_btn = ttk.Button(
            btn_container,
            text="📊 Export Comparison Report",
            command=self._export_comparison_report,
            style='Primary.TButton',
            state='disabled'
        )
        self.export_report_btn.pack(side='left', padx=(0, 10))

        # Generate PDF report button
        self.generate_pdf_btn = ttk.Button(
            btn_container,
            text="📄 Generate PDF Report",
            command=self._generate_pdf_report,
            state='disabled'
        )
        self.generate_pdf_btn.pack(side='left', padx=(0, 10))

        # View individual tracks button
        self.view_tracks_btn = ttk.Button(
            btn_container,
            text="👁 View Individual Tracks",
            command=self._view_individual_tracks,
            state='disabled'
        )
        self.view_tracks_btn.pack(side='left')

        # Status label
        self.status_label = ttk.Label(
            btn_container,
            text="",
            font=('Segoe UI', 10),
            foreground=self.colors['text_secondary']
        )
        self.status_label.pack(side='right')

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
        """Analyze a track file and find related tracks."""
        self.unit_info_label.config(text=f"Analyzing: {file_path.name}...")
        self.status_label.config(text="Processing...")
        
        # Run analysis in background thread
        thread = threading.Thread(
            target=self._run_multi_track_analysis,
            args=(file_path,),
            daemon=True
        )
        thread.start()

    def _analyze_folder_tracks(self, folder_path: Path):
        """Analyze all track files in a folder and group by units."""
        self.unit_info_label.config(text=f"Analyzing folder: {folder_path.name}...")
        self.status_label.config(text="Scanning folder...")
        
        # Run analysis in background thread
        thread = threading.Thread(
            target=self._run_folder_analysis,
            args=(folder_path,),
            daemon=True
        )
        thread.start()

    def _run_multi_track_analysis(self, file_path: Path):
        """Run multi-track analysis in background thread."""
        try:
            if not self.main_window.processor:
                raise ValueError("Processor not available")

            # Use the new multi-track analysis method
            import asyncio
            
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the analysis
            unit_data = loop.run_until_complete(
                self.main_window.processor.analyze_multi_track_unit(file_path)
            )
            
            if unit_data:
                self.current_unit_data = unit_data
                self.comparison_data = unit_data.get('comparison')
                
                # Update UI in main thread
                self.after(0, self._update_multi_track_display)
            else:
                self.after(0, lambda: messagebox.showerror(
                    "Error", "Failed to analyze track file"
                ))

        except Exception as e:
            self.logger.error(f"Multi-track analysis failed: {e}")
            self.after(0, lambda: messagebox.showerror(
                "Error", f"Analysis failed:\n{str(e)}"
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
        listbox.config(yscrollcommand=scrollbar.set)

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
            self.unit_info_label.config(text="No multi-track data loaded. Select a file, folder, or unit from database to begin analysis.")
            
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
                
                self.unit_info_label.config(
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
                
                self.unit_info_label.config(
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
                self.export_report_btn.config(state='normal')
            if hasattr(self, 'generate_pdf_btn'):
                self.generate_pdf_btn.config(state='normal')
            
            self.logger.info("Successfully updated multi-track display")
            
        except Exception as e:
            self.logger.error(f"Error updating multi-track display: {e}")
            self.unit_info_label.config(text="Error displaying multi-track data - check logs")

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
        """Update consistency analysis text fields."""
        if not self.comparison_data:
            # Clear text fields and show no data message
            self.issues_text.config(state='normal')
            self.issues_text.delete(1.0, tk.END)
            self.issues_text.insert(tk.END, "No comparison data available.")
            self.issues_text.config(state='disabled')
            
            self.recommendations_text.config(state='normal')
            self.recommendations_text.delete(1.0, tk.END)
            self.recommendations_text.insert(tk.END, "Load multi-track data to see recommendations.")
            self.recommendations_text.config(state='disabled')
            return

        try:
            comparison = self.comparison_data
            
            # Update issues text
            self.issues_text.config(state='normal')
            self.issues_text.delete(1.0, tk.END)
            
            if comparison.get('consistency_issues'):
                for i, issue in enumerate(comparison['consistency_issues'], 1):
                    self.issues_text.insert(tk.END, f"{i}. {issue}\n")
            else:
                self.issues_text.insert(tk.END, "No consistency issues found. All tracks show good agreement.")
            
            self.issues_text.config(state='disabled')
            
            # Update recommendations text
            self.recommendations_text.config(state='normal')
            self.recommendations_text.delete(1.0, tk.END)
            
            if comparison.get('recommendations'):
                for i, rec in enumerate(comparison['recommendations'], 1):
                    self.recommendations_text.insert(tk.END, f"{i}. {rec}\n")
            else:
                # Generate default recommendations based on analysis
                recommendations = []
                if comparison.get('has_issues'):
                    recommendations.append("Review tracks with high variation for potential manufacturing issues.")
                    recommendations.append("Consider additional quality control measures for this unit type.")
                else:
                    recommendations.append("Unit shows good track consistency.")
                    recommendations.append("Continue with current manufacturing process.")
                
                for i, rec in enumerate(recommendations, 1):
                    self.recommendations_text.insert(tk.END, f"{i}. {rec}\n")
            
            self.recommendations_text.config(state='disabled')
            
        except Exception as e:
            self.logger.error(f"Error updating consistency analysis: {e}")
            
            # Show error state
            self.issues_text.config(state='normal')
            self.issues_text.delete(1.0, tk.END)
            self.issues_text.insert(tk.END, f"Error loading consistency data: {str(e)}")
            self.issues_text.config(state='disabled')
            
            self.recommendations_text.config(state='normal')
            self.recommendations_text.delete(1.0, tk.END)
            self.recommendations_text.insert(tk.END, "Unable to generate recommendations due to data error.")
            self.recommendations_text.config(state='disabled')

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
        messagebox.showinfo("Coming Soon", "PDF report generation will be implemented in the next version.")

    def _view_individual_tracks(self):
        """Open individual track analysis in separate windows."""
        messagebox.showinfo("Coming Soon", "Individual track viewer will be implemented in the next version.")

    def _load_multi_track_from_database(self, model: str, serial: str):
        """Load multi-track data for a specific unit from database."""
        try:
            if not self.db_manager:
                raise ValueError("Database not connected")

            # Get all analyses for this model/serial combination
            historical_data = self.db_manager.get_historical_data(
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
            # Get all unique model/serial combinations
            with self.db_manager.get_session() as session:
                from laser_trim_analyzer.database.manager import DBAnalysisResult
                
                results = session.query(
                    DBAnalysisResult.model,
                    DBAnalysisResult.serial,
                    func.count(DBAnalysisResult.id).label('file_count')
                ).filter(
                    DBAnalysisResult.model.isnot(None),
                    DBAnalysisResult.serial.isnot(None)
                ).group_by(
                    DBAnalysisResult.model,
                    DBAnalysisResult.serial
                ).having(
                    func.count(DBAnalysisResult.id) > 1  # Only units with multiple files
                ).order_by(
                    DBAnalysisResult.model,
                    DBAnalysisResult.serial
                ).all()

            if not results:
                messagebox.showinfo(
                    "No Multi-Track Units",
                    "No units with multiple files found in database."
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
                text="Units with Multiple Files:",
                font=('Segoe UI', 14, 'bold')
            ).pack(pady=(10, 20))

            # Create listbox with scrollbar
            list_frame = ttk.Frame(dialog)
            list_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))

            listbox = tk.Listbox(list_frame, font=('Segoe UI', 10))
            scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=listbox.yview)
            listbox.config(yscrollcommand=scrollbar.set)

            # Populate listbox
            unit_list = []
            for model, serial, file_count in results:
                display_text = f"{model} / {serial} ({file_count} files)"
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
                    self.unit_info_label.config(text=f"Loading data for {model}/{serial}...")
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