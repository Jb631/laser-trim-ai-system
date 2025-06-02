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
        """Update all UI elements with multi-track data."""
        if not self.current_unit_data:
            return

        unit_data = self.current_unit_data
        comparison = self.comparison_data

        # Update unit info
        unit_id = unit_data['unit_id']
        track_count = len(unit_data['tracks'])
        overall_status = unit_data['overall_status'].value

        self.unit_info_label.config(
            text=f"Unit: {unit_id} | {track_count} tracks | Status: {overall_status}"
        )

        # Update overview cards
        self.overview_cards['unit_id'].update_value(unit_id)
        self.overview_cards['track_count'].update_value(track_count)
        self.overview_cards['overall_status'].update_value(overall_status)

        # Set status color
        if overall_status == "Pass":
            self.overview_cards['overall_status'].set_color_scheme('success')
        elif overall_status == "Fail":
            self.overview_cards['overall_status'].set_color_scheme('danger')
        else:
            self.overview_cards['overall_status'].set_color_scheme('warning')

        if comparison and comparison.get('comparison_performed'):
            # Update consistency metrics
            consistency_status = "Good" if not comparison.get('has_issues') else "Issues Found"
            self.overview_cards['consistency'].update_value(consistency_status)
            
            if comparison.get('has_issues'):
                self.overview_cards['consistency'].set_color_scheme('danger')
            else:
                self.overview_cards['consistency'].set_color_scheme('success')

            # Update variation metrics
            sigma_cv = comparison.get('sigma_comparison', {}).get('cv_percent', 0)
            linearity_cv = comparison.get('linearity_comparison', {}).get('cv_percent', 0)
            issues_count = len(comparison.get('consistency_issues', []))

            self.overview_cards['sigma_cv'].update_value(f"{sigma_cv:.1f}")
            self.overview_cards['linearity_cv'].update_value(f"{linearity_cv:.1f}")
            self.overview_cards['issues_found'].update_value(issues_count)

            # Calculate overall validation grade for the unit
            validation_grades = []
            for result in unit_data['tracks'].values():
                if hasattr(result, 'validation_grade') and result.validation_grade:
                    grade = result.validation_grade
                    if grade not in ["Not Validated", "Incomplete"]:
                        validation_grades.append(grade)

            if validation_grades:
                grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0, "F": 0}
                avg_grade_value = sum(grade_values.get(g, 0) for g in validation_grades) / len(validation_grades)
                
                if avg_grade_value >= 3.5:
                    overall_validation_grade = "A"
                elif avg_grade_value >= 2.5:
                    overall_validation_grade = "B"
                elif avg_grade_value >= 1.5:
                    overall_validation_grade = "C"
                elif avg_grade_value >= 0.5:
                    overall_validation_grade = "D"
                else:
                    overall_validation_grade = "F"
            else:
                overall_validation_grade = "Not Validated"
            
            self.overview_cards['validation_grade'].update_value(overall_validation_grade)
            
            # Set validation grade color scheme
            if overall_validation_grade in ["A", "B"]:
                self.overview_cards['validation_grade'].set_color_scheme('success')
            elif overall_validation_grade in ["C", "D"]:
                self.overview_cards['validation_grade'].set_color_scheme('warning')
            elif overall_validation_grade == "F":
                self.overview_cards['validation_grade'].set_color_scheme('danger')
            else:
                self.overview_cards['validation_grade'].set_color_scheme('default')

            # Set color schemes based on values
            if sigma_cv > 10:
                self.overview_cards['sigma_cv'].set_color_scheme('danger')
            elif sigma_cv > 5:
                self.overview_cards['sigma_cv'].set_color_scheme('warning')
            else:
                self.overview_cards['sigma_cv'].set_color_scheme('success')

            # Update charts
            self._update_comparison_charts()
            
            # Update consistency analysis
            self._update_consistency_analysis()
        else:
            # Single track or no comparison
            self.overview_cards['consistency'].update_value("N/A (Single Track)")
            self.overview_cards['sigma_cv'].update_value("N/A")
            self.overview_cards['linearity_cv'].update_value("N/A")
            self.overview_cards['validation_grade'].update_value("N/A")
            self.overview_cards['issues_found'].update_value("0")

        # Enable action buttons
        self.export_report_btn.config(state='normal')
        self.generate_pdf_btn.config(state='normal')
        self.view_tracks_btn.config(state='normal')
        
        self.status_label.config(text=f"Analysis complete: {datetime.now().strftime('%H:%M:%S')}")

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
            return

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
        
        recommendations = []
        
        if comparison.get('has_issues'):
            recommendations.append("• Investigate manufacturing process variations between tracks")
            recommendations.append("• Check tooling alignment and calibration")
            recommendations.append("• Review trim parameters for consistency")
            
            if comparison.get('sigma_comparison', {}).get('cv_percent', 0) > 10:
                recommendations.append("• High sigma variation detected - review cutting parameters")
                
            if comparison.get('linearity_comparison', {}).get('cv_percent', 0) > 15:
                recommendations.append("• High linearity variation - check mechanical alignment")
        else:
            recommendations.append("• Unit shows good track-to-track consistency")
            recommendations.append("• Continue current manufacturing process")
            recommendations.append("• Regular monitoring recommended")
        
        for rec in recommendations:
            self.recommendations_text.insert(tk.END, rec + "\n")
        
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

    def on_show(self):
        """Called when page is shown."""
        pass 