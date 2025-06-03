"""
Historical Data Page for Laser Trim Analyzer

Provides interface for querying and analyzing historical QA data
with charts and export functionality.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.core.models import AnalysisResult, FileMetadata, AnalysisStatus
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.gui.widgets import add_mousewheel_support
from laser_trim_analyzer.utils.date_utils import safe_datetime_convert


class HistoricalPage(BasePage):
    """Historical data analysis page."""

    def __init__(self, parent, main_window):
        self.current_data = None
        super().__init__(parent, main_window)

    def _create_page(self):
        """Set up the historical data page with scrollable layout."""
        # Create scrollable main frame without shifting
        main_container = ttk.Frame(self)
        main_container.pack(fill='both', expand=True)
        
        # Canvas and scrollbar
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add mouse wheel scrolling support
        add_mousewheel_support(scrollable_frame, canvas)
        
        # Pack scrollbar first to avoid shifting
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Title in scrollable frame
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill='x', padx=20, pady=(20, 10))

        ttk.Label(
            title_frame,
            text="Historical Data Analysis",
            font=('Segoe UI', 24, 'bold')
        ).pack(side='left')

        # Create main sections in scrollable frame
        self._create_query_section(scrollable_frame)
        self._create_results_section(scrollable_frame)
        self._create_charts_section(scrollable_frame)

    def _create_query_section(self, parent):
        """Create query filters section with responsive layout."""
        query_frame = ttk.LabelFrame(
            parent,
            text="Query Filters",
            padding=15
        )
        query_frame.pack(fill='x', padx=20, pady=10)

        # Create responsive grid for filters
        filters_grid = ttk.Frame(query_frame)
        filters_grid.pack(fill='x')
        
        # Configure grid for responsive layout
        for i in range(6):  # 6 columns
            filters_grid.grid_columnconfigure(i, weight=1, minsize=120)

        # Model filter
        ttk.Label(filters_grid, text="Model:").grid(
            row=0, column=0, sticky='w', padx=(0, 10), pady=5
        )
        self.model_var = tk.StringVar()
        self.model_entry = ttk.Entry(
            filters_grid,
            textvariable=self.model_var
        )
        self.model_entry.grid(row=0, column=1, sticky='ew', padx=(0, 20), pady=5)

        # Serial filter
        ttk.Label(filters_grid, text="Serial:").grid(
            row=0, column=2, sticky='w', padx=(0, 10), pady=5
        )
        self.serial_var = tk.StringVar()
        self.serial_entry = ttk.Entry(
            filters_grid,
            textvariable=self.serial_var
        )
        self.serial_entry.grid(row=0, column=3, sticky='ew', padx=(0, 20), pady=5)

        # Date range
        ttk.Label(filters_grid, text="Date Range:").grid(
            row=0, column=4, sticky='w', padx=(0, 10), pady=5
        )
        self.date_range_var = tk.StringVar(value="Last 30 days")
        self.date_combo = ttk.Combobox(
            filters_grid,
            textvariable=self.date_range_var,
            values=[
                "Today", "Last 7 days", "Last 30 days",
                "Last 90 days", "Last year", "All time"
            ],
            state='readonly'
        )
        self.date_combo.grid(row=0, column=5, sticky='ew', pady=5)

        # Second row of filters
        # Status filter
        ttk.Label(filters_grid, text="Status:").grid(
            row=1, column=0, sticky='w', padx=(0, 10), pady=5
        )
        self.status_var = tk.StringVar(value="All")
        self.status_combo = ttk.Combobox(
            filters_grid,
            textvariable=self.status_var,
            values=["All", "Pass", "Fail", "Warning"],
            state='readonly'
        )
        self.status_combo.grid(row=1, column=1, sticky='ew', pady=5)

        # Risk category filter
        ttk.Label(filters_grid, text="Risk:").grid(
            row=1, column=2, sticky='w', padx=(0, 10), pady=5
        )
        self.risk_var = tk.StringVar(value="All")
        self.risk_combo = ttk.Combobox(
            filters_grid,
            textvariable=self.risk_var,
            values=["All", "High", "Medium", "Low"],
            state='readonly'
        )
        self.risk_combo.grid(row=1, column=3, sticky='ew', pady=5)

        # Limit results
        ttk.Label(filters_grid, text="Limit:").grid(
            row=1, column=4, sticky='w', padx=(0, 10), pady=5
        )
        self.limit_var = tk.StringVar(value="100")
        self.limit_combo = ttk.Combobox(
            filters_grid,
            textvariable=self.limit_var,
            values=["50", "100", "500", "1000", "All"],
            state='readonly'
        )
        self.limit_combo.grid(row=1, column=5, sticky='ew', pady=5)

        # Action buttons with responsive layout
        button_frame = ttk.Frame(query_frame)
        button_frame.pack(fill='x', pady=(15, 0))
        
        # Configure button frame for responsive layout
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_columnconfigure(2, weight=1)
        button_frame.grid_columnconfigure(3, weight=2)  # Extra space

        self.query_btn = ttk.Button(
            button_frame,
            text="Run Query",
            command=self._run_query,
            style='Primary.TButton'
        )
        self.query_btn.grid(row=0, column=0, sticky='ew', padx=(0, 10))

        clear_btn = ttk.Button(
            button_frame,
            text="Clear Filters",
            command=self._clear_filters
        )
        clear_btn.grid(row=0, column=1, sticky='ew', padx=(0, 10))

        export_btn = ttk.Button(
            button_frame,
            text="Export Results",
            command=self._export_results
        )
        export_btn.grid(row=0, column=2, sticky='ew', padx=(0, 10))

    def _create_results_section(self, parent):
        """Create results table section with responsive height management."""
        results_frame = ttk.LabelFrame(
            parent,
            text="Query Results",
            padding=10
        )
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)  # Changed to expand

        # Create treeview with scrollbars and responsive layout
        tree_frame = ttk.Frame(results_frame)
        tree_frame.pack(fill='both', expand=True)  # Changed to expand
        
        # Configure tree frame for responsiveness
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient='vertical')
        hsb = ttk.Scrollbar(tree_frame, orient='horizontal')

        # Treeview with responsive dimensions
        columns = (
            'Date', 'Model', 'Serial', 'System', 'Status',
            'Sigma Gradient', 'Sigma Pass', 'Linearity Pass',
            'Risk Category', 'Processing Time'
        )

        self.results_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show='tree headings',
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set
            # Remove fixed height for responsive behavior
        )

        # Configure scrollbars
        vsb.config(command=self.results_tree.yview)
        hsb.config(command=self.results_tree.xview)

        # Configure columns with responsive widths
        self.results_tree.column('#0', width=0, stretch=False)
        
        # Define responsive column widths
        column_configs = [
            ('Date', 150, True),
            ('Model', 80, True),
            ('Serial', 100, True),
            ('System', 60, True),
            ('Status', 80, True),
            ('Sigma Gradient', 100, True),
            ('Sigma Pass', 80, True),
            ('Linearity Pass', 100, True),
            ('Risk Category', 90, True),
            ('Processing Time', 100, True)
        ]
        
        for col, width, stretch in column_configs:
            self.results_tree.column(col, width=width, stretch=stretch, minwidth=50)

        # Configure headings
        for col in columns:
            self.results_tree.heading(col, text=col, command=lambda c=col: self._sort_column(c))

        # Grid layout for responsive control
        self.results_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        # Add mouse wheel scrolling support to treeview
        add_mousewheel_support(self.results_tree)

        # Summary label with responsive positioning
        self.summary_label = ttk.Label(
            results_frame,
            text="No data loaded",
            font=('Segoe UI', 10)
        )
        self.summary_label.pack(pady=(10, 0), expand=True)

        # Bind double-click to view details
        self.results_tree.bind('<Double-1>', self._view_details)

    def _create_charts_section(self, parent):
        """Create charts section with responsive layout."""
        charts_frame = ttk.LabelFrame(
            parent,
            text="Data Visualization",
            padding=10
        )
        charts_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Create notebook for different charts with responsive sizing
        self.chart_notebook = ttk.Notebook(charts_frame)
        self.chart_notebook.pack(fill='both', expand=True)

        # Pass rate trend chart with responsive dimensions
        trend_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(trend_frame, text="Pass Rate Trend")

        self.trend_chart = ChartWidget(
            trend_frame,
            chart_type='line',
            title="Pass Rate Trend Over Time"
            # Remove fixed figsize for responsive behavior
        )
        self.trend_chart.pack(fill='both', expand=True, padx=10, pady=10)

        # Sigma distribution chart with responsive dimensions
        dist_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(dist_frame, text="Sigma Distribution")

        self.dist_chart = ChartWidget(
            dist_frame,
            chart_type='histogram',
            title="Sigma Gradient Distribution"
            # Remove fixed figsize for responsive behavior
        )
        self.dist_chart.pack(fill='both', expand=True, padx=10, pady=10)

        # Model comparison chart with responsive dimensions
        comp_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(comp_frame, text="Model Comparison")

        self.comp_chart = ChartWidget(
            comp_frame,
            chart_type='bar',
            title="Pass Rate by Model"
            # Remove fixed figsize for responsive behavior
        )
        self.comp_chart.pack(fill='both', expand=True, padx=10, pady=10)

    def _run_query(self):
        """Execute database query with current filters."""
        if not self.db_manager:
            messagebox.showerror("Error", "Database not connected")
            return

        # Disable query button
        self.query_btn.config(state='disabled', text='Querying...')
        self.update()

        try:
            # Get filter values
            model = self.model_var.get() or None
            serial = self.serial_var.get() or None
            status = None if self.status_var.get() == "All" else self.status_var.get()
            risk = None if self.risk_var.get() == "All" else self.risk_var.get()

            # Calculate date range
            days_back = self._get_days_back()

            # Get limit
            limit_str = self.limit_var.get()
            limit = None if limit_str == "All" else int(limit_str)

            # Query database
            results = self.db_manager.get_historical_data(
                model=model,
                serial=serial,
                days_back=days_back,
                status=status,
                risk_category=risk,
                limit=limit,
                include_tracks=True
            )

            # Update UI with results
            self._display_results(results)

            # Update charts
            self._update_charts(results)

        except Exception as e:
            messagebox.showerror("Query Error", f"Failed to query database:\n{str(e)}")

        finally:
            self.query_btn.config(state='normal', text='Run Query')

    def _display_results(self, results):
        """Display query results in treeview."""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        if not results:
            self.summary_label.config(text="No results found")
            return

        # Convert to DataFrame for easier handling
        data = []
        for result in results:
            # Get primary track data
            primary_track = None
            if result.tracks:
                primary_track = result.tracks[0]

            row = {
                'id': result.id,
                'timestamp': safe_datetime_convert(result.timestamp),
                'file_date': safe_datetime_convert(result.file_date),
                'model': result.model,
                'serial': result.serial,
                'system': result.system.value,
                'status': result.overall_status.value,
                'sigma_gradient': primary_track.sigma_gradient if primary_track else None,
                'sigma_pass': primary_track.sigma_pass if primary_track else None,
                'linearity_pass': primary_track.linearity_pass if primary_track else None,
                'risk_category': primary_track.risk_category.value if primary_track and primary_track.risk_category else None,
                'processing_time': result.processing_time
            }
            data.append(row)

        self.current_data = pd.DataFrame(data)

        # Insert into treeview
        for _, row in self.current_data.iterrows():
            # Use file_date if available, otherwise fall back to timestamp
            date_to_display = row['file_date'] if pd.notna(row['file_date']) else row['timestamp']
            
            values = (
                date_to_display.strftime('%Y-%m-%d %H:%M') if pd.notna(date_to_display) else '',
                row['model'],
                row['serial'],
                row['system'],
                row['status'],
                f"{row['sigma_gradient']:.6f}" if pd.notna(row['sigma_gradient']) else '',
                '✓' if row['sigma_pass'] else '✗' if pd.notna(row['sigma_pass']) else '',
                '✓' if row['linearity_pass'] else '✗' if pd.notna(row['linearity_pass']) else '',
                row['risk_category'] if pd.notna(row['risk_category']) else '',
                f"{row['processing_time']:.2f}s" if pd.notna(row['processing_time']) else ''
            )

            # Color based on status
            tags = []
            if row['status'] == 'Pass':
                tags.append('pass')
            elif row['status'] == 'Fail':
                tags.append('fail')
            elif row['status'] == 'Warning':
                tags.append('warning')

            self.results_tree.insert('', 'end', values=values, tags=tags)

        # Configure tag colors
        self.results_tree.tag_configure('pass', foreground='#27ae60')
        self.results_tree.tag_configure('fail', foreground='#e74c3c')
        self.results_tree.tag_configure('warning', foreground='#f39c12')

        # Update summary
        total = len(self.current_data)
        passed = (self.current_data['status'] == 'Pass').sum()
        failed = (self.current_data['status'] == 'Fail').sum()

        self.summary_label.config(
            text=f"Total: {total} | Passed: {passed} | Failed: {failed} | Pass Rate: {passed / total * 100:.1f}%"
        )

    def _update_charts(self, results):
        """Update all charts with query results."""
        if not results or self.current_data is None:
            return

        # Update pass rate trend
        self._update_trend_chart()

        # Update sigma distribution
        self._update_distribution_chart()

        # Update model comparison
        self._update_comparison_chart()

    def _update_trend_chart(self):
        """Update pass rate trend chart."""
        if self.current_data is None or len(self.current_data) == 0:
            return

        # Group by date and calculate pass rate
        df = self.current_data.copy()
        # Use file_date if available, otherwise timestamp
        df['date'] = pd.to_datetime(df['file_date'].fillna(df['timestamp'])).dt.date

        daily_stats = df.groupby('date').agg({
            'status': lambda x: (x == 'Pass').mean() * 100
        }).reset_index()
        daily_stats.columns = ['date', 'pass_rate']

        # Sort by date
        daily_stats = daily_stats.sort_values('date')

        # Clear and plot
        self.trend_chart.clear_chart()

        if len(daily_stats) > 1:
            self.trend_chart.plot_line(
                x_data=daily_stats['date'].tolist(),
                y_data=daily_stats['pass_rate'].tolist(),
                label="Pass Rate",
                color='primary',
                marker='o',
                xlabel="Date",
                ylabel="Pass Rate (%)"
            )

            # Add average line
            avg_pass_rate = daily_stats['pass_rate'].mean()
            self.trend_chart.add_threshold_lines(
                {'Average': avg_pass_rate},
                orientation='horizontal'
            )

    def _update_distribution_chart(self):
        """Update sigma gradient distribution chart."""
        if self.current_data is None or 'sigma_gradient' not in self.current_data.columns:
            return

        # Get sigma values
        sigma_values = self.current_data['sigma_gradient'].dropna()

        if len(sigma_values) == 0:
            return

        # Clear and plot
        self.dist_chart.clear_chart()
        self.dist_chart.plot_histogram(
            data=sigma_values.tolist(),
            bins=30,
            color='primary',
            xlabel="Sigma Gradient",
            ylabel="Frequency"
        )

    def _update_comparison_chart(self):
        """Update model comparison chart."""
        if self.current_data is None or len(self.current_data) == 0:
            return

        # Calculate pass rate by model
        model_stats = self.current_data.groupby('model').agg({
            'status': [
                lambda x: (x == 'Pass').mean() * 100,
                'count'
            ]
        }).reset_index()

        model_stats.columns = ['model', 'pass_rate', 'count']

        # Filter models with sufficient data
        model_stats = model_stats[model_stats['count'] >= 5]

        if len(model_stats) == 0:
            return

        # Sort by pass rate
        model_stats = model_stats.sort_values('pass_rate', ascending=False)

        # Determine colors based on pass rate
        colors = []
        for rate in model_stats['pass_rate']:
            if rate >= 95:
                colors.append('pass')
            elif rate >= 90:
                colors.append('warning')
            else:
                colors.append('fail')

        # Clear and plot
        self.comp_chart.clear_chart()
        self.comp_chart.plot_bar(
            categories=model_stats['model'].tolist(),
            values=model_stats['pass_rate'].tolist(),
            colors=colors,
            xlabel="Model",
            ylabel="Pass Rate (%)"
        )

    def _get_days_back(self) -> Optional[int]:
        """Convert date range selection to days."""
        date_range = self.date_range_var.get()

        mapping = {
            "Today": 1,
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
            "Last year": 365,
            "All time": None
        }

        return mapping.get(date_range, 30)

    def _clear_filters(self):
        """Clear all filter inputs."""
        self.model_var.set("")
        self.serial_var.set("")
        self.date_range_var.set("Last 30 days")
        self.status_var.set("All")
        self.risk_var.set("All")
        self.limit_var.set("100")

    def _export_results(self):
        """Export current results to file."""
        if self.current_data is None or len(self.current_data) == 0:
            messagebox.showwarning("No Data", "No data to export")
            return

        # Ask for file location
        filename = filedialog.asksaveasfilename(
            defaultextension='.xlsx',
            filetypes=[
                ('Excel files', '*.xlsx'),
                ('CSV files', '*.csv'),
                ('All files', '*.*')
            ],
            initialfile=f'historical_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

        if not filename:
            return

        try:
            if filename.endswith('.xlsx'):
                # Export to Excel with formatting
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    self.current_data.to_excel(writer, sheet_name='Historical Data', index=False)

                    # Add summary sheet
                    summary = pd.DataFrame({
                        'Metric': ['Total Records', 'Pass Rate', 'Average Sigma', 'Date Range'],
                        'Value': [
                            len(self.current_data),
                            f"{(self.current_data['status'] == 'Pass').mean() * 100:.2f}%",
                            f"{self.current_data['sigma_gradient'].mean():.6f}",
                            f"{self.current_data['timestamp'].min()} to {self.current_data['timestamp'].max()}"
                        ]
                    })
                    summary.to_excel(writer, sheet_name='Summary', index=False)

            else:
                # Export to CSV
                self.current_data.to_csv(filename, index=False)

            messagebox.showinfo("Export Complete", f"Data exported to:\n{filename}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")

    def _sort_column(self, col):
        """Sort treeview by column."""
        # Get data
        data = [(self.results_tree.set(child, col), child)
                for child in self.results_tree.get_children('')]

        # Sort data
        data.sort(reverse=False)

        # Rearrange items
        for index, (_, child) in enumerate(data):
            self.results_tree.move(child, '', index)

    def _view_details(self, event):
        """View detailed information for selected item."""
        selection = self.results_tree.selection()
        if not selection:
            return

        # Get selected item data
        item = self.results_tree.item(selection[0])
        values = item['values']

        if not values:
            return

        # Create details dialog
        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title("Analysis Details")
        dialog.geometry("600x400")

        # Create text widget with scrollbar
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        text = tk.Text(text_frame, wrap='word', width=70, height=20)
        scroll = ttk.Scrollbar(text_frame, command=text.yview)
        text.config(yscrollcommand=scroll.set)

        text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        # Add details
        details = f"""Analysis Details
{'=' * 50}
Date: {values[0]}
Model: {values[1]}
Serial: {values[2]}
System: {values[3]}
Status: {values[4]}

Metrics:
- Sigma Gradient: {values[5]}
- Sigma Pass: {values[6]}
- Linearity Pass: {values[7]}
- Risk Category: {values[8]}
- Processing Time: {values[9]}
"""

        text.insert('1.0', details)
        text.config(state='disabled')

        # Close button
        ttk.Button(
            dialog,
            text="Close",
            command=dialog.destroy
        ).pack(pady=(0, 10))

    def _handle_selection(self, event):
        """Handle row selection in the tree."""
        try:
            selected_items = self.results_tree.selection()
            self.selected_ids = []
            
            for item in selected_items:
                values = self.results_tree.item(item, 'values')
                if values:
                    self.selected_ids.append(int(values[0]))  # ID is first column
                    
        except Exception as e:
            self.logger.error(f"Selection handling error: {e}")

    def export_results(self):
        """Export historical data to Excel."""
        if self.current_data is None or len(self.current_data) == 0:
            messagebox.showwarning("Export", "No data available to export")
            return
            
        try:
            # Get save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Export Historical Data"
            )
            
            if not filename:
                return
                
            # Export to Excel
            self.current_data.to_excel(filename, index=False)
            messagebox.showinfo("Export", f"Data exported successfully to:\n{filename}")
            self.logger.info(f"Exported historical data to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
            self.logger.error(f"Export failed: {e}")