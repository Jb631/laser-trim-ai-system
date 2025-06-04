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
import threading

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
        """Create historical page content with consistent theme (matching batch processing)."""
        # Main scrollable container (matching batch processing theme)
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create sections in order (matching batch processing pattern)
        self._create_header()
        self._create_query_section_ctk()
        self._create_results_section_ctk()
        self._create_charts_section_ctk()

    def _create_header(self):
        """Create header section (matching batch processing theme)."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Historical Data Analysis",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)

    def _create_query_section_ctk(self):
        """Create query filters section (matching batch processing theme)."""
        self.query_frame = ctk.CTkFrame(self.main_container)
        self.query_frame.pack(fill='x', pady=(0, 20))

        self.query_label = ctk.CTkLabel(
            self.query_frame,
            text="Query Filters:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.query_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Filters container
        self.filters_container = ctk.CTkFrame(self.query_frame)
        self.filters_container.pack(fill='x', padx=15, pady=(0, 15))

        # First row of filters
        filter_row1 = ctk.CTkFrame(self.filters_container)
        filter_row1.pack(fill='x', padx=10, pady=(10, 5))

        # Model filter
        model_label = ctk.CTkLabel(filter_row1, text="Model:")
        model_label.pack(side='left', padx=10, pady=10)

        self.model_var = tk.StringVar()
        self.model_entry = ctk.CTkEntry(
            filter_row1,
            textvariable=self.model_var,
            width=120,
            height=30
        )
        self.model_entry.pack(side='left', padx=(0, 20), pady=10)

        # Serial filter
        serial_label = ctk.CTkLabel(filter_row1, text="Serial:")
        serial_label.pack(side='left', padx=10, pady=10)

        self.serial_var = tk.StringVar()
        self.serial_entry = ctk.CTkEntry(
            filter_row1,
            textvariable=self.serial_var,
            width=120,
            height=30
        )
        self.serial_entry.pack(side='left', padx=(0, 20), pady=10)

        # Date range
        date_label = ctk.CTkLabel(filter_row1, text="Date Range:")
        date_label.pack(side='left', padx=10, pady=10)

        self.date_range_var = tk.StringVar(value="Last 30 days")
        self.date_combo = ctk.CTkComboBox(
            filter_row1,
            variable=self.date_range_var,
            values=[
                "Today", "Last 7 days", "Last 30 days",
                "Last 90 days", "Last year", "All time"
            ],
            width=120,
            height=30
        )
        self.date_combo.pack(side='left', padx=(0, 10), pady=10)

        # Second row of filters
        filter_row2 = ctk.CTkFrame(self.filters_container)
        filter_row2.pack(fill='x', padx=10, pady=(5, 10))

        # Status filter
        status_label = ctk.CTkLabel(filter_row2, text="Status:")
        status_label.pack(side='left', padx=10, pady=10)

        self.status_var = tk.StringVar(value="All")
        self.status_combo = ctk.CTkComboBox(
            filter_row2,
            variable=self.status_var,
            values=["All", "Pass", "Fail", "Warning"],
            width=100,
            height=30
        )
        self.status_combo.pack(side='left', padx=(0, 20), pady=10)

        # Risk filter
        risk_label = ctk.CTkLabel(filter_row2, text="Risk:")
        risk_label.pack(side='left', padx=10, pady=10)

        self.risk_var = tk.StringVar(value="All")
        self.risk_combo = ctk.CTkComboBox(
            filter_row2,
            variable=self.risk_var,
            values=["All", "High", "Medium", "Low"],
            width=100,
            height=30
        )
        self.risk_combo.pack(side='left', padx=(0, 20), pady=10)

        # Limit filter
        limit_label = ctk.CTkLabel(filter_row2, text="Limit:")
        limit_label.pack(side='left', padx=10, pady=10)

        self.limit_var = tk.StringVar(value="100")
        self.limit_combo = ctk.CTkComboBox(
            filter_row2,
            variable=self.limit_var,
            values=["50", "100", "500", "1000", "All"],
            width=100,
            height=30
        )
        self.limit_combo.pack(side='left', padx=(0, 10), pady=10)

        # Action buttons
        button_frame = ctk.CTkFrame(self.filters_container)
        button_frame.pack(fill='x', padx=10, pady=(10, 10))

        self.query_btn = ctk.CTkButton(
            button_frame,
            text="Run Query",
            command=self._run_query,
            width=120,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="blue",
            hover_color="darkblue"
        )
        self.query_btn.pack(side='left', padx=(10, 10), pady=10)

        clear_btn = ctk.CTkButton(
            button_frame,
            text="Clear Filters",
            command=self._clear_filters,
            width=120,
            height=40
        )
        clear_btn.pack(side='left', padx=(0, 10), pady=10)

        export_btn = ctk.CTkButton(
            button_frame,
            text="Export Results",
            command=self._export_results,
            width=120,
            height=40
        )
        export_btn.pack(side='left', padx=(0, 10), pady=10)

    def _create_results_section_ctk(self):
        """Create results display section (matching batch processing theme)."""
        self.results_frame = ctk.CTkFrame(self.main_container)
        self.results_frame.pack(fill='x', pady=(0, 20))

        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="Query Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.results_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Results display
        self.results_display = ctk.CTkTextbox(
            self.results_frame,
            height=200,
            state="disabled"
        )
        self.results_display.pack(fill='both', expand=True, padx=15, pady=(0, 15))

    def _create_charts_section_ctk(self):
        """Create data visualization section (matching batch processing theme)."""
        self.charts_frame = ctk.CTkFrame(self.main_container)
        self.charts_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.charts_label = ctk.CTkLabel(
            self.charts_frame,
            text="Data Visualization:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.charts_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Charts container
        self.charts_container = ctk.CTkFrame(self.charts_frame)
        self.charts_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Chart tabs
        self.chart_tabview = ctk.CTkTabview(self.charts_container)
        self.chart_tabview.pack(fill='both', expand=True, padx=10, pady=10)

        # Add tabs
        self.chart_tabview.add("Pass Rate Trend")
        self.chart_tabview.add("Sigma Distribution")
        self.chart_tabview.add("Model Comparison")

        # Placeholder for charts
        self.trend_chart_label = ctk.CTkLabel(
            self.chart_tabview.tab("Pass Rate Trend"),
            text="No data loaded",
            font=ctk.CTkFont(size=12)
        )
        self.trend_chart_label.pack(expand=True)

        self.distribution_chart_label = ctk.CTkLabel(
            self.chart_tabview.tab("Sigma Distribution"),
            text="No data loaded",
            font=ctk.CTkFont(size=12)
        )
        self.distribution_chart_label.pack(expand=True)

        self.comparison_chart_label = ctk.CTkLabel(
            self.chart_tabview.tab("Model Comparison"),
            text="No data loaded",
            font=ctk.CTkFont(size=12)
        )
        self.comparison_chart_label.pack(expand=True)

    def _run_query(self):
        """Run database query with current filters."""
        if not self.db_manager:
            messagebox.showerror("Error", "Database not connected")
            return

        # Update UI and run query in background
        self.query_btn.configure(state='disabled', text='Querying...')
        thread = threading.Thread(target=self._run_query_background, daemon=True)
        thread.start()

    def _run_query_background(self):
        """Run database query in background thread."""
        try:
            # Get filter values - use the actual variables that exist in the UI
            model = self.model_var.get().strip() if self.model_var.get().strip() else None
            serial = self.serial_var.get().strip() if self.serial_var.get().strip() else None
            status = self.status_var.get() if self.status_var.get() != "All" else None
            risk = self.risk_var.get() if self.risk_var.get() != "All" else None
            
            # Get date range
            start_date = None
            end_date = None
            date_range = self.date_range_var.get()
            
            if date_range != "All time":
                days_map = {
                    "Today": 1,
                    "Last 7 days": 7,
                    "Last 30 days": 30,
                    "Last 90 days": 90,
                    "Last year": 365
                }
                days_back = days_map.get(date_range, 30)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)

            # Get limit
            limit_str = self.limit_var.get()
            limit = None if limit_str == "All" else int(limit_str)

            # Query database using the correct parameters
            results = self.db_manager.get_historical_data(
                model=model,
                serial=serial,
                start_date=start_date,
                end_date=end_date,
                status=status,
                risk_category=risk,
                limit=limit,
                include_tracks=True
            )

            # Update UI in main thread
            self.after(0, self._display_results, results)
            self.after(0, lambda: self.query_btn.configure(state='normal', text='Run Query'))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Query Error", f"Failed to query database:\n{str(e)}"))
            self.after(0, lambda: self.query_btn.configure(state='normal', text='Run Query'))

    def _display_results(self, results):
        """Display query results in the CTk textbox."""
        try:
            # Clear existing results
            self.results_display.configure(state="normal")
            self.results_display.delete('1.0', 'end')
            
            if not results:
                self.results_display.insert('1.0', "No data found matching the criteria")
                self.results_display.configure(state="disabled")
                return
                
            # Display summary
            total_count = len(results)
            pass_count = sum(1 for r in results if r.overall_status.value == "Pass")
            fail_count = total_count - pass_count
            pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0
            
            summary = f"Query Results Summary:\n"
            summary += f"Total Records: {total_count}\n"
            summary += f"Pass: {pass_count} ({pass_rate:.1f}%)\n"
            summary += f"Fail: {fail_count} ({100-pass_rate:.1f}%)\n\n"
            
            # Display detailed results
            summary += "Detailed Results:\n"
            summary += "-" * 80 + "\n"
            summary += f"{'Date':<12} {'Model':<8} {'Serial':<12} {'Status':<8} {'Sigma':<10} {'Pass':<6}\n"
            summary += "-" * 80 + "\n"
            
            for result in results:
                date_str = result.timestamp.strftime('%Y-%m-%d') if hasattr(result, 'timestamp') else 'Unknown'
                model = getattr(result, 'model', 'Unknown')[:8]
                serial = getattr(result, 'serial', 'Unknown')[:12]
                status = result.overall_status.value[:8]
                
                # Get sigma and pass info from first track
                sigma = "N/A"
                sigma_pass = "N/A"
                if result.tracks and len(result.tracks) > 0:
                    track = result.tracks[0]
                    if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                        sigma = f"{track.sigma_gradient:.4f}"
                    if hasattr(track, 'sigma_pass'):
                        sigma_pass = "✓" if track.sigma_pass else "✗"
                
                line = f"{date_str:<12} {model:<8} {serial:<12} {status:<8} {sigma:<10} {sigma_pass:<6}\n"
                summary += line
            
            self.results_display.insert('1.0', summary)
            self.results_display.configure(state="disabled")
            
            # Store current data for potential export
            self.current_data = results
            
            logger.info(f"Displayed {len(results)} query results")
            
        except Exception as e:
            logger.error(f"Error displaying results: {e}")
            self.results_display.configure(state="normal")
            self.results_display.delete('1.0', 'end')
            self.results_display.insert('1.0', f"Error displaying results: {str(e)}")
            self.results_display.configure(state="disabled")

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
        text.configure(yscrollcommand=scroll.set)

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
        text.configure(state='disabled')

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