"""
ChartWidget for QA Dashboard

A versatile matplotlib wrapper widget with zoom, pan, and export functionality.
Supports multiple chart types for QA data visualization.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type
)
from datetime import datetime
import pandas as pd
import matplotlib.dates as mdates


class ChartWidget(ctk.CTkFrame):
    """
    Enhanced matplotlib chart widget for QA data visualization.

    Features:
    - Multiple chart types (line, bar, scatter, histogram, heatmap)
    - Interactive zoom and pan
    - Export to various formats
    - Customizable styling
    - Real-time updates
    - CustomTkinter integration
    """

    def __init__(self, parent, chart_type: str = 'line',
                 title: str = "", figsize: Tuple[int, int] = (8, 6),
                 style: str = 'default', **kwargs):
        """
        Initialize ChartWidget.

        Args:
            parent: Parent widget
            chart_type: Type of chart ('line', 'bar', 'scatter', 'histogram', 'heatmap')
            title: Chart title
            figsize: Figure size in inches
            style: Matplotlib style
        """
        super().__init__(parent, **kwargs)

        self.chart_type = chart_type
        self.title = title
        self.figsize = figsize
        self.data = {}

        # Apply matplotlib style safely
        available_styles = plt.style.available
        if 'seaborn-v0_8' in available_styles:
            plt.style.use('seaborn-v0_8')
        elif 'ggplot' in available_styles:
            plt.style.use('ggplot')
        else:
            plt.style.use('default')

        # Colors for QA metrics
        self.qa_colors = {
            'pass': '#27ae60',
            'fail': '#e74c3c',
            'warning': '#f39c12',
            'primary': '#3498db',
            'secondary': '#9b59b6',
            'neutral': '#95a5a6'
        }

        self._setup_ui()

    def _setup_ui(self):
        """Set up the chart widget UI."""
        # Create matplotlib figure
        self.figure = Figure(figsize=self.figsize, dpi=100)
        self.figure.patch.set_facecolor('white')

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Create toolbar frame
        toolbar_frame = ctk.CTkFrame(self)
        toolbar_frame.pack(fill='x', side='bottom')

        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Add custom buttons
        # Export button
        export_btn = ctk.CTkButton(toolbar_frame, text="Export",
                                command=self._export_chart, width=80, height=25)
        export_btn.pack(side='left', padx=2, pady=2)

        # Clear button
        clear_btn = ctk.CTkButton(toolbar_frame, text="Clear",
                               command=self.clear_chart, width=80, height=25)
        clear_btn.pack(side='left', padx=2, pady=2)

        # Chart type selector
        ctk.CTkLabel(toolbar_frame, text="Type:").pack(side='left', padx=(10, 2))
        self.type_var = tk.StringVar(value=self.chart_type)
        type_combo = ctk.CTkComboBox(toolbar_frame, variable=self.type_var,
                                   values=['line', 'bar', 'scatter', 'histogram', 'heatmap'],
                                   width=120,
                                   command=self._change_chart_type)
        type_combo.pack(side='left', padx=2)

    def update_chart_data(self, data: pd.DataFrame):
        """
        Update chart with new data based on chart type.
        
        Args:
            data: DataFrame with columns appropriate for the chart type
        """
        # Store data for chart type changes
        self._last_data = data.copy() if data is not None and len(data) > 0 else None
        
        if data is None or len(data) == 0:
            self.clear_chart()
            return
            
        try:
            self.clear_chart()
            
            if self.chart_type == 'line':
                self._plot_line_from_data(data)
            elif self.chart_type == 'bar':
                self._plot_bar_from_data(data)
            elif self.chart_type == 'scatter':
                self._plot_scatter_from_data(data)
            elif self.chart_type == 'histogram':
                self._plot_histogram_from_data(data)
            elif self.chart_type == 'heatmap':
                self._plot_heatmap_from_data(data)
            else:
                # Default to line plot
                self._plot_line_from_data(data)
                
        except Exception as e:
            import traceback
            print(f"Error updating chart: {e}")
            print(traceback.format_exc())
            self.clear_chart()
            
    def _plot_line_from_data(self, data: pd.DataFrame):
        """Plot line chart from DataFrame."""
        ax = self.figure.add_subplot(111)
        
        if 'trim_date' in data.columns and 'sigma_gradient' in data.columns:
            # Sort by date
            data_sorted = data.sort_values('trim_date')
            x_data = data_sorted['trim_date']
            y_data = data_sorted['sigma_gradient']
            
            # Convert dates to numeric for trend calculation
            x_numeric = mdates.date2num(pd.to_datetime(x_data))
            
            # Plot the actual data points and line
            ax.plot(x_data, y_data, marker='o', linewidth=2, markersize=4, 
                   color=self.qa_colors['primary'], label='Sigma Gradient', alpha=0.8)
            
            # Add trend line if we have enough data points
            if len(x_data) >= 3:
                # Calculate linear trend
                z = np.polyfit(x_numeric, y_data, 1)
                p = np.poly1d(z)
                trend_y = p(x_numeric)
                
                ax.plot(x_data, trend_y, "--", color=self.qa_colors['warning'], 
                       linewidth=2, alpha=0.7, label=f'Trend (slope: {z[0]:.6f})')
                
                # Add legend
                ax.legend(loc='upper right', fontsize=9)
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Sigma Gradient')
            
            # Format x-axis dates
            if len(x_data) > 0:
                ax.tick_params(axis='x', rotation=45)
                
        if self.title:
            ax.set_title(self.title)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def _plot_bar_from_data(self, data: pd.DataFrame):
        """Plot bar chart from DataFrame."""
        ax = self.figure.add_subplot(111)
        
        if 'month_year' in data.columns and 'track_status' in data.columns:
            categories = [str(m) for m in data['month_year']]
            values = data['track_status'].tolist()
            
            # Color code based on pass rate
            colors = []
            for rate in values:
                if rate >= 95:
                    colors.append(self.qa_colors['pass'])
                elif rate >= 90:
                    colors.append(self.qa_colors['warning'])
                else:
                    colors.append(self.qa_colors['fail'])
            
            bars = ax.bar(categories, values, color=colors)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Pass Rate (%)')
            ax.set_ylim(0, 100)
            
            if len(categories) > 5:
                ax.tick_params(axis='x', rotation=45)
                
        if self.title:
            ax.set_title(self.title)
        ax.grid(True, alpha=0.3, axis='y')
        self.figure.tight_layout()
        self.canvas.draw()
        
    def _plot_scatter_from_data(self, data: pd.DataFrame):
        """Plot scatter chart from DataFrame."""
        ax = self.figure.add_subplot(111)
        
        if 'x' in data.columns and 'y' in data.columns:
            x_data = data['x']
            y_data = data['y']
            
            ax.scatter(x_data, y_data, alpha=0.6, s=50, color=self.qa_colors['primary'])
            ax.set_xlabel('Sigma Gradient')
            ax.set_ylabel('Linearity Error')
            
            # Add correlation if enough points
            if len(x_data) > 5:
                correlation = np.corrcoef(x_data, y_data)[0, 1]
                if not np.isnan(correlation):
                    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                           transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                
        if self.title:
            ax.set_title(self.title)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def _plot_histogram_from_data(self, data: pd.DataFrame):
        """Plot histogram from DataFrame."""
        ax = self.figure.add_subplot(111)
        
        if 'sigma_gradient' in data.columns:
            sigma_data = data['sigma_gradient'].dropna()
            
            if len(sigma_data) > 0:
                ax.hist(sigma_data, bins=min(30, max(10, len(sigma_data) // 5)), 
                       alpha=0.7, edgecolor='black', color=self.qa_colors['primary'])
                
                # Add statistics
                mean = sigma_data.mean()
                std = sigma_data.std()
                ax.axvline(mean, color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {mean:.4f}')
                ax.axvline(mean + std, color='orange', linestyle='--', linewidth=1,
                          label=f'±1σ: {std:.4f}')
                ax.axvline(mean - std, color='orange', linestyle='--', linewidth=1)
                
                ax.set_xlabel('Sigma Gradient')
                ax.set_ylabel('Frequency')
                ax.legend()
                
        if self.title:
            ax.set_title(self.title)
        ax.grid(True, alpha=0.3, axis='y')
        self.figure.tight_layout()
        self.canvas.draw()
        
    def _plot_heatmap_from_data(self, data: pd.DataFrame):
        """Plot heatmap from DataFrame."""
        ax = self.figure.add_subplot(111)
        
        # Check for required columns
        if 'x_values' in data.columns and 'y_values' in data.columns and 'values' in data.columns:
            try:
                # Pivot data for heatmap
                pivot_data = data.pivot(index='y_values', columns='x_values', values='values')
                
                # Create heatmap
                im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', interpolation='nearest')
                
                # Set labels
                ax.set_xticks(range(len(pivot_data.columns)))
                ax.set_yticks(range(len(pivot_data.index)))
                ax.set_xticklabels(pivot_data.columns)
                ax.set_yticklabels(pivot_data.index)
                
                # Add colorbar
                cbar = self.figure.colorbar(im, ax=ax)
                
                # Add text annotations if not too many cells
                if pivot_data.size <= 100:  # Only add text for reasonably sized heatmaps
                    for i in range(len(pivot_data.index)):
                        for j in range(len(pivot_data.columns)):
                            value = pivot_data.values[i, j]
                            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                                   color='black' if 0.3 < im.norm(value) < 0.7 else 'white')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error creating heatmap: {str(e)}", 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Alternative format with matrix data
        elif 'matrix_data' in data.columns and 'row_labels' in data.columns and 'col_labels' in data.columns:
            try:
                matrix = data['matrix_data'].iloc[0]  # Assuming it's stored as a single cell
                row_labels = data['row_labels'].iloc[0]
                col_labels = data['col_labels'].iloc[0]
                
                if isinstance(matrix, str):
                    # Try to parse if it's stored as string
                    import json
                    matrix = json.loads(matrix)
                
                # Create heatmap
                im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
                
                # Set labels
                ax.set_xticks(range(len(col_labels)))
                ax.set_yticks(range(len(row_labels)))
                ax.set_xticklabels(col_labels)
                ax.set_yticklabels(row_labels)
                
                # Add colorbar
                cbar = self.figure.colorbar(im, ax=ax)
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error parsing matrix data: {str(e)}", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Heatmap requires 'x_values', 'y_values', and 'values' columns", 
                   ha='center', va='center', transform=ax.transAxes)
        
        if self.title:
            ax.set_title(self.title)
            
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        """Clear the chart (alias for clear_chart for compatibility)."""
        self.clear_chart()

    def plot_line(self, x_data: List, y_data: List, label: str = "",
                  color: Optional[str] = None, marker: Optional[str] = None,
                  xlabel: str = "", ylabel: str = "", **kwargs):
        """Plot line chart."""
        ax = self.figure.add_subplot(111) if not self.figure.axes else self.figure.axes[0]

        # Use QA color if specified
        if color and color in self.qa_colors:
            color = self.qa_colors[color]

        # Plot data
        line = ax.plot(x_data, y_data, label=label, color=color,
                       marker=marker, **kwargs)[0]

        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if self.title:
            ax.set_title(self.title)

        # Show legend if labels exist
        if label:
            ax.legend()

        # Grid
        ax.grid(True, alpha=0.3)

        # Refresh canvas
        self.canvas.draw()

        return line

    def plot_bar(self, categories: List[str], values: List[float],
                 colors: Optional[List[str]] = None, xlabel: str = "",
                 ylabel: str = "", **kwargs):
        """Plot bar chart."""
        ax = self.figure.add_subplot(111) if not self.figure.axes else self.figure.axes[0]

        # Map QA colors
        if colors:
            colors = [self.qa_colors.get(c, c) for c in colors]
        else:
            colors = self.qa_colors['primary']

        # Plot bars
        bars = ax.bar(categories, values, color=colors, **kwargs)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if self.title:
            ax.set_title(self.title)

        # Rotate x labels if many categories
        if len(categories) > 10:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Grid
        ax.grid(True, alpha=0.3, axis='y')

        # Tight layout
        self.figure.tight_layout()

        # Refresh canvas
        self.canvas.draw()

        return bars

    def plot_scatter(self, x_data: List, y_data: List,
                     colors: Optional[List] = None, sizes: Optional[List] = None,
                     labels: Optional[List[str]] = None, xlabel: str = "",
                     ylabel: str = "", alpha: float = 0.6, **kwargs):
        """Plot scatter chart."""
        ax = self.figure.add_subplot(111) if not self.figure.axes else self.figure.axes[0]

        # Default size
        if sizes is None:
            sizes = 50

        # Map colors
        if colors and isinstance(colors[0], str):
            colors = [self.qa_colors.get(c, c) for c in colors]

        # Remove alpha from kwargs if it exists to avoid conflict
        kwargs.pop('alpha', None)

        # Plot scatter
        scatter = ax.scatter(x_data, y_data, c=colors, s=sizes,
                             alpha=alpha, **kwargs)

        # Add annotations if labels provided
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (x_data[i], y_data[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.7)

        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if self.title:
            ax.set_title(self.title)

        # Grid
        ax.grid(True, alpha=0.3)

        # Refresh canvas
        self.canvas.draw()

        return scatter

    def plot_histogram(self, data: List[float], bins: int = 20,
                       color: Optional[str] = None, xlabel: str = "",
                       ylabel: str = "Frequency", **kwargs):
        """Plot histogram."""
        ax = self.figure.add_subplot(111) if not self.figure.axes else self.figure.axes[0]

        # Use QA color if specified
        if color and color in self.qa_colors:
            color = self.qa_colors[color]
        else:
            color = self.qa_colors['primary']

        # Plot histogram
        n, bins, patches = ax.hist(data, bins=bins, color=color,
                                   alpha=0.7, edgecolor='black', **kwargs)

        # Add statistics
        mean = np.mean(data)
        std = np.std(data)
        ax.axvline(mean, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean:.3f}')
        ax.axvline(mean + std, color='orange', linestyle='--', linewidth=1,
                   label=f'±1σ: {std:.3f}')
        ax.axvline(mean - std, color='orange', linestyle='--', linewidth=1)

        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if self.title:
            ax.set_title(self.title)

        # Legend
        ax.legend()

        # Grid
        ax.grid(True, alpha=0.3, axis='y')

        # Refresh canvas
        self.canvas.draw()

        return n, bins, patches

    def plot_box(self, data: List[List[float]], labels: List[str] = None,
                 xlabel: str = "", ylabel: str = "", **kwargs):
        """Plot box plot."""
        ax = self.figure.add_subplot(111) if not self.figure.axes else self.figure.axes[0]

        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True, **kwargs)

        # Color the boxes
        colors = [self.qa_colors['primary'], self.qa_colors['secondary'], 
                  self.qa_colors['warning'], self.qa_colors['pass']]
        
        for patch, color in zip(bp['boxes'], colors * (len(bp['boxes']) // len(colors) + 1)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if self.title:
            ax.set_title(self.title)

        # Grid
        ax.grid(True, alpha=0.3, axis='y')

        # Refresh canvas
        self.canvas.draw()

        return bp

    def plot_heatmap(self, data: np.ndarray, xlabels: List[str],
                     ylabels: List[str], cmap: str = 'RdYlGn',
                     xlabel: str = "", ylabel: str = "", **kwargs):
        """Plot heatmap."""
        ax = self.figure.add_subplot(111) if not self.figure.axes else self.figure.axes[0]

        # Create heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto', **kwargs)

        # Set ticks
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)

        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add colorbar
        cbar = self.figure.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(len(ylabels)):
            for j in range(len(xlabels)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                               ha='center', va='center', color='black',
                               fontsize=9)

        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if self.title:
            ax.set_title(self.title)

        # Tight layout
        self.figure.tight_layout()

        # Refresh canvas
        self.canvas.draw()

        return im

    def plot_multi_series(self, data_dict: Dict[str, Dict[str, List]],
                          xlabel: str = "", ylabel: str = ""):
        """
        Plot multiple series on the same chart.

        Args:
            data_dict: Dictionary with series names as keys and
                      {'x': x_data, 'y': y_data, 'color': color} as values
        """
        ax = self.figure.add_subplot(111) if not self.figure.axes else self.figure.axes[0]

        # Plot each series
        for series_name, series_data in data_dict.items():
            x_data = series_data['x']
            y_data = series_data['y']
            color = series_data.get('color')

            if color and color in self.qa_colors:
                color = self.qa_colors[color]

            ax.plot(x_data, y_data, label=series_name, color=color,
                    marker='o', markersize=4)

        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if self.title:
            ax.set_title(self.title)

        # Legend
        ax.legend()

        # Grid
        ax.grid(True, alpha=0.3)

        # Refresh canvas
        self.canvas.draw()

    def add_threshold_lines(self, thresholds: Dict[str, float],
                            orientation: str = 'horizontal'):
        """Add threshold lines to the chart."""
        if not self.figure.axes:
            return

        ax = self.figure.axes[0]

        for label, value in thresholds.items():
            color = self.qa_colors.get(label.lower(), 'red')

            if orientation == 'horizontal':
                ax.axhline(y=value, color=color, linestyle='--',
                           label=f'{label}: {value}', alpha=0.7)
            else:
                ax.axvline(x=value, color=color, linestyle='--',
                           label=f'{label}: {value}', alpha=0.7)

        # Update legend
        ax.legend()

        # Refresh canvas
        self.canvas.draw()

    def clear_chart(self):
        """Clear the current chart."""
        self.figure.clear()
        self.canvas.draw()

    def _change_chart_type(self, new_value=None):
        """Change chart type and refresh if data exists."""
        if new_value:
            self.chart_type = new_value
        else:
            self.chart_type = self.type_var.get()
        
        # If we have data stored, re-plot with new chart type
        if hasattr(self, '_last_data') and self._last_data is not None:
            self.update_chart_data(self._last_data)

    def _export_chart(self):
        """Export chart to file."""
        filetypes = [
            ('PNG files', '*.png'),
            ('PDF files', '*.pdf'),
            ('SVG files', '*.svg'),
            ('All files', '*.*')
        ]

        # Use parent window for dialog to ensure proper modal behavior
        parent = self.winfo_toplevel()
        filename = filedialog.asksaveasfilename(
            parent=parent,
            defaultextension='.png',
            filetypes=filetypes,
            title='Export Chart'
        )

        if filename:
            try:
                # Ensure directory exists
                from pathlib import Path
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                
                # Save with high quality
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Export Successful",
                                    f"Chart exported to {filename}")
            except Exception as e:
                import traceback
                print(f"Export error: {e}")
                print(traceback.format_exc())
                messagebox.showerror("Export Error",
                                     f"Failed to export chart: {str(e)}")

    def update_chart(self, update_func: Callable):
        """Update chart with a custom function."""
        update_func(self.figure, self.figure.axes[0] if self.figure.axes else None)
        self.canvas.draw()


# Example usage and testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("ChartWidget Demo")
    root.geometry("1000x700")

    # Create notebook for different chart examples
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # Line chart example
    line_frame = ttk.Frame(notebook)
    notebook.add(line_frame, text="Line Chart")

    line_chart = ChartWidget(line_frame, chart_type='line',
                             title="Sigma Gradient Trend")
    line_chart.pack(fill='both', expand=True)

    # Sample data
    days = list(range(1, 31))
    sigma_values = [0.02 + 0.01 * np.sin(d / 5) + np.random.normal(0, 0.002)
                    for d in days]

    line_chart.plot_line(days, sigma_values, label="Sigma Gradient",
                         color='primary', marker='o', markersize=4,
                         xlabel="Day", ylabel="Sigma Gradient")
    line_chart.add_threshold_lines({'warning': 0.025, 'critical': 0.03})

    # Bar chart example
    bar_frame = ttk.Frame(notebook)
    notebook.add(bar_frame, text="Bar Chart")

    bar_chart = ChartWidget(bar_frame, chart_type='bar',
                            title="Model Performance Comparison")
    bar_chart.pack(fill='both', expand=True)

    models = ['8340', '8555', '6845', '7825']
    pass_rates = [94.5, 96.2, 91.8, 98.1]
    colors = ['warning', 'pass', 'fail', 'pass']

    bar_chart.plot_bar(models, pass_rates, colors=colors,
                       xlabel="Model", ylabel="Pass Rate (%)")

    # Scatter plot example
    scatter_frame = ttk.Frame(notebook)
    notebook.add(scatter_frame, text="Scatter Plot")

    scatter_chart = ChartWidget(scatter_frame, chart_type='scatter',
                                title="Sigma vs Resistance Change")
    scatter_chart.pack(fill='both', expand=True)

    # Generate sample data
    n_points = 50
    sigma_data = np.random.normal(0.025, 0.005, n_points)
    resistance_data = np.random.normal(2, 0.5, n_points)
    risk_colors = ['pass' if s < 0.025 else 'warning' if s < 0.03 else 'fail'
                   for s in sigma_data]

    scatter_chart.plot_scatter(sigma_data, resistance_data, colors=risk_colors,
                               xlabel="Sigma Gradient",
                               ylabel="Resistance Change (%)")

    # Histogram example
    hist_frame = ttk.Frame(notebook)
    notebook.add(hist_frame, text="Histogram")

    hist_chart = ChartWidget(hist_frame, chart_type='histogram',
                             title="Sigma Gradient Distribution")
    hist_chart.pack(fill='both', expand=True)

    # Generate sample data
    hist_data = np.random.normal(0.025, 0.008, 200)
    hist_chart.plot_histogram(hist_data, bins=30, color='primary',
                              xlabel="Sigma Gradient")

    # Heatmap example
    heatmap_frame = ttk.Frame(notebook)
    notebook.add(heatmap_frame, text="Heatmap")

    heatmap_chart = ChartWidget(heatmap_frame, chart_type='heatmap',
                                title="Model Performance Matrix")
    heatmap_chart.pack(fill='both', expand=True)

    # Generate sample data
    models_h = ['8340', '8555', '6845', '7825']
    metrics = ['Pass Rate', 'Avg Sigma', 'Risk Score', 'Efficiency']
    data = np.random.rand(4, 4) * 100

    heatmap_chart.plot_heatmap(data, models_h, metrics,
                               xlabel="Model", ylabel="Metric")

    root.mainloop()
