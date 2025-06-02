"""
ChartWidget for QA Dashboard

A versatile matplotlib wrapper widget with zoom, pan, and export functionality.
Supports multiple chart types for QA data visualization.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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


class ChartWidget(ttk.Frame):
    """
    Enhanced matplotlib chart widget for QA data visualization.

    Features:
    - Multiple chart types (line, bar, scatter, histogram, heatmap)
    - Interactive zoom and pan
    - Export to various formats
    - Customizable styling
    - Real-time updates
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
        self.configure(relief='solid', borderwidth=1, padding=5)

        # Create matplotlib figure
        self.figure = Figure(figsize=self.figsize, dpi=100)
        self.figure.patch.set_facecolor('white')

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Create toolbar frame
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill='x', side='bottom')

        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Add custom buttons
        ttk.Separator(toolbar_frame, orient='vertical').pack(side='left', fill='y', padx=5)

        # Export button
        export_btn = ttk.Button(toolbar_frame, text="Export",
                                command=self._export_chart)
        export_btn.pack(side='left', padx=2)

        # Clear button
        clear_btn = ttk.Button(toolbar_frame, text="Clear",
                               command=self.clear_chart)
        clear_btn.pack(side='left', padx=2)

        # Chart type selector
        ttk.Label(toolbar_frame, text="Type:").pack(side='left', padx=(10, 2))
        self.type_var = tk.StringVar(value=self.chart_type)
        type_combo = ttk.Combobox(toolbar_frame, textvariable=self.type_var,
                                  values=['line', 'bar', 'scatter', 'histogram', 'heatmap'],
                                  width=10, state='readonly')
        type_combo.pack(side='left', padx=2)
        type_combo.bind('<<ComboboxSelected>>', lambda e: self._change_chart_type())

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
                  self.qa_colors['warning'], self.qa_colors['success']]
        
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

    def _change_chart_type(self):
        """Change chart type."""
        self.chart_type = self.type_var.get()
        # Optionally redraw with new type

    def _export_chart(self):
        """Export chart to file."""
        filetypes = [
            ('PNG files', '*.png'),
            ('PDF files', '*.pdf'),
            ('SVG files', '*.svg'),
            ('All files', '*.*')
        ]

        filename = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=filetypes,
            title='Export Chart'
        )

        if filename:
            try:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Export Successful",
                                    f"Chart exported to {filename}")
            except Exception as e:
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