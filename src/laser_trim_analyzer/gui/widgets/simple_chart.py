"""
SimpleChartWidget - Clean, industry-standard charting for QA dashboards.

This is a redesigned chart widget that follows the same visual style as the
single file analysis charts (create_analysis_plot). It provides:

- Clean, professional visualizations
- Consistent color scheme with QA_COLORS
- Industry-standard SPC control charts
- Auto-detection of data columns
- Clear error messages when data issues occur

Usage:
    chart = SimpleChartWidget(parent, title="My Chart")
    chart.plot_control_chart(df, value_col='sigma_gradient', date_col='trim_date')
    chart.plot_trend(df, y_col='value', x_col='date')
    chart.plot_distribution(df, value_col='sigma_gradient')
    chart.plot_bar(df, category_col='model', value_col='pass_rate')
"""

import customtkinter as ctk
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# QA Color scheme matching plotting_utils.py
QA_COLORS = {
    'pass': '#27ae60',      # Success green
    'fail': '#e74c3c',      # Error red
    'warning': '#f39c12',   # Warning orange
    'info': '#3498db',      # Primary blue
    'untrimmed': '#3498db', # Primary blue
    'trimmed': '#27ae60',   # Success green
    'filtered': '#9b59b6',  # Purple
    'spec_limit': '#e74c3c',# Red
    'threshold': '#f39c12', # Orange
    'grid': '#95a5a6',      # Gray
    'text': '#2c3e50',      # Dark text
    'background': '#ffffff', # White
    # SPC colors
    'mean_line': '#2F4F4F',     # Dark slate gray
    'control_limit': '#FF8C00', # Dark orange
    'warning_limit': '#FFD700', # Gold
    'target': '#228B22',        # Forest green
}


class SimpleChartWidget(ctk.CTkFrame):
    """
    Clean, simple chart widget following the single file analysis visual style.

    This widget provides easy-to-use methods for common chart types:
    - Control charts (SPC X-bar)
    - Trend lines
    - Distributions/Histograms
    - Bar charts
    - Pie charts
    """

    def __init__(self, parent, title: str = "", figsize: Tuple[float, float] = (10, 6),
                 show_toolbar: bool = False, **kwargs):
        """
        Initialize SimpleChartWidget.

        Args:
            parent: Parent widget
            title: Chart title (can be changed per plot)
            figsize: Figure size in inches (width, height)
            show_toolbar: Whether to show matplotlib toolbar
        """
        super().__init__(parent, **kwargs)

        self.title = title
        self.figsize = figsize
        self._has_data = False

        # Create matplotlib figure
        self.figure = Figure(figsize=figsize, dpi=100, facecolor='white')
        self.figure.set_tight_layout(True)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        if show_toolbar:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            self.toolbar = NavigationToolbar2Tk(self.canvas, self)
            self.toolbar.update()

        # Show initial placeholder
        self._show_placeholder("No Data", "Complete an analysis or query to view chart")

    def show_placeholder(self, title: str, message: str):
        """Display a placeholder message (public method for compatibility)."""
        self._show_placeholder(title, message)

    def _show_placeholder(self, title: str, message: str):
        """Display a placeholder message."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#f8f9fa')

        # Center text
        ax.text(0.5, 0.5, f"{title}\n\n{message}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='#6c757d', style='italic',
                wrap=True)
        ax.axis('off')

        self._has_data = False
        self.canvas.draw_idle()

    def _show_error(self, title: str, message: str):
        """Display an error message."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#fff5f5')

        ax.text(0.5, 0.5, f"⚠ {title}\n\n{message}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color=QA_COLORS['fail'],
                wrap=True)
        ax.axis('off')

        self._has_data = False
        self.canvas.draw_idle()

    def _apply_style(self, ax):
        """Apply consistent styling to axes."""
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.3, color=QA_COLORS['grid'])
        ax.tick_params(colors=QA_COLORS['text'])
        for spine in ax.spines.values():
            spine.set_color(QA_COLORS['grid'])

    def clear(self):
        """Clear the chart and show placeholder."""
        self._show_placeholder("No Data", "Complete an analysis or query to view chart")

    def plot_control_chart(self, data: pd.DataFrame,
                           value_col: str,
                           date_col: str = 'trim_date',
                           spec_limits: Optional[Tuple[float, float]] = None,
                           target_value: Optional[float] = None,
                           title: str = None,
                           show_moving_average: bool = True):
        """
        Plot an industry-standard SPC control chart.

        Args:
            data: DataFrame with time series data
            value_col: Column name for measured values
            date_col: Column name for dates/timestamps
            spec_limits: Tuple of (LSL, USL) specification limits
            target_value: Target/nominal value
            title: Chart title (overrides widget title)
            show_moving_average: Whether to show 7-day moving average
        """
        # Validate data
        if data is None or len(data) == 0:
            self._show_placeholder("No Data", "DataFrame is empty")
            return

        # Check columns
        if value_col not in data.columns:
            self._show_error("Missing Column",
                f"Column '{value_col}' not found.\nAvailable: {list(data.columns)}")
            return

        if date_col not in data.columns:
            # Try to detect a date column
            for col in ['date', 'Date', 'timestamp', 'time']:
                if col in data.columns:
                    date_col = col
                    break
            else:
                self._show_error("Missing Date Column",
                    f"Column '{date_col}' not found and no date column detected")
                return

        # Prepare data
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])

        if len(df) < 3:
            self._show_placeholder("Insufficient Data",
                f"Need at least 3 data points. Found: {len(df)}")
            return

        df = df.sort_values(date_col)
        dates = df[date_col]
        values = df[value_col].astype(float)

        # Clear and create new plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._apply_style(ax)

        # Calculate SPC control limits using moving range method
        mean_val = values.mean()
        vals = values.values

        # Moving range estimate of sigma
        mr = np.abs(np.diff(vals))
        if len(mr) > 0 and np.isfinite(mr).all():
            mrbar = np.mean(mr)
            d2 = 1.128  # for n=2 in moving range
            sigma_est = mrbar / d2 if mrbar > 0 else values.std()
        else:
            sigma_est = values.std()

        # 3-sigma control limits
        ucl = mean_val + 3 * sigma_est
        lcl = mean_val - 3 * sigma_est

        # 2-sigma warning limits
        uwl = mean_val + 2 * sigma_est
        lwl = mean_val - 2 * sigma_est

        # Plot data points
        ax.plot(dates, values, 'o-', color=QA_COLORS['info'],
                linewidth=1.5, markersize=4, alpha=0.8, label='Individual Values')

        # Plot 7-day moving average if requested and enough data
        if show_moving_average and len(values) >= 7:
            ma = values.rolling(window=7, min_periods=3).mean()
            ax.plot(dates, ma, '-', color=QA_COLORS['trimmed'],
                   linewidth=2.5, alpha=0.8, label='7-Day Moving Avg')

        # Plot mean line
        ax.axhline(y=mean_val, color=QA_COLORS['mean_line'], linestyle='-',
                  linewidth=2, alpha=0.9, label=f'Mean: {mean_val:.4f}')

        # Plot control limits (3-sigma)
        ax.axhline(y=ucl, color=QA_COLORS['control_limit'], linestyle='--',
                  linewidth=1.5, alpha=0.8, label=f'UCL (3σ): {ucl:.4f}')
        ax.axhline(y=lcl, color=QA_COLORS['control_limit'], linestyle='--',
                  linewidth=1.5, alpha=0.8, label=f'LCL (3σ): {lcl:.4f}')

        # Plot warning limits (2-sigma) with lighter styling
        ax.axhline(y=uwl, color=QA_COLORS['warning_limit'], linestyle=':',
                  linewidth=1, alpha=0.6)
        ax.axhline(y=lwl, color=QA_COLORS['warning_limit'], linestyle=':',
                  linewidth=1, alpha=0.6)

        # Plot target if provided
        if target_value is not None:
            ax.axhline(y=target_value, color=QA_COLORS['target'], linestyle='-.',
                      linewidth=2, alpha=0.8, label=f'Target: {target_value:.4f}')

        # Plot spec limits if provided
        if spec_limits:
            lsl, usl = spec_limits
            ax.axhline(y=usl, color=QA_COLORS['spec_limit'], linestyle='-',
                      linewidth=2, alpha=0.7, label=f'USL: {usl:.4f}')
            ax.axhline(y=lsl, color=QA_COLORS['spec_limit'], linestyle='-',
                      linewidth=2, alpha=0.7, label=f'LSL: {lsl:.4f}')
            # Shade spec region
            ax.fill_between(dates, lsl, usl, alpha=0.05, color=QA_COLORS['spec_limit'])

        # Highlight out-of-control points
        out_of_control = (values > ucl) | (values < lcl)
        if out_of_control.any():
            ax.scatter(dates[out_of_control], values[out_of_control],
                      color=QA_COLORS['fail'], s=100, zorder=5, marker='X',
                      label='Out of Control')

        # Formatting
        chart_title = title or self.title or f"{value_col.replace('_', ' ').title()} Control Chart"
        ax.set_title(chart_title, fontsize=12, fontweight='bold', color=QA_COLORS['text'])
        ax.set_xlabel('Date', fontsize=10, color=QA_COLORS['text'])
        ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=10, color=QA_COLORS['text'])

        # Date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Legend - compact, outside if many items
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            if len(handles) > 5:
                ax.legend(handles[:5], labels[:5], loc='upper left', fontsize=8,
                         framealpha=0.9, ncol=2)
            else:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        self.figure.tight_layout()
        self._has_data = True
        self.canvas.draw_idle()

    def plot_trend(self, data: pd.DataFrame,
                   y_col: str,
                   x_col: str = None,
                   title: str = None,
                   show_trend_line: bool = True,
                   color: str = None):
        """
        Plot a simple trend line chart.

        Args:
            data: DataFrame with data
            y_col: Column for Y values
            x_col: Column for X values (auto-detects date columns if None)
            title: Chart title
            show_trend_line: Whether to show linear trend line
            color: Line color (uses default if None)
        """
        if data is None or len(data) == 0:
            self._show_placeholder("No Data", "DataFrame is empty")
            return

        if y_col not in data.columns:
            self._show_error("Missing Column",
                f"Column '{y_col}' not found.\nAvailable: {list(data.columns)}")
            return

        # Auto-detect x column if not provided
        if x_col is None:
            for col in ['trim_date', 'date', 'Date', 'timestamp', 'time', 'index']:
                if col in data.columns:
                    x_col = col
                    break

        if x_col is None and len(data.columns) > 1:
            x_col = data.columns[0] if data.columns[0] != y_col else data.columns[1]

        if x_col is None:
            self._show_error("Missing X Column", "Could not determine X column for trend chart")
            return

        # Prepare data
        df = data.copy()
        df = df.dropna(subset=[y_col])

        if len(df) < 2:
            self._show_placeholder("Insufficient Data",
                f"Need at least 2 data points. Found: {len(df)}")
            return

        # Try to convert x to datetime
        is_datetime = False
        try:
            df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            if df[x_col].notna().any():
                is_datetime = True
                df = df.sort_values(x_col)
        except:
            pass

        x_vals = df[x_col]
        y_vals = df[y_col].astype(float)

        # Clear and plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._apply_style(ax)

        line_color = color or QA_COLORS['info']

        ax.plot(x_vals, y_vals, 'o-', color=line_color, linewidth=2, markersize=5, alpha=0.8)

        # Add trend line if requested
        if show_trend_line and len(y_vals) >= 3:
            try:
                if is_datetime:
                    x_numeric = mdates.date2num(x_vals)
                else:
                    x_numeric = np.arange(len(x_vals))

                z = np.polyfit(x_numeric, y_vals, 1)
                p = np.poly1d(z)
                trend_y = p(x_numeric)

                slope_dir = "↑" if z[0] > 0 else "↓" if z[0] < 0 else "→"
                ax.plot(x_vals, trend_y, '--', color=QA_COLORS['warning'],
                       linewidth=2, alpha=0.7, label=f'Trend {slope_dir} (slope: {z[0]:.4f})')
                ax.legend(loc='upper right', fontsize=9)
            except Exception as e:
                logger.debug(f"Could not add trend line: {e}")

        # Formatting
        chart_title = title or self.title or f"{y_col.replace('_', ' ').title()} Trend"
        ax.set_title(chart_title, fontsize=12, fontweight='bold', color=QA_COLORS['text'])
        ax.set_xlabel(x_col.replace('_', ' ').title() if x_col else 'Index', fontsize=10)
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=10)

        if is_datetime:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        self.figure.tight_layout()
        self._has_data = True
        self.canvas.draw_idle()

    def plot_distribution(self, data: pd.DataFrame,
                          value_col: str,
                          spec_limits: Optional[Tuple[float, float]] = None,
                          target_value: Optional[float] = None,
                          title: str = None,
                          bins: int = None):
        """
        Plot a histogram/distribution with optional spec limits.

        Args:
            data: DataFrame with data
            value_col: Column for values to plot
            spec_limits: Tuple of (LSL, USL)
            target_value: Target/nominal value
            title: Chart title
            bins: Number of histogram bins (auto if None)
        """
        if data is None or len(data) == 0:
            self._show_placeholder("No Data", "DataFrame is empty")
            return

        if value_col not in data.columns:
            self._show_error("Missing Column",
                f"Column '{value_col}' not found.\nAvailable: {list(data.columns)}")
            return

        values = data[value_col].dropna().astype(float)

        if len(values) < 5:
            self._show_placeholder("Insufficient Data",
                f"Need at least 5 data points. Found: {len(values)}")
            return

        # Clear and plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._apply_style(ax)

        # Calculate histogram bins
        n_bins = bins or min(30, int(np.sqrt(len(values))))

        # Plot histogram
        ax.hist(values, bins=n_bins, density=True, alpha=0.7,
               color=QA_COLORS['info'], edgecolor='white', linewidth=1)

        # Add normal curve
        mean_val = values.mean()
        std_val = values.std()
        x_range = np.linspace(values.min(), values.max(), 100)
        normal_curve = (1 / (std_val * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * ((x_range - mean_val) / std_val) ** 2)
        ax.plot(x_range, normal_curve, '-', color=QA_COLORS['trimmed'],
               linewidth=2, alpha=0.8, label=f'Normal (μ={mean_val:.3f}, σ={std_val:.3f})')

        # Add mean line
        ax.axvline(x=mean_val, color=QA_COLORS['mean_line'], linestyle='-',
                  linewidth=2, label=f'Mean: {mean_val:.4f}')

        # Add target if provided
        if target_value is not None:
            ax.axvline(x=target_value, color=QA_COLORS['target'], linestyle='--',
                      linewidth=2, label=f'Target: {target_value:.4f}')

        # Add spec limits if provided
        if spec_limits:
            lsl, usl = spec_limits
            ax.axvline(x=lsl, color=QA_COLORS['spec_limit'], linestyle='-.',
                      linewidth=2, label=f'LSL: {lsl:.4f}')
            ax.axvline(x=usl, color=QA_COLORS['spec_limit'], linestyle='-.',
                      linewidth=2, label=f'USL: {usl:.4f}')
            # Shade out-of-spec regions
            ax.axvspan(values.min(), lsl, alpha=0.1, color=QA_COLORS['fail'])
            ax.axvspan(usl, values.max(), alpha=0.1, color=QA_COLORS['fail'])

        # Formatting
        chart_title = title or self.title or f"{value_col.replace('_', ' ').title()} Distribution"
        ax.set_title(chart_title, fontsize=12, fontweight='bold', color=QA_COLORS['text'])
        ax.set_xlabel(value_col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        self.figure.tight_layout()
        self._has_data = True
        self.canvas.draw_idle()

    def plot_bar(self, data: pd.DataFrame,
                 category_col: str = None,
                 value_col: str = None,
                 title: str = None,
                 horizontal: bool = False,
                 color_by_value: bool = True):
        """
        Plot a bar chart.

        Args:
            data: DataFrame with categorical data
            category_col: Column for categories (auto-detects if None)
            value_col: Column for values (auto-detects if None)
            title: Chart title
            horizontal: Whether to use horizontal bars
            color_by_value: Whether to color bars by value (green/yellow/red for percentages)
        """
        if data is None or len(data) == 0:
            self._show_placeholder("No Data", "DataFrame is empty")
            return

        # Auto-detect columns
        if category_col is None:
            for col in ['category', 'model', 'name', 'label', 'month_year']:
                if col in data.columns:
                    category_col = col
                    break
        if category_col is None and len(data.columns) > 0:
            category_col = data.columns[0]

        if value_col is None:
            for col in ['value', 'count', 'rate', 'pass_rate', 'track_status']:
                if col in data.columns:
                    value_col = col
                    break
        if value_col is None:
            for col in data.columns:
                if col != category_col and pd.api.types.is_numeric_dtype(data[col]):
                    value_col = col
                    break

        if category_col is None or value_col is None:
            self._show_error("Column Detection Failed",
                f"Could not find category/value columns.\nAvailable: {list(data.columns)}")
            return

        categories = [str(c) for c in data[category_col]]
        values = data[value_col].astype(float).tolist()

        if len(values) == 0:
            self._show_placeholder("No Data", "No values to plot")
            return

        # Clear and plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._apply_style(ax)

        # Determine colors
        max_val = max(values) if values else 0
        is_percentage = 'rate' in value_col.lower() or (0 <= min(values) and max_val <= 100)

        if color_by_value and is_percentage:
            colors = []
            for v in values:
                if v >= 95:
                    colors.append(QA_COLORS['pass'])
                elif v >= 90:
                    colors.append(QA_COLORS['warning'])
                else:
                    colors.append(QA_COLORS['fail'])
        else:
            colors = [QA_COLORS['info']] * len(values)

        # Plot bars
        if horizontal:
            bars = ax.barh(categories, values, color=colors, alpha=0.8)
            for bar, val in zip(bars, values):
                fmt = f'{val:.1f}%' if is_percentage else f'{val:.1f}'
                ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                       fmt, va='center', fontsize=9, color=QA_COLORS['text'])
        else:
            bars = ax.bar(categories, values, color=colors, alpha=0.8)
            for bar, val in zip(bars, values):
                fmt = f'{val:.1f}%' if is_percentage else f'{val:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.01,
                       fmt, ha='center', va='bottom', fontsize=9, color=QA_COLORS['text'])

        # Formatting
        chart_title = title or self.title or f"{value_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}"
        ax.set_title(chart_title, fontsize=12, fontweight='bold', color=QA_COLORS['text'])

        if horizontal:
            ax.set_xlabel(value_col.replace('_', ' ').title() + (' (%)' if is_percentage else ''))
            ax.set_ylabel(category_col.replace('_', ' ').title())
        else:
            ax.set_ylabel(value_col.replace('_', ' ').title() + (' (%)' if is_percentage else ''))
            ax.set_xlabel(category_col.replace('_', ' ').title())
            if len(categories) > 3:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        self.figure.tight_layout()
        self._has_data = True
        self.canvas.draw_idle()

    def plot_pie(self, data: Union[pd.DataFrame, Dict[str, float], pd.Series],
                 category_col: str = None,
                 value_col: str = None,
                 title: str = None):
        """
        Plot a pie chart.

        Args:
            data: DataFrame, dict, or Series with category/value data
            category_col: Column for categories (if DataFrame)
            value_col: Column for values (if DataFrame)
            title: Chart title
        """
        # Handle different input types
        if isinstance(data, dict):
            labels = list(data.keys())
            values = list(data.values())
        elif isinstance(data, pd.Series):
            labels = list(data.index)
            values = list(data.values)
        elif isinstance(data, pd.DataFrame):
            if category_col is None or value_col is None:
                # Try to auto-detect
                if len(data.columns) >= 2:
                    category_col = data.columns[0]
                    for col in data.columns[1:]:
                        if pd.api.types.is_numeric_dtype(data[col]):
                            value_col = col
                            break
                    if value_col is None:
                        value_col = data.columns[1]
                else:
                    self._show_error("Invalid Data", "Need at least 2 columns for pie chart")
                    return
            labels = [str(c) for c in data[category_col]]
            values = data[value_col].astype(float).tolist()
        else:
            self._show_error("Invalid Data Type", "Expected DataFrame, dict, or Series")
            return

        if len(values) == 0:
            self._show_placeholder("No Data", "No values to plot")
            return

        # Filter out zero/negative values
        filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
        if not filtered:
            self._show_placeholder("No Positive Values", "All values are zero or negative")
            return

        labels, values = zip(*filtered)

        # Clear and plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Color scheme
        colors = [QA_COLORS['pass'], QA_COLORS['fail'], QA_COLORS['warning'],
                 QA_COLORS['info'], QA_COLORS['filtered']]
        pie_colors = [colors[i % len(colors)] for i in range(len(values))]

        wedges, texts, autotexts = ax.pie(values, labels=labels, colors=pie_colors,
                                          autopct='%1.1f%%', startangle=90,
                                          explode=[0.02] * len(values))

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')

        ax.axis('equal')

        chart_title = title or self.title or 'Distribution'
        ax.set_title(chart_title, fontsize=12, fontweight='bold', color=QA_COLORS['text'])

        self.figure.tight_layout()
        self._has_data = True
        self.canvas.draw_idle()

    @property
    def has_data(self) -> bool:
        """Check if chart has data."""
        return self._has_data

    def save(self, filepath: str, dpi: int = 150):
        """Save chart to file."""
        self.figure.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
