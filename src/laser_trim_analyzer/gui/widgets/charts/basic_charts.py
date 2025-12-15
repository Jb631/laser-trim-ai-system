"""
BasicChartMixin - Standard chart types for ChartWidget.

This module provides basic plotting methods:
- update_chart_data: Update chart with DataFrame data
- plot_line: Line charts
- plot_bar: Bar charts
- plot_scatter: Scatter plots
- plot_histogram: Histograms
- plot_box: Box plots
- plot_heatmap: Heatmaps
- plot_multi_series: Multiple series on same chart
- plot_pie: Pie charts
- add_threshold_lines: Add reference lines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import customtkinter as ctk
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from laser_trim_analyzer.gui.theme_helper import ThemeHelper

logger = logging.getLogger(__name__)


class BasicChartMixin:
    """
    Mixin providing basic chart plotting methods.

    Requires ChartWidgetBase as parent class with:
    - self.figure, self.canvas
    - self.qa_colors
    - self._get_or_create_axes()
    - self._apply_theme_to_axes()
    - self.show_placeholder(), self.show_error()
    - self._style_legend()
    """

    def update_chart_data(self, data: pd.DataFrame):
        """
        Update chart with new data based on chart type.

        Args:
            data: DataFrame with columns appropriate for the chart type
        """
        self.show_loading()
        self.after(10, self._process_chart_update, data)

    def _process_chart_update(self, data: pd.DataFrame):
        """Process the actual chart update after showing loading state."""
        self._last_data = data.copy() if data is not None and len(data) > 0 else None

        if data is None or len(data) == 0:
            logger.warning(f"ChartWidget.update_chart_data called with empty data for {self.title}")
            self.show_placeholder("No data available", "Load or analyze data to display chart")
            return

        if isinstance(data, pd.DataFrame):
            logger.debug(f"ChartWidget received data with columns: {data.columns.tolist()}, shape: {data.shape}")
            if all(data[col].isna().all() for col in data.columns):
                logger.warning(f"All columns in data are empty for {self.title}")
                self.show_placeholder("All data values are empty", "Check data processing pipeline")
                return

        self._has_data = True
        logger.info(f"ChartWidget updating {self.chart_type} chart '{self.title}' with {len(data)} rows")

        try:
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
            elif self.chart_type == 'grouped_bar':
                self._plot_bar_from_data(data)
            else:
                logger.warning(f"Unknown chart type '{self.chart_type}', defaulting to line plot")
                self._plot_line_from_data(data)

        except KeyError as e:
            logger.error(f"Missing required column for {self.chart_type} chart: {e}", exc_info=True)
            self.show_error("Data Format Error", f"Missing required column: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid data values: {e}", exc_info=True)
            self.show_error("Invalid Data", f"The data contains invalid values:\n{str(e)}")
        except Exception as e:
            logger.error(f"Error updating chart: {e}", exc_info=True)
            self.show_error("Chart Display Error", f"Unable to display the chart:\n{str(e)}")

    def _set_y_limits_with_padding(self, ax, y_values: Optional[np.ndarray] = None, pad_ratio: float = 0.08):
        """Apply 5-10% y-axis padding for readability."""
        try:
            if y_values is None:
                ymin, ymax = ax.get_ylim()
            else:
                if len(y_values) == 0 or not np.all(np.isfinite(y_values)):
                    return
                ymin = float(np.nanmin(y_values))
                ymax = float(np.nanmax(y_values))
            if not np.isfinite([ymin, ymax]).all():
                return
            if ymax == ymin:
                delta = max(abs(ymin), 1.0) * pad_ratio
                ax.set_ylim(ymin - delta, ymax + delta)
                return
            span = ymax - ymin
            pad = span * pad_ratio
            ax.set_ylim(ymin - pad, ymax + pad)
        except Exception:
            pass

    def _detect_columns(self, data: pd.DataFrame, preferred_x: list = None, preferred_y: list = None):
        """
        Auto-detect appropriate X (date/index) and Y (value) columns.

        Args:
            data: DataFrame to analyze
            preferred_x: List of preferred column names for X axis (in order of preference)
            preferred_y: List of preferred column names for Y axis (in order of preference)

        Returns:
            Tuple of (x_col, y_col) or (None, None) if not found
        """
        if preferred_x is None:
            preferred_x = ['trim_date', 'date', 'Date', 'timestamp', 'time', 'x', 'index']
        if preferred_y is None:
            preferred_y = ['sigma_gradient', 'value', 'y', 'values', 'measurement', 'data']

        x_col = None
        y_col = None

        # Find X column (prefer date-like columns)
        for col in preferred_x:
            if col in data.columns:
                x_col = col
                break

        # If no preferred X column found, look for datetime columns
        if x_col is None:
            for col in data.columns:
                if data[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                    x_col = col
                    break

        # If still no X column, use first column or index
        if x_col is None and len(data.columns) > 0:
            x_col = data.columns[0]

        # Find Y column (prefer numeric columns)
        for col in preferred_y:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                y_col = col
                break

        # If no preferred Y column found, look for any numeric column
        if y_col is None:
            for col in data.columns:
                if col != x_col and pd.api.types.is_numeric_dtype(data[col]):
                    y_col = col
                    break

        return x_col, y_col

    def _plot_line_from_data(self, data: pd.DataFrame):
        """Plot line chart from DataFrame with flexible column detection."""
        if data is None or not isinstance(data, pd.DataFrame):
            self.show_placeholder("Invalid Data", "Expected pandas DataFrame for line chart")
            return

        if len(data) == 0:
            self.show_placeholder("No Data", "DataFrame is empty")
            return

        # Auto-detect columns if specific ones aren't present
        x_col, y_col = self._detect_columns(data)

        # Log what we found for debugging
        logger.debug(f"Line chart: detected x_col={x_col}, y_col={y_col} from columns: {list(data.columns)}")

        if x_col is None or y_col is None:
            self.show_placeholder("Column Detection Failed",
                f"Could not find suitable X/Y columns.\nAvailable: {list(data.columns)}")
            return

        if data[y_col].isna().all():
            self.show_placeholder("Empty Y Column", f"Column '{y_col}' contains no valid data")
            return

        valid_data = data.dropna(subset=[y_col])
        if len(valid_data) < 2:
            self.show_placeholder("Insufficient Data",
                f"Need at least 2 valid data points for line chart. Found: {len(valid_data)}")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._apply_theme_to_axes(ax)

        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]

        # Work with detected columns (use valid_data which already has NaN dropped)
        work_data = valid_data.copy()

        # Check if x_col is datetime-like and convert if needed
        is_datetime = False
        try:
            work_data[x_col] = pd.to_datetime(work_data[x_col], errors='coerce')
            if work_data[x_col].notna().any():
                is_datetime = True
        except (TypeError, ValueError):
            pass

        # Aggregate by date if datetime and multiple points
        if is_datetime and len(work_data) > 1:
            try:
                data_agg = work_data.groupby(work_data[x_col].dt.date).agg({
                    y_col: 'mean'
                }).reset_index()
                data_agg.columns = [x_col, y_col]
                data_agg[x_col] = pd.to_datetime(data_agg[x_col])
                data_sorted = data_agg.sort_values(x_col)
            except Exception:
                data_sorted = work_data.sort_values(x_col) if x_col else work_data
        else:
            data_sorted = work_data.sort_values(x_col) if x_col else work_data

        x_data = data_sorted[x_col]
        y_data = data_sorted[y_col]

        # Create y_label from column name (format nicely)
        y_label = y_col.replace('_', ' ').title() if y_col else 'Value'
        x_label = 'Date' if is_datetime else (x_col.replace('_', ' ').title() if x_col else 'Index')

        # Plot main data
        ax.plot(x_data, y_data, marker='o', linewidth=2, markersize=4,
               color=self.qa_colors['primary'], label=y_label, alpha=0.8)

        # Add trend line for datetime data with enough points
        if is_datetime and len(x_data) >= 3:
            try:
                x_numeric = mdates.date2num(pd.to_datetime(x_data))
                z = np.polyfit(x_numeric, y_data, 1)
                p = np.poly1d(z)
                trend_y = p(x_numeric)

                ax.plot(x_data, trend_y, "--", color=self.qa_colors['warning'],
                       linewidth=2, alpha=0.7, label=f'Trend (slope: {z[0]:.6f})')

                legend = ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True)
                self._style_legend(legend)
            except Exception as e:
                logger.debug(f"Could not add trend line: {e}")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        try:
            self._set_y_limits_with_padding(ax, y_data.to_numpy())
        except Exception:
            pass

        # Format dates if applicable
        if is_datetime and len(x_data) > 0:
            try:
                date_range = (x_data.max() - x_data.min()).days if hasattr(x_data.max() - x_data.min(), 'days') else 0

                if date_range <= 7:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
                elif date_range <= 31:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, date_range // 10)))
                elif date_range <= 365:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                else:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                ax.tick_params(axis='x', pad=5)
            except Exception as e:
                logger.debug(f"Date formatting skipped: {e}")

        if self.title:
            ax.set_title(self.title, color=text_color)
        ax.grid(True, alpha=0.3, color=grid_color)

        self._has_data = True
        self.canvas.draw()
        self.canvas.flush_events()

    def _plot_bar_from_data(self, data: pd.DataFrame):
        """Plot bar chart from DataFrame with flexible column detection."""
        if data is None or not isinstance(data, pd.DataFrame):
            self.show_placeholder("Invalid Data", "Expected pandas DataFrame for bar chart")
            return

        if len(data) == 0:
            self.show_placeholder("No Data", "DataFrame is empty")
            return

        # Auto-detect columns for bar chart
        # For category column, prefer: month_year, category, name, label, x
        # For value column, prefer: track_status, value, count, y, rate
        cat_cols = ['month_year', 'category', 'name', 'label', 'model', 'x']
        val_cols = ['track_status', 'value', 'count', 'rate', 'y', 'pass_rate']

        cat_col = None
        val_col = None

        for col in cat_cols:
            if col in data.columns:
                cat_col = col
                break
        if cat_col is None and len(data.columns) >= 1:
            cat_col = data.columns[0]  # Use first column as category

        for col in val_cols:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                val_col = col
                break
        if val_col is None:
            # Find first numeric column that isn't the category
            for col in data.columns:
                if col != cat_col and pd.api.types.is_numeric_dtype(data[col]):
                    val_col = col
                    break

        logger.debug(f"Bar chart: detected cat_col={cat_col}, val_col={val_col} from columns: {list(data.columns)}")

        if cat_col is None or val_col is None:
            self.show_placeholder("Column Detection Failed",
                f"Could not find suitable category/value columns.\nAvailable: {list(data.columns)}")
            return

        if data[val_col].isna().all():
            self.show_placeholder("Empty Values", f"Column '{val_col}' contains no valid data")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]

        ax.tick_params(colors=text_color, labelcolor=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)

        categories = [str(m) for m in data[cat_col]]
        values = data[val_col].tolist()

        # Determine if this looks like percentage data (0-100 range)
        max_val = max(values) if values else 0
        is_percentage = 'rate' in val_col.lower() or 'percent' in val_col.lower() or (0 <= min(values) and max_val <= 100)

        # Color bars based on values (for percentage data, color by thresholds)
        colors = []
        for val in values:
            if is_percentage:
                if val >= 95:
                    colors.append(self.qa_colors['pass'])
                elif val >= 90:
                    colors.append(self.qa_colors['warning'])
                else:
                    colors.append(self.qa_colors['fail'])
            else:
                colors.append(self.qa_colors['primary'])

        bars = ax.bar(categories, values, color=colors)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            label_fmt = f'{height:.1f}%' if is_percentage else f'{height:.1f}'
            ax.text(bar.get_x() + bar.get_width() / 2., height + max_val * 0.02,
                   label_fmt, ha='center', va='bottom', fontsize=9,
                   color=text_color, fontweight='bold')

        # Use detected column names for labels
        x_label = cat_col.replace('_', ' ').title() if cat_col else 'Category'
        y_label = val_col.replace('_', ' ').title() if val_col else 'Value'
        if is_percentage and '%' not in y_label:
            y_label += ' (%)'

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if is_percentage:
            ax.set_ylim(0, max(105, max_val * 1.1))  # Leave room for labels

        if len(categories) > 3:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        if self.title:
            ax.set_title(self.title, color=text_color)
        ax.grid(True, alpha=0.3, axis='y', color=grid_color)

        self._has_data = True
        self.canvas.draw()

    def _plot_scatter_from_data(self, data: pd.DataFrame):
        """Plot scatter chart from DataFrame with validation."""
        if data is None or not isinstance(data, pd.DataFrame):
            self.show_placeholder("Invalid Data", "Expected pandas DataFrame for scatter chart")
            return

        if len(data) == 0:
            self.show_placeholder("No Data", "DataFrame is empty")
            return

        if 'x' not in data.columns or 'y' not in data.columns:
            self.show_placeholder("Missing Columns",
                f"Scatter chart requires 'x' and 'y' columns.\nFound: {list(data.columns)}")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._apply_theme_to_axes(ax)

        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]

        x_data = data['x']
        y_data = data['y']

        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask]
        y_data = y_data[mask]

        if len(x_data) > 0:
            ax.scatter(x_data, y_data, alpha=0.6, s=50, color=self.qa_colors['primary'],
                      edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Sigma Gradient')
            ax.set_ylabel('Linearity Error')

            if len(x_data) > 5:
                correlation = np.corrcoef(x_data, y_data)[0, 1]
                if not np.isnan(correlation):
                    box_color = '#2b2b2b' if is_dark else 'white'
                    text_color_inv = 'white' if is_dark else 'black'
                    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                           transform=ax.transAxes, fontsize=10, color=text_color_inv,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color,
                                    edgecolor=grid_color, alpha=0.9))

            try:
                self._set_y_limits_with_padding(ax, y_data.to_numpy())
            except Exception:
                pass

        if self.title:
            ax.set_title(self.title, color=text_color)
        ax.grid(True, alpha=0.3, color=grid_color)

        self.canvas.draw()

    def _plot_histogram_from_data(self, data: pd.DataFrame):
        """Plot histogram from DataFrame with validation."""
        if data is None or not isinstance(data, pd.DataFrame):
            self.show_placeholder("Invalid Data", "Expected pandas DataFrame for histogram")
            return

        if len(data) == 0:
            self.show_placeholder("No Data", "DataFrame is empty")
            return

        if 'sigma_gradient' not in data.columns:
            self.show_placeholder("Missing Column",
                f"Histogram requires 'sigma_gradient' column.\nFound: {list(data.columns)}")
            return

        valid_data = data['sigma_gradient'].dropna()
        if len(valid_data) < 5:
            self.show_placeholder("Insufficient Data",
                f"Need at least 5 valid data points for histogram. Found: {len(valid_data)}")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]

        ax.tick_params(colors=text_color, labelcolor=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)

        sigma_data = valid_data
        mean = sigma_data.mean()
        std = sigma_data.std()

        n, bins, patches = ax.hist(sigma_data, bins=min(30, max(10, len(sigma_data) // 5)),
                                  alpha=0.7, edgecolor='black', density=True)

        cm = plt.cm.RdYlGn_r
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = (bin_centers - mean) / (2 * std) if std > 0 else np.zeros_like(bin_centers)
        col = np.clip(col, -1, 1)

        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm((c + 1) / 2))

        if std > 0:
            x = np.linspace(sigma_data.min(), sigma_data.max(), 100)
            from scipy import stats
            ax.plot(x, stats.norm.pdf(x, mean, std), 'k-', linewidth=2,
                   label=f'Normal (u={mean:.3f}, s={std:.3f})')

        ax.axvline(0.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Spec Limits')
        ax.axvline(0.7, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvspan(0.3, 0.7, alpha=0.1, color='green', label='Target Range')
        ax.axvline(mean, color='blue', linestyle='-', linewidth=2, label=f'Mean: {mean:.3f}')

        cpk = min((0.7-mean)/(3*std), (mean-0.3)/(3*std)) if std > 0 else 0
        textstr = f'n = {len(sigma_data)}\nMean = {mean:.3f}\nStd = {std:.3f}\nCpk = {cpk:.2f}'
        props = dict(boxstyle='round', facecolor='wheat' if not is_dark else '#3a3a3a', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props, color=text_color)

        ax.set_xlabel('Sigma Gradient')
        ax.set_ylabel('Probability Density')
        ax.set_xlim(max(0, sigma_data.min() - 0.1), min(1, sigma_data.max() + 0.1))

        legend = ax.legend(loc='upper right')
        if legend:
            self._style_legend(legend)

        if self.title:
            ax.set_title(self.title, color=text_color)
        ax.grid(True, alpha=0.3, axis='y', color=grid_color)

        self.canvas.draw()

    def _plot_heatmap_from_data(self, data: pd.DataFrame):
        """Plot heatmap from DataFrame with validation."""
        if data is None or not isinstance(data, pd.DataFrame):
            self.show_placeholder("Invalid Data", "Expected pandas DataFrame for heatmap")
            return

        if len(data) == 0:
            self.show_placeholder("No Data", "DataFrame is empty")
            return

        required_cols = ['x_values', 'y_values', 'values']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            self.show_placeholder("Missing Columns",
                f"Heatmap requires {required_cols} columns.\nMissing: {missing_cols}")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]

        ax.tick_params(colors=text_color, labelcolor=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)

        try:
            pivot_data = data.pivot(index='y_values', columns='x_values', values='values')

            im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', interpolation='nearest')

            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_xticklabels(pivot_data.columns)
            ax.set_yticklabels(pivot_data.index)

            self.figure.colorbar(im, ax=ax)

            if pivot_data.size <= 100:
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        value = pivot_data.values[i, j]
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                               color='black' if 0.3 < im.norm(value) < 0.7 else 'white')

        except Exception as e:
            ax.text(0.5, 0.5, f"Error creating heatmap: {str(e)}",
                   ha='center', va='center', transform=ax.transAxes)

        if self.title:
            ax.set_title(self.title, color=text_color)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        self.canvas.draw()

    def clear(self):
        """Clear the chart (alias for clear_chart for compatibility)."""
        self.clear_chart()

    def plot_line(self, x_data: List, y_data: List, label: str = "",
                  color: Optional[str] = None, marker: Optional[str] = None,
                  xlabel: str = "", ylabel: str = "", **kwargs):
        """Plot line chart with data validation."""
        if x_data is None or y_data is None:
            self.show_placeholder("No Data", "Cannot plot without X and Y data")
            return None

        if not x_data or not y_data:
            self.show_placeholder("Empty Data", "X and Y data arrays are empty")
            return None

        if len(x_data) != len(y_data):
            self.show_error("Data Mismatch",
                f"X data has {len(x_data)} points but Y data has {len(y_data)} points.")
            return None

        if len(x_data) < 2:
            self.show_placeholder("Insufficient Data", "Need at least 2 data points to plot a line")
            return None

        try:
            ax = self._get_or_create_axes()
            self._apply_theme_to_axes(ax)
            self._has_data = True

            if color and color in self.qa_colors:
                color = self.qa_colors[color]

            line = ax.plot(x_data, y_data, label=label, color=color, marker=marker, **kwargs)[0]

            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]
            grid_color = theme_colors["border"]["primary"]

            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if self.title:
                ax.set_title(self.title, color=text_color)
            if label:
                ax.legend()

            ax.grid(True, alpha=0.3, color=grid_color)

            try:
                y_array = np.array(y_data)
                if len(y_array) > 0 and np.all(np.isfinite(y_array)):
                    self._set_y_limits_with_padding(ax, y_array)
            except (ValueError, TypeError) as e:
                self.logger.debug(f"Y-axis padding skipped: {e}")

            self.canvas.draw_idle()
            return line

        except Exception as e:
            self.show_error("Line Plot Error", f"Failed to create line chart: {str(e)}")
            return None

    def plot_bar(self, categories: List[str], values: List[float],
                 colors: Optional[List[str]] = None, xlabel: str = "",
                 ylabel: str = "", **kwargs):
        """Plot bar chart with data validation."""
        if categories is None or values is None:
            self.show_placeholder("No Data", "Cannot plot without categories and values")
            return None

        if not categories or not values:
            self.show_placeholder("Empty Data", "Categories and values arrays are empty")
            return None

        if len(categories) != len(values):
            self.show_error("Data Mismatch",
                f"Categories has {len(categories)} items but values has {len(values)} items.")
            return None

        try:
            ax = self._get_or_create_axes()
            self._apply_theme_to_axes(ax)
            self._has_data = True

            if colors:
                colors = [self.qa_colors.get(c, c) for c in colors]
            else:
                colors = self.qa_colors['primary']

            bars = ax.bar(categories, values, color=colors, **kwargs)

            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]
            grid_color = theme_colors["border"]["primary"]

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9, color=text_color)

            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if self.title:
                ax.set_title(self.title, color=text_color)

            if len(categories) > 10:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            ax.grid(True, alpha=0.3, axis='y', color=grid_color)
            self.canvas.draw_idle()

            return bars

        except Exception as e:
            self.show_error("Bar Chart Error", f"Failed to create bar chart: {str(e)}")
            return None

    def plot_scatter(self, x_data: List, y_data: List,
                     colors: Optional[List] = None, sizes: Optional[List] = None,
                     labels: Optional[List[str]] = None, xlabel: str = "",
                     ylabel: str = "", alpha: float = 0.6, **kwargs):
        """Plot scatter chart with data validation."""
        if x_data is None or y_data is None:
            self.show_placeholder("No Data", "Cannot plot without X and Y data")
            return None

        if not x_data or not y_data:
            self.show_placeholder("Empty Data", "X and Y data arrays are empty")
            return None

        if len(x_data) != len(y_data):
            self.show_error("Data Mismatch",
                f"X data has {len(x_data)} points but Y data has {len(y_data)} points.")
            return None

        try:
            ax = self._get_or_create_axes()
            self._apply_theme_to_axes(ax)
            self._has_data = True

            if sizes is None:
                sizes = 50

            if colors and isinstance(colors, (list, tuple)) and len(colors) > 0 and isinstance(colors[0], str):
                colors = [self.qa_colors.get(c, c) for c in colors]

            kwargs.pop('alpha', None)
            scatter = ax.scatter(x_data, y_data, c=colors, s=sizes, alpha=alpha, **kwargs)

            if labels:
                for i, label in enumerate(labels):
                    ax.annotate(label, (x_data[i], y_data[i]), xytext=(5, 5),
                               textcoords='offset points', fontsize=8, alpha=0.7)

            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]
            grid_color = theme_colors["border"]["primary"]

            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if self.title:
                ax.set_title(self.title, color=text_color)

            ax.grid(True, alpha=0.3, color=grid_color)

            try:
                y_array = np.array(y_data)
                if len(y_array) > 0 and np.all(np.isfinite(y_array)):
                    self._set_y_limits_with_padding(ax, y_array)
            except:
                pass

            self.canvas.draw_idle()
            return scatter

        except Exception as e:
            self.show_error("Scatter Plot Error", f"Failed to create scatter chart: {str(e)}")
            return None

    def plot_histogram(self, data: List[float], bins: int = 20,
                       color: Optional[str] = None, xlabel: str = "",
                       ylabel: str = "Frequency", **kwargs):
        """Plot histogram with data validation."""
        if data is None:
            self.show_placeholder("No Data", "Cannot plot histogram without data")
            return None

        if not data:
            self.show_placeholder("Empty Data", "Data array is empty")
            return None

        if len(data) < 2:
            self.show_placeholder("Insufficient Data", "Need at least 2 data points for a histogram")
            return None

        try:
            ax = self._get_or_create_axes()
            self._apply_theme_to_axes(ax)
            self._has_data = True

            if color and color in self.qa_colors:
                color = self.qa_colors[color]
            else:
                color = self.qa_colors['primary']

            n, bins_edges, patches = ax.hist(data, bins=bins, color=color,
                                       alpha=0.7, edgecolor='black', **kwargs)

            mean = np.mean(data)
            std = np.std(data)
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
            ax.axvline(mean + std, color='orange', linestyle='--', linewidth=1, label=f'+/-1s: {std:.3f}')
            ax.axvline(mean - std, color='orange', linestyle='--', linewidth=1)

            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]
            grid_color = theme_colors["border"]["primary"]

            if xlabel:
                ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if self.title:
                ax.set_title(self.title, color=text_color)

            ax.legend()
            ax.grid(True, alpha=0.3, axis='y', color=grid_color)

            self.canvas.draw_idle()
            return n, bins_edges, patches

        except Exception as e:
            self.show_error("Histogram Error", f"Failed to create histogram: {str(e)}")
            return None

    def plot_box(self, data: List[List[float]], labels: List[str] = None,
                 xlabel: str = "", ylabel: str = "", **kwargs):
        """Plot box plot with data validation."""
        if data is None:
            self.show_placeholder("No Data", "Cannot plot boxplot without data")
            return None

        if not data:
            self.show_placeholder("Empty Data", "Data array is empty")
            return None

        try:
            ax = self._get_or_create_axes()
            self._apply_theme_to_axes(ax)
            self._has_data = True

            bp = ax.boxplot(data, labels=labels, patch_artist=True, **kwargs)

            colors = [self.qa_colors['primary'], self.qa_colors['secondary'],
                      self.qa_colors['warning'], self.qa_colors['pass']]

            for patch, color in zip(bp['boxes'], colors * (len(bp['boxes']) // len(colors) + 1)):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]
            grid_color = theme_colors["border"]["primary"]

            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if self.title:
                ax.set_title(self.title, color=text_color)

            ax.grid(True, alpha=0.3, axis='y', color=grid_color)
            self.canvas.draw_idle()

            return bp

        except Exception as e:
            self.show_error("Box Plot Error", f"Failed to create box plot: {str(e)}")
            return None

    def plot_heatmap(self, data: np.ndarray, xlabels: List[str],
                     ylabels: List[str], cmap: str = 'RdYlGn',
                     xlabel: str = "", ylabel: str = "", **kwargs):
        """Plot heatmap with data validation."""
        if data is None:
            self.show_placeholder("No Data", "Cannot plot heatmap without data")
            return None

        if xlabels is None or ylabels is None:
            self.show_placeholder("No Labels", "Cannot plot heatmap without axis labels")
            return None

        if not hasattr(data, 'shape') or len(data.shape) != 2:
            self.show_error("Invalid Data", "Heatmap requires 2D data array (matrix)")
            return None

        try:
            ax = self._get_or_create_axes()
            self._apply_theme_to_axes(ax)
            self._has_data = True

            im = ax.imshow(data, cmap=cmap, aspect='auto', **kwargs)

            ax.set_xticks(np.arange(len(xlabels)))
            ax.set_yticks(np.arange(len(ylabels)))
            ax.set_xticklabels(xlabels)
            ax.set_yticklabels(ylabels)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            self.figure.colorbar(im, ax=ax)

            for i in range(len(ylabels)):
                for j in range(len(xlabels)):
                    value = data[i, j]
                    text_color = 'black' if 0.3 < value < 0.7 else 'white'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=text_color, fontsize=9)

            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]

            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if self.title:
                ax.set_title(self.title, color=text_color)

            self.canvas.draw_idle()
            return im

        except Exception as e:
            self.show_error("Heatmap Error", f"Failed to create heatmap: {str(e)}")
            return None

    def plot_multi_series(self, data_dict: Dict[str, Dict[str, List]],
                          xlabel: str = "", ylabel: str = ""):
        """Plot multiple series on the same chart."""
        if data_dict is None:
            self.show_placeholder("No Data", "Cannot plot without data dictionary")
            return None

        if not data_dict:
            self.show_placeholder("Empty Data", "Data dictionary is empty")
            return None

        try:
            ax = self._get_or_create_axes()
            self._apply_theme_to_axes(ax)
            self._has_data = True

            for series_name, series_data in data_dict.items():
                x_data = series_data['x']
                y_data = series_data['y']
                color = series_data.get('color')

                if color and color in self.qa_colors:
                    color = self.qa_colors[color]

                ax.plot(x_data, y_data, label=series_name, color=color, marker='o', markersize=4)

            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]
            grid_color = theme_colors["border"]["primary"]

            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if self.title:
                ax.set_title(self.title, color=text_color)

            ax.legend()
            ax.grid(True, alpha=0.3, color=grid_color)

            self.canvas.draw_idle()
            return True

        except Exception as e:
            self.show_error("Multi-Series Plot Error", f"Failed to create multi-series chart: {str(e)}")
            return None

    def plot_pie(self, values: List[float], labels: List[str],
                 colors: List[str] = None, explode: List[float] = None):
        """Plot a pie chart with data validation."""
        if values is None or labels is None:
            self.show_placeholder("No Data", "Cannot plot pie chart without values and labels")
            return None

        if not values or not labels:
            self.show_placeholder("Empty Data", "Values and labels arrays are empty")
            return None

        if len(values) != len(labels):
            self.show_error("Data Mismatch",
                f"Values has {len(values)} items but labels has {len(labels)} items.")
            return None

        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            self._has_data = True

            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]

            if colors is None:
                colors = [self.qa_colors['pass'], self.qa_colors['warning'],
                         self.qa_colors['fail'], self.qa_colors['primary']]

            wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors,
                                             explode=explode, autopct='%1.1f%%',
                                             shadow=True, startangle=90)

            for text in texts:
                text.set_color(text_color)
                text.set_fontsize(10)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(9)
                autotext.set_weight('bold')

            ax.axis('equal')

            if self.title:
                ax.set_title(self.title, color=text_color, pad=20)

            self._update_figure_theme()
            self.canvas.draw_idle()

            return wedges, texts, autotexts

        except Exception as e:
            self.show_error("Pie Chart Error", f"Failed to create pie chart: {str(e)}")
            return None

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

        ax.legend()
        self.canvas.draw_idle()
