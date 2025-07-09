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
import logging
from laser_trim_analyzer.gui.theme_helper import ThemeHelper

logger = logging.getLogger(__name__)


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
        self._has_data = False
        self._placeholder_message = "No data to display"
        self._placeholder_instruction = "Complete an analysis or select data to view chart"

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
        
        # Show initial placeholder
        self.show_placeholder()
        
    def _style_legend(self, legend):
        """Apply theme-appropriate styling to legend."""
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        
        if is_dark:
            legend.get_frame().set_facecolor('#2b2b2b')
            legend.get_frame().set_edgecolor('#555555')
            legend.get_frame().set_alpha(0.9)
            # Set text color to white for dark theme
            for text in legend.get_texts():
                text.set_color('white')
        else:
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('#cccccc')
            legend.get_frame().set_alpha(0.9)
            # Set text color to black for light theme
            for text in legend.get_texts():
                text.set_color('black')
    
    def _apply_theme_to_axes(self, ax):
        """Apply theme colors to matplotlib axes."""
        # Get current theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        
        # Set background based on theme
        if is_dark:
            ax.set_facecolor(theme_colors["bg"]["secondary"])  # Dark background
            text_color = theme_colors["fg"]["primary"]  # White text
            grid_color = theme_colors["border"]["primary"]  # Subtle grid
        else:
            ax.set_facecolor(theme_colors["bg"]["primary"])  # White background
            text_color = theme_colors["fg"]["primary"]  # Black text
            grid_color = theme_colors["border"]["primary"]  # Light gray grid
            
        # Apply text colors
        ax.tick_params(colors=text_color, labelcolor=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        if hasattr(ax, 'title'):
            ax.title.set_color(text_color)
            
        # Set grid style
        ax.grid(True, alpha=0.3, color=grid_color)
        
    def _update_figure_theme(self):
        """Update figure background based on current theme."""
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        
        if is_dark:
            self.figure.patch.set_facecolor(theme_colors["bg"]["primary"])
        else:
            self.figure.patch.set_facecolor(theme_colors["bg"]["secondary"])
            
    def _get_or_create_axes(self):
        """Get existing axes or create new one with theme applied."""
        if not self.figure.axes:
            ax = self.figure.add_subplot(111)
            self._apply_theme_to_axes(ax)
        else:
            ax = self.figure.axes[0]
        return ax

    def _setup_ui(self):
        """Set up the chart widget UI."""
        # Create matplotlib figure with theme-aware colors
        self.figure = Figure(figsize=self.figsize, dpi=100)
        
        # Apply theme-aware background
        self._update_figure_theme()

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
        self.type_combo = ctk.CTkComboBox(toolbar_frame,
                                   values=['line', 'bar', 'scatter', 'histogram', 'heatmap'],
                                   width=120,
                                   command=self._change_chart_type)
        self.type_combo.set(self.chart_type)
        self.type_combo.pack(side='left', padx=2)

    def update_chart_data(self, data: pd.DataFrame):
        """
        Update chart with new data based on chart type.
        
        Args:
            data: DataFrame with columns appropriate for the chart type
        """
        # Store data for chart type changes
        self._last_data = data.copy() if data is not None and len(data) > 0 else None
        
        if data is None or len(data) == 0:
            logger.warning(f"ChartWidget.update_chart_data called with empty data for {self.title}")
            self.show_placeholder("No data available", "Load or analyze data to display chart")
            return
            
        self._has_data = True
        logger.info(f"ChartWidget updating {self.chart_type} chart '{self.title}' with {len(data)} rows")
        logger.debug(f"Data columns: {data.columns.tolist()}")
            
        try:
            # Don't call clear_chart here since each plot method clears the figure
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
                # Treat grouped_bar as regular bar for now
                self._plot_bar_from_data(data)
            else:
                # Default to line plot
                logger.warning(f"Unknown chart type '{self.chart_type}', defaulting to line plot")
                self._plot_line_from_data(data)
                
        except Exception as e:
            logger.error(f"Error updating chart: {e}", exc_info=True)
            self.clear_chart()
            # Show error message on the chart
            ax = self.figure.gca()
            ax.text(0.5, 0.5, f'Error displaying chart:\n{str(e)}', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=12,
                    color='red',
                    wrap=True)
            self.canvas.draw_idle()
            
    def _plot_line_from_data(self, data: pd.DataFrame):
        """Plot line chart from DataFrame."""
        # Clear the figure first to prevent overlapping plots
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Apply theme to axes
        self._apply_theme_to_axes(ax)
        
        # Get theme colors for use in this method
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]
        
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
                
                # Add legend with better contrast
                legend = ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True)
                self._style_legend(legend)
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Sigma Gradient')
            
            # Format x-axis dates properly
            if len(x_data) > 0:
                # Determine appropriate date format based on date range
                date_range = (x_data.max() - x_data.min()).days if hasattr(x_data.max() - x_data.min(), 'days') else 0
                
                if date_range <= 7:
                    # Less than a week - show day and time
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
                elif date_range <= 31:
                    # Less than a month - show day/month
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, date_range // 10)))
                elif date_range <= 365:
                    # Less than a year - show month/day
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                else:
                    # More than a year - show year/month
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                
                # Rotate labels for better readability
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
        if self.title:
            ax.set_title(self.title, color=text_color)
        ax.grid(True, alpha=0.3, color=grid_color)
        
        # Use tight_layout with padding to prevent label cutoff
        try:
            self.figure.tight_layout(pad=1.5)
        except Exception as e:
            logger.debug(f"tight_layout warning: {e}")
            # Fallback to subplots_adjust if tight_layout fails
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        self.canvas.draw()
        
    def _plot_bar_from_data(self, data: pd.DataFrame):
        """Plot bar chart from DataFrame."""
        # Clear the figure first to prevent overlapping plots
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Use white background for better visibility
        # Apply theme background
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])
        # Apply theme colors to text and spines
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]
        
        ax.tick_params(colors=text_color, labelcolor=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        
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
            
            # Add value labels on bars with proper color
            for bar in bars:
                height = bar.get_height()
                # Use contrasting color for text - white on dark bars, dark on light bars
                bar_color = bar.get_facecolor()
                # Calculate luminance to determine if we need light or dark text
                r, g, b, _ = bar_color
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                label_color = 'white' if luminance < 0.5 else 'black'
                
                ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9,
                       color=text_color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white' if is_dark else 'black', 
                                alpha=0.7, edgecolor='none'))
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Pass Rate (%)')
            ax.set_ylim(0, 100)
            
            if len(categories) > 5:
                ax.tick_params(axis='x', rotation=45)
                
        if self.title:
            ax.set_title(self.title, color=text_color)
        ax.grid(True, alpha=0.3, axis='y', color=grid_color)
        
        # Use tight_layout with padding to prevent label cutoff
        try:
            self.figure.tight_layout(pad=1.5)
        except Exception as e:
            logger.debug(f"tight_layout warning: {e}")
            # Fallback to subplots_adjust if tight_layout fails
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        self.canvas.draw()
        
    def _plot_scatter_from_data(self, data: pd.DataFrame):
        """Plot scatter chart from DataFrame."""
        # Clear the figure first to prevent overlapping plots
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Use white background for better visibility
        # Apply theme to axes
        self._apply_theme_to_axes(ax)
        
        # Get theme colors for use in this method
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]
        
        if 'x' in data.columns and 'y' in data.columns:
            x_data = data['x']
            y_data = data['y']
            
            # Filter out any NaN values
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_data = x_data[mask]
            y_data = y_data[mask]
            
            if len(x_data) > 0:
                ax.scatter(x_data, y_data, alpha=0.6, s=50, color=self.qa_colors['primary'], edgecolors='black', linewidth=0.5)
                ax.set_xlabel('Sigma Gradient')
                ax.set_ylabel('Linearity Error')
                
                # Add correlation if enough points
                if len(x_data) > 5:
                    correlation = np.corrcoef(x_data, y_data)[0, 1]
                    if not np.isnan(correlation):
                        # Better contrast for text box
                        box_color = '#2b2b2b' if is_dark else 'white'
                        text_color_inv = 'white' if is_dark else 'black'
                        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                               transform=ax.transAxes, fontsize=10, color=text_color_inv,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, edgecolor=grid_color, alpha=0.9))
                else:
                    ax.text(0.5, 0.5, 'Insufficient data for correlation\n(minimum 5 points required)', 
                           transform=ax.transAxes, fontsize=12, ha='center', va='center',
                           color=text_color, alpha=0.7)
                
        if self.title:
            ax.set_title(self.title, color=text_color)
        ax.grid(True, alpha=0.3, color=grid_color)
        
        # Use tight_layout with padding to prevent label cutoff
        try:
            self.figure.tight_layout(pad=1.5)
        except Exception as e:
            logger.debug(f"tight_layout warning: {e}")
            # Fallback to subplots_adjust if tight_layout fails
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        self.canvas.draw()
        
    def _plot_histogram_from_data(self, data: pd.DataFrame):
        """Plot histogram from DataFrame."""
        # Clear the figure first to prevent overlapping plots
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Use white background for better visibility
        # Apply theme background
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])
        # Apply theme colors to text and spines
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]
        
        ax.tick_params(colors=text_color, labelcolor=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        
        if 'sigma_gradient' in data.columns:
            sigma_data = data['sigma_gradient'].dropna()
            
            if len(sigma_data) > 0:
                # Calculate statistics
                mean = sigma_data.mean()
                std = sigma_data.std()
                
                # Create histogram with better bins
                n, bins, patches = ax.hist(sigma_data, bins=min(30, max(10, len(sigma_data) // 5)), 
                                          alpha=0.7, edgecolor='black', density=True)
                
                # Color bars based on distance from mean (gradient effect)
                cm = plt.cm.RdYlGn_r  # Red-Yellow-Green reversed
                bin_centers = 0.5 * (bins[:-1] + bins[1:])  # compute bin centers
                col = (bin_centers - mean) / (2 * std)  # normalize to standard deviations
                col = np.clip(col, -1, 1)  # clip to [-1, 1]
                
                for c, p in zip(col, patches):
                    plt.setp(p, 'facecolor', cm((c + 1) / 2))  # map to [0, 1] for colormap
                
                # Add normal distribution overlay
                if std > 0:
                    x = np.linspace(sigma_data.min(), sigma_data.max(), 100)
                    from scipy import stats
                    ax.plot(x, stats.norm.pdf(x, mean, std), 'k-', linewidth=2, 
                           label=f'Normal (μ={mean:.3f}, σ={std:.3f})')
                
                # Add specification limits (0.3 to 0.7 as typical range)
                ax.axvline(0.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Spec Limits')
                ax.axvline(0.7, color='red', linestyle='--', linewidth=2, alpha=0.7)
                ax.axvspan(0.3, 0.7, alpha=0.1, color='green', label='Target Range')
                
                # Add mean line
                ax.axvline(mean, color='blue', linestyle='-', linewidth=2, label=f'Mean: {mean:.3f}')
                
                # Add text box with statistics
                textstr = f'n = {len(sigma_data)}\nMean = {mean:.3f}\nStd = {std:.3f}\nCpk = {min((0.7-mean)/(3*std), (mean-0.3)/(3*std)):.2f}' if std > 0 else f'n = {len(sigma_data)}\nMean = {mean:.3f}'
                props = dict(boxstyle='round', facecolor='wheat' if not is_dark else '#3a3a3a', alpha=0.8)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props, color=text_color)
                
                ax.set_xlabel('Sigma Gradient')
                ax.set_ylabel('Probability Density')
                ax.set_xlim(max(0, sigma_data.min() - 0.1), min(1, sigma_data.max() + 0.1))
                
                # Style legend
                legend = ax.legend(loc='upper right')
                if legend:
                    self._style_legend(legend)
                
        if self.title:
            ax.set_title(self.title, color=text_color)
        ax.grid(True, alpha=0.3, axis='y', color=grid_color)
        
        # Use tight_layout with padding to prevent label cutoff
        try:
            self.figure.tight_layout(pad=1.5)
        except Exception as e:
            logger.debug(f"tight_layout warning: {e}")
            # Fallback to subplots_adjust if tight_layout fails
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        self.canvas.draw()
        
    def _plot_heatmap_from_data(self, data: pd.DataFrame):
        """Plot heatmap from DataFrame."""
        # Clear the figure first to prevent overlapping plots
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Use white background for better visibility
        # Apply theme background
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])
        # Apply theme colors to text and spines
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]
        
        ax.tick_params(colors=text_color, labelcolor=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        
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
                    try:
                        matrix = json.loads(matrix)
                    except (json.JSONDecodeError, ValueError):
                        # If parsing fails, create empty matrix
                        matrix = [[]]
                
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
            ax.set_title(self.title, color=text_color)
            
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Use tight_layout with padding to prevent label cutoff
        try:
            self.figure.tight_layout(pad=1.5)
        except Exception as e:
            logger.debug(f"tight_layout warning: {e}")
            # Fallback to subplots_adjust if tight_layout fails
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
        
        self.canvas.draw()

    def clear(self):
        """Clear the chart (alias for clear_chart for compatibility)."""
        self.clear_chart()

    def plot_line(self, x_data: List, y_data: List, label: str = "",
                  color: Optional[str] = None, marker: Optional[str] = None,
                  xlabel: str = "", ylabel: str = "", **kwargs):
        """Plot line chart."""
        ax = self._get_or_create_axes()
        self._apply_theme_to_axes(ax)
        self._has_data = True  # Mark that we have data

        # Use QA color if specified
        if color and color in self.qa_colors:
            color = self.qa_colors[color]

        # Plot data
        line = ax.plot(x_data, y_data, label=label, color=color,
                       marker=marker, **kwargs)[0]

        # Get theme colors for labels
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]

        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if self.title:
            ax.set_title(self.title, color=text_color)

        # Show legend if labels exist
        if label:
            ax.legend()

        # Grid
        ax.grid(True, alpha=0.3, color=grid_color)

        # Refresh canvas with idle callback to prevent threading issues
        self.canvas.draw_idle()

        return line

    def plot_bar(self, categories: List[str], values: List[float],
                 colors: Optional[List[str]] = None, xlabel: str = "",
                 ylabel: str = "", **kwargs):
        """Plot bar chart."""
        ax = self._get_or_create_axes()
        self._apply_theme_to_axes(ax)
        self._has_data = True  # Mark that we have data

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
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]
        
        if self.title:
            ax.set_title(self.title, color=text_color)

        # Rotate x labels if many categories
        if len(categories) > 10:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Grid
        ax.grid(True, alpha=0.3, axis='y', color=grid_color)

        # Use tight_layout with padding to prevent label cutoff
        try:
            self.figure.tight_layout(pad=1.5)
        except Exception as e:
            logger.debug(f"tight_layout warning: {e}")
            # Fallback to subplots_adjust if tight_layout fails
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

        # Refresh canvas with idle callback to prevent threading issues
        self.canvas.draw_idle()

        return bars

    def plot_scatter(self, x_data: List, y_data: List,
                     colors: Optional[List] = None, sizes: Optional[List] = None,
                     labels: Optional[List[str]] = None, xlabel: str = "",
                     ylabel: str = "", alpha: float = 0.6, **kwargs):
        """Plot scatter chart."""
        ax = self._get_or_create_axes()
        self._apply_theme_to_axes(ax)
        self._has_data = True  # Mark that we have data

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
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]
        
        if self.title:
            ax.set_title(self.title, color=text_color)

        # Grid
        ax.grid(True, alpha=0.3, color=grid_color)

        # Refresh canvas with idle callback to prevent threading issues
        self.canvas.draw_idle()

        return scatter

    def plot_histogram(self, data: List[float], bins: int = 20,
                       color: Optional[str] = None, xlabel: str = "",
                       ylabel: str = "Frequency", **kwargs):
        """Plot histogram."""
        ax = self._get_or_create_axes()
        self._apply_theme_to_axes(ax)
        self._has_data = True  # Mark that we have data

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
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]
        
        if self.title:
            ax.set_title(self.title, color=text_color)

        # Legend
        ax.legend()

        # Grid
        ax.grid(True, alpha=0.3, axis='y', color=grid_color)

        # Refresh canvas with idle callback to prevent threading issues
        self.canvas.draw_idle()

        return n, bins, patches

    def plot_box(self, data: List[List[float]], labels: List[str] = None,
                 xlabel: str = "", ylabel: str = "", **kwargs):
        """Plot box plot."""
        ax = self._get_or_create_axes()
        self._apply_theme_to_axes(ax)
        self._has_data = True  # Mark that we have data

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
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]
        
        if self.title:
            ax.set_title(self.title, color=text_color)

        # Grid
        ax.grid(True, alpha=0.3, axis='y', color=grid_color)

        # Refresh canvas with idle callback to prevent threading issues
        self.canvas.draw_idle()

        return bp

    def plot_heatmap(self, data: np.ndarray, xlabels: List[str],
                     ylabels: List[str], cmap: str = 'RdYlGn',
                     xlabel: str = "", ylabel: str = "", **kwargs):
        """Plot heatmap."""
        ax = self._get_or_create_axes()
        self._apply_theme_to_axes(ax)
        self._has_data = True  # Mark that we have data

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
                value = data[i, j]
                # Choose text color based on background
                text_color = 'black' if 0.3 < value < 0.7 else 'white'
                text = ax.text(j, i, f'{value:.2f}',
                               ha='center', va='center', color=text_color,
                               fontsize=9)

        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        
        if self.title:
            ax.set_title(self.title, color=text_color)

        # Use tight_layout with padding to prevent label cutoff
        try:
            self.figure.tight_layout(pad=1.5)
        except Exception as e:
            logger.debug(f"tight_layout warning: {e}")
            # Fallback to subplots_adjust if tight_layout fails
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

        # Refresh canvas with idle callback to prevent threading issues
        self.canvas.draw_idle()

        return im

    def plot_multi_series(self, data_dict: Dict[str, Dict[str, List]],
                          xlabel: str = "", ylabel: str = ""):
        """
        Plot multiple series on the same chart.

        Args:
            data_dict: Dictionary with series names as keys and
                      {'x': x_data, 'y': y_data, 'color': color} as values
        """
        ax = self._get_or_create_axes()
        self._apply_theme_to_axes(ax)
        self._has_data = True  # Mark that we have data

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
        
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        grid_color = theme_colors["border"]["primary"]
        
        if self.title:
            ax.set_title(self.title, color=text_color)

        # Legend
        ax.legend()

        # Grid
        ax.grid(True, alpha=0.3, color=grid_color)

        # Refresh canvas with idle callback to prevent threading issues
        self.canvas.draw_idle()

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

        # Refresh canvas with idle callback to prevent threading issues
        self.canvas.draw_idle()

    def clear_chart(self):
        """Clear the current chart."""
        self.show_placeholder(self._placeholder_message, self._placeholder_instruction)
        
    def show_placeholder(self, message: str = "No data to display", instruction: str = ""):
        """Show a placeholder with custom message."""
        self._has_data = False
        self.figure.clear()
        # Reset figure background color
        # Update figure theme
        self._update_figure_theme()
        
        # Add empty axes to show clean background
        ax = self.figure.add_subplot(111)
        # Apply theme background
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])
        
        # Create centered text with icon
        text_parts = ['[Chart]']  # Chart icon placeholder (emoji not supported in Arial)
        text_parts.append('')
        text_parts.append(message)
        if instruction:
            text_parts.append('')
            text_parts.append(instruction)
            
        # Use theme-aware text color
        text_color = theme_colors["fg"]["tertiary"]
        
        ax.text(0.5, 0.5, '\n'.join(text_parts), 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=14,
                color=text_color,
                alpha=0.8,
                linespacing=2)
        
        # Remove axes spines and ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()
    
    def refresh_theme(self):
        """Refresh chart colors when theme changes."""
        # Update figure background
        self._update_figure_theme()
        
        # If we have data, redraw the chart with new theme
        if self._has_data and hasattr(self, '_last_data') and self._last_data is not None:
            self.update_chart_data(self._last_data)
        else:
            # Otherwise refresh placeholder
            self.show_placeholder(self._placeholder_message, self._placeholder_instruction)
        
        # Update all axes if they exist
        for ax in self.figure.axes:
            self._apply_theme_to_axes(ax)
        
        self.canvas.draw()

    def _change_chart_type(self, new_value=None):
        """Change chart type and refresh if data exists."""
        if new_value:
            self.chart_type = new_value
        else:
            self.chart_type = self.type_combo.get()
        
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
