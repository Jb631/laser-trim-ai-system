"""
ChartWidget for QA Dashboard

A versatile matplotlib wrapper widget with zoom, pan, and export functionality.
Supports multiple chart types for QA data visualization.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk

# Configure matplotlib before importing pyplot
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better integration

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
                 title: str = "", figsize: Tuple[int, int] = (10, 6),
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
            
        # Configure matplotlib for better theme integration
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        
        # Set matplotlib parameters for theme
        if is_dark:
            plt.rcParams['figure.facecolor'] = theme_colors["bg"]["primary"]
            plt.rcParams['axes.facecolor'] = theme_colors["bg"]["secondary"]
            plt.rcParams['savefig.facecolor'] = theme_colors["bg"]["primary"]
            plt.rcParams['axes.edgecolor'] = theme_colors["border"]["primary"]
            plt.rcParams['text.color'] = theme_colors["fg"]["primary"]
            plt.rcParams['axes.labelcolor'] = theme_colors["fg"]["primary"]
            plt.rcParams['xtick.color'] = theme_colors["fg"]["primary"]
            plt.rcParams['ytick.color'] = theme_colors["fg"]["primary"]
            plt.rcParams['grid.color'] = theme_colors["border"]["primary"]
            plt.rcParams['legend.facecolor'] = theme_colors["bg"]["secondary"]
            plt.rcParams['legend.edgecolor'] = theme_colors["border"]["primary"]
        else:
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['savefig.facecolor'] = 'white'
            plt.rcParams['axes.edgecolor'] = theme_colors["border"]["primary"]
            plt.rcParams['text.color'] = theme_colors["fg"]["primary"]
            plt.rcParams['axes.labelcolor'] = theme_colors["fg"]["primary"]
            plt.rcParams['xtick.color'] = theme_colors["fg"]["primary"]
            plt.rcParams['ytick.color'] = theme_colors["fg"]["primary"]
            plt.rcParams['grid.color'] = theme_colors["border"]["primary"]
            plt.rcParams['legend.facecolor'] = 'white'
            plt.rcParams['legend.edgecolor'] = theme_colors["border"]["primary"]

        # Colors for QA metrics
        self.qa_colors = {
            'pass': '#27ae60',
            'fail': '#e74c3c',
            'warning': '#f39c12',
            'primary': '#3498db',
            'secondary': '#9b59b6',
            'good': '#27ae60',  # Same as pass
            'bad': '#e74c3c',   # Same as fail
            'neutral': '#95a5a6'
        }

        self._setup_ui()
        
        # Show initial placeholder
        self.show_placeholder()
        
    def _style_legend(self, legend):
        """Apply theme-appropriate styling to legend with enhanced readability."""
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        
        # Use theme colors instead of hardcoded values
        if is_dark:
            legend.get_frame().set_facecolor(theme_colors["bg"]["secondary"])
            legend.get_frame().set_edgecolor(theme_colors["border"]["primary"])
            legend.get_frame().set_alpha(0.95)
            text_color = theme_colors["fg"]["primary"]
        else:
            legend.get_frame().set_facecolor(theme_colors["bg"]["primary"])
            legend.get_frame().set_edgecolor(theme_colors["border"]["primary"])
            legend.get_frame().set_alpha(0.95)
            text_color = theme_colors["fg"]["primary"]
            
        # Set consistent font size for readability
        font_size = 8 if self.figsize[0] < 8 else 9
        
        # Apply to all text elements
        for text in legend.get_texts():
            text.set_color(text_color)
            text.set_fontsize(font_size)
            
        # Style legend title if present
        if legend.get_title():
            legend.get_title().set_color(text_color)
            legend.get_title().set_fontsize(font_size + 1)
            legend.get_title().set_weight('bold')
            
        # Add padding for better appearance
        legend.set_frame_on(True)
        legend.get_frame().set_linewidth(0.8)
        
        # Ensure legend is visible against background
        legend.set_zorder(999)
    
    def add_chart_annotation(self, ax, text, position='top', **kwargs):
        """Add informative annotation to chart with theme-aware styling.
        
        Args:
            ax: Matplotlib axes
            text: Annotation text
            position: 'top', 'bottom', 'right', or tuple (x, y) in axes coordinates
            **kwargs: Additional text properties
        """
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["secondary"]
        
        # Default text properties
        default_props = {
            'fontsize': 8,
            'color': text_color,
            'alpha': 0.8,
            'ha': 'center',
            'va': 'top',
            'transform': ax.transAxes,
            'wrap': True
        }
        default_props.update(kwargs)
        
        # Determine position
        if position == 'top':
            x, y = 0.5, 0.98
        elif position == 'bottom':
            x, y = 0.5, 0.02
            default_props['va'] = 'bottom'
        elif position == 'right':
            x, y = 0.98, 0.5
            default_props['ha'] = 'right'
            default_props['va'] = 'center'
        elif isinstance(position, tuple) and len(position) == 2:
            x, y = position
        else:
            x, y = 0.5, 0.98
            
        # Add the text
        return ax.text(x, y, text, **default_props)
    
    def add_reference_lines(self, ax, values_dict, orientation='horizontal'):
        """Add reference lines with labels to chart.
        
        Args:
            ax: Matplotlib axes
            values_dict: Dict of {label: value} for reference lines
            orientation: 'horizontal' or 'vertical'
        """
        theme_colors = ThemeHelper.get_theme_colors()
        
        for label, value in values_dict.items():
            # Determine color based on label keywords
            if any(word in label.lower() for word in ['target', 'goal', 'optimal']):
                color = self.qa_colors.get('good', 'green')
                style = '--'
            elif any(word in label.lower() for word in ['warning', 'caution']):
                color = self.qa_colors.get('warning', 'orange')
                style = ':'
            elif any(word in label.lower() for word in ['limit', 'critical', 'fail']):
                color = self.qa_colors.get('bad', 'red')
                style = '--'
            else:
                color = theme_colors["fg"]["disabled"]
                style = '-'
                
            # Add line
            if orientation == 'horizontal':
                line = ax.axhline(y=value, color=color, linestyle=style, 
                                alpha=0.7, linewidth=1.5, label=f'{label}: {value}')
            else:
                line = ax.axvline(x=value, color=color, linestyle=style, 
                                alpha=0.7, linewidth=1.5, label=f'{label}: {value}')
    
    def _apply_theme_to_axes(self, ax):
        """Apply theme colors to matplotlib axes with enhanced readability."""
        # Get current theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        
        # Set background based on theme
        if is_dark:
            ax.set_facecolor(theme_colors["bg"]["secondary"])  # Dark background
            text_color = theme_colors["fg"]["primary"]  # White text
            grid_color = theme_colors["border"]["primary"]  # Subtle grid
            spine_alpha = 0.7
        else:
            ax.set_facecolor("#ffffff")  # Pure white background for light mode
            text_color = theme_colors["fg"]["primary"]  # Black text
            grid_color = theme_colors["border"]["primary"]  # Light gray grid
            spine_alpha = 0.5
            
        # Ensure figure background matches
        self.figure.patch.set_facecolor(theme_colors["bg"]["primary"] if is_dark else "#ffffff")
            
        # Enhanced font sizes for better readability
        base_size = 10 if self.figsize[0] > 8 else 9
        title_size = base_size + 2
        label_size = base_size
        tick_size = base_size - 1
        
        # Apply text colors with proper sizes
        ax.tick_params(colors=text_color, labelcolor=text_color, labelsize=tick_size, which='both')
        ax.tick_params(axis='x', colors=text_color, labelcolor=text_color)
        ax.tick_params(axis='y', colors=text_color, labelcolor=text_color)
        
        for spine in ax.spines.values():
            spine.set_color(grid_color)
            spine.set_alpha(spine_alpha)
            spine.set_linewidth(1.0)
        
        # Set label properties
        ax.xaxis.label.set_color(text_color)
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_color(text_color)
        ax.yaxis.label.set_fontsize(label_size)
        
        # Set title properties
        if ax.get_title():
            ax.set_title(ax.get_title(), color=text_color, fontsize=title_size, weight='bold')
            
        # Ensure all text elements have proper color
        for text in ax.get_xticklabels():
            text.set_color(text_color)
        for text in ax.get_yticklabels():
            text.set_color(text_color)
            
        # Set grid style with theme-appropriate alpha
        grid_alpha = 0.2 if is_dark else 0.3
        ax.grid(True, alpha=grid_alpha, color=grid_color, linestyle='-', linewidth=0.5)
        
    def _update_figure_theme(self):
        """Update figure background based on current theme."""
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        
        if is_dark:
            # Dark theme colors
            self.figure.patch.set_facecolor(theme_colors["bg"]["primary"])
            # Also update canvas background
            if hasattr(self, 'canvas'):
                self.canvas.get_tk_widget().configure(bg=theme_colors["bg"]["primary"])
        else:
            # Light theme colors
            self.figure.patch.set_facecolor(theme_colors["bg"]["secondary"])
            if hasattr(self, 'canvas'):
                self.canvas.get_tk_widget().configure(bg=theme_colors["bg"]["secondary"])
            
    def _get_or_create_axes(self):
        """Get existing axes or create new one with theme applied."""
        # Clear figure if we're transitioning from placeholder to data
        if not self._has_data and self.figure.axes:
            self.figure.clear()
            # Reapply figure theme after clearing
            self._update_figure_theme()
            
        if not self.figure.axes:
            ax = self.figure.add_subplot(111)
            self._apply_theme_to_axes(ax)
        else:
            ax = self.figure.axes[0]
            ax.clear()  # Clear existing plot data
            self._apply_theme_to_axes(ax)  # Reapply theme
        return ax

    def _setup_ui(self):
        """Set up the chart widget UI."""
        # Get theme colors first
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        fig_bg = theme_colors["bg"]["primary"] if is_dark else theme_colors["bg"]["secondary"]
        
        # Create matplotlib figure with theme-aware colors
        self.figure = Figure(figsize=self.figsize, dpi=100, facecolor=fig_bg, edgecolor='none')
        # Set constrained layout to prevent overlapping
        self.figure.set_constrained_layout(True)
        self.figure.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1)
        
        # Apply theme-aware background
        self._update_figure_theme()

        # Create canvas with theme-aware background
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        bg_color = theme_colors["bg"]["primary"] if is_dark else theme_colors["bg"]["secondary"]
        self.canvas.get_tk_widget().configure(bg=bg_color)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Bind resize event for responsive design
        self.canvas.get_tk_widget().bind('<Configure>', self._on_resize)

        # Create toolbar frame
        toolbar_frame = ctk.CTkFrame(self)
        toolbar_frame.pack(fill='x', side='bottom')

        # Add navigation toolbar with theme styling
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Style the toolbar to match theme
        self._style_toolbar()

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
        
        # Initialize theme tracking
        self._current_theme = ctk.get_appearance_mode()
        
        # Start checking for theme changes
        self.after(1000, self._check_theme_change)

    def update_chart_data(self, data: pd.DataFrame):
        """
        Update chart with new data based on chart type.
        
        Args:
            data: DataFrame with columns appropriate for the chart type
        """
        # Show loading state immediately
        self.show_loading()
        
        # Process on next event loop iteration to allow UI update
        self.after(10, self._process_chart_update, data)
    
    def _process_chart_update(self, data: pd.DataFrame):
        """Process the actual chart update after showing loading state."""
        # Store data for chart type changes
        self._last_data = data.copy() if data is not None and len(data) > 0 else None
        
        if data is None or len(data) == 0:
            logger.warning(f"ChartWidget.update_chart_data called with empty data for {self.title}")
            self.show_placeholder("No data available", "Load or analyze data to display chart")
            return
            
        # Additional validation - check if required columns exist
        if isinstance(data, pd.DataFrame):
            logger.debug(f"ChartWidget received data with columns: {data.columns.tolist()}, shape: {data.shape}")
            # Check for empty columns
            if all(data[col].isna().all() for col in data.columns):
                logger.warning(f"All columns in data are empty for {self.title}")
                self.show_placeholder("All data values are empty", "Check data processing pipeline")
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
                
        except KeyError as e:
            logger.error(f"Missing required column for {self.chart_type} chart: {e}", exc_info=True)
            self.show_error(
                "Data Format Error",
                f"Missing required column: {str(e)}\n\nPlease check the data processing."
            )
        except ValueError as e:
            logger.error(f"Invalid data values: {e}", exc_info=True)
            self.show_error(
                "Invalid Data",
                f"The data contains invalid values:\n{str(e)}\n\nPlease verify the data source."
            )
        except Exception as e:
            logger.error(f"Error updating chart: {e}", exc_info=True)
            self.show_error(
                "Chart Display Error",
                f"Unable to display the chart:\n{str(e)}\n\nTry refreshing the data."
            )
            
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
            # Ensure trim_date is datetime
            data['trim_date'] = pd.to_datetime(data['trim_date'])
            
            # Aggregate data by date to avoid multiple points per date
            # This prevents erratic vertical lines
            if len(data) > 1:
                # Group by date and calculate mean for each date
                data_agg = data.groupby(data['trim_date'].dt.date).agg({
                    'sigma_gradient': 'mean'
                }).reset_index()
                data_agg.columns = ['trim_date', 'sigma_gradient']
                data_agg['trim_date'] = pd.to_datetime(data_agg['trim_date'])
                data_sorted = data_agg.sort_values('trim_date')
            else:
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
                    # Less than a week - show full date and time
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
                elif date_range <= 31:
                    # Less than a month - show full date
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, date_range // 10)))
                elif date_range <= 365:
                    # Less than a year - show full date
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                else:
                    # More than a year - show year/month/day
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                
                # Rotate labels for better readability with more spacing
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Add padding to prevent label cutoff
                ax.tick_params(axis='x', pad=5)
                
                # Reduce number of ticks if too crowded
                num_ticks = len(ax.get_xticks())
                if num_ticks > 15:
                    # Show only every nth tick to prevent crowding
                    n = max(2, num_ticks // 10)
                    for i, label in enumerate(ax.xaxis.get_ticklabels()):
                        if i % n != 0:
                            label.set_visible(False)
                
        if self.title:
            ax.set_title(self.title, color=text_color)
        ax.grid(True, alpha=0.3, color=grid_color)
        
        # Layout is handled by constrained_layout set in __init__
        # No need for manual adjustments
        
        # Force redraw with flush_events to ensure proper rendering
        self.canvas.draw()
        self.canvas.flush_events()
        
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
            
            # Always rotate labels if more than a few categories for readability
            if len(categories) > 3:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
            # Reduce font size for many categories
            if len(categories) > 10:
                ax.tick_params(axis='x', labelsize=8)
            elif len(categories) > 7:
                ax.tick_params(axis='x', labelsize=9)
                
        if self.title:
            ax.set_title(self.title, color=text_color)
        ax.grid(True, alpha=0.3, axis='y', color=grid_color)
        
        # Layout is handled by constrained_layout set in __init__
        # No need for manual adjustments
        
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
        
        # Layout is handled by constrained_layout set in __init__
        # No need for manual adjustments
        
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
        
        # Layout is handled by constrained_layout set in __init__
        # No need for manual adjustments
        
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
        
        # Layout is handled by constrained_layout set in __init__
        # No need for manual adjustments
        
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

        # Get theme colors for text
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        
        # Add value labels on bars with theme-appropriate color
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, color=text_color)

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

        # Layout is handled by constrained_layout set in __init__
        # No need for manual adjustments

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

        # Layout is handled by constrained_layout set in __init__
        # No need for manual adjustments

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

    def plot_pie(self, values: List[float], labels: List[str], 
                 colors: List[str] = None, explode: List[float] = None):
        """
        Plot a pie chart.
        
        Args:
            values: List of values for each pie slice
            labels: List of labels for each slice
            colors: Optional list of colors for each slice
            explode: Optional list of values to explode slices
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._has_data = True
        
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        
        # Use provided colors or default palette
        if colors is None:
            colors = [self.qa_colors['pass'], self.qa_colors['warning'], 
                     self.qa_colors['fail'], self.qa_colors['primary']]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors,
                                         explode=explode, autopct='%1.1f%%',
                                         shadow=True, startangle=90)
        
        # Style the text
        for text in texts:
            text.set_color(text_color)
            text.set_fontsize(10)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_weight('bold')
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        if self.title:
            ax.set_title(self.title, color=text_color, pad=20)
        
        # Apply theme to figure background
        self._update_figure_theme()
        
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
    
    def show_loading(self):
        """Show loading state in the chart area."""
        self.clear_chart()
        
        # Update figure theme
        self._update_figure_theme()
        
        # Add empty axes to show clean background
        ax = self.figure.add_subplot(111)
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])
        
        # Show loading message
        text_color = theme_colors["fg"]["tertiary"]
        ax.text(0.5, 0.5, 'Loading chart data...', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=16,
                color=text_color,
                alpha=0.8)
        
        # Remove axes elements
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()
    
    def show_error(self, error_title: str, error_message: str):
        """Show error state in the chart area with user-friendly message."""
        self.clear_chart()
        
        # Update figure theme
        self._update_figure_theme()
        
        # Add empty axes to show clean background
        ax = self.figure.add_subplot(111)
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])
        
        # Create error message with icon and formatting
        text_parts = ['⚠ ' + error_title]  # Warning icon
        text_parts.append('')
        # Split long error messages
        import textwrap
        wrapped_message = textwrap.fill(error_message, width=60)
        text_parts.extend(wrapped_message.split('\n'))
        
        # Use error color for title, normal color for message
        ax.text(0.5, 0.55, text_parts[0], 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=14,
                color='#e74c3c',  # Error red
                weight='bold')
        
        # Show rest of message
        if len(text_parts) > 1:
            ax.text(0.5, 0.45, '\n'.join(text_parts[1:]), 
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes,
                    fontsize=11,
                    color=theme_colors["fg"]["secondary"],
                    alpha=0.9,
                    linespacing=1.5)
        
        # Remove axes elements
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()
    
    def refresh_theme(self):
        """Refresh chart colors when theme changes."""
        # Update matplotlib rcParams for new theme
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        
        # Update matplotlib parameters
        if is_dark:
            plt.rcParams['figure.facecolor'] = theme_colors["bg"]["primary"]
            plt.rcParams['axes.facecolor'] = theme_colors["bg"]["secondary"]
            plt.rcParams['savefig.facecolor'] = theme_colors["bg"]["primary"]
            plt.rcParams['text.color'] = theme_colors["fg"]["primary"]
            plt.rcParams['axes.labelcolor'] = theme_colors["fg"]["primary"]
            plt.rcParams['xtick.color'] = theme_colors["fg"]["primary"]
            plt.rcParams['ytick.color'] = theme_colors["fg"]["primary"]
        else:
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['savefig.facecolor'] = 'white'
            plt.rcParams['text.color'] = theme_colors["fg"]["primary"]
            plt.rcParams['axes.labelcolor'] = theme_colors["fg"]["primary"]
            plt.rcParams['xtick.color'] = theme_colors["fg"]["primary"]
            plt.rcParams['ytick.color'] = theme_colors["fg"]["primary"]
        
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
    
    def _on_resize(self, event):
        """Handle widget resize for responsive design."""
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = None
            
        # Cancel previous timer to prevent excessive redraws
        if self._resize_timer:
            self.after_cancel(self._resize_timer)
            
        # Schedule redraw after a short delay to prevent flickering
        self._resize_timer = self.after(300, self._handle_resize, event)
    
    def _handle_resize(self, event):
        """Handle the actual resize after timer delay."""
        # Get new size
        width = event.width
        height = event.height
        
        # Don't resize if too small
        if width < 100 or height < 100:
            return
            
        # Calculate new figure size in inches (accounting for DPI)
        dpi = self.figure.dpi
        new_width_inches = max(4, width / dpi)
        new_height_inches = max(3, height / dpi)
        
        # Only resize if size changed significantly (more than 10%)
        old_width, old_height = self.figure.get_size_inches()
        if abs(new_width_inches - old_width) / old_width > 0.1 or abs(new_height_inches - old_height) / old_height > 0.1:
            # Update figure size
            self.figure.set_size_inches(new_width_inches, new_height_inches)
            
            # Adjust font sizes based on new size
            self._adjust_font_sizes(new_width_inches, new_height_inches)
            
            # Layout is handled by constrained_layout set in __init__
            # No need for tight_layout
                
            self.canvas.draw_idle()
    
    def _adjust_font_sizes(self, width_inches, height_inches):
        """Adjust font sizes based on figure size for responsive design."""
        # Calculate scale factor based on figure size
        base_width = 8  # Original figsize width
        scale_factor = min(width_inches / base_width, 1.5)  # Cap at 1.5x
        scale_factor = max(scale_factor, 0.7)  # Minimum 0.7x
        
        # Update font sizes for all text elements
        for ax in self.figure.axes:
            # Title - use set_title to update or get_title to access
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=14 * scale_factor)
            
            # Axis labels
            ax.xaxis.label.set_fontsize(12 * scale_factor)
            ax.yaxis.label.set_fontsize(12 * scale_factor)
            
            # Tick labels
            ax.tick_params(axis='both', labelsize=10 * scale_factor)
            
            # Legend
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(9 * scale_factor)

    def _change_chart_type(self, new_value=None):
        """Change chart type and refresh if data exists."""
        if new_value:
            self.chart_type = new_value
        else:
            self.chart_type = self.type_combo.get()
        
        # If we have data stored, re-plot with new chart type
        if hasattr(self, '_last_data') and self._last_data is not None:
            self.update_chart_data(self._last_data)

    def _style_toolbar(self):
        """Apply theme styling to the matplotlib navigation toolbar."""
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        
        # Choose appropriate colors for the toolbar
        if is_dark:
            toolbar_bg = "#2b2b2b"  # Dark gray background
            toolbar_fg = "#ffffff"  # White text
            button_bg = "#3c3c3c"  # Slightly lighter for buttons
            button_active_bg = "#4a4a4a"  # Even lighter for hover
        else:
            toolbar_bg = "#f0f0f0"  # Light gray background
            toolbar_fg = "#000000"  # Black text
            button_bg = "#ffffff"  # White for buttons
            button_active_bg = "#e0e0e0"  # Slightly darker for hover
        
        # Style the toolbar itself
        try:
            self.toolbar.configure(bg=toolbar_bg)
            self.toolbar._message_label.configure(bg=toolbar_bg, fg=toolbar_fg)
        except:
            pass
        
        # Style all toolbar buttons and widgets
        for widget in self.toolbar.winfo_children():
            try:
                widget_class = widget.winfo_class()
                if widget_class in ['Button', 'Checkbutton']:
                    widget.configure(
                        bg=button_bg,
                        fg=toolbar_fg,
                        activebackground=button_active_bg,
                        activeforeground=toolbar_fg,
                        highlightbackground=toolbar_bg,
                        highlightcolor=toolbar_bg,
                        relief='flat',
                        bd=1
                    )
                elif widget_class == 'Frame':
                    widget.configure(bg=toolbar_bg)
                elif widget_class == 'Label':
                    widget.configure(bg=toolbar_bg, fg=toolbar_fg)
            except:
                pass

    def _check_theme_change(self):
        """Check if theme has changed and update if necessary."""
        current_theme = ctk.get_appearance_mode()
        if current_theme != self._current_theme:
            self._current_theme = current_theme
            self._on_theme_change()
        
        # Schedule next check
        self.after(1000, self._check_theme_change)
    
    def _on_theme_change(self):
        """Handle theme change events."""
        # Update figure theme
        self._update_figure_theme()
        
        # Update toolbar styling
        self._style_toolbar()
        
        # Redraw the chart
        if hasattr(self, '_last_data') and self._last_data is not None:
            self.update_chart_data(self._last_data)
        else:
            self.canvas.draw_idle()

    def _cleanup(self):
        """Clean up resources when widget is destroyed."""
        # Currently no cleanup needed
        pass

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

    def plot_quality_dashboard(self, metrics: Dict[str, Dict[str, Any]]):
        """
        Create a quality health dashboard with traffic lights, gauges, and sparklines.
        
        Args:
            metrics: Dictionary with metric names as keys and values containing:
                    - 'value': Current value
                    - 'status': 'green', 'yellow', or 'red'
                    - 'trend': 'up', 'down', or 'stable'
                    - 'history': List of recent values for sparkline
                    - 'target': Target value (optional)
                    - 'label': Display label (optional)
        """
        self.figure.clear()
        self._has_data = True
        
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        bg_color = theme_colors["bg"]["secondary"]
        
        # Create single main axis for better layout control
        ax = self.figure.add_subplot(111)
        self._apply_theme_to_axes(ax)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Calculate card layout
        n_metrics = len(metrics)
        n_cols = 2  # Fixed 2 columns for 4 metrics
        n_rows = 2  # Fixed 2 rows for 4 metrics
        
        # Card dimensions and spacing
        card_width = 0.42
        card_height = 0.35
        spacing_x = 0.06
        spacing_y = 0.1
        
        # Center the grid
        total_width = n_cols * card_width + (n_cols - 1) * spacing_x
        total_height = n_rows * card_height + (n_rows - 1) * spacing_y
        start_x = (1 - total_width) / 2
        start_y = (1 - total_height) / 2 - 0.05  # Shift up slightly for title
        
        # Draw each metric card
        for idx, (metric_name, metric_data) in enumerate(metrics.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            # Calculate card position
            x = start_x + col * (card_width + spacing_x)
            y = start_y + (n_rows - 1 - row) * (card_height + spacing_y)
            
            # Draw card background
            card_rect = plt.Rectangle((x, y), card_width, card_height,
                                    facecolor=bg_color, edgecolor=theme_colors["border"]["primary"],
                                    linewidth=1, alpha=0.3, transform=ax.transAxes)
            ax.add_patch(card_rect)
            
            # Create metric content
            self._draw_metric_card(ax, x, y, card_width, card_height, metric_name, metric_data, theme_colors)
        
        # Add title
        ax.text(0.5, 0.95, 'Quality Health Dashboard', 
                ha='center', va='top', fontsize=14, weight='bold',
                color=text_color, transform=ax.transAxes)
        ax.text(0.5, 0.91, 'Real-time metrics with trend indicators', 
                ha='center', va='top', fontsize=10,
                color=text_color, alpha=0.7, transform=ax.transAxes)
        
        self.canvas.draw_idle()
    
    def _draw_metric_card(self, ax, x, y, width, height, metric_name, metric_data, theme_colors):
        """Draw a single metric card with value, status indicator, and trend."""
        # Extract data
        value = metric_data.get('value', 0)
        status = metric_data.get('status', 'gray')
        trend = metric_data.get('trend', 'stable')
        history = metric_data.get('history', [])
        target = metric_data.get('target', None)
        label = metric_data.get('label', metric_name)
        
        # Colors
        text_color = theme_colors["fg"]["primary"]
        status_colors = {
            'green': self.qa_colors.get('good', '#27ae60'),
            'yellow': self.qa_colors.get('warning', '#f39c12'),
            'red': self.qa_colors.get('bad', '#e74c3c'),
            'gray': '#95a5a6'
        }
        
        # Status indicator circle (left side)
        circle_x = x + 0.05
        circle_y = y + height - 0.08
        circle = plt.Circle((circle_x, circle_y), 0.02, 
                          color=status_colors.get(status, '#95a5a6'),
                          transform=ax.transAxes, zorder=10)
        ax.add_patch(circle)
        
        # Metric label (top)
        ax.text(x + width/2, y + height - 0.05, label,
                ha='center', va='top', fontsize=11, weight='bold',
                color=text_color, transform=ax.transAxes)
        
        # Value (center, large)
        value_text = f'{value:.1f}%' if isinstance(value, (int, float)) else str(value)
        ax.text(x + width/2, y + height/2 + 0.02, value_text,
                ha='center', va='center', fontsize=20, weight='bold',
                color=status_colors.get(status, text_color), transform=ax.transAxes)
        
        # Target (below value)
        if target is not None:
            ax.text(x + width/2, y + height/2 - 0.05, f'Target: {target}%',
                    ha='center', va='center', fontsize=9,
                    color=text_color, alpha=0.7, transform=ax.transAxes)
        
        # Trend arrow (right side)
        trend_symbols = {'up': '↑', 'down': '↓', 'stable': '→'}
        trend_colors = {'up': self.qa_colors.get('good', '#27ae60'), 
                       'down': self.qa_colors.get('bad', '#e74c3c'), 
                       'stable': '#3498db'}
        ax.text(x + width - 0.05, y + height - 0.08, trend_symbols.get(trend, '→'),
                ha='right', va='top', fontsize=16,
                color=trend_colors.get(trend, '#3498db'), transform=ax.transAxes)
        
        # Mini sparkline (bottom)
        if history and len(history) > 1:
            # Create inset axes for sparkline
            spark_height = 0.08
            spark_y = y + 0.02
            spark_ax = ax.inset_axes([x + 0.05, spark_y, width - 0.1, spark_height],
                                   transform=ax.transAxes)
            
            # Plot sparkline
            spark_ax.plot(range(len(history)), history, 
                         color=status_colors.get(status, '#3498db'), 
                         linewidth=2, alpha=0.8)
            spark_ax.fill_between(range(len(history)), history, 
                                 alpha=0.2, color=status_colors.get(status, '#3498db'))
            
            # Remove all decorations
            spark_ax.set_xticks([])
            spark_ax.set_yticks([])
            for spine in spark_ax.spines.values():
                spine.set_visible(False)
            spark_ax.set_xlim(0, len(history) - 1)
            
            # Add min/max labels
            if len(history) > 0:
                min_val = min(history)
                max_val = max(history)
                spark_ax.text(0, min_val, f'{min_val:.0f}', 
                            fontsize=7, va='top', ha='left',
                            color=text_color, alpha=0.5)
                spark_ax.text(len(history)-1, max_val, f'{max_val:.0f}', 
                            fontsize=7, va='bottom', ha='right',
                            color=text_color, alpha=0.5)
    
    def _create_metric_dashboard(self, ax, metric_name, metric_data, theme_colors):
        """Create a single metric dashboard with traffic light, value, and sparkline."""
        # Clear axes
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Extract data
        value = metric_data.get('value', 0)
        status = metric_data.get('status', 'gray')
        trend = metric_data.get('trend', 'stable')
        history = metric_data.get('history', [])
        target = metric_data.get('target', None)
        label = metric_data.get('label', metric_name)
        
        # Traffic light (top left)
        light_colors = {
            'green': '#27ae60',
            'yellow': '#f39c12', 
            'red': '#e74c3c',
            'gray': '#95a5a6'
        }
        circle = plt.Circle((1.5, 8), 0.8, color=light_colors.get(status, '#95a5a6'), 
                           ec='black', linewidth=2)
        ax.add_patch(circle)
        
        # Metric name and value (top center/right)
        ax.text(3, 8.5, label, fontsize=10, weight='bold', 
                color=theme_colors["fg"]["primary"])
        ax.text(3, 7.5, f'{value:.1f}%' if isinstance(value, (int, float)) else str(value), 
                fontsize=16, color=theme_colors["fg"]["primary"])
        
        # Trend arrow
        trend_symbols = {'up': '↑', 'down': '↓', 'stable': '→'}
        trend_colors = {'up': '#27ae60', 'down': '#e74c3c', 'stable': '#3498db'}
        ax.text(6.5, 7.5, trend_symbols.get(trend, '→'), fontsize=20,
                color=trend_colors.get(trend, '#3498db'))
        
        # Target line if provided
        if target is not None:
            ax.text(8, 8.5, f'Target: {target:.1f}%', fontsize=8,
                   color=theme_colors["fg"]["secondary"])
        
        # Sparkline (bottom half)
        if history and len(history) > 1:
            spark_ax = ax.inset_axes([0.1, 0.1, 0.8, 0.4])
            spark_ax.plot(range(len(history)), history, 
                         color=light_colors.get(status, '#3498db'), linewidth=2)
            spark_ax.fill_between(range(len(history)), history, 
                                 alpha=0.3, color=light_colors.get(status, '#3498db'))
            
            # Remove all spines and ticks
            spark_ax.set_xticks([])
            spark_ax.set_yticks([])
            for spine in spark_ax.spines.values():
                spine.set_visible(False)
            
            # Add min/max labels
            if len(history) > 0:
                spark_ax.text(0, min(history), f'{min(history):.0f}', 
                            fontsize=7, va='top', color=theme_colors["fg"]["tertiary"])
                spark_ax.text(len(history)-1, max(history), f'{max(history):.0f}', 
                            fontsize=7, va='bottom', ha='right', 
                            color=theme_colors["fg"]["tertiary"])
    
    def plot_gauge(self, value: float, min_val: float = 0, max_val: float = 100,
                   target: Optional[float] = None, title: str = "",
                   zones: Optional[List[Tuple[float, float, str]]] = None):
        """
        Create a gauge chart.
        
        Args:
            value: Current value
            min_val: Minimum value
            max_val: Maximum value  
            target: Target value (optional)
            title: Gauge title
            zones: List of (start, end, color) tuples for colored zones
        """
        self.figure.clear()
        self._has_data = True
        
        ax = self.figure.add_subplot(111, projection='polar')
        
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        
        # Set up the gauge
        theta_min = np.pi * 0.75  # Start at 135 degrees
        theta_max = np.pi * 0.25  # End at 45 degrees
        
        # Draw colored zones if provided
        if zones:
            for start, end, color in zones:
                # Normalize to gauge range
                start_norm = (start - min_val) / (max_val - min_val)
                end_norm = (end - min_val) / (max_val - min_val)
                
                # Convert to angles
                start_angle = theta_min + start_norm * (theta_max - theta_min + 2*np.pi)
                end_angle = theta_min + end_norm * (theta_max - theta_min + 2*np.pi)
                
                # Draw wedge
                wedge = plt.matplotlib.patches.Wedge((0, 0), 1, 
                                                    np.degrees(start_angle),
                                                    np.degrees(end_angle),
                                                    width=0.3, 
                                                    facecolor=color,
                                                    alpha=0.3)
                ax.add_patch(wedge)
        
        # Draw the gauge outline
        theta = np.linspace(theta_min, theta_max + 2*np.pi, 100)
        ax.plot(theta, np.ones_like(theta), 'k-', linewidth=2)
        ax.plot(theta, np.ones_like(theta) * 0.7, 'k-', linewidth=1)
        
        # Add value indicator
        value_norm = (value - min_val) / (max_val - min_val)
        value_angle = theta_min + value_norm * (theta_max - theta_min + 2*np.pi)
        ax.plot([value_angle, value_angle], [0.6, 1.1], 'r-', linewidth=3)
        
        # Add target line if provided
        if target is not None:
            target_norm = (target - min_val) / (max_val - min_val)
            target_angle = theta_min + target_norm * (theta_max - theta_min + 2*np.pi)
            ax.plot([target_angle, target_angle], [0.7, 1], 'g--', linewidth=2, alpha=0.7)
        
        # Add scale labels
        for i in range(0, 101, 20):
            val = min_val + (max_val - min_val) * i / 100
            angle = theta_min + i/100 * (theta_max - theta_min + 2*np.pi)
            ax.text(angle, 1.15, f'{val:.0f}', ha='center', va='center',
                   fontsize=8, color=text_color)
        
        # Clear default polar labels
        ax.set_ylim(0, 1.3)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        
        # Add title and value
        ax.text(0, -0.2, title, ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, weight='bold',
                color=text_color)
        ax.text(0, -0.35, f'{value:.1f}', ha='center', va='center',
                transform=ax.transAxes, fontsize=20, color=text_color)
        
        self.canvas.draw_idle()
    
    def plot_early_warning_system(self, data: pd.DataFrame):
        """
        Create an early warning system with moving range and CUSUM charts.
        
        Args:
            data: DataFrame with 'trim_date' and 'sigma_gradient' columns
        """
        self.figure.clear()
        self._has_data = True
        
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        
        # Create subplots
        fig = self.figure
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1.5, 0.5], hspace=0.3)
        
        # 1. Main control chart with violations
        ax1 = fig.add_subplot(gs[0])
        self._apply_theme_to_axes(ax1)
        
        # Plot sigma gradient with control limits
        dates = data['trim_date']
        values = data['sigma_gradient']
        
        # Calculate control limits
        mean_val = values.mean()
        std_val = values.std()
        ucl = mean_val + 3 * std_val
        lcl = mean_val - 3 * std_val
        uwl = mean_val + 2 * std_val
        lwl = mean_val - 2 * std_val
        
        # Plot main data
        ax1.plot(dates, values, 'o-', color=self.qa_colors['primary'], 
                markersize=4, linewidth=1.5, label='Sigma Gradient')
        
        # Plot control limits
        ax1.axhline(ucl, color='red', linestyle='--', alpha=0.5, label=f'UCL ({ucl:.3f})')
        ax1.axhline(uwl, color='orange', linestyle=':', alpha=0.5, label=f'UWL ({uwl:.3f})')
        ax1.axhline(mean_val, color='green', linestyle='-', alpha=0.5, label=f'Mean ({mean_val:.3f})')
        ax1.axhline(lwl, color='orange', linestyle=':', alpha=0.5, label=f'LWL ({lwl:.3f})')
        ax1.axhline(lcl, color='red', linestyle='--', alpha=0.5, label=f'LCL ({lcl:.3f})')
        
        # Highlight violations
        violations = ((values > ucl) | (values < lcl))
        warnings = ((values > uwl) & (values <= ucl)) | ((values < lwl) & (values >= lcl))
        
        if violations.any():
            ax1.scatter(dates[violations], values[violations], 
                       color='red', s=100, marker='x', linewidth=3, label='Violations', zorder=5)
        if warnings.any():
            ax1.scatter(dates[warnings], values[warnings], 
                       color='orange', s=80, marker='^', label='Warnings', zorder=5)
        
        ax1.set_ylabel('Sigma Gradient', fontsize=10)
        ax1.set_title('Control Chart with Violation Detection', fontsize=12, color=text_color)
        ax1.legend(loc='upper left', fontsize=8, ncol=3)
        ax1.grid(True, alpha=0.3)
        
        # Format dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Moving Range Chart
        ax2 = fig.add_subplot(gs[1])
        self._apply_theme_to_axes(ax2)
        
        # Calculate moving range
        moving_range = np.abs(values.diff())
        avg_mr = moving_range.mean()
        ucl_mr = avg_mr * 3.267  # D4 constant for n=2
        
        ax2.plot(dates[1:], moving_range[1:], 'o-', color=self.qa_colors['secondary'], 
                markersize=3, linewidth=1, label='Moving Range')
        ax2.axhline(avg_mr, color='green', linestyle='-', alpha=0.5, label=f'Avg MR ({avg_mr:.3f})')
        ax2.axhline(ucl_mr, color='red', linestyle='--', alpha=0.5, label=f'UCL ({ucl_mr:.3f})')
        
        # Highlight large variations
        large_variations = moving_range > ucl_mr
        if large_variations.any():
            ax2.scatter(dates[large_variations], moving_range[large_variations],
                       color='red', s=80, marker='x', linewidth=2, label='High Variation', zorder=5)
        
        ax2.set_ylabel('Moving Range', fontsize=10)
        ax2.set_title('Moving Range Chart - Variation Detection', fontsize=11, color=text_color)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Format dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. CUSUM Indicator Bar
        ax3 = fig.add_subplot(gs[2])
        self._apply_theme_to_axes(ax3)
        
        # Calculate CUSUM
        target = 0.5  # Target sigma gradient
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))
        
        for i in range(1, len(values)):
            cusum_pos[i] = max(0, values.iloc[i] - target + cusum_pos[i-1])
            cusum_neg[i] = min(0, values.iloc[i] - target + cusum_neg[i-1])
        
        # Detect shifts
        h = 4 * std_val  # Decision interval
        shift_up = cusum_pos > h
        shift_down = cusum_neg < -h
        
        # Create status bar
        status = np.zeros(len(dates))
        status[shift_up] = 1  # Upward shift
        status[shift_down] = -1  # Downward shift
        
        # Plot as colored bars
        for i in range(len(dates)):
            if status[i] == 1:
                ax3.axvspan(i-0.5, i+0.5, color='red', alpha=0.7)
            elif status[i] == -1:
                ax3.axvspan(i-0.5, i+0.5, color='blue', alpha=0.7)
            else:
                ax3.axvspan(i-0.5, i+0.5, color='green', alpha=0.3)
        
        ax3.set_xlim(-0.5, len(dates)-0.5)
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_xlabel('Time', fontsize=10)
        ax3.set_title('CUSUM Shift Detection: Green=Normal, Red=Upward Shift, Blue=Downward Shift', 
                     fontsize=10, color=text_color)
        ax3.set_yticks([])
        
        # Add text indicators
        if shift_up.any():
            first_up = np.where(shift_up)[0][0]
            ax3.text(first_up, 0.5, 'SHIFT UP', ha='center', va='center', 
                    fontsize=8, color='white', weight='bold')
        if shift_down.any():
            first_down = np.where(shift_down)[0][0]
            ax3.text(first_down, 0.5, 'SHIFT DOWN', ha='center', va='center',
                    fontsize=8, color='white', weight='bold')
        
        # Main title (without overlapping subtitle due to layout constraints)
        fig.suptitle('Early Warning System Dashboard', fontsize=14, color=text_color)
        
        # Layout is handled by constrained_layout
        self.canvas.draw_idle()
    
    def plot_quality_dashboard_cards(self, metrics: Dict[str, Dict]):
        """
        Create a quality health dashboard with KPI cards and sparklines.
        
        Args:
            metrics: Dictionary of metrics with format:
                    {
                        'Pass Rate': {
                            'value': 95.5,
                            'status': 'green',  # 'green', 'yellow', 'red'
                            'trend': 'up',      # 'up', 'down', 'stable'
                            'history': [94, 95, 95.5],  # Optional sparkline data
                            'target': 95,
                            'label': 'Pass Rate'
                        },
                        ...
                    }
        """
        self.figure.clear()
        self._has_data = True
        
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        bg_color = theme_colors["bg"]["secondary"]
        
        # Create single axis for dashboard
        ax = self.figure.add_subplot(111)
        self._apply_theme_to_axes(ax)
        ax.axis('off')
        
        # Calculate card positions
        num_metrics = len(metrics)
        cols = min(num_metrics, 2)  # Max 2 columns
        rows = (num_metrics + 1) // 2
        
        # Card dimensions
        card_width = 0.4
        card_height = 0.35
        spacing_x = 0.1
        spacing_y = 0.15
        
        # Starting position
        start_x = (1 - (cols * card_width + (cols - 1) * spacing_x)) / 2
        start_y = (1 - (rows * card_height + (rows - 1) * spacing_y)) / 2
        
        # Status colors
        status_colors = {
            'green': self.qa_colors['good'],
            'yellow': self.qa_colors['warning'],
            'red': self.qa_colors['bad']
        }
        
        # Draw each metric card
        for idx, (metric_name, metric_data) in enumerate(metrics.items()):
            row = idx // cols
            col = idx % cols
            
            # Calculate card position
            x = start_x + col * (card_width + spacing_x)
            y = start_y + (rows - 1 - row) * (card_height + spacing_y)  # Top to bottom
            
            # Draw card background
            card_bg = plt.Rectangle((x, y), card_width, card_height,
                                  facecolor=bg_color, edgecolor=text_color,
                                  linewidth=1, alpha=0.1, transform=ax.transAxes)
            ax.add_patch(card_bg)
            
            # Metric label
            ax.text(x + card_width/2, y + card_height - 0.05, metric_data['label'],
                   ha='center', va='top', fontsize=11, weight='bold',
                   color=text_color, transform=ax.transAxes)
            
            # Value with color based on status
            value_color = status_colors.get(metric_data['status'], text_color)
            ax.text(x + card_width/2, y + card_height/2, f"{metric_data['value']:.1f}%",
                   ha='center', va='center', fontsize=24, weight='bold',
                   color=value_color, transform=ax.transAxes)
            
            # Target line
            if 'target' in metric_data:
                ax.text(x + card_width/2, y + card_height/2 - 0.08, 
                       f"Target: {metric_data['target']}%",
                       ha='center', va='center', fontsize=9,
                       color=text_color, alpha=0.7, transform=ax.transAxes)
            
            # Trend arrow
            if 'trend' in metric_data:
                trend_symbols = {'up': '↑', 'down': '↓', 'stable': '→'}
                trend_colors = {'up': 'green', 'down': 'red', 'stable': 'gray'}
                trend_symbol = trend_symbols.get(metric_data['trend'], '')
                trend_color = trend_colors.get(metric_data['trend'], text_color)
                
                ax.text(x + card_width - 0.05, y + card_height - 0.05, trend_symbol,
                       ha='right', va='top', fontsize=16, weight='bold',
                       color=trend_color, transform=ax.transAxes)
            
            # Mini sparkline if history is provided
            if 'history' in metric_data and len(metric_data['history']) > 1:
                spark_ax = ax.inset_axes([x + 0.05, y + 0.05, card_width - 0.1, 0.15],
                                       transform=ax.transAxes)
                spark_ax.plot(metric_data['history'], color=value_color, linewidth=2)
                spark_ax.axis('off')
                spark_ax.set_xlim(0, len(metric_data['history']) - 1)
                
        # Title
        self.figure.suptitle('Quality Health Dashboard', fontsize=14, color=text_color)
        
        # Update info text based on overall performance
        avg_value = sum(m['value'] for m in metrics.values()) / len(metrics)
        if avg_value >= 90:
            info_text = "All systems operating within target parameters"
        elif avg_value >= 75:
            info_text = "Some metrics below target - monitoring required"
        else:
            info_text = "Multiple metrics below target - action needed"
            
        ax.text(0.5, 0.02, info_text, ha='center', va='bottom',
               fontsize=10, color=text_color, alpha=0.7, transform=ax.transAxes)
        
        # Layout is handled by constrained_layout
        self.canvas.draw_idle()
    
    def plot_early_warning_system(self, data: pd.DataFrame):
        """
        Create early warning system with control charts and violation detection.
        
        Args:
            data: DataFrame with 'trim_date' and measurement columns
        """
        self.figure.clear()
        self._has_data = True
        
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        
        # Create subplots
        fig = self.figure
        gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 0.8], hspace=0.3)
        
        # 1. Control Chart with Violation Detection (top)
        ax1 = fig.add_subplot(gs[0])
        self._apply_theme_to_axes(ax1)
        
        # Sort by date and get values
        data_sorted = data.sort_values('trim_date')
        
        # Check if we have multiple measurements per date
        measurements_per_date = data_sorted.groupby(data_sorted['trim_date'].dt.date).size()
        
        if measurements_per_date.max() > 1:
            # Aggregate to daily averages to avoid erratic lines
            daily_data = data_sorted.groupby(data_sorted['trim_date'].dt.date).agg({
                data_sorted.columns[1]: 'mean'  # Average the measurement column
            }).reset_index()
            daily_data.columns = ['date', 'value']
            daily_data['date'] = pd.to_datetime(daily_data['date'])
            dates = daily_data['date']
            values = daily_data['value']
        else:
            # Use original data if no duplicates
            dates = data_sorted['trim_date']
            values = data_sorted.iloc[:, 1]  # Assume second column is the measurement
        
        # Calculate control limits
        mean_val = values.mean()
        std_val = values.std()
        ucl = mean_val + 3 * std_val
        lcl = mean_val - 3 * std_val
        uwl = mean_val + 2 * std_val
        lwl = mean_val - 2 * std_val
        
        # Plot data
        ax1.plot(dates, values, 'b-', linewidth=1.5, label='Measurements', marker='o', markersize=4)
        
        # Plot control limits
        ax1.axhline(mean_val, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax1.axhline(ucl, color='red', linestyle='--', linewidth=1.5, label=f'UCL: {ucl:.3f}')
        ax1.axhline(lcl, color='red', linestyle='--', linewidth=1.5, label=f'LCL: {lcl:.3f}')
        ax1.axhline(uwl, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'UWL: {uwl:.3f}')
        ax1.axhline(lwl, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'LWL: {lwl:.3f}')
        
        # Highlight violations
        violations = (values > ucl) | (values < lcl)
        if violations.any():
            ax1.scatter(dates[violations], values[violations], color='red', s=100, 
                       marker='X', zorder=5, label=f'Violations ({violations.sum()})')
        
        # Highlight warnings
        warnings = ((values > uwl) & (values <= ucl)) | ((values < lwl) & (values >= lcl))
        if warnings.any():
            ax1.scatter(dates[warnings], values[warnings], color='orange', s=80, 
                       marker='^', zorder=4, label=f'Warnings ({warnings.sum()})')
        
        # Labels and title
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sigma Gradient')
        ax1.set_title('Control Chart with Violation Detection', fontweight='bold')
        
        # Create comprehensive legend with better positioning
        handles, labels = ax1.get_legend_handles_labels()
        if len(handles) > 6:  # Many items - use columns
            legend = ax1.legend(handles, labels, loc='upper center', 
                              bbox_to_anchor=(0.5, -0.15), ncol=3, 
                              frameon=True, fancybox=True, shadow=True)
        else:
            legend = ax1.legend(handles, labels, loc='best', 
                              frameon=True, fancybox=True, shadow=True)
        self._style_legend(legend)
        
        # Add informative annotation
        info_text = f"UCL/LCL: ±3σ | UWL/LWL: ±2σ | Mean: {mean_val:.3f}"
        self.add_chart_annotation(ax1, info_text, position=(0.02, 0.02), 
                                va='bottom', ha='left')
        
        # Format dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Moving Range Chart - Variation Detection (bottom)
        ax2 = fig.add_subplot(gs[1])
        self._apply_theme_to_axes(ax2)
        
        # Calculate moving range
        if hasattr(values, 'values'):  # If it's a pandas Series
            mr = np.abs(np.diff(values.values))
        else:  # If it's already a numpy array
            mr = np.abs(np.diff(values))
        
        # Handle dates based on type
        if hasattr(dates, 'iloc'):  # If it's a pandas Series
            mr_dates = dates.iloc[1:]
        else:  # If it's already an array-like
            mr_dates = dates[1:]
        mr_mean = mr.mean()
        mr_ucl = mr_mean * 3.267  # D4 factor for n=2
        
        # Plot moving range
        ax2.plot(mr_dates, mr, 'purple', linewidth=1.5, marker='o', markersize=4, label='Moving Range')
        ax2.axhline(mr_mean, color='green', linestyle='-', linewidth=2, label=f'MR Mean: {mr_mean:.3f}')
        ax2.axhline(mr_ucl, color='red', linestyle='--', linewidth=1.5, label=f'MR UCL: {mr_ucl:.3f}')
        
        # Highlight large variations
        large_variations = mr > mr_ucl
        if large_variations.any():
            ax2.scatter(mr_dates[large_variations], mr[large_variations], 
                       color='red', s=100, marker='X', zorder=5, label=f'Large Variations ({large_variations.sum()})')
        
        # Labels and title
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Moving Range')
        ax2.set_title('Moving Range Chart - Variation Detection', fontweight='bold')
        
        # Legend
        legend = ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        self._style_legend(legend)
        
        # Add informative annotation about what this shows
        info_text = f"D4 factor: 3.267 | Mean MR: {mr_mean:.3f}"
        self.add_chart_annotation(ax2, info_text, position=(0.02, 0.98), 
                                va='top', ha='left')
        
        # Format dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add shift detection annotations
        # Simple CUSUM calculation for shift detection
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))
        k = 0.5 * std_val  # Allowance
        
        for i in range(1, len(values)):
            cusum_pos[i] = max(0, values.iloc[i] - (mean_val + k) + cusum_pos[i-1])
            cusum_neg[i] = max(0, (mean_val - k) - values.iloc[i] + cusum_neg[i-1])
        
        # Detect shifts
        h = 5 * std_val  # Decision interval
        shifts_up = cusum_pos > h
        shifts_down = cusum_neg > h
        
        # Add annotations for shift detection
        shift_text = "CUSUM Shift Detection: "
        if shifts_up.any():
            shift_text += f"Upward shift detected "
            # Mark the shift regions
            ax1.axvspan(dates.iloc[np.where(shifts_up)[0][0]], dates.iloc[-1], 
                       alpha=0.2, color='red', label='Upward Shift')
        if shifts_down.any():
            shift_text += f"Downward shift detected"
            ax1.axvspan(dates.iloc[np.where(shifts_down)[0][0]], dates.iloc[-1], 
                       alpha=0.2, color='blue', label='Downward Shift')
        if not shifts_up.any() and not shifts_down.any():
            shift_text += "No shifts detected"
            
        fig.text(0.5, 0.01, shift_text, ha='center', fontsize=10, 
                color=text_color, alpha=0.7)
        
        # Main title
        fig.suptitle('Early Warning System', fontsize=14, color=text_color)
        
        # Layout is handled by constrained_layout
        self.canvas.draw_idle()
    
    def plot_failure_pattern_analysis(self, data: pd.DataFrame):
        """
        Create failure pattern analysis with heat map, Pareto chart, and projection.
        
        Args:
            data: DataFrame with columns including 'trim_date', 'track_status', 
                  'sigma_gradient', 'linearity_pass', etc.
        """
        self.figure.clear()
        self._has_data = True
        
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        
        # Create subplots
        fig = self.figure
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        # 1. Time-based Heat Map (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._apply_theme_to_axes(ax1)
        
        # Prepare heat map data
        fail_data = data[data['track_status'] == 'Fail'].copy()
        
        if len(fail_data) > 0:
            # Create failure categories
            fail_data['failure_type'] = 'Other'
            # Note: Without ML threshold context, we categorize based on failure status
            # The actual threshold determination should come from ML models
            if 'linearity_pass' in fail_data.columns:
                fail_data.loc[fail_data['linearity_pass'] == False, 'failure_type'] = 'Linearity'
            if 'risk_category' in fail_data.columns:
                fail_data.loc[fail_data['risk_category'] == 'High', 'failure_type'] = 'High Risk'
            
            # Group by week and failure type
            fail_data['week'] = fail_data['trim_date'].dt.to_period('W')
            heatmap_data = fail_data.groupby(['week', 'failure_type']).size().unstack(fill_value=0)
            
            if len(heatmap_data) > 0:
                # Create heat map
                im = ax1.imshow(heatmap_data.T, aspect='auto', cmap='Reds', interpolation='nearest')
                
                # Set ticks
                ax1.set_xticks(range(len(heatmap_data.index)))
                ax1.set_xticklabels([str(w).split('/')[1] if '/' in str(w) else str(w) 
                                    for w in heatmap_data.index], rotation=45, ha='right')
                ax1.set_yticks(range(len(heatmap_data.columns)))
                ax1.set_yticklabels(heatmap_data.columns)
                
                # Add text annotations
                for i in range(len(heatmap_data.columns)):
                    for j in range(len(heatmap_data.index)):
                        value = heatmap_data.iloc[j, i]
                        if value > 0:
                            ax1.text(j, i, str(value), ha='center', va='center',
                                    color='white' if value > heatmap_data.max().max()/2 else 'black',
                                    fontsize=8)
                
                # Colorbar - use figure's colorbar method to avoid warning
                cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
                cbar.set_label('Failure Count', fontsize=8)
                
                ax1.set_title('Failure Pattern Heat Map (by Week)', fontsize=11, color=text_color)
                ax1.set_xlabel('Week', fontsize=9)
                ax1.set_ylabel('Failure Type', fontsize=9)
            else:
                ax1.text(0.5, 0.5, 'No failure patterns to display', 
                        ha='center', va='center', transform=ax1.transAxes,
                        fontsize=12, color=text_color)
                ax1.axis('off')
        else:
            ax1.text(0.5, 0.5, 'No failures detected', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=12, color='green')
            ax1.axis('off')
        
        # 2. Pareto Chart (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._apply_theme_to_axes(ax2)
        
        if len(fail_data) > 0:
            # Count failure causes
            failure_counts = fail_data['failure_type'].value_counts()
            
            # Sort and calculate cumulative percentage
            cumulative_percent = failure_counts.cumsum() / failure_counts.sum() * 100
            
            # Create bar chart
            bars = ax2.bar(range(len(failure_counts)), failure_counts.values, 
                           color=self.qa_colors['fail'], alpha=0.7)
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, failure_counts.values)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=8)
            
            # Add cumulative line
            ax2_twin = ax2.twinx()
            ax2_twin.plot(range(len(failure_counts)), cumulative_percent.values, 
                         'o-', color=self.qa_colors['warning'], linewidth=2, markersize=6)
            ax2_twin.set_ylabel('Cumulative %', fontsize=9)
            ax2_twin.set_ylim(0, 105)
            
            # Add 80% reference line
            ax2_twin.axhline(80, color='green', linestyle='--', alpha=0.5)
            ax2_twin.text(len(failure_counts)-0.5, 80, '80%', ha='right', va='bottom', 
                         fontsize=8, color='green')
            
            # Labels
            ax2.set_xticks(range(len(failure_counts)))
            ax2.set_xticklabels(failure_counts.index, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('Count', fontsize=9)
            ax2.set_title('Failure Pareto Chart', fontsize=11, color=text_color)
        else:
            ax2.text(0.5, 0.5, 'No failures\nto analyze', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, color='green')
            ax2.axis('off')
        
        # 3. Failure Rate Projection (bottom)
        ax3 = fig.add_subplot(gs[1, :])
        self._apply_theme_to_axes(ax3)
        
        # Calculate daily failure rates
        daily_data = data.groupby(data['trim_date'].dt.date).agg({
            'track_status': lambda x: (x == 'Fail').mean() * 100
        })
        
        if len(daily_data) > 3:
            # Plot historical failure rate
            dates = pd.to_datetime(daily_data.index)
            fail_rates = daily_data['track_status'].values
            
            ax3.plot(dates, fail_rates, 'o-', color=self.qa_colors['fail'], 
                    linewidth=1.5, markersize=4, label='Actual Failure Rate')
            
            # Fit polynomial for projection
            x_numeric = np.arange(len(dates))
            
            # Use 2nd degree polynomial for smooth projection
            if len(dates) > 5:
                z = np.polyfit(x_numeric, fail_rates, 2)
                p = np.poly1d(z)
                
                # Project forward 7 days
                future_days = 7
                future_x = np.arange(len(dates), len(dates) + future_days)
                future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), 
                                           periods=future_days)
                
                # Calculate projection with confidence bounds
                projected_rates = np.maximum(0, p(future_x))  # Ensure no negative rates
                
                # Simple confidence bounds based on historical variability
                std_error = np.std(fail_rates - p(x_numeric))
                upper_bound = np.minimum(100, projected_rates + 2 * std_error)  # Cap at 100%
                lower_bound = np.maximum(0, projected_rates - 2 * std_error)
                
                # Plot projection
                ax3.plot(future_dates, projected_rates, '--', 
                        color=self.qa_colors['warning'], linewidth=2, 
                        label='Projected Rate')
                ax3.fill_between(future_dates, lower_bound, upper_bound, 
                               color=self.qa_colors['warning'], alpha=0.2, 
                               label='Confidence Interval')
                
                # Add vertical line at today
                ax3.axvline(dates[-1], color='gray', linestyle=':', alpha=0.5)
                ax3.text(dates[-1], ax3.get_ylim()[1]*0.9, 'Today', 
                        ha='center', fontsize=8, color='gray')
            
            # Target line
            ax3.axhline(5, color='green', linestyle='--', alpha=0.5, label='Target (5%)')
            
            # Format
            ax3.set_xlabel('Date', fontsize=10)
            ax3.set_ylabel('Failure Rate (%)', fontsize=10)
            ax3.set_title('Failure Rate Trend & 7-Day Projection', fontsize=12, color=text_color)
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # Format dates
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for projection analysis', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, color=text_color)
            ax3.axis('off')
        
        # Main title with explanation
        fig.suptitle('Failure Pattern Analysis Dashboard', fontsize=14, color=text_color, y=0.98)
        fig.text(0.5, 0.94, 'Heat map shows when failures occur • Pareto identifies top issues • Projection forecasts future risk', 
                ha='center', fontsize=10, color=text_color, alpha=0.7)
        
        # Layout is handled by constrained_layout
        self.canvas.draw_idle()
    
    def plot_performance_scorecard(self, data: pd.DataFrame):
        """
        Create a performance scorecard with quality score, yield/efficiency, and comparisons.
        
        Args:
            data: DataFrame with performance data including dates, pass rates, sigma values, etc.
        """
        self.figure.clear()
        self._has_data = True
        
        # Get theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        text_color = theme_colors["fg"]["primary"]
        
        # Create subplots
        fig = self.figure
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], hspace=0.3, wspace=0.3)
        
        # 1. Quality Score Timeline (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._apply_theme_to_axes(ax1)
        
        # Calculate daily quality scores
        daily_scores = data.groupby(data['trim_date'].dt.date).agg({
            'track_status': lambda x: (x == 'Pass').mean() * 100,
            'sigma_gradient': lambda x: ((x >= 0.3) & (x <= 0.7)).mean() * 100 if len(x) > 0 else 0,
            'linearity_pass': lambda x: x.mean() * 100 if x.notna().any() else 100
        })
        
        # Calculate composite quality score (weighted average)
        weights = {'pass': 0.5, 'sigma': 0.3, 'linearity': 0.2}
        daily_scores['quality_score'] = (
            daily_scores['track_status'] * weights['pass'] +
            daily_scores['sigma_gradient'] * weights['sigma'] +
            daily_scores['linearity_pass'] * weights['linearity']
        )
        
        # Plot quality score
        dates = pd.to_datetime(daily_scores.index)
        scores = daily_scores['quality_score'].values
        
        ax1.plot(dates, scores, 'o-', color=self.qa_colors['primary'], 
                linewidth=2, markersize=5)
        
        # Add color zones
        ax1.axhspan(90, 100, color='green', alpha=0.1, label='Excellent')
        ax1.axhspan(80, 90, color='yellow', alpha=0.1, label='Good')
        ax1.axhspan(70, 80, color='orange', alpha=0.1, label='Fair')
        ax1.axhspan(0, 70, color='red', alpha=0.1, label='Poor')
        
        # Add moving average
        if len(scores) > 7:
            ma = pd.Series(scores).rolling(window=7, center=True).mean()
            ax1.plot(dates, ma, '--', color=self.qa_colors['secondary'], 
                    linewidth=2, alpha=0.7, label='7-day MA')
        
        ax1.set_ylim(0, 105)
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Quality Score (%)', fontsize=10)
        ax1.set_title('Composite Quality Score Timeline', fontsize=11, color=text_color)
        ax1.legend(loc='lower left', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Format dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add score breakdown text
        if len(daily_scores) > 0:
            latest = daily_scores.iloc[-1]
            breakdown = f"Latest: Pass={latest['track_status']:.1f}%, Sigma={latest['sigma_gradient']:.1f}%, Lin={latest['linearity_pass']:.1f}%"
            ax1.text(0.02, 0.98, breakdown, transform=ax1.transAxes, 
                    fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Yield & Efficiency Chart (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._apply_theme_to_axes(ax2)
        
        # Calculate yield and efficiency metrics
        daily_metrics = data.groupby(data['trim_date'].dt.date).agg({
            'track_status': lambda x: (x == 'Pass').mean() * 100,  # Yield
            'sigma_gradient': lambda x: ((x >= 0.4) & (x <= 0.6)).mean() * 100 if len(x) > 0 else 0  # Efficiency (tight spec)
        })
        
        # Plot dual axis
        dates = pd.to_datetime(daily_metrics.index)
        yield_line = ax2.plot(dates, daily_metrics['track_status'], 'o-', 
                             color=self.qa_colors['pass'], linewidth=2, 
                             markersize=4, label='Yield %')
        
        # Add target lines
        ax2.axhline(95, color=self.qa_colors['pass'], linestyle='--', 
                   alpha=0.5, linewidth=1)
        ax2.text(dates[0], 95.5, 'Yield Target', fontsize=8, 
                color=self.qa_colors['pass'])
        
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Yield (%)', fontsize=10, color=self.qa_colors['pass'])
        ax2.tick_params(axis='y', labelcolor=self.qa_colors['pass'])
        
        # Second y-axis for efficiency
        ax2_twin = ax2.twinx()
        eff_line = ax2_twin.plot(dates, daily_metrics['sigma_gradient'], 's-', 
                                color=self.qa_colors['warning'], linewidth=2, 
                                markersize=4, label='Efficiency %')
        
        ax2_twin.axhline(80, color=self.qa_colors['warning'], linestyle='--', 
                        alpha=0.5, linewidth=1)
        ax2_twin.text(dates[-1], 80.5, 'Eff Target', fontsize=8, 
                     color=self.qa_colors['warning'], ha='right')
        
        ax2_twin.set_ylabel('Efficiency (%)', fontsize=10, color=self.qa_colors['warning'])
        ax2_twin.tick_params(axis='y', labelcolor=self.qa_colors['warning'])
        
        # Combined legend
        lines = yield_line + eff_line
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='lower left', fontsize=8)
        
        ax2.set_title('Yield & Process Efficiency', fontsize=11, color=text_color)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Format dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Comparative Performance Table (bottom)
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('tight')
        ax3.axis('off')
        
        # Calculate period comparisons
        if len(data) > 0:
            # Define periods
            now = data['trim_date'].max()
            week_ago = now - pd.Timedelta(days=7)
            month_ago = now - pd.Timedelta(days=30)
            
            # Current week
            current_week = data[data['trim_date'] > week_ago]
            # Previous week
            prev_week = data[(data['trim_date'] > week_ago - pd.Timedelta(days=7)) & 
                           (data['trim_date'] <= week_ago)]
            # Current month
            current_month = data[data['trim_date'] > month_ago]
            
            # Calculate metrics for each period
            def calc_metrics(df_period):
                if len(df_period) == 0:
                    return {'Pass Rate': 0, 'Avg Sigma': 0, 'In Spec': 0, 'Units': 0}
                return {
                    'Pass Rate': f"{(df_period['track_status'] == 'Pass').mean() * 100:.1f}%",
                    'Avg Sigma': f"{df_period['sigma_gradient'].mean():.3f}",
                    'In Spec': f"{((df_period['sigma_gradient'] >= 0.3) & (df_period['sigma_gradient'] <= 0.7)).mean() * 100:.1f}%",
                    'Units': len(df_period)
                }
            
            # Create comparison data
            comparison_data = {
                'Metric': ['Pass Rate', 'Avg Sigma', 'In Spec %', 'Units Tested'],
                'This Week': list(calc_metrics(current_week).values()),
                'Last Week': list(calc_metrics(prev_week).values()),
                'Month Total': list(calc_metrics(current_month).values()),
                'All Time': [
                    f"{(data['track_status'] == 'Pass').mean() * 100:.1f}%",
                    f"{data['sigma_gradient'].mean():.3f}",
                    f"{((data['sigma_gradient'] >= 0.3) & (data['sigma_gradient'] <= 0.7)).mean() * 100:.1f}%",
                    len(data)
                ]
            }
            
            # Create table
            table_data = []
            for i in range(len(comparison_data['Metric'])):
                row = [comparison_data[col][i] for col in comparison_data.keys()]
                table_data.append(row)
            
            # Create table with styling
            table = ax3.table(cellText=table_data,
                            colLabels=list(comparison_data.keys()),
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Color code cells based on performance
            for i in range(1, len(table_data)):  # Skip header row
                for j in range(1, len(comparison_data.keys())):  # Skip metric column
                    cell = table[(i, j)]
                    if comparison_data['Metric'][i-1] == 'Pass Rate':
                        # Color based on pass rate
                        value = float(table_data[i-1][j].rstrip('%')) if '%' in str(table_data[i-1][j]) else 0
                        if value >= 95:
                            cell.set_facecolor('#90EE90')  # Light green
                        elif value >= 85:
                            cell.set_facecolor('#FFFFE0')  # Light yellow
                        else:
                            cell.set_facecolor('#FFB6C1')  # Light red
            
            # Header styling
            for j in range(len(comparison_data.keys())):
                table[(0, j)].set_facecolor('#4CAF50')
                table[(0, j)].set_text_props(weight='bold', color='white')
            
            ax3.set_title('Performance Comparison Table', fontsize=12, 
                         color=text_color, pad=20)
        else:
            ax3.text(0.5, 0.5, 'No data available for comparison', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, color=text_color)
        
        # Main title with explanation
        fig.suptitle('Performance Scorecard Dashboard', fontsize=14, color=text_color, y=0.98)
        fig.text(0.5, 0.94, 'Quality score combines multiple metrics • Dual-axis shows yield vs efficiency • Table compares time periods', 
                ha='center', fontsize=10, color=text_color, alpha=0.7)
        
        # Layout is handled by constrained_layout
        self.canvas.draw_idle()


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
