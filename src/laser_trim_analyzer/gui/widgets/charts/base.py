"""
ChartWidgetBase - Core infrastructure for chart widgets.

This module provides the base class with:
- UI setup and matplotlib integration
- Theme management and styling
- Resize handling
- Export functionality
- Placeholder/error states
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
from matplotlib.ticker import MaxNLocator
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


class ChartWidgetBase(ctk.CTkFrame):
    """
    Base class for chart widgets with core infrastructure.

    Provides:
    - Matplotlib figure/canvas integration
    - Theme-aware styling
    - Toolbar and export functionality
    - Placeholder and error states
    - Responsive resize handling
    """

    def __init__(self, parent, chart_type: str = 'line',
                 title: str = "", figsize: Tuple[int, int] = (10, 6),
                 style: str = 'default', **kwargs):
        """
        Initialize ChartWidgetBase.

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

        # Enhanced QA colors optimized for manufacturing/aerospace documentation
        # Print-friendly colors with B&W compatibility
        self.qa_colors = {
            # Status colors (primary indicators)
            'pass': '#2E8B57',      # Sea Green - works in B&W as medium gray
            'fail': '#CD5C5C',      # Indian Red - works in B&W as dark gray
            'warning': '#FF8C00',   # Dark Orange - works in B&W as medium-dark gray
            'critical': '#8B0000',  # Dark Red - works in B&W as very dark gray

            # Technical measurement colors
            'primary': '#4169E1',   # Royal Blue - good B&W contrast
            'secondary': '#663399', # Rebecca Purple - medium dark in B&W
            'tertiary': '#708090',  # Slate Gray - consistent in B&W

            # Control chart specific colors
            'control_center': '#2F4F4F',  # Dark Slate Gray - always visible
            'control_limits': '#CD853F',  # Peru - distinct from other colors
            'spec_limits': '#B22222',     # Fire Brick - critical boundaries
            'trend_line': '#000080',      # Navy - strong contrast

            # Data point states
            'in_control': '#228B22',      # Forest Green
            'out_of_control': '#DC143C',  # Crimson
            'warning_zone': '#DAA520',    # Goldenrod

            # Background and grid
            'grid_major': '#D3D3D3',      # Light Gray
            'grid_minor': '#F5F5F5',      # White Smoke
            'background': '#FFFFFF',      # White
            'background_alt': '#F8F8FF',  # Ghost White

            # Additional colors for quality charts
            'good': '#2E8B57',
            'bad': '#CD5C5C',
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

        # Determine position with better spacing
        if position == 'top':
            x, y = 0.5, 0.95  # Moved down to avoid overlap with title
        elif position == 'bottom':
            x, y = 0.5, 0.05  # Moved up to avoid overlap with x-axis labels
            default_props['va'] = 'bottom'
        elif position == 'right':
            x, y = 0.95, 0.5  # Moved left to avoid edge
            default_props['ha'] = 'right'
            default_props['va'] = 'center'
        elif isinstance(position, tuple) and len(position) == 2:
            x, y = position
        else:
            x, y = 0.5, 0.95  # Default to top with spacing

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
                color = theme_colors["fg"].get("tertiary", "#808080")
                style = '-'

            # Add line
            if orientation == 'horizontal':
                ax.axhline(y=value, color=color, linestyle=style,
                           alpha=0.7, linewidth=1.5, label=f'{label}: {value}')
            else:
                ax.axvline(x=value, color=color, linestyle=style,
                           alpha=0.7, linewidth=1.5, label=f'{label}: {value}')

    def _apply_theme_to_axes(self, ax):
        """Apply theme colors to matplotlib axes with enhanced readability."""
        # Get current theme colors
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"

        # Set background based on theme
        if is_dark:
            ax.set_facecolor(theme_colors["bg"]["secondary"])
            text_color = theme_colors["fg"]["primary"]
            grid_color = theme_colors["border"]["primary"]
            spine_alpha = 0.7
        else:
            ax.set_facecolor("#ffffff")
            text_color = theme_colors["fg"]["primary"]
            grid_color = theme_colors["border"]["primary"]
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

        # Control tick density to prevent MAXTICKS warnings
        max_ticks_x = 15 if self.figsize[0] > 10 else 10
        max_ticks_y = 12 if self.figsize[1] > 8 else 8

        # Apply MaxNLocator to both axes
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks_x, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks_y, prune='both'))

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
            self.figure.patch.set_facecolor(theme_colors["bg"]["primary"])
            if hasattr(self, 'canvas'):
                self.canvas.get_tk_widget().configure(bg=theme_colors["bg"]["primary"])
        else:
            self.figure.patch.set_facecolor(theme_colors["bg"]["secondary"])
            if hasattr(self, 'canvas'):
                self.canvas.get_tk_widget().configure(bg=theme_colors["bg"]["secondary"])

    def _get_or_create_axes(self):
        """Get existing axes or create new one with theme applied."""
        # Clear figure if we're transitioning from placeholder to data
        if not self._has_data and self.figure.axes:
            self.figure.clear()
            self._update_figure_theme()

        if not self.figure.axes:
            ax = self.figure.add_subplot(111)
            self._apply_theme_to_axes(ax)
        else:
            ax = self.figure.axes[0]
            ax.clear()
            self._apply_theme_to_axes(ax)
        return ax

    def _setup_ui(self):
        """Set up the chart widget UI."""
        # Get theme colors first
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        fig_bg = theme_colors["bg"]["primary"] if is_dark else theme_colors["bg"]["secondary"]

        # Create matplotlib figure with theme-aware colors
        self.figure = Figure(figsize=self.figsize, dpi=100, facecolor=fig_bg, edgecolor='none')
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
        export_btn = ctk.CTkButton(toolbar_frame, text="Export",
                                   command=self._export_chart, width=80, height=25)
        export_btn.pack(side='left', padx=2, pady=2)

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
        self._theme_check_scheduled = False
        self._shutting_down = False

        # Start checking for theme changes
        self._schedule_theme_check()

    def clear_chart(self):
        """Clear the current chart."""
        self.show_placeholder(self._placeholder_message, self._placeholder_instruction)

    def show_placeholder(self, message: str = "No data to display", instruction: str = ""):
        """Show a placeholder with custom message."""
        self._has_data = False
        self.figure.clear()
        self._update_figure_theme()

        ax = self.figure.add_subplot(111)
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])

        # Create centered text with icon
        text_parts = ['[Chart]']
        text_parts.append('')
        text_parts.append(message)
        if instruction:
            text_parts.append('')
            text_parts.append(instruction)

        text_color = theme_colors["fg"]["tertiary"]

        ax.text(0.5, 0.5, '\n'.join(text_parts),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=14,
                color=text_color,
                alpha=0.8,
                linespacing=2)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()

    def show_loading(self):
        """Show loading state in the chart area."""
        self.clear_chart()
        self._update_figure_theme()

        ax = self.figure.add_subplot(111)
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])

        text_color = theme_colors["fg"]["tertiary"]
        ax.text(0.5, 0.5, 'Loading chart data...',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=16,
                color=text_color,
                alpha=0.8)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()

    def show_error(self, error_title: str, error_message: str):
        """Show error state in the chart area with user-friendly message."""
        self.clear_chart()
        self._update_figure_theme()

        ax = self.figure.add_subplot(111)
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        ax.set_facecolor(theme_colors["bg"]["secondary" if is_dark else "primary"])

        import textwrap
        text_parts = ['Warning: ' + error_title]
        text_parts.append('')
        wrapped_message = textwrap.fill(error_message, width=60)
        text_parts.extend(wrapped_message.split('\n'))

        ax.text(0.5, 0.55, text_parts[0],
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=14,
                color='#e74c3c',
                weight='bold')

        if len(text_parts) > 1:
            ax.text(0.5, 0.45, '\n'.join(text_parts[1:]),
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes,
                    fontsize=11,
                    color=theme_colors["fg"]["secondary"],
                    alpha=0.9,
                    linespacing=1.5)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()

    def refresh_theme(self):
        """Refresh chart colors when theme changes."""
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"

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

        self._update_figure_theme()

        if self._has_data and hasattr(self, '_last_data') and self._last_data is not None:
            self.update_chart_data(self._last_data)
        else:
            self.show_placeholder(self._placeholder_message, self._placeholder_instruction)

        for ax in self.figure.axes:
            self._apply_theme_to_axes(ax)

        self.canvas.draw()

    def _on_resize(self, event):
        """Handle widget resize for responsive design."""
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = None

        if self._resize_timer:
            self.after_cancel(self._resize_timer)

        self._resize_timer = self.after(300, self._handle_resize, event)

    def _handle_resize(self, event):
        """Handle the actual resize after timer delay."""
        width = event.width
        height = event.height

        if width < 100 or height < 100:
            return

        dpi = self.figure.dpi
        new_width_inches = max(4, width / dpi)
        new_height_inches = max(3, height / dpi)

        old_width, old_height = self.figure.get_size_inches()
        if abs(new_width_inches - old_width) / old_width > 0.1 or abs(new_height_inches - old_height) / old_height > 0.1:
            self.figure.set_size_inches(new_width_inches, new_height_inches)
            self._adjust_font_sizes(new_width_inches, new_height_inches)
            self.canvas.draw_idle()

    def _adjust_font_sizes(self, width_inches, height_inches):
        """Adjust font sizes based on figure size for responsive design."""
        base_width = 8
        scale_factor = min(width_inches / base_width, 1.5)
        scale_factor = max(scale_factor, 0.7)

        for ax in self.figure.axes:
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=14 * scale_factor)

            ax.xaxis.label.set_fontsize(12 * scale_factor)
            ax.yaxis.label.set_fontsize(12 * scale_factor)
            ax.tick_params(axis='both', labelsize=10 * scale_factor)

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

        if hasattr(self, '_last_data') and self._last_data is not None:
            self.update_chart_data(self._last_data)

    def _style_toolbar(self):
        """Apply theme styling to the matplotlib navigation toolbar."""
        theme_colors = ThemeHelper.get_theme_colors()
        is_dark = ctk.get_appearance_mode().lower() == "dark"

        if is_dark:
            toolbar_bg = "#2b2b2b"
            toolbar_fg = "#ffffff"
            button_bg = "#3c3c3c"
            button_active_bg = "#4a4a4a"
        else:
            toolbar_bg = "#f0f0f0"
            toolbar_fg = "#000000"
            button_bg = "#ffffff"
            button_active_bg = "#e0e0e0"

        try:
            self.toolbar.configure(bg=toolbar_bg)
            self.toolbar._message_label.configure(bg=toolbar_bg, fg=toolbar_fg)
        except:
            pass

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

    def _schedule_theme_check(self):
        """Schedule theme change checking."""
        if not self._shutting_down and not self._theme_check_scheduled:
            self._theme_check_scheduled = True
            try:
                self.after(1000, self._check_theme_change)
            except:
                self._theme_check_scheduled = False

    def _check_theme_change(self):
        """Check if theme has changed and update if necessary."""
        self._theme_check_scheduled = False

        if self._shutting_down:
            return

        try:
            current_theme = ctk.get_appearance_mode()
            if current_theme != self._current_theme:
                self._current_theme = current_theme
                self._on_theme_change()
        except:
            pass

        self._schedule_theme_check()

    def cleanup(self):
        """Cleanup method to prevent after() callback errors."""
        self._shutting_down = True
        self._theme_check_scheduled = False

    def _on_theme_change(self):
        """Handle theme change events."""
        self._update_figure_theme()
        self._style_toolbar()

        if hasattr(self, '_last_data') and self._last_data is not None:
            self.update_chart_data(self._last_data)
        else:
            self.canvas.draw_idle()

    def _cleanup(self):
        """Clean up resources when widget is destroyed."""
        try:
            if hasattr(self, 'figure') and self.figure is not None:
                plt.close(self.figure)
                self.figure = None

            if hasattr(self, 'canvas') and self.canvas is not None:
                try:
                    self.canvas.get_tk_widget().destroy()
                except:
                    pass
                self.canvas = None

            if hasattr(self, 'toolbar'):
                self.toolbar = None

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during chart widget cleanup: {e}")
            pass

    def destroy(self):
        """Override destroy to ensure proper cleanup."""
        self._cleanup()
        super().destroy()

    def _export_chart(self):
        """Export chart to file."""
        filetypes = [
            ('PNG files', '*.png'),
            ('PDF files', '*.pdf'),
            ('SVG files', '*.svg'),
            ('All files', '*.*')
        ]

        parent = self.winfo_toplevel()
        filename = filedialog.asksaveasfilename(
            parent=parent,
            defaultextension='.png',
            filetypes=filetypes,
            title='Export Chart'
        )

        if filename:
            try:
                from pathlib import Path
                Path(filename).parent.mkdir(parents=True, exist_ok=True)

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
