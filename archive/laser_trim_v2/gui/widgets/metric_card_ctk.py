"""
MetricCard Widget for QA Dashboard - CustomTkinter Version

A modern metric display card showing value, trend, and sparkline visualization.
Perfect for displaying key QA metrics like pass rates, sigma values, etc.
"""

import customtkinter as ctk
from typing import Callable, Dict, List, Optional, Tuple
import math


class SparkLine(ctk.CTkCanvas):
    """Simple sparkline chart for metric visualization."""

    def __init__(self, parent, width=80, height=30, **kwargs):
        # Set background to match theme
        kwargs['highlightthickness'] = 0
        # Get theme-appropriate background
        current_theme = ctk.get_appearance_mode()
        if current_theme == "Dark":
            kwargs['bg'] = '#2b2b2b'  # Dark theme background
        else:
            kwargs['bg'] = '#dbdbdb'  # Light theme background
        super().__init__(parent, width=width, height=height, **kwargs)
        self.data_points = []
        self.line_color = '#3498db'

    def set_data(self, values: List[float], color: str = '#3498db'):
        """Set sparkline data points."""
        self.data_points = values
        self.line_color = color
        self._draw_sparkline()

    def _draw_sparkline(self):
        """Draw the sparkline chart."""
        self.delete("all")

        if len(self.data_points) < 2:
            return

        # Calculate dimensions
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        padding = 2

        # Normalize data
        min_val = min(self.data_points)
        max_val = max(self.data_points)
        range_val = max_val - min_val if max_val != min_val else 1

        # Calculate points
        x_step = (width - 2 * padding) / (len(self.data_points) - 1)
        points = []

        for i, value in enumerate(self.data_points):
            x = padding + i * x_step
            y = height - padding - ((value - min_val) / range_val) * (height - 2 * padding)
            points.extend([x, y])

        # Draw line
        if len(points) >= 4:
            self.create_line(points, fill=self.line_color, width=2, smooth=True)

            # Draw end point
            self.create_oval(points[-2] - 2, points[-1] - 2,
                             points[-2] + 2, points[-1] + 2,
                             fill=self.line_color, outline='')


class MetricCard(ctk.CTkFrame):
    """
    Modern metric display card with value, trend, and sparkline.

    Features:
    - Large metric value display
    - Trend arrow indicator
    - Sparkline visualization
    - Color-coded based on thresholds
    - Click for detailed view
    """

    def __init__(self, parent, title: str = "Metric",
                 value: float = 0.0, unit: str = "",
                 thresholds: Optional[dict] = None,
                 show_sparkline: bool = True,
                 on_click: Optional[Callable] = None,
                 color_scheme: str = "neutral",
                 status: str = None,  # For backward compatibility
                 **kwargs):
        """
        Initialize metric card.

        Args:
            parent: Parent widget
            title: Card title
            value: Initial value
            unit: Unit of measurement (e.g., '%', 'mm')
            thresholds: Dict with 'good', 'warning', 'critical' values
            show_sparkline: Whether to show sparkline
            on_click: Callback when card is clicked
            color_scheme: Color scheme to use ("neutral", "success", "warning", "danger", "info")
            status: Alias for color_scheme (for backward compatibility)
        """
        # Handle status parameter as alias for color_scheme
        if status is not None:
            color_scheme = status
            
        # Filter out parameters that shouldn't be passed to CTkFrame
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['color_scheme', 'status']}
        
        super().__init__(parent, **filtered_kwargs)

        self.title = title
        self.value = value
        self.unit = unit
        self.thresholds = thresholds or {'good': 95, 'warning': 90, 'critical': 80}
        self.show_sparkline = show_sparkline
        self.on_click = on_click
        self.color_scheme = color_scheme
        self.historical_values = []

        # Colors - adjust for theme
        # Get current theme to adjust text colors
        current_theme = ctk.get_appearance_mode()
        is_dark = current_theme == "Dark"
        
        self.colors = {
            'good': '#27ae60',
            'warning': '#f39c12',
            'critical': '#e74c3c',
            'neutral': '#3498db',
            'success': '#27ae60',
            'danger': '#e74c3c',
            'info': '#3498db',
            'bg_light': '#ecf0f1',
            'text_dark': '#ffffff' if is_dark else '#2c3e50',
            'text_light': '#b0b0b0' if is_dark else '#7f8c8d'
        }

        self._setup_ui()
        self._update_display()

    def _setup_ui(self):
        """Set up the card UI."""
        # Configure the frame - ensure no white background
        self.configure(corner_radius=10, fg_color=("gray90", "gray20"))

        # Use self as the main container instead of creating nested frame
        main_frame = self

        # Title
        self.title_label = ctk.CTkLabel(
            main_frame, 
            text=self.title,
            font=ctk.CTkFont(size=12),
            text_color=self.colors['text_light']
        )
        self.title_label.pack(anchor='w', padx=10, pady=(10, 0))

        # Value and trend container
        value_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        value_frame.pack(fill='x', pady=(5, 0), padx=10)

        # Value display
        self.value_label = ctk.CTkLabel(
            value_frame, 
            text="0",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors['text_dark']
        )
        self.value_label.pack(side='left')

        # Unit
        if self.unit:
            self.unit_label = ctk.CTkLabel(
                value_frame, 
                text=self.unit,
                font=ctk.CTkFont(size=12),
                text_color=self.colors['text_light']
            )
            self.unit_label.pack(side='left', padx=(2, 0))

        # Trend arrow
        self.trend_label = ctk.CTkLabel(
            value_frame, 
            text="",
            font=ctk.CTkFont(size=16),
            text_color=self.colors['neutral']
        )
        self.trend_label.pack(side='left', padx=(10, 0))

        # Sparkline
        if self.show_sparkline:
            self.sparkline = SparkLine(main_frame, width=100, height=30)
            self.sparkline.pack(fill='x', padx=10, pady=(10, 10))

        # Bind click event
        if self.on_click:
            main_frame.bind("<Button-1>", lambda e: self.on_click())
            self.bind("<Button-1>", lambda e: self.on_click())

    def set_value(self, value: float, add_to_history: bool = True):
        """Update the metric value."""
        old_value = self.value
        self.value = value

        if add_to_history:
            # Only add numeric values to history
            try:
                numeric_value = float(value)
                self.historical_values.append(numeric_value)
                # Keep last 20 values
                if len(self.historical_values) > 20:
                    self.historical_values.pop(0)
            except (ValueError, TypeError):
                # Skip non-numeric values
                pass

        self._update_display()
        self._update_trend(old_value, value)

    def set_historical_data(self, values: List[float]):
        """Set historical data for sparkline."""
        self.historical_values = values[-20:]  # Keep last 20
        self._update_sparkline()

    def _update_display(self):
        """Update the display based on current value."""
        # Format value
        if isinstance(self.value, float):
            if self.value >= 100 or self.value <= -100:
                formatted_value = f"{self.value:.0f}"
            elif self.value >= 10 or self.value <= -10:
                formatted_value = f"{self.value:.1f}"
            else:
                formatted_value = f"{self.value:.2f}"
        else:
            formatted_value = str(self.value)

        self.value_label.configure(text=formatted_value)

        # Update color based on thresholds
        color = self._get_value_color()
        self.value_label.configure(text_color=color)

        # Update sparkline
        if self.show_sparkline and self.historical_values:
            self._update_sparkline()

    def _get_value_color(self) -> str:
        """Get color based on value and thresholds."""
        # Use explicit color scheme if it's not neutral or if there are no thresholds
        if self.color_scheme != "neutral" or not self.thresholds:
            return self.colors.get(self.color_scheme, self.colors['neutral'])
            
        # Check if value is numeric for threshold-based coloring
        try:
            numeric_value = float(self.value)
        except (ValueError, TypeError):
            # For non-numeric values, use color scheme
            return self.colors.get(self.color_scheme, self.colors['neutral'])

        # Determine if higher or lower is better based on threshold order
        if self.thresholds.get('good', 0) > self.thresholds.get('critical', 0):
            # Higher is better (e.g., pass rate)
            if numeric_value >= self.thresholds.get('good', 95):
                return self.colors['good']
            elif numeric_value >= self.thresholds.get('warning', 90):
                return self.colors['warning']
            else:
                return self.colors['critical']
        else:
            # Lower is better (e.g., failure rate)
            if numeric_value <= self.thresholds.get('good', 5):
                return self.colors['good']
            elif numeric_value <= self.thresholds.get('warning', 10):
                return self.colors['warning']
            else:
                return self.colors['critical']

    def _update_trend(self, old_value: float, new_value: float):
        """Update trend indicator."""
        # Check if values are numeric
        try:
            old_numeric = float(old_value)
            new_numeric = float(new_value)
        except (ValueError, TypeError):
            # For non-numeric values, don't show trend
            self.trend_label.configure(text="", text_color=self.colors['neutral'])
            return
            
        if old_numeric == new_numeric:
            self.trend_label.configure(text="→", text_color=self.colors['neutral'])
        elif new_numeric > old_numeric:
            # Check if increase is good or bad
            if self.thresholds.get('good', 0) > self.thresholds.get('critical', 0):
                # Higher is better
                self.trend_label.configure(text="↑", text_color=self.colors['good'])
            else:
                # Lower is better
                self.trend_label.configure(text="↑", text_color=self.colors['critical'])
        else:
            # Check if decrease is good or bad
            if self.thresholds.get('good', 0) > self.thresholds.get('critical', 0):
                # Higher is better
                self.trend_label.configure(text="↓", text_color=self.colors['critical'])
            else:
                # Lower is better
                self.trend_label.configure(text="↓", text_color=self.colors['good'])

    def _update_sparkline(self):
        """Update sparkline with current data."""
        if self.show_sparkline and hasattr(self, 'sparkline') and self.historical_values:
            color = self._get_value_color()
            self.sparkline.set_data(self.historical_values, color)

    def update_thresholds(self, thresholds: dict):
        """Update threshold values."""
        self.thresholds = thresholds
        self._update_display()

    def update_value(self, value, color_scheme: str = None):
        """Update the card value and optionally color scheme."""
        self.value = value
        if color_scheme:
            self.color_scheme = color_scheme
        self._update_display()

    def set_color_scheme(self, color_scheme: str):
        """Set the color scheme for the card."""
        self.color_scheme = color_scheme
        self._update_display()
    
    def update_theme_colors(self):
        """Update colors based on current theme."""
        current_theme = ctk.get_appearance_mode()
        is_dark = current_theme == "Dark"
        
        self.colors['text_dark'] = '#ffffff' if is_dark else '#2c3e50'
        self.colors['text_light'] = '#b0b0b0' if is_dark else '#7f8c8d'
        
        # Update sparkline background if it exists
        if hasattr(self, 'sparkline'):
            bg_color = '#2b2b2b' if is_dark else '#dbdbdb'
            self.sparkline.configure(bg=bg_color)
        
        # Update displayed colors
        self._update_display()