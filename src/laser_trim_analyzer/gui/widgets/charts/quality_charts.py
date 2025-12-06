"""
QualityChartMixin - Quality dashboard charts for ChartWidget.

This module provides quality-focused chart methods:
- plot_quality_dashboard: Traffic light dashboard with gauges
- plot_gauge: Gauge/meter charts
- plot_quality_dashboard_cards: KPI cards with sparklines

Migrated from chart_widget.py lines 2433-2847 during Phase 4 file splitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import customtkinter as ctk
from typing import Any, Dict, List, Optional, Tuple
import logging

from laser_trim_analyzer.gui.theme_helper import ThemeHelper

logger = logging.getLogger(__name__)


class QualityChartMixin:
    """
    Mixin providing quality dashboard chart methods.

    Requires ChartWidgetBase as parent class.
    """

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

        Returns:
            True if successful, None if validation fails
        """
        # Data validation - check for empty or None data
        if metrics is None:
            self.show_placeholder("No Metrics", "Cannot create dashboard without metrics data")
            return None

        if not isinstance(metrics, dict):
            self.show_error("Invalid Data Type", "Metrics must be a dictionary")
            return None

        if not metrics:
            self.show_placeholder("Empty Metrics", "Metrics dictionary is empty")
            return None

        # Validate that each metric has required fields
        for metric_name, metric_data in metrics.items():
            if not isinstance(metric_data, dict):
                self.show_error(
                    "Invalid Metric Format",
                    f"Metric '{metric_name}' must be a dictionary with 'value', 'status', etc."
                )
                return None

            # Check for required 'value' key
            if 'value' not in metric_data:
                self.show_error(
                    "Missing Value",
                    f"Metric '{metric_name}' is missing required 'value' key"
                )
                return None

        try:
            # Proceed with plotting
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

            return True

        except Exception as e:
            # Surface any plotting errors with clear message
            self.show_error("Quality Dashboard Error", f"Failed to create quality dashboard: {str(e)}")
            return None

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
        Create a gauge chart with data validation and error handling.

        Args:
            value: Current value
            min_val: Minimum value
            max_val: Maximum value
            target: Target value (optional)
            title: Gauge title
            zones: List of (start, end, color) tuples for colored zones

        Returns:
            True if successful, None if validation fails
        """
        # Data validation - check for None value
        if value is None:
            self.show_placeholder("No Value", "Cannot plot gauge without a value")
            return None

        # Validate min < max
        if min_val >= max_val:
            self.show_error(
                "Invalid Range",
                f"Minimum value ({min_val}) must be less than maximum value ({max_val})"
            )
            return None

        # Check if value is numeric
        try:
            value = float(value)
            min_val = float(min_val)
            max_val = float(max_val)
        except (TypeError, ValueError) as e:
            self.show_error(
                "Invalid Value Type",
                f"Value, min_val, and max_val must be numeric: {str(e)}"
            )
            return None

        # Validate target if provided
        if target is not None:
            try:
                target = float(target)
            except (TypeError, ValueError):
                self.show_error(
                    "Invalid Target",
                    "Target value must be numeric"
                )
                return None

        # Validate zones if provided
        if zones is not None:
            if not zones:
                self.show_error("Empty Zones", "Zones list is empty")
                return None

            for i, zone in enumerate(zones):
                if not isinstance(zone, (tuple, list)) or len(zone) != 3:
                    self.show_error(
                        "Invalid Zone Format",
                        f"Zone {i+1} must be a tuple/list of (start, end, color)"
                    )
                    return None

        try:
            # Proceed with plotting
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

            return True

        except Exception as e:
            # Surface any plotting errors with clear message
            self.show_error("Gauge Chart Error", f"Failed to create gauge chart: {str(e)}")
            return None

    def plot_quality_dashboard_cards(self, metrics: Dict[str, Dict]):
        """
        Create a quality health dashboard with KPI cards and sparklines.

        This is an alias for plot_quality_dashboard for backward compatibility.
        """
        return self.plot_quality_dashboard(metrics)
