"""
AnalyticsChartMixin - Advanced analytics charts for ChartWidget.

This module provides advanced analytics chart methods:
- plot_early_warning_system: Control charts with violation detection
- plot_failure_pattern_analysis: Pattern analysis for failures
- plot_performance_scorecard: Performance metrics scorecard
- plot_enhanced_control_chart: SPC control charts
- plot_process_capability_histogram: Capability analysis

Migrated from chart_widget.py lines 2848-4173 during Phase 4 file splitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import customtkinter as ctk
from typing import Any, Dict, List, Optional, Tuple
import logging

from laser_trim_analyzer.gui.theme_helper import ThemeHelper

logger = logging.getLogger(__name__)


class AnalyticsChartMixin:
    """
    Mixin providing advanced analytics chart methods.

    Requires ChartWidgetBase as parent class.
    """

    def plot_early_warning_system(self, data: pd.DataFrame):
        """
        Create an early warning system with moving range and CUSUM charts.

        Validates data and creates a comprehensive dashboard with:
        - Control chart with violation detection
        - Moving range chart for variation detection
        - CUSUM shift detection indicator

        Args:
            data: DataFrame with 'trim_date' and 'sigma_gradient' columns

        Returns:
            True if successful, None if validation fails
        """
        # Data validation - check for empty or None data
        if data is None:
            self.show_placeholder("No Data", "Cannot create early warning system without data")
            return None

        if not isinstance(data, pd.DataFrame):
            self.show_error("Invalid Data Type", "Data must be a pandas DataFrame")
            return None

        if data.empty:
            self.show_placeholder("Empty Data", "DataFrame has no rows")
            return None

        # Check for required columns
        required_columns = ['trim_date', 'sigma_gradient']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.show_error(
                "Missing Columns",
                f"DataFrame must have columns: {', '.join(required_columns)}\nMissing: {', '.join(missing_columns)}"
            )
            return None

        # Check minimum data points for moving range (need at least 2)
        if len(data) < 2:
            self.show_placeholder(
                "Insufficient Data",
                "Need at least 2 data points for early warning system (moving range requires 2+ points)"
            )
            return None

        try:
            self.figure.clear()
            self._has_data = True

            # Get theme colors
            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]

            # Create subplots with better spacing
            fig = self.figure
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1.5, 0.5], hspace=0.4)

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

            # Plot control limits with shorter labels
            ax1.axhline(ucl, color='red', linestyle='--', alpha=0.5, label=f'UCL: {ucl:.2f}')
            ax1.axhline(uwl, color='orange', linestyle=':', alpha=0.5, label=f'UWL: {uwl:.2f}')
            ax1.axhline(mean_val, color='green', linestyle='-', alpha=0.5, label=f'Mean: {mean_val:.2f}')
            ax1.axhline(lwl, color='orange', linestyle=':', alpha=0.5, label=f'LWL: {lwl:.2f}')
            ax1.axhline(lcl, color='red', linestyle='--', alpha=0.5, label=f'LCL: {lcl:.2f}')

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
            ax1.set_title('Control Chart with Violation Detection', fontsize=12, color=text_color, pad=15)
            # Place legend outside plot area to avoid overlap
            ax1.legend(loc='center left', fontsize=6, ncol=2, bbox_to_anchor=(1.02, 0.5), framealpha=0.9)
            ax1.grid(True, alpha=0.3)

            # Format dates with better spacing
            date_range = (dates.max() - dates.min()).days if hasattr(dates.max() - dates.min(), 'days') else 30
            if date_range > 60:
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
                ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            elif date_range > 30:
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
                ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            else:
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, date_range // 10)))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

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

            ax2.set_ylabel('Moving Range', fontsize=9)
            ax2.set_title('Moving Range Chart - Variation Detection', fontsize=11, color=text_color, pad=10)
            ax2.legend(loc='upper left', fontsize=7, bbox_to_anchor=(0.02, 0.98))
            ax2.grid(True, alpha=0.3)

            # Format dates with better spacing
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

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
            fig.suptitle('Early Warning System Dashboard', fontsize=14, color=text_color, y=0.98)

            # Apply tight layout to prevent overlaps
            plt.tight_layout(rect=[0, 0, 0.85, 0.96])  # Leave space for legend on right and title on top

            # Layout is handled by constrained_layout
            self.canvas.draw_idle()

            return True

        except Exception as e:
            # Surface any plotting errors with clear message
            self.show_error("Early Warning System Error", f"Failed to create early warning system: {str(e)}")
            return None

    def plot_failure_pattern_analysis(self, data: pd.DataFrame):
        """
        Create failure pattern analysis with heat map, Pareto chart, and projection.

        Validates data and creates comprehensive failure analysis with:
        - Time-based heat map showing failure patterns by week
        - Pareto chart identifying top failure types
        - Failure rate trend with 7-day projection

        Args:
            data: DataFrame with columns including 'trim_date', 'track_status',
                  'sigma_gradient', 'linearity_pass', etc.

        Returns:
            True if successful, None if validation fails
        """
        # Data validation - check for empty or None data
        if data is None:
            self.show_placeholder("No Data", "Cannot create failure pattern analysis without data")
            return None

        if not isinstance(data, pd.DataFrame):
            self.show_error("Invalid Data Type", "Data must be a pandas DataFrame")
            return None

        if data.empty:
            self.show_placeholder("Empty Data", "DataFrame has no rows")
            return None

        # Check for required columns
        required_columns = ['trim_date', 'track_status']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.show_error(
                "Missing Columns",
                f"DataFrame must have columns: {', '.join(required_columns)}\nMissing: {', '.join(missing_columns)}"
            )
            return None

        # Check minimum data points for meaningful analysis (need at least 4 for projection)
        if len(data) < 2:
            self.show_placeholder(
                "Insufficient Data",
                "Need at least 2 data points for failure pattern analysis"
            )
            return None

        try:
            self.figure.clear()
            self._has_data = True

            # Get theme colors
            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]

            # Create subplots with better spacing
            fig = self.figure
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1], hspace=0.4, wspace=0.35)

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

                    ax1.set_title('Failure Pattern Heat Map (by Week)', fontsize=10, color=text_color, pad=10)
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
                ax2.set_title('Failure Pareto Chart', fontsize=10, color=text_color, pad=10)
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
                ax3.set_title('Failure Rate Trend & 7-Day Projection', fontsize=10, color=text_color, pad=10)
                ax3.legend(loc='upper left', fontsize=8)
                ax3.grid(True, alpha=0.3)

                # Format dates
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
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

            return True

        except Exception as e:
            # Surface any plotting errors with clear message
            self.show_error("Failure Pattern Analysis Error", f"Failed to create failure pattern analysis: {str(e)}")
            return None

    def plot_performance_scorecard(self, data: pd.DataFrame):
        """
        Create a performance scorecard with quality score, yield/efficiency, and comparisons.

        Validates data and creates comprehensive performance dashboard with:
        - Quality score timeline with trending
        - Yield and efficiency dual-axis chart
        - Comparative performance table across time periods

        Args:
            data: DataFrame with performance data including dates, pass rates, sigma values, etc.

        Returns:
            True if successful, None if validation fails
        """
        # Data validation - check for empty or None data
        if data is None:
            self.show_placeholder("No Data", "Cannot create performance scorecard without data")
            return None

        if not isinstance(data, pd.DataFrame):
            self.show_error("Invalid Data Type", "Data must be a pandas DataFrame")
            return None

        if data.empty:
            self.show_placeholder("Empty Data", "DataFrame has no rows")
            return None

        # Check for required columns
        required_columns = ['trim_date', 'track_status', 'sigma_gradient', 'linearity_pass']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.show_error(
                "Missing Columns",
                f"DataFrame must have columns: {', '.join(required_columns)}\nMissing: {', '.join(missing_columns)}"
            )
            return None

        # Check minimum data points for meaningful analysis
        if len(data) < 2:
            self.show_placeholder(
                "Insufficient Data",
                "Need at least 2 data points for performance scorecard"
            )
            return None

        try:
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
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
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
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
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

            return True

        except Exception as e:
            # Surface any plotting errors with clear message
            self.show_error("Performance Scorecard Error", f"Failed to create performance scorecard: {str(e)}")
            return None

    def plot_enhanced_control_chart(self, data: pd.DataFrame, value_column: str,
                                    date_column: str = 'trim_date',
                                    spec_limits: Optional[Tuple[float, float]] = None,
                                    target_value: Optional[float] = None,
                                    title: str = "Control Chart"):
        """
        Simplified control chart with moving averages for trend smoothing.

        Args:
            data: DataFrame with time series data
            value_column: Column name containing measured values
            date_column: Column name containing dates
            spec_limits: Tuple of (LSL, USL) specification limits
            target_value: Target/nominal value
            title: Chart title
        """
        # Defensive gating and error surfaces
        try:
            if data is None or (hasattr(data, 'empty') and data.empty):
                self.show_placeholder("No Data", "No rows available for control chart")
                return

            df = data.copy()
            missing_cols = [c for c in (value_column, date_column) if c not in df.columns]
            if missing_cols:
                self.show_error("Missing Columns", f"Required columns not found: {', '.join(missing_cols)}")
                return

            # Prepare data
            # Convert and drop invalid rows without substituting current time
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df = df[[date_column, value_column]].dropna()
            if df.empty or len(df) < 3:
                self.show_placeholder("Insufficient Data", "Need at least 3 data points for control chart")
                return

            df = df.sort_values(date_column)
            values = df[value_column]
            dates = df[date_column]

            # Ready to draw
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            self._apply_theme_to_axes(ax)
            self._has_data = True

            # Group by date for daily averages if many points
            if len(values) > 100:
                df_daily = df.groupby(df[date_column].dt.date)[value_column].mean().reset_index()
                df_daily['date'] = pd.to_datetime(df_daily[date_column])
                dates = df_daily['date']
                values = df_daily[value_column]

            # Calculate moving averages for smoothing
            values_series = pd.Series(values.values, index=dates)
            ma_7 = values_series.rolling(window=7, min_periods=3).mean()
            ma_30 = values_series.rolling(window=30, min_periods=15).mean() if len(values) >= 30 else None

            # Individuals chart control limits (3-sigma) using MR-based sigma estimate when possible
            mean_val = values.mean()
            vals = values.values.astype(float)
            sigma_est = values.std(ddof=1) if len(vals) >= 2 else 0.0
            if len(vals) >= 2:
                mr = np.abs(np.diff(vals))
                if len(mr) > 0 and np.isfinite(mr).all():
                    mrbar = float(np.mean(mr))
                    d2 = 1.128  # for n=2 in moving range
                    if mrbar > 0:
                        sigma_est = mrbar / d2
            # 3-sigma limits
            ucl = mean_val + 3 * sigma_est
            lcl = mean_val - 3 * sigma_est

            # Plot raw data lightly
            ax.plot(dates, values, '-', linewidth=0.5, color=self.qa_colors['primary'],
                   alpha=0.3, label='Raw Data')

            # Plot moving averages for trend (main focus)
            ax.plot(dates, ma_7, '-', linewidth=2, color=self.qa_colors['secondary'],
                   alpha=0.9, label='7-day Trend')

            if ma_30 is not None:
                ax.plot(dates, ma_30, '-', linewidth=3, color=self.qa_colors['tertiary'],
                       alpha=0.8, label='30-day Trend')

            # Add reference lines: mean and control limits (3σ)
            ax.axhline(y=mean_val, color=self.qa_colors['control_center'], linestyle='-',
                      linewidth=2, alpha=0.9, label=f'Mean (X̄): {mean_val:.3f}')
            # Show control limits if variance non-zero
            if sigma_est and np.isfinite(sigma_est):
                ax.axhline(y=ucl, color=self.qa_colors.get('control_limits', self.qa_colors['warning']), linestyle='--',
                          linewidth=1.8, alpha=0.8, label=f'UCL (3σ): {ucl:.3f}')
                ax.axhline(y=lcl, color=self.qa_colors.get('control_limits', self.qa_colors['warning']), linestyle='--',
                          linewidth=1.8, alpha=0.8, label=f'LCL (3σ): {lcl:.3f}')

            # Add target line if provided (engineering target)
            if target_value is not None:
                ax.axhline(y=target_value, color=self.qa_colors['pass'], linestyle=':',
                          linewidth=2, alpha=0.8, label=f'Target: {target_value:.3f}')

            # Add specification limits if provided (engineering spec), visually distinct from control limits
            if spec_limits is not None:
                try:
                    lsl, usl = spec_limits
                    ax.axhline(y=lsl, color=self.qa_colors.get('spec_limits', '#B22222'), linestyle='-.',
                              linewidth=1.6, alpha=0.9, label=f'LSL: {lsl:.3f}')
                    ax.axhline(y=usl, color=self.qa_colors.get('spec_limits', '#B22222'), linestyle='-.',
                              linewidth=1.6, alpha=0.9, label=f'USL: {usl:.3f}')
                except Exception:
                    pass

            # Clean formatting
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel(value_column.replace('_', ' ').title(), fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')

            # Simple legend with capped items to prevent clutter
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                # Keep unique labels in order and cap at 6
                uniq = []
                for h, l in zip(handles, labels):
                    if l not in [ul for _, ul in uniq]:
                        uniq.append((h, l))
                uniq = uniq[:6]
                ax.legend([h for h, _ in uniq], [l for _, l in uniq], loc='upper right', fontsize=9, framealpha=0.9)

            # Clean grid and layout
            ax.grid(True, alpha=0.3)

            # Simple date formatting with full year
            if len(dates) > 30:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            self.figure.tight_layout()
            self.canvas.draw_idle()
        except Exception as e:
            # Surface any unexpected error as a chart error instead of blank canvas
            self.show_error("Control Chart Error", str(e))

    def plot_process_capability_histogram(self, data: pd.DataFrame, value_column: str,
                                        spec_limits: Optional[Tuple[float, float]] = None,
                                        target_value: Optional[float] = None,
                                        title: str = "Process Distribution"):
        """
        Simplified histogram showing process distribution.

        Args:
            data: DataFrame with process data
            value_column: Column name containing measured values
            spec_limits: Tuple of (LSL, USL) specification limits
            target_value: Target/nominal value
            title: Chart title
        """
        # Defensive gating and error surfaces
        try:
            if data is None or (hasattr(data, 'empty') and data.empty):
                self.show_placeholder("No Data", "No rows available for histogram")
                return

            df = data.copy()
            if value_column not in df.columns:
                self.show_error("Missing Column", f"Required column not found: {value_column}")
                return

            values = df[value_column].dropna()
            if len(values) < 10:
                self.show_placeholder("Insufficient Data", "Need at least 10 data points for capability histogram")
                return

            # Ready to draw
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            self._apply_theme_to_axes(ax)
            self._has_data = True

            # Calculate basic statistics
            mean_val = values.mean()
            std_val = values.std()

            # Simple histogram
            n_bins = min(20, int(np.sqrt(len(values))))
            ax.hist(values, bins=n_bins, density=True,
                   color=self.qa_colors['primary'], alpha=0.6,
                   edgecolor='white', linewidth=1)

            # Add normal curve for reference
            x_range = np.linspace(values.min(), values.max(), 100)
            normal_curve = (1 / (std_val * np.sqrt(2 * np.pi))) * \
                          np.exp(-0.5 * ((x_range - mean_val) / std_val) ** 2)
            ax.plot(x_range, normal_curve, color=self.qa_colors['secondary'],
                   linewidth=2, alpha=0.8, label='Normal Fit')

            # Add mean line
            ax.axvline(mean_val, color=self.qa_colors['control_center'],
                      linestyle='-', linewidth=2, alpha=0.8,
                      label=f'Average ({mean_val:.3f})')

            # Add target if provided
            if target_value:
                ax.axvline(target_value, color=self.qa_colors['pass'],
                          linestyle='--', linewidth=2, alpha=0.8,
                          label=f'Target ({target_value:.3f})')

            # Add spec limits if provided (simplified)
            if spec_limits:
                lsl, usl = spec_limits
                ax.axvline(lsl, color=self.qa_colors['warning'],
                          linestyle=':', linewidth=2, alpha=0.7, label='Lower Limit')
                ax.axvline(usl, color=self.qa_colors['warning'],
                          linestyle=':', linewidth=2, alpha=0.7, label='Upper Limit')

            # Formatting
            ax.set_xlabel(value_column.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')

            # Simple legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper right', fontsize=9, framealpha=0.9)

            ax.grid(True, alpha=0.3)

            self.figure.tight_layout()
            self.canvas.draw_idle()
        except Exception as e:
            self.show_error("Capability Chart Error", str(e))
