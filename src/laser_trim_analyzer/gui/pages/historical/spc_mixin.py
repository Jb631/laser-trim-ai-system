"""
SPCMixin - Statistical Process Control methods for HistoricalPage.

This module provides SPC analysis functionality:
- _generate_all_spc_analyses: Run all SPC analyses
- _generate_control_charts: Enhanced control charts
- _run_capability_study: Process capability (Cp, Cpk)
- _run_pareto_analysis: Defect Pareto charts
- _detect_process_drift: ML-first drift detection with formula fallback
- _analyze_failure_modes: Failure mode analysis

Migrated from historical_page.py lines 3663-4411 during Phase 4 file splitting.
"""

import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
import logging
import customtkinter as ctk

logger = logging.getLogger(__name__)

# Import UnifiedProcessor for ML drift detection (Phase 3)
try:
    from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
    HAS_UNIFIED_PROCESSOR = True
except ImportError:
    HAS_UNIFIED_PROCESSOR = False
    logger.warning("UnifiedProcessor not available - using formula-based drift detection")

from laser_trim_analyzer.core.config import get_config


class SPCMixin:
    """
    Mixin providing Statistical Process Control (SPC) analysis methods.

    Requires HistoricalPage as parent class with:
    - self.current_data: List of analysis results
    - self.spc_tabview: Tab view widget for SPC charts
    - self.control_chart, self.capability_chart, etc.: Chart widgets
    - self.generate_spc_btn: Button widget
    - self.logger: Logger instance
    """

    def _generate_all_spc_analyses(self):
        """Generate all SPC analyses at once."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return

        try:
            # Show progress
            self.generate_spc_btn.configure(text="Generating Analyses...", state='disabled')

            # Generate all analyses
            self._generate_control_charts()
            self._run_capability_study()
            self._run_pareto_analysis()
            self._detect_process_drift()
            self._analyze_failure_modes()

            # Switch to first tab
            self.spc_tabview.set("Control Charts")

            # Show success message
            messagebox.showinfo("SPC Analysis Complete",
                              "All statistical analyses have been generated.\n"
                              "Use the tabs to view different analyses.")

        except Exception as e:
            self.logger.error(f"Error generating SPC analyses: {e}")
            messagebox.showerror("Analysis Error", f"Failed to generate analyses:\n{str(e)}")
        finally:
            # Reset button
            self.generate_spc_btn.configure(text="Generate All SPC Analyses", state='normal')

    def _generate_control_charts(self):
        """Generate statistical control charts for key parameters."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return

        try:
            # Switch to control charts tab
            self.spc_tabview.set("Control Charts")

            # Prepare data
            chart_data = []
            for result in self.current_data:
                if result.tracks:
                    for track in result.tracks:
                        if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                            chart_data.append({
                                'date': result.file_date or result.timestamp,
                                'value': track.sigma_gradient,
                                'threshold': getattr(track, 'sigma_threshold', None),
                                'model': result.model
                            })

            if not chart_data:
                messagebox.showwarning("No Data", "No sigma gradient data available")
                return

            # Create enhanced control chart using the new method
            df = pd.DataFrame(chart_data).sort_values('date')

            # Rename columns to match expected format
            df = df.rename(columns={'date': 'trim_date', 'value': 'sigma_gradient', 'threshold': 'sigma_threshold'})

            # M3: Use model-specific sigma threshold (one-sided: LSL=0, USL=threshold)
            sigma_thresholds = df['sigma_threshold'].dropna()
            if len(sigma_thresholds) > 0:
                sigma_threshold_usl = float(sigma_thresholds.median())
                spec_limits = (0.0, sigma_threshold_usl)
                target_value = sigma_threshold_usl * 0.5
                self.logger.info(f"Historical trends using model-specific threshold USL: {sigma_threshold_usl:.4f}")
            else:
                # Fallback if threshold missing
                spec_limits = (0.0, 0.250)
                target_value = 0.125
                self.logger.warning("No sigma_threshold for historical trends, using fallback")

            # Use the enhanced control chart method
            self.control_chart.plot_enhanced_control_chart(
                data=df,
                value_column='sigma_gradient',
                date_column='trim_date',
                spec_limits=spec_limits,
                target_value=target_value,
                title="Historical Sigma Trends"
            )

            self.logger.info(f"Enhanced control chart created with {len(df)} data points")

        except Exception as e:
            self.logger.error(f"Error generating control charts: {e}")
            messagebox.showerror("Error", f"Failed to generate control charts:\n{str(e)}")

    def _run_capability_study(self):
        """Run process capability study."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return

        try:
            # Switch to capability tab
            self.spc_tabview.set("Process Capability")

            # Collect data
            sigma_values = []
            for i, result in enumerate(self.current_data):
                if result.tracks:
                    for track in result.tracks:
                        # Debug logging for sigma values
                        sigma_val = getattr(track, 'sigma_gradient', None)
                        if i == 0:  # Only log for first result to avoid spam
                            self.logger.debug(f"Track {getattr(track, 'track_id', 'unknown')}: sigma_gradient = {sigma_val}")
                        if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                            sigma_values.append(track.sigma_gradient)

            if len(sigma_values) < 30:
                messagebox.showwarning("Insufficient Data",
                                     f"Need at least 30 samples for capability study. Found: {len(sigma_values)}")
                return

            # Calculate statistics
            mean = np.mean(sigma_values)
            std = np.std(sigma_values, ddof=1)

            # Specification limits (example)
            usl = 0.7
            lsl = 0.3

            # Calculate capability indices
            cp = (usl - lsl) / (6 * std) if std > 0 else 0
            cpu = (usl - mean) / (3 * std) if std > 0 else 0
            cpl = (mean - lsl) / (3 * std) if std > 0 else 0
            cpk = min(cpu, cpl)

            # Percentage within spec
            within_spec = sum(1 for v in sigma_values if lsl <= v <= usl)
            pct_within_spec = (within_spec / len(sigma_values)) * 100

            # Generate report
            report = f"""PROCESS CAPABILITY STUDY REPORT
{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Parameter: Sigma Gradient

SAMPLE STATISTICS:
    Sample Size: {len(sigma_values)}
    Mean: {mean:.4f}
    Std Dev: {std:.4f}
    Min: {min(sigma_values):.4f}
    Max: {max(sigma_values):.4f}

SPECIFICATION LIMITS:
    Lower Spec Limit (LSL): {lsl}
    Upper Spec Limit (USL): {usl}
    Target: {(usl + lsl) / 2:.4f}

CAPABILITY INDICES:
    Cp: {cp:.3f}
    Cpu: {cpu:.3f}
    Cpl: {cpl:.3f}
    Cpk: {cpk:.3f}

PERFORMANCE:
    Within Spec: {within_spec} of {len(sigma_values)} ({pct_within_spec:.1f}%)

INTERPRETATION:
"""
            if cpk >= 1.33:
                report += "    Process is CAPABLE (Cpk >= 1.33)\n"
                report += "    Process is well-centered and has low variability.\n"
            elif cpk >= 1.0:
                report += "    Process is MARGINALLY CAPABLE (1.0 <= Cpk < 1.33)\n"
                report += "    Consider process improvement to increase margin.\n"
            else:
                report += "    Process is NOT CAPABLE (Cpk < 1.0)\n"
                report += "    Immediate action required to reduce variability or re-center process.\n"

            # Display report
            self.capability_display.configure(state='normal')
            self.capability_display.delete('1.0', tk.END)
            self.capability_display.insert('1.0', report)
            self.capability_display.configure(state='disabled')

            # Plot histogram
            self.capability_chart.clear_chart()
            fig = self.capability_chart.figure
            ax = fig.add_subplot(111)

            # Apply theme
            self.capability_chart._apply_theme_to_axes(ax)

            # Plot histogram
            n, bins, patches = ax.hist(sigma_values, bins=30, density=True,
                                       alpha=0.7, color='steelblue', edgecolor='white')

            # Add normal curve
            x = np.linspace(min(sigma_values), max(sigma_values), 100)
            y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
            ax.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')

            # Add spec limits
            ax.axvline(lsl, color='red', linestyle='--', linewidth=2, label=f'LSL ({lsl})')
            ax.axvline(usl, color='red', linestyle='--', linewidth=2, label=f'USL ({usl})')
            ax.axvline(mean, color='green', linestyle='-', linewidth=2, label=f'Mean ({mean:.3f})')

            ax.set_xlabel('Sigma Gradient', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'Process Capability: Cpk = {cpk:.3f}', fontsize=14, fontweight='bold')

            legend = ax.legend(loc='upper right')
            if legend:
                self.capability_chart._style_legend(legend)

            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            self.capability_chart.canvas.draw()

        except Exception as e:
            self.logger.error(f"Error running capability study: {e}")
            messagebox.showerror("Error", f"Failed to run capability study:\n{str(e)}")

    def _run_pareto_analysis(self):
        """Run Pareto analysis of defects and issues."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return

        try:
            # Switch to Pareto tab
            self.spc_tabview.set("Pareto Analysis")

            # Collect defect data
            defects = {}
            for result in self.current_data:
                if result.tracks:
                    for track in result.tracks:
                        # Count sigma failures
                        if hasattr(track, 'sigma_pass') and not track.sigma_pass:
                            defects['Sigma Failure'] = defects.get('Sigma Failure', 0) + 1

                        # Count linearity failures
                        if hasattr(track, 'linearity_pass') and not track.linearity_pass:
                            defects['Linearity Failure'] = defects.get('Linearity Failure', 0) + 1

                        # Count high risk
                        if hasattr(track, 'risk_category') and track.risk_category:
                            risk = track.risk_category.value if hasattr(track.risk_category, 'value') else str(track.risk_category)
                            if risk == 'High':
                                defects['High Risk'] = defects.get('High Risk', 0) + 1

                        # Count resistance issues
                        if hasattr(track, 'resistance_change_pct') and track.resistance_change_pct is not None:
                            if abs(track.resistance_change_pct) > 5:
                                defects['Large Resistance Change'] = defects.get('Large Resistance Change', 0) + 1

            if not defects:
                messagebox.showinfo("No Defects", "No defects or issues found in the data.")
                return

            # Sort defects by count (descending)
            sorted_defects = sorted(defects.items(), key=lambda x: x[1], reverse=True)
            categories = [item[0] for item in sorted_defects]
            counts = [item[1] for item in sorted_defects]

            # Calculate cumulative percentage
            total = sum(counts)
            cumulative = []
            running_sum = 0
            for count in counts:
                running_sum += count
                cumulative.append((running_sum / total) * 100)

            # Plot Pareto chart
            self.pareto_chart.clear_chart()
            fig = self.pareto_chart.figure
            ax1 = fig.add_subplot(111)

            # Apply theme
            self.pareto_chart._apply_theme_to_axes(ax1)

            # Get theme colors
            from laser_trim_analyzer.gui.theme_helper import ThemeHelper
            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]

            # Bar chart
            x_pos = range(len(categories))
            bars = ax1.bar(x_pos, counts, color='steelblue', alpha=0.7, edgecolor='white')

            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax1.set_xlabel('Defect Type', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12, color='steelblue')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(categories, rotation=45, ha='right')

            # Cumulative line on secondary y-axis
            ax2 = ax1.twinx()
            ax2.plot(x_pos, cumulative, 'ro-', linewidth=2, markersize=8)
            ax2.set_ylabel('Cumulative %', fontsize=12, color='red')
            ax2.set_ylim(0, 105)

            # Add 80% reference line
            ax2.axhline(80, color='green', linestyle='--', alpha=0.5)
            ax2.text(len(categories)-0.5, 82, '80%', color='green', fontsize=10)

            ax1.set_title('Pareto Analysis of Defects', fontsize=14, fontweight='bold')

            # Grid only on primary axis
            ax1.grid(True, alpha=0.3, axis='y')

            fig.tight_layout()
            self.pareto_chart.canvas.draw()

        except Exception as e:
            self.logger.error(f"Error running Pareto analysis: {e}")
            messagebox.showerror("Error", f"Failed to run Pareto analysis:\n{str(e)}")

    def _detect_process_drift(self):
        """
        Detect process drift using ML-first approach with formula fallback.

        Phase 3 ML Integration: Uses UnifiedProcessor.detect_drift() which
        follows the ThresholdOptimizer pattern (ADR-005):
        1. Check feature flag first
        2. Try ML prediction if model trained
        3. Fall back to statistical CUSUM method if ML not available
        4. Log which method used
        """
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return

        try:
            # Switch to drift detection tab
            self.spc_tabview.set("Drift Detection")

            # Use UnifiedProcessor for drift detection (ML-first with formula fallback)
            drift_report = self._run_drift_detection()

            if drift_report is None:
                messagebox.showwarning("Insufficient Data",
                                     "Need at least 20 samples for drift detection.")
                return

            # Get method used for logging
            method_used = drift_report.get('method_used', 'formula')
            self.logger.info(f"Drift detection using {method_used} method")

            # Prepare visualization data (extract sigma values for chart)
            drift_data = []
            for result in self.current_data:
                if result.tracks:
                    for track in result.tracks:
                        if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                            drift_data.append({
                                'date': result.file_date or result.timestamp,
                                'value': track.sigma_gradient,
                                'model': result.model
                            })

            # Convert to DataFrame and sort by date
            df = pd.DataFrame(drift_data).sort_values('date').reset_index(drop=True)

            # Calculate moving averages for visualization
            window_size = min(10, len(df) // 4)
            if window_size < 3:
                window_size = min(3, len(df))
            df['ma'] = df['value'].rolling(window=window_size).mean()
            df['ma_std'] = df['value'].rolling(window=window_size).std()

            # Get target from drift report or calculate
            target = drift_report.get('target_value', df['value'].iloc[:window_size].mean())

            # Plot drift analysis with visualization
            self.drift_chart.clear_chart()
            fig = self.drift_chart.figure
            ax = fig.add_subplot(111)

            # Apply theme to axes
            self.drift_chart._apply_theme_to_axes(ax)

            # Get theme colors
            from laser_trim_analyzer.gui.theme_helper import ThemeHelper
            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]

            # Map drift severity to color intensity
            severity = drift_report.get('drift_severity', 'negligible')
            severity_map = {
                'negligible': 0.0, 'low': 0.3, 'moderate': 0.6,
                'high': 0.8, 'critical': 1.0, 'insufficient_data': 0.0
            }
            drift_intensity = severity_map.get(severity, 0.0)

            # Plot individual values as gray dots
            ax.scatter(df.index, df['value'], color='gray', alpha=0.3, s=30, label='Individual Values')

            # Plot moving average with color based on drift severity from report
            for i in range(1, len(df)):
                if pd.notna(df['ma'].iloc[i]) and pd.notna(df['ma'].iloc[i-1]):
                    # Color based on overall drift severity
                    if drift_intensity < 0.3:
                        color = 'green'
                    elif drift_intensity < 0.6:
                        color = 'orange'
                    else:
                        color = 'red'

                    ax.plot([i-1, i], [df['ma'].iloc[i-1], df['ma'].iloc[i]],
                           color=color, linewidth=3, alpha=0.8)

            # Add reference lines
            ax.axhline(y=target, color='blue', linestyle='--', alpha=0.5, label=f'Target: {target:.3f}')
            ax.axhline(y=target + 2*df['value'].std(), color='orange', linestyle=':', alpha=0.5)
            ax.axhline(y=target - 2*df['value'].std(), color='orange', linestyle=':', alpha=0.5)

            # Add drift zones
            ax.fill_between([0, len(df)], [target - 3*df['value'].std()]*2, [target - 2*df['value'].std()]*2,
                           color='yellow', alpha=0.1, label='Warning Zone')
            ax.fill_between([0, len(df)], [target + 2*df['value'].std()]*2, [target + 3*df['value'].std()]*2,
                           color='yellow', alpha=0.1)
            ax.fill_between([0, len(df)], [ax.get_ylim()[0]]*2, [target - 3*df['value'].std()]*2,
                           color='red', alpha=0.1, label='Drift Zone')
            ax.fill_between([0, len(df)], [target + 3*df['value'].std()]*2, [ax.get_ylim()[1]]*2,
                           color='red', alpha=0.1)

            # Mark drift points from the report
            drift_point_indices = [p['index'] for p in drift_report.get('drift_points', [])]
            if drift_point_indices:
                drift_df = df.iloc[drift_point_indices]
                ax.scatter(drift_df.index, drift_df['value'],
                          color='red', s=100, marker='v',
                          label=f'Drift Detected ({len(drift_point_indices)} points)')

            ax.set_xlabel('Sample Number', fontsize=12)
            ax.set_ylabel('Sigma Gradient', fontsize=12)

            # Update title to show method used
            method_label = "ML" if method_used == 'ml' else "Statistical"
            ax.set_title(f'Process Drift Detection ({method_label} Method)',
                        fontsize=14, fontweight='bold')

            # Create custom legend
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='gray', marker='o', linestyle='', alpha=0.3, label='Individual Values'),
                Line2D([0], [0], color='green', linewidth=3, label='Stable (Moving Avg)'),
                Line2D([0], [0], color='orange', linewidth=3, label='Warning'),
                Line2D([0], [0], color='red', linewidth=3, label='Drift Detected'),
                Line2D([0], [0], color='blue', linestyle='--', label='Target'),
                Patch(facecolor='yellow', alpha=0.3, label='Warning Zone'),
                Patch(facecolor='red', alpha=0.3, label='Drift Zone')
            ]

            legend = ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
            if legend:
                self.drift_chart._style_legend(legend)

            ax.grid(True, alpha=0.3)

            # Add explanation text box with recommendations
            recommendations = drift_report.get('recommendations', [])
            rec_text = "\n".join(f"* {r}" for r in recommendations[:3]) if recommendations else "No recommendations"

            explanation = (
                f"Drift Analysis ({method_label}):\n"
                f"* Severity: {severity.upper()}\n"
                f"* Drift Rate: {drift_report.get('drift_rate', 0):.1%}\n"
                f"* Trend: {drift_report.get('drift_trend', 'unknown')}\n"
                f"* Samples: {drift_report.get('samples_analyzed', len(df))}\n\n"
                f"Recommendations:\n{rec_text}"
            )

            ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5',
                            facecolor='white' if ctk.get_appearance_mode().lower() == "light" else '#2b2b2b',
                            alpha=0.8, edgecolor=text_color),
                   color=text_color)

            fig.tight_layout()
            self.drift_chart.canvas.draw()

            # Update drift alert metric
            drift_detected = drift_report.get('drift_detected', False)
            drift_points_count = len(drift_report.get('drift_points', []))

            if drift_detected:
                self._drift_alerts = drift_points_count
                self.drift_alert_card.update_value(str(drift_points_count))
                self.drift_alert_card.set_color_scheme("error")

                # Show drift details with recommendations
                rec_text_msg = "\n".join(f"* {r}" for r in recommendations[:3])
                messagebox.showinfo("Drift Detected",
                                  f"Process drift detected!\n\n"
                                  f"Severity: {severity.upper()}\n"
                                  f"Drift Rate: {drift_report.get('drift_rate', 0):.1%}\n"
                                  f"Points: {drift_points_count}\n"
                                  f"Method: {method_label}\n\n"
                                  f"Recommendations:\n{rec_text_msg}")
            else:
                self._drift_alerts = 0
                self.drift_alert_card.update_value("0")
                self.drift_alert_card.set_color_scheme("success")

                messagebox.showinfo("No Drift",
                                  f"No significant process drift detected.\n"
                                  f"Severity: {severity}\n"
                                  f"Method: {method_label}\n\n"
                                  f"Process appears to be stable.")

        except Exception as e:
            self.logger.error(f"Error detecting process drift: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to detect process drift:\n{str(e)}")

    def _run_drift_detection(self) -> Optional[Dict[str, Any]]:
        """
        Run drift detection using UnifiedProcessor with ML-first approach.

        Returns:
            Drift report dictionary or None if insufficient data
        """
        try:
            # Get config for UnifiedProcessor
            config = get_config()

            # Create UnifiedProcessor for drift detection
            if HAS_UNIFIED_PROCESSOR:
                processor = UnifiedProcessor(config=config)
                drift_report = processor.detect_drift(self.current_data)

                # Check for insufficient data
                if drift_report.get('drift_severity') == 'insufficient_data':
                    return None

                return drift_report
            else:
                # Fallback to manual formula-based detection if UnifiedProcessor unavailable
                self.logger.warning("UnifiedProcessor not available, using inline formula")
                return self._detect_drift_inline_fallback()

        except Exception as e:
            self.logger.error(f"Error in drift detection: {e}")
            return self._detect_drift_inline_fallback()

    def _detect_drift_inline_fallback(self) -> Optional[Dict[str, Any]]:
        """
        Inline fallback drift detection when UnifiedProcessor is not available.

        Uses CUSUM statistical method similar to original implementation.
        """
        # Extract time series data
        drift_data = []
        for result in self.current_data:
            if result.tracks:
                for track in result.tracks:
                    if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                        drift_data.append({
                            'date': result.file_date or result.timestamp,
                            'value': track.sigma_gradient
                        })

        if len(drift_data) < 20:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(drift_data).sort_values('date').reset_index(drop=True)

        # Calculate baseline statistics from first 25% of data
        baseline_size = max(5, len(df) // 4)
        baseline = df['value'].iloc[:baseline_size]
        target = baseline.mean()
        std = baseline.std()

        if std == 0:
            std = 0.001  # Avoid division by zero

        # CUSUM parameters
        k = 0.5  # Allowable slack
        h = 4.0  # Decision interval

        # Calculate CUSUM
        cusum_pos = np.zeros(len(df))
        cusum_neg = np.zeros(len(df))

        for i in range(1, len(df)):
            z = (df['value'].iloc[i] - target) / std
            cusum_pos[i] = max(0, cusum_pos[i-1] + z - k)
            cusum_neg[i] = min(0, cusum_neg[i-1] + z + k)

        # Detect drift points
        drift_points = []
        for i in range(len(df)):
            if cusum_pos[i] > h or abs(cusum_neg[i]) > h:
                drift_points.append({
                    'index': i,
                    'value': df['value'].iloc[i],
                    'cusum_pos': cusum_pos[i],
                    'cusum_neg': cusum_neg[i]
                })

        # Calculate drift rate
        drift_rate = len(drift_points) / len(df) if len(df) > 0 else 0

        # Determine severity
        if drift_rate < 0.05:
            severity = 'negligible'
        elif drift_rate < 0.1:
            severity = 'low'
        elif drift_rate < 0.2:
            severity = 'moderate'
        elif drift_rate < 0.3:
            severity = 'high'
        else:
            severity = 'critical'

        # Determine trend
        if len(df) > 10:
            recent_mean = df['value'].iloc[-10:].mean()
            if recent_mean > target + std:
                trend = 'upward'
            elif recent_mean < target - std:
                trend = 'downward'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'

        # Generate recommendations
        recommendations = []
        if severity in ['moderate', 'high', 'critical']:
            recommendations.append("Investigate recent process changes")
            recommendations.append("Review equipment calibration")
            if trend == 'upward':
                recommendations.append("Check for material or environmental changes causing increase")
            elif trend == 'downward':
                recommendations.append("Verify measurement system for potential degradation")

        return {
            'drift_detected': len(drift_points) > 0,
            'drift_severity': severity,
            'drift_rate': drift_rate,
            'drift_trend': trend,
            'drift_points': drift_points,
            'recommendations': recommendations,
            'feature_drift': {},
            'method_used': 'formula',
            'cusum_threshold': float(h),
            'target_value': float(target),
            'samples_analyzed': len(df)
        }

    def _analyze_failure_modes(self):
        """Analyze failure modes and patterns."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return

        try:
            # Switch to failure modes tab
            self.spc_tabview.set("Failure Modes")

            # Analyze failure patterns
            failure_analysis = {
                'total_units': 0,
                'failed_units': 0,
                'failure_modes': {},
                'correlations': {},
                'recommendations': []
            }

            for result in self.current_data:
                failure_analysis['total_units'] += 1

                if result.overall_status.value != "Pass":
                    failure_analysis['failed_units'] += 1

                if result.tracks:
                    for track in result.tracks:
                        # Identify failure modes
                        failures = []

                        if hasattr(track, 'sigma_pass') and not track.sigma_pass:
                            failures.append('Sigma Failure')

                        if hasattr(track, 'linearity_pass') and not track.linearity_pass:
                            failures.append('Linearity Failure')

                        if hasattr(track, 'risk_category') and track.risk_category:
                            risk = track.risk_category.value if hasattr(track.risk_category, 'value') else str(track.risk_category)
                            if risk == 'High':
                                failures.append('High Risk Classification')

                        # Count failure combinations
                        if failures:
                            failure_key = ' + '.join(sorted(failures))
                            failure_analysis['failure_modes'][failure_key] = \
                                failure_analysis['failure_modes'].get(failure_key, 0) + 1

            # Generate recommendations based on analysis
            if failure_analysis['failure_modes']:
                top_failure = max(failure_analysis['failure_modes'].items(), key=lambda x: x[1])

                if 'Sigma Failure' in top_failure[0]:
                    failure_analysis['recommendations'].append(
                        "* Review and optimize trim parameters - sigma failures are prevalent"
                    )

                if 'Linearity Failure' in top_failure[0]:
                    failure_analysis['recommendations'].append(
                        "* Investigate mechanical alignment and calibration procedures"
                    )

                if 'High Risk' in top_failure[0]:
                    failure_analysis['recommendations'].append(
                        "* Implement additional quality checks for high-risk units"
                    )

            # Generate report
            report = f"""FAILURE MODE ANALYSIS REPORT
{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
    Total Units Analyzed: {failure_analysis['total_units']}
    Failed Units: {failure_analysis['failed_units']}
    Failure Rate: {(failure_analysis['failed_units'] / failure_analysis['total_units'] * 100) if failure_analysis['total_units'] > 0 else 0:.2f}%

FAILURE MODE BREAKDOWN:
"""

            # Sort failure modes by frequency
            sorted_modes = sorted(failure_analysis['failure_modes'].items(),
                                key=lambda x: x[1], reverse=True)

            for mode, count in sorted_modes:
                percentage = (count / failure_analysis['failed_units'] * 100) if failure_analysis['failed_units'] > 0 else 0
                report += f"    {mode}: {count} occurrences ({percentage:.1f}%)\n"

            report += f"\nRECOMMENDATIONS:\n"
            for rec in failure_analysis['recommendations']:
                report += f"{rec}\n"

            # Display report
            self.failure_mode_display.configure(state='normal')
            self.failure_mode_display.delete('1.0', tk.END)
            self.failure_mode_display.insert('1.0', report)
            self.failure_mode_display.configure(state='disabled')

        except Exception as e:
            self.logger.error(f"Error analyzing failure modes: {e}")
            messagebox.showerror("Error", f"Failed to analyze failure modes:\n{str(e)}")
