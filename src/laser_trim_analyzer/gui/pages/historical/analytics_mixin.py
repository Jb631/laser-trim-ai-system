"""
AnalyticsMixin - Advanced analytics methods for HistoricalPage.

This module provides analytics functionality:
- _run_trend_analysis: Trend analysis with forecasting
- _run_correlation_analysis: Correlation matrix calculations
- _generate_statistical_summary: Comprehensive statistics
- _run_predictive_analysis: ML-based predictions
- _detect_anomalies: Outlier detection using IQR method

Migrated from historical_page.py lines 2202-2838 during Phase 4 file splitting.
"""

import threading
from tkinter import messagebox
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import logging
import customtkinter as ctk

logger = logging.getLogger(__name__)

# Optional analytics libraries
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available - some analytics features will be disabled")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available - ML features will be disabled")


class AnalyticsMixin:
    """
    Mixin providing advanced analytics methods.

    Requires HistoricalPage as parent class with:
    - self.current_data: List of analysis results
    - self._analytics_data: Prepared analytics data
    - self.trend_analysis_btn, self.correlation_analysis_btn, etc.: Button widgets
    - self.stats_display, self.anomaly_display: Text display widgets
    - self.trend_chart, self.correlation_chart, etc.: Chart widgets
    - self.logger: Logger instance
    """

    def _run_trend_analysis(self):
        """Run comprehensive trend analysis."""
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return

        self._update_analytics_status("Running Trend Analysis...", "orange")
        self.trend_analysis_btn.configure(state="disabled", text="Analyzing...")

        def analyze():
            try:
                # Use the analytics data that has the proper structure
                if hasattr(self, '_analytics_data') and self._analytics_data:
                    trend_data = self._calculate_trend_analysis(self._analytics_data)
                else:
                    # Fallback to preparing data if not available
                    self._prepare_and_update_analytics(self.current_data)
                    trend_data = self._calculate_trend_analysis(self._analytics_data)
                self.trend_analysis_data = trend_data

                # Update UI
                self.after(0, lambda: self._display_trend_analysis(trend_data))

            except Exception as e:
                logger.error(f"Trend analysis failed: {e}")
                self.after(0, lambda: messagebox.showerror(
                    "Analysis Error", f"Trend analysis failed:\n{str(e)}"
                ))
            finally:
                self.after(0, lambda: self.trend_analysis_btn.configure(
                    state="normal", text="Trend Analysis"
                ))
                self.after(0, lambda: self._update_analytics_status("Ready", "green"))

        threading.Thread(target=analyze, daemon=True).start()

    def _calculate_trend_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive trend analysis from historical data."""
        trend_data = {
            'time_series': {},
            'trends': {},
            'seasonality': {},
            'forecasts': {},
            'change_points': {}
        }

        try:
            df = pd.DataFrame(data)

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')

                # Analyze sigma gradient trends
                if 'sigma_gradient' in df.columns:
                    sigma_trends = self._analyze_parameter_trend(df, 'sigma_gradient', 'timestamp')
                    trend_data['trends']['sigma_gradient'] = sigma_trends

                # Analyze linearity error trends
                if 'linearity_error' in df.columns:
                    linearity_trends = self._analyze_parameter_trend(df, 'linearity_error', 'timestamp')
                    trend_data['trends']['linearity_error'] = linearity_trends

                # Analyze pass rate trends over time
                if 'overall_status' in df.columns:
                    pass_rate_trends = self._analyze_pass_rate_trends()
                    trend_data['trends']['pass_rate'] = pass_rate_trends

                # Detect change points
                trend_data['change_points'] = self._detect_change_points(df)

                # Generate forecasts
                trend_data['forecasts'] = self._generate_forecasts(df)

        except Exception as e:
            logger.error(f"Error in trend analysis calculation: {e}")

        return trend_data

    def _analyze_parameter_trend(self, df: pd.DataFrame, parameter: str, time_col: str) -> Dict[str, Any]:
        """Analyze trend for a specific parameter."""
        analysis = {
            'slope': 0,
            'r_squared': 0,
            'trend_direction': 'stable',
            'significance': 'low',
            'volatility': 0
        }

        try:
            if parameter not in df.columns or df[parameter].isna().all():
                return analysis

            # Remove NaN values
            clean_df = df.dropna(subset=[parameter, time_col])
            if len(clean_df) < 3:
                return analysis

            # Convert timestamps to numeric for regression
            x_numeric = (clean_df[time_col] - clean_df[time_col].min()).dt.total_seconds()
            y = clean_df[parameter]

            # Linear regression if scipy is available
            if HAS_SCIPY:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y)
            else:
                # Simple linear regression without scipy
                n = len(x_numeric)
                if n > 1:
                    x_mean = x_numeric.mean()
                    y_mean = y.mean()

                    numerator = ((x_numeric - x_mean) * (y - y_mean)).sum()
                    denominator = ((x_numeric - x_mean) ** 2).sum()

                    slope = numerator / denominator if denominator != 0 else 0
                    intercept = y_mean - slope * x_mean

                    # Calculate R-squared
                    y_pred = slope * x_numeric + intercept
                    ss_res = ((y - y_pred) ** 2).sum()
                    ss_tot = ((y - y_mean) ** 2).sum()
                    # FIXED: Protect against negative values in sqrt
                    r_squared = max(0.0, min(1.0, 1 - (ss_res / ss_tot))) if ss_tot != 0 else 0
                    r_value = np.sqrt(r_squared)

                    p_value = 0.05  # Placeholder
                    std_err = 0.1  # Placeholder
                else:
                    slope, intercept, r_value, p_value, std_err = 0, 0, 0, 1, 0

            analysis['slope'] = slope
            analysis['r_squared'] = r_value ** 2
            analysis['p_value'] = p_value

            # Determine trend direction
            if abs(slope) < std_err:
                analysis['trend_direction'] = 'stable'
            elif slope > 0:
                analysis['trend_direction'] = 'increasing'
            else:
                analysis['trend_direction'] = 'decreasing'

            # Determine significance
            if p_value < 0.01:
                analysis['significance'] = 'high'
            elif p_value < 0.05:
                analysis['significance'] = 'medium'
            else:
                analysis['significance'] = 'low'

            # Calculate volatility (coefficient of variation)
            analysis['volatility'] = y.std() / y.mean() if y.mean() != 0 else 0

        except Exception as e:
            logger.error(f"Error analyzing parameter trend for {parameter}: {e}")

        return analysis

    def _detect_change_points(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect change points in time series data."""
        # Placeholder - can be expanded with CUSUM or other methods
        return {}

    def _generate_forecasts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate simple forecasts from historical data."""
        # Placeholder - can be expanded with ARIMA or other methods
        return {}

    def _run_correlation_analysis(self):
        """Run correlation analysis between parameters."""
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return

        self._update_analytics_status("Running Correlation Analysis...", "orange")
        self.correlation_analysis_btn.configure(state="disabled", text="Analyzing...")

        def analyze():
            try:
                # Use the analytics data that has the proper structure
                if hasattr(self, '_analytics_data') and self._analytics_data:
                    correlation_data = self._calculate_correlation_matrix(self._analytics_data)
                else:
                    # Fallback to preparing data if not available
                    self._prepare_and_update_analytics(self.current_data)
                    correlation_data = self._calculate_correlation_matrix(self._analytics_data)
                self.correlation_matrix = correlation_data

                # Update UI
                self.after(0, lambda: self._display_correlation_analysis(correlation_data))

            except Exception as e:
                logger.error(f"Correlation analysis failed: {e}")
                self.after(0, lambda: messagebox.showerror(
                    "Analysis Error", f"Correlation analysis failed:\n{str(e)}"
                ))
            finally:
                self.after(0, lambda: self.correlation_analysis_btn.configure(
                    state="normal", text="Correlation Analysis"
                ))
                self.after(0, lambda: self._update_analytics_status("Ready", "green"))

        threading.Thread(target=analyze, daemon=True).start()

    def _generate_statistical_summary(self):
        """Generate comprehensive statistical summary."""
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return

        self._update_analytics_status("Generating Statistical Summary...", "orange")
        self.statistical_summary_btn.configure(state="disabled", text="Generating...")

        def generate():
            try:
                summary_data = self._calculate_statistical_summary(self.current_data)
                self.after(0, lambda: self._display_statistical_summary(summary_data))

            except Exception as e:
                logger.error(f"Statistical summary generation failed: {e}")
                self.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to generate statistical summary:\n{str(e)}"
                ))
            finally:
                self.after(0, lambda: self.statistical_summary_btn.configure(
                    state="normal", text="Statistical Summary"
                ))
                self.after(0, lambda: self._update_analytics_status("Ready", "green"))

        threading.Thread(target=generate, daemon=True).start()

    def _calculate_statistical_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistical summary of historical data."""
        summary = {
            'total_records': 0,
            'pass_rate': 0,
            'parameter_stats': {},
            'model_breakdown': {},
            'temporal_analysis': {}
        }

        try:
            df = pd.DataFrame(data)
            summary['total_records'] = len(df)

            if 'status' in df.columns:
                pass_count = len(df[df['status'] == 'Pass'])
                summary['pass_rate'] = (pass_count / len(df)) * 100 if len(df) > 0 else 0

            # Analyze numerical parameters
            numerical_params = ['sigma_gradient', 'linearity_error', 'resistance_change_percent']
            for param in numerical_params:
                if param in df.columns:
                    values = df[param].dropna()
                    if len(values) > 0:
                        summary['parameter_stats'][param] = {
                            'mean': values.mean(),
                            'median': values.median(),
                            'std': values.std(),
                            'min': values.min(),
                            'max': values.max(),
                            'count': len(values)
                        }

            # Model breakdown
            if 'model' in df.columns:
                model_counts = df['model'].value_counts()
                summary['model_breakdown'] = model_counts.to_dict()

            # Temporal analysis
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                date_range = df['timestamp'].max() - df['timestamp'].min()
                summary['temporal_analysis'] = {
                    'date_range_days': date_range.days,
                    'earliest_record': df['timestamp'].min().isoformat(),
                    'latest_record': df['timestamp'].max().isoformat()
                }

        except Exception as e:
            logger.error(f"Error calculating statistical summary: {e}")
            summary['error'] = str(e)

        return summary

    def _display_statistical_summary(self, summary_data: Dict[str, Any]):
        """Display statistical summary in the UI."""
        try:
            self.stats_display.configure(state='normal')
            self.stats_display.delete('1.0', ctk.END)

            content = "COMPREHENSIVE STATISTICAL SUMMARY\n"
            content += "=" * 60 + "\n\n"

            # Basic statistics
            content += f"Total Records: {summary_data.get('total_records', 0)}\n"
            content += f"Pass Rate: {summary_data.get('pass_rate', 0):.2f}%\n\n"

            # Parameter statistics
            param_stats = summary_data.get('parameter_stats', {})
            if param_stats:
                content += "PARAMETER STATISTICS:\n"
                for param, stats in param_stats.items():
                    content += f"\n{param.replace('_', ' ').title()}:\n"
                    content += f"  Mean: {stats['mean']:.6f}\n"
                    content += f"  Median: {stats['median']:.6f}\n"
                    content += f"  Std Dev: {stats['std']:.6f}\n"
                    content += f"  Range: {stats['min']:.6f} - {stats['max']:.6f}\n"
                    content += f"  Sample Count: {stats['count']}\n"
                content += "\n"

            # Model breakdown
            model_breakdown = summary_data.get('model_breakdown', {})
            if model_breakdown:
                content += "MODEL BREAKDOWN:\n"
                for model, count in sorted(model_breakdown.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / summary_data.get('total_records', 1)) * 100
                    content += f"  {model}: {count} records ({percentage:.1f}%)\n"
                content += "\n"

            # Temporal analysis
            temporal = summary_data.get('temporal_analysis', {})
            if temporal:
                content += "TEMPORAL ANALYSIS:\n"
                content += f"  Date Range: {temporal.get('date_range_days', 0)} days\n"
                content += f"  Earliest: {temporal.get('earliest_record', 'Unknown')}\n"
                content += f"  Latest: {temporal.get('latest_record', 'Unknown')}\n"

            self.stats_display.insert('1.0', content)
            self.stats_display.configure(state='disabled')

        except Exception as e:
            logger.error(f"Error displaying statistical summary: {e}")

    def _calculate_correlation_matrix(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate correlation matrix for numerical parameters."""
        try:
            df = pd.DataFrame(data)

            # Select numerical columns for correlation
            numerical_cols = ['sigma_gradient', 'linearity_error', 'resistance_change_percent',
                            'unit_length', 'travel_length']

            # Filter to available columns
            available_cols = [col for col in numerical_cols if col in df.columns]

            if len(available_cols) < 2:
                return {'matrix': None, 'error': 'Insufficient numerical columns for correlation'}

            correlation_df = df[available_cols].corr()

            # Find strong correlations
            strong_correlations = []
            for i, col1 in enumerate(available_cols):
                for j, col2 in enumerate(available_cols):
                    if i < j:  # Avoid duplicates
                        corr_val = correlation_df.loc[col1, col2]
                        if abs(corr_val) > 0.5:  # Threshold for "strong" correlation
                            strong_correlations.append({
                                'param1': col1,
                                'param2': col2,
                                'correlation': corr_val,
                                'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                            })

            return {
                'matrix': correlation_df,
                'strong_correlations': strong_correlations,
                'parameters': available_cols
            }

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return {'matrix': None, 'error': str(e)}

    def _run_predictive_analysis(self):
        """Run predictive analysis to forecast future performance."""
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return

        self._update_analytics_status("Running Predictive Analysis...", "orange")
        self.predictive_analysis_btn.configure(state="disabled", text="Predicting...")

        def analyze():
            try:
                # Use the analytics data that has the proper structure
                if hasattr(self, '_analytics_data') and self._analytics_data:
                    prediction_data = self._build_predictive_models(self._analytics_data)
                else:
                    # Fallback to preparing data if not available
                    self._prepare_and_update_analytics(self.current_data)
                    prediction_data = self._build_predictive_models(self._analytics_data)

                # Update UI
                self.after(0, lambda: self._display_predictive_analysis(prediction_data))

            except Exception as e:
                logger.error(f"Predictive analysis failed: {e}")
                self.after(0, lambda: messagebox.showerror(
                    "Analysis Error", f"Predictive analysis failed:\n{str(e)}"
                ))
            finally:
                self.after(0, lambda: self.predictive_analysis_btn.configure(
                    state="normal", text="Predictive Analysis"
                ))
                self.after(0, lambda: self._update_analytics_status("Ready", "green"))

        threading.Thread(target=analyze, daemon=True).start()

    def _build_predictive_models(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build simple predictive models from historical data."""
        if not HAS_SKLEARN:
            return {'error': 'Machine learning libraries not available. Install scikit-learn for predictive features.'}

        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score

            df = pd.DataFrame(data)

            # Prepare features and targets
            feature_cols = ['sigma_gradient', 'linearity_error', 'unit_length', 'travel_length']
            available_features = [col for col in feature_cols if col in df.columns]

            if len(available_features) < 2:
                return {'error': 'Insufficient features for predictive modeling'}

            # Remove rows with missing values
            clean_df = df.dropna(subset=available_features)

            if len(clean_df) < 10:
                return {'error': 'Insufficient data points for reliable prediction'}

            models = {}

            # Predict sigma gradient
            if 'sigma_gradient' in clean_df.columns and len(available_features) > 1:
                target_features = [col for col in available_features if col != 'sigma_gradient']

                X = clean_df[target_features]
                y = clean_df['sigma_gradient']

                if len(X) > 5:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    models['sigma_gradient'] = {
                        'model': model,
                        'features': target_features,
                        'mse': mean_squared_error(y_test, y_pred),
                        'r2_score': r2_score(y_test, y_pred),
                        'actual': y_test.tolist(),
                        'predicted': y_pred.tolist(),
                        'feature_importance': dict(zip(target_features, model.feature_importances_))
                    }

            return models

        except ImportError:
            return {'error': 'Scikit-learn not available for predictive modeling'}
        except Exception as e:
            logger.error(f"Error building predictive models: {e}")
            return {'error': str(e)}

    def _detect_anomalies(self):
        """Detect anomalies in the historical data."""
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return

        self._update_analytics_status("Detecting Anomalies...", "orange")
        self.anomaly_detection_btn.configure(state="disabled", text="Detecting...")

        def detect():
            try:
                anomaly_data = self._find_anomalies(self.current_data)

                # Update dashboard
                anomaly_count = len(anomaly_data.get('anomalies', []))
                self.anomaly_count_card.update_value(str(anomaly_count))

                if anomaly_count == 0:
                    self.anomaly_count_card.set_color_scheme('success')
                elif anomaly_count <= 5:
                    self.anomaly_count_card.set_color_scheme('warning')
                else:
                    self.anomaly_count_card.set_color_scheme('danger')

                # Update UI
                self.after(0, lambda: self._display_anomaly_results(anomaly_data))

            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
                self.after(0, lambda: messagebox.showerror(
                    "Analysis Error", f"Anomaly detection failed:\n{str(e)}"
                ))
            finally:
                self.after(0, lambda: self.anomaly_detection_btn.configure(
                    state="normal", text="Detect Anomalies"
                ))
                self.after(0, lambda: self._update_analytics_status("Ready", "green"))

        threading.Thread(target=detect, daemon=True).start()

    def _find_anomalies(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find anomalies using statistical methods."""
        try:
            df = pd.DataFrame(data)
            anomalies = []

            # Parameters to check for anomalies
            numerical_params = ['sigma_gradient', 'linearity_error', 'resistance_change_percent']

            for param in numerical_params:
                if param in df.columns:
                    values = df[param].dropna()

                    if len(values) > 3:
                        # Use IQR method for anomaly detection
                        Q1 = values.quantile(0.25)
                        Q3 = values.quantile(0.75)
                        IQR = Q3 - Q1

                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        # Find outliers
                        outliers = df[(df[param] < lower_bound) | (df[param] > upper_bound)]

                        for idx, row in outliers.iterrows():
                            anomalies.append({
                                'index': idx,
                                'parameter': param,
                                'value': row[param],
                                'expected_range': f"{lower_bound:.4f} - {upper_bound:.4f}",
                                'severity': 'high' if (row[param] < Q1 - 3*IQR or row[param] > Q3 + 3*IQR) else 'medium',
                                'timestamp': row.get('timestamp', 'Unknown'),
                                'model': row.get('model', 'Unknown'),
                                'serial': row.get('serial', 'Unknown')
                            })

            # Statistical summary
            summary = {
                'total_records': len(df),
                'anomalies_found': len(anomalies),
                'anomaly_rate': (len(anomalies) / len(df)) * 100 if len(df) > 0 else 0,
                'parameters_checked': numerical_params,
                'severity_breakdown': {
                    'high': len([a for a in anomalies if a['severity'] == 'high']),
                    'medium': len([a for a in anomalies if a['severity'] == 'medium'])
                }
            }

            return {
                'anomalies': anomalies,
                'summary': summary
            }

        except Exception as e:
            logger.error(f"Error finding anomalies: {e}")
            return {'anomalies': [], 'summary': {}, 'error': str(e)}

    def _display_trend_analysis(self, trend_data: Dict[str, Any]):
        """Display trend analysis results."""
        try:
            # Update trend chart
            self._update_trend_analysis_chart(trend_data)

        except Exception as e:
            logger.error(f"Error displaying trend analysis: {e}")

    def _display_correlation_analysis(self, correlation_data: Dict[str, Any]):
        """Display correlation analysis results."""
        try:
            # Update correlation heatmap
            self._update_correlation_heatmap(correlation_data)

        except Exception as e:
            logger.error(f"Error displaying correlation analysis: {e}")

    def _display_predictive_analysis(self, prediction_data: Dict[str, Any]):
        """Display predictive analysis results."""
        try:
            if 'error' in prediction_data:
                # Show error message
                self.prediction_chart.clear_chart()
                return

            # Update prediction chart
            self._update_prediction_chart(prediction_data)

        except Exception as e:
            logger.error(f"Error displaying predictive analysis: {e}")

    def _display_anomaly_results(self, anomaly_data: Dict[str, Any]):
        """Display anomaly detection results."""
        try:
            self.anomaly_display.configure(state='normal')
            self.anomaly_display.delete('1.0', ctk.END)

            content = "ANOMALY DETECTION RESULTS\n"
            content += "=" * 50 + "\n\n"

            summary = anomaly_data.get('summary', {})
            anomalies = anomaly_data.get('anomalies', [])

            content += f"Total Records Analyzed: {summary.get('total_records', 0)}\n"
            content += f"Anomalies Found: {summary.get('anomalies_found', 0)}\n"
            content += f"Anomaly Rate: {summary.get('anomaly_rate', 0):.2f}%\n\n"

            # Severity breakdown
            severity = summary.get('severity_breakdown', {})
            content += "SEVERITY BREAKDOWN:\n"
            content += f"  High: {severity.get('high', 0)}\n"
            content += f"  Medium: {severity.get('medium', 0)}\n\n"

            # List anomalies
            if anomalies:
                content += "DETECTED ANOMALIES:\n"
                for i, anomaly in enumerate(anomalies[:20], 1):  # Show first 20
                    content += f"\n{i}. {anomaly['parameter'].upper()} ANOMALY\n"
                    content += f"   Value: {anomaly['value']:.6f}\n"
                    content += f"   Expected Range: {anomaly['expected_range']}\n"
                    content += f"   Severity: {anomaly['severity']}\n"
                    content += f"   Model/Serial: {anomaly['model']}/{anomaly['serial']}\n"
                    content += f"   Timestamp: {anomaly['timestamp']}\n"

                if len(anomalies) > 20:
                    content += f"\n... and {len(anomalies) - 20} more anomalies\n"
            else:
                content += "No anomalies detected. All data points are within expected ranges.\n"

            self.anomaly_display.insert('1.0', content)
            self.anomaly_display.configure(state='disabled')

        except Exception as e:
            logger.error(f"Error displaying anomaly results: {e}")

    def _update_trend_analysis_chart(self, trend_data: Dict[str, Any]):
        """Update the trend analysis chart."""
        try:
            self.trend_chart.clear_chart()

            if not trend_data or 'trends' not in trend_data:
                return

            trends = trend_data.get('trends', {})
            if not trends:
                return

            # Create chart
            ax = self.trend_chart.figure.add_subplot(111)
            self.trend_chart._apply_theme_to_axes(ax)

            # Display trend summary as text for now
            content = []
            for param, trend_info in trends.items():
                if isinstance(trend_info, dict):
                    direction = trend_info.get('trend_direction', 'unknown')
                    significance = trend_info.get('significance', 'unknown')
                    content.append(f"{param}: {direction} ({significance})")

            if content:
                ax.text(0.5, 0.5, '\n'.join(content),
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No trend data available',
                       ha='center', va='center', transform=ax.transAxes)

            ax.set_title('Trend Analysis Summary')
            ax.axis('off')

            self.trend_chart.figure.tight_layout()
            self.trend_chart.canvas.draw()

        except Exception as e:
            self.logger.error(f"Error updating trend analysis chart: {e}")

    def _update_correlation_heatmap(self, correlation_data: Dict[str, Any]):
        """Update the correlation heatmap."""
        try:
            self.correlation_chart.clear_chart()

            matrix = correlation_data.get('matrix')
            if matrix is None or matrix.empty:
                return

            # Create heatmap
            ax = self.correlation_chart.figure.add_subplot(111)
            self.correlation_chart._apply_theme_to_axes(ax)

            # Create heatmap
            im = ax.imshow(matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

            # Set ticks
            params = correlation_data.get('parameters', list(matrix.columns))
            ax.set_xticks(np.arange(len(params)))
            ax.set_yticks(np.arange(len(params)))
            ax.set_xticklabels(params, rotation=45, ha='right')
            ax.set_yticklabels(params)

            # Add colorbar
            cbar = self.correlation_chart.figure.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient')

            # Add values
            for i in range(len(params)):
                for j in range(len(params)):
                    text = ax.text(j, i, f'{matrix.iloc[i, j]:.2f}',
                                 ha='center', va='center',
                                 color='black' if abs(matrix.iloc[i, j]) < 0.5 else 'white')

            ax.set_title('Feature Correlation Heatmap')
            self.correlation_chart.figure.tight_layout()
            self.correlation_chart.canvas.draw()

        except Exception as e:
            self.logger.error(f"Error updating correlation heatmap: {e}")

    def _update_prediction_chart(self, prediction_data: Dict[str, Any]):
        """Update the predictive analysis chart."""
        try:
            self.prediction_chart.clear_chart()

            if not prediction_data:
                return

            # Create chart
            ax = self.prediction_chart.figure.add_subplot(111)
            self.prediction_chart._apply_theme_to_axes(ax)

            # Show model performance if available
            if 'sigma_gradient' in prediction_data:
                model_info = prediction_data['sigma_gradient']
                r2 = model_info.get('r2_score', 0)
                mse = model_info.get('mse', 0)

                ax.text(0.5, 0.6, f'Sigma Gradient Prediction Model',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, fontweight='bold')
                ax.text(0.5, 0.4, f'R² Score: {r2:.3f}\nMSE: {mse:.6f}',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12)

                # Feature importance
                importance = model_info.get('feature_importance', {})
                if importance:
                    imp_text = '\n'.join([f'  {k}: {v:.3f}' for k, v in importance.items()])
                    ax.text(0.5, 0.2, f'Feature Importance:\n{imp_text}',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No prediction data available',
                       ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')
            self.prediction_chart.canvas.draw()

        except Exception as e:
            self.logger.error(f"Error updating prediction chart: {e}")
