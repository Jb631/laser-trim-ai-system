"""
Historical Data Analyzer for Laser Trim AI System

This module provides advanced analysis of historical data to identify trends,
patterns, and opportunities for continuous improvement.

Author: Laser Trim AI System
Date: 2024
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from .database_manager import DatabaseManager
from ..config import Config


class HistoricalAnalyzer:
    """Analyzes historical data for trends and insights."""

    def __init__(self, db_manager: DatabaseManager, config: Config):
        """
        Initialize historical analyzer.

        Args:
            db_manager: Database manager instance
            config: System configuration
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)

    def analyze_model_trends(self, model: str, days_back: int = 90) -> Dict[str, Any]:
        """
        Analyze performance trends for a specific model.

        Args:
            model: Model number to analyze
            days_back: Number of days to analyze

        Returns:
            Dictionary containing trend analysis results
        """
        # Get historical data
        df = self.db.get_model_performance_history(model, days_back)

        if df.empty:
            return {'error': 'No data available for analysis'}

        analysis = {
            'model': model,
            'period': f'{days_back} days',
            'data_points': len(df)
        }

        # Pass rate trend
        if len(df) > 7:
            trend_result = self._calculate_trend(df['date'], df['pass_rate'])
            analysis['pass_rate_trend'] = trend_result

        # Sigma gradient analysis
        analysis['sigma_analysis'] = {
            'mean': df['avg_sigma'].mean(),
            'std': df['avg_sigma'].std(),
            'trend': self._calculate_trend(df['date'], df['avg_sigma']) if len(df) > 7 else None,
            'volatility': df['avg_sigma'].std() / df['avg_sigma'].mean() if df['avg_sigma'].mean() > 0 else 0
        }

        # Identify problematic periods
        analysis['problem_periods'] = self._identify_problem_periods(df)

        # Predict future performance
        if len(df) > 14:
            analysis['predictions'] = self._predict_future_performance(df)

        # Seasonal patterns
        if len(df) > 30:
            analysis['seasonal_patterns'] = self._detect_seasonal_patterns(df)

        return analysis

    def _calculate_trend(self, dates: pd.Series, values: pd.Series) -> Dict[str, Any]:
        """Calculate linear trend and statistical significance."""
        # Convert dates to numeric values
        x = (dates - dates.min()).dt.total_seconds() / 86400  # Days from start
        y = values.values

        # Remove NaN values
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            return {'error': 'Insufficient data for trend analysis'}

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Calculate percentage change
        if len(y) > 0 and y[0] != 0:
            pct_change = ((y[-1] - y[0]) / y[0]) * 100
        else:
            pct_change = 0

        return {
            'slope': slope,
            'direction': 'improving' if slope > 0 else 'declining',
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'percentage_change': pct_change,
            'interpretation': self._interpret_trend(slope, p_value, pct_change)
        }

    def _interpret_trend(self, slope: float, p_value: float, pct_change: float) -> str:
        """Interpret trend analysis results."""
        if p_value > 0.05:
            return "No significant trend detected"

        if abs(pct_change) < 5:
            return "Stable performance with minor variations"
        elif pct_change > 5:
            return f"Significant improvement of {pct_change:.1f}%"
        else:
            return f"Concerning decline of {abs(pct_change):.1f}%"

    def _identify_problem_periods(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify periods with performance issues."""
        problems = []

        # Define thresholds
        low_pass_threshold = 0.8
        high_risk_threshold = 3

        for _, row in df.iterrows():
            issues = []

            if row['pass_rate'] < low_pass_threshold:
                issues.append(f"Low pass rate: {row['pass_rate']:.1%}")

            if row['high_risk_count'] > high_risk_threshold:
                issues.append(f"High risk units: {row['high_risk_count']}")

            if issues:
                problems.append({
                    'date': row['date'],
                    'issues': issues,
                    'severity': 'high' if row['pass_rate'] < 0.7 else 'medium'
                })

        return problems

    def _predict_future_performance(self, df: pd.DataFrame, days_ahead: int = 7) -> Dict[str, Any]:
        """Predict future performance using simple time series analysis."""
        # Use exponential smoothing for prediction
        alpha = 0.3  # Smoothing factor

        # Prepare data
        pass_rates = df['pass_rate'].values

        # Simple exponential smoothing
        forecast = [pass_rates[0]]
        for i in range(1, len(pass_rates)):
            forecast.append(alpha * pass_rates[i] + (1 - alpha) * forecast[i - 1])

        # Project forward
        last_value = forecast[-1]
        trend = (forecast[-1] - forecast[-7]) / 7 if len(forecast) > 7 else 0

        predictions = []
        for i in range(days_ahead):
            predicted_value = last_value + trend * (i + 1)
            predicted_value = max(0, min(1, predicted_value))  # Bound between 0 and 1
            predictions.append({
                'days_ahead': i + 1,
                'predicted_pass_rate': predicted_value,
                'confidence_interval': (predicted_value - 0.1, predicted_value + 0.1)
            })

        return {
            'method': 'exponential_smoothing',
            'predictions': predictions,
            'trend_continuation': 'improving' if trend > 0 else 'declining'
        }

    def _detect_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal or cyclical patterns in the data."""
        # Analyze by day of week
        df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()

        weekly_stats = df.groupby('day_of_week').agg({
            'pass_rate': ['mean', 'std'],
            'total_units': 'sum'
        }).round(3)

        # Find best and worst days
        best_day = weekly_stats['pass_rate']['mean'].idxmax()
        worst_day = weekly_stats['pass_rate']['mean'].idxmin()

        return {
            'weekly_pattern': weekly_stats.to_dict(),
            'best_day': best_day,
            'worst_day': worst_day,
            'variation': weekly_stats['pass_rate']['std'].mean()
        }

    def compare_models(self, models: List[str], metric: str = 'pass_rate',
                       days_back: int = 30) -> pd.DataFrame:
        """
        Compare performance across multiple models.

        Args:
            models: List of model numbers to compare
            metric: Metric to compare (pass_rate, sigma_gradient, etc.)
            days_back: Number of days to analyze

        Returns:
            DataFrame with comparison results
        """
        comparison_data = []

        for model in models:
            df = self.db.get_historical_data(model=model, days_back=days_back)

            if not df.empty:
                if metric == 'pass_rate':
                    value = (df['sigma_pass'] & df['linearity_pass']).mean()
                else:
                    value = df[metric].mean() if metric in df.columns else None

                comparison_data.append({
                    'model': model,
                    'sample_size': len(df),
                    metric: value,
                    f'{metric}_std': df[metric].std() if metric in df.columns else None,
                    'latest_date': df['timestamp'].max()
                })

        return pd.DataFrame(comparison_data)

    def detect_anomaly_clusters(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Detect clusters of anomalies that might indicate systematic issues.

        Args:
            days_back: Number of days to analyze

        Returns:
            Dictionary containing anomaly cluster analysis
        """
        # Get recent anomalies with file data
        query = '''
            SELECT a.*, f.model, f.serial, f.sigma_gradient, f.failure_probability
            FROM anomalies a
            JOIN file_results f ON a.file_id = f.id
            WHERE a.timestamp > datetime('now', ?)
        '''

        with self.db.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(f'-{days_back} days',),
                                   parse_dates=['timestamp'])

        if df.empty:
            return {'clusters': [], 'summary': 'No anomalies found'}

        # Prepare features for clustering
        features = ['sigma_gradient', 'failure_probability']
        X = df[features].fillna(0).values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        clustering = DBSCAN(eps=0.5, min_samples=3)
        df['cluster'] = clustering.fit_predict(X_scaled)

        # Analyze clusters
        clusters = []
        for cluster_id in df['cluster'].unique():
            if cluster_id == -1:  # Noise points
                continue

            cluster_data = df[df['cluster'] == cluster_id]

            clusters.append({
                'cluster_id': int(cluster_id),
                'size': len(cluster_data),
                'models_affected': cluster_data['model'].unique().tolist(),
                'anomaly_types': cluster_data['anomaly_type'].value_counts().to_dict(),
                'date_range': {
                    'start': cluster_data['timestamp'].min().isoformat(),
                    'end': cluster_data['timestamp'].max().isoformat()
                },
                'characteristics': {
                    'avg_sigma': cluster_data['sigma_gradient'].mean(),
                    'avg_failure_prob': cluster_data['failure_probability'].mean()
                }
            })

        return {
            'clusters': clusters,
            'total_anomalies': len(df),
            'clustered_anomalies': len(df[df['cluster'] != -1]),
            'noise_points': len(df[df['cluster'] == -1]),
            'interpretation': self._interpret_clusters(clusters)
        }

    def _interpret_clusters(self, clusters: List[Dict]) -> str:
        """Interpret anomaly clusters."""
        if not clusters:
            return "No significant anomaly clusters detected"

        if len(clusters) == 1:
            return f"Single anomaly cluster detected affecting {clusters[0]['size']} units"

        # Check for model-specific issues
        model_specific = any(len(c['models_affected']) == 1 for c in clusters)
        if model_specific:
            return "Model-specific anomaly patterns detected - investigate individual models"

        return f"Multiple anomaly clusters detected ({len(clusters)} clusters) - possible systematic issues"

    def generate_improvement_recommendations(self, model: str) -> List[Dict[str, Any]]:
        """
        Generate specific recommendations for improving model performance.

        Args:
            model: Model number to analyze

        Returns:
            List of recommendations with priority and expected impact
        """
        recommendations = []

        # Get recent performance data
        df = self.db.get_historical_data(model=model, days_back=30)

        if df.empty:
            return [{'recommendation': 'Insufficient data for analysis', 'priority': 'low'}]

        # Analyze failure patterns
        avg_sigma = df['sigma_gradient'].mean()
        std_sigma = df['sigma_gradient'].std()
        pass_rate = (df['sigma_pass'] & df['linearity_pass']).mean()
        high_risk_rate = (df['risk_category'] == 'High').mean()

        # High sigma gradient
        if avg_sigma > self.config.SIGMA_THRESHOLD * 0.8:
            recommendations.append({
                'issue': 'High average sigma gradient',
                'recommendation': 'Review laser trimming parameters and calibration',
                'priority': 'high',
                'expected_impact': 'Could improve pass rate by 10-15%',
                'metrics': {'current_avg_sigma': avg_sigma, 'target': self.config.SIGMA_THRESHOLD * 0.7}
            })

        # High variability
        if std_sigma > avg_sigma * 0.3:
            recommendations.append({
                'issue': 'High sigma gradient variability',
                'recommendation': 'Investigate process consistency and material quality',
                'priority': 'medium',
                'expected_impact': 'Could reduce defect rate by 5-10%',
                'metrics': {'current_std': std_sigma, 'target': avg_sigma * 0.2}
            })

        # Low pass rate
        if pass_rate < 0.9:
            recommendations.append({
                'issue': f'Low pass rate ({pass_rate:.1%})',
                'recommendation': 'Perform root cause analysis on failed units',
                'priority': 'high',
                'expected_impact': 'Target 95% pass rate achievable',
                'metrics': {'current_pass_rate': pass_rate, 'target': 0.95}
            })

        # High risk units
        if high_risk_rate > 0.1:
            recommendations.append({
                'issue': f'High percentage of high-risk units ({high_risk_rate:.1%})',
                'recommendation': 'Implement additional quality checks for at-risk units',
                'priority': 'medium',
                'expected_impact': 'Could prevent field failures',
                'metrics': {'current_high_risk_rate': high_risk_rate, 'target': 0.05}
            })

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))

        return recommendations

    def calculate_cost_impact(self, model: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Calculate the potential cost impact of quality issues.

        Args:
            model: Model number to analyze
            days_back: Number of days to analyze

        Returns:
            Dictionary containing cost impact analysis
        """
        df = self.db.get_historical_data(model=model, days_back=days_back)

        if df.empty:
            return {'error': 'No data available for cost analysis'}

        # Define cost parameters (these should come from config in production)
        costs = {
            'rework_cost': 50,  # Cost to rework a failed unit
            'field_failure_cost': 500,  # Cost of a field failure
            'scrap_cost': 100,  # Cost to scrap a unit
            'inspection_cost': 10  # Additional inspection cost
        }

        # Calculate quantities
        total_units = len(df)
        failed_units = len(df[~(df['sigma_pass'] & df['linearity_pass'])])
        high_risk_units = len(df[df['risk_category'] == 'High'])

        # Estimate field failure rate based on risk category
        field_failure_rate = {
            'High': 0.1,
            'Medium': 0.02,
            'Low': 0.001
        }

        estimated_field_failures = sum(
            field_failure_rate.get(risk, 0)
            for risk in df['risk_category']
        )

        # Calculate costs
        rework_cost_total = failed_units * costs['rework_cost']
        field_failure_cost_total = estimated_field_failures * costs['field_failure_cost']
        inspection_cost_total = high_risk_units * costs['inspection_cost']

        total_cost_impact = rework_cost_total + field_failure_cost_total + inspection_cost_total
        cost_per_unit = total_cost_impact / total_units if total_units > 0 else 0

        return {
            'period': f'{days_back} days',
            'total_units': total_units,
            'failed_units': failed_units,
            'high_risk_units': high_risk_units,
            'costs': {
                'rework': rework_cost_total,
                'field_failures': field_failure_cost_total,
                'additional_inspection': inspection_cost_total,
                'total': total_cost_impact
            },
            'cost_per_unit': cost_per_unit,
            'potential_savings': {
                'reduce_failures_10pct': total_cost_impact * 0.1,
                'eliminate_high_risk': high_risk_units * costs['field_failure_cost'] * 0.1
            }
        }