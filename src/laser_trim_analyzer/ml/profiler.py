"""
Per-Model Profiler for Laser Trim Analyzer v3.

Builds statistical profiles for each product model, providing insights
about quality distributions, correlations, and comparative analysis.

Part of the per-model ML redesign - provides data exploration and insights.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ProfileStatistics:
    """Statistical profile for a single metric."""
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p5: float = 0.0
    p25: float = 0.0
    p50: float = 0.0  # median
    p75: float = 0.0
    p95: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0


@dataclass
class ModelProfile:
    """Complete statistical profile for a product model."""
    model_name: str
    sample_count: int = 0
    trim_count: int = 0
    final_test_count: int = 0

    # Sigma gradient statistics
    sigma: Optional[ProfileStatistics] = None

    # Linearity error statistics
    linearity_error: Optional[ProfileStatistics] = None

    # Quality metrics
    pass_rate: float = 0.0  # Overall pass rate
    fail_rate: float = 0.0
    linearity_pass_rate: float = 0.0
    sigma_pass_rate: float = 0.0
    avg_fail_points: float = 0.0  # Average fail points when failed

    # Spec analysis
    linearity_spec: Optional[float] = None  # Common spec for this model
    spec_margin_percent: float = 0.0  # How much margin to spec on average
    tight_margin_count: int = 0  # Count of units within 10% of spec

    # Correlations
    track_correlation: float = 0.0  # Track 1 vs Track 2 sigma correlation
    sigma_error_correlation: float = 0.0  # Sigma vs linearity error

    # Comparative metrics (relative to other models)
    difficulty_score: float = 0.5  # 0=easiest, 1=hardest
    quality_percentile: float = 0.5  # Quality ranking among all models

    # Time analysis
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None
    date_range_days: int = 0

    # Calculated timestamp
    profiled_date: Optional[datetime] = None


@dataclass
class ModelInsight:
    """A single insight about a model."""
    category: str  # 'quality', 'spec', 'correlation', 'trend', 'comparison'
    severity: str  # 'info', 'warning', 'critical'
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None


class ModelProfiler:
    """
    Per-model statistical profiling for insights.

    Analyzes all available data for a product model to provide:
    - Distribution statistics (sigma, error, etc.)
    - Quality metrics (pass rates, severity)
    - Correlations (track-to-track, sigma-to-error)
    - Comparative analysis (difficulty ranking)
    - Human-readable insights
    """

    def __init__(self, model_name: str):
        """
        Initialize profiler for a specific product model.

        Args:
            model_name: Product model number (e.g., "6828", "8340-1")
        """
        self.model_name = model_name
        self.profile: Optional[ModelProfile] = None
        self.insights: List[ModelInsight] = []
        self.is_profiled: bool = False

    def build_profile(
        self,
        data: pd.DataFrame,
        track_pairs: Optional[pd.DataFrame] = None
    ) -> ModelProfile:
        """
        Build complete statistical profile from data.

        Args:
            data: DataFrame with track-level data containing:
                - sigma_gradient
                - final_linearity_error_shifted or linearity_error
                - linearity_pass
                - sigma_pass
                - linearity_fail_points
                - linearity_spec
                - file_date (optional)
            track_pairs: Optional DataFrame with paired Track 1/Track 2 data
                for correlation analysis

        Returns:
            ModelProfile with complete statistics
        """
        profile = ModelProfile(model_name=self.model_name)

        if len(data) == 0:
            self.profile = profile
            return profile

        profile.sample_count = len(data)

        # Count by source if available
        if 'source' in data.columns:
            profile.trim_count = len(data[data['source'] == 'trim'])
            profile.final_test_count = len(data[data['source'] == 'final_test'])
        else:
            profile.trim_count = len(data)

        # Sigma statistics
        if 'sigma_gradient' in data.columns:
            sigma_values = data['sigma_gradient'].dropna()
            if len(sigma_values) > 0:
                profile.sigma = self._calculate_statistics(sigma_values)

        # Linearity error statistics
        error_col = 'final_linearity_error_shifted' if 'final_linearity_error_shifted' in data.columns else 'linearity_error'
        if error_col in data.columns:
            error_values = data[error_col].dropna().abs()
            if len(error_values) > 0:
                profile.linearity_error = self._calculate_statistics(error_values)

        # Quality metrics
        if 'linearity_pass' in data.columns:
            linearity_pass = data['linearity_pass'].dropna()
            if len(linearity_pass) > 0:
                profile.linearity_pass_rate = float(linearity_pass.mean())

        if 'sigma_pass' in data.columns:
            sigma_pass = data['sigma_pass'].dropna()
            if len(sigma_pass) > 0:
                profile.sigma_pass_rate = float(sigma_pass.mean())

        # Overall pass rate (both must pass)
        if 'linearity_pass' in data.columns and 'sigma_pass' in data.columns:
            both_pass = data['linearity_pass'] & data['sigma_pass']
            profile.pass_rate = float(both_pass.mean())
            profile.fail_rate = 1.0 - profile.pass_rate

        # Fail points analysis
        if 'linearity_fail_points' in data.columns:
            failed = data[~data.get('linearity_pass', pd.Series([True] * len(data)))]
            if len(failed) > 0:
                profile.avg_fail_points = float(failed['linearity_fail_points'].mean())

        # Spec analysis
        if 'linearity_spec' in data.columns:
            specs = data['linearity_spec'].dropna()
            if len(specs) > 0:
                # Use most common spec (mode), fall back to mean if mode is empty
                mode_result = specs.mode()
                if len(mode_result) > 0:
                    profile.linearity_spec = float(mode_result.iloc[0])
                else:
                    profile.linearity_spec = float(specs.mean())

                # Calculate margin to spec
                if error_col in data.columns and profile.linearity_spec > 0:
                    errors = data[error_col].dropna().abs()
                    margins = (profile.linearity_spec - errors) / profile.linearity_spec * 100
                    profile.spec_margin_percent = float(margins.mean())
                    profile.tight_margin_count = int((margins < 10).sum())

        # Track correlation (if paired data available)
        if track_pairs is not None and len(track_pairs) > 0:
            if 'sigma_t1' in track_pairs.columns and 'sigma_t2' in track_pairs.columns:
                t1 = track_pairs['sigma_t1'].dropna()
                t2 = track_pairs['sigma_t2'].dropna()
                if len(t1) > 10 and len(t2) > 10:
                    # Align indices
                    common = t1.index.intersection(t2.index)
                    if len(common) > 10:
                        corr, _ = stats.pearsonr(t1[common], t2[common])
                        profile.track_correlation = float(corr)

        # Sigma vs error correlation
        if 'sigma_gradient' in data.columns and error_col in data.columns:
            sigma = data['sigma_gradient'].dropna()
            error = data[error_col].dropna().abs()
            common = sigma.index.intersection(error.index)
            if len(common) > 10:
                corr, _ = stats.pearsonr(sigma[common], error[common])
                profile.sigma_error_correlation = float(corr)

        # Date analysis
        if 'file_date' in data.columns:
            dates = pd.to_datetime(data['file_date'].dropna())
            if len(dates) > 0:
                profile.earliest_date = dates.min().to_pydatetime()
                profile.latest_date = dates.max().to_pydatetime()
                profile.date_range_days = (profile.latest_date - profile.earliest_date).days

        profile.profiled_date = datetime.now()
        self.profile = profile
        self.is_profiled = True

        # Generate insights
        self._generate_insights()

        logger.info(
            f"ModelProfiler[{self.model_name}] complete - "
            f"Samples: {profile.sample_count}, Pass rate: {profile.pass_rate:.1%}, "
            f"Sigma mean: {profile.sigma.mean if profile.sigma else 0:.6f}"
        )

        return profile

    def _calculate_statistics(self, values: pd.Series) -> ProfileStatistics:
        """Calculate comprehensive statistics for a series of values."""
        arr = np.array(values)

        return ProfileStatistics(
            count=len(arr),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p5=float(np.percentile(arr, 5)),
            p25=float(np.percentile(arr, 25)),
            p50=float(np.percentile(arr, 50)),
            p75=float(np.percentile(arr, 75)),
            p95=float(np.percentile(arr, 95)),
            skewness=float(stats.skew(arr)) if len(arr) > 2 else 0.0,
            kurtosis=float(stats.kurtosis(arr)) if len(arr) > 3 else 0.0,
        )

    def set_comparative_metrics(
        self,
        difficulty_score: float,
        quality_percentile: float
    ) -> None:
        """
        Set comparative metrics (calculated across all models).

        Args:
            difficulty_score: 0-1, where 1 = hardest model to trim
            quality_percentile: 0-1, quality ranking among all models
        """
        if self.profile:
            self.profile.difficulty_score = difficulty_score
            self.profile.quality_percentile = quality_percentile
            self._generate_insights()  # Regenerate with new info

    def _generate_insights(self) -> None:
        """Generate human-readable insights from profile."""
        self.insights = []

        if not self.profile:
            return

        p = self.profile

        # Quality insights
        if p.pass_rate < 0.9:
            severity = 'critical' if p.pass_rate < 0.8 else 'warning'
            self.insights.append(ModelInsight(
                category='quality',
                severity=severity,
                message=f"Low pass rate: {p.pass_rate:.1%} (below 90% target)",
                metric_name='pass_rate',
                metric_value=p.pass_rate
            ))
        elif p.pass_rate > 0.98:
            self.insights.append(ModelInsight(
                category='quality',
                severity='info',
                message=f"Excellent pass rate: {p.pass_rate:.1%}",
                metric_name='pass_rate',
                metric_value=p.pass_rate
            ))

        # Spec margin insights
        if p.spec_margin_percent < 10:
            self.insights.append(ModelInsight(
                category='spec',
                severity='warning',
                message=f"Tight spec margin: {p.spec_margin_percent:.1f}% average margin to spec",
                metric_name='spec_margin_percent',
                metric_value=p.spec_margin_percent
            ))

        if p.tight_margin_count > p.sample_count * 0.1:
            self.insights.append(ModelInsight(
                category='spec',
                severity='warning',
                message=f"{p.tight_margin_count} units ({p.tight_margin_count/p.sample_count:.1%}) within 10% of spec limit",
                metric_name='tight_margin_count',
                metric_value=float(p.tight_margin_count)
            ))

        # Correlation insights
        if abs(p.track_correlation) > 0.8:
            self.insights.append(ModelInsight(
                category='correlation',
                severity='info',
                message=f"High Track 1/Track 2 correlation: {p.track_correlation:.2f} - failures likely correlated",
                metric_name='track_correlation',
                metric_value=p.track_correlation
            ))

        # Difficulty insights
        if p.difficulty_score > 0.8:
            self.insights.append(ModelInsight(
                category='comparison',
                severity='warning',
                message=f"High difficulty model - harder to trim than {(1-p.difficulty_score)*100:.0f}% of models",
                metric_name='difficulty_score',
                metric_value=p.difficulty_score
            ))
        elif p.difficulty_score < 0.2:
            self.insights.append(ModelInsight(
                category='comparison',
                severity='info',
                message=f"Low difficulty model - easier than {(1-p.difficulty_score)*100:.0f}% of models",
                metric_name='difficulty_score',
                metric_value=p.difficulty_score
            ))

        # Sample size insights
        if p.sample_count < 50:
            self.insights.append(ModelInsight(
                category='data',
                severity='info',
                message=f"Limited data: {p.sample_count} samples - more data will improve ML accuracy",
                metric_name='sample_count',
                metric_value=float(p.sample_count)
            ))

        # Severity insights
        if p.avg_fail_points > 3:
            self.insights.append(ModelInsight(
                category='quality',
                severity='warning',
                message=f"High failure severity: average {p.avg_fail_points:.1f} fail points per failure",
                metric_name='avg_fail_points',
                metric_value=p.avg_fail_points
            ))

    def get_insights(self, category: Optional[str] = None) -> List[ModelInsight]:
        """
        Get insights, optionally filtered by category.

        Args:
            category: Optional category filter ('quality', 'spec', 'correlation', 'comparison', 'data')

        Returns:
            List of ModelInsight objects
        """
        if category:
            return [i for i in self.insights if i.category == category]
        return self.insights.copy()

    def get_insight_summary(self) -> str:
        """Get a brief text summary of key insights."""
        if not self.insights:
            return f"Model {self.model_name}: No significant issues detected"

        critical = [i for i in self.insights if i.severity == 'critical']
        warnings = [i for i in self.insights if i.severity == 'warning']

        parts = [f"Model {self.model_name}:"]
        if critical:
            parts.append(f" {len(critical)} critical issues")
        if warnings:
            parts.append(f" {len(warnings)} warnings")

        return ''.join(parts)

    def get_profile_dict(self) -> Dict[str, Any]:
        """Get profile as dictionary for storage/display."""
        if not self.profile:
            return {'model_name': self.model_name, 'is_profiled': False}

        p = self.profile

        return {
            'model_name': p.model_name,
            'sample_count': p.sample_count,
            'trim_count': p.trim_count,
            'final_test_count': p.final_test_count,

            # Sigma stats
            'sigma_mean': p.sigma.mean if p.sigma else None,
            'sigma_std': p.sigma.std if p.sigma else None,
            'sigma_p5': p.sigma.p5 if p.sigma else None,
            'sigma_p50': p.sigma.p50 if p.sigma else None,
            'sigma_p95': p.sigma.p95 if p.sigma else None,

            # Error stats
            'error_mean': p.linearity_error.mean if p.linearity_error else None,
            'error_std': p.linearity_error.std if p.linearity_error else None,

            # Quality
            'pass_rate': p.pass_rate,
            'fail_rate': p.fail_rate,
            'linearity_pass_rate': p.linearity_pass_rate,
            'sigma_pass_rate': p.sigma_pass_rate,
            'avg_fail_points': p.avg_fail_points,

            # Spec
            'linearity_spec': p.linearity_spec,
            'spec_margin_percent': p.spec_margin_percent,
            'tight_margin_count': p.tight_margin_count,

            # Correlations
            'track_correlation': p.track_correlation,
            'sigma_error_correlation': p.sigma_error_correlation,

            # Comparative
            'difficulty_score': p.difficulty_score,
            'quality_percentile': p.quality_percentile,

            # Time
            'earliest_date': p.earliest_date.isoformat() if p.earliest_date else None,
            'latest_date': p.latest_date.isoformat() if p.latest_date else None,
            'date_range_days': p.date_range_days,

            'profiled_date': p.profiled_date.isoformat() if p.profiled_date else None,
            'is_profiled': True,
        }


def calculate_cross_model_metrics(
    profiles: Dict[str, ModelProfiler]
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate comparative metrics across all models.

    Args:
        profiles: Dict of model_name -> ModelProfiler

    Returns:
        Dict of model_name -> (difficulty_score, quality_percentile)
    """
    if not profiles:
        return {}

    # Collect pass rates and sigma means
    model_data = {}
    for model_name, profiler in profiles.items():
        if profiler.profile and profiler.profile.sample_count > 0:
            model_data[model_name] = {
                'pass_rate': profiler.profile.pass_rate,
                'sigma_mean': profiler.profile.sigma.mean if profiler.profile.sigma else 0,
                'fail_rate': profiler.profile.fail_rate,
            }

    if not model_data:
        return {}

    # Calculate difficulty (based on fail rate and sigma)
    # Higher fail rate + higher sigma = more difficult
    fail_rates = [d['fail_rate'] for d in model_data.values()]
    sigma_means = [d['sigma_mean'] for d in model_data.values()]

    fail_min, fail_max = min(fail_rates), max(fail_rates)
    sigma_min, sigma_max = min(sigma_means), max(sigma_means)

    result = {}
    for model_name, data in model_data.items():
        # Normalize fail rate to 0-1
        if fail_max > fail_min:
            fail_norm = (data['fail_rate'] - fail_min) / (fail_max - fail_min)
        else:
            fail_norm = 0.5

        # Normalize sigma to 0-1
        if sigma_max > sigma_min:
            sigma_norm = (data['sigma_mean'] - sigma_min) / (sigma_max - sigma_min)
        else:
            sigma_norm = 0.5

        # Difficulty = weighted combination
        difficulty = 0.6 * fail_norm + 0.4 * sigma_norm

        # Quality percentile (inverse of difficulty, but based on pass rate)
        # Rank models by pass rate
        sorted_by_quality = sorted(model_data.items(), key=lambda x: x[1]['pass_rate'])
        rank = next(i for i, (name, _) in enumerate(sorted_by_quality) if name == model_name)
        quality_percentile = rank / len(sorted_by_quality)

        result[model_name] = (difficulty, quality_percentile)

    return result
