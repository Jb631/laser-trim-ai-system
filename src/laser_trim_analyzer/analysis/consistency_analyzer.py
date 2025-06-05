"""
Consistency Analyzer for Multi-Track Units

Analyzes consistency between multiple tracks in a unit,
identifying deviations and potential issues.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from statistics import stdev, mean

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyMetrics:
    """Metrics for track consistency analysis."""
    sigma_cv: float  # Coefficient of variation for sigma
    linearity_cv: float  # Coefficient of variation for linearity
    resistance_cv: float  # Coefficient of variation for resistance
    overall_consistency: str  # EXCELLENT, GOOD, FAIR, POOR
    issues: List[str]
    recommendations: List[str]


class ConsistencyAnalyzer:
    """Analyzes consistency between multiple tracks in a unit."""
    
    # Thresholds for consistency evaluation
    SIGMA_CV_EXCELLENT = 5.0
    SIGMA_CV_GOOD = 10.0
    SIGMA_CV_FAIR = 15.0
    
    LINEARITY_CV_EXCELLENT = 10.0
    LINEARITY_CV_GOOD = 20.0
    LINEARITY_CV_FAIR = 30.0
    
    RESISTANCE_CV_EXCELLENT = 2.0
    RESISTANCE_CV_GOOD = 5.0
    RESISTANCE_CV_FAIR = 10.0
    
    def __init__(self):
        """Initialize consistency analyzer."""
        self.logger = logger
        
    def analyze_tracks(self, tracks_data: Dict[str, Any]) -> ConsistencyMetrics:
        """
        Analyze consistency between multiple tracks.
        
        Args:
            tracks_data: Dictionary of track_id -> track_data
            
        Returns:
            ConsistencyMetrics with analysis results
        """
        if not tracks_data or len(tracks_data) < 2:
            return ConsistencyMetrics(
                sigma_cv=0.0,
                linearity_cv=0.0,
                resistance_cv=0.0,
                overall_consistency="N/A - Single Track",
                issues=["Only one track available - consistency analysis requires multiple tracks"],
                recommendations=[]
            )
            
        # Extract metrics from all tracks
        sigma_values = []
        linearity_values = []
        resistance_values = []
        
        for track_id, track_data in tracks_data.items():
            # Extract sigma gradient
            sigma = self._extract_value(track_data, ['sigma_gradient', 'sigma_analysis.sigma_gradient'])
            if sigma is not None:
                sigma_values.append(sigma)
                
            # Extract linearity error
            linearity = self._extract_value(track_data, ['linearity_error', 'linearity_analysis.final_linearity_error_shifted'])
            if linearity is not None:
                linearity_values.append(abs(linearity))
                
            # Extract resistance change
            resistance = self._extract_value(track_data, ['resistance_change', 'resistance_change_percent', 'unit_properties.resistance_change_percent'])
            if resistance is not None:
                resistance_values.append(abs(resistance))
        
        # Calculate coefficients of variation
        sigma_cv = self._calculate_cv(sigma_values)
        linearity_cv = self._calculate_cv(linearity_values)
        resistance_cv = self._calculate_cv(resistance_values)
        
        # Identify issues
        issues = []
        recommendations = []
        
        # Check sigma consistency
        if sigma_cv > self.SIGMA_CV_FAIR:
            issues.append(f"High sigma gradient variation (CV={sigma_cv:.1f}%) indicates process instability")
            recommendations.append("Investigate laser trimming process parameters for consistency")
        elif sigma_cv > self.SIGMA_CV_GOOD:
            issues.append(f"Moderate sigma gradient variation (CV={sigma_cv:.1f}%)")
            
        # Check linearity consistency
        if linearity_cv > self.LINEARITY_CV_FAIR:
            issues.append(f"High linearity error variation (CV={linearity_cv:.1f}%) suggests measurement or process issues")
            recommendations.append("Check measurement setup and calibration across tracks")
        elif linearity_cv > self.LINEARITY_CV_GOOD:
            issues.append(f"Moderate linearity error variation (CV={linearity_cv:.1f}%)")
            
        # Check resistance consistency
        if resistance_cv > self.RESISTANCE_CV_FAIR:
            issues.append(f"High resistance change variation (CV={resistance_cv:.1f}%) indicates material or process variation")
            recommendations.append("Verify substrate material consistency and trim conditions")
        elif resistance_cv > self.RESISTANCE_CV_GOOD:
            issues.append(f"Moderate resistance change variation (CV={resistance_cv:.1f}%)")
            
        # Check for outliers
        outlier_issues = self._check_outliers(tracks_data, sigma_values, linearity_values, resistance_values)
        issues.extend(outlier_issues['issues'])
        recommendations.extend(outlier_issues['recommendations'])
        
        # Determine overall consistency
        overall_consistency = self._determine_overall_consistency(sigma_cv, linearity_cv, resistance_cv)
        
        # Add general recommendations based on consistency
        if overall_consistency == "POOR":
            recommendations.append("Consider unit rework or additional quality checks")
            recommendations.append("Review manufacturing process parameters")
        elif overall_consistency == "FAIR":
            recommendations.append("Monitor unit performance closely")
            recommendations.append("Consider additional validation testing")
            
        # If no issues found
        if not issues:
            issues.append("All tracks show excellent consistency")
            
        return ConsistencyMetrics(
            sigma_cv=sigma_cv,
            linearity_cv=linearity_cv,
            resistance_cv=resistance_cv,
            overall_consistency=overall_consistency,
            issues=issues,
            recommendations=recommendations
        )
    
    def _extract_value(self, data: Dict[str, Any], paths: List[str]) -> Optional[float]:
        """Extract value from nested dictionary using multiple possible paths."""
        for path in paths:
            value = data
            for key in path.split('.'):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            if value is not None and isinstance(value, (int, float)):
                return float(value)
        return None
    
    def _calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation (CV) as percentage."""
        if not values or len(values) < 2:
            return 0.0
            
        avg = mean(values)
        if avg == 0:
            return 0.0
            
        std = stdev(values)
        return (std / abs(avg)) * 100
    
    def _check_outliers(self, tracks_data: Dict[str, Any], 
                       sigma_values: List[float], 
                       linearity_values: List[float], 
                       resistance_values: List[float]) -> Dict[str, List[str]]:
        """Check for outlier tracks."""
        issues = []
        recommendations = []
        
        # Use 2-sigma rule for outlier detection
        outlier_tracks = set()
        
        # Check sigma outliers
        if len(sigma_values) >= 3:
            sigma_mean = mean(sigma_values)
            sigma_std = stdev(sigma_values)
            
            for track_id, track_data in tracks_data.items():
                sigma = self._extract_value(track_data, ['sigma_gradient', 'sigma_analysis.sigma_gradient'])
                if sigma is not None and abs(sigma - sigma_mean) > 2 * sigma_std:
                    outlier_tracks.add(track_id)
                    issues.append(f"Track {track_id} has outlier sigma gradient ({sigma:.6f})")
        
        # Check linearity outliers
        if len(linearity_values) >= 3:
            lin_mean = mean(linearity_values)
            lin_std = stdev(linearity_values)
            
            for track_id, track_data in tracks_data.items():
                linearity = self._extract_value(track_data, ['linearity_error', 'linearity_analysis.final_linearity_error_shifted'])
                if linearity is not None and abs(abs(linearity) - lin_mean) > 2 * lin_std:
                    outlier_tracks.add(track_id)
                    issues.append(f"Track {track_id} has outlier linearity error ({abs(linearity):.3f}%)")
        
        if outlier_tracks:
            recommendations.append(f"Investigate outlier tracks: {', '.join(sorted(outlier_tracks))}")
            
        return {'issues': issues, 'recommendations': recommendations}
    
    def _determine_overall_consistency(self, sigma_cv: float, linearity_cv: float, resistance_cv: float) -> str:
        """Determine overall consistency rating."""
        # Count how many metrics are in each category
        excellent_count = 0
        good_count = 0
        fair_count = 0
        poor_count = 0
        
        # Evaluate sigma CV
        if sigma_cv <= self.SIGMA_CV_EXCELLENT:
            excellent_count += 1
        elif sigma_cv <= self.SIGMA_CV_GOOD:
            good_count += 1
        elif sigma_cv <= self.SIGMA_CV_FAIR:
            fair_count += 1
        else:
            poor_count += 1
            
        # Evaluate linearity CV
        if linearity_cv <= self.LINEARITY_CV_EXCELLENT:
            excellent_count += 1
        elif linearity_cv <= self.LINEARITY_CV_GOOD:
            good_count += 1
        elif linearity_cv <= self.LINEARITY_CV_FAIR:
            fair_count += 1
        else:
            poor_count += 1
            
        # Evaluate resistance CV
        if resistance_cv <= self.RESISTANCE_CV_EXCELLENT:
            excellent_count += 1
        elif resistance_cv <= self.RESISTANCE_CV_GOOD:
            good_count += 1
        elif resistance_cv <= self.RESISTANCE_CV_FAIR:
            fair_count += 1
        else:
            poor_count += 1
            
        # Determine overall rating
        if poor_count > 0:
            return "POOR"
        elif fair_count >= 2:
            return "FAIR"
        elif excellent_count >= 2:
            return "EXCELLENT"
        else:
            return "GOOD"
    
    def generate_consistency_report(self, metrics: ConsistencyMetrics) -> str:
        """Generate a formatted consistency report."""
        report = "CONSISTENCY ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Overall Consistency: {metrics.overall_consistency}\n\n"
        
        report += "Variation Metrics:\n"
        report += f"  • Sigma CV: {metrics.sigma_cv:.1f}%\n"
        report += f"  • Linearity CV: {metrics.linearity_cv:.1f}%\n"
        report += f"  • Resistance CV: {metrics.resistance_cv:.1f}%\n\n"
        
        report += "Issues Identified:\n"
        if metrics.issues:
            for i, issue in enumerate(metrics.issues, 1):
                report += f"  {i}. {issue}\n"
        else:
            report += "  No issues identified\n"
            
        report += "\nRecommendations:\n"
        if metrics.recommendations:
            for i, rec in enumerate(metrics.recommendations, 1):
                report += f"  {i}. {rec}\n"
        else:
            report += "  No specific recommendations\n"
            
        return report