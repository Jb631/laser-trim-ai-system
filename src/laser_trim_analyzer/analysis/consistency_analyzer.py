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
            # Log the structure for debugging
            self.logger.debug(f"Analyzing track {track_id}, data type: {type(track_data)}")
            if isinstance(track_data, dict):
                self.logger.debug(f"Track {track_id} keys: {list(track_data.keys())}")
                # Log nested structure if present
                if 'sigma_analysis' in track_data and isinstance(track_data['sigma_analysis'], dict):
                    self.logger.debug(f"Track {track_id} sigma_analysis keys: {list(track_data['sigma_analysis'].keys())}")
                if 'linearity_analysis' in track_data and isinstance(track_data['linearity_analysis'], dict):
                    self.logger.debug(f"Track {track_id} linearity_analysis keys: {list(track_data['linearity_analysis'].keys())}")
                if 'unit_properties' in track_data and isinstance(track_data['unit_properties'], dict):
                    self.logger.debug(f"Track {track_id} unit_properties keys: {list(track_data['unit_properties'].keys())}")
            
            # Extract sigma gradient
            sigma = self._extract_value(track_data, ['sigma_gradient', 'sigma_analysis.sigma_gradient'])
            if sigma is not None:
                sigma_values.append(sigma)
                self.logger.info(f"Track {track_id} sigma: {sigma}")
            else:
                self.logger.warning(f"Track {track_id}: No sigma gradient found")
                
            # Extract linearity error - don't take absolute value yet to preserve variations
            linearity = self._extract_value(track_data, ['linearity_error', 'linearity_analysis.final_linearity_error_shifted'])
            if linearity is not None:
                # Store the actual value, not absolute
                linearity_values.append(linearity)
                self.logger.info(f"Track {track_id} linearity: {linearity} (raw value)")
            else:
                self.logger.warning(f"Track {track_id}: No linearity error found")
                
            # Extract resistance change
            resistance = self._extract_value(track_data, ['resistance_change_percent', 'unit_properties.resistance_change_percent', 'resistance_change'])
            if resistance is not None:
                resistance_values.append(resistance)
                self.logger.info(f"Track {track_id} resistance: {resistance}%")
            else:
                self.logger.warning(f"Track {track_id}: No resistance change found")
        
        self.logger.info(f"Consistency analysis summary: {len(sigma_values)} sigma, {len(linearity_values)} linearity, {len(resistance_values)} resistance values")
        if sigma_values:
            self.logger.info(f"Sigma values: {sigma_values}")
        if linearity_values:
            self.logger.info(f"Linearity values: {linearity_values}")
        if resistance_values:
            self.logger.info(f"Resistance values: {resistance_values}")
        
        # Calculate coefficients of variation
        sigma_cv = self._calculate_cv(sigma_values)
        # For linearity, use absolute values for CV calculation
        linearity_cv = self._calculate_cv([abs(v) for v in linearity_values]) if linearity_values else 0.0
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
        if not isinstance(data, dict):
            self.logger.debug(f"Data is not a dictionary: {type(data)}")
            return None
            
        for path in paths:
            value = data
            path_parts = path.split('.')
            
            # Navigate through the path
            for i, key in enumerate(path_parts):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    # Log what went wrong
                    if not isinstance(value, dict):
                        self.logger.debug(f"Path {path} failed at part {i} ({key}): parent is {type(value)}, not dict")
                    else:
                        self.logger.debug(f"Path {path} failed at part {i}: key '{key}' not found in {list(value.keys())}")
                    value = None
                    break
                    
            # Check if we found a valid numeric value
            if value is not None:
                if isinstance(value, (int, float)):
                    self.logger.debug(f"Found value {value} at path {path}")
                    return float(value)
                else:
                    self.logger.debug(f"Value at path {path} is not numeric: {type(value)} = {value}")
                    
        self.logger.debug(f"No value found in any of the paths: {paths}")
        return None
    
    def _calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation (CV) as percentage."""
        if not values:
            self.logger.debug("CV calculation: No values provided")
            return 0.0
            
        if len(values) < 2:
            self.logger.debug(f"CV calculation: Only {len(values)} value(s), need at least 2")
            return 0.0
        
        # Check if all values are identical
        if len(set(values)) == 1:
            self.logger.info(f"CV calculation: All {len(values)} values are identical ({values[0]})")
            return 0.0
            
        avg = mean(values)
        std = stdev(values)
        
        self.logger.debug(f"CV calculation: mean={avg}, std={std}, n={len(values)}")
        
        # Handle near-zero mean
        if abs(avg) < 1e-10:
            self.logger.warning(f"CV calculation: Mean is near zero ({avg}), using alternative calculation")
            # For near-zero mean, report relative to the range
            value_range = max(values) - min(values)
            if value_range > 0:
                return (std / value_range) * 100
            else:
                return 0.0
            
        cv = (std / abs(avg)) * 100
        self.logger.info(f"CV calculation result: {cv:.2f}%")
        return cv
    
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
        
        # Sigma CV with interpretation
        if metrics.sigma_cv == 0.0:
            report += f"  • Sigma CV: {metrics.sigma_cv:.1f}% (All tracks have identical sigma gradients)\n"
        else:
            report += f"  • Sigma CV: {metrics.sigma_cv:.1f}%\n"
            
        # Linearity CV with interpretation
        if metrics.linearity_cv == 0.0:
            report += f"  • Linearity CV: {metrics.linearity_cv:.1f}% (All tracks have identical linearity errors)\n"
        else:
            report += f"  • Linearity CV: {metrics.linearity_cv:.1f}%\n"
            
        # Resistance CV with interpretation
        if metrics.resistance_cv == 0.0:
            report += f"  • Resistance CV: {metrics.resistance_cv:.1f}% (All tracks have identical resistance changes)\n"
        else:
            report += f"  • Resistance CV: {metrics.resistance_cv:.1f}%\n"
            
        report += "\n"
        
        # Add note if all CVs are zero
        if metrics.sigma_cv == 0.0 and metrics.linearity_cv == 0.0 and metrics.resistance_cv == 0.0:
            report += "NOTE: All variation metrics show 0.0%, indicating either:\n"
            report += "  - All tracks have identical values (perfect consistency)\n"
            report += "  - Data may be missing or not properly loaded\n"
            report += "  - Check the log files for detailed analysis information\n\n"
        
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