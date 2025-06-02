"""
Calculation Validation Module for Laser Trim Analyzer

Validates calculations against industry standards for:
- Potentiometer linearity analysis
- Sigma gradient calculations  
- Independent linearity measurements
- Industry standard formulas and tolerances

Based on:
- VRCI (Variable Resistive Components Institute) standards
- Bourns technical specifications
- ATP Laser Resistor Trimming Design Rules
- IPC standards for electronic components
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    RELAXED = "relaxed"
    STANDARD = "standard"
    STRICT = "strict"


class CalculationType(Enum):
    """Types of calculations to validate"""
    SIGMA_GRADIENT = "sigma_gradient"
    LINEARITY_ERROR = "linearity_error"
    INDEPENDENT_LINEARITY = "independent_linearity"
    RESISTANCE_CALCULATION = "resistance_calculation"
    TEMPERATURE_COEFFICIENT = "temperature_coefficient"


@dataclass
class ValidationResult:
    """Result of calculation validation"""
    calculation_type: CalculationType
    is_valid: bool
    expected_value: float
    actual_value: float
    deviation_percent: float
    tolerance_used: float
    standard_reference: str
    warnings: List[str]
    recommendations: List[str]


@dataclass
class IndustryStandards:
    """Industry standard values and tolerances"""
    
    # Linearity tolerances (as percentage of full scale)
    LINEARITY_TOLERANCE_PRECISION = 0.1   # ±0.1% for precision pots
    LINEARITY_TOLERANCE_STANDARD = 0.5    # ±0.5% for standard pots  
    LINEARITY_TOLERANCE_COMMERCIAL = 2.0  # ±2.0% for commercial pots
    
    # Sigma gradient tolerances
    SIGMA_GRADIENT_MIN = 0.001  # Minimum acceptable sigma
    SIGMA_GRADIENT_MAX = 10.0   # Maximum acceptable sigma
    SIGMA_GRADIENT_TYPICAL = 0.1  # Typical good sigma value
    
    # Resistance calculation tolerances
    RESISTANCE_TOLERANCE_PRECISION = 0.01  # ±0.01% for precision
    RESISTANCE_TOLERANCE_STANDARD = 0.1    # ±0.1% for standard
    RESISTANCE_TOLERANCE_COMMERCIAL = 1.0  # ±1.0% for commercial
    
    # Temperature coefficient limits (ppm/°C)
    TEMP_COEFF_RESISTANCE_MAX = 200    # ppm/°C for resistance
    TEMP_COEFF_LINEARITY_MAX = 5       # ppm/°C for linearity
    
    # Contact resistance limits
    CONTACT_RESISTANCE_MAX = 100       # Ohms maximum
    CONTACT_RESISTANCE_TYPICAL = 10    # Ohms typical


class CalculationValidator:
    """Validates laser trim analyzer calculations against industry standards"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.standards = IndustryStandards()
        self.logger = logging.getLogger(__name__)
        
    def validate_sigma_gradient(self, 
                               calculated_sigma: float,
                               position_data: List[float],
                               error_data: List[float],
                               target_function: str = "linear") -> ValidationResult:
        """
        Validate sigma gradient calculation against industry standards.
        
        Sigma gradient is calculated as the standard deviation of the error
        between actual and theoretical resistance values.
        
        Industry standard: σ = sqrt(Σ(xi - x̄)² / (n-1))
        """
        warnings = []
        recommendations = []
        
        try:
            # Recalculate sigma using industry standard formula
            if len(position_data) != len(error_data):
                warnings.append("Position and error data arrays have different lengths")
                return ValidationResult(
                    calculation_type=CalculationType.SIGMA_GRADIENT,
                    is_valid=False,
                    expected_value=0.0,
                    actual_value=calculated_sigma,
                    deviation_percent=100.0,
                    tolerance_used=0.0,
                    standard_reference="Data validation failed",
                    warnings=warnings,
                    recommendations=["Ensure position and error data have same length"]
                )
            
            # Calculate theoretical values based on target function
            if target_function.lower() == "linear":
                # Linear: y = mx + b
                theoretical_values = self._calculate_linear_theoretical(position_data)
            else:
                # For other functions, use polynomial fit
                theoretical_values = self._calculate_polynomial_theoretical(position_data, error_data)
            
            # Calculate deviations from theoretical
            if len(theoretical_values) == len(error_data):
                deviations = np.array(error_data) - np.array(theoretical_values)
            else:
                deviations = np.array(error_data)
                warnings.append("Using raw error data due to theoretical calculation mismatch")
            
            # Industry standard sigma calculation: σ = sqrt(Σ(xi - x̄)² / (n-1))
            expected_sigma = float(np.std(deviations, ddof=1))  # Sample standard deviation
            
            # Calculate deviation percentage
            if expected_sigma > 0:
                deviation_percent = abs(calculated_sigma - expected_sigma) / expected_sigma * 100
            else:
                deviation_percent = 0.0 if calculated_sigma == 0 else 100.0
            
            # Determine tolerance based on validation level
            tolerance_map = {
                ValidationLevel.RELAXED: 10.0,   # ±10%
                ValidationLevel.STANDARD: 5.0,   # ±5%
                ValidationLevel.STRICT: 2.0      # ±2%
            }
            tolerance = tolerance_map[self.validation_level]
            
            # Check if within tolerance
            is_valid = deviation_percent <= tolerance
            
            # Generate recommendations based on sigma value
            if calculated_sigma > self.standards.SIGMA_GRADIENT_MAX:
                recommendations.append(f"Sigma gradient {calculated_sigma:.3f} exceeds maximum recommended value of {self.standards.SIGMA_GRADIENT_MAX}")
                recommendations.append("Consider reviewing trim parameters or equipment calibration")
            elif calculated_sigma > self.standards.SIGMA_GRADIENT_TYPICAL * 2:
                recommendations.append(f"Sigma gradient {calculated_sigma:.3f} is higher than typical good value of {self.standards.SIGMA_GRADIENT_TYPICAL}")
                recommendations.append("Monitor trim quality and consider process optimization")
            elif calculated_sigma < self.standards.SIGMA_GRADIENT_MIN:
                warnings.append(f"Sigma gradient {calculated_sigma:.3f} is unusually low - verify calculation")
            
            return ValidationResult(
                calculation_type=CalculationType.SIGMA_GRADIENT,
                is_valid=is_valid,
                expected_value=expected_sigma,
                actual_value=calculated_sigma,
                deviation_percent=deviation_percent,
                tolerance_used=tolerance,
                standard_reference="IEEE/VRCI Standard σ = sqrt(Σ(xi - x̄)² / (n-1))",
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error validating sigma gradient: {e}")
            return ValidationResult(
                calculation_type=CalculationType.SIGMA_GRADIENT,
                is_valid=False,
                expected_value=0.0,
                actual_value=calculated_sigma,
                deviation_percent=100.0,
                tolerance_used=0.0,
                standard_reference="Validation failed due to error",
                warnings=[f"Calculation error: {str(e)}"],
                recommendations=["Review input data and calculation method"]
            )
    
    def validate_linearity_error(self,
                                calculated_linearity: float,
                                position_data: List[float],
                                resistance_data: List[float]) -> ValidationResult:
        """
        Validate linearity error calculation against VRCI/Bourns standards.
        
        Independent linearity is the maximum deviation from the best-fit line
        that minimizes deviations over the electrical travel.
        """
        warnings = []
        recommendations = []
        
        try:
            if len(position_data) != len(resistance_data):
                warnings.append("Position and resistance data arrays have different lengths")
                return ValidationResult(
                    calculation_type=CalculationType.LINEARITY_ERROR,
                    is_valid=False,
                    expected_value=0.0,
                    actual_value=calculated_linearity,
                    deviation_percent=100.0,
                    tolerance_used=0.0,
                    standard_reference="Data validation failed",
                    warnings=warnings,
                    recommendations=["Ensure position and resistance data have same length"]
                )
            
            # Calculate independent linearity using best-fit line method
            # This is the industry standard approach per VRCI specifications
            positions = np.array(position_data)
            resistances = np.array(resistance_data)
            
            # Normalize positions to 0-1 range
            if len(positions) > 1:
                pos_normalized = (positions - positions.min()) / (positions.max() - positions.min())
            else:
                pos_normalized = np.array([0.0])
                warnings.append("Insufficient data points for linearity calculation")
            
            # Calculate best-fit line (slope and intercept chosen to minimize deviations)
            if len(pos_normalized) > 1:
                slope, intercept = np.polyfit(pos_normalized, resistances, 1)
                theoretical_resistances = slope * pos_normalized + intercept
                
                # Calculate deviations from best-fit line
                deviations = resistances - theoretical_resistances
                
                # Independent linearity = maximum deviation as % of full scale
                full_scale = resistances.max() - resistances.min() if len(resistances) > 1 else 1.0
                if full_scale > 0:
                    max_deviation = np.max(np.abs(deviations))
                    expected_linearity = (max_deviation / full_scale) * 100.0
                else:
                    expected_linearity = 0.0
                    warnings.append("Zero full-scale range detected")
            else:
                expected_linearity = 0.0
                warnings.append("Insufficient data for linearity calculation")
            
            # Calculate deviation percentage
            if expected_linearity > 0:
                deviation_percent = abs(calculated_linearity - expected_linearity) / expected_linearity * 100
            else:
                deviation_percent = 0.0 if calculated_linearity == 0 else 100.0
            
            # Determine tolerance and classification
            tolerance_map = {
                ValidationLevel.RELAXED: 15.0,   # ±15%
                ValidationLevel.STANDARD: 10.0,  # ±10%
                ValidationLevel.STRICT: 5.0      # ±5%
            }
            tolerance = tolerance_map[self.validation_level]
            
            is_valid = deviation_percent <= tolerance
            
            # Generate recommendations based on linearity value
            if calculated_linearity > self.standards.LINEARITY_TOLERANCE_COMMERCIAL:
                recommendations.append(f"Linearity error {calculated_linearity:.2f}% exceeds commercial grade tolerance")
                recommendations.append("Consider trim parameter optimization or equipment maintenance")
            elif calculated_linearity > self.standards.LINEARITY_TOLERANCE_STANDARD:
                recommendations.append(f"Linearity error {calculated_linearity:.2f}% exceeds standard grade tolerance")
                recommendations.append("Monitor trim quality for consistency")
            elif calculated_linearity <= self.standards.LINEARITY_TOLERANCE_PRECISION:
                recommendations.append(f"Excellent linearity {calculated_linearity:.2f}% meets precision grade standards")
            
            return ValidationResult(
                calculation_type=CalculationType.LINEARITY_ERROR,
                is_valid=is_valid,
                expected_value=expected_linearity,
                actual_value=calculated_linearity,
                deviation_percent=deviation_percent,
                tolerance_used=tolerance,
                standard_reference="VRCI/Bourns Independent Linearity Standard",
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error validating linearity error: {e}")
            return ValidationResult(
                calculation_type=CalculationType.LINEARITY_ERROR,
                is_valid=False,
                expected_value=0.0,
                actual_value=calculated_linearity,
                deviation_percent=100.0,
                tolerance_used=0.0,
                standard_reference="Validation failed due to error",
                warnings=[f"Calculation error: {str(e)}"],
                recommendations=["Review input data and calculation method"]
            )
    
    def validate_resistance_calculation(self,
                                      calculated_resistance: float,
                                      length: float,
                                      width: float,
                                      sheet_resistance: float) -> ValidationResult:
        """
        Validate resistance calculation using industry standard formula: R = Rs * (L/W)
        
        Based on ATP Laser Resistor Trimming Design Rules and industry standards.
        """
        warnings = []
        recommendations = []
        
        try:
            # Industry standard formula: R = Rs * (L/W)
            # Where Rs = sheet resistance (ohms/square), L = length, W = width
            if width <= 0:
                warnings.append("Invalid width value (must be positive)")
                return ValidationResult(
                    calculation_type=CalculationType.RESISTANCE_CALCULATION,
                    is_valid=False,
                    expected_value=0.0,
                    actual_value=calculated_resistance,
                    deviation_percent=100.0,
                    tolerance_used=0.0,
                    standard_reference="Data validation failed",
                    warnings=warnings,
                    recommendations=["Verify geometric measurements"]
                )
            
            # Calculate expected resistance using industry formula
            aspect_ratio = length / width  # Number of "squares"
            expected_resistance = sheet_resistance * aspect_ratio
            
            # Calculate deviation percentage
            if expected_resistance > 0:
                deviation_percent = abs(calculated_resistance - expected_resistance) / expected_resistance * 100
            else:
                deviation_percent = 100.0 if calculated_resistance != 0 else 0.0
            
            # Determine tolerance based on validation level
            tolerance_map = {
                ValidationLevel.RELAXED: self.standards.RESISTANCE_TOLERANCE_COMMERCIAL,
                ValidationLevel.STANDARD: self.standards.RESISTANCE_TOLERANCE_STANDARD,
                ValidationLevel.STRICT: self.standards.RESISTANCE_TOLERANCE_PRECISION
            }
            tolerance = tolerance_map[self.validation_level]
            
            is_valid = deviation_percent <= tolerance
            
            # Generate recommendations
            if deviation_percent > 5.0:
                recommendations.append("Large deviation suggests possible measurement or calculation error")
                recommendations.append("Verify sheet resistance value and geometric measurements")
            
            if aspect_ratio < 0.1:
                warnings.append("Very low aspect ratio - consider wider geometry for better accuracy")
            elif aspect_ratio > 100:
                warnings.append("Very high aspect ratio - consider design optimization")
            
            return ValidationResult(
                calculation_type=CalculationType.RESISTANCE_CALCULATION,
                is_valid=is_valid,
                expected_value=expected_resistance,
                actual_value=calculated_resistance,
                deviation_percent=deviation_percent,
                tolerance_used=tolerance,
                standard_reference="ATP Design Rules: R = Rs * (L/W)",
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error validating resistance calculation: {e}")
            return ValidationResult(
                calculation_type=CalculationType.RESISTANCE_CALCULATION,
                is_valid=False,
                expected_value=0.0,
                actual_value=calculated_resistance,
                deviation_percent=100.0,
                tolerance_used=0.0,
                standard_reference="Validation failed due to error",
                warnings=[f"Calculation error: {str(e)}"],
                recommendations=["Review input parameters and calculation method"]
            )
    
    def _calculate_linear_theoretical(self, position_data: List[float]) -> List[float]:
        """Calculate theoretical linear values for given positions"""
        if len(position_data) < 2:
            return position_data.copy()
        
        positions = np.array(position_data)
        min_pos, max_pos = positions.min(), positions.max()
        
        if max_pos == min_pos:
            return [0.0] * len(position_data)
        
        # Linear interpolation from 0 to 1 over the range
        normalized = (positions - min_pos) / (max_pos - min_pos)
        return normalized.tolist()
    
    def _calculate_polynomial_theoretical(self, position_data: List[float], 
                                        error_data: List[float], degree: int = 2) -> List[float]:
        """Calculate theoretical values using polynomial fit"""
        if len(position_data) < degree + 1:
            return self._calculate_linear_theoretical(position_data)
        
        try:
            positions = np.array(position_data)
            errors = np.array(error_data)
            
            # Fit polynomial
            coeffs = np.polyfit(positions, errors, degree)
            theoretical = np.polyval(coeffs, positions)
            
            return theoretical.tolist()
        except:
            return self._calculate_linear_theoretical(position_data)
    
    def validate_multi_track_consistency(self,
                                       track_data: Dict[str, Dict]) -> Dict[str, ValidationResult]:
        """
        Validate multi-track unit consistency.
        
        Checks that multiple tracks on the same unit show consistent performance
        within acceptable industry tolerances.
        """
        results = {}
        
        if len(track_data) < 2:
            return {"insufficient_tracks": ValidationResult(
                calculation_type=CalculationType.SIGMA_GRADIENT,
                is_valid=True,
                expected_value=0.0,
                actual_value=0.0,
                deviation_percent=0.0,
                tolerance_used=0.0,
                standard_reference="Single track unit",
                warnings=["Only one track available - no multi-track validation possible"],
                recommendations=["Multi-track validation requires 2+ tracks"]
            )}
        
        # Extract metrics from all tracks
        sigma_values = []
        linearity_values = []
        resistance_values = []
        
        for track_id, data in track_data.items():
            if 'sigma_gradient' in data:
                sigma_values.append(data['sigma_gradient'])
            if 'linearity_error' in data:
                linearity_values.append(data['linearity_error'])
            if 'resistance' in data:
                resistance_values.append(data['resistance'])
        
        # Validate sigma consistency
        if len(sigma_values) >= 2:
            sigma_cv = (np.std(sigma_values) / np.mean(sigma_values)) * 100 if np.mean(sigma_values) > 0 else 0
            results['sigma_consistency'] = ValidationResult(
                calculation_type=CalculationType.SIGMA_GRADIENT,
                is_valid=sigma_cv <= 20.0,  # 20% CV is reasonable for multi-track
                expected_value=np.mean(sigma_values),
                actual_value=sigma_cv,
                deviation_percent=sigma_cv,
                tolerance_used=20.0,
                standard_reference="Multi-track sigma consistency (CV < 20%)",
                warnings=["High sigma variation between tracks"] if sigma_cv > 20.0 else [],
                recommendations=["Review trim parameters for consistency"] if sigma_cv > 20.0 else ["Good track-to-track consistency"]
            )
        
        # Validate linearity consistency
        if len(linearity_values) >= 2:
            linearity_cv = (np.std(linearity_values) / np.mean(linearity_values)) * 100 if np.mean(linearity_values) > 0 else 0
            results['linearity_consistency'] = ValidationResult(
                calculation_type=CalculationType.LINEARITY_ERROR,
                is_valid=linearity_cv <= 25.0,  # 25% CV for linearity
                expected_value=np.mean(linearity_values),
                actual_value=linearity_cv,
                deviation_percent=linearity_cv,
                tolerance_used=25.0,
                standard_reference="Multi-track linearity consistency (CV < 25%)",
                warnings=["High linearity variation between tracks"] if linearity_cv > 25.0 else [],
                recommendations=["Check mechanical alignment"] if linearity_cv > 25.0 else ["Good linearity consistency"]
            )
        
        return results
    
    def generate_validation_report(self, validation_results: List[ValidationResult]) -> str:
        """Generate a comprehensive validation report"""
        report = []
        report.append("=" * 60)
        report.append("LASER TRIM ANALYZER CALCULATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Validation Level: {self.validation_level.value.upper()}")
        report.append("")
        
        # Summary
        total_checks = len(validation_results)
        passed_checks = sum(1 for r in validation_results if r.is_valid)
        report.append(f"SUMMARY: {passed_checks}/{total_checks} validations passed")
        report.append("")
        
        # Detailed results
        for result in validation_results:
            report.append(f"Calculation: {result.calculation_type.value.replace('_', ' ').title()}")
            report.append(f"Status: {'PASS' if result.is_valid else 'FAIL'}")
            report.append(f"Expected: {result.expected_value:.6f}")
            report.append(f"Actual: {result.actual_value:.6f}")
            report.append(f"Deviation: {result.deviation_percent:.2f}% (tolerance: ±{result.tolerance_used:.1f}%)")
            report.append(f"Standard: {result.standard_reference}")
            
            if result.warnings:
                report.append("Warnings:")
                for warning in result.warnings:
                    report.append(f"  • {warning}")
            
            if result.recommendations:
                report.append("Recommendations:")
                for rec in result.recommendations:
                    report.append(f"  • {rec}")
            
            report.append("-" * 40)
        
        return "\n".join(report) 