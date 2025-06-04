# src/laser_trim_analyzer/core/models.py
"""
Pydantic data models for the Laser Trim Analyzer.

These models provide type validation and serialization for all analysis data.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator, computed_field
import numpy as np


class SystemType(str, Enum):
    """Laser trim system types."""
    SYSTEM_A = "A"
    SYSTEM_B = "B"
    UNKNOWN = "Unknown"


class RiskCategory(str, Enum):
    """Risk assessment categories."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"


class AnalysisStatus(str, Enum):
    """Overall analysis status."""
    PASS = "Pass"
    FAIL = "Fail"
    WARNING = "Warning"
    ERROR = "Error"
    PENDING = "Pending"


class ProcessingMode(str, Enum):
    """Processing mode options."""
    DETAIL = "detail"  # Sequential with plots
    SPEED = "speed"  # Parallel without plots


class ValidationStatus(str, Enum):
    """Validation status for calculations."""
    VALIDATED = "Validated"
    WARNING = "Warning"
    FAILED = "Failed"
    NOT_VALIDATED = "Not Validated"


# Base configuration for all models
class BaseAnalysisModel(BaseModel):
    """Base model with common configuration."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v),
            np.ndarray: lambda v: v.tolist(),
        }
    )


class ValidationResult(BaseAnalysisModel):
    """Result of calculation validation against industry standards."""
    calculation_type: str = Field(..., description="Type of calculation validated")
    is_valid: bool = Field(..., description="Whether calculation passes validation")
    expected_value: float = Field(..., description="Expected value per industry standards")
    actual_value: float = Field(..., description="Actual calculated value")
    deviation_percent: float = Field(..., description="Percentage deviation from expected")
    tolerance_used: float = Field(..., description="Tolerance threshold used")
    standard_reference: str = Field(..., description="Industry standard reference")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    recommendations: List[str] = Field(default_factory=list, description="Industry recommendations")
    validation_status: ValidationStatus = Field(..., description="Overall validation status")

    @computed_field
    @property
    def validation_grade(self) -> str:
        """Get validation grade (A-F)."""
        if not self.is_valid:
            return "F"
        elif self.deviation_percent <= 1.0:
            return "A"
        elif self.deviation_percent <= 3.0:
            return "B"
        elif self.deviation_percent <= 5.0:
            return "C"
        elif self.deviation_percent <= 10.0:
            return "D"
        else:
            return "E"


class FileMetadata(BaseAnalysisModel):
    """File information and metadata."""
    filename: str = Field(..., description="Name of the file")
    file_path: Path = Field(..., description="Full path to the file")
    file_date: datetime = Field(..., description="File modification date")
    model: str = Field(..., description="Potentiometer model number")
    serial: str = Field(..., description="Unit serial number")
    system: SystemType = Field(..., description="Trim system type")
    has_multi_tracks: bool = Field(default=False, description="Whether file contains multiple tracks")
    track_identifier: Optional[str] = Field(None, description="Track identifier for System B multi-track (e.g., TA, TB)")

    @field_validator('file_path')
    @classmethod
    def validate_file_exists(cls, v: Path) -> Path:
        """Ensure file exists."""
        if not v.exists():
            raise ValueError(f"File does not exist: {v}")
        return v

    @computed_field
    @property
    def file_size_mb(self) -> float:
        """Calculate file size in MB."""
        return self.file_path.stat().st_size / (1024 * 1024)


class UnitProperties(BaseAnalysisModel):
    """Physical properties of the potentiometer unit."""
    unit_length: Optional[float] = Field(None, ge=0, description="Unit angle/length in degrees")
    untrimmed_resistance: Optional[float] = Field(None, ge=0, description="Resistance before trimming (Ω)")
    trimmed_resistance: Optional[float] = Field(None, ge=0, description="Resistance after trimming (Ω)")

    @computed_field
    @property
    def resistance_change(self) -> Optional[float]:
        """Calculate absolute resistance change."""
        if self.untrimmed_resistance and self.trimmed_resistance:
            return self.trimmed_resistance - self.untrimmed_resistance
        return None

    @computed_field
    @property
    def resistance_change_percent(self) -> Optional[float]:
        """Calculate percentage resistance change."""
        if self.untrimmed_resistance and self.trimmed_resistance and self.untrimmed_resistance > 0:
            return ((self.trimmed_resistance - self.untrimmed_resistance) / self.untrimmed_resistance) * 100
        return None


class SigmaAnalysis(BaseAnalysisModel):
    """Sigma gradient analysis results."""
    sigma_gradient: float = Field(..., description="Calculated sigma gradient")
    sigma_threshold: float = Field(..., description="Sigma threshold value")
    sigma_pass: bool = Field(..., description="Whether sigma test passed")
    gradient_margin: float = Field(..., description="Margin to threshold")
    scaling_factor: float = Field(default=24.0, description="Sigma scaling factor used")
    
    # Validation fields
    validation_result: Optional[ValidationResult] = Field(None, description="Industry standard validation")
    validation_status: ValidationStatus = Field(default=ValidationStatus.NOT_VALIDATED, description="Validation status")

    @computed_field
    @property
    def sigma_ratio(self) -> float:
        """Calculate ratio of gradient to threshold."""
        return self.sigma_gradient / self.sigma_threshold if self.sigma_threshold > 0 else float('inf')

    @computed_field
    @property
    def industry_compliance(self) -> str:
        """Get industry compliance level."""
        if not self.validation_result:
            return "Not Validated"
        
        if self.validation_result.is_valid:
            if self.validation_result.deviation_percent <= 2.0:
                return "Precision Grade"
            elif self.validation_result.deviation_percent <= 5.0:
                return "Standard Grade"
            else:
                return "Commercial Grade"
        else:
            return "Non-Compliant"


class LinearityAnalysis(BaseAnalysisModel):
    """Linearity analysis results."""
    linearity_spec: float = Field(..., ge=0, description="Linearity specification")
    optimal_offset: float = Field(..., description="Optimal offset for linearity")
    final_linearity_error_raw: float = Field(..., ge=0, description="Raw linearity error")
    final_linearity_error_shifted: float = Field(..., ge=0, description="Shifted linearity error")
    linearity_pass: bool = Field(..., description="Whether linearity test passed")
    linearity_fail_points: int = Field(default=0, ge=0, description="Number of failing points")
    max_deviation: Optional[float] = Field(None, description="Maximum deviation from linear")
    max_deviation_position: Optional[float] = Field(None, description="Position of max deviation")
    
    # Validation fields
    validation_result: Optional[ValidationResult] = Field(None, description="Industry standard validation")
    validation_status: ValidationStatus = Field(default=ValidationStatus.NOT_VALIDATED, description="Validation status")

    @computed_field
    @property
    def industry_grade(self) -> str:
        """Get industry grade classification."""
        if not self.validation_result:
            return "Not Classified"
        
        linearity_percent = self.final_linearity_error_shifted
        if linearity_percent <= 0.1:
            return "Precision Grade (±0.1%)"
        elif linearity_percent <= 0.5:
            return "Standard Grade (±0.5%)"
        elif linearity_percent <= 2.0:
            return "Commercial Grade (±2.0%)"
        else:
            return "Below Commercial Grade"


class ResistanceAnalysis(BaseAnalysisModel):
    """Resistance change analysis results."""
    untrimmed_resistance: Optional[float] = Field(None, ge=0)
    trimmed_resistance: Optional[float] = Field(None, ge=0)
    resistance_change: Optional[float] = Field(None)
    resistance_change_percent: Optional[float] = Field(None)
    
    # Validation fields
    validation_result: Optional[ValidationResult] = Field(None, description="Industry standard validation")
    validation_status: ValidationStatus = Field(default=ValidationStatus.NOT_VALIDATED, description="Validation status")

    @field_validator('resistance_change_percent')
    @classmethod
    def validate_percent(cls, v: Optional[float]) -> Optional[float]:
        """Ensure percentage is reasonable."""
        if v is not None and abs(v) > 100:
            raise ValueError(f"Resistance change percent seems unrealistic: {v}%")
        return v

    @computed_field
    @property
    def resistance_stability_grade(self) -> str:
        """Get resistance stability grade appropriate for laser trimming processes."""
        if self.resistance_change_percent is None:
            return "Not Available"
        
        abs_change = abs(self.resistance_change_percent)
        
        # Laser trimming appropriate thresholds
        # Large changes are normal and expected for achieving target specs
        if abs_change <= 50.0:  # Up to 50% change is acceptable for laser trimming
            if abs_change <= 5.0:
                return "Minimal Trim (<5%)"
            elif abs_change <= 15.0:
                return "Light Trim (5-15%)"
            elif abs_change <= 30.0:
                return "Standard Trim (15-30%)"
            else:
                return "Heavy Trim (30-50%)"
        else:
            # Only flag as concerning if change is extreme (>50%)
            return "Extreme Trim (>50%)"


class TrimEffectiveness(BaseAnalysisModel):
    """Trim process effectiveness metrics."""
    improvement_percent: Optional[float] = Field(None, description="Overall improvement percentage")
    untrimmed_rms_error: Optional[float] = Field(None, ge=0, description="RMS error before trim")
    trimmed_rms_error: Optional[float] = Field(None, ge=0, description="RMS error after trim")
    max_error_reduction_percent: Optional[float] = Field(None, description="Maximum error reduction")
    
    # Validation fields
    validation_result: Optional[ValidationResult] = Field(None, description="Trim effectiveness validation")
    validation_status: ValidationStatus = Field(default=ValidationStatus.NOT_VALIDATED, description="Validation status")

    @computed_field
    @property
    def trim_quality_grade(self) -> str:
        """Get trim quality grade."""
        if self.improvement_percent is None:
            return "Not Available"
        
        if self.improvement_percent >= 80:
            return "Excellent (≥80%)"
        elif self.improvement_percent >= 60:
            return "Good (≥60%)"
        elif self.improvement_percent >= 40:
            return "Fair (≥40%)"
        else:
            return "Poor (<40%)"


class ZoneAnalysis(BaseAnalysisModel):
    """Travel zone analysis results."""
    num_zones: int = Field(default=5, ge=1, le=20, description="Number of analysis zones")
    worst_zone: Optional[int] = Field(None, ge=1, description="Zone with highest error")
    worst_zone_position: Optional[tuple[float, float]] = Field(None, description="Position range of worst zone")
    zone_results: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed zone metrics")


class FailurePrediction(BaseAnalysisModel):
    """Failure probability prediction results."""
    failure_probability: float = Field(..., ge=0, le=1, description="Probability of early failure")
    risk_category: RiskCategory = Field(..., description="Risk classification")
    gradient_margin: float = Field(..., description="Margin to failure threshold")
    contributing_factors: Dict[str, float] = Field(default_factory=dict, description="Factor contributions")


class DynamicRangeAnalysis(BaseAnalysisModel):
    """Dynamic range utilization analysis."""
    range_utilization_percent: Optional[float] = Field(None, ge=0, le=100)
    minimum_margin: Optional[float] = Field(None, description="Minimum margin to limits")
    minimum_margin_position: Optional[float] = Field(None, description="Position of minimum margin")
    margin_bias: Optional[str] = Field(None, description="Tendency towards upper/lower limit")


class TrackData(BaseAnalysisModel):
    """Data for a single track analysis."""
    track_id: str = Field(..., description="Track identifier (e.g., TRK1, TRK2, default)")
    status: AnalysisStatus = Field(..., description="Track analysis status")

    # Core measurements
    travel_length: float = Field(..., gt=0, description="Total travel length")
    position_data: Optional[List[float]] = Field(None, description="Position measurements")
    error_data: Optional[List[float]] = Field(None, description="Error measurements")
    
    # Store untrimmed data separately for plotting comparison
    untrimmed_positions: Optional[List[float]] = Field(None, description="Untrimmed position measurements")
    untrimmed_errors: Optional[List[float]] = Field(None, description="Untrimmed error measurements")

    # Analysis results
    unit_properties: UnitProperties
    sigma_analysis: SigmaAnalysis
    linearity_analysis: LinearityAnalysis
    resistance_analysis: ResistanceAnalysis

    # Advanced analytics
    trim_effectiveness: Optional[TrimEffectiveness] = None
    zone_analysis: Optional[ZoneAnalysis] = None
    failure_prediction: Optional[FailurePrediction] = None
    dynamic_range: Optional[DynamicRangeAnalysis] = None

    # Validation summary
    overall_validation_status: ValidationStatus = Field(default=ValidationStatus.NOT_VALIDATED, description="Overall validation status")
    validation_warnings: List[str] = Field(default_factory=list, description="All validation warnings")
    validation_recommendations: List[str] = Field(default_factory=list, description="All validation recommendations")

    # Visualization
    plot_path: Optional[Path] = Field(None, description="Path to generated plot")

    @field_validator('position_data', 'error_data')
    @classmethod
    def validate_data_length(cls, v: Optional[List[float]], info) -> Optional[List[float]]:
        """Ensure data arrays have consistent length."""
        if v is not None and len(v) < 10:
            raise ValueError(f"Data array too short: {len(v)} points")
        return v

    @computed_field
    @property
    def validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        summary = {
            "overall_status": self.overall_validation_status.value,
            "total_validations": 0,
            "passed_validations": 0,
            "warnings_count": len(self.validation_warnings),
            "recommendations_count": len(self.validation_recommendations),
            "grades": {}
        }
        
        # Count validations
        validations = [
            self.sigma_analysis.validation_result,
            self.linearity_analysis.validation_result,
            self.resistance_analysis.validation_result
        ]
        
        if self.trim_effectiveness and self.trim_effectiveness.validation_result:
            validations.append(self.trim_effectiveness.validation_result)
        
        valid_validations = [v for v in validations if v is not None]
        summary["total_validations"] = len(valid_validations)
        summary["passed_validations"] = sum(1 for v in valid_validations if v.is_valid)
        
        # Add grades
        summary["grades"]["sigma"] = self.sigma_analysis.industry_compliance
        summary["grades"]["linearity"] = self.linearity_analysis.industry_grade
        summary["grades"]["resistance"] = self.resistance_analysis.resistance_stability_grade
        
        if self.trim_effectiveness:
            summary["grades"]["trim_effectiveness"] = self.trim_effectiveness.trim_quality_grade
        
        return summary


class AnalysisResult(BaseAnalysisModel):
    """Complete analysis result for a file."""
    # File metadata
    metadata: FileMetadata

    # Overall results
    overall_status: AnalysisStatus = Field(..., description="Overall file status")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")

    # Track data (one or more tracks)
    tracks: Dict[str, TrackData] = Field(..., description="Track analysis results")

    # Validation and errors
    validation_issues: List[str] = Field(default_factory=list, description="Validation warnings/errors")
    processing_errors: List[str] = Field(default_factory=list, description="Processing errors")
    
    # Missing validation attributes that UI expects
    validation_warnings: List[str] = Field(default_factory=list, description="All validation warnings")
    validation_recommendations: List[str] = Field(default_factory=list, description="All validation recommendations")

    # Overall validation status
    overall_validation_status: ValidationStatus = Field(default=ValidationStatus.NOT_VALIDATED, description="Overall validation status")
    validation_summary: Dict[str, Any] = Field(default_factory=dict, description="Comprehensive validation summary")

    # Database reference
    db_id: Optional[int] = Field(None, description="Database record ID")

    @field_validator('tracks')
    @classmethod
    def validate_tracks(cls, v: Dict[str, TrackData]) -> Dict[str, TrackData]:
        """Ensure at least one track exists."""
        if not v:
            raise ValueError("Analysis must contain at least one track")
        return v

    @computed_field
    @property
    def primary_track(self) -> TrackData:
        """Get the primary track (TRK1 or first available)."""
        if "TRK1" in self.tracks:
            return self.tracks["TRK1"]
        elif "default" in self.tracks:
            return self.tracks["default"]
        return next(iter(self.tracks.values()))

    @computed_field
    @property
    def all_tracks_pass(self) -> bool:
        """Check if all tracks pass."""
        return all(
            track.sigma_analysis.sigma_pass and
            track.linearity_analysis.linearity_pass
            for track in self.tracks.values()
        )

    @computed_field
    @property
    def validation_grade(self) -> str:
        """Get overall validation grade."""
        if self.overall_validation_status == ValidationStatus.NOT_VALIDATED:
            return "Not Validated"
        elif self.overall_validation_status == ValidationStatus.FAILED:
            return "F"
        
        # Calculate average grade from all tracks
        grades = []
        for track in self.tracks.values():
            if track.sigma_analysis.validation_result:
                grades.append(track.sigma_analysis.validation_result.validation_grade)
            if track.linearity_analysis.validation_result:
                grades.append(track.linearity_analysis.validation_result.validation_grade)
        
        if not grades:
            return "Incomplete"
        
        # Convert grades to numeric and average
        grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0, "F": 0}
        avg_grade = sum(grade_values.get(g, 0) for g in grades) / len(grades)
        
        if avg_grade >= 3.5:
            return "A"
        elif avg_grade >= 2.5:
            return "B"
        elif avg_grade >= 1.5:
            return "C"
        elif avg_grade >= 0.5:
            return "D"
        else:
            return "F"

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for Excel/database storage."""
        # Start with file-level data
        flat = {
            "filename": self.metadata.filename,
            "file_path": str(self.metadata.file_path),
            "file_date": self.metadata.file_date,
            "model": self.metadata.model,
            "serial": self.metadata.serial,
            "system": self.metadata.system.value,
            "has_multi_tracks": self.metadata.has_multi_tracks,
            "overall_status": self.overall_status.value,
            "processing_time": self.processing_time,
            
            # Validation fields
            "overall_validation_status": self.overall_validation_status.value,
            "validation_grade": self.validation_grade,
            "validation_issues_count": len(self.validation_issues),
        }

        # Add primary track data for backward compatibility
        primary = self.primary_track
        flat.update({
            "sigma_gradient": primary.sigma_analysis.sigma_gradient,
            "sigma_threshold": primary.sigma_analysis.sigma_threshold,
            "sigma_pass": primary.sigma_analysis.sigma_pass,
            "sigma_validation_status": primary.sigma_analysis.validation_status.value,
            "sigma_industry_compliance": primary.sigma_analysis.industry_compliance,
            
            "linearity_pass": primary.linearity_analysis.linearity_pass,
            "linearity_error": primary.linearity_analysis.final_linearity_error_shifted,
            "linearity_validation_status": primary.linearity_analysis.validation_status.value,
            "linearity_industry_grade": primary.linearity_analysis.industry_grade,
            
            "resistance_validation_status": primary.resistance_analysis.validation_status.value,
            "resistance_stability_grade": primary.resistance_analysis.resistance_stability_grade,
            
            "risk_category": primary.failure_prediction.risk_category.value if primary.failure_prediction else None,
        })

        # Add validation summary
        validation_summary = primary.validation_summary
        flat.update({
            "total_validations": validation_summary["total_validations"],
            "passed_validations": validation_summary["passed_validations"],
            "validation_success_rate": (validation_summary["passed_validations"] / validation_summary["total_validations"] * 100) if validation_summary["total_validations"] > 0 else 0,
        })

        return flat


# Batch processing models
class BatchConfig(BaseAnalysisModel):
    """Configuration for batch processing."""
    batch_name: str = Field(..., description="Batch identifier")
    input_directory: Path = Field(..., description="Input directory path")
    output_directory: Path = Field(..., description="Output directory path")
    processing_mode: ProcessingMode = Field(default=ProcessingMode.DETAIL)
    max_workers: int = Field(default=4, ge=1, le=16)
    file_pattern: str = Field(default="*.xlsx", description="File pattern to match")

    # ML configuration
    enable_ml_predictions: bool = Field(default=True)
    enable_threshold_optimization: bool = Field(default=True)
    
    # Validation configuration
    enable_validation: bool = Field(default=True, description="Enable industry standard validation")
    validation_level: str = Field(default="standard", description="Validation strictness level")

    # Database configuration
    save_to_database: bool = Field(default=True)
    database_path: Optional[Path] = None


class BatchResult(BaseAnalysisModel):
    """Results from batch processing."""
    batch_config: BatchConfig
    start_time: datetime
    end_time: datetime
    total_files: int
    successful_files: int
    failed_files: int
    results: List[AnalysisResult]
    
    # Validation summary
    validation_summary: Dict[str, Any] = Field(default_factory=dict, description="Batch validation summary")

    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.successful_files / self.total_files * 100) if self.total_files > 0 else 0.0

    @computed_field
    @property
    def processing_duration(self) -> float:
        """Calculate total processing time in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @computed_field
    @property
    def validation_success_rate(self) -> float:
        """Calculate validation success rate."""
        if not self.results:
            return 0.0
        
        validated_results = [r for r in self.results if r.overall_validation_status != ValidationStatus.NOT_VALIDATED]
        if not validated_results:
            return 0.0
        
        passed_validations = sum(1 for r in validated_results if r.overall_validation_status == ValidationStatus.VALIDATED)
        return (passed_validations / len(validated_results)) * 100