"""
Pydantic data models for Laser Trim Analyzer v3.

Simplified from v2 - focused on essential data structures.
Target: ~400 lines (v2 was 600+)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import numpy as np


# ============================================================================
# Enums
# ============================================================================

class SystemType(str, Enum):
    """Laser trim system types.

    C = the third trim system (LTS3, added 2026). Its files are FORMAT-
    identical to System B — detection is by the 'LTS3' folder in the file
    path, not by sheet structure. Parse logic treats C as B-format.
    """
    A = "A"
    B = "B"
    C = "C"
    UNKNOWN = "Unknown"


class AnalysisStatus(str, Enum):
    """Analysis status."""
    PASS = "Pass"
    FAIL = "Fail"
    WARNING = "Warning"
    ERROR = "Error"
    # Test-sweep file with no laser-trim runs. Sigma/linearity metrics are
    # not defined; only the untrimmed sweep is recorded.
    UNTRIMMED = "Untrimmed"


class RiskCategory(str, Enum):
    """Risk categories for failure prediction."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"


# ============================================================================
# Base Model
# ============================================================================

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

    @model_validator(mode="before")
    @classmethod
    def _nan_means_missing(cls, data):
        """NaN in a measurement column means 'not measured' — treat as None.

        Work incident 2026-07-10: pydantic ≥2.12 started rejecting NaN on
        ge=0-constrained floats, so every file with an unmeasured field (2,106
        that day) ERRORED at TrackData construction. Older pydantic let NaN
        through, which is how home kept working while the fresh work venv
        broke. Coercing NaN→None is version-proof AND more honest: NaN was
        never a real measurement, and downstream code already handles None.
        """
        if isinstance(data, dict):
            return {k: (None if isinstance(v, float) and v != v else v)
                    for k, v in data.items()}
        return data


# ============================================================================
# File Metadata
# ============================================================================

class FileMetadata(BaseAnalysisModel):
    """File information and metadata."""
    filename: str = Field(..., description="Name of the file")
    file_path: Path = Field(..., description="Full path to the file")
    file_date: Optional[datetime] = Field(None, description="File modification date")
    test_date: Optional[datetime] = Field(None, description="Trim date from Excel")
    model: str = Field(..., description="Model number")
    serial: str = Field(..., description="Serial number")
    system: SystemType = Field(..., description="System type (A or B)")
    has_multi_tracks: bool = Field(default=False, description="Multi-track file")
    track_identifier: Optional[str] = Field(None, description="Track ID (e.g., TA, TB)")


# ============================================================================
# Track Data
# ============================================================================

class TrackData(BaseAnalysisModel):
    """
    Data for a single track analysis.

    Simplified from v2 - contains only essential fields.
    Advanced analytics are computed on-demand, not stored.
    """
    track_id: str = Field(..., description="Track identifier")
    status: AnalysisStatus = Field(..., description="Track status")

    # Core measurements
    travel_length: float = Field(..., ge=0, description="Travel length")
    linearity_spec: float = Field(..., ge=0, description="Linearity spec")

    # Sigma analysis (None for UNTRIMMED tracks — no trim run, sigma undefined)
    sigma_gradient: Optional[float] = Field(None, ge=0, description="Sigma gradient value")
    untrimmed_sigma_gradient: Optional[float] = Field(
        None, ge=0,
        description="Sigma gradient calculated on untrimmed (pre-trim) arrays. "
                    "Upstream element-quality signal; independent of post-trim "
                    "process. NULL when untrimmed arrays absent or all-NaN.",
    )
    sigma_threshold: Optional[float] = Field(None, gt=0, description="Sigma threshold")
    sigma_pass: Optional[bool] = Field(None, description="Sigma test passed")

    # Linearity analysis (None for UNTRIMMED tracks — no trim result to judge)
    # optimal_offset is Optional so UNTRIMMED rows can record "no measurement"
    # rather than a placeholder 0.0 that would skew any future histogram.
    optimal_offset: Optional[float] = Field(default=0.0, description="Optimal offset")
    linearity_error: Optional[float] = Field(None, ge=0, description="Linearity error")
    linearity_pass: Optional[bool] = Field(None, description="Linearity test passed")
    linearity_fail_points: int = Field(default=0, ge=0, description="Failing points count")
    # Set when the source file's limit columns do not describe a usable spec
    # band (see LaserTrimParser._validate_limit_columns). When present,
    # linearity_pass is forced to None: the unit is NOT graded, because
    # grading it against a corrupt limit would manufacture a false PASS on a
    # zero-tolerance customer disposition.
    linearity_spec_warning: Optional[str] = Field(
        None, description="Why linearity_spec is untrustworthy (None = spec is usable)"
    )

    # Spec-aware optimization results (Phase 2)
    optimal_slope: float = Field(default=0.0, description="Theory rotation factor k (0.0 = no rotation)")
    station_compensation: Optional[float] = Field(None, description="Compensation value from station file (offset applied by machine)")
    linearity_type: Optional[str] = Field(None, description="Linearity type from model specs (Absolute, Independent, Term Base, Zero-Based)")
    raw_linearity_error: Optional[float] = Field(None, ge=0, description="Max error before any optimization")
    optimized_linearity_error: Optional[float] = Field(None, ge=0, description="Max error after optimal offset+slope adjustment")
    raw_fail_points: Optional[int] = Field(None, ge=0, description="Fail points before optimization")

    # Unit properties (optional)
    unit_length: Optional[float] = Field(None, ge=0, description="Unit length")
    untrimmed_resistance: Optional[float] = Field(None, ge=0, description="Untrimmed resistance")
    trimmed_resistance: Optional[float] = Field(None, ge=0, description="Trimmed resistance")
    measured_electrical_angle: Optional[float] = Field(None, description="Measured electrical angle from trim sheet")

    # Risk assessment (from ML)
    failure_probability: Optional[float] = Field(None, ge=0, le=1, description="Failure probability")
    risk_category: RiskCategory = Field(default=RiskCategory.UNKNOWN, description="Risk category")

    # Anomaly detection (trim failures with linear slope pattern)
    is_anomaly: bool = Field(default=False, description="Flagged as anomalous (likely trim failure)")
    anomaly_reason: Optional[str] = Field(None, description="Reason for anomaly flag")

    # Trim difficulty: how many laser-trim passes the equipment ran on this
    # track. Counted from the file's "Trim N" / "TRK<n> M" sheet structure.
    # 1 = single pass to spec, 2+ = retrim was needed. None for FT files.
    trim_pass_count: Optional[int] = Field(None, ge=0, description="Number of trim passes the laser ran for this track")

    # Theory/test voltage data (from parser, used for slope optimization bounds)
    theory_volts: Optional[List[float]] = Field(None, description="Theoretical output values at each position")
    test_volts: Optional[float] = Field(None, description="Reference voltage for slope optimization bounds")

    # Raw data for plotting (optional - can be large)
    position_data: Optional[List[float]] = Field(None, description="Position values")
    error_data: Optional[List[float]] = Field(None, description="Error values")
    # Note: Limits can have None values at positions with no specification (unlimited)
    upper_limits: Optional[List[Optional[float]]] = Field(None, description="Upper spec limits (None = no limit)")
    lower_limits: Optional[List[Optional[float]]] = Field(None, description="Lower spec limits (None = no limit)")

    # Untrimmed data for comparison (optional)
    untrimmed_positions: Optional[List[float]] = Field(None, description="Untrimmed positions")
    untrimmed_errors: Optional[List[float]] = Field(None, description="Untrimmed errors")

    # Max deviation metrics
    max_deviation: Optional[float] = Field(None, ge=0, description="Max absolute error after offset (same as linearity_error)")
    max_deviation_position: Optional[float] = Field(None, description="Position where max deviation occurs")
    deviation_uniformity: Optional[float] = Field(None, ge=0, description="Coefficient of variation of absolute errors (std/mean)")

    # Failure margin metrics — how far from spec limits
    max_violation: Optional[float] = Field(None, ge=0, description="Max amount any point exceeded spec (absolute)")
    avg_violation: Optional[float] = Field(None, ge=0, description="Average violation across all fail points")
    margin_to_spec: Optional[float] = Field(None, description="For passing tracks: closest margin to spec limit (% of spec width)")

    # Trim effectiveness metrics (calculated when untrimmed data available)
    resistance_change: Optional[float] = Field(None, description="Resistance change (ohms)")
    trim_improvement_percent: Optional[float] = Field(None, description="RMS error improvement from trim (%)")
    untrimmed_rms_error: Optional[float] = Field(None, description="RMS error before trim")
    untrimmed_error_max: Optional[float] = Field(
        None, ge=0,
        description="Max |error| on the untrimmed (pre-trim) sweep. Zero-tolerance "
                    "linearity is governed by the worst point; this is the strongest "
                    "single upstream-drift signal. Exclude-aware, NaN-safe.",
    )
    trimmed_rms_error: Optional[float] = Field(None, description="RMS error after trim")
    max_error_reduction_percent: Optional[float] = Field(None, description="Max error reduction from trim (%)")

    # Composite trim-risk score (0..1). Written during live processing for
    # deployed CompositeRiskModel instances; NULL if no deployed model exists.
    # This score is the primary input for the group-level drift early-warning.
    composite_trim_risk_score: Optional[float] = Field(
        None, ge=0, le=1,
        description="Per-unit composite trim-risk score from the deployed logistic "
                    "regression. Group trend of this field is the drift early-warning "
                    "signal. NULL means no deployed model for this product model.",
    )

    # Plot reference
    plot_path: Optional[Path] = Field(None, description="Path to plot image")

    @property
    def gradient_margin(self) -> Optional[float]:
        """Margin between gradient and threshold. None for untrimmed tracks."""
        if self.sigma_threshold is None or self.sigma_gradient is None:
            return None
        return self.sigma_threshold - self.sigma_gradient

    @property
    def sigma_ratio(self) -> Optional[float]:
        """Ratio of gradient to threshold (lower is better). None for untrimmed tracks."""
        if self.sigma_threshold is None or self.sigma_gradient is None:
            return None
        return self.sigma_gradient / self.sigma_threshold if self.sigma_threshold > 0 else float('inf')

    @property
    def resistance_change_percent(self) -> Optional[float]:
        """Percentage change in resistance."""
        if self.untrimmed_resistance and self.trimmed_resistance and self.untrimmed_resistance > 0:
            return ((self.trimmed_resistance - self.untrimmed_resistance) / self.untrimmed_resistance) * 100
        return None


# ============================================================================
# Analysis Result
# ============================================================================

class AnalysisResult(BaseAnalysisModel):
    """
    Complete analysis result for a file.

    Contains metadata, track results, and processing info.
    """
    # File metadata
    metadata: FileMetadata

    # Overall results
    overall_status: AnalysisStatus = Field(..., description="Overall file status")
    processing_time: float = Field(..., ge=0, description="Processing time (seconds)")

    # Track data (one or more tracks)
    tracks: List[TrackData] = Field(..., description="Track results")

    # Errors/warnings
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")

    # Database reference
    db_id: Optional[int] = Field(None, description="Database record ID")

    # File type marker (for distinguishing trim vs final test)
    file_type: str = Field(default="trim", description="File type: 'trim', 'final_test', or 'smoothness'")
    final_test_id: Optional[int] = Field(None, description="Database ID for final test records")
    smoothness_id: Optional[int] = Field(None, description="Linked smoothness result ID")

    # Data quality flags — validation issues found during ingest
    data_quality: str = Field(default="good", description="Data quality: 'good' or 'suspect'")
    data_quality_issues: List[str] = Field(default_factory=list, description="List of validation issues found")

    @field_validator('tracks')
    @classmethod
    def validate_tracks(cls, v: List[TrackData], info) -> List[TrackData]:
        """Ensure at least one track exists (except for ERROR status)."""
        # Allow empty tracks only for ERROR status
        # info.data contains the other field values being validated
        if not v:
            status = info.data.get('overall_status')
            if status != AnalysisStatus.ERROR:
                raise ValueError("Analysis must contain at least one track")
        return v

    @property
    def primary_track(self) -> Optional[TrackData]:
        """Get the primary track (first one), or None if no tracks."""
        return self.tracks[0] if self.tracks else None

    @property
    def all_tracks_pass(self) -> bool:
        """Check if all tracks pass both sigma and linearity. Untrimmed tracks never count as passing."""
        return all(
            t.sigma_pass is True and t.linearity_pass is True
            for t in self.tracks
        )

    @property
    def any_high_risk(self) -> bool:
        """Check if any track is high risk."""
        return any(t.risk_category == RiskCategory.HIGH for t in self.tracks)

    @property
    def track_count(self) -> int:
        """Number of tracks."""
        return len(self.tracks)

    def get_track(self, track_id: str) -> Optional[TrackData]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for display/export."""
        primary = self.primary_track
        result = {
            "filename": self.metadata.filename,
            "model": self.metadata.model,
            "serial": self.metadata.serial,
            "system": self.metadata.system.value,
            "status": self.overall_status.value,
            "processing_time": self.processing_time,
            "track_count": self.track_count,
            "test_date": self.metadata.test_date.isoformat() if self.metadata.test_date else None,
        }

        # Add track data if available
        if primary:
            result.update({
                "sigma_gradient": primary.sigma_gradient,
                "sigma_threshold": primary.sigma_threshold,
                "sigma_pass": primary.sigma_pass,
                "linearity_error": primary.linearity_error,
                "linearity_pass": primary.linearity_pass,
                "risk_category": primary.risk_category.value,
                "failure_probability": primary.failure_probability,
            })
        else:
            # Default values for error results
            result.update({
                "sigma_gradient": None,
                "sigma_threshold": None,
                "sigma_pass": None,
                "linearity_error": None,
                "linearity_pass": None,
                "risk_category": RiskCategory.UNKNOWN.value,
                "failure_probability": None,
            })

        return result


# ============================================================================
# Processing Status (for progress tracking)
# ============================================================================

class ProcessingStatus(BaseAnalysisModel):
    """Status of file processing."""
    filename: str
    status: str  # "pending", "processing", "completed", "failed", "skipped"
    message: Optional[str] = None
    progress_percent: float = 0.0

    # For completed files
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None


# ============================================================================
# Batch Summary
# ============================================================================

class BatchSummary(BaseAnalysisModel):
    """Summary of batch processing results."""
    total_files: int = 0
    processed: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0  # Pass linearity but fail sigma (or vice versa)
    skipped: int = 0
    errors: int = 0
    # Test-sweep files with no laser-trim run. A valid, expected state -- NOT an
    # error -- and excluded from the yield denominator (see gradeable_count).
    untrimmed: int = 0
    anomalies: int = 0  # Trim failures with linear slope pattern

    def record_status(self, status: "AnalysisStatus") -> None:
        """Bucket one result's overall_status into the right counter.

        Single source of truth so the processor and GUI never disagree.
        UNTRIMMED is its own bucket -- never an error, never in the yield
        denominator.
        """
        if status == AnalysisStatus.PASS:
            self.passed += 1
        elif status == AnalysisStatus.WARNING:
            self.warnings += 1
        elif status == AnalysisStatus.FAIL:
            self.failed += 1
        elif status == AnalysisStatus.UNTRIMMED:
            self.untrimmed += 1
        else:  # ERROR or anything unexpected
            self.errors += 1

    @property
    def gradeable_count(self) -> int:
        """Processed files that actually have a trim result.

        Excludes UNTRIMMED test-sweeps -- 'no trim result' is not 'a fail',
        so it must not sit in the yield denominator.
        """
        return max(0, self.processed - self.untrimmed)

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_processing_time: float = 0.0

    # Stats
    avg_sigma_gradient: Optional[float] = None
    pass_rate: Optional[float] = None
    high_risk_count: int = 0

    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return self.total_processing_time

    @property
    def files_per_second(self) -> float:
        """Processing speed."""
        duration = self.duration_seconds
        return self.processed / duration if duration > 0 else 0.0
