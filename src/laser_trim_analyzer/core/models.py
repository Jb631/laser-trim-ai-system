"""
Pydantic data models for Laser Trim Analyzer v3.

Simplified from v2 - focused on essential data structures.
Target: ~400 lines (v2 was 600+)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator
import numpy as np


# ============================================================================
# Enums
# ============================================================================

class SystemType(str, Enum):
    """Laser trim system types."""
    A = "A"
    B = "B"
    UNKNOWN = "Unknown"


class AnalysisStatus(str, Enum):
    """Analysis status."""
    PASS = "Pass"
    FAIL = "Fail"
    WARNING = "Warning"
    ERROR = "Error"


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


# ============================================================================
# File Metadata
# ============================================================================

class FileMetadata(BaseAnalysisModel):
    """File information and metadata."""
    filename: str = Field(..., description="Name of the file")
    file_path: Path = Field(..., description="Full path to the file")
    file_date: datetime = Field(..., description="File modification date")
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

    # Sigma analysis (required)
    sigma_gradient: float = Field(..., ge=0, description="Sigma gradient value")
    sigma_threshold: float = Field(..., gt=0, description="Sigma threshold")
    sigma_pass: bool = Field(..., description="Sigma test passed")

    # Linearity analysis (required)
    optimal_offset: float = Field(..., description="Optimal offset")
    linearity_error: float = Field(..., ge=0, description="Linearity error")
    linearity_pass: bool = Field(..., description="Linearity test passed")
    linearity_fail_points: int = Field(default=0, ge=0, description="Failing points count")

    # Unit properties (optional)
    unit_length: Optional[float] = Field(None, ge=0, description="Unit length")
    untrimmed_resistance: Optional[float] = Field(None, ge=0, description="Untrimmed resistance")
    trimmed_resistance: Optional[float] = Field(None, ge=0, description="Trimmed resistance")

    # Risk assessment (from ML)
    failure_probability: Optional[float] = Field(None, ge=0, le=1, description="Failure probability")
    risk_category: RiskCategory = Field(default=RiskCategory.UNKNOWN, description="Risk category")

    # Raw data for plotting (optional - can be large)
    position_data: Optional[List[float]] = Field(None, description="Position values")
    error_data: Optional[List[float]] = Field(None, description="Error values")
    # Note: Limits can have None values at positions with no specification (unlimited)
    upper_limits: Optional[List[Optional[float]]] = Field(None, description="Upper spec limits (None = no limit)")
    lower_limits: Optional[List[Optional[float]]] = Field(None, description="Lower spec limits (None = no limit)")

    # Untrimmed data for comparison (optional)
    untrimmed_positions: Optional[List[float]] = Field(None, description="Untrimmed positions")
    untrimmed_errors: Optional[List[float]] = Field(None, description="Untrimmed errors")

    # Plot reference
    plot_path: Optional[Path] = Field(None, description="Path to plot image")

    @property
    def gradient_margin(self) -> float:
        """Margin between gradient and threshold."""
        return self.sigma_threshold - self.sigma_gradient

    @property
    def sigma_ratio(self) -> float:
        """Ratio of gradient to threshold (lower is better)."""
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
        """Check if all tracks pass both sigma and linearity."""
        return all(t.sigma_pass and t.linearity_pass for t in self.tracks)

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
