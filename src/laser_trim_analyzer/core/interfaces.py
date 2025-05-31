# laser_trim_analyzer/core/interfaces.py
"""
Core interfaces and data models for the laser trim analyzer.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime
from enum import Enum
import numpy as np


class SystemType(Enum):
    """Potentiometer system types."""
    SYSTEM_A = "A"
    SYSTEM_B = "B"


class RiskCategory(Enum):
    """Risk categories for failure probability."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNKNOWN = "Unknown"


class Status(Enum):
    """Processing status for files and tracks."""
    PASS = "Pass"
    FAIL = "Fail"
    WARNING = "Warning"
    ERROR = "Error"
    UNKNOWN = "Unknown"


@dataclass
class Position:
    """Position data point."""
    value: float
    index: int


@dataclass
class Limits:
    """Upper and lower tolerance limits."""
    upper: Optional[float] = None
    lower: Optional[float] = None


@dataclass
class UnitProperties:
    """Physical properties of the potentiometer unit."""
    unit_length: Optional[float] = None
    untrimmed_resistance: Optional[float] = None
    trimmed_resistance: Optional[float] = None
    resistance_change: Optional[float] = None
    resistance_change_percent: Optional[float] = None


@dataclass
class TrimData:
    """Data from a trim operation."""
    positions: List[float]
    errors: List[float]
    upper_limits: List[Optional[float]]
    lower_limits: List[Optional[float]]
    sheet_name: str
    full_travel_length: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        """Check if data contains valid measurements."""
        return bool(self.positions and self.errors and
                    len(self.positions) == len(self.errors))


@dataclass
class SigmaMetrics:
    """Sigma gradient analysis metrics."""
    gradient: float
    threshold: float
    passed: bool
    margin: float
    scaling_factor: float = 24.0


@dataclass
class LinearityMetrics:
    """Linearity analysis metrics."""
    spec: float
    optimal_offset: float
    final_error_raw: float
    final_error_shifted: float
    passed: bool
    fail_points: int
    max_deviation: Optional[float] = None
    max_deviation_position: Optional[float] = None


@dataclass
class TrimEffectiveness:
    """Metrics for trim effectiveness analysis."""
    improvement_percent: float
    untrimmed_rms_error: float
    trimmed_rms_error: float
    max_error_reduction_percent: float


@dataclass
class ZoneAnalysis:
    """Zone-based analysis results."""
    worst_zone: int
    worst_zone_position: tuple[float, float]
    zone_results: List[Dict[str, Any]]


@dataclass
class TrackResult:
    """Complete analysis results for a single track."""
    track_id: str
    status: Status

    # Core data
    untrimmed_data: TrimData
    final_data: Optional[TrimData]
    unit_properties: UnitProperties

    # Metrics
    sigma_metrics: SigmaMetrics
    linearity_metrics: Optional[LinearityMetrics]
    trim_effectiveness: Optional[TrimEffectiveness]
    zone_analysis: Optional[ZoneAnalysis]

    # Risk assessment
    failure_probability: float
    risk_category: RiskCategory

    # Additional metadata
    plot_path: Optional[str] = None
    processing_time: Optional[float] = None
    validation_issues: List[str] = field(default_factory=list)


@dataclass
class FileResult:
    """Complete analysis results for a file."""
    filename: str
    filepath: str
    model: str
    serial: str
    system: SystemType
    overall_status: Status

    # Track results
    tracks: Dict[str, TrackResult]
    is_multi_track: bool

    # File metadata
    file_date: datetime
    processing_time: float
    output_directory: str

    # Validation
    validation_issues: List[str] = field(default_factory=list)

    @property
    def primary_track(self) -> Optional[TrackResult]:
        """Get primary track for backwards compatibility."""
        if "TRK1" in self.tracks:
            return self.tracks["TRK1"]
        elif "default" in self.tracks:
            return self.tracks["default"]
        return next(iter(self.tracks.values())) if self.tracks else None


# Protocol definitions for clean interfaces

class FileReader(Protocol):
    """Protocol for reading Excel files."""

    async def read_file(self, filepath: str) -> Dict[str, Any]:
        """Read Excel file and return sheet information."""
        ...

    def get_sheet_names(self, filepath: str) -> List[str]:
        """Get list of sheet names in file."""
        ...


class DataExtractor(Protocol):
    """Protocol for extracting data from sheets."""

    async def extract_trim_data(
            self,
            filepath: str,
            sheet_name: str,
            system: SystemType
    ) -> TrimData:
        """Extract trim data from a sheet."""
        ...

    async def extract_unit_properties(
            self,
            filepath: str,
            sheet_info: Dict[str, Any],
            system: SystemType
    ) -> UnitProperties:
        """Extract unit properties from sheets."""
        ...


class MetricsCalculator(Protocol):
    """Protocol for calculating analysis metrics."""

    def calculate_sigma_metrics(
            self,
            data: TrimData,
            unit_properties: UnitProperties,
            model: str
    ) -> SigmaMetrics:
        """Calculate sigma gradient metrics."""
        ...

    def calculate_linearity_metrics(
            self,
            data: TrimData,
            spec: float
    ) -> LinearityMetrics:
        """Calculate linearity metrics."""
        ...

    def calculate_trim_effectiveness(
            self,
            untrimmed: TrimData,
            trimmed: TrimData
    ) -> TrimEffectiveness:
        """Calculate trim effectiveness metrics."""
        ...


class ResultsFormatter(Protocol):
    """Protocol for formatting analysis results."""

    def format_for_excel(self, results: List[FileResult]) -> Any:
        """Format results for Excel export."""
        ...

    def format_for_html(self, results: List[FileResult]) -> str:
        """Format results as HTML report."""
        ...

    def format_for_database(self, results: List[FileResult]) -> List[Dict[str, Any]]:
        """Format results for database storage."""
        ...