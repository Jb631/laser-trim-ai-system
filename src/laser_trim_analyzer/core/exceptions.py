"""
Custom exceptions for the Laser Trim Analyzer.

These exceptions provide specific error handling for different
stages of the analysis process.
"""


class LaserTrimAnalyzerError(Exception):
    """Base exception for all laser trim analyzer errors."""
    pass


class ProcessingError(LaserTrimAnalyzerError):
    """Raised when file processing fails."""
    pass


class DataExtractionError(LaserTrimAnalyzerError):
    """Raised when data extraction from Excel files fails."""
    pass


class AnalysisError(LaserTrimAnalyzerError):
    """Raised when analysis calculations fail."""
    pass


class ValidationError(LaserTrimAnalyzerError):
    """Raised when data validation fails."""
    pass


class ConfigurationError(LaserTrimAnalyzerError):
    """Raised when configuration is invalid."""
    pass


class DatabaseError(LaserTrimAnalyzerError):
    """Raised when database operations fail."""
    pass


class MLPredictionError(LaserTrimAnalyzerError):
    """Raised when ML prediction fails."""
    pass


class PlottingError(LaserTrimAnalyzerError):
    """Raised when plot generation fails."""
    pass


class SystemDetectionError(DataExtractionError):
    """Raised when system type cannot be detected."""
    pass


class SheetNotFoundError(DataExtractionError):
    """Raised when required sheet is not found in Excel file."""
    pass


class InvalidDataError(ValidationError):
    """Raised when data is invalid or corrupted."""
    pass


class ThresholdExceededError(AnalysisError):
    """Raised when analysis values exceed critical thresholds."""

    def __init__(self, message: str, metric: str, value: float, threshold: float):
        super().__init__(message)
        self.metric = metric
        self.value = value
        self.threshold = threshold


class CacheError(LaserTrimAnalyzerError):
    """Raised when cache operations fail."""
    pass