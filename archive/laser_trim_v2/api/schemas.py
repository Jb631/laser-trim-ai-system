"""
API schemas for laser trim analyzer.

Pydantic models for API requests and responses.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator
import numpy as np


class AnalysisMode(str, Enum):
    """Analysis processing modes."""
    QUICK = "quick"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class AIProvider(str, Enum):
    """Supported AI providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    LOCAL = "local"


class PredictionType(str, Enum):
    """Types of ML predictions."""
    FAILURE_RISK = "failure_risk"
    THRESHOLD_OPTIMIZATION = "threshold_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    QUALITY_SCORE = "quality_score"


# Request schemas
class AnalysisRequest(BaseModel):
    """Request schema for analysis API."""

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist(),
        }
    )

    # File data
    filename: str = Field(..., description="Name of the file to analyze")
    file_content: Optional[str] = Field(None, description="Base64 encoded file content")
    file_url: Optional[str] = Field(None, description="URL to download file from")

    # Analysis options
    mode: AnalysisMode = Field(
        default=AnalysisMode.STANDARD,
        description="Analysis mode"
    )

    # Processing options
    generate_plots: bool = Field(default=True, description="Generate analysis plots")
    save_to_database: bool = Field(default=True, description="Save results to database")

    # ML options
    enable_ml_predictions: bool = Field(default=True, description="Enable ML predictions")
    prediction_types: List[PredictionType] = Field(
        default=[PredictionType.FAILURE_RISK, PredictionType.QUALITY_SCORE],
        description="Types of predictions to generate"
    )

    # AI options
    enable_ai_insights: bool = Field(default=False, description="Generate AI insights")
    ai_provider: Optional[AIProvider] = Field(None, description="AI provider to use")

    # Metadata
    operator: Optional[str] = Field(None, description="Operator name")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    @field_validator('file_content', 'file_url')
    @classmethod
    def validate_file_source(cls, v, info):
        """Ensure at least one file source is provided."""
        if info.field_name == 'file_url' and not v and not info.data.get('file_content'):
            raise ValueError("Either file_content or file_url must be provided")
        return v


class BatchAnalysisRequest(BaseModel):
    """Request schema for batch analysis."""

    files: List[AnalysisRequest] = Field(..., description="List of files to analyze")

    # Batch options
    parallel_processing: bool = Field(default=True, description="Process files in parallel")
    max_workers: int = Field(default=4, ge=1, le=16, description="Maximum parallel workers")
    stop_on_error: bool = Field(default=False, description="Stop batch on first error")

    # Batch metadata
    batch_name: str = Field(..., description="Batch name for identification")
    batch_type: str = Field(default="production", description="Type of batch")

    @field_validator('files')
    @classmethod
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if len(v) == 0:
            raise ValueError("Batch must contain at least one file")
        if len(v) > 1000:
            raise ValueError("Batch size exceeds maximum of 1000 files")
        return v


class AIInsightRequest(BaseModel):
    """Request schema for AI insights."""

    # Data for analysis
    analysis_results: List[Dict[str, Any]] = Field(
        ...,
        description="Analysis results to generate insights from"
    )

    # AI options
    provider: AIProvider = Field(..., description="AI provider to use")
    model: Optional[str] = Field(None, description="Specific model to use")

    # Insight types
    insight_types: List[str] = Field(
        default=["failure_patterns", "process_improvements", "quality_trends"],
        description="Types of insights to generate"
    )

    # Context
    historical_context: bool = Field(
        default=True,
        description="Include historical context in analysis"
    )
    compare_to_baseline: bool = Field(
        default=False,
        description="Compare to baseline metrics"
    )

    # Output options
    format: str = Field(default="structured", description="Output format")
    language: str = Field(default="en", description="Output language")
    max_recommendations: int = Field(default=5, ge=1, le=20)


# Response schemas
class PredictionResult(BaseModel):
    """Schema for ML prediction results."""

    prediction_type: PredictionType
    timestamp: datetime = Field(default_factory=datetime.now)

    # Prediction values
    value: float = Field(..., description="Primary prediction value")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")

    # Additional metrics
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional prediction metrics"
    )

    # Explanations
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    feature_importance: Optional[Dict[str, float]] = Field(
        None,
        description="Feature importance scores"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Action recommendations"
    )

    # Model info
    model_version: str = Field(..., description="Model version used")
    model_type: str = Field(..., description="Type of model")


class TrackAnalysisResult(BaseModel):
    """Schema for track analysis results."""

    track_id: str
    status: str

    # Core metrics
    sigma_gradient: float
    sigma_threshold: float
    sigma_pass: bool
    linearity_pass: bool

    # Risk assessment
    failure_probability: float
    risk_category: str

    # Additional data
    data_points: int
    processing_time: float

    # ML predictions if available
    predictions: Optional[List[PredictionResult]] = None

    # Validation
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Response schema for analysis API."""

    # Request tracking
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.now)

    # File info
    filename: str
    model: str
    serial: str
    system_type: str

    # Overall results
    overall_status: str
    processing_time: float

    # Track results
    tracks: Dict[str, TrackAnalysisResult]

    # ML predictions
    predictions: Optional[List[PredictionResult]] = None

    # AI insights
    ai_insights: Optional[Dict[str, Any]] = None

    # File references
    plot_urls: Dict[str, str] = Field(
        default_factory=dict,
        description="URLs to generated plots"
    )
    report_url: Optional[str] = Field(None, description="URL to detailed report")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Status
    success: bool = Field(True, description="Whether analysis succeeded")
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class BatchAnalysisResponse(BaseModel):
    """Response schema for batch analysis."""

    batch_id: str
    batch_name: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Summary statistics
    total_files: int
    successful_files: int
    failed_files: int

    # Aggregate metrics
    overall_pass_rate: float
    average_sigma: float
    high_risk_count: int

    # Individual results
    results: List[AnalysisResponse]

    # Batch insights
    batch_insights: Optional[Dict[str, Any]] = None

    # Reports
    summary_report_url: Optional[str] = None
    detailed_report_url: Optional[str] = None

    # Processing info
    processing_time: float
    parallel_processed: bool


class AIInsightResponse(BaseModel):
    """Response schema for AI insights."""

    request_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Provider info
    provider: AIProvider
    model_used: str

    # Insights
    insights: Dict[str, Any] = Field(..., description="Generated insights")

    # Key findings
    key_findings: List[str] = Field(..., description="Summary of key findings")

    # Recommendations
    recommendations: List[Dict[str, Any]] = Field(
        ...,
        description="Prioritized recommendations"
    )

    # Metrics
    confidence_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for insights"
    )

    # Usage info
    tokens_used: Optional[int] = None
    processing_time: float
    cost_estimate: Optional[float] = None


class HealthCheckResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(default="healthy")
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Service status
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of dependent services"
    )

    # System info
    system_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="System information"
    )


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error_id: str = Field(..., description="Unique error identifier")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Error details
    error_type: str
    error_message: str

    # Context
    request_id: Optional[str] = None
    file_name: Optional[str] = None

    # Debug info (only in development)
    stack_trace: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# WebSocket schemas for real-time updates
class ProgressUpdate(BaseModel):
    """Schema for progress updates."""

    request_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Progress info
    current_file: Optional[str] = None
    current_step: str
    progress_percent: float = Field(..., ge=0, le=100)

    # Time estimates
    elapsed_time: float
    estimated_remaining: Optional[float] = None

    # Status
    status: str
    message: Optional[str] = None


class RealtimeAlert(BaseModel):
    """Schema for real-time alerts."""

    alert_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Alert info
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity")

    # Context
    file_name: Optional[str] = None
    model: Optional[str] = None

    # Alert details
    title: str
    message: str
    details: Optional[Dict[str, Any]] = None

    # Actions
    recommended_actions: List[str] = Field(default_factory=list)
    auto_resolved: bool = Field(default=False)


# Validation helpers
def validate_analysis_response(response: AnalysisResponse) -> bool:
    """Validate analysis response completeness."""
    if not response.tracks:
        return False

    # Check each track has required data
    for track_id, track in response.tracks.items():
        if track.sigma_gradient < 0 or track.sigma_threshold <= 0:
            return False

    return True