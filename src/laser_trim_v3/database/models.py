"""
SQLAlchemy database models for Laser Trim Analyzer v3.

Reused from v2 - the database schema is well-designed and production-ready.
v3 uses the same database schema for backwards compatibility.

Tables:
- AnalysisResult: File-level analysis results
- TrackResult: Track-level measurements
- MLPrediction: ML model outputs
- QAAlert: Quality assurance alerts
- BatchInfo: Production batch metadata
- ProcessedFile: Incremental processing tracking
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum as PyEnum

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, Index, UniqueConstraint,
    Enum, JSON, CheckConstraint, event, TypeDecorator
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, validates
from sqlalchemy.sql import func
import json as json_lib


class SafeJSON(TypeDecorator):
    """A JSON type that safely handles empty strings and None values."""
    
    impl = JSON
    cache_ok = True
    
    def __init__(self, none_as=None):
        """Initialize with default value for None."""
        super().__init__()
        self.none_as = none_as if none_as is not None else []
    
    def process_bind_param(self, value, dialect):
        """Process value before saving to database."""
        if value is None or (isinstance(value, list) and len(value) == 0):
            return None
        # Ensure we're saving valid JSON
        try:
            # Test serialization
            json_lib.dumps(value)
            return value
        except (TypeError, ValueError):
            # Log error and return None to avoid corruption
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Cannot serialize value to JSON: {type(value)}, returning None")
            return None
    
    def process_result_value(self, value, dialect):
        """Process value when loading from database with robust error handling."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Handle various edge cases
        if value is None:
            return self.none_as
            
        if isinstance(value, str):
            # Handle empty strings and common empty JSON representations
            if value in ('', '[]', '{}', 'null', 'NULL', 'None'):
                return self.none_as
                
            # Try to parse JSON
            try:
                # Strip whitespace
                value = value.strip()
                if not value:
                    return self.none_as
                    
                parsed = json_lib.loads(value)
                
                # Ensure it's the expected type
                if self.none_as == [] and not isinstance(parsed, list):
                    logger.warning(f"Expected list but got {type(parsed)}, converting to list")
                    if isinstance(parsed, dict):
                        return [parsed]  # Wrap dict in list
                    else:
                        return self.none_as
                        
                return parsed
                
            except json_lib.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}, value: {value[:100]}...")
                return self.none_as
            except ValueError as e:
                logger.error(f"Value error parsing JSON: {e}, value: {value[:100]}...")
                return self.none_as
            except Exception as e:
                logger.error(f"Unexpected error parsing JSON: {e}, value type: {type(value)}")
                return self.none_as
                
        # If already parsed (list, dict, etc.), return as is
        if isinstance(value, (list, dict)):
            return value
            
        # Unknown type, log and return default
        logger.warning(f"Unexpected type for JSON field: {type(value)}, returning default")
        return self.none_as


# Create base class for all models
Base = declarative_base()


class SystemType(PyEnum):
    """System types for potentiometer testing."""
    A = "A"
    B = "B"


class StatusType(PyEnum):
    """Processing status types."""
    PASS = "Pass"
    FAIL = "Fail"
    WARNING = "Warning"
    ERROR = "Error"
    PROCESSING_FAILED = "Processing Failed"


class RiskCategory(PyEnum):
    """Risk categories for failure prediction."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"


class AlertType(PyEnum):
    """Types of QA alerts."""
    CARBON_SCREEN = "Carbon Screen Check"
    HIGH_RISK = "High Risk Unit"
    DRIFT_DETECTED = "Manufacturing Drift"
    THRESHOLD_EXCEEDED = "Threshold Exceeded"
    MAINTENANCE_REQUIRED = "Maintenance Required"
    SIGMA_FAIL = "Sigma Validation Failed"
    PROCESS_ERROR = "Process Error"


class AnalysisResult(Base):
    """
    File-level analysis results.

    This is the main table that stores information about each analyzed file.
    One file can have multiple tracks (for System A multi-track files).
    
    Production-ready with proper validation and constraints.
    """
    __tablename__ = 'analysis_results'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # File identification - required fields
    filename = Column(String(255), nullable=False)
    file_path = Column(Text)
    file_date = Column(DateTime)
    file_hash = Column(String(64))  # SHA256 hash for deduplication

    # Basic properties - required for production
    model = Column(String(50), nullable=False)
    serial = Column(String(100), nullable=False)
    system = Column(Enum(SystemType), nullable=False)
    has_multi_tracks = Column(Boolean, default=False, nullable=False)

    # Overall results - required
    overall_status = Column(Enum(StatusType), nullable=False)
    processing_time = Column(Float)  # seconds

    # Analysis metadata
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)  # Always store in UTC
    output_dir = Column(Text)
    software_version = Column(String(20))
    operator = Column(String(100))

    # Configuration used
    sigma_scaling_factor = Column(Float)
    filter_cutoff_frequency = Column(Float)

    # Relationships with proper cascade for production
    tracks = relationship(
        "TrackResult", 
        back_populates="analysis", 
        cascade="all, delete-orphan",
        lazy="select"  # Explicit loading strategy
    )
    ml_predictions = relationship(
        "MLPrediction", 
        back_populates="analysis", 
        cascade="all, delete-orphan",
        lazy="select"
    )
    qa_alerts = relationship(
        "QAAlert", 
        back_populates="analysis", 
        cascade="all, delete-orphan",
        lazy="select"
    )

    # Production-ready indexes for performance
    __table_args__ = (
        Index('idx_filename_date', 'filename', 'file_date'),
        Index('idx_model_serial', 'model', 'serial'),
        Index('idx_model_serial_date', 'model', 'serial', 'file_date'),
        Index('idx_timestamp', 'timestamp'),
        Index('idx_status', 'overall_status'),
        Index('idx_system', 'system'),
        # Prevent duplicate processing of same file on same date
        UniqueConstraint('filename', 'file_date', 'model', 'serial', name='uq_file_analysis'),
        # Data validation constraints
        CheckConstraint("LENGTH(TRIM(filename)) > 0", name='check_filename_not_empty'),
        CheckConstraint("LENGTH(TRIM(model)) > 0", name='check_model_not_empty'),
        CheckConstraint("LENGTH(TRIM(serial)) > 0", name='check_serial_not_empty'),
        CheckConstraint("processing_time >= 0", name='check_processing_time_positive'),
    )

    @validates('filename')
    def validate_filename(self, key, filename):
        """Validate filename is not empty."""
        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")
        return filename.strip()

    @validates('model')
    def validate_model(self, key, model):
        """Validate model is not empty."""
        if not model or not model.strip():
            raise ValueError("Model cannot be empty")
        return model.strip()

    @validates('serial')
    def validate_serial(self, key, serial):
        """Validate serial is not empty."""
        if not serial or not serial.strip():
            raise ValueError("Serial cannot be empty")
        return serial.strip()

    @validates('processing_time')
    def validate_processing_time(self, key, processing_time):
        """Validate processing time is non-negative."""
        if processing_time is not None and processing_time < 0:
            raise ValueError("Processing time cannot be negative")
        return processing_time

    @validates('overall_status')
    def validate_overall_status(self, key, overall_status):
        """Validate overall_status is a valid enum value."""
        if overall_status is None:
            # Default to ERROR if not specified
            return StatusType.ERROR
        if isinstance(overall_status, str):
            # Handle empty strings or invalid values
            if not overall_status or overall_status.strip() == '':
                return StatusType.ERROR
            # Try to convert string to enum
            try:
                return StatusType(overall_status)
            except ValueError:
                return StatusType.ERROR
        return overall_status

    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, filename='{self.filename}', model='{self.model}', serial='{self.serial}')>"


class TrackResult(Base):
    """
    Track-level results.

    Each track represents one potentiometer element (TRK1, TRK2 for System A).
    System B files have a single 'default' track.
    
    Production-ready with comprehensive validation.
    """
    __tablename__ = 'track_results'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to analysis - required
    analysis_id = Column(Integer, ForeignKey('analysis_results.id'), nullable=False)

    # Track identification - required
    track_id = Column(String(20), nullable=False)  # 'TRK1', 'TRK2', or 'default'
    status = Column(Enum(StatusType), nullable=False)

    # Core measurements - sigma analysis is required for production
    travel_length = Column(Float)
    linearity_spec = Column(Float)
    sigma_gradient = Column(Float, nullable=False)
    sigma_threshold = Column(Float, nullable=False)
    sigma_pass = Column(Boolean, nullable=False)

    # Unit properties
    unit_length = Column(Float)
    untrimmed_resistance = Column(Float)
    trimmed_resistance = Column(Float)
    resistance_change = Column(Float)
    resistance_change_percent = Column(Float)

    # Linearity analysis
    optimal_offset = Column(Float)
    final_linearity_error_raw = Column(Float)
    final_linearity_error_shifted = Column(Float)
    linearity_pass = Column(Boolean)
    linearity_fail_points = Column(Integer)

    # Advanced analytics
    max_deviation = Column(Float)
    max_deviation_position = Column(Float)
    deviation_uniformity = Column(Float)

    # Trim effectiveness
    trim_improvement_percent = Column(Float)
    untrimmed_rms_error = Column(Float)
    trimmed_rms_error = Column(Float)
    max_error_reduction_percent = Column(Float)

    # Raw data for detailed analysis and comparison
    position_data = Column(SafeJSON, nullable=True)  # Array of position values
    error_data = Column(SafeJSON, nullable=True)     # Array of error values
    
    # Zone analysis
    worst_zone = Column(Integer)
    worst_zone_position = Column(Float)
    zone_details = Column(SafeJSON)  # Detailed zone-by-zone results

    # Risk assessment
    failure_probability = Column(Float)
    risk_category = Column(Enum(RiskCategory))
    gradient_margin = Column(Float)

    # Dynamic range
    range_utilization_percent = Column(Float)
    minimum_margin = Column(Float)
    minimum_margin_position = Column(Float)
    margin_bias = Column(String(20))

    # Plot reference
    plot_path = Column(Text)

    # Relationships
    analysis = relationship("AnalysisResult", back_populates="tracks")

    # Production-ready indexes and constraints
    __table_args__ = (
        Index('idx_track_analysis', 'analysis_id', 'track_id'),
        Index('idx_sigma_gradient', 'sigma_gradient'),
        Index('idx_sigma_pass', 'sigma_pass'),
        Index('idx_linearity_pass', 'linearity_pass'),
        Index('idx_risk_category', 'risk_category'),
        Index('idx_failure_probability', 'failure_probability'),
        Index('idx_status', 'status'),
        # Ensure unique track per analysis
        UniqueConstraint('analysis_id', 'track_id', name='uq_analysis_track'),
        # Data validation constraints for production
        CheckConstraint('sigma_gradient >= 0', name='check_sigma_gradient_positive'),
        CheckConstraint('sigma_threshold > 0', name='check_sigma_threshold_positive'),
        CheckConstraint('failure_probability >= 0 AND failure_probability <= 1', name='check_probability_range'),
        CheckConstraint('travel_length > 0', name='check_travel_length_positive'),
        CheckConstraint('unit_length > 0', name='check_unit_length_positive'),
        CheckConstraint("LENGTH(TRIM(track_id)) > 0", name='check_track_id_not_empty'),
        CheckConstraint('linearity_fail_points >= 0', name='check_linearity_fail_points_positive'),
        CheckConstraint('worst_zone >= 0', name='check_worst_zone_positive'),
        CheckConstraint('range_utilization_percent >= 0 AND range_utilization_percent <= 100', name='check_range_utilization_percent'),
    )

    @validates('track_id')
    def validate_track_id(self, key, track_id):
        """Validate track_id is not empty."""
        if not track_id or not track_id.strip():
            raise ValueError("Track ID cannot be empty")
        return track_id.strip()

    @validates('sigma_gradient')
    def validate_sigma_gradient(self, key, sigma_gradient):
        """Validate sigma_gradient is non-negative."""
        if sigma_gradient is not None and sigma_gradient < 0:
            raise ValueError("Sigma gradient cannot be negative")
        return sigma_gradient

    @validates('sigma_threshold')
    def validate_sigma_threshold(self, key, sigma_threshold):
        """Validate sigma_threshold is positive."""
        if sigma_threshold is not None and sigma_threshold <= 0:
            raise ValueError("Sigma threshold must be positive")
        return sigma_threshold

    @validates('failure_probability')
    def validate_failure_probability(self, key, failure_probability):
        """Validate failure_probability is between 0 and 1."""
        if failure_probability is not None:
            if failure_probability < 0 or failure_probability > 1:
                raise ValueError("Failure probability must be between 0 and 1")
        return failure_probability

    @validates('travel_length')
    def validate_travel_length(self, key, travel_length):
        """Validate travel_length is positive."""
        if travel_length is not None and travel_length <= 0:
            raise ValueError("Travel length must be positive")
        return travel_length

    @validates('unit_length')
    def validate_unit_length(self, key, unit_length):
        """Validate unit_length is positive."""
        if unit_length is not None and unit_length <= 0:
            raise ValueError("Unit length must be positive")
        return unit_length

    @validates('linearity_fail_points')
    def validate_linearity_fail_points(self, key, linearity_fail_points):
        """Validate linearity_fail_points is non-negative."""
        if linearity_fail_points is not None and linearity_fail_points < 0:
            raise ValueError("Linearity fail points cannot be negative")
        return linearity_fail_points

    @validates('range_utilization_percent')
    def validate_range_utilization_percent(self, key, range_utilization_percent):
        """Validate range_utilization_percent is between 0 and 100."""
        if range_utilization_percent is not None:
            if range_utilization_percent < 0 or range_utilization_percent > 100:
                raise ValueError("Range utilization percent must be between 0 and 100")
        return range_utilization_percent

    @validates('status')
    def validate_status(self, key, status):
        """Validate status is a valid enum value."""
        if status is None:
            # Default to ERROR if not specified
            return StatusType.ERROR
        if isinstance(status, str):
            # Handle empty strings or invalid values
            if not status or status.strip() == '':
                return StatusType.ERROR
            # Try to convert string to enum
            try:
                return StatusType(status)
            except ValueError:
                return StatusType.ERROR
        return status
    
    @validates('risk_category')
    def validate_risk_category(self, key, risk_category):
        """Validate risk_category is a valid enum value."""
        if risk_category is None:
            # None is acceptable for risk_category
            return None
        if isinstance(risk_category, str):
            # Handle lowercase or mixed case strings
            risk_category_upper = risk_category.upper()
            if risk_category_upper in ['HIGH', 'MEDIUM', 'LOW', 'UNKNOWN']:
                return RiskCategory[risk_category_upper]
            # Default to UNKNOWN for invalid values
            return RiskCategory.UNKNOWN
        return risk_category

    def __repr__(self):
        return f"<TrackResult(id={self.id}, analysis_id={self.analysis_id}, track_id='{self.track_id}', status='{self.status}')>"


class MLPrediction(Base):
    """
    Machine learning predictions and recommendations.

    Stores ML model outputs for threshold optimization and failure prediction.
    Production-ready with proper validation.
    """
    __tablename__ = 'ml_predictions'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to analysis - required
    analysis_id = Column(Integer, ForeignKey('analysis_results.id'), nullable=False)

    # Prediction metadata
    prediction_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_version = Column(String(20))
    prediction_type = Column(String(50))  # 'threshold_optimization', 'failure_prediction', etc.

    # Threshold optimization
    current_threshold = Column(Float)
    recommended_threshold = Column(Float)
    threshold_change_percent = Column(Float)
    false_positives = Column(Integer)
    false_negatives = Column(Integer)

    # Failure prediction
    predicted_failure_probability = Column(Float)
    predicted_risk_category = Column(Enum(RiskCategory))
    confidence_score = Column(Float)

    # Feature importance
    feature_importance = Column(JSON)  # Dict of feature names and importance scores

    # Drift detection
    drift_detected = Column(Boolean, default=False, nullable=False)
    drift_percentage = Column(Float)
    drift_direction = Column(String(20))  # 'increasing', 'decreasing'

    # Recommendations
    recommendations = Column(JSON)  # List of recommendation strings

    # Relationships
    analysis = relationship("AnalysisResult", back_populates="ml_predictions")

    # Production-ready indexes and constraints
    __table_args__ = (
        Index('idx_ml_prediction_date', 'prediction_date'),
        Index('idx_ml_analysis', 'analysis_id'),
        Index('idx_ml_prediction_type', 'prediction_type'),
        Index('idx_ml_drift_detected', 'drift_detected'),
        Index('idx_ml_predicted_risk', 'predicted_risk_category'),
        # Data validation constraints
        CheckConstraint('predicted_failure_probability >= 0 AND predicted_failure_probability <= 1', name='check_ml_probability_range'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_range'),
        CheckConstraint('false_positives >= 0', name='check_false_positives_positive'),
        CheckConstraint('false_negatives >= 0', name='check_false_negatives_positive'),
        CheckConstraint("drift_direction IN ('increasing', 'decreasing', 'stable') OR drift_direction IS NULL", name='check_drift_direction_valid'),
    )

    @validates('predicted_failure_probability')
    def validate_predicted_failure_probability(self, key, predicted_failure_probability):
        """Validate predicted_failure_probability is between 0 and 1."""
        if predicted_failure_probability is not None:
            if predicted_failure_probability < 0 or predicted_failure_probability > 1:
                raise ValueError("Predicted failure probability must be between 0 and 1")
        return predicted_failure_probability

    @validates('confidence_score')
    def validate_confidence_score(self, key, confidence_score):
        """Validate confidence_score is between 0 and 1."""
        if confidence_score is not None:
            if confidence_score < 0 or confidence_score > 1:
                raise ValueError("Confidence score must be between 0 and 1")
        return confidence_score

    @validates('false_positives')
    def validate_false_positives(self, key, false_positives):
        """Validate false_positives is non-negative."""
        if false_positives is not None and false_positives < 0:
            raise ValueError("False positives cannot be negative")
        return false_positives

    @validates('false_negatives')
    def validate_false_negatives(self, key, false_negatives):
        """Validate false_negatives is non-negative."""
        if false_negatives is not None and false_negatives < 0:
            raise ValueError("False negatives cannot be negative")
        return false_negatives

    @validates('drift_direction')
    def validate_drift_direction(self, key, drift_direction):
        """Validate drift_direction is valid."""
        if drift_direction is not None:
            valid_directions = ['increasing', 'decreasing', 'stable']
            if drift_direction not in valid_directions:
                raise ValueError(f"Drift direction must be one of: {valid_directions}")
        return drift_direction
    
    @validates('predicted_risk_category')
    def validate_predicted_risk_category(self, key, predicted_risk_category):
        """Validate predicted_risk_category is a valid enum value."""
        if predicted_risk_category is None:
            # None is acceptable for predicted_risk_category
            return None
        if isinstance(predicted_risk_category, str):
            # Handle lowercase or mixed case strings
            risk_category_upper = predicted_risk_category.upper()
            if risk_category_upper in ['HIGH', 'MEDIUM', 'LOW', 'UNKNOWN']:
                return RiskCategory[risk_category_upper]
            # Default to UNKNOWN for invalid values
            return RiskCategory.UNKNOWN
        return predicted_risk_category

    def __repr__(self):
        return f"<MLPrediction(id={self.id}, analysis_id={self.analysis_id}, type='{self.prediction_type}', date='{self.prediction_date}')>"


class QAAlert(Base):
    """
    Quality assurance alerts and maintenance notifications.

    Tracks important events that require QA attention.
    Production-ready with proper validation and tracking.
    """
    __tablename__ = 'qa_alerts'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to analysis - required
    analysis_id = Column(Integer, ForeignKey('analysis_results.id'), nullable=False)

    # Alert details - required for production
    alert_type = Column(Enum(AlertType), nullable=False)
    severity = Column(String(20), nullable=False)  # 'Critical', 'High', 'Medium', 'Low'
    created_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Alert specifics
    track_id = Column(String(20))  # Which track triggered the alert
    metric_name = Column(String(50))  # e.g., 'sigma_gradient'
    metric_value = Column(Float)
    threshold_value = Column(Float)

    # Alert message and details - required
    message = Column(Text, nullable=False)
    details = Column(JSON)  # Additional context

    # Resolution tracking for production workflow
    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_by = Column(String(100))
    acknowledged_date = Column(DateTime)
    resolved = Column(Boolean, default=False, nullable=False)
    resolved_by = Column(String(100))
    resolved_date = Column(DateTime)
    resolution_notes = Column(Text)

    # Relationships
    analysis = relationship("AnalysisResult", back_populates="qa_alerts")

    # Production-ready indexes and constraints
    __table_args__ = (
        Index('idx_alert_type_date', 'alert_type', 'created_date'),
        Index('idx_alert_severity', 'severity'),
        Index('idx_alert_resolved', 'resolved'),
        Index('idx_alert_acknowledged', 'acknowledged'),
        Index('idx_alert_analysis', 'analysis_id'),
        Index('idx_alert_unresolved', 'resolved', 'severity', 'created_date'),
        # Data validation constraints
        CheckConstraint("LENGTH(TRIM(message)) > 0", name='check_alert_message_not_empty'),
        CheckConstraint("severity IN ('Critical', 'High', 'Medium', 'Low')", name='check_severity_valid'),
        CheckConstraint("acknowledged_date IS NULL OR acknowledged = TRUE", name='check_acknowledged_consistency'),
        CheckConstraint("resolved_date IS NULL OR resolved = TRUE", name='check_resolved_consistency'),
        CheckConstraint("resolved_date IS NULL OR acknowledged = TRUE", name='check_resolved_requires_acknowledged'),
    )

    @validates('message')
    def validate_message(self, key, message):
        """Validate message is not empty."""
        if not message or not message.strip():
            raise ValueError("Alert message cannot be empty")
        return message.strip()

    @validates('severity')
    def validate_severity(self, key, severity):
        """Validate severity is valid."""
        if severity is not None:
            valid_severities = ['Critical', 'High', 'Medium', 'Low']
            if severity not in valid_severities:
                raise ValueError(f"Severity must be one of: {valid_severities}")
        return severity

    @validates('acknowledged')
    def validate_acknowledged(self, key, acknowledged):
        """Validate acknowledged consistency."""
        if acknowledged and not self.acknowledged_date:
            # This will be set by the application logic
            pass
        return acknowledged

    @validates('resolved')
    def validate_resolved(self, key, resolved):
        """Validate resolved consistency."""
        if resolved and not self.acknowledged:
            raise ValueError("Alert must be acknowledged before it can be resolved")
        return resolved

    def __repr__(self):
        return f"<QAAlert(id={self.id}, analysis_id={self.analysis_id}, type='{self.alert_type}', severity='{self.severity}', resolved={self.resolved})>"


class BatchInfo(Base):
    """
    Production batch information.

    Groups analysis results by production batch for trend analysis.
    Production-ready with proper validation.
    """
    __tablename__ = 'batch_info'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Batch identification - required
    batch_name = Column(String(100), nullable=False, unique=True)
    batch_type = Column(String(50), nullable=False)  # 'production', 'rework', 'test'

    # Batch metadata
    created_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    production_date = Column(DateTime)
    model = Column(String(50))
    operator = Column(String(100))

    # Batch statistics (aggregated from linked analyses)
    total_units = Column(Integer, default=0)
    passed_units = Column(Integer, default=0)
    failed_units = Column(Integer, default=0)
    average_sigma_gradient = Column(Float)
    average_failure_probability = Column(Float)

    # Quality metrics
    sigma_pass_rate = Column(Float)
    linearity_pass_rate = Column(Float)
    high_risk_count = Column(Integer, default=0)

    # Notes and tags
    notes = Column(Text)
    tags = Column(JSON)  # List of string tags

    # Production-ready indexes and constraints
    __table_args__ = (
        Index('idx_batch_name', 'batch_name'),
        Index('idx_batch_date', 'created_date'),
        Index('idx_batch_model', 'model'),
        Index('idx_batch_type', 'batch_type'),
        Index('idx_batch_production_date', 'production_date'),
        # Data validation constraints
        CheckConstraint("LENGTH(TRIM(batch_name)) > 0", name='check_batch_name_not_empty'),
        CheckConstraint("batch_type IN ('production', 'rework', 'test')", name='check_batch_type_valid'),
        CheckConstraint('total_units >= 0', name='check_total_units_positive'),
        CheckConstraint('passed_units >= 0', name='check_passed_units_positive'),
        CheckConstraint('failed_units >= 0', name='check_failed_units_positive'),
        CheckConstraint('passed_units + failed_units <= total_units', name='check_units_consistency'),
        CheckConstraint('high_risk_count >= 0', name='check_high_risk_count_positive'),
        CheckConstraint('sigma_pass_rate >= 0 AND sigma_pass_rate <= 100', name='check_sigma_pass_rate_range'),
        CheckConstraint('linearity_pass_rate >= 0 AND linearity_pass_rate <= 100', name='check_linearity_pass_rate_range'),
        CheckConstraint('average_failure_probability >= 0 AND average_failure_probability <= 1', name='check_avg_failure_prob_range'),
    )

    @validates('batch_name')
    def validate_batch_name(self, key, batch_name):
        """Validate batch_name is not empty."""
        if not batch_name or not batch_name.strip():
            raise ValueError("Batch name cannot be empty")
        return batch_name.strip()

    @validates('batch_type')
    def validate_batch_type(self, key, batch_type):
        """Validate batch_type is valid."""
        if batch_type is not None:
            valid_types = ['production', 'rework', 'test']
            if batch_type not in valid_types:
                raise ValueError(f"Batch type must be one of: {valid_types}")
        return batch_type

    @validates('total_units')
    def validate_total_units(self, key, total_units):
        """Validate total_units is non-negative."""
        if total_units is not None and total_units < 0:
            raise ValueError("Total units cannot be negative")
        return total_units

    @validates('passed_units')
    def validate_passed_units(self, key, passed_units):
        """Validate passed_units is non-negative."""
        if passed_units is not None and passed_units < 0:
            raise ValueError("Passed units cannot be negative")
        return passed_units

    @validates('failed_units')
    def validate_failed_units(self, key, failed_units):
        """Validate failed_units is non-negative."""
        if failed_units is not None and failed_units < 0:
            raise ValueError("Failed units cannot be negative")
        return failed_units

    @validates('sigma_pass_rate')
    def validate_sigma_pass_rate(self, key, sigma_pass_rate):
        """Validate sigma_pass_rate is between 0 and 100."""
        if sigma_pass_rate is not None:
            if sigma_pass_rate < 0 or sigma_pass_rate > 100:
                raise ValueError("Sigma pass rate must be between 0 and 100")
        return sigma_pass_rate

    @validates('linearity_pass_rate')
    def validate_linearity_pass_rate(self, key, linearity_pass_rate):
        """Validate linearity_pass_rate is between 0 and 100."""
        if linearity_pass_rate is not None:
            if linearity_pass_rate < 0 or linearity_pass_rate > 100:
                raise ValueError("Linearity pass rate must be between 0 and 100")
        return linearity_pass_rate

    @validates('average_failure_probability')
    def validate_average_failure_probability(self, key, average_failure_probability):
        """Validate average_failure_probability is between 0 and 1."""
        if average_failure_probability is not None:
            if average_failure_probability < 0 or average_failure_probability > 1:
                raise ValueError("Average failure probability must be between 0 and 1")
        return average_failure_probability

    def __repr__(self):
        return f"<BatchInfo(id={self.id}, name='{self.batch_name}', type='{self.batch_type}', units={self.total_units})>"


# Association table for many-to-many relationship between analyses and batches
class AnalysisBatch(Base):
    """
    Links analysis results to batches.
    
    Production-ready association table with proper constraints.
    """
    __tablename__ = 'analysis_batch'

    # Composite primary key
    analysis_id = Column(Integer, ForeignKey('analysis_results.id'), primary_key=True)
    batch_id = Column(Integer, ForeignKey('batch_info.id'), primary_key=True)
    added_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Production-ready indexes
    __table_args__ = (
        Index('idx_analysis_batch_analysis', 'analysis_id'),
        Index('idx_analysis_batch_batch', 'batch_id'),
        Index('idx_analysis_batch_date', 'added_date'),
    )

    def __repr__(self):
        return f"<AnalysisBatch(analysis_id={self.analysis_id}, batch_id={self.batch_id})>"


# Event listeners for production data integrity
@event.listens_for(QAAlert, 'before_update')
def update_alert_timestamps(mapper, connection, target):
    """Update timestamps when alert status changes."""
    if target.acknowledged and not target.acknowledged_date:
        target.acknowledged_date = datetime.utcnow()
    if target.resolved and not target.resolved_date:
        target.resolved_date = datetime.utcnow()


@event.listens_for(BatchInfo, 'before_update')
def validate_batch_units_consistency(mapper, connection, target):
    """Ensure batch unit counts are consistent."""
    if target.total_units is not None and target.passed_units is not None and target.failed_units is not None:
        if target.passed_units + target.failed_units > target.total_units:
            raise ValueError("Sum of passed and failed units cannot exceed total units")


class ProcessedFile(Base):
    """
    Tracks files that have been processed to enable incremental processing.

    This table allows the system to skip already-processed files, resulting in
    10x faster processing for daily incremental updates (100 new files vs 1000 total).

    Production-ready with proper validation, indexes, and duplicate detection.

    Related: ADR-002 (Incremental Processing via Database Tracking)
    Phase: 1, Day 2 (Refactoring)
    """
    __tablename__ = 'processed_files'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # File identification - required fields
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_hash = Column(String(64), nullable=False)  # SHA-256 hash for duplicate detection

    # File metadata
    file_size = Column(Integer)  # File size in bytes
    file_modified_date = Column(DateTime)  # Original file modification date

    # Processing metadata
    processed_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    processing_time_ms = Column(Integer)  # How long processing took in milliseconds
    software_version = Column(String(20))  # Version that processed the file

    # Processing result summary (for quick filtering without joining analysis_results)
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text)  # Only populated if success=False

    # Link to analysis result (optional - for files that created analysis records)
    analysis_id = Column(Integer, ForeignKey('analysis_results.id', ondelete='SET NULL'), nullable=True)

    # Production-ready indexes for performance
    __table_args__ = (
        # Fast lookups by filename (most common query)
        Index('idx_processed_filename', 'filename'),
        # Fast duplicate detection by hash
        Index('idx_processed_hash', 'file_hash'),
        # Fast path lookups
        Index('idx_processed_path', 'file_path'),
        # Combined index for common query pattern
        Index('idx_processed_filename_hash', 'filename', 'file_hash'),
        # Date-based filtering
        Index('idx_processed_date', 'processed_date'),
        # Success filtering
        Index('idx_processed_success', 'success'),
        # Unique constraint: same file (by hash) should only be processed once
        UniqueConstraint('file_hash', name='uq_processed_file_hash'),
        # Data validation constraints
        CheckConstraint("LENGTH(TRIM(filename)) > 0", name='check_pf_filename_not_empty'),
        CheckConstraint("LENGTH(TRIM(file_path)) > 0", name='check_pf_file_path_not_empty'),
        CheckConstraint("LENGTH(file_hash) = 64", name='check_pf_hash_length'),
        CheckConstraint('file_size >= 0 OR file_size IS NULL', name='check_pf_file_size_positive'),
        CheckConstraint('processing_time_ms >= 0 OR processing_time_ms IS NULL', name='check_pf_processing_time_positive'),
    )

    # Relationship to analysis result
    analysis = relationship("AnalysisResult", backref="processed_file_record", foreign_keys=[analysis_id])

    @validates('filename')
    def validate_filename(self, key, filename):
        """Validate filename is not empty."""
        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")
        return filename.strip()

    @validates('file_path')
    def validate_file_path(self, key, file_path):
        """Validate file_path is not empty."""
        if not file_path or not file_path.strip():
            raise ValueError("File path cannot be empty")
        return file_path.strip()

    @validates('file_hash')
    def validate_file_hash(self, key, file_hash):
        """Validate file_hash is a valid SHA-256 hash (64 hex characters)."""
        if not file_hash or len(file_hash) != 64:
            raise ValueError("File hash must be a 64-character SHA-256 hash")
        # Validate it's hexadecimal
        try:
            int(file_hash, 16)
        except ValueError:
            raise ValueError("File hash must be a valid hexadecimal string")
        return file_hash.lower()  # Normalize to lowercase

    @validates('file_size')
    def validate_file_size(self, key, file_size):
        """Validate file_size is non-negative."""
        if file_size is not None and file_size < 0:
            raise ValueError("File size cannot be negative")
        return file_size

    @validates('processing_time_ms')
    def validate_processing_time_ms(self, key, processing_time_ms):
        """Validate processing_time_ms is non-negative."""
        if processing_time_ms is not None and processing_time_ms < 0:
            raise ValueError("Processing time cannot be negative")
        return processing_time_ms

    def __repr__(self):
        return f"<ProcessedFile(id={self.id}, filename='{self.filename}', hash='{self.file_hash[:8]}...', success={self.success})>"
