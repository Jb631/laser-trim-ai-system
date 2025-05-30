"""
SQLAlchemy database models for Laser Trim Analyzer v2.

This module defines all database tables and their relationships using SQLAlchemy ORM.
Designed for QA specialists to store and analyze potentiometer test results.
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum as PyEnum

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, Index, UniqueConstraint,
    Enum, JSON, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func

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


class AnalysisResult(Base):
    """
    File-level analysis results.

    This is the main table that stores information about each analyzed file.
    One file can have multiple tracks (for System A multi-track files).
    """
    __tablename__ = 'analysis_results'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # File identification
    filename = Column(String(255), nullable=False)
    file_path = Column(Text)
    file_date = Column(DateTime)
    file_hash = Column(String(64))  # SHA256 hash for deduplication

    # Basic properties
    model = Column(String(50), nullable=False)
    serial = Column(String(100), nullable=False)
    system = Column(Enum(SystemType), nullable=False)
    has_multi_tracks = Column(Boolean, default=False)

    # Overall results
    overall_status = Column(Enum(StatusType), nullable=False)
    processing_time = Column(Float)  # seconds

    # Analysis metadata
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    output_dir = Column(Text)
    software_version = Column(String(20))
    operator = Column(String(100))

    # Configuration used
    sigma_scaling_factor = Column(Float)
    filter_cutoff_frequency = Column(Float)

    # Relationships
    tracks = relationship("TrackResult", back_populates="analysis", cascade="all, delete-orphan")
    ml_predictions = relationship("MLPrediction", back_populates="analysis", cascade="all, delete-orphan")
    qa_alerts = relationship("QAAlert", back_populates="analysis", cascade="all, delete-orphan")

    # Indexes for performance
    __table_args__ = (
        Index('idx_filename_date', 'filename', 'file_date'),
        Index('idx_model_serial', 'model', 'serial'),
        Index('idx_timestamp', 'timestamp'),
        Index('idx_status', 'overall_status'),
        UniqueConstraint('file_hash', 'timestamp', name='uq_file_timestamp'),
    )

    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, filename='{self.filename}', model='{self.model}')>"


class TrackResult(Base):
    """
    Track-level results.

    Each track represents one potentiometer element (TRK1, TRK2 for System A).
    System B files have a single 'default' track.
    """
    __tablename__ = 'track_results'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to analysis
    analysis_id = Column(Integer, ForeignKey('analysis_results.id'), nullable=False)

    # Track identification
    track_id = Column(String(20), nullable=False)  # 'TRK1', 'TRK2', or 'default'
    status = Column(Enum(StatusType), nullable=False)

    # Core measurements
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

    # Zone analysis
    worst_zone = Column(Integer)
    worst_zone_position = Column(Float)
    zone_details = Column(JSON)  # Detailed zone-by-zone results

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

    # Indexes
    __table_args__ = (
        Index('idx_track_analysis', 'analysis_id', 'track_id'),
        Index('idx_sigma_gradient', 'sigma_gradient'),
        Index('idx_risk_category', 'risk_category'),
        Index('idx_failure_probability', 'failure_probability'),
        CheckConstraint('sigma_gradient >= 0', name='check_sigma_positive'),
        CheckConstraint('failure_probability >= 0 AND failure_probability <= 1',
                        name='check_probability_range'),
    )

    def __repr__(self):
        return f"<TrackResult(id={self.id}, track_id='{self.track_id}', status='{self.status}')>"


class MLPrediction(Base):
    """
    Machine learning predictions and recommendations.

    Stores ML model outputs for threshold optimization and failure prediction.
    """
    __tablename__ = 'ml_predictions'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to analysis
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
    drift_detected = Column(Boolean, default=False)
    drift_percentage = Column(Float)
    drift_direction = Column(String(20))  # 'increasing', 'decreasing'

    # Recommendations
    recommendations = Column(JSON)  # List of recommendation strings

    # Relationships
    analysis = relationship("AnalysisResult", back_populates="ml_predictions")

    # Indexes
    __table_args__ = (
        Index('idx_ml_prediction_date', 'prediction_date'),
        Index('idx_ml_analysis', 'analysis_id'),
    )

    def __repr__(self):
        return f"<MLPrediction(id={self.id}, type='{self.prediction_type}', date='{self.prediction_date}')>"


class QAAlert(Base):
    """
    Quality assurance alerts and maintenance notifications.

    Tracks important events that require QA attention.
    """
    __tablename__ = 'qa_alerts'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to analysis
    analysis_id = Column(Integer, ForeignKey('analysis_results.id'), nullable=False)

    # Alert details
    alert_type = Column(Enum(AlertType), nullable=False)
    severity = Column(String(20))  # 'Critical', 'High', 'Medium', 'Low'
    created_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Alert specifics
    track_id = Column(String(20))  # Which track triggered the alert
    metric_name = Column(String(50))  # e.g., 'sigma_gradient'
    metric_value = Column(Float)
    threshold_value = Column(Float)

    # Alert message and details
    message = Column(Text, nullable=False)
    details = Column(JSON)  # Additional context

    # Resolution tracking
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_date = Column(DateTime)
    resolved = Column(Boolean, default=False)
    resolved_by = Column(String(100))
    resolved_date = Column(DateTime)
    resolution_notes = Column(Text)

    # Relationships
    analysis = relationship("AnalysisResult", back_populates="qa_alerts")

    # Indexes
    __table_args__ = (
        Index('idx_alert_type_date', 'alert_type', 'created_date'),
        Index('idx_alert_severity', 'severity'),
        Index('idx_alert_resolved', 'resolved'),
    )

    def __repr__(self):
        return f"<QAAlert(id={self.id}, type='{self.alert_type}', severity='{self.severity}')>"


class BatchInfo(Base):
    """
    Production batch information.

    Groups analysis results by production batch for trend analysis.
    """
    __tablename__ = 'batch_info'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Batch identification
    batch_name = Column(String(100), nullable=False, unique=True)
    batch_type = Column(String(50))  # 'production', 'rework', 'test'

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

    # Indexes
    __table_args__ = (
        Index('idx_batch_name', 'batch_name'),
        Index('idx_batch_date', 'created_date'),
        Index('idx_batch_model', 'model'),
    )

    def __repr__(self):
        return f"<BatchInfo(id={self.id}, name='{self.batch_name}', units={self.total_units})>"


# Association table for many-to-many relationship between analyses and batches
class AnalysisBatch(Base):
    """Links analysis results to batches."""
    __tablename__ = 'analysis_batch'

    analysis_id = Column(Integer, ForeignKey('analysis_results.id'), primary_key=True)
    batch_id = Column(Integer, ForeignKey('batch_info.id'), primary_key=True)
    added_date = Column(DateTime, default=datetime.utcnow)