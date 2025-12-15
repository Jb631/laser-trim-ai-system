"""
Database modules for v3.

Reuses v2's well-designed database schema.

Modules:
- models: SQLAlchemy ORM models
- manager: Simplified database operations (~600 lines vs v2's 2,900+)
"""

from laser_trim_v3.database.models import (
    Base,
    SystemType,
    StatusType,
    RiskCategory,
    AlertType,
    AnalysisResult,
    TrackResult,
    MLPrediction,
    QAAlert,
    BatchInfo,
    ProcessedFile,
)

from laser_trim_v3.database.manager import (
    DatabaseManager,
    DatabaseError,
    get_database,
    reset_database,
)

__all__ = [
    # Models
    "Base",
    "SystemType",
    "StatusType",
    "RiskCategory",
    "AlertType",
    "AnalysisResult",
    "TrackResult",
    "MLPrediction",
    "QAAlert",
    "BatchInfo",
    "ProcessedFile",
    # Manager
    "DatabaseManager",
    "DatabaseError",
    "get_database",
    "reset_database",
]
