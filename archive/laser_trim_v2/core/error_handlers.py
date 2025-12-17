"""
Comprehensive error handling utilities and recovery mechanisms.

This module provides centralized error handling, recovery strategies,
and user-friendly error messaging for the application.
"""

import functools
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from .exceptions import (
    LaserTrimAnalyzerError,
    ProcessingError,
    DataExtractionError,
    DatabaseError,
    ValidationError,
    ConfigurationError,
    MLPredictionError
)

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for proper handling and user notification."""
    INFO = auto()      # Informational, no action required
    WARNING = auto()   # Warning, operation continues
    ERROR = auto()     # Error, operation fails but app continues
    CRITICAL = auto()  # Critical, app may need to restart
    FATAL = auto()     # Fatal, app must shut down


class ErrorCategory(Enum):
    """Error categories for grouping and handling similar errors."""
    FILE_IO = auto()
    DATABASE = auto()
    NETWORK = auto()
    VALIDATION = auto()
    CONFIGURATION = auto()
    RESOURCE = auto()
    PERMISSION = auto()
    ML_MODEL = auto()
    USER_INPUT = auto()
    SYSTEM = auto()


class ErrorCode(Enum):
    """Standardized error codes for support and debugging."""
    # File I/O Errors (1000-1999)
    FILE_NOT_FOUND = 1001
    FILE_ACCESS_DENIED = 1002
    FILE_CORRUPTED = 1003
    FILE_TOO_LARGE = 1004
    FILE_WRONG_FORMAT = 1005
    FILE_LOCKED = 1006
    EXCEL_CORRUPT = 1007
    EXCEL_PASSWORD_PROTECTED = 1008
    EXCEL_SHEET_MISSING = 1009
    
    # Database Errors (2000-2999)
    DB_CONNECTION_FAILED = 2001
    DB_QUERY_FAILED = 2002
    DB_INTEGRITY_ERROR = 2003
    DB_MIGRATION_FAILED = 2004
    DB_POOL_EXHAUSTED = 2005
    DB_DUPLICATE_ENTRY = 2006
    DB_TIMEOUT = 2007
    
    # Network Errors (3000-3999)
    NETWORK_TIMEOUT = 3001
    NETWORK_CONNECTION_FAILED = 3002
    API_RATE_LIMITED = 3003
    API_AUTHENTICATION_FAILED = 3004
    
    # Validation Errors (4000-4999)
    INVALID_INPUT = 4001
    INVALID_RANGE = 4002
    MISSING_REQUIRED_FIELD = 4003
    DATA_INTEGRITY_ERROR = 4004
    
    # Resource Errors (5000-5999)
    INSUFFICIENT_MEMORY = 5001
    INSUFFICIENT_DISK_SPACE = 5002
    CPU_LIMIT_EXCEEDED = 5003
    
    # ML Model Errors (6000-6999)
    MODEL_NOT_FOUND = 6001
    MODEL_INITIALIZATION_FAILED = 6002
    MODEL_PREDICTION_FAILED = 6003
    MODEL_VERSION_MISMATCH = 6004
    INSUFFICIENT_TRAINING_DATA = 6005
    FEATURE_MISMATCH = 6006
    
    # Permission Errors (7000-7999)
    PERMISSION_DENIED = 7001
    ADMIN_REQUIRED = 7002
    
    # Configuration Errors (8000-8999)
    CONFIG_MISSING = 8001
    CONFIG_INVALID = 8002
    CONFIG_VERSION_MISMATCH = 8003
    
    # System Errors (9000-9999)
    UNKNOWN_ERROR = 9999


class ErrorContext:
    """Container for error context information."""
    
    def __init__(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        code: ErrorCode,
        user_message: str,
        technical_details: Optional[str] = None,
        recovery_suggestions: Optional[List[str]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        self.error = error
        self.category = category
        self.severity = severity
        self.code = code
        self.user_message = user_message
        self.technical_details = technical_details or str(error)
        self.recovery_suggestions = recovery_suggestions or []
        self.additional_data = additional_data or {}
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()


class ErrorHandler:
    """Centralized error handler with recovery strategies."""
    
    # Maximum number of errors to keep in history
    MAX_ERROR_HISTORY = 1000
    
    # Error rate limiting (max errors per minute)
    ERROR_RATE_LIMIT = 10
    ERROR_RATE_WINDOW = 60  # seconds
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.error_timestamps: List[float] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies for each error category."""
        self.recovery_strategies = {
            ErrorCategory.FILE_IO: [
                self._retry_file_operation,
                self._check_file_permissions,
                self._suggest_alternative_file
            ],
            ErrorCategory.DATABASE: [
                self._retry_database_operation,
                self._reconnect_database,
                self._use_fallback_storage
            ],
            ErrorCategory.NETWORK: [
                self._retry_with_backoff,
                self._check_network_connection,
                self._use_offline_mode
            ],
            ErrorCategory.RESOURCE: [
                self._free_memory,
                self._check_disk_space,
                self._reduce_batch_size
            ],
            ErrorCategory.ML_MODEL: [
                self._reload_model,
                self._use_fallback_model,
                self._disable_ml_features
            ]
        }
    
    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        code: Optional[ErrorCode] = None,
        user_message: Optional[str] = None,
        recovery_suggestions: Optional[List[str]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        retry_func: Optional[Callable] = None,
        max_retries: int = 3
    ) -> Optional[Any]:
        """
        Handle an error with appropriate logging, user notification, and recovery attempts.
        
        Args:
            error: The exception that occurred
            category: Category of the error
            severity: Severity level
            code: Specific error code
            user_message: User-friendly error message
            recovery_suggestions: List of recovery suggestions
            additional_data: Additional context data
            retry_func: Function to retry if recovery succeeds
            max_retries: Maximum number of retry attempts
            
        Returns:
            Result of retry_func if recovery succeeds, None otherwise
        """
        # Check rate limiting
        if not self._check_rate_limit():
            logger.warning("Error rate limit exceeded, suppressing error dialog")
            return None
        
        # Create error context
        if code is None:
            code = self._determine_error_code(error, category)
        
        if user_message is None:
            user_message = self._generate_user_message(error, category, code)
        
        context = ErrorContext(
            error=error,
            category=category,
            severity=severity,
            code=code,
            user_message=user_message,
            recovery_suggestions=recovery_suggestions,
            additional_data=additional_data
        )
        
        # Add to history
        self._add_to_history(context)
        
        # Log the error
        self._log_error(context)
        
        # Attempt recovery
        if retry_func and severity not in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            recovery_result = self._attempt_recovery(context, retry_func, max_retries)
            if recovery_result is not None:
                return recovery_result
        
        # Handle based on severity
        if severity == ErrorSeverity.FATAL:
            self._handle_fatal_error(context)
        
        return None
    
    def _check_rate_limit(self) -> bool:
        """Check if error rate limit has been exceeded."""
        current_time = time.time()
        
        # Remove old timestamps
        self.error_timestamps = [
            ts for ts in self.error_timestamps 
            if current_time - ts < self.ERROR_RATE_WINDOW
        ]
        
        # Check limit
        if len(self.error_timestamps) >= self.ERROR_RATE_LIMIT:
            return False
        
        # Add current timestamp
        self.error_timestamps.append(current_time)
        return True
    
    def _add_to_history(self, context: ErrorContext):
        """Add error to history with size limit."""
        self.error_history.append(context)
        
        # Maintain size limit
        if len(self.error_history) > self.MAX_ERROR_HISTORY:
            self.error_history = self.error_history[-self.MAX_ERROR_HISTORY:]
    
    def _log_error(self, context: ErrorContext):
        """Log error with appropriate level."""
        log_func = {
            ErrorSeverity.INFO: logger.info,
            ErrorSeverity.WARNING: logger.warning,
            ErrorSeverity.ERROR: logger.error,
            ErrorSeverity.CRITICAL: logger.critical,
            ErrorSeverity.FATAL: logger.critical
        }.get(context.severity, logger.error)
        
        log_func(
            f"[{context.code.name}] {context.user_message}",
            extra={
                'error_code': context.code.value,
                'category': context.category.name,
                'severity': context.severity.name,
                'technical_details': context.technical_details,
                'additional_data': context.additional_data
            }
        )
    
    def _determine_error_code(self, error: Exception, category: ErrorCategory) -> ErrorCode:
        """Determine appropriate error code based on exception type."""
        error_type = type(error)
        
        # Map exception types to error codes
        error_code_map = {
            FileNotFoundError: ErrorCode.FILE_NOT_FOUND,
            PermissionError: ErrorCode.PERMISSION_DENIED,
            MemoryError: ErrorCode.INSUFFICIENT_MEMORY,
            ValidationError: ErrorCode.INVALID_INPUT,
            DatabaseError: ErrorCode.DB_QUERY_FAILED,
            ConfigurationError: ErrorCode.CONFIG_INVALID,
            MLPredictionError: ErrorCode.MODEL_PREDICTION_FAILED,
        }
        
        return error_code_map.get(error_type, ErrorCode.UNKNOWN_ERROR)
    
    def _generate_user_message(self, error: Exception, category: ErrorCategory, code: ErrorCode) -> str:
        """Generate user-friendly error message."""
        messages = {
            ErrorCode.FILE_NOT_FOUND: "The file could not be found. Please check the file path.",
            ErrorCode.FILE_ACCESS_DENIED: "Access denied. Please check file permissions.",
            ErrorCode.FILE_CORRUPTED: "The file appears to be corrupted and cannot be processed.",
            ErrorCode.FILE_TOO_LARGE: "The file is too large to process. Please try a smaller file.",
            ErrorCode.FILE_WRONG_FORMAT: "Invalid file format. Please provide a valid Excel file.",
            ErrorCode.FILE_LOCKED: "The file is locked by another process. Please close it and try again.",
            ErrorCode.EXCEL_CORRUPT: "The Excel file is corrupted. Please check the file integrity.",
            ErrorCode.EXCEL_PASSWORD_PROTECTED: "The Excel file is password protected. Please remove protection.",
            ErrorCode.EXCEL_SHEET_MISSING: "Required sheet not found in Excel file.",
            
            ErrorCode.DB_CONNECTION_FAILED: "Could not connect to database. Please check your connection.",
            ErrorCode.DB_QUERY_FAILED: "Database operation failed. Please try again.",
            ErrorCode.DB_INTEGRITY_ERROR: "Data integrity error. The data may be corrupted.",
            ErrorCode.DB_DUPLICATE_ENTRY: "This entry already exists in the database.",
            
            ErrorCode.NETWORK_TIMEOUT: "Network request timed out. Please check your connection.",
            ErrorCode.NETWORK_CONNECTION_FAILED: "Network connection failed. Please check your internet.",
            ErrorCode.API_RATE_LIMITED: "API rate limit exceeded. Please wait before trying again.",
            
            ErrorCode.INVALID_INPUT: "Invalid input provided. Please check your data.",
            ErrorCode.INVALID_RANGE: "Value out of valid range. Please check the limits.",
            ErrorCode.MISSING_REQUIRED_FIELD: "Required field is missing. Please provide all required data.",
            
            ErrorCode.INSUFFICIENT_MEMORY: "Not enough memory available. Please close other applications.",
            ErrorCode.INSUFFICIENT_DISK_SPACE: "Not enough disk space. Please free up some space.",
            
            ErrorCode.MODEL_NOT_FOUND: "ML model not found. Please check installation.",
            ErrorCode.MODEL_INITIALIZATION_FAILED: "Failed to initialize ML model.",
            ErrorCode.MODEL_PREDICTION_FAILED: "ML prediction failed. Please check your data.",
            ErrorCode.FEATURE_MISMATCH: "Input features don't match model requirements.",
            
            ErrorCode.PERMISSION_DENIED: "Permission denied. Please run with appropriate permissions.",
            ErrorCode.CONFIG_MISSING: "Configuration file missing. Please check installation.",
            ErrorCode.CONFIG_INVALID: "Invalid configuration. Please check settings.",
            
            ErrorCode.UNKNOWN_ERROR: f"An unexpected error occurred: {str(error)}"
        }
        
        return messages.get(code, f"Error occurred: {str(error)}")
    
    def _attempt_recovery(self, context: ErrorContext, retry_func: Callable, max_retries: int) -> Optional[Any]:
        """Attempt to recover from error using registered strategies."""
        strategies = self.recovery_strategies.get(context.category, [])
        
        for strategy in strategies:
            try:
                if strategy(context):
                    # Recovery succeeded, retry operation
                    for attempt in range(max_retries):
                        try:
                            result = retry_func()
                            logger.info(f"Recovery successful after {attempt + 1} attempts")
                            return result
                        except Exception as e:
                            if attempt == max_retries - 1:
                                raise
                            time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.warning(f"Recovery strategy failed: {e}")
        
        return None
    
    def _handle_fatal_error(self, context: ErrorContext):
        """Handle fatal errors that require application shutdown."""
        logger.critical(f"Fatal error occurred: {context.code.name}")
        
        # Save error report
        self._save_error_report(context)
        
        # Perform cleanup if possible
        try:
            # Add cleanup logic here
            pass
        except Exception:
            pass
        
        # Exit application
        sys.exit(1)
    
    def _save_error_report(self, context: ErrorContext):
        """Save detailed error report for debugging."""
        try:
            report_dir = Path.home() / ".laser_trim_analyzer" / "error_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = context.timestamp.strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"error_report_{timestamp}.txt"
            
            with open(report_file, 'w') as f:
                f.write(f"Error Report - {context.timestamp}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Error Code: {context.code.name} ({context.code.value})\n")
                f.write(f"Category: {context.category.name}\n")
                f.write(f"Severity: {context.severity.name}\n")
                f.write(f"User Message: {context.user_message}\n\n")
                f.write(f"Technical Details:\n{context.technical_details}\n\n")
                f.write(f"Traceback:\n{context.traceback}\n\n")
                
                if context.additional_data:
                    f.write("Additional Data:\n")
                    for key, value in context.additional_data.items():
                        f.write(f"  {key}: {value}\n")
                
                if context.recovery_suggestions:
                    f.write("\nRecovery Suggestions:\n")
                    for suggestion in context.recovery_suggestions:
                        f.write(f"  - {suggestion}\n")
            
            logger.info(f"Error report saved to: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")
    
    # Recovery strategy implementations
    def _retry_file_operation(self, context: ErrorContext) -> bool:
        """Retry file operation after brief delay."""
        if context.code in [ErrorCode.FILE_LOCKED]:
            time.sleep(1)  # Wait for file to be unlocked
            return True
        return False
    
    def _check_file_permissions(self, context: ErrorContext) -> bool:
        """Check and potentially fix file permissions."""
        if context.code == ErrorCode.FILE_ACCESS_DENIED:
            file_path = context.additional_data.get('file_path')
            if file_path and os.path.exists(file_path):
                try:
                    os.chmod(file_path, 0o666)
                    return True
                except Exception:
                    pass
        return False
    
    def _suggest_alternative_file(self, context: ErrorContext) -> bool:
        """Suggest alternative file locations."""
        # This would be implemented in the GUI layer
        return False
    
    def _retry_database_operation(self, context: ErrorContext) -> bool:
        """Retry database operation with fresh connection."""
        if context.code in [ErrorCode.DB_CONNECTION_FAILED, ErrorCode.DB_TIMEOUT]:
            time.sleep(0.5)
            return True
        return False
    
    def _reconnect_database(self, context: ErrorContext) -> bool:
        """Attempt to reconnect to database."""
        # This would call the database manager's reconnect method
        return False
    
    def _use_fallback_storage(self, context: ErrorContext) -> bool:
        """Use local file storage as fallback."""
        # This would switch to file-based storage temporarily
        return False
    
    def _retry_with_backoff(self, context: ErrorContext) -> bool:
        """Retry with exponential backoff."""
        if context.code in [ErrorCode.NETWORK_TIMEOUT, ErrorCode.API_RATE_LIMITED]:
            wait_time = context.additional_data.get('retry_after', 5)
            time.sleep(wait_time)
            return True
        return False
    
    def _check_network_connection(self, context: ErrorContext) -> bool:
        """Check network connectivity."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except Exception:
            return False
    
    def _use_offline_mode(self, context: ErrorContext) -> bool:
        """Switch to offline mode."""
        # This would disable network features
        return False
    
    def _free_memory(self, context: ErrorContext) -> bool:
        """Attempt to free memory."""
        if context.code == ErrorCode.INSUFFICIENT_MEMORY:
            import gc
            gc.collect()
            return True
        return False
    
    def _check_disk_space(self, context: ErrorContext) -> bool:
        """Check available disk space."""
        try:
            import shutil
            path = context.additional_data.get('path', '.')
            stat = shutil.disk_usage(path)
            
            # Need at least 100MB free
            if stat.free > 100 * 1024 * 1024:
                return True
        except Exception:
            pass
        return False
    
    def _reduce_batch_size(self, context: ErrorContext) -> bool:
        """Reduce batch processing size."""
        # This would adjust batch size in configuration
        return False
    
    def _reload_model(self, context: ErrorContext) -> bool:
        """Attempt to reload ML model."""
        # This would trigger model reload
        return False
    
    def _use_fallback_model(self, context: ErrorContext) -> bool:
        """Use simpler fallback model."""
        # This would switch to a simpler model
        return False
    
    def _disable_ml_features(self, context: ErrorContext) -> bool:
        """Disable ML features temporarily."""
        # This would disable ML predictions
        return False
    
    def get_error_summary(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get summary of recent errors."""
        recent_errors = self.error_history[-limit:]
        
        return [
            {
                'timestamp': err.timestamp.isoformat(),
                'code': err.code.name,
                'category': err.category.name,
                'severity': err.severity.name,
                'message': err.user_message
            }
            for err in recent_errors
        ]
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.error_timestamps.clear()


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    code: Optional[ErrorCode] = None,
    user_message: Optional[str] = None,
    max_retries: int = 3,
    reraise: bool = True
):
    """
    Decorator for comprehensive error handling.
    
    Args:
        category: Error category
        severity: Error severity level
        code: Specific error code
        user_message: User-friendly message
        max_retries: Maximum retry attempts
        reraise: Whether to re-raise the exception after handling
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                result = error_handler.handle_error(
                    error=e,
                    category=category,
                    severity=severity,
                    code=code,
                    user_message=user_message,
                    retry_func=lambda: func(*args, **kwargs),
                    max_retries=max_retries
                )
                
                if result is not None:
                    return result
                
                if reraise:
                    raise
                
                return None
        
        return wrapper
    return decorator


def validate_file_upload(
    file_path: Union[str, Path],
    max_size_mb: float = 100,
    allowed_extensions: Optional[List[str]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate file before processing.
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    file_path = Path(file_path)
    
    # Check file exists
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"
    
    # Check extension
    if allowed_extensions:
        if file_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            return False, f"Invalid file type: {file_path.suffix}"
    
    # Check if file is readable
    try:
        with open(file_path, 'rb') as f:
            f.read(1)
    except PermissionError:
        return False, "File access denied. Please check permissions."
    except Exception as e:
        return False, f"Cannot read file: {str(e)}"
    
    return True, None


def check_system_resources() -> Dict[str, Any]:
    """Check system resources and return status."""
    import psutil
    
    try:
        # Memory check
        memory = psutil.virtual_memory()
        memory_status = {
            'total_mb': memory.total / (1024 * 1024),
            'available_mb': memory.available / (1024 * 1024),
            'percent_used': memory.percent,
            'status': 'ok' if memory.percent < 90 else 'warning' if memory.percent < 95 else 'critical'
        }
        
        # Disk check
        disk = psutil.disk_usage('/')
        disk_status = {
            'total_gb': disk.total / (1024 ** 3),
            'free_gb': disk.free / (1024 ** 3),
            'percent_used': disk.percent,
            'status': 'ok' if disk.percent < 90 else 'warning' if disk.percent < 95 else 'critical'
        }
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_status = {
            'percent_used': cpu_percent,
            'status': 'ok' if cpu_percent < 80 else 'warning' if cpu_percent < 90 else 'critical'
        }
        
        return {
            'memory': memory_status,
            'disk': disk_status,
            'cpu': cpu_status,
            'overall_status': 'ok' if all(
                s['status'] == 'ok' for s in [memory_status, disk_status, cpu_status]
            ) else 'warning' if any(
                s['status'] == 'critical' for s in [memory_status, disk_status, cpu_status]
            ) else 'warning'
        }
        
    except Exception as e:
        logger.error(f"Failed to check system resources: {e}")
        return {
            'error': str(e),
            'overall_status': 'unknown'
        }