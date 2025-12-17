"""
Secure Logging Module for Laser Trim Analyzer

Provides comprehensive logging with:
- Input/output sanitization
- Sensitive data masking
- Structured logging
- Performance tracking
- Debug mode support
"""

import logging
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import threading
import hashlib
import re
from dataclasses import dataclass, asdict
from enum import Enum

# Import security if available
try:
    from laser_trim_analyzer.core.security import InputSanitizer, get_security_validator
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    
    # Fallback sanitizer
    class InputSanitizer:
        @staticmethod
        def sanitize_for_logging(text: str, max_length: int = 1000) -> str:
            # Basic sanitization
            text = str(text)[:max_length]
            # Remove control characters
            text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
            return text


class LogLevel(Enum):
    """Extended log levels for comprehensive logging."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60


@dataclass
class LogContext:
    """Context information for structured logging."""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    context_data: Dict[str, Any]
    performance_data: Optional[Dict[str, float]] = None
    security_data: Optional[Dict[str, Any]] = None
    error_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.name,
            'logger': self.logger_name,
            'message': self.message,
            'context': self.context_data
        }
        
        if self.performance_data:
            data['performance'] = self.performance_data
        if self.security_data:
            data['security'] = self.security_data
        if self.error_data:
            data['error'] = self.error_data
            
        return data


class SecureLogger:
    """
    Enhanced logger with security features and comprehensive logging.
    """
    
    # Patterns for sensitive data
    SENSITIVE_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),  # SSN
        (r'\b\d{16}\b', 'CARD'),  # Credit card
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),
        (r'\b(?:password|pwd|pass|secret|token|key)[\s=:]+[\S]+', 'CREDENTIAL'),
        (r'Bearer\s+[\w\-._~+/]+=*', 'TOKEN'),
    ]
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        log_level: LogLevel = LogLevel.INFO,
        max_file_size_mb: int = 100,
        backup_count: int = 5,
        enable_performance_tracking: bool = True,
        enable_input_sanitization: bool = True,
        enable_sensitive_masking: bool = True
    ):
        """Initialize secure logger."""
        self.name = name
        self.log_dir = log_dir or Path.home() / ".laser_trim_analyzer" / "logs"
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.log_level = log_level
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_input_sanitization = enable_input_sanitization
        self.enable_sensitive_masking = enable_sensitive_masking
        
        # Performance tracking
        self._performance_data = threading.local()
        
        # Initialize logger
        self._setup_logger()
        
        # Log initialization
        self.info("SecureLogger initialized", context={
            'log_dir': str(self.log_dir),
            'log_level': log_level.name,
            'features': {
                'file_logging': enable_file_logging,
                'console_logging': enable_console_logging,
                'performance_tracking': enable_performance_tracking,
                'input_sanitization': enable_input_sanitization,
                'sensitive_masking': enable_sensitive_masking
            }
        })
    
    def _setup_logger(self):
        """Set up the underlying Python logger."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level.value)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level.value)
            console_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create rotating file handler
            from logging.handlers import RotatingFileHandler
            
            log_file = self.log_dir / f"{self.name.replace('.', '_')}.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level.value)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
            
            # Create JSON file handler for structured logs
            json_log_file = self.log_dir / f"{self.name.replace('.', '_')}_structured.jsonl"
            self.json_handler = RotatingFileHandler(
                json_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            self.json_handler.setLevel(LogLevel.DEBUG.value)
    
    def _sanitize_input(self, data: Any) -> Any:
        """Sanitize input data for safe logging."""
        if not self.enable_input_sanitization:
            return data
        
        if isinstance(data, str):
            # Use security module if available
            if HAS_SECURITY:
                return InputSanitizer.sanitize_for_logging(data)
            else:
                # Basic sanitization
                return data[:1000] if len(data) > 1000 else data
        elif isinstance(data, dict):
            return {k: self._sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_input(v) for v in data[:100]]  # Limit list size
        elif isinstance(data, (int, float, bool, type(None))):
            return data
        else:
            return str(data)[:100]
    
    def _mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in log messages."""
        if not self.enable_sensitive_masking:
            return text
        
        masked_text = text
        for pattern, label in self.SENSITIVE_PATTERNS:
            masked_text = re.sub(pattern, f'[{label}_MASKED]', masked_text, flags=re.IGNORECASE)
        
        return masked_text
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, float]] = None,
        security: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ):
        """Internal logging method with full context."""
        # Sanitize and mask message
        safe_message = self._mask_sensitive_data(str(message))
        
        # Create log context
        log_context = LogContext(
            timestamp=datetime.now(),
            level=level,
            logger_name=self.name,
            message=safe_message,
            context_data=self._sanitize_input(context or {}),
            performance_data=performance,
            security_data=security,
            error_data={
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            } if error else None
        )
        
        # Log to standard logger
        self.logger.log(level.value, safe_message)
        
        # Log structured data to JSON file
        if hasattr(self, 'json_handler'):
            try:
                json_line = json.dumps(log_context.to_dict()) + '\n'
                self.json_handler.stream.write(json_line)
                self.json_handler.stream.flush()
            except Exception as e:
                self.logger.error(f"Failed to write structured log: {e}")
    
    # Convenience methods for different log levels
    def trace(self, message: str, **kwargs):
        """Log trace level message."""
        self._log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level message."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def security(self, message: str, **kwargs):
        """Log security-related message."""
        self._log(LogLevel.SECURITY, message, **kwargs)
    
    def log_input(self, function_name: str, args: tuple, kwargs: dict):
        """Log function inputs."""
        self.debug(
            f"Function input: {function_name}",
            context={
                'function': function_name,
                'args': self._sanitize_input(args),
                'kwargs': self._sanitize_input(kwargs)
            }
        )
    
    def log_output(self, function_name: str, result: Any, execution_time: float):
        """Log function outputs."""
        self.debug(
            f"Function output: {function_name}",
            context={
                'function': function_name,
                'result_type': type(result).__name__,
                'result_preview': self._sanitize_input(result)
            },
            performance={
                'execution_time_ms': execution_time * 1000
            }
        )
    
    def log_exception(self, function_name: str, exception: Exception):
        """Log exception with full context."""
        self.error(
            f"Exception in {function_name}: {type(exception).__name__}",
            context={
                'function': function_name,
                'exception_type': type(exception).__name__,
                'exception_args': str(exception.args)
            },
            error=exception
        )
    
    def start_performance_tracking(self, operation: str):
        """Start tracking performance for an operation."""
        if not self.enable_performance_tracking:
            return
        
        if not hasattr(self._performance_data, 'operations'):
            self._performance_data.operations = {}
        
        self._performance_data.operations[operation] = time.time()
    
    def end_performance_tracking(self, operation: str) -> float:
        """End performance tracking and return duration."""
        if not self.enable_performance_tracking:
            return 0.0
        
        if not hasattr(self._performance_data, 'operations'):
            return 0.0
        
        start_time = self._performance_data.operations.get(operation)
        if start_time:
            duration = time.time() - start_time
            del self._performance_data.operations[operation]
            return duration
        
        return 0.0


def logged_function(
    logger: Optional[SecureLogger] = None,
    log_inputs: bool = True,
    log_outputs: bool = True,
    log_performance: bool = True,
    log_exceptions: bool = True
):
    """
    Decorator for comprehensive function logging.
    
    Args:
        logger: Logger instance to use
        log_inputs: Whether to log function inputs
        log_outputs: Whether to log function outputs
        log_performance: Whether to log execution time
        log_exceptions: Whether to log exceptions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            func_logger = logger or get_logger(func.__module__)
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Log inputs
            if log_inputs:
                func_logger.log_input(func_name, args, kwargs)
            
            # Track performance
            start_time = time.time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Log output
                if log_outputs:
                    func_logger.log_output(func_name, result, execution_time)
                elif log_performance:
                    func_logger.debug(
                        f"Function completed: {func_name}",
                        performance={'execution_time_ms': execution_time * 1000}
                    )
                
                return result
                
            except Exception as e:
                # Log exception
                if log_exceptions:
                    func_logger.log_exception(func_name, e)
                raise
        
        return wrapper
    return decorator


# Global logger registry
_loggers: Dict[str, SecureLogger] = {}
_logger_lock = threading.Lock()


def get_logger(
    name: str,
    log_level: Optional[LogLevel] = None,
    **kwargs
) -> SecureLogger:
    """Get or create a logger instance."""
    with _logger_lock:
        if name not in _loggers:
            _loggers[name] = SecureLogger(
                name=name,
                log_level=log_level or LogLevel.INFO,
                **kwargs
            )
        
        return _loggers[name]


def configure_logging(
    log_dir: Optional[Path] = None,
    log_level: LogLevel = LogLevel.INFO,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    **kwargs
):
    """Configure global logging settings."""
    # Set up root logger
    root_logger = get_logger(
        'laser_trim_analyzer',
        log_level=log_level,
        log_dir=log_dir,
        enable_file_logging=enable_file_logging,
        enable_console_logging=enable_console_logging,
        **kwargs
    )
    
    # Log configuration
    root_logger.info(
        "Logging configured",
        context={
            'log_dir': str(log_dir) if log_dir else 'default',
            'log_level': log_level.name,
            'file_logging': enable_file_logging,
            'console_logging': enable_console_logging
        }
    )


# Export main components
__all__ = [
    'SecureLogger',
    'LogLevel',
    'LogContext',
    'logged_function',
    'get_logger',
    'configure_logging'
]