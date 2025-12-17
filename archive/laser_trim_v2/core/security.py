"""
Security and Validation Framework for Laser Trim Analyzer

Provides comprehensive security measures including input validation,
sanitization, and protection against common vulnerabilities.
"""

import re
import os
import hashlib
import hmac
import secrets
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, TypeVar
from datetime import datetime, timedelta
import json
import pickle
from functools import wraps
import threading
from dataclasses import dataclass
from enum import Enum

# Import validators for enhanced validation
from laser_trim_analyzer.utils.validators import ValidationResult
from laser_trim_analyzer.core.error_handlers import (
    ErrorCode, ErrorCategory, ErrorSeverity,
    error_handler, handle_errors
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SecurityLevel(Enum):
    """Security validation levels."""
    LOW = "low"          # Basic validation
    MEDIUM = "medium"    # Standard validation + sanitization
    HIGH = "high"        # Comprehensive validation + security checks
    PARANOID = "paranoid"  # Maximum security, may impact usability


class ThreatType(Enum):
    """Types of security threats."""
    SQL_INJECTION = "sql_injection"
    PATH_TRAVERSAL = "path_traversal"
    XSS = "cross_site_scripting"
    COMMAND_INJECTION = "command_injection"
    FILE_UPLOAD = "malicious_file_upload"
    BUFFER_OVERFLOW = "buffer_overflow"
    DOS = "denial_of_service"
    DATA_TAMPERING = "data_tampering"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class SecurityValidationResult:
    """Result of security validation."""
    is_safe: bool
    threats_detected: List[ThreatType]
    sanitized_value: Any
    validation_errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class SecurityValidator:
    """
    Comprehensive security validator for all inputs.
    """
    
    # Regex patterns for common injection attacks
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|#|/\*|\*/)",
        r"(\bOR\b\s*\d+\s*=\s*\d+)",
        r"(\bAND\b\s*\d+\s*=\s*\d+)",
        r"('\s*OR\s*'[^']*'\s*=\s*'[^']*')",
        r"(;\s*(DROP|DELETE|UPDATE|INSERT))"
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%252e%252e%252f",
        r"\.\.%2f",
        r"\.\.%5c"
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$()]",
        r"\$\([^)]+\)",
        r"`[^`]+`",
        r"\|\|",
        r"&&"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>"
    ]
    
    # File upload restrictions
    DANGEROUS_EXTENSIONS = [
        '.exe', '.dll', '.com', '.bat', '.cmd', '.scr', '.vbs', '.js',
        '.jar', '.zip', '.rar', '.sh', '.ps1', '.psm1', '.app', '.deb',
        '.rpm', '.msi', '.pkg', '.dmg', '.iso', '.img', '.vhd', '.py',
        '.rb', '.pl', '.php', '.asp', '.aspx', '.jsp', '.cgi'
    ]
    
    # Safe file extensions for our application
    ALLOWED_EXTENSIONS = ['.xlsx', '.xls', '.csv', '.json', '.txt', '.log']
    
    # Maximum sizes for different inputs
    MAX_STRING_LENGTH = 10000
    MAX_FILENAME_LENGTH = 255
    MAX_PATH_LENGTH = 4096
    MAX_NUMBER_VALUE = 1e10
    MAX_ARRAY_SIZE = 10000
    MAX_DICT_SIZE = 1000
    MAX_FILE_SIZE_MB = 100
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        """Initialize security validator with specified security level."""
        self.security_level = security_level
        self.logger = logging.getLogger(__name__)
        self._validation_cache = {}
        self._cache_lock = threading.Lock()
    
    @handle_errors(
        category=ErrorCategory.VALIDATION,
        severity=ErrorSeverity.WARNING
    )
    def validate_input(
        self,
        value: Any,
        input_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SecurityValidationResult:
        """
        Validate and sanitize any input based on type and context.
        
        Args:
            value: The value to validate
            input_type: Type of input (string, number, file_path, etc.)
            context: Additional context for validation
            
        Returns:
            SecurityValidationResult with validation status and sanitized value
        """
        if value is None:
            return SecurityValidationResult(
                is_safe=True,
                threats_detected=[],
                sanitized_value=None,
                validation_errors=[],
                warnings=[],
                metadata={}
            )
        
        # Dispatch to appropriate validator
        validators = {
            'string': self._validate_string,
            'number': self._validate_number,
            'file_path': self._validate_file_path,
            'filename': self._validate_filename,
            'sql_parameter': self._validate_sql_parameter,
            'array': self._validate_array,
            'dict': self._validate_dict,
            'model_number': self._validate_model_number,
            'serial_number': self._validate_serial_number,
            'date': self._validate_date,
            'email': self._validate_email,
            'url': self._validate_url
        }
        
        validator = validators.get(input_type, self._validate_generic)
        return validator(value, context or {})
    
    def _validate_string(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate string input for security threats."""
        if not isinstance(value, str):
            return SecurityValidationResult(
                is_safe=False,
                threats_detected=[],
                sanitized_value=None,
                validation_errors=["Value must be a string"],
                warnings=[],
                metadata={}
            )
        
        threats = []
        errors = []
        warnings = []
        
        # Check length
        max_length = context.get('max_length', self.MAX_STRING_LENGTH)
        if len(value) > max_length:
            errors.append(f"String exceeds maximum length of {max_length}")
            threats.append(ThreatType.BUFFER_OVERFLOW)
        
        # Check for null bytes
        if '\0' in value:
            threats.append(ThreatType.DATA_TAMPERING)
            errors.append("String contains null bytes")
        
        # Check for SQL injection patterns
        if self.security_level.value in ['high', 'paranoid']:
            for pattern in self.SQL_INJECTION_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    threats.append(ThreatType.SQL_INJECTION)
                    warnings.append("Potential SQL injection pattern detected")
                    break
        
        # Check for XSS patterns
        if context.get('check_xss', True):
            for pattern in self.XSS_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    threats.append(ThreatType.XSS)
                    warnings.append("Potential XSS pattern detected")
                    break
        
        # Sanitize string
        sanitized = self._sanitize_string(value, context)
        
        return SecurityValidationResult(
            is_safe=len(errors) == 0 and len(threats) == 0,
            threats_detected=threats,
            sanitized_value=sanitized,
            validation_errors=errors,
            warnings=warnings,
            metadata={'original_length': len(value), 'sanitized_length': len(sanitized)}
        )
    
    def _sanitize_string(self, value: str, context: Dict[str, Any]) -> str:
        """Sanitize string by removing or escaping dangerous characters."""
        # Remove null bytes
        sanitized = value.replace('\0', '')
        
        # Remove control characters (except newline and tab)
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\t')
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        # Apply context-specific sanitization
        if context.get('alphanumeric_only'):
            sanitized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', sanitized)
        
        if context.get('no_special_chars'):
            sanitized = re.sub(r'[^a-zA-Z0-9\s\-_\.]', '', sanitized)
        
        # Truncate if needed
        max_length = context.get('max_length', self.MAX_STRING_LENGTH)
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def _validate_number(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate numeric input."""
        errors = []
        warnings = []
        threats = []
        
        try:
            # Convert to float
            if isinstance(value, str):
                # Check for potential code injection in string
                if any(char in value for char in ['e', 'E']) and len(value) > 10:
                    warnings.append("Large exponential notation detected")
                num_value = float(value)
            else:
                num_value = float(value)
            
            # Check range
            min_val = context.get('min', -self.MAX_NUMBER_VALUE)
            max_val = context.get('max', self.MAX_NUMBER_VALUE)
            
            if num_value < min_val or num_value > max_val:
                errors.append(f"Number {num_value} outside allowed range [{min_val}, {max_val}]")
                threats.append(ThreatType.BUFFER_OVERFLOW)
            
            # Check for special values
            import math
            if math.isnan(num_value):
                errors.append("NaN values not allowed")
                threats.append(ThreatType.DATA_TAMPERING)
            elif math.isinf(num_value):
                errors.append("Infinite values not allowed")
                threats.append(ThreatType.DATA_TAMPERING)
            
            # Sanitize precision if needed
            if context.get('precision'):
                num_value = round(num_value, context['precision'])
            
            return SecurityValidationResult(
                is_safe=len(errors) == 0,
                threats_detected=threats,
                sanitized_value=num_value,
                validation_errors=errors,
                warnings=warnings,
                metadata={'original_value': value, 'numeric_value': num_value}
            )
            
        except (ValueError, TypeError) as e:
            return SecurityValidationResult(
                is_safe=False,
                threats_detected=[ThreatType.DATA_TAMPERING],
                sanitized_value=None,
                validation_errors=[f"Invalid number format: {str(e)}"],
                warnings=[],
                metadata={}
            )
    
    def _validate_file_path(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate file path for security issues."""
        if not isinstance(value, (str, Path)):
            return SecurityValidationResult(
                is_safe=False,
                threats_detected=[],
                sanitized_value=None,
                validation_errors=["File path must be string or Path object"],
                warnings=[],
                metadata={}
            )
        
        threats = []
        errors = []
        warnings = []
        
        path_str = str(value)
        
        # Check length
        if len(path_str) > self.MAX_PATH_LENGTH:
            errors.append(f"Path exceeds maximum length of {self.MAX_PATH_LENGTH}")
            threats.append(ThreatType.BUFFER_OVERFLOW)
        
        # Check for path traversal
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path_str, re.IGNORECASE):
                threats.append(ThreatType.PATH_TRAVERSAL)
                errors.append("Path traversal attempt detected")
                break
        
        # Check for null bytes
        if '\0' in path_str:
            threats.append(ThreatType.DATA_TAMPERING)
            errors.append("Path contains null bytes")
        
        # Validate path structure
        try:
            path = Path(path_str)
            
            # Check if absolute path when required
            if context.get('require_absolute') and not path.is_absolute():
                errors.append("Absolute path required")
            
            # Check if path is within allowed directories
            allowed_dirs = context.get('allowed_directories', [])
            if allowed_dirs:
                is_allowed = False
                for allowed_dir in allowed_dirs:
                    try:
                        allowed_path = Path(allowed_dir).resolve()
                        if path.resolve().is_relative_to(allowed_path):
                            is_allowed = True
                            break
                    except Exception:
                        pass
                
                if not is_allowed:
                    threats.append(ThreatType.UNAUTHORIZED_ACCESS)
                    errors.append("Path is outside allowed directories")
            
            # Check extension if file
            if context.get('check_extension', True) and path.suffix:
                if path.suffix.lower() in self.DANGEROUS_EXTENSIONS:
                    threats.append(ThreatType.FILE_UPLOAD)
                    errors.append(f"Dangerous file extension: {path.suffix}")
                elif context.get('allowed_extensions'):
                    if path.suffix.lower() not in context['allowed_extensions']:
                        errors.append(f"File extension {path.suffix} not allowed")
            
            # Sanitize path
            sanitized_path = self._sanitize_path(path)
            
            return SecurityValidationResult(
                is_safe=len(errors) == 0 and len(threats) == 0,
                threats_detected=threats,
                sanitized_value=sanitized_path,
                validation_errors=errors,
                warnings=warnings,
                metadata={'original_path': path_str, 'resolved_path': str(path.resolve())}
            )
            
        except Exception as e:
            return SecurityValidationResult(
                is_safe=False,
                threats_detected=[ThreatType.DATA_TAMPERING],
                sanitized_value=None,
                validation_errors=[f"Invalid path: {str(e)}"],
                warnings=[],
                metadata={}
            )
    
    def _sanitize_path(self, path: Path) -> Path:
        """Sanitize file path."""
        # Resolve to absolute path to prevent traversal
        try:
            sanitized = path.resolve()
            return sanitized
        except Exception:
            # If resolve fails, return the original path
            return path
    
    def _validate_filename(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate filename for security issues."""
        if not isinstance(value, str):
            return SecurityValidationResult(
                is_safe=False,
                threats_detected=[],
                sanitized_value=None,
                validation_errors=["Filename must be a string"],
                warnings=[],
                metadata={}
            )
        
        threats = []
        errors = []
        warnings = []
        
        # Check length
        if len(value) > self.MAX_FILENAME_LENGTH:
            errors.append(f"Filename exceeds maximum length of {self.MAX_FILENAME_LENGTH}")
            threats.append(ThreatType.BUFFER_OVERFLOW)
        
        # Check for directory separators
        if any(sep in value for sep in ['/', '\\', '\0']):
            threats.append(ThreatType.PATH_TRAVERSAL)
            errors.append("Filename contains path separators")
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        if any(char in value for char in dangerous_chars):
            errors.append("Filename contains invalid characters")
        
        # Check extension
        extension = os.path.splitext(value)[1].lower()
        if extension in self.DANGEROUS_EXTENSIONS:
            threats.append(ThreatType.FILE_UPLOAD)
            errors.append(f"Dangerous file extension: {extension}")
        
        # Sanitize filename
        sanitized = self._sanitize_filename(value)
        
        return SecurityValidationResult(
            is_safe=len(errors) == 0 and len(threats) == 0,
            threats_detected=threats,
            sanitized_value=sanitized,
            validation_errors=errors,
            warnings=warnings,
            metadata={'original_filename': value, 'extension': extension}
        )
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing dangerous characters."""
        # Remove path separators and null bytes
        sanitized = filename.replace('/', '_').replace('\\', '_').replace('\0', '')
        
        # Remove other dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Ensure filename is not empty
        if not sanitized:
            sanitized = 'unnamed_file'
        
        # Truncate if too long
        if len(sanitized) > self.MAX_FILENAME_LENGTH:
            name, ext = os.path.splitext(sanitized)
            max_name_length = self.MAX_FILENAME_LENGTH - len(ext)
            sanitized = name[:max_name_length] + ext
        
        return sanitized
    
    def _validate_sql_parameter(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate SQL parameter to prevent injection."""
        # First do general string validation
        result = self._validate_string(str(value), context)
        
        # Additional SQL-specific checks
        value_str = str(value)
        
        # Check for SQL keywords and operators
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
                       'ALTER', 'EXEC', 'UNION', 'OR', 'AND', '--', '/*', '*/']
        
        for keyword in sql_keywords:
            if keyword in value_str.upper():
                result.threats_detected.append(ThreatType.SQL_INJECTION)
                result.warnings.append(f"SQL keyword '{keyword}' detected in parameter")
                result.is_safe = False
        
        # For parameterized queries, we can be less strict
        if context.get('parameterized_query', True):
            # Just ensure proper escaping
            result.sanitized_value = value_str.replace("'", "''")
        else:
            # Strict sanitization for direct SQL
            result.sanitized_value = re.sub(r'[^a-zA-Z0-9\s\-_]', '', value_str)
        
        return result
    
    def _validate_array(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate array/list input."""
        if not isinstance(value, (list, tuple)):
            return SecurityValidationResult(
                is_safe=False,
                threats_detected=[],
                sanitized_value=None,
                validation_errors=["Value must be a list or tuple"],
                warnings=[],
                metadata={}
            )
        
        threats = []
        errors = []
        warnings = []
        
        # Check size
        max_size = context.get('max_size', self.MAX_ARRAY_SIZE)
        if len(value) > max_size:
            errors.append(f"Array size {len(value)} exceeds maximum {max_size}")
            threats.append(ThreatType.DOS)
        
        # Validate each element
        element_type = context.get('element_type', 'string')
        sanitized_array = []
        
        for i, element in enumerate(value):
            element_result = self.validate_input(element, element_type, context)
            if not element_result.is_safe:
                errors.extend([f"Element {i}: {err}" for err in element_result.validation_errors])
                threats.extend(element_result.threats_detected)
            sanitized_array.append(element_result.sanitized_value)
        
        return SecurityValidationResult(
            is_safe=len(errors) == 0 and len(threats) == 0,
            threats_detected=list(set(threats)),
            sanitized_value=sanitized_array,
            validation_errors=errors,
            warnings=warnings,
            metadata={'array_size': len(value)}
        )
    
    def _validate_dict(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate dictionary input."""
        if not isinstance(value, dict):
            return SecurityValidationResult(
                is_safe=False,
                threats_detected=[],
                sanitized_value=None,
                validation_errors=["Value must be a dictionary"],
                warnings=[],
                metadata={}
            )
        
        threats = []
        errors = []
        warnings = []
        
        # Check size
        max_size = context.get('max_size', self.MAX_DICT_SIZE)
        if len(value) > max_size:
            errors.append(f"Dictionary size {len(value)} exceeds maximum {max_size}")
            threats.append(ThreatType.DOS)
        
        # Validate keys and values
        sanitized_dict = {}
        allowed_keys = context.get('allowed_keys')
        
        for key, val in value.items():
            # Validate key
            key_result = self._validate_string(str(key), {'max_length': 100})
            if not key_result.is_safe:
                errors.append(f"Invalid key '{key}': {key_result.validation_errors}")
                continue
            
            # Check if key is allowed
            if allowed_keys and key not in allowed_keys:
                errors.append(f"Key '{key}' not in allowed keys")
                threats.append(ThreatType.DATA_TAMPERING)
                continue
            
            # Validate value
            value_type = context.get('value_types', {}).get(key, 'string')
            val_result = self.validate_input(val, value_type, context)
            if not val_result.is_safe:
                errors.extend([f"Value for '{key}': {err}" for err in val_result.validation_errors])
                threats.extend(val_result.threats_detected)
            
            sanitized_dict[key_result.sanitized_value] = val_result.sanitized_value
        
        return SecurityValidationResult(
            is_safe=len(errors) == 0 and len(threats) == 0,
            threats_detected=list(set(threats)),
            sanitized_value=sanitized_dict,
            validation_errors=errors,
            warnings=warnings,
            metadata={'dict_size': len(value)}
        )
    
    def _validate_model_number(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate potentiometer model number."""
        # First do string validation
        result = self._validate_string(str(value), {
            'max_length': 50,
            'alphanumeric_only': False
        })
        
        # Model-specific validation
        model_pattern = r'^[A-Z0-9]{3,10}(-[A-Z0-9]+)*$'
        if not re.match(model_pattern, result.sanitized_value):
            result.warnings.append("Model number doesn't match expected pattern")
        
        return result
    
    def _validate_serial_number(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate serial number."""
        # First do string validation
        result = self._validate_string(str(value), {
            'max_length': 50,
            'no_special_chars': True
        })
        
        # Serial-specific validation
        serial_pattern = r'^[A-Z0-9\-_]+$'
        if not re.match(serial_pattern, result.sanitized_value):
            result.warnings.append("Serial number contains unexpected characters")
        
        return result
    
    def _validate_date(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate date input."""
        from datetime import datetime
        
        errors = []
        warnings = []
        threats = []
        
        # Try to parse date
        date_value = None
        if isinstance(value, datetime):
            date_value = value
        elif isinstance(value, str):
            # Limit string length to prevent DOS
            if len(value) > 50:
                errors.append("Date string too long")
                threats.append(ThreatType.DOS)
            else:
                # Try common formats
                formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y']
                for fmt in formats:
                    try:
                        date_value = datetime.strptime(value, fmt)
                        break
                    except ValueError:
                        continue
                
                if not date_value:
                    errors.append("Invalid date format")
        else:
            errors.append("Date must be string or datetime object")
        
        # Validate date range
        if date_value:
            min_date = context.get('min_date', datetime(1900, 1, 1))
            max_date = context.get('max_date', datetime(2100, 12, 31))
            
            if date_value < min_date or date_value > max_date:
                errors.append(f"Date outside allowed range")
                threats.append(ThreatType.DATA_TAMPERING)
        
        return SecurityValidationResult(
            is_safe=len(errors) == 0,
            threats_detected=threats,
            sanitized_value=date_value,
            validation_errors=errors,
            warnings=warnings,
            metadata={}
        )
    
    def _validate_email(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate email address."""
        if not isinstance(value, str):
            return SecurityValidationResult(
                is_safe=False,
                threats_detected=[],
                sanitized_value=None,
                validation_errors=["Email must be a string"],
                warnings=[],
                metadata={}
            )
        
        errors = []
        warnings = []
        
        # Basic email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, value):
            errors.append("Invalid email format")
        
        # Check length
        if len(value) > 254:  # RFC 5321
            errors.append("Email address too long")
        
        # Sanitize
        sanitized = value.lower().strip()
        
        return SecurityValidationResult(
            is_safe=len(errors) == 0,
            threats_detected=[],
            sanitized_value=sanitized,
            validation_errors=errors,
            warnings=warnings,
            metadata={}
        )
    
    def _validate_url(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Validate URL."""
        if not isinstance(value, str):
            return SecurityValidationResult(
                is_safe=False,
                threats_detected=[],
                sanitized_value=None,
                validation_errors=["URL must be a string"],
                warnings=[],
                metadata={}
            )
        
        errors = []
        warnings = []
        threats = []
        
        # Check length
        if len(value) > 2048:
            errors.append("URL too long")
            threats.append(ThreatType.DOS)
        
        # Basic URL validation
        url_pattern = r'^https?://[^\s]+$'
        if not re.match(url_pattern, value):
            errors.append("Invalid URL format")
        
        # Check for dangerous protocols
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:', 'file:']
        for protocol in dangerous_protocols:
            if value.lower().startswith(protocol):
                threats.append(ThreatType.XSS)
                errors.append(f"Dangerous protocol: {protocol}")
        
        return SecurityValidationResult(
            is_safe=len(errors) == 0 and len(threats) == 0,
            threats_detected=threats,
            sanitized_value=value,
            validation_errors=errors,
            warnings=warnings,
            metadata={}
        )
    
    def _validate_generic(self, value: Any, context: Dict[str, Any]) -> SecurityValidationResult:
        """Generic validation for unknown types."""
        # Convert to string and validate
        return self._validate_string(str(value), context)


class SecureFileProcessor:
    """
    Secure file upload and processing handler.
    """
    
    def __init__(self, upload_dir: Path, max_file_size_mb: float = 100):
        """Initialize secure file processor."""
        self.upload_dir = Path(upload_dir)
        self.max_file_size_mb = max_file_size_mb
        self.validator = SecurityValidator()
        self.logger = logging.getLogger(__name__)
        
        # Ensure upload directory exists and is secure
        self._secure_upload_directory()
    
    def _secure_upload_directory(self):
        """Ensure upload directory is secure."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Set restrictive permissions on Unix-like systems
        if os.name != 'nt':
            os.chmod(self.upload_dir, 0o700)
    
    @handle_errors(
        category=ErrorCategory.FILE_IO,
        severity=ErrorSeverity.ERROR
    )
    def process_upload(
        self,
        file_path: Path,
        original_filename: str,
        content_type: Optional[str] = None
    ) -> Tuple[bool, Path, Dict[str, Any]]:
        """
        Securely process an uploaded file.
        
        Args:
            file_path: Path to uploaded file
            original_filename: Original filename from upload
            content_type: MIME type if available
            
        Returns:
            Tuple of (success, secure_path, metadata)
        """
        metadata = {
            'original_filename': original_filename,
            'upload_time': datetime.now().isoformat(),
            'content_type': content_type
        }
        
        # Validate filename
        filename_result = self.validator.validate_input(
            original_filename,
            'filename'
        )
        
        if not filename_result.is_safe:
            raise ValueError(f"Invalid filename: {filename_result.validation_errors}")
        
        # Validate file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)")
        
        # Generate secure filename
        secure_filename = self._generate_secure_filename(filename_result.sanitized_value)
        secure_path = self.upload_dir / secure_filename
        
        # Validate file content
        is_valid, file_type = self._validate_file_content(file_path)
        if not is_valid:
            raise ValueError(f"Invalid file content or type")
        
        metadata['detected_type'] = file_type
        
        # Move file to secure location
        import shutil
        shutil.move(str(file_path), str(secure_path))
        
        # Set secure permissions
        if os.name != 'nt':
            os.chmod(secure_path, 0o600)
        
        metadata['secure_path'] = str(secure_path)
        metadata['file_size_mb'] = file_size_mb
        
        return True, secure_path, metadata
    
    def _generate_secure_filename(self, original_filename: str) -> str:
        """Generate a secure filename with random component."""
        # Extract extension
        _, ext = os.path.splitext(original_filename)
        
        # Generate secure name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_component = secrets.token_hex(8)
        
        return f"{timestamp}_{random_component}{ext}"
    
    def _validate_file_content(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate file content to ensure it matches expected type.
        
        Returns:
            Tuple of (is_valid, detected_type)
        """
        # Read file header
        with open(file_path, 'rb') as f:
            header = f.read(1024)
        
        # Check for Excel files
        excel_signatures = [
            b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1',  # OLE2
            b'PK\x03\x04',  # XLSX (ZIP)
            b'\x09\x08\x10\x00\x00\x06\x05\x00',  # XLS
        ]
        
        for sig in excel_signatures:
            if header.startswith(sig):
                return True, 'excel'
        
        # Check for CSV (text-based)
        try:
            header_text = header.decode('utf-8', errors='strict')
            if ',' in header_text or '\t' in header_text:
                return True, 'csv'
        except UnicodeDecodeError:
            pass
        
        return False, 'unknown'


class InputSanitizer:
    """
    Provides methods to sanitize various types of inputs.
    """
    
    @staticmethod
    def sanitize_for_html(text: str) -> str:
        """Sanitize text for safe HTML display."""
        html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#x27;",
            ">": "&gt;",
            "<": "&lt;",
        }
        return "".join(html_escape_table.get(c, c) for c in text)
    
    @staticmethod
    def sanitize_for_sql(text: str, quote_char: str = "'") -> str:
        """Sanitize text for SQL queries (use parameterized queries instead!)."""
        # This is a fallback - always use parameterized queries
        if quote_char == "'":
            return text.replace("'", "''")
        elif quote_char == '"':
            return text.replace('"', '""')
        return text
    
    @staticmethod
    def sanitize_for_shell(text: str) -> str:
        """Sanitize text for shell commands (avoid shell commands if possible!)."""
        # Remove all potentially dangerous characters
        return re.sub(r'[^a-zA-Z0-9\s\-_\./]', '', text)
    
    @staticmethod
    def sanitize_for_logging(text: str, max_length: int = 1000) -> str:
        """Sanitize text for safe logging."""
        # Remove control characters
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + '...'
        
        return sanitized


# Decorator for input validation
def validate_inputs(**validation_rules):
    """
    Decorator to validate function inputs.
    
    Usage:
        @validate_inputs(
            filename={'type': 'filename'},
            count={'type': 'number', 'min': 0, 'max': 1000}
        )
        def process_file(filename: str, count: int):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            validator = SecurityValidator()
            
            # Validate each parameter
            for param_name, rules in validation_rules.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    input_type = rules.get('type', 'string')
                    context = {k: v for k, v in rules.items() if k != 'type'}
                    
                    result = validator.validate_input(value, input_type, context)
                    
                    if not result.is_safe:
                        raise ValueError(
                            f"Invalid input for '{param_name}': {result.validation_errors}"
                        )
                    
                    # Replace with sanitized value
                    bound.arguments[param_name] = result.sanitized_value
            
            # Call function with sanitized inputs
            return func(**bound.arguments)
        
        return wrapper
    return decorator


# Global security validator instance
_security_validator: Optional[SecurityValidator] = None


def get_security_validator() -> SecurityValidator:
    """Get or create global security validator instance."""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator()
    return _security_validator