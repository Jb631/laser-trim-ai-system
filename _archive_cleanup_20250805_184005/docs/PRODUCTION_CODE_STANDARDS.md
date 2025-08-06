# Production Code Quality Standards

**Document Version:** 1.0  
**Date:** November 6, 2025  
**Application:** Laser Trim Analyzer v2  
**Purpose:** Define mandatory code quality standards for all production fixes

---

## Executive Summary

This document establishes mandatory code quality standards for all production fixes and future development. These standards ensure security, reliability, performance, and maintainability while reducing technical debt and operational risks.

**Compliance is mandatory** - All code must pass automated quality gates before deployment.

---

## 1. Code Quality Requirements

### 1.1 Error Handling Standards

#### **Exception Handling Requirements**

```python
# MANDATORY: All functions must follow this pattern

from typing import Optional, Union, Any
from laser_trim_analyzer.core.exceptions import (
    ProcessingError, ValidationError, DatabaseError
)
from laser_trim_analyzer.core.error_handlers import handle_errors, ErrorCategory, ErrorSeverity

@handle_errors(
    category=ErrorCategory.PROCESSING,
    severity=ErrorSeverity.ERROR,
    max_retries=3
)
def process_data(
    file_path: Path,
    options: ProcessOptions
) -> ProcessResult:
    """
    Process data with comprehensive error handling.
    
    Args:
        file_path: Path to input file
        options: Processing configuration
        
    Returns:
        ProcessResult: Processed data result
        
    Raises:
        ValidationError: If input validation fails
        ProcessingError: If processing fails
        DatabaseError: If database operation fails
    """
    try:
        # Input validation
        if not file_path.exists():
            raise ValidationError(
                f"File not found: {file_path}",
                error_code="FILE_NOT_FOUND",
                context={"file_path": str(file_path)}
            )
            
        # Main processing
        result = _perform_processing(file_path, options)
        
        # Success logging
        logger.info(
            "Processing completed successfully",
            extra={
                "file_path": str(file_path),
                "duration_ms": result.duration_ms,
                "record_count": result.record_count
            }
        )
        
        return result
        
    except ValidationError:
        # Re-raise validation errors with context
        raise
        
    except Exception as e:
        # Log unexpected errors with full context
        logger.error(
            f"Unexpected error in process_data: {str(e)}",
            exc_info=True,
            extra={
                "file_path": str(file_path),
                "options": options.dict()
            }
        )
        raise ProcessingError(
            "Processing failed due to unexpected error",
            original_error=e,
            error_code="PROCESSING_FAILED"
        )
        
    finally:
        # Cleanup resources
        _cleanup_temp_files()
```

#### **Error Handling Checklist**
- [ ] All functions wrapped with `@handle_errors` decorator
- [ ] Specific exception types used (not generic Exception)
- [ ] Error context included in all exceptions
- [ ] Cleanup code in finally blocks
- [ ] No bare except clauses
- [ ] All exceptions logged with appropriate level
- [ ] User-friendly error messages provided

### 1.2 Logging Standards

#### **Logging Level Guidelines**

```python
import logging
from laser_trim_analyzer.core.secure_logging import get_logger, logged_function

# Logger initialization (module level)
logger = get_logger(__name__)

# Logging levels usage:
# DEBUG: Detailed diagnostic information
logger.debug(
    "Starting processing iteration",
    extra={"iteration": i, "batch_size": batch_size}
)

# INFO: General informational messages
logger.info(
    "File processed successfully",
    extra={"file": filename, "records": count, "duration_ms": duration}
)

# WARNING: Warning conditions that should be reviewed
logger.warning(
    "Processing approaching memory limit",
    extra={"memory_used_mb": used, "memory_limit_mb": limit}
)

# ERROR: Error conditions that don't stop execution
logger.error(
    "Failed to save to cache, continuing without cache",
    exc_info=True,
    extra={"cache_key": key, "error": str(e)}
)

# CRITICAL: Critical conditions requiring immediate attention
logger.critical(
    "Database connection lost, application cannot continue",
    exc_info=True,
    extra={"connection_string": sanitized_conn_str}
)
```

#### **Structured Logging Requirements**
```python
# All log entries must include:
logger.info(
    "Operation completed",  # Clear message
    extra={
        # Mandatory fields
        "operation": "process_batch",
        "duration_ms": 1234,
        "success": True,
        
        # Context fields
        "user_id": user_id,  # If authenticated
        "session_id": session_id,
        "request_id": request_id,
        
        # Business fields
        "file_count": 10,
        "total_records": 5000,
        "errors": 0
    }
)
```

#### **Sensitive Data Logging**
```python
# NEVER log:
# - Passwords, API keys, tokens
# - Full file paths with usernames
# - Personally identifiable information
# - Full stack traces in production

# Sanitize before logging:
from laser_trim_analyzer.core.security import sanitize_for_logging

logger.info(
    "User login attempt",
    extra={
        "username": sanitize_for_logging(username),
        "ip_address": sanitize_for_logging(ip_address),
        # NEVER: "password": password
    }
)
```

### 1.3 Input Validation Standards

#### **Validation Requirements**

```python
from typing import Union, List, Optional
from pydantic import BaseModel, Field, validator
from laser_trim_analyzer.core.security import SecurityValidator
from laser_trim_analyzer.utils.validators import (
    validate_file_path, validate_model_number, validate_date_range
)

class ProcessingRequest(BaseModel):
    """All input models must use Pydantic for validation."""
    
    file_path: Path = Field(
        ...,
        description="Path to input file"
    )
    
    model_number: str = Field(
        ...,
        min_length=3,
        max_length=50,
        regex=r"^[A-Z0-9\-_]+$"
    )
    
    date_range: Optional[DateRange] = Field(
        None,
        description="Optional date range filter"
    )
    
    max_records: int = Field(
        default=10000,
        ge=1,
        le=1000000,
        description="Maximum records to process"
    )
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Custom validation for file paths."""
        # Security validation
        security_result = SecurityValidator().validate_input(
            str(v),
            'file_path'
        )
        if not security_result.is_safe:
            raise ValueError(f"Security validation failed: {security_result.threats_detected}")
            
        # Business validation
        if not v.suffix.lower() in ['.xlsx', '.xls']:
            raise ValueError(f"Invalid file type: {v.suffix}")
            
        if not v.exists():
            raise ValueError(f"File not found: {v}")
            
        return v
    
    @validator('model_number')
    def validate_model(cls, v):
        """Validate model number format."""
        if not validate_model_number(v):
            raise ValueError(f"Invalid model number format: {v}")
        return v.upper()  # Normalize to uppercase
```

#### **Validation Checklist**
- [ ] All inputs validated at entry points
- [ ] Pydantic models used for complex inputs
- [ ] Security validation for all user inputs
- [ ] Business rule validation implemented
- [ ] Clear validation error messages
- [ ] Input sanitization applied
- [ ] Length limits enforced

### 1.4 Documentation Standards

#### **Function Documentation Template**

```python
def calculate_metrics(
    data: pd.DataFrame,
    config: MetricConfig,
    cache: Optional[CacheManager] = None
) -> MetricResult:
    """
    Calculate quality metrics for the provided data.
    
    This function processes the input DataFrame to calculate various
    quality metrics including sigma, linearity, and resistance values.
    Results are cached if cache manager is provided.
    
    Args:
        data: Input DataFrame containing measurement data.
            Must include columns: ['timestamp', 'value', 'temperature']
        config: Configuration for metric calculations including
            thresholds and calculation parameters
        cache: Optional cache manager for storing results.
            If None, caching is disabled.
    
    Returns:
        MetricResult containing calculated metrics:
        - sigma_value: Calculated sigma metric
        - linearity_score: Linearity assessment (0-100)
        - resistance_stats: Statistical analysis of resistance
        
    Raises:
        ValidationError: If input data is invalid or missing required columns
        CalculationError: If metric calculation fails
        CacheError: If caching fails (non-fatal, logs warning)
        
    Example:
        >>> config = MetricConfig(threshold=0.5)
        >>> result = calculate_metrics(df, config)
        >>> print(f"Sigma: {result.sigma_value}")
        
    Note:
        Large datasets (>100k rows) may require significant memory.
        Consider using batch processing for very large inputs.
        
    See Also:
        - validate_dataframe: Input validation function
        - MetricConfig: Configuration class documentation
        - MetricResult: Result class documentation
    """
```

#### **Class Documentation Template**

```python
class DataProcessor:
    """
    High-performance data processor for laser trim analysis.
    
    This class handles the core processing pipeline including validation,
    transformation, analysis, and result generation. It supports both
    single file and batch processing modes with configurable parallelism.
    
    Attributes:
        config: Processing configuration
        validator: Data validator instance
        analyzer: Analysis engine instance
        cache: Cache manager for results
        
    Thread Safety:
        This class is thread-safe for read operations.
        Write operations require external synchronization.
        
    Performance Characteristics:
        - Memory usage: O(n) where n is input size
        - Time complexity: O(n log n) for sorting operations
        - Supports streaming for large files
        
    Example:
        >>> processor = DataProcessor(config)
        >>> result = processor.process_file("data.xlsx")
        >>> print(result.summary())
        
    See Also:
        - ProcessorConfig: Configuration options
        - StreamingProcessor: For very large files
        - BatchProcessor: For multiple file processing
    """
```

### 1.5 Type Hints Standards

#### **Type Annotation Requirements**

```python
from typing import (
    Dict, List, Optional, Union, Tuple, Any, 
    TypeVar, Generic, Protocol, Literal, Final,
    Callable, Awaitable, Generator, AsyncGenerator
)
from typing_extensions import TypedDict, NotRequired
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Type aliases for clarity
UserId = str
SessionId = str
FileHash = str
Timestamp = float

# Generic types
T = TypeVar('T')
NumberType = Union[int, float, np.number]

# Typed dictionaries for complex structures
class AnalysisResult(TypedDict):
    """Typed dictionary for analysis results."""
    file_path: Path
    timestamp: datetime
    metrics: Dict[str, NumberType]
    warnings: List[str]
    metadata: NotRequired[Dict[str, Any]]

# Protocol for interface definitions
class DataSource(Protocol):
    """Protocol defining data source interface."""
    
    def read_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Read data for date range."""
        ...
    
    def validate(self) -> bool:
        """Validate data source availability."""
        ...

# Complex function signatures
def process_batch(
    files: List[Path],
    processor: Callable[[Path], AnalysisResult],
    parallel: bool = True,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[Path, Union[AnalysisResult, Exception]]:
    """
    Process batch of files with optional parallelism.
    
    Args:
        files: List of file paths to process
        processor: Function to process individual files
        parallel: Enable parallel processing
        max_workers: Maximum worker threads
        progress_callback: Called with (completed, total)
        
    Returns:
        Dictionary mapping file paths to results or exceptions
    """
    ...

# Async function signatures
async def fetch_data_async(
    source: DataSource,
    date_range: Tuple[datetime, datetime],
    timeout: timedelta = timedelta(seconds=30)
) -> AsyncGenerator[pd.DataFrame, None]:
    """Asynchronously fetch data in chunks."""
    ...
```

### 1.6 Testing Standards

#### **Test Requirements**

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np

from laser_trim_analyzer.core.processor import DataProcessor
from laser_trim_analyzer.core.exceptions import ValidationError, ProcessingError

class TestDataProcessor:
    """Test coverage requirements: minimum 85% for all modules."""
    
    @pytest.fixture
    def processor(self):
        """Fixture providing configured processor instance."""
        config = ProcessorConfig(
            max_memory_mb=100,
            enable_caching=False  # Disable for testing
        )
        return DataProcessor(config)
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample test data."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100),
            'value': np.random.randn(100),
            'temperature': np.random.uniform(20, 30, 100)
        })
    
    # Positive test cases
    def test_process_valid_data(self, processor, sample_data, tmp_path):
        """Test processing with valid input data."""
        # Arrange
        input_file = tmp_path / "test_data.xlsx"
        sample_data.to_excel(input_file)
        
        # Act
        result = processor.process_file(input_file)
        
        # Assert
        assert result.status == "success"
        assert result.record_count == 100
        assert 'sigma_value' in result.metrics
        assert result.warnings == []
    
    # Negative test cases
    def test_process_invalid_file(self, processor):
        """Test processing with invalid file path."""
        # Arrange
        invalid_path = Path("/nonexistent/file.xlsx")
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            processor.process_file(invalid_path)
        
        assert "File not found" in str(exc_info.value)
        assert exc_info.value.error_code == "FILE_NOT_FOUND"
    
    # Edge cases
    @pytest.mark.parametrize("row_count", [0, 1, 10000])
    def test_process_edge_cases(self, processor, tmp_path, row_count):
        """Test processing with edge case data sizes."""
        # Arrange
        data = pd.DataFrame({
            'value': np.random.randn(row_count) if row_count > 0 else []
        })
        input_file = tmp_path / f"test_{row_count}_rows.xlsx"
        data.to_excel(input_file)
        
        # Act
        result = processor.process_file(input_file)
        
        # Assert
        if row_count == 0:
            assert result.status == "warning"
            assert "No data" in result.warnings[0]
        else:
            assert result.status == "success"
            assert result.record_count == row_count
    
    # Performance tests
    @pytest.mark.performance
    def test_process_large_file_performance(self, processor, tmp_path, benchmark):
        """Test processing performance with large files."""
        # Arrange
        large_data = pd.DataFrame({
            'value': np.random.randn(100000)
        })
        input_file = tmp_path / "large_test.xlsx"
        large_data.to_excel(input_file)
        
        # Act & Assert
        result = benchmark(processor.process_file, input_file)
        assert result.status == "success"
        
        # Performance assertion
        assert benchmark.stats['mean'] < 5.0  # Must complete in < 5 seconds
    
    # Integration tests
    @pytest.mark.integration
    def test_end_to_end_processing(self, tmp_path):
        """Test complete processing pipeline."""
        # This test uses real components, no mocks
        pass
    
    # Security tests
    @pytest.mark.security
    def test_path_traversal_prevention(self, processor):
        """Test that path traversal attacks are prevented."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f",
        ]
        
        for path in malicious_paths:
            with pytest.raises(ValidationError) as exc_info:
                processor.process_file(Path(path))
            assert "Security validation failed" in str(exc_info.value)
```

#### **Test Categories**
- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **Performance Tests**: Validate performance requirements
- **Security Tests**: Validate security measures
- **Regression Tests**: Prevent bug reintroduction

---

## 2. Security Requirements

### 2.1 Input Sanitization Standards

#### **Sanitization Functions**

```python
from laser_trim_analyzer.core.security import (
    sanitize_string, sanitize_path, sanitize_sql_parameter,
    sanitize_html, sanitize_filename
)

# String sanitization
def process_user_input(raw_input: str) -> str:
    """All user inputs must be sanitized."""
    # Remove control characters
    cleaned = sanitize_string(raw_input, max_length=1000)
    
    # Validate against allowed patterns
    if not re.match(r'^[\w\s\-\.]+$', cleaned):
        raise ValidationError("Invalid characters in input")
    
    return cleaned

# Path sanitization
def process_file_path(user_path: str) -> Path:
    """Sanitize and validate file paths."""
    # Sanitize path components
    safe_path = sanitize_path(user_path)
    
    # Resolve to absolute path
    abs_path = Path(safe_path).resolve()
    
    # Ensure within allowed directory
    allowed_dir = Path(get_config().data_directory).resolve()
    if not abs_path.is_relative_to(allowed_dir):
        raise SecurityError("Path traversal attempt detected")
    
    return abs_path

# SQL parameter sanitization
def build_query(model_filter: str) -> str:
    """Sanitize SQL parameters."""
    # Use parameterized queries (preferred)
    safe_param = sanitize_sql_parameter(model_filter)
    
    # Additional validation
    if safe_param.count('%') > 2:
        raise ValidationError("Too many wildcards")
    
    return safe_param
```

### 2.2 Output Encoding Requirements

#### **Output Encoding Standards**

```python
import html
import json
from urllib.parse import quote

# HTML encoding for web output
def render_user_content(content: str) -> str:
    """Encode all user content for HTML display."""
    return html.escape(content, quote=True)

# JSON encoding for API responses
def build_api_response(data: Dict[str, Any]) -> str:
    """Safely encode API responses."""
    # Sanitize data before encoding
    sanitized = {
        k: sanitize_for_output(v) for k, v in data.items()
    }
    
    # Use safe JSON encoder
    return json.dumps(
        sanitized,
        ensure_ascii=True,  # Escape Unicode
        cls=SecureJSONEncoder
    )

# File path encoding for OS compatibility
def encode_file_path(filename: str) -> str:
    """Encode filename for safe file system usage."""
    # Remove dangerous characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(safe_name) > 255:
        safe_name = safe_name[:255]
    
    # URL encode for extra safety
    return quote(safe_name, safe='')
```

### 2.3 Authentication Verification

#### **Authentication Check Requirements**

```python
from functools import wraps
from laser_trim_analyzer.core.auth import (
    get_current_user, require_auth, require_role
)

# All protected endpoints must verify authentication
@require_auth
def protected_operation(request: Request) -> Response:
    """Example of protected operation."""
    user = get_current_user(request)
    logger.info(f"User {user.id} accessing protected operation")
    
    # Perform operation
    return Response(data={"status": "success"})

# Role-based access control
@require_role(['admin', 'analyst'])
def admin_operation(request: Request) -> Response:
    """Operation requiring specific roles."""
    user = get_current_user(request)
    
    # Additional permission check
    if not user.has_permission('write'):
        raise PermissionError("Write permission required")
    
    # Perform operation
    return Response(data={"status": "success"})

# Manual authentication check pattern
def process_request(request: Request) -> Response:
    """Manual authentication verification."""
    # Verify authentication
    if not request.authenticated:
        logger.warning(
            "Unauthenticated access attempt",
            extra={"ip": request.ip_address}
        )
        raise AuthenticationError("Authentication required")
    
    # Verify session validity
    if request.session.expired:
        raise SessionExpiredError("Session expired")
    
    # Continue with processing
    return handle_request(request)
```

### 2.4 Authorization Checks

#### **Authorization Standards**

```python
from laser_trim_analyzer.core.auth import check_permission, check_resource_access

class ResourceAuthorization:
    """Authorization checker for resources."""
    
    def can_read(self, user: User, resource: Resource) -> bool:
        """Check read permission."""
        # Check role-based permission
        if user.has_role('admin'):
            return True
        
        # Check resource ownership
        if resource.owner_id == user.id:
            return True
        
        # Check explicit permissions
        return check_permission(user, resource, 'read')
    
    def can_write(self, user: User, resource: Resource) -> bool:
        """Check write permission."""
        # Admin can write anything
        if user.has_role('admin'):
            return True
        
        # Owner can write their resources
        if resource.owner_id == user.id:
            return True
        
        # Check explicit permissions
        return check_permission(user, resource, 'write')
    
    def can_delete(self, user: User, resource: Resource) -> bool:
        """Check delete permission."""
        # Only admin and owner can delete
        return (
            user.has_role('admin') or 
            resource.owner_id == user.id
        )

# Authorization decorator
def require_resource_permission(permission: str):
    """Decorator to check resource permissions."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, resource_id: str, *args, **kwargs):
            user = get_current_user()
            resource = get_resource(resource_id)
            
            if not check_resource_access(user, resource, permission):
                logger.warning(
                    f"Unauthorized access attempt",
                    extra={
                        "user_id": user.id,
                        "resource_id": resource_id,
                        "permission": permission
                    }
                )
                raise PermissionError(
                    f"Permission '{permission}' required for resource"
                )
            
            return func(self, resource_id, *args, **kwargs)
        return wrapper
    return decorator
```

### 2.5 Audit Trail Requirements

#### **Audit Logging Standards**

```python
from laser_trim_analyzer.core.audit import audit_log, AuditEvent

# Audit all data modifications
@audit_log(AuditEvent.DATA_CREATE)
def create_analysis(data: AnalysisData) -> AnalysisResult:
    """Create new analysis with audit trail."""
    result = perform_analysis(data)
    
    # Audit log includes:
    # - User ID
    # - Timestamp
    # - Operation type
    # - Resource ID
    # - Before/after values
    # - IP address
    # - Session ID
    
    return result

# Manual audit logging for complex operations
def bulk_update_operation(updates: List[Update]) -> None:
    """Perform bulk update with detailed audit trail."""
    user = get_current_user()
    
    # Start audit transaction
    with audit_transaction() as audit:
        for update in updates:
            # Capture before state
            before = get_current_state(update.resource_id)
            
            # Perform update
            result = apply_update(update)
            
            # Log audit event
            audit.log_event(
                event_type=AuditEvent.DATA_UPDATE,
                user_id=user.id,
                resource_id=update.resource_id,
                before_value=before,
                after_value=result,
                metadata={
                    "bulk_operation_id": audit.transaction_id,
                    "update_reason": update.reason
                }
            )
```

### 2.6 Data Encryption Standards

#### **Encryption Requirements**

```python
from laser_trim_analyzer.core.crypto import (
    encrypt_sensitive_data, decrypt_sensitive_data,
    hash_password, verify_password
)

# Encryption for data at rest
class EncryptedField:
    """Encrypted database field."""
    
    def __init__(self, encryption_key: bytes):
        self.key = encryption_key
    
    def encrypt(self, value: str) -> str:
        """Encrypt value for storage."""
        if not value:
            return value
        
        return encrypt_sensitive_data(value, self.key)
    
    def decrypt(self, encrypted: str) -> str:
        """Decrypt value from storage."""
        if not encrypted:
            return encrypted
        
        return decrypt_sensitive_data(encrypted, self.key)

# Password hashing
def store_user_password(password: str) -> str:
    """Hash password for storage."""
    # Use bcrypt with cost factor 12
    return hash_password(password, cost_factor=12)

def verify_user_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash."""
    return verify_password(password, stored_hash)

# Encryption for data in transit
class SecureAPIClient:
    """API client with encryption."""
    
    def __init__(self, api_key: str, use_tls: bool = True):
        self.api_key = api_key
        self.use_tls = use_tls
    
    def make_request(self, endpoint: str, data: Dict) -> Response:
        """Make encrypted API request."""
        # Enforce HTTPS
        if not endpoint.startswith('https://'):
            raise SecurityError("HTTPS required for API calls")
        
        # Encrypt sensitive fields
        encrypted_data = self._encrypt_sensitive_fields(data)
        
        # Sign request
        signature = self._sign_request(encrypted_data)
        
        # Make request with certificate validation
        return requests.post(
            endpoint,
            json=encrypted_data,
            headers={
                'X-API-Key': self.api_key,
                'X-Signature': signature
            },
            verify=True  # Verify SSL certificate
        )
```

---

## 3. Performance Requirements

### 3.1 Database Query Optimization

#### **Query Optimization Standards**

```python
from sqlalchemy import select, and_, or_, func, Index
from sqlalchemy.orm import selectinload, joinedload, contains_eager

# Optimized query patterns
class OptimizedQueries:
    """Database query optimization patterns."""
    
    # Use eager loading to prevent N+1 queries
    def get_analysis_with_tracks(self, analysis_id: str) -> AnalysisResult:
        """Fetch analysis with all tracks in one query."""
        return (
            self.session.query(DBAnalysisResult)
            .options(
                joinedload(DBAnalysisResult.tracks)
                .selectinload(DBTrackResult.metrics)
            )
            .filter(DBAnalysisResult.id == analysis_id)
            .first()
        )
    
    # Use query result caching
    @cached_query(ttl=300)  # Cache for 5 minutes
    def get_model_statistics(self, model: str) -> Dict[str, float]:
        """Get cached model statistics."""
        return (
            self.session.query(
                func.avg(DBAnalysisResult.sigma_value).label('avg_sigma'),
                func.min(DBAnalysisResult.sigma_value).label('min_sigma'),
                func.max(DBAnalysisResult.sigma_value).label('max_sigma'),
                func.count(DBAnalysisResult.id).label('count')
            )
            .filter(DBAnalysisResult.model == model)
            .first()
        )
    
    # Use bulk operations
    def bulk_insert_tracks(self, tracks: List[TrackData]) -> None:
        """Bulk insert tracks for performance."""
        # Use bulk_insert_mappings for best performance
        self.session.bulk_insert_mappings(
            DBTrackResult,
            [track.dict() for track in tracks]
        )
        
        # Or use bulk_save_objects for ORM features
        track_objects = [
            DBTrackResult(**track.dict()) for track in tracks
        ]
        self.session.bulk_save_objects(track_objects)
    
    # Use indexed queries
    def search_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[AnalysisResult]:
        """Search using indexed date columns."""
        # Ensure index exists: CREATE INDEX ix_analysis_date ON analysis_results(analysis_date)
        return (
            self.session.query(DBAnalysisResult)
            .filter(
                and_(
                    DBAnalysisResult.analysis_date >= start_date,
                    DBAnalysisResult.analysis_date <= end_date
                )
            )
            .order_by(DBAnalysisResult.analysis_date.desc())
            .limit(1000)  # Always limit results
            .all()
        )
```

#### **Query Performance Checklist**
- [ ] All queries use appropriate indexes
- [ ] Eager loading used to prevent N+1 queries
- [ ] Query results limited (no unlimited queries)
- [ ] Bulk operations for multiple inserts/updates
- [ ] Query execution plans reviewed
- [ ] Slow query logging enabled
- [ ] Connection pooling configured

### 3.2 Caching Implementation

#### **Caching Standards**

```python
from functools import lru_cache, wraps
from laser_trim_analyzer.core.cache_manager import (
    CacheManager, cache_key, invalidate_cache
)

class CachingPatterns:
    """Standard caching implementation patterns."""
    
    def __init__(self):
        self.cache = CacheManager(
            max_size_mb=1024,
            ttl_seconds=3600,
            eviction_policy='lru'
        )
    
    # Method-level caching
    @lru_cache(maxsize=128)
    def get_calculated_metric(
        self,
        data_hash: str,
        metric_type: str
    ) -> float:
        """Cache expensive calculations."""
        return self._perform_calculation(data_hash, metric_type)
    
    # Flexible caching with TTL
    def get_analysis_result(
        self,
        file_path: Path,
        force_refresh: bool = False
    ) -> AnalysisResult:
        """Get cached or fresh analysis result."""
        # Generate cache key
        key = cache_key('analysis', file_path, self._get_file_hash(file_path))
        
        # Check cache unless forced refresh
        if not force_refresh:
            cached = self.cache.get(key)
            if cached:
                logger.debug(f"Cache hit for {file_path}")
                return cached
        
        # Compute result
        logger.debug(f"Cache miss for {file_path}")
        result = self._analyze_file(file_path)
        
        # Store in cache with metadata
        self.cache.set(
            key,
            result,
            ttl=3600,  # 1 hour
            metadata={
                'file_path': str(file_path),
                'timestamp': datetime.now(),
                'version': self.VERSION
            }
        )
        
        return result
    
    # Cache invalidation patterns
    def update_analysis(
        self,
        file_path: Path,
        updates: Dict[str, Any]
    ) -> None:
        """Update analysis with cache invalidation."""
        # Perform update
        self._apply_updates(file_path, updates)
        
        # Invalidate affected caches
        invalidate_cache(
            pattern=f"analysis:{file_path}:*",
            reason="Data updated"
        )
        
        # Also invalidate dependent caches
        model = self._get_model_from_file(file_path)
        invalidate_cache(
            pattern=f"model_stats:{model}:*",
            reason="Model data updated"
        )
```

### 3.3 Resource Usage Limits

#### **Resource Management Standards**

```python
from laser_trim_analyzer.core.resource_manager import (
    ResourceLimiter, ResourceMonitor, ResourceExhaustedError
)

class ResourceConstraints:
    """Resource usage enforcement."""
    
    # Memory limits
    MAX_MEMORY_MB = 2048
    MAX_MEMORY_PER_REQUEST_MB = 512
    
    # CPU limits
    MAX_CPU_PERCENT = 80
    MAX_THREADS = 16
    
    # I/O limits
    MAX_OPEN_FILES = 100
    MAX_FILE_SIZE_MB = 100
    
    # Connection limits
    MAX_DB_CONNECTIONS = 20
    MAX_API_CONNECTIONS = 10

# Resource-aware processing
class ResourceAwareProcessor:
    """Processor with resource management."""
    
    def __init__(self):
        self.limiter = ResourceLimiter(
            max_memory_mb=ResourceConstraints.MAX_MEMORY_MB,
            max_cpu_percent=ResourceConstraints.MAX_CPU_PERCENT
        )
        self.monitor = ResourceMonitor()
    
    def process_with_limits(
        self,
        data: pd.DataFrame
    ) -> ProcessResult:
        """Process data with resource constraints."""
        # Check available resources
        if not self.limiter.can_allocate(
            memory_mb=data.memory_usage().sum() / 1024 / 1024
        ):
            raise ResourceExhaustedError("Insufficient memory")
        
        # Acquire resource lock
        with self.limiter.acquire_resources(
            memory_mb=512,
            cpu_percent=50,
            timeout=30
        ) as resources:
            # Monitor resource usage
            with self.monitor.track_operation("process_data"):
                result = self._process(data)
            
            # Log resource usage
            logger.info(
                "Processing completed",
                extra={
                    "memory_used_mb": resources.memory_used_mb,
                    "cpu_time_seconds": resources.cpu_time,
                    "duration_seconds": resources.duration
                }
            )
            
            return result
```

### 3.4 Response Time Requirements

#### **Performance SLA Standards**

```python
from laser_trim_analyzer.core.performance import (
    track_performance, PerformanceTimer, SLAViolationError
)

# Response time SLAs
class ResponseTimeSLA:
    """Service Level Agreement for response times."""
    
    # API endpoints (milliseconds)
    API_ANALYSIS = 1000      # 1 second
    API_QUERY = 500          # 500ms
    API_HEALTH = 100         # 100ms
    
    # Processing operations (seconds)
    FILE_SMALL = 5           # < 10MB
    FILE_MEDIUM = 30         # 10-50MB
    FILE_LARGE = 120         # 50-100MB
    
    # UI operations (milliseconds)
    UI_INTERACTION = 100     # User action feedback
    UI_NAVIGATION = 500      # Page transitions
    UI_DATA_LOAD = 2000      # Data loading

# Performance tracking decorator
@track_performance(sla_ms=ResponseTimeSLA.API_ANALYSIS)
async def analyze_file_api(
    file_path: Path
) -> AnalysisResult:
    """API endpoint with SLA tracking."""
    async with PerformanceTimer() as timer:
        result = await process_file_async(file_path)
        
        # Check SLA
        if timer.elapsed_ms > ResponseTimeSLA.API_ANALYSIS:
            logger.warning(
                f"SLA violation: {timer.elapsed_ms}ms > {ResponseTimeSLA.API_ANALYSIS}ms",
                extra={
                    "endpoint": "analyze_file",
                    "file_size_mb": file_path.stat().st_size / 1024 / 1024
                }
            )
        
        return result

# Timeout enforcement
async def process_with_timeout(
    data: pd.DataFrame,
    timeout_seconds: int = 30
) -> ProcessResult:
    """Process with enforced timeout."""
    try:
        return await asyncio.wait_for(
            process_async(data),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        logger.error(
            f"Processing timeout after {timeout_seconds}s",
            extra={"data_size": len(data)}
        )
        raise TimeoutError(
            f"Processing exceeded {timeout_seconds}s limit"
        )
```

### 3.5 Memory Usage Constraints

#### **Memory Management Standards**

```python
import gc
import psutil
from memory_profiler import profile

class MemoryManager:
    """Memory usage management."""
    
    @staticmethod
    def check_memory_available(required_mb: int) -> bool:
        """Check if sufficient memory is available."""
        available = psutil.virtual_memory().available / 1024 / 1024
        return available > required_mb * 1.5  # 50% buffer
    
    @staticmethod
    def estimate_dataframe_memory(df: pd.DataFrame) -> int:
        """Estimate DataFrame memory usage in MB."""
        return df.memory_usage(deep=True).sum() / 1024 / 1024
    
    @profile  # Memory profiling decorator
    def process_large_file(
        self,
        file_path: Path,
        chunk_size: int = 10000
    ) -> ProcessResult:
        """Process large file with memory constraints."""
        # Estimate memory requirement
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        estimated_memory_mb = file_size_mb * 3  # Conservative estimate
        
        if not self.check_memory_available(estimated_memory_mb):
            # Use chunked processing
            return self._process_in_chunks(file_path, chunk_size)
        
        # Process normally
        return self._process_full_file(file_path)
    
    def _process_in_chunks(
        self,
        file_path: Path,
        chunk_size: int
    ) -> ProcessResult:
        """Process file in memory-efficient chunks."""
        results = []
        
        # Read and process in chunks
        for chunk in pd.read_excel(
            file_path,
            chunksize=chunk_size
        ):
            # Process chunk
            chunk_result = self._process_chunk(chunk)
            results.append(chunk_result)
            
            # Force garbage collection
            del chunk
            gc.collect()
        
        # Combine results
        return self._combine_results(results)
```

### 3.6 Connection Pooling Standards

#### **Connection Pool Configuration**

```python
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from contextlib import contextmanager

class ConnectionPoolConfig:
    """Database connection pool standards."""
    
    # Pool configuration by environment
    PRODUCTION = {
        'pool_class': QueuePool,
        'pool_size': 20,
        'max_overflow': 10,
        'pool_timeout': 30,
        'pool_recycle': 3600,  # Recycle connections after 1 hour
        'pool_pre_ping': True,  # Test connections before use
    }
    
    DEVELOPMENT = {
        'pool_class': StaticPool,
        'pool_size': 5,
        'max_overflow': 0,
        'pool_timeout': 10,
        'pool_recycle': -1,
        'pool_pre_ping': False,
    }

class PooledDatabaseManager:
    """Database manager with connection pooling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.engine = create_engine(
            config['database_url'],
            **ConnectionPoolConfig.PRODUCTION,
            echo=False,
            echo_pool=config.get('debug', False),
            pool_logging_name='db_pool',
            connect_args={
                'connect_timeout': 10,
                'options': '-c statement_timeout=30000'  # 30s statement timeout
            }
        )
        
        # Monitor pool usage
        self._setup_pool_monitoring()
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with monitoring."""
        start_time = time.time()
        conn = None
        
        try:
            # Get connection
            conn = self.engine.connect()
            wait_time = time.time() - start_time
            
            # Log if wait was long
            if wait_time > 1.0:
                logger.warning(
                    f"Connection pool wait time: {wait_time:.2f}s",
                    extra={
                        'pool_size': self.engine.pool.size(),
                        'checked_out': self.engine.pool.checkedout()
                    }
                )
            
            yield conn
            
        finally:
            if conn:
                conn.close()
```

---

## 4. Maintainability Requirements

### 4.1 Code Complexity Limits

#### **Complexity Standards**

```python
# Maximum cyclomatic complexity: 10
# Use cognitive complexity metric for better assessment

# BAD: High complexity
def process_data_complex(data, options):
    result = []
    for item in data:
        if item.type == 'A':
            if item.value > 100:
                if item.status == 'active':
                    result.append(process_a_high(item))
                else:
                    result.append(process_a_low(item))
            else:
                if item.priority == 1:
                    result.append(process_a_priority(item))
                else:
                    result.append(process_a_normal(item))
        elif item.type == 'B':
            # More nested conditions...
            pass
    return result

# GOOD: Reduced complexity
def process_data_simple(data: List[Item], options: Options) -> List[Result]:
    """Process data with reduced complexity."""
    processors = {
        'A': _process_type_a,
        'B': _process_type_b,
        'C': _process_type_c,
    }
    
    results = []
    for item in data:
        processor = processors.get(item.type, _process_default)
        result = processor(item, options)
        results.append(result)
    
    return results

def _process_type_a(item: Item, options: Options) -> Result:
    """Process type A items."""
    if item.value > 100 and item.status == 'active':
        return process_a_high(item)
    
    if item.priority == 1:
        return process_a_priority(item)
    
    return process_a_normal(item)
```

### 4.2 Function Size Restrictions

#### **Function Size Standards**

```python
# Maximum function length: 50 lines (excluding docstring)
# Maximum function parameters: 5

# BAD: Too long function
def process_and_analyze_file_long(
    file_path, config, validator, analyzer,
    cache, logger, monitor, options, callbacks
):  # Too many parameters
    # 200+ lines of code...
    pass

# GOOD: Properly sized functions
class FileProcessor:
    """File processor with appropriate function sizes."""
    
    def __init__(self, config: ProcessConfig):
        """Initialize processor with configuration."""
        self.config = config
        self.validator = DataValidator(config)
        self.analyzer = DataAnalyzer(config)
        self.cache = CacheManager(config.cache)
    
    def process_file(
        self,
        file_path: Path,
        options: ProcessOptions = None
    ) -> ProcessResult:
        """
        Process single file with validation and analysis.
        
        Main orchestration function that delegates to specific methods.
        """
        options = options or ProcessOptions()
        
        # Validate input
        validation_result = self._validate_file(file_path)
        if not validation_result.is_valid:
            return ProcessResult.validation_failed(validation_result)
        
        # Load data
        data = self._load_data(file_path, options)
        
        # Process data
        processed = self._process_data(data, options)
        
        # Analyze results
        analysis = self._analyze_results(processed, options)
        
        # Cache results
        self._cache_results(file_path, analysis)
        
        return ProcessResult.success(analysis)
    
    def _validate_file(self, file_path: Path) -> ValidationResult:
        """Validate file before processing."""
        return self.validator.validate_file(file_path)
    
    def _load_data(
        self,
        file_path: Path,
        options: ProcessOptions
    ) -> pd.DataFrame:
        """Load data from file."""
        # 20-30 lines of focused loading logic
        pass
```

### 4.3 Module Organization Standards

#### **Module Structure**

```
laser_trim_analyzer/
├── __init__.py          # Package initialization
├── core/                # Core business logic
│   ├── __init__.py
│   ├── models.py        # Data models (< 500 lines)
│   ├── processor.py     # Main processing logic
│   ├── validators.py    # Input validation
│   └── exceptions.py    # Custom exceptions
├── api/                 # External interfaces
│   ├── __init__.py
│   ├── client.py        # API client
│   ├── schemas.py       # API schemas
│   └── auth.py          # Authentication
├── database/            # Data persistence
│   ├── __init__.py
│   ├── models.py        # ORM models
│   ├── manager.py       # Database operations
│   └── migrations/      # Schema migrations
├── services/            # Business services
│   ├── __init__.py
│   ├── analysis.py      # Analysis service
│   ├── reporting.py     # Report generation
│   └── ml_service.py    # ML operations
├── utils/               # Shared utilities
│   ├── __init__.py
│   ├── logging.py       # Logging utilities
│   ├── cache.py         # Caching utilities
│   └── security.py      # Security utilities
└── tests/               # Test modules
    ├── unit/            # Unit tests
    ├── integration/     # Integration tests
    └── fixtures/        # Test fixtures
```

#### **Module Standards**
- Single responsibility per module
- Clear public API in `__init__.py`
- Maximum 500 lines per module
- Logical grouping of related functionality
- Minimal cross-module dependencies

### 4.4 Naming Convention Enforcement

#### **Naming Standards**

```python
# Module names: lowercase with underscores
# laser_trim_analyzer/core/data_processor.py

# Class names: PascalCase
class DataProcessor:
    """Process laser trim data."""
    pass

class HTTPAPIClient:
    """HTTP API client implementation."""
    pass

# Function names: lowercase with underscores
def calculate_sigma_value(data: pd.DataFrame) -> float:
    """Calculate sigma value from data."""
    pass

def validate_model_number(model: str) -> bool:
    """Validate model number format."""
    pass

# Constants: UPPERCASE with underscores
MAX_FILE_SIZE_MB = 100
DEFAULT_TIMEOUT_SECONDS = 30
API_VERSION = "2.0"

# Private members: leading underscore
class Analyzer:
    def __init__(self):
        self._cache = {}  # Private attribute
    
    def _calculate_internal(self):  # Private method
        pass

# Variable names: descriptive, no abbreviations
# BAD
res = calc_val(d)
for i in rng:
    proc(i)

# GOOD
result = calculate_value(data)
for item in item_range:
    process_item(item)

# Boolean names: is/has prefix
is_valid = True
has_permission = False
can_process = True
should_retry = False

# Collection names: plural
users = []
analysis_results = {}
file_paths = set()

# SQL table names: snake_case, plural
# analysis_results, track_data, user_sessions
```

### 4.5 Configuration Management

#### **Configuration Standards**

```python
from pydantic import BaseSettings, Field, validator
from typing import Optional, Dict, Any
import yaml
import os

class AppConfig(BaseSettings):
    """Application configuration with validation."""
    
    # Environment
    environment: str = Field(
        default="development",
        regex="^(development|staging|production)$"
    )
    
    # Database
    database_url: str = Field(
        ...,
        description="Database connection URL"
    )
    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=100
    )
    
    # API
    api_key: Optional[str] = Field(
        None,
        description="API key for external services"
    )
    api_timeout: int = Field(
        default=30,
        ge=1,
        le=300
    )
    
    # Performance
    max_memory_mb: int = Field(
        default=2048,
        ge=512,
        le=8192
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=16
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "LTA_"  # Laser Trim Analyzer
        case_sensitive = False
    
    @validator('database_url')
    def validate_database_url(cls, v, values):
        """Validate database URL format."""
        if values.get('environment') == 'production':
            if 'sqlite' in v:
                raise ValueError(
                    "SQLite not allowed in production"
                )
        return v

# Configuration loading with overrides
def load_config(
    config_file: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> AppConfig:
    """Load configuration with proper precedence."""
    # 1. Default values
    config_data = {}
    
    # 2. Configuration file
    if config_file and config_file.exists():
        with open(config_file) as f:
            config_data.update(yaml.safe_load(f))
    
    # 3. Environment variables (handled by pydantic)
    
    # 4. Explicit overrides
    if overrides:
        config_data.update(overrides)
    
    # Create and validate configuration
    return AppConfig(**config_data)
```

### 4.6 Dependency Management

#### **Dependency Standards**

```python
# requirements.txt format with version pinning
"""
# Core dependencies (pin to minor version)
pandas==2.0.3
numpy==1.24.3
sqlalchemy==2.0.19

# Security updates (pin to patch version)
cryptography==41.0.3
requests==2.31.0

# Development dependencies (in requirements-dev.txt)
pytest==7.4.0
mypy==1.4.1
black==23.7.0
"""

# Dependency injection pattern
from typing import Protocol
from abc import abstractmethod

class CacheProtocol(Protocol):
    """Cache interface for dependency injection."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache."""
        ...

class Processor:
    """Processor with injected dependencies."""
    
    def __init__(
        self,
        cache: CacheProtocol,
        logger: logging.Logger,
        config: ProcessConfig
    ):
        """Initialize with injected dependencies."""
        self.cache = cache
        self.logger = logger
        self.config = config
    
    def process(self, data: Any) -> Any:
        """Process using injected dependencies."""
        # Use injected cache
        cached = self.cache.get(data.id)
        if cached:
            self.logger.info("Cache hit")
            return cached
        
        # Process data
        result = self._process_internal(data)
        
        # Store in cache
        self.cache.set(data.id, result, ttl=3600)
        
        return result
```

---

## 5. Quality Gates and Enforcement

### 5.1 Automated Quality Checks

```yaml
# .github/workflows/quality-checks.yml
name: Code Quality Gates

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      # Code formatting
      - name: Check formatting (Black)
        run: black --check src/
      
      # Linting
      - name: Lint code (Ruff)
        run: ruff check src/
      
      # Type checking
      - name: Type check (MyPy)
        run: mypy src/ --strict
      
      # Security scanning
      - name: Security scan (Bandit)
        run: bandit -r src/ -ll
      
      # Complexity check
      - name: Complexity check
        run: radon cc src/ -nc -s
      
      # Test coverage
      - name: Test coverage
        run: pytest --cov=src --cov-fail-under=85
      
      # Documentation check
      - name: Documentation coverage
        run: interrogate -c pyproject.toml src/
```

### 5.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.285
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.4.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-ll]

  - repo: local
    hooks:
      - id: no-pickle
        name: Check for pickle usage
        entry: '(pickle\.dump|pickle\.load)'
        language: pygrep
        types: [python]
```

### 5.3 Code Review Checklist

```markdown
## Code Review Checklist

### Security
- [ ] No hardcoded secrets or credentials
- [ ] Input validation on all user inputs
- [ ] No use of pickle for serialization
- [ ] SQL queries use parameterization
- [ ] Authentication checks in place
- [ ] Audit logging for sensitive operations

### Performance  
- [ ] Database queries optimized with proper indexes
- [ ] Caching implemented where appropriate
- [ ] Resource limits enforced
- [ ] No N+1 query problems
- [ ] Memory usage within limits

### Quality
- [ ] All functions have type hints
- [ ] Comprehensive docstrings
- [ ] Error handling with specific exceptions
- [ ] Appropriate logging levels
- [ ] Test coverage > 85%
- [ ] No code complexity violations

### Maintainability
- [ ] Follows naming conventions
- [ ] Functions under 50 lines
- [ ] Single responsibility principle
- [ ] Dependencies injected
- [ ] Configuration properly managed
```

---

## 6. Conclusion

These production code quality standards are mandatory for all fixes and new development. Automated tooling enforces most standards, while code reviews ensure complete compliance. Regular audits verify ongoing adherence to these standards.

**Remember:** Quality is not negotiable in production systems. Every line of code must meet these standards before deployment.

---

**Document Approval:**
- Engineering Lead: _________________
- Security Officer: _________________
- QA Manager: _________________
- CTO: _________________

**Next Review Date:** Quarterly review scheduled