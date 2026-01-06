"""
File utility functions for the Laser Trim Analyzer.

Provides common file operations with error handling and safety features.
"""

import os
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Tuple
)
from datetime import datetime
from functools import wraps
import logging
import stat
import psutil
import platform

from laser_trim_analyzer.core.error_handlers import (
    ErrorCode, ErrorCategory, ErrorSeverity,
    error_handler, handle_errors
)

# Import memory safety if available
try:
    from laser_trim_analyzer.core.memory_safety import (
        SafeFileHandle, memory_safe_string, get_memory_validator
    )
    HAS_MEMORY_SAFETY = True
except ImportError:
    HAS_MEMORY_SAFETY = False

logger = logging.getLogger(__name__)

# Maximum limits for file operations
MAX_FILENAME_LENGTH = 255
MAX_PATH_LENGTH = 4096
MAX_FILE_SIZE_MB = 1000  # 1GB max for single file operations

def calculate_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file with memory safety."""
    file_path = Path(file_path)
    
    # Check file size to prevent excessive memory usage
    if file_path.exists():
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large for hashing: {file_size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)"
            )
    
    sha256_hash = hashlib.sha256()
    total_bytes_read = 0
    max_total_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    
    try:
        if HAS_MEMORY_SAFETY:
            # Use SafeFileHandle for memory-safe file operations
            with SafeFileHandle(file_path, 'rb', max_size_mb=MAX_FILE_SIZE_MB) as f:
                for byte_block in iter(lambda: f.read(chunk_size), b""):
                    total_bytes_read += len(byte_block)
                    if total_bytes_read > max_total_bytes:
                        raise MemoryError("File read exceeded maximum size during hashing")
                    sha256_hash.update(byte_block)
        else:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(chunk_size), b""):
                    total_bytes_read += len(byte_block)
                    if total_bytes_read > max_total_bytes:
                        raise MemoryError("File read exceeded maximum size during hashing")
                    sha256_hash.update(byte_block)
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        raise
        
    return sha256_hash.hexdigest()

@handle_errors(
    category=ErrorCategory.FILE_IO,
    severity=ErrorSeverity.ERROR,
    max_retries=2
)
def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Handles permission errors, disk space issues, and path validation.

    Args:
        path: Directory path

    Returns:
        Path object for the directory

    Raises:
        OSError: If directory cannot be created
    """
    path = Path(path)
    
    # Check if path already exists and is accessible
    if path.exists():
        if not path.is_dir():
            error_handler.handle_error(
                error=ValueError(f"Path exists but is not a directory: {path}"),
                category=ErrorCategory.FILE_IO,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.INVALID_INPUT,
                user_message=f"Path '{path}' exists but is not a directory.",
                recovery_suggestions=[
                    "Choose a different path",
                    "Remove the existing file"
                ]
            )
            raise ValueError(f"Path exists but is not a directory: {path}")
        
        # Check write permissions
        if not os.access(path, os.W_OK):
            error_handler.handle_error(
                error=PermissionError(f"No write access to directory: {path}"),
                category=ErrorCategory.PERMISSION,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.PERMISSION_DENIED,
                user_message=f"No write permission for directory '{path}'.",
                recovery_suggestions=[
                    "Check directory permissions",
                    "Run application with appropriate permissions",
                    "Choose a different directory"
                ]
            )
            raise PermissionError(f"No write access to directory: {path}")
        
        return path
    
    # Check parent directory permissions
    parent = path.parent
    if parent.exists() and not os.access(parent, os.W_OK):
        error_handler.handle_error(
            error=PermissionError(f"Cannot create directory in: {parent}"),
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.ERROR,
            code=ErrorCode.PERMISSION_DENIED,
            user_message=f"Cannot create directory in '{parent}' - permission denied.",
            recovery_suggestions=[
                "Check parent directory permissions",
                "Choose a location where you have write access",
                "Run application as administrator" if platform.system() == "Windows" else "Run with appropriate permissions"
            ]
        )
        raise PermissionError(f"Cannot create directory in: {parent}")
    
    # Check disk space before creating
    required_space_mb = 10  # Minimum space needed
    if not check_disk_space(path.parent if path.parent.exists() else Path.home(), required_space_mb):
        error_handler.handle_error(
            error=OSError("Insufficient disk space"),
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.ERROR,
            code=ErrorCode.INSUFFICIENT_DISK_SPACE,
            user_message=f"Not enough disk space to create directory. Need at least {required_space_mb}MB free.",
            recovery_suggestions=[
                "Free up disk space",
                "Choose a different drive with more space",
                "Clean up temporary files"
            ]
        )
        raise OSError("Insufficient disk space")
    
    try:
        path.mkdir(parents=True, exist_ok=True)
        
        # Verify creation was successful
        if not path.exists() or not path.is_dir():
            raise OSError(f"Directory creation failed silently: {path}")
        
        return path
        
    except PermissionError as e:
        error_handler.handle_error(
            error=e,
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.ERROR,
            code=ErrorCode.PERMISSION_DENIED,
            user_message=f"Permission denied creating directory '{path}'.",
            recovery_suggestions=[
                "Check file system permissions",
                "Run with administrator/root privileges",
                "Choose a different directory"
            ],
            additional_data={'path': str(path), 'parent': str(path.parent)}
        )
        raise
        
    except OSError as e:
        if "read-only" in str(e).lower():
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.FILE_IO,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.PERMISSION_DENIED,
                user_message="Cannot create directory on read-only file system.",
                recovery_suggestions=[
                    "Check if drive is write-protected",
                    "Choose a different location",
                    "Check file system status"
                ]
            )
        else:
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.FILE_IO,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.FILE_ACCESS_DENIED,
                user_message=f"Failed to create directory '{path}'.",
                technical_details=str(e)
            )
        raise


def get_excel_files(
        directory: Path,
        pattern: str = "*.xlsx",
        recursive: bool = False
) -> List[Path]:
    """
    Get all Excel files in a directory.

    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search subdirectories

    Returns:
        List of Excel file paths
    """
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))

    # Also include .xls files
    if pattern.endswith(".xlsx"):
        xls_pattern = pattern.replace(".xlsx", ".xls")
        if recursive:
            files.extend(directory.rglob(xls_pattern))
        else:
            files.extend(directory.glob(xls_pattern))

    # Filter out temporary files
    files = [f for f in files if not f.name.startswith("~$")]

    return sorted(files)


def calculate_file_hash(
        file_path: Path,
        algorithm: str = "sha256",
        chunk_size: int = 8192
) -> str:
    """
    Calculate hash of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use
        chunk_size: Size of chunks to read

    Returns:
        Hex digest of file hash
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hash_obj = hashlib.new(algorithm)

    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        raise


def safe_file_operation(
        operation: str = "read",
        backup: bool = False,
        max_retries: int = 3
) -> Callable:
    """
    Decorator for safe file operations with error handling and optional backup.

    Args:
        operation: Type of operation ('read', 'write', 'modify')
        backup: Whether to create backup before write operations
        max_retries: Maximum number of retry attempts

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(file_path: Path, *args, **kwargs) -> Any:
            file_path = Path(file_path)

            # Create backup if needed
            backup_path = None
            if backup and operation in ["write", "modify"] and file_path.exists():
                backup_path = create_backup(file_path)

            # Try operation with retries
            last_error = None
            for attempt in range(max_retries):
                try:
                    result = func(file_path, *args, **kwargs)

                    # Clean up backup on success
                    if backup_path and backup_path.exists():
                        backup_path.unlink()

                    return result

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for "
                        f"{operation} on {file_path}: {e}"
                    )

                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff

            # All attempts failed
            if backup_path and backup_path.exists():
                logger.info(f"Restoring backup from {backup_path}")
                shutil.copy2(backup_path, file_path)
                backup_path.unlink()

            raise last_error

        return wrapper

    return decorator


def create_backup(file_path: Path, suffix: str = ".backup") -> Path:
    """
    Create a backup copy of a file.

    Args:
        file_path: File to backup
        suffix: Suffix for backup file

    Returns:
        Path to backup file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Cannot backup non-existent file: {file_path}")

    # Create backup with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.parent / f"{file_path.stem}_{timestamp}{suffix}{file_path.suffix}"

    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise


def safe_remove(file_path: Path) -> bool:
    """
    Safely remove a file with error handling.

    Args:
        file_path: File to remove

    Returns:
        True if removed successfully, False otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return True

    try:
        file_path.unlink()
        logger.info(f"Removed file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove {file_path}: {e}")
        return False


def get_unique_filename(
        directory: Path,
        base_name: str,
        extension: str = ""
) -> Path:
    """
    Get a unique filename in a directory.

    Args:
        directory: Target directory
        base_name: Base filename
        extension: File extension

    Returns:
        Unique file path
    """
    directory = Path(directory)
    ensure_directory(directory)

    # Try base name first
    file_path = directory / f"{base_name}{extension}"
    if not file_path.exists():
        return file_path

    # Add number suffix
    counter = 1
    while True:
        file_path = directory / f"{base_name}_{counter}{extension}"
        if not file_path.exists():
            return file_path
        counter += 1


def is_file_locked(file_path: Path) -> bool:
    """
    Check if a file is locked (being used by another process).

    Args:
        file_path: File to check

    Returns:
        True if file is locked, False otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False

    try:
        # Try to open file in exclusive mode
        with open(file_path, 'r+b'):
            pass
        return False
    except (IOError, OSError):
        return True


def atomic_write(file_path: Path, content: bytes, mode: str = 'wb') -> None:
    """
    Write file atomically to prevent corruption.

    Args:
        file_path: Target file path
        content: Content to write
        mode: File open mode
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
            mode=mode,
            dir=file_path.parent,
            delete=False
    ) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    # Move temporary file to target
    try:
        temp_path.replace(file_path)
    except Exception as e:
        temp_path.unlink()
        raise


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """
    Get detailed file information.

    Args:
        file_path: File path

    Returns:
        Dictionary with file information
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stat = file_path.stat()

    return {
        'name': file_path.name,
        'path': str(file_path.absolute()),
        'size': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'accessed': datetime.fromtimestamp(stat.st_atime),
        'is_file': file_path.is_file(),
        'is_dir': file_path.is_dir(),
        'extension': file_path.suffix,
        'parent': str(file_path.parent)
    }


def check_disk_space(path: Path, required_mb: float) -> bool:
    """
    Check if there's enough disk space available.
    
    Args:
        path: Path to check (uses the disk/volume of this path)
        required_mb: Required space in megabytes
        
    Returns:
        True if enough space available, False otherwise
    """
    try:
        # Get disk usage statistics
        stat = psutil.disk_usage(str(path))
        available_mb = stat.free / (1024 * 1024)
        
        return available_mb >= required_mb
        
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        # If we can't check, assume there's enough space
        return True


def check_file_permissions(path: Path, mode: str = 'r') -> Tuple[bool, Optional[str]]:
    """
    Check if we have required permissions for a file/directory.
    
    Args:
        path: Path to check
        mode: Permission mode ('r', 'w', 'x', 'rw', etc.)
        
    Returns:
        Tuple of (has_permission, error_message)
    """
    path = Path(path)
    
    if not path.exists():
        parent = path.parent
        if not parent.exists():
            return False, f"Parent directory does not exist: {parent}"
        # Check parent directory write permission for new files
        if 'w' in mode:
            if not os.access(parent, os.W_OK):
                return False, f"No write permission in parent directory: {parent}"
        return True, None
    
    # Check requested permissions
    checks = []
    if 'r' in mode:
        if not os.access(path, os.R_OK):
            checks.append("read")
    if 'w' in mode:
        if not os.access(path, os.W_OK):
            checks.append("write")
    if 'x' in mode:
        if not os.access(path, os.X_OK):
            checks.append("execute")
    
    if checks:
        return False, f"Missing {', '.join(checks)} permission for: {path}"
    
    return True, None


@handle_errors(
    category=ErrorCategory.FILE_IO,
    severity=ErrorSeverity.ERROR
)
def safe_delete(path: Path, recursive: bool = False) -> bool:
    """
    Safely delete a file or directory with permission checking.
    
    Args:
        path: Path to delete
        recursive: If True, delete directories recursively
        
    Returns:
        True if successful, False otherwise
    """
    path = Path(path)
    
    if not path.exists():
        return True
    
    # Check permissions
    has_perm, error_msg = check_file_permissions(path, 'w')
    if not has_perm:
        error_handler.handle_error(
            error=PermissionError(error_msg),
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.WARNING,
            code=ErrorCode.PERMISSION_DENIED,
            user_message=f"Cannot delete '{path}': {error_msg}",
            recovery_suggestions=[
                "Check file permissions",
                "Close any programs using the file",
                "Run with appropriate permissions"
            ]
        )
        return False
    
    try:
        if path.is_file():
            # Check if file is locked
            if is_file_locked(path):
                error_handler.handle_error(
                    error=OSError("File is locked"),
                    category=ErrorCategory.FILE_IO,
                    severity=ErrorSeverity.WARNING,
                    code=ErrorCode.FILE_LOCKED,
                    user_message=f"Cannot delete '{path}': file is in use",
                    recovery_suggestions=[
                        "Close any programs using the file",
                        "Try again later"
                    ]
                )
                return False
            
            path.unlink()
            
        elif path.is_dir():
            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()
        
        return True
        
    except PermissionError as e:
        error_handler.handle_error(
            error=e,
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.ERROR,
            code=ErrorCode.PERMISSION_DENIED,
            user_message=f"Permission denied deleting '{path}'",
            recovery_suggestions=[
                "Check if file is read-only",
                "Run with administrator privileges",
                "Check parent directory permissions"
            ]
        )
        return False
        
    except OSError as e:
        if "directory not empty" in str(e).lower():
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.FILE_IO,
                severity=ErrorSeverity.WARNING,
                code=ErrorCode.INVALID_INPUT,
                user_message=f"Cannot delete non-empty directory '{path}'",
                recovery_suggestions=[
                    "Use recursive=True to delete directory contents",
                    "Empty the directory first"
                ]
            )
        else:
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.FILE_IO,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.FILE_ACCESS_DENIED,
                user_message=f"Failed to delete '{path}'",
                technical_details=str(e)
            )
        return False


def get_safe_filename(filename: str) -> str:
    """
    Convert a string to a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Memory safety check for string length
    if HAS_MEMORY_SAFETY:
        validator = get_memory_validator()
        valid, issue = validator.validate_string_operation(
            filename, 
            'get_safe_filename',
            max_length=MAX_PATH_LENGTH
        )
        if not valid:
            # Truncate to safe length
            filename = filename[:MAX_FILENAME_LENGTH]
    else:
        # Basic length check
        if len(filename) > MAX_PATH_LENGTH:
            logger.warning(f"Filename too long: {len(filename)} chars, truncating")
            filename = filename[:MAX_FILENAME_LENGTH]
    
    # Remove or replace invalid characters
    invalid_chars = '<>:"|?*'
    if platform.system() == 'Windows':
        invalid_chars += '\\'
    
    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip(' .')
    
    # Ensure it's not empty
    if not safe_name:
        safe_name = 'unnamed'
    
    # Truncate if too long (255 is typical limit)
    if len(safe_name) > MAX_FILENAME_LENGTH:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:MAX_FILENAME_LENGTH - len(ext)] + ext
    
    return safe_name
