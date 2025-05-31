"""
File utility functions for the Laser Trim Analyzer.

Provides common file operations with error handling and safety features.
"""

import os
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Callable, Any
from functools import wraps
import logging

import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

def calculate_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def ensure_directory(path: Path) -> None:
    """Ensure directory exists, create if not."""
    path.mkdir(parents=True, exist_ok=True)

def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory

    Raises:
        OSError: If directory cannot be created
    """
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
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