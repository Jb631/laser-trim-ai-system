"""
File hashing utilities.

Provides centralized file hashing to avoid duplicate hash calculations
across parser, processor, and database manager.
"""

import hashlib
import logging
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)

# Cache for file hashes to avoid recalculating
_hash_cache: dict[str, str] = {}
_cache_max_size = 1000  # Limit cache size to prevent memory issues


def calculate_file_hash(file_path: Union[str, Path], use_cache: bool = True) -> str:
    """
    Calculate SHA256 hash of a file.

    Args:
        file_path: Path to the file
        use_cache: Whether to use/update the hash cache

    Returns:
        Hex string of the SHA256 hash

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read
    """
    path = Path(file_path)
    path_str = str(path.resolve())

    # Check cache first
    if use_cache and path_str in _hash_cache:
        return _hash_cache[path_str]

    # Calculate hash
    sha256 = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
    except FileNotFoundError:
        logger.error(f"File not found for hashing: {path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied reading file: {path}")
        raise

    file_hash = sha256.hexdigest()

    # Update cache (with size limit)
    if use_cache:
        if len(_hash_cache) >= _cache_max_size:
            # Remove oldest entries (simple FIFO - dict maintains insertion order in Python 3.7+)
            keys_to_remove = list(_hash_cache.keys())[:_cache_max_size // 2]
            for key in keys_to_remove:
                del _hash_cache[key]
        _hash_cache[path_str] = file_hash

    return file_hash


def clear_hash_cache():
    """Clear the hash cache (useful after processing batches)."""
    global _hash_cache
    _hash_cache = {}
    logger.debug("Hash cache cleared")


def get_cache_size() -> int:
    """Get current hash cache size."""
    return len(_hash_cache)
