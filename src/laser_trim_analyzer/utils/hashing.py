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

    # Include modification time in cache key to detect file changes
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0  # File doesn't exist yet, will fail in hash calculation
    cache_key = f"{path_str}:{mtime}"

    # Check cache first
    if use_cache and cache_key in _hash_cache:
        return _hash_cache[cache_key]

    # Calculate hash
    sha256 = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            # 1 MB chunks: trim files are typically <1 MB, so this is one read
            # per file. Small (8 KB) chunks turned every hash into thousands of
            # network round-trips on SMB shares, which made the incremental
            # "already processed?" scan crawl.
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
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
        _hash_cache[cache_key] = file_hash

    return file_hash


def hash_bytes_for(file_path: Union[str, Path], data: bytes) -> str:
    """SHA256 of bytes ALREADY in memory, recorded under the file's cache key.

    For a caller that has read the file for its own reasons and would
    otherwise pay for a second read just to hash it. The result is stored
    under the same key `calculate_file_hash` uses, so a later call for this
    path -- `save_analysis` makes one -- is a dictionary lookup, not I/O.
    """
    path = Path(file_path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0
    cache_key = f"{path.resolve()}:{mtime}"
    file_hash = hashlib.sha256(data).hexdigest()
    if len(_hash_cache) >= _cache_max_size:
        for key in list(_hash_cache.keys())[:_cache_max_size // 2]:
            del _hash_cache[key]
    _hash_cache[cache_key] = file_hash
    return file_hash


def clear_hash_cache():
    """Clear the hash cache (useful after processing batches)."""
    global _hash_cache
    _hash_cache = {}
    logger.debug("Hash cache cleared")


def get_cache_size() -> int:
    """Get current hash cache size."""
    return len(_hash_cache)
