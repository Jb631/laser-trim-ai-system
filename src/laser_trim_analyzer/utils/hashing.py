"""
File hashing utilities.

Provides centralized file hashing to avoid duplicate hash calculations
across parser, processor, and database manager.
"""

import hashlib
import logging
import functools
import os
import threading
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)

# Cache for file hashes to avoid recalculating
_hash_cache: dict[str, str] = {}
_cache_max_size = 1000  # Limit cache size to prevent memory issues
# Every ingest worker thread shares this dict. `in` then `[]`, and
# `list(keys)` then `del`, are each two steps; a worker evicting between
# another worker's two steps raised KeyError, which process_file records as an
# ERROR row for a perfectly good file. Hashing itself stays OUTSIDE the lock.
_hash_lock = threading.Lock()


def _remember(cache_key: str, file_hash: str) -> None:
    """Store one hash, evicting the oldest half first when full (FIFO: dicts keep insertion order)."""
    with _hash_lock:
        if len(_hash_cache) >= _cache_max_size:
            for key in list(_hash_cache.keys())[:_cache_max_size // 2]:
                _hash_cache.pop(key, None)
        _hash_cache[cache_key] = file_hash


# --- one stat per PARSE -------------------------------------------------------
# On the work share every os.stat() opens a handle over SMB: 113 ms per call
# (two 54 ms round trips through the VPN, measured 2026-09-20), never cached by
# Windows, and the parse path made about eight of them per file.
#
# First attempt (74924ae) shared one stat for TEN SECONDS. That was wrong: the
# "is this file already processed?" decision reads the same stat, so a file
# rewritten inside the window looked unchanged and was skipped -- exactly what
# test_incremental_skip_confirms_by_content_hash and
# test_same_path_reexport_stops_being_offered exist to forbid.
#
# The sharing is now scoped to ONE PARSE OF ONE FILE, by call extent rather
# than by clock: inside a function marked @shares_one_stat every stat of a
# path is the same answer (it describes the bytes that parse is reading);
# outside one, every stat is real. So every DECISION about whether a file has
# changed sees the file as it is now.
_parse_scope = threading.local()
_RESOLVE_MEMO: dict = {}     # path -> resolved path; does not go stale


def shares_one_stat(fn):
    """Within one call of `fn`, stat each path at most once. Re-entrant."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        outer = getattr(_parse_scope, "memo", None)
        if outer is not None:                 # already inside a parse: share its memo
            return fn(*args, **kwargs)
        _parse_scope.memo = {}
        try:
            return fn(*args, **kwargs)
        finally:
            _parse_scope.memo = None
    return wrapper


def stat_once(file_path: Union[str, Path]) -> os.stat_result:
    """os.stat -- shared inside a @shares_one_stat call, always REAL outside one.

    Raises FileNotFoundError exactly as os.stat does; failures are not kept.
    """
    key = str(file_path)
    memo = getattr(_parse_scope, "memo", None)
    if memo is None:
        return os.stat(key)
    st = memo.get(key)
    if st is None:
        st = memo[key] = os.stat(key)
    return st


def _cache_key(path: Path, known_stat: Optional[os.stat_result] = None) -> str:
    """The hash cache's key: resolved path + mtime. `known_stat` spares a caller
    that has JUST statted the file a second 113 ms conversation."""
    raw = str(path)
    resolved = _RESOLVE_MEMO.get(raw)
    if resolved is None:
        if len(_RESOLVE_MEMO) > 4096:
            _RESOLVE_MEMO.clear()
        resolved = str(path.resolve())
        _RESOLVE_MEMO[raw] = resolved
    try:
        mtime = (known_stat or stat_once(path)).st_mtime
    except OSError:
        mtime = 0  # File doesn't exist yet, will fail in hash calculation
    return f"{resolved}:{mtime}"


def calculate_file_hash(file_path: Union[str, Path], use_cache: bool = True,
                        known_stat: Optional[os.stat_result] = None) -> str:
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
    # Resolved path + mtime (mtime detects file changes); one stat, memoized.
    cache_key = _cache_key(path, known_stat)

    # Check cache first
    if use_cache:
        with _hash_lock:
            cached = _hash_cache.get(cache_key)
        if cached is not None:
            return cached

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
        _remember(cache_key, file_hash)

    return file_hash


def hash_bytes_for(file_path: Union[str, Path], data: bytes) -> str:
    """SHA256 of bytes ALREADY in memory, recorded under the file's cache key.

    For a caller that has read the file for its own reasons and would
    otherwise pay for a second read just to hash it. The result is stored
    under the same key `calculate_file_hash` uses, so a later call for this
    path -- `save_analysis` makes one -- is a dictionary lookup, not I/O.
    """
    path = Path(file_path)
    cache_key = _cache_key(path)
    file_hash = hashlib.sha256(data).hexdigest()
    _remember(cache_key, file_hash)
    return file_hash


def clear_hash_cache():
    """Clear the hash cache (useful after processing batches)."""
    # Cleared IN PLACE: rebinding the name would leave a worker that is midway
    # through _remember() writing into a dict nobody reads any more.
    with _hash_lock:
        _hash_cache.clear()
    logger.debug("Hash cache cleared")


def get_cache_size() -> int:
    """Get current hash cache size."""
    return len(_hash_cache)
