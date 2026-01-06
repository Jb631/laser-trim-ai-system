"""
Utility modules for v3.

Modules:
- constants: Shared constants
- validators: Input validation
- threads: Thread management for graceful shutdown
- hashing: File hashing utilities
"""

from laser_trim_analyzer.utils.threads import (
    ThreadManager,
    get_thread_manager,
    background_thread,
)
from laser_trim_analyzer.utils.hashing import (
    calculate_file_hash,
    clear_hash_cache,
)

__all__ = [
    'ThreadManager',
    'get_thread_manager',
    'background_thread',
    'calculate_file_hash',
    'clear_hash_cache',
]
