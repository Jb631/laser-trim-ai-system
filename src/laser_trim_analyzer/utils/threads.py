"""
Thread management utilities for graceful shutdown.

Provides a centralized way to track background threads and ensure
they complete before application exit to prevent data corruption.
"""

import threading
import logging
from typing import Callable, Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)


class ThreadManager:
    """
    Manages background threads with graceful shutdown support.

    Usage:
        # Get the singleton instance
        manager = get_thread_manager()

        # Start a tracked thread
        manager.start_thread(target=my_function, name="my-task")

        # On app shutdown
        manager.shutdown(timeout=5.0)
    """

    _instance: Optional['ThreadManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'ThreadManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._threads: list[threading.Thread] = []
        self._threads_lock = threading.Lock()
        self._shutting_down = False

    def start_thread(
        self,
        target: Callable,
        name: Optional[str] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None
    ) -> Optional[threading.Thread]:
        """
        Start a tracked background thread.

        Args:
            target: Function to run in thread
            name: Optional thread name for debugging
            args: Positional arguments for target
            kwargs: Keyword arguments for target

        Returns:
            The started thread, or None if shutting down
        """
        if self._shutting_down:
            logger.warning(f"Cannot start thread '{name}' - shutdown in progress")
            return None

        kwargs = kwargs or {}

        def wrapper():
            try:
                target(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Thread '{name}' raised exception: {e}")
            finally:
                self._remove_thread(threading.current_thread())

        thread = threading.Thread(target=wrapper, name=name, daemon=False)

        with self._threads_lock:
            self._threads.append(thread)

        thread.start()
        return thread

    def _remove_thread(self, thread: threading.Thread):
        """Remove completed thread from tracking list."""
        with self._threads_lock:
            if thread in self._threads:
                self._threads.remove(thread)

    @property
    def active_count(self) -> int:
        """Number of active tracked threads."""
        with self._threads_lock:
            # Clean up dead threads
            self._threads = [t for t in self._threads if t.is_alive()]
            return len(self._threads)

    def shutdown(self, timeout: float = 5.0) -> bool:
        """
        Wait for all tracked threads to complete.

        Args:
            timeout: Maximum seconds to wait for each thread

        Returns:
            True if all threads completed, False if timeout occurred
        """
        self._shutting_down = True

        with self._threads_lock:
            threads_to_join = list(self._threads)

        if not threads_to_join:
            logger.debug("No active threads to wait for")
            return True

        logger.info(f"Waiting for {len(threads_to_join)} background threads to complete...")

        all_completed = True
        for thread in threads_to_join:
            if thread.is_alive():
                logger.debug(f"Waiting for thread '{thread.name}'...")
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"Thread '{thread.name}' did not complete within timeout")
                    all_completed = False
                else:
                    logger.debug(f"Thread '{thread.name}' completed")

        return all_completed

    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutting_down


# Singleton accessor
_manager: Optional[ThreadManager] = None


def get_thread_manager() -> ThreadManager:
    """Get the singleton ThreadManager instance."""
    global _manager
    if _manager is None:
        _manager = ThreadManager()
    return _manager


def background_thread(name: Optional[str] = None):
    """
    Decorator to run a function in a tracked background thread.

    Usage:
        @background_thread(name="data-loader")
        def load_data():
            # This runs in background
            pass

        load_data()  # Starts thread immediately
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_thread_manager()
            thread_name = name or func.__name__
            return manager.start_thread(
                target=func,
                name=thread_name,
                args=args,
                kwargs=kwargs
            )
        return wrapper
    return decorator
