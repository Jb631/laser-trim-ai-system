"""
UI Responsiveness Optimization Module

This module provides tools and utilities for keeping the UI responsive during
long-running operations by managing background tasks, thread pools, and
asynchronous execution with proper callbacks.
"""

import asyncio
import functools
import logging
import queue
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
from collections import defaultdict, deque

import customtkinter as ctk

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TaskPriority(Enum):
    """Task priority levels for the background task queue."""
    CRITICAL = auto()  # UI updates, user interactions
    HIGH = auto()      # File processing, analysis
    NORMAL = auto()    # Background calculations
    LOW = auto()       # Cache updates, cleanup


class TaskState(Enum):
    """States for async tasks."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass
class AsyncTask:
    """Represents an asynchronous task."""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Optional[Exception] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def __lt__(self, other):
        """Compare tasks by priority and creation time."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


class UIStateManager:
    """Manages UI state during async operations."""
    
    def __init__(self):
        self._states: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._state_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def set_state(self, component_id: str, **state):
        """Set state for a UI component."""
        with self._lock:
            if component_id not in self._states:
                self._states[component_id] = {}
            self._states[component_id].update(state)
            self._notify_callbacks(component_id, state)
    
    def get_state(self, component_id: str, key: Optional[str] = None):
        """Get state for a UI component."""
        with self._lock:
            if component_id not in self._states:
                return None
            if key is None:
                return self._states[component_id].copy()
            return self._states[component_id].get(key)
    
    def clear_state(self, component_id: str):
        """Clear state for a UI component."""
        with self._lock:
            self._states.pop(component_id, None)
    
    def register_callback(self, component_id: str, callback: Callable):
        """Register a callback for state changes."""
        self._state_callbacks[component_id].append(callback)
    
    def unregister_callback(self, component_id: str, callback: Callable):
        """Unregister a callback."""
        if component_id in self._state_callbacks:
            self._state_callbacks[component_id].remove(callback)
    
    def _notify_callbacks(self, component_id: str, state: Dict[str, Any]):
        """Notify registered callbacks of state changes."""
        for callback in self._state_callbacks.get(component_id, []):
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")


class Debouncer:
    """Debounces function calls."""
    
    def __init__(self, delay: float = 0.3):
        self.delay = delay
        self._timers: Dict[str, Optional[threading.Timer]] = {}
        self._lock = threading.Lock()
    
    def debounce(self, key: str, func: Callable, *args, **kwargs):
        """Debounce a function call."""
        with self._lock:
            # Cancel existing timer
            if key in self._timers and self._timers[key]:
                self._timers[key].cancel()
            
            # Create new timer
            timer = threading.Timer(self.delay, func, args, kwargs)
            self._timers[key] = timer
            timer.start()
    
    def cancel(self, key: str):
        """Cancel a debounced call."""
        with self._lock:
            if key in self._timers and self._timers[key]:
                self._timers[key].cancel()
                self._timers[key] = None
    
    def cancel_all(self):
        """Cancel all debounced calls."""
        with self._lock:
            for timer in self._timers.values():
                if timer:
                    timer.cancel()
            self._timers.clear()


class Throttler:
    """Throttles function calls."""
    
    def __init__(self, rate_limit: float = 0.1):
        self.rate_limit = rate_limit
        self._last_calls: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def throttle(self, key: str, func: Callable, *args, **kwargs) -> bool:
        """
        Throttle a function call.
        Returns True if the call was executed, False if throttled.
        """
        with self._lock:
            now = time.time()
            last_call = self._last_calls.get(key, 0)
            
            if now - last_call >= self.rate_limit:
                self._last_calls[key] = now
                func(*args, **kwargs)
                return True
            return False
    
    def reset(self, key: str):
        """Reset throttle for a specific key."""
        with self._lock:
            self._last_calls.pop(key, None)
    
    def reset_all(self):
        """Reset all throttles."""
        with self._lock:
            self._last_calls.clear()


class AsyncHandler:
    """
    Main async handler for UI operations.
    Manages thread pools, task queues, and UI updates.
    """
    
    def __init__(self, max_workers: int = 4, max_queue_size: int = 100):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # Thread pool for background tasks
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task management
        self._task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self._active_tasks: Dict[str, AsyncTask] = {}
        self._completed_tasks: deque = deque(maxlen=100)
        self._task_lock = threading.Lock()
        
        # UI update queue
        self._ui_queue = queue.Queue()
        self._ui_callbacks: Set[weakref.ref] = set()
        
        # State management
        self.state_manager = UIStateManager()
        
        # Debouncing and throttling
        self.debouncer = Debouncer()
        self.throttler = Throttler()
        
        # Worker thread for processing tasks
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self._worker_thread.start()
        
        # UI update timer
        self._ui_update_id = None
        
        logger.info(f"AsyncHandler initialized with {max_workers} workers")
    
    def submit_task(self, 
                   func: Callable,
                   *args,
                   task_id: Optional[str] = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   callback: Optional[Callable] = None,
                   error_callback: Optional[Callable] = None,
                   **kwargs) -> str:
        """Submit a task for async execution."""
        if task_id is None:
            task_id = f"task_{id(func)}_{time.time()}"
        
        task = AsyncTask(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            callback=callback,
            error_callback=error_callback
        )
        
        try:
            self._task_queue.put_nowait((priority.value, task))
            logger.debug(f"Task {task_id} submitted with priority {priority.name}")
            return task_id
        except queue.Full:
            logger.error(f"Task queue full, rejecting task {task_id}")
            raise RuntimeError("Task queue is full")
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self._task_lock:
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                if task.state == TaskState.PENDING:
                    task.state = TaskState.CANCELLED
                    return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskState]:
        """Get the status of a task."""
        with self._task_lock:
            if task_id in self._active_tasks:
                return self._active_tasks[task_id].state
            
            for task in self._completed_tasks:
                if task.id == task_id:
                    return task.state
        return None
    
    def run_in_background(self, func: Callable, *args, **kwargs) -> Future:
        """Run a function in the background thread pool."""
        return self._executor.submit(func, *args, **kwargs)
    
    def run_in_ui_thread(self, func: Callable, *args, **kwargs):
        """Schedule a function to run in the UI thread."""
        self._ui_queue.put((func, args, kwargs))
    
    def register_ui_callback(self, root: ctk.CTk):
        """Register the UI update callback with the root window."""
        # Store weak reference to avoid circular references
        self._ui_callbacks.add(weakref.ref(root))
        self._schedule_ui_update(root)
    
    def _schedule_ui_update(self, root: ctk.CTk):
        """Schedule the next UI update."""
        if not self._running:
            return
        
        try:
            # Process UI updates
            self._process_ui_updates()
            
            # Schedule next update
            self._ui_update_id = root.after(16, lambda: self._schedule_ui_update(root))  # ~60 FPS
        except Exception as e:
            logger.error(f"Error in UI update: {e}")
    
    def _process_ui_updates(self):
        """Process pending UI updates."""
        batch_size = 10  # Process up to 10 updates per frame
        
        for _ in range(batch_size):
            try:
                func, args, kwargs = self._ui_queue.get_nowait()
                func(*args, **kwargs)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing UI update: {e}")
    
    def _process_tasks(self):
        """Worker thread for processing tasks."""
        while self._running:
            try:
                # Get next task with timeout
                _, task = self._task_queue.get(timeout=0.1)
                
                with self._task_lock:
                    if task.state == TaskState.CANCELLED:
                        continue
                    
                    task.state = TaskState.RUNNING
                    task.started_at = time.time()
                    self._active_tasks[task.id] = task
                
                # Execute task
                try:
                    result = task.func(*task.args, **task.kwargs)
                    task.result = result
                    task.state = TaskState.COMPLETED
                    
                    # Call success callback in UI thread
                    if task.callback:
                        self.run_in_ui_thread(task.callback, result)
                
                except Exception as e:
                    logger.error(f"Task {task.id} failed: {e}")
                    task.error = e
                    task.state = TaskState.FAILED
                    
                    # Call error callback in UI thread
                    if task.error_callback:
                        self.run_in_ui_thread(task.error_callback, e)
                
                finally:
                    task.completed_at = time.time()
                    
                    with self._task_lock:
                        self._active_tasks.pop(task.id, None)
                        self._completed_tasks.append(task)
                
                self._task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
    
    @contextmanager
    def ui_operation(self, component_id: str, loading_state: bool = True):
        """
        Context manager for UI operations.
        Automatically manages loading states and error handling.
        """
        if loading_state:
            self.state_manager.set_state(component_id, loading=True, error=None)
        
        try:
            yield
        except Exception as e:
            logger.error(f"Error in UI operation for {component_id}: {e}")
            self.state_manager.set_state(component_id, error=str(e))
            raise
        finally:
            if loading_state:
                self.state_manager.set_state(component_id, loading=False)
    
    def create_progress_callback(self, component_id: str) -> Callable:
        """Create a progress callback for long operations."""
        def update_progress(current: int, total: int, message: str = ""):
            progress = current / total if total > 0 else 0
            self.run_in_ui_thread(
                lambda: self.state_manager.set_state(
                    component_id,
                    progress=progress,
                    progress_text=f"{message} ({current}/{total})"
                )
            )
        return update_progress
    
    def batch_ui_updates(self, updates: List[Callable]):
        """Batch multiple UI updates together."""
        def batch_update():
            for update in updates:
                try:
                    update()
                except Exception as e:
                    logger.error(f"Error in batch update: {e}")
        
        self.run_in_ui_thread(batch_update)
    
    def shutdown(self):
        """Shutdown the async handler."""
        logger.info("Shutting down AsyncHandler")
        
        self._running = False
        
        # Cancel all pending tasks
        while not self._task_queue.empty():
            try:
                _, task = self._task_queue.get_nowait()
                task.state = TaskState.CANCELLED
            except queue.Empty:
                break
        
        # Wait for worker thread
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        
        # Shutdown executor
        self._executor.shutdown(wait=True, cancel_futures=True)
        
        # Clear UI callbacks
        for root_ref in self._ui_callbacks:
            root = root_ref()
            if root and self._ui_update_id:
                try:
                    root.after_cancel(self._ui_update_id)
                except Exception:
                    pass
        
        # Clear debouncer and throttler
        self.debouncer.cancel_all()
        self.throttler.reset_all()
        
        logger.info("AsyncHandler shutdown complete")


# Global instance
_async_handler: Optional[AsyncHandler] = None


def get_async_handler() -> AsyncHandler:
    """Get or create the global async handler instance."""
    global _async_handler
    if _async_handler is None:
        _async_handler = AsyncHandler()
    return _async_handler


def async_method(priority: TaskPriority = TaskPriority.NORMAL):
    """
    Decorator to make a method run asynchronously.
    The decorated method will return immediately and run in background.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            handler = get_async_handler()
            
            # Extract callbacks if provided
            callback = kwargs.pop('callback', None)
            error_callback = kwargs.pop('error_callback', None)
            
            # Submit task
            task_id = handler.submit_task(
                functools.partial(func, self),
                *args,
                priority=priority,
                callback=callback,
                error_callback=error_callback,
                **kwargs
            )
            
            return task_id
        
        return wrapper
    return decorator


def debounced(delay: float = 0.3, key: Optional[str] = None):
    """Decorator to debounce method calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            handler = get_async_handler()
            debounce_key = key or f"{self.__class__.__name__}.{func.__name__}"
            handler.debouncer.debounce(
                debounce_key,
                functools.partial(func, self),
                *args,
                **kwargs
            )
        return wrapper
    return decorator


def throttled(rate_limit: float = 0.1, key: Optional[str] = None):
    """Decorator to throttle method calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            handler = get_async_handler()
            throttle_key = key or f"{self.__class__.__name__}.{func.__name__}"
            return handler.throttler.throttle(
                throttle_key,
                functools.partial(func, self),
                *args,
                **kwargs
            )
        return wrapper
    return decorator


class AsyncWidget(ctk.CTkFrame):
    """Base class for widgets with async support."""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._async_handler = get_async_handler()
        self._component_id = f"{self.__class__.__name__}_{id(self)}"
        self._active_tasks: Set[str] = set()
        
        # Register state callback
        self._async_handler.state_manager.register_callback(
            self._component_id,
            self._on_state_change
        )
    
    def _on_state_change(self, state: Dict[str, Any]):
        """Handle state changes."""
        # Override in subclasses
        pass
    
    def run_async(self, 
                  func: Callable,
                  *args,
                  priority: TaskPriority = TaskPriority.NORMAL,
                  show_loading: bool = True,
                  **kwargs) -> str:
        """Run a function asynchronously."""
        if show_loading:
            self.set_loading(True)
        
        def on_complete(result):
            self._active_tasks.discard(task_id)
            if show_loading and not self._active_tasks:
                self.set_loading(False)
            self.on_async_complete(result)
        
        def on_error(error):
            self._active_tasks.discard(task_id)
            if show_loading and not self._active_tasks:
                self.set_loading(False)
            self.on_async_error(error)
        
        task_id = self._async_handler.submit_task(
            func,
            *args,
            priority=priority,
            callback=on_complete,
            error_callback=on_error,
            **kwargs
        )
        
        self._active_tasks.add(task_id)
        return task_id
    
    def set_loading(self, loading: bool):
        """Set loading state."""
        self._async_handler.state_manager.set_state(
            self._component_id,
            loading=loading
        )
    
    def set_error(self, error: Optional[str]):
        """Set error state."""
        self._async_handler.state_manager.set_state(
            self._component_id,
            error=error
        )
    
    def on_async_complete(self, result: Any):
        """Called when async operation completes. Override in subclasses."""
        pass
    
    def on_async_error(self, error: Exception):
        """Called when async operation fails. Override in subclasses."""
        logger.error(f"Async error in {self.__class__.__name__}: {error}")
    
    def destroy(self):
        """Clean up resources."""
        # Cancel active tasks
        for task_id in self._active_tasks:
            self._async_handler.cancel_task(task_id)
        
        # Unregister state callback
        self._async_handler.state_manager.unregister_callback(
            self._component_id,
            self._on_state_change
        )
        
        super().destroy()