"""
Session Management for Laser Trim Analyzer

Provides secure session handling with:
- Session creation and validation
- Timeout management
- Activity tracking
- Secure session storage
- Multi-user support (future)
"""

import uuid
import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
import secrets

# Import security and error handling
from laser_trim_analyzer.core.security import (
    SecurityValidator, InputSanitizer, get_security_validator
)
from laser_trim_analyzer.core.error_handlers import (
    ErrorCode, ErrorCategory, ErrorSeverity,
    error_handler, handle_errors
)

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session status states."""
    ACTIVE = "active"
    EXPIRED = "expired"
    INVALID = "invalid"
    LOCKED = "locked"


@dataclass
class SessionData:
    """Container for session data."""
    session_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    status: SessionStatus
    user_data: Dict[str, Any]
    activity_count: int = 0
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now() > self.expires_at
    
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == SessionStatus.ACTIVE and not self.is_expired()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_activity'] = self.last_activity.isoformat()
        data['expires_at'] = self.expires_at.isoformat()
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        data['status'] = SessionStatus(data['status'])
        return cls(**data)


class SessionManager:
    """
    Manages user sessions for the application.
    
    Features:
    - Secure session token generation
    - Automatic session expiration
    - Activity tracking
    - Session persistence
    - Concurrent access handling
    """
    
    def __init__(
        self,
        session_timeout_minutes: int = 30,
        max_sessions: int = 100,
        storage_path: Optional[Path] = None,
        cleanup_interval_seconds: int = 300  # 5 minutes
    ):
        """Initialize session manager."""
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_sessions = max_sessions
        self.storage_path = storage_path or Path.home() / ".laser_trim_analyzer" / "sessions"
        self.cleanup_interval = cleanup_interval_seconds
        
        # Session storage
        self._sessions: Dict[str, SessionData] = {}
        self._session_lock = threading.RLock()
        
        # Security validator
        self._security = get_security_validator()
        
        # Initialize storage
        self._init_storage()
        
        # Load existing sessions
        self._load_sessions()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_expired_sessions,
            daemon=True
        )
        self._cleanup_thread.start()
        
        logger.info(f"Session manager initialized: timeout={session_timeout_minutes}min, "
                   f"max_sessions={max_sessions}")
    
    def _init_storage(self):
        """Initialize session storage directory."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Set secure permissions on Unix-like systems
            if hasattr(os, 'chmod'):
                import stat
                os.chmod(self.storage_path, stat.S_IRWXU)  # 700 permissions
        except Exception as e:
            logger.error(f"Failed to initialize session storage: {e}")
    
    def _load_sessions(self):
        """Load existing sessions from storage."""
        session_file = self.storage_path / "sessions.json"
        
        if not session_file.exists():
            return
        
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
                
            for session_id, session_data in data.items():
                try:
                    session = SessionData.from_dict(session_data)
                    
                    # Only load non-expired sessions
                    if not session.is_expired():
                        self._sessions[session_id] = session
                except Exception as e:
                    logger.warning(f"Failed to load session {session_id}: {e}")
                    
            logger.info(f"Loaded {len(self._sessions)} active sessions")
            
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
    
    def _save_sessions(self):
        """Save sessions to storage."""
        session_file = self.storage_path / "sessions.json"
        
        try:
            with self._session_lock:
                data = {
                    sid: session.to_dict()
                    for sid, session in self._sessions.items()
                    if session.is_active()
                }
            
            # Write to temporary file first
            temp_file = session_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            temp_file.replace(session_file)
            
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
    
    @handle_errors(
        category=ErrorCategory.VALIDATION,
        severity=ErrorSeverity.WARNING
    )
    def create_session(
        self,
        user_data: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a new session.
        
        Args:
            user_data: Optional user data to store
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session ID if created, None otherwise
        """
        # Validate inputs
        if ip_address:
            result = self._security.validate_input(ip_address, 'string', {'max_length': 45})
            if not result.is_safe:
                logger.warning(f"Invalid IP address: {result.validation_errors}")
                ip_address = None
        
        if user_agent:
            result = self._security.validate_input(user_agent, 'string', {'max_length': 500})
            if not result.is_safe:
                user_agent = result.sanitized_value
        
        with self._session_lock:
            # Check session limit
            if len(self._sessions) >= self.max_sessions:
                # Remove oldest expired session
                self._cleanup_expired_sessions_sync()
                
                if len(self._sessions) >= self.max_sessions:
                    logger.warning("Maximum session limit reached")
                    return None
            
            # Generate secure session ID
            session_id = self._generate_session_id()
            
            # Create session
            now = datetime.now()
            session = SessionData(
                session_id=session_id,
                created_at=now,
                last_activity=now,
                expires_at=now + self.session_timeout,
                status=SessionStatus.ACTIVE,
                user_data=user_data or {},
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self._sessions[session_id] = session
            self._save_sessions()
            
            logger.info(f"Created session: {session_id[:8]}...")
            return session_id
    
    def _generate_session_id(self) -> str:
        """Generate cryptographically secure session ID."""
        # Use secrets for secure random generation
        random_bytes = secrets.token_bytes(32)
        
        # Add timestamp for uniqueness
        timestamp = str(time.time()).encode()
        
        # Create session ID
        session_data = random_bytes + timestamp
        session_id = hashlib.sha256(session_data).hexdigest()
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """
        Validate if a session is active.
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            True if session is valid and active
        """
        # Validate session ID format
        if not session_id or not isinstance(session_id, str):
            return False
        
        result = self._security.validate_input(
            session_id,
            'string',
            {'max_length': 64, 'alphanumeric_only': True}
        )
        
        if not result.is_safe:
            return False
        
        with self._session_lock:
            session = self._sessions.get(session_id)
            
            if not session:
                return False
            
            if session.is_expired():
                session.status = SessionStatus.EXPIRED
                return False
            
            if session.status != SessionStatus.ACTIVE:
                return False
            
            # Update last activity
            session.last_activity = datetime.now()
            session.activity_count += 1
            
            # Extend expiration on activity
            session.expires_at = session.last_activity + self.session_timeout
            
            return True
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get data for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session user data if valid, None otherwise
        """
        if not self.validate_session(session_id):
            return None
        
        with self._session_lock:
            session = self._sessions.get(session_id)
            return session.user_data if session else None
    
    def update_session_data(
        self,
        session_id: str,
        data: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session ID
            data: Data to store
            merge: Whether to merge with existing data
            
        Returns:
            True if updated successfully
        """
        if not self.validate_session(session_id):
            return False
        
        with self._session_lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            if merge:
                session.user_data.update(data)
            else:
                session.user_data = data
            
            self._save_sessions()
            return True
    
    def end_session(self, session_id: str) -> bool:
        """
        End a session.
        
        Args:
            session_id: Session ID to end
            
        Returns:
            True if session was ended
        """
        with self._session_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._save_sessions()
                logger.info(f"Ended session: {session_id[:8]}...")
                return True
            return False
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions."""
        with self._session_lock:
            active_sessions = []
            
            for session in self._sessions.values():
                if session.is_active():
                    active_sessions.append({
                        'session_id': session.session_id[:8] + '...',
                        'created_at': session.created_at.isoformat(),
                        'last_activity': session.last_activity.isoformat(),
                        'activity_count': session.activity_count,
                        'expires_in': (session.expires_at - datetime.now()).total_seconds()
                    })
            
            return active_sessions
    
    def _cleanup_expired_sessions_sync(self):
        """Synchronously clean up expired sessions."""
        expired = []
        
        for session_id, session in self._sessions.items():
            if session.is_expired():
                expired.append(session_id)
        
        for session_id in expired:
            del self._sessions[session_id]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
            self._save_sessions()
    
    def _cleanup_expired_sessions(self):
        """Background thread to clean up expired sessions."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                
                with self._session_lock:
                    self._cleanup_expired_sessions_sync()
                    
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
    
    def lock_session(self, session_id: str) -> bool:
        """Lock a session (e.g., for security reasons)."""
        with self._session_lock:
            session = self._sessions.get(session_id)
            if session:
                session.status = SessionStatus.LOCKED
                self._save_sessions()
                logger.warning(f"Locked session: {session_id[:8]}...")
                return True
            return False
    
    def unlock_session(self, session_id: str) -> bool:
        """Unlock a locked session."""
        with self._session_lock:
            session = self._sessions.get(session_id)
            if session and session.status == SessionStatus.LOCKED:
                session.status = SessionStatus.ACTIVE
                self._save_sessions()
                logger.info(f"Unlocked session: {session_id[:8]}...")
                return True
            return False
    
    def clear_all_sessions(self):
        """Clear all sessions (admin function)."""
        with self._session_lock:
            self._sessions.clear()
            self._save_sessions()
            logger.warning("All sessions cleared")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        with self._session_lock:
            active_count = sum(1 for s in self._sessions.values() if s.is_active())
            expired_count = sum(1 for s in self._sessions.values() if s.is_expired())
            locked_count = sum(1 for s in self._sessions.values() if s.status == SessionStatus.LOCKED)
            
            return {
                'total_sessions': len(self._sessions),
                'active_sessions': active_count,
                'expired_sessions': expired_count,
                'locked_sessions': locked_count,
                'max_sessions': self.max_sessions,
                'timeout_minutes': self.session_timeout.total_seconds() / 60
            }


# Global session manager instance
_session_manager: Optional[SessionManager] = None
_manager_lock = threading.Lock()


def get_session_manager(
    session_timeout_minutes: int = 30,
    max_sessions: int = 100,
    storage_path: Optional[Path] = None
) -> SessionManager:
    """Get or create the global session manager instance."""
    global _session_manager
    
    with _manager_lock:
        if _session_manager is None:
            _session_manager = SessionManager(
                session_timeout_minutes=session_timeout_minutes,
                max_sessions=max_sessions,
                storage_path=storage_path
            )
        
        return _session_manager


def cleanup_session_manager():
    """Cleanup the global session manager."""
    global _session_manager
    
    with _manager_lock:
        if _session_manager:
            _session_manager.clear_all_sessions()
            _session_manager = None


# Decorator for session-protected functions
def require_session(func):
    """Decorator to require valid session for function access."""
    def wrapper(*args, **kwargs):
        # Get session_id from kwargs or first positional arg
        session_id = kwargs.get('session_id')
        if not session_id and args:
            session_id = args[0]
        
        if not session_id:
            raise ValueError("Session ID required")
        
        manager = get_session_manager()
        if not manager.validate_session(session_id):
            raise PermissionError("Invalid or expired session")
        
        return func(*args, **kwargs)
    
    return wrapper


# Export main components
__all__ = [
    'SessionManager',
    'SessionData',
    'SessionStatus',
    'get_session_manager',
    'cleanup_session_manager',
    'require_session'
]