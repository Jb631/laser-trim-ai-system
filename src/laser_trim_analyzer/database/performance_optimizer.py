"""
Database Performance Optimizer for Laser Trim Analyzer v2.

This module provides comprehensive tools for optimizing database query performance,
including query profiling, index management, connection pooling optimization,
query result caching, and prepared statement usage.

Production-ready implementation with focus on performance and reliability.
"""

import time
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import (
    Any, Dict, List, Optional, Tuple, Set, Union, Type, Callable
)
from functools import wraps, lru_cache
from collections import defaultdict, deque
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import weakref

from sqlalchemy import (
    create_engine, event, text, Index, inspect, select, func,
    and_, or_, Table, MetaData, Column, pool
)
from sqlalchemy.orm import Session, Query, sessionmaker
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.sql import ClauseElement
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.dialects import sqlite, postgresql, mysql

from ..core.exceptions import DatabaseError
from .models import (
    Base, AnalysisResult, TrackResult, MLPrediction, 
    QAAlert, BatchInfo, AnalysisBatch
)


class QueryProfile:
    """Stores profiling information for a single query."""
    
    def __init__(self, query_hash: str, query_text: str):
        self.query_hash = query_hash
        self.query_text = query_text
        self.execution_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.avg_time = 0.0
        self.last_executed = None
        self.row_counts = []
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
    def record_execution(self, execution_time: float, row_count: int = 0):
        """Record a query execution."""
        self.execution_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.execution_count
        self.last_executed = datetime.utcnow()
        self.row_counts.append(row_count)
        
    def record_error(self):
        """Record a query error."""
        self.error_count += 1
        self.last_executed = datetime.utcnow()
        
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
        
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
        
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_attempts = self.cache_hits + self.cache_misses
        if total_cache_attempts == 0:
            return 0.0
        return (self.cache_hits / total_cache_attempts) * 100
        
    @property
    def avg_row_count(self) -> float:
        """Calculate average row count."""
        if not self.row_counts:
            return 0.0
        return sum(self.row_counts) / len(self.row_counts)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'query_hash': self.query_hash,
            'query_text': self.query_text[:200] + '...' if len(self.query_text) > 200 else self.query_text,
            'execution_count': self.execution_count,
            'total_time': round(self.total_time, 4),
            'avg_time': round(self.avg_time, 4),
            'min_time': round(self.min_time, 4),
            'max_time': round(self.max_time, 4),
            'avg_row_count': round(self.avg_row_count, 2),
            'error_count': self.error_count,
            'cache_hit_rate': round(self.cache_hit_rate, 2),
            'last_executed': self.last_executed.isoformat() if self.last_executed else None
        }


class QueryCache:
    """Thread-safe query result cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum number of cached queries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_order = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if datetime.utcnow() < expiry:
                    # Move to end (most recently used)
                    self._access_order.remove(key)
                    self._access_order.append(key)
                    self._hit_count += 1
                    return value
                else:
                    # Expired, remove it
                    del self._cache[key]
                    self._access_order.remove(key)
            
            self._miss_count += 1
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with TTL."""
        if ttl is None:
            ttl = self.default_ttl
            
        with self._lock:
            expiry = datetime.utcnow() + timedelta(seconds=ttl)
            
            # Remove oldest if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = self._access_order.popleft()
                del self._cache[oldest_key]
                
            self._cache[key] = (value, expiry)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
    def invalidate(self, key: str):
        """Remove specific key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
                
    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern."""
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
                self._access_order.remove(key)
                
    def clear(self):
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return (self._hit_count / total) * 100
        
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
        
    def cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            now = datetime.utcnow()
            expired_keys = []
            
            for key, (_, expiry) in self._cache.items():
                if now >= expiry:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                del self._cache[key]
                self._access_order.remove(key)


class PreparedStatementManager:
    """Manages prepared statements for improved performance."""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self._statements: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
    def prepare(self, name: str, query: Union[str, ClauseElement]) -> Any:
        """Prepare a statement for reuse."""
        with self._lock:
            if name not in self._statements:
                try:
                    if isinstance(query, str):
                        stmt = text(query)
                    else:
                        stmt = query
                    
                    # Store the prepared statement
                    self._statements[name] = stmt
                    self.logger.debug(f"Prepared statement '{name}'")
                    
                except Exception as e:
                    self.logger.error(f"Failed to prepare statement '{name}': {e}")
                    raise
                    
            return self._statements[name]
            
    def execute(self, name: str, params: Optional[Dict[str, Any]] = None, 
                connection: Optional[Connection] = None) -> Any:
        """Execute a prepared statement."""
        if name not in self._statements:
            raise ValueError(f"Statement '{name}' not prepared")
            
        stmt = self._statements[name]
        
        try:
            if connection:
                return connection.execute(stmt, params or {})
            else:
                with self.engine.connect() as conn:
                    return conn.execute(stmt, params or {})
        except Exception as e:
            self.logger.error(f"Failed to execute prepared statement '{name}': {e}")
            raise
            
    def remove(self, name: str):
        """Remove a prepared statement."""
        with self._lock:
            if name in self._statements:
                del self._statements[name]
                
    def clear(self):
        """Clear all prepared statements."""
        with self._lock:
            self._statements.clear()


class IndexOptimizer:
    """Manages database indexes for optimal performance."""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        self._existing_indexes: Set[str] = set()
        self._suggested_indexes: List[Dict[str, Any]] = []
        self._refresh_existing_indexes()
        
    def _refresh_existing_indexes(self):
        """Refresh list of existing indexes."""
        try:
            inspector = inspect(self.engine)
            self._existing_indexes.clear()
            
            for table_name in inspector.get_table_names():
                indexes = inspector.get_indexes(table_name)
                for index in indexes:
                    self._existing_indexes.add(index['name'])
                    
        except Exception as e:
            self.logger.error(f"Failed to refresh indexes: {e}")
            
    def analyze_query_patterns(self, profiles: Dict[str, QueryProfile]) -> List[Dict[str, Any]]:
        """Analyze query patterns and suggest indexes."""
        suggestions = []
        
        # Analyze slow queries
        slow_queries = [p for p in profiles.values() if p.avg_time > 0.1]  # 100ms threshold
        
        for profile in slow_queries:
            # Simple heuristic: look for WHERE clauses
            query_lower = profile.query_text.lower()
            
            # Extract potential index candidates
            if 'where' in query_lower:
                # Look for common patterns
                patterns = [
                    (r'model\s*=', 'analysis_results', ['model']),
                    (r'serial\s*=', 'analysis_results', ['serial']),
                    (r'timestamp\s*>=', 'analysis_results', ['timestamp']),
                    (r'risk_category\s*=', 'track_results', ['risk_category']),
                    (r'sigma_gradient\s*>', 'track_results', ['sigma_gradient']),
                    (r'failure_probability\s*>', 'track_results', ['failure_probability']),
                ]
                
                import re
                for pattern, table, columns in patterns:
                    if re.search(pattern, query_lower):
                        index_name = f"idx_{table}_{'_'.join(columns)}_perf"
                        if index_name not in self._existing_indexes:
                            suggestions.append({
                                'table': table,
                                'columns': columns,
                                'name': index_name,
                                'reason': f"Slow query pattern detected (avg {profile.avg_time:.3f}s)",
                                'query_hash': profile.query_hash
                            })
                            
        self._suggested_indexes = suggestions
        return suggestions
        
    def create_index(self, table_name: str, columns: List[str], 
                    index_name: Optional[str] = None) -> bool:
        """Create an index on specified columns."""
        if not index_name:
            index_name = f"idx_{table_name}_{'_'.join(columns)}"
            
        try:
            # Get table object
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            table = metadata.tables.get(table_name)
            
            if not table:
                self.logger.error(f"Table '{table_name}' not found")
                return False
                
            # Create index
            index = Index(index_name, *[table.c[col] for col in columns])
            index.create(self.engine)
            
            self._existing_indexes.add(index_name)
            self.logger.info(f"Created index '{index_name}' on {table_name}({', '.join(columns)})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create index '{index_name}': {e}")
            return False
            
    def drop_index(self, index_name: str, table_name: str) -> bool:
        """Drop an index."""
        try:
            with self.engine.connect() as conn:
                # Handle different database dialects
                if self.engine.dialect.name == 'sqlite':
                    conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
                elif self.engine.dialect.name == 'postgresql':
                    conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
                elif self.engine.dialect.name == 'mysql':
                    conn.execute(text(f"DROP INDEX {index_name} ON {table_name}"))
                else:
                    self.logger.warning(f"Unsupported dialect for dropping index: {self.engine.dialect.name}")
                    return False
                    
                conn.commit()
                
            self._existing_indexes.discard(index_name)
            self.logger.info(f"Dropped index '{index_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to drop index '{index_name}': {e}")
            return False
            
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexes."""
        stats = {
            'total_indexes': len(self._existing_indexes),
            'suggested_indexes': len(self._suggested_indexes),
            'tables': {}
        }
        
        try:
            inspector = inspect(self.engine)
            
            for table_name in inspector.get_table_names():
                indexes = inspector.get_indexes(table_name)
                stats['tables'][table_name] = {
                    'index_count': len(indexes),
                    'indexes': [idx['name'] for idx in indexes]
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get index statistics: {e}")
            
        return stats


class ConnectionPoolOptimizer:
    """Optimizes database connection pool settings."""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        self._metrics = {
            'active_connections': 0,
            'idle_connections': 0,
            'overflow_connections': 0,
            'connection_wait_time': deque(maxlen=1000),
            'connection_errors': 0
        }
        self._lock = threading.Lock()
        self._setup_listeners()
        
    def _setup_listeners(self):
        """Setup event listeners for connection pool monitoring."""
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            with self._lock:
                self._metrics['active_connections'] += 1
                
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            connection_record.checkout_time = time.time()
            
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            if hasattr(connection_record, 'checkout_time'):
                wait_time = time.time() - connection_record.checkout_time
                with self._lock:
                    self._metrics['connection_wait_time'].append(wait_time)
                    self._metrics['active_connections'] = max(0, self._metrics['active_connections'] - 1)
                    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current connection pool status."""
        pool = self.engine.pool
        
        status = {
            'size': getattr(pool, 'size', 0),
            'checked_out': getattr(pool, 'checkedout', 0),
            'overflow': getattr(pool, 'overflow', 0),
            'total': getattr(pool, 'total', 0)
        }
        
        with self._lock:
            if self._metrics['connection_wait_time']:
                avg_wait = sum(self._metrics['connection_wait_time']) / len(self._metrics['connection_wait_time'])
                status['avg_wait_time'] = round(avg_wait, 4)
            else:
                status['avg_wait_time'] = 0.0
                
            status['connection_errors'] = self._metrics['connection_errors']
            
        return status
        
    def optimize_pool_size(self, target_concurrency: int = 10) -> Dict[str, Any]:
        """Suggest optimal pool size based on usage patterns."""
        current_status = self.get_pool_status()
        
        recommendations = {
            'current_size': current_status['size'],
            'current_overflow': current_status['overflow'],
            'recommendations': []
        }
        
        # Analyze usage patterns
        avg_checked_out = current_status['checked_out']
        avg_wait_time = current_status.get('avg_wait_time', 0)
        
        # Recommend pool size
        if avg_wait_time > 0.1:  # 100ms wait time
            recommended_size = min(target_concurrency * 2, 20)
            recommendations['recommendations'].append({
                'parameter': 'pool_size',
                'current': current_status['size'],
                'recommended': recommended_size,
                'reason': f"High average wait time ({avg_wait_time:.3f}s)"
            })
            
        # Recommend overflow
        if current_status['overflow'] > current_status['size']:
            recommended_overflow = current_status['size'] * 2
            recommendations['recommendations'].append({
                'parameter': 'max_overflow',
                'current': current_status['overflow'],
                'recommended': recommended_overflow,
                'reason': "Overflow exceeds pool size frequently"
            })
            
        return recommendations
        
    def recycle_connections(self, max_age: int = 3600):
        """Force recycling of old connections."""
        try:
            # Dispose of current pool and create new one
            self.engine.dispose()
            self.logger.info(f"Recycled all database connections (max age: {max_age}s)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to recycle connections: {e}")
            return False


class BatchQueryOptimizer:
    """Optimizes batch query operations."""
    
    def __init__(self, engine: Engine, batch_size: int = 1000):
        self.engine = engine
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
    def batch_insert(self, table: Type[Base], records: List[Dict[str, Any]], 
                    batch_size: Optional[int] = None) -> int:
        """Perform batch insert with optimal chunk size."""
        if not records:
            return 0
            
        batch_size = batch_size or self.batch_size
        total_inserted = 0
        
        try:
            Session = sessionmaker(bind=self.engine)
            
            for i in range(0, len(records), batch_size):
                chunk = records[i:i + batch_size]
                
                with Session() as session:
                    # Use bulk_insert_mappings for performance
                    session.bulk_insert_mappings(table, chunk)
                    session.commit()
                    total_inserted += len(chunk)
                    
                self.logger.debug(f"Inserted batch of {len(chunk)} records")
                
            self.logger.info(f"Batch inserted {total_inserted} records")
            return total_inserted
            
        except Exception as e:
            self.logger.error(f"Batch insert failed: {e}")
            raise
            
    def batch_update(self, table: Type[Base], updates: List[Dict[str, Any]], 
                    batch_size: Optional[int] = None) -> int:
        """Perform batch update with optimal chunk size."""
        if not updates:
            return 0
            
        batch_size = batch_size or self.batch_size
        total_updated = 0
        
        try:
            Session = sessionmaker(bind=self.engine)
            
            for i in range(0, len(updates), batch_size):
                chunk = updates[i:i + batch_size]
                
                with Session() as session:
                    # Use bulk_update_mappings for performance
                    session.bulk_update_mappings(table, chunk)
                    session.commit()
                    total_updated += len(chunk)
                    
                self.logger.debug(f"Updated batch of {len(chunk)} records")
                
            self.logger.info(f"Batch updated {total_updated} records")
            return total_updated
            
        except Exception as e:
            self.logger.error(f"Batch update failed: {e}")
            raise
            
    def batch_query(self, query_func: Callable, ids: List[Any], 
                   batch_size: Optional[int] = None) -> List[Any]:
        """Execute queries in batches to avoid memory issues."""
        if not ids:
            return []
            
        batch_size = batch_size or self.batch_size
        results = []
        
        try:
            for i in range(0, len(ids), batch_size):
                chunk_ids = ids[i:i + batch_size]
                chunk_results = query_func(chunk_ids)
                results.extend(chunk_results)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Batch query failed: {e}")
            raise


class PerformanceOptimizer:
    """Main performance optimization manager."""
    
    def __init__(self, engine: Engine, enable_profiling: bool = True,
                 enable_caching: bool = True, cache_size: int = 1000,
                 cache_ttl: int = 300):
        """
        Initialize performance optimizer.
        
        Args:
            engine: SQLAlchemy engine
            enable_profiling: Enable query profiling
            enable_caching: Enable query result caching
            cache_size: Maximum cache size
            cache_ttl: Default cache TTL in seconds
        """
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.profiling_enabled = enable_profiling
        self.caching_enabled = enable_caching
        
        # Query profiling
        self._profiles: Dict[str, QueryProfile] = {}
        self._profile_lock = threading.RLock()
        
        # Query cache
        self.cache = QueryCache(max_size=cache_size, default_ttl=cache_ttl)
        
        # Other optimizers
        self.prepared_statements = PreparedStatementManager(engine)
        self.index_optimizer = IndexOptimizer(engine)
        self.pool_optimizer = ConnectionPoolOptimizer(engine)
        self.batch_optimizer = BatchQueryOptimizer(engine)
        
        # Setup event listeners
        if enable_profiling:
            self._setup_profiling()
            
        # Start background cleanup
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
    def _setup_profiling(self):
        """Setup query profiling event listeners."""
        @event.listens_for(self.engine, "before_execute")
        def receive_before_execute(conn, clauseelement, multiparams, params, execution_options):
            conn.info['query_start_time'] = time.time()
            
        @event.listens_for(self.engine, "after_execute")
        def receive_after_execute(conn, clauseelement, multiparams, params, execution_options, result):
            if 'query_start_time' in conn.info:
                execution_time = time.time() - conn.info['query_start_time']
                query_text = str(clauseelement)
                self._record_query_execution(query_text, execution_time, result.rowcount)
                
    def _record_query_execution(self, query_text: str, execution_time: float, row_count: int):
        """Record query execution for profiling."""
        if not self.profiling_enabled:
            return
            
        # Generate query hash
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        
        with self._profile_lock:
            if query_hash not in self._profiles:
                self._profiles[query_hash] = QueryProfile(query_hash, query_text)
                
            self._profiles[query_hash].record_execution(execution_time, row_count)
            
    def _cleanup_loop(self):
        """Background cleanup thread."""
        while True:
            try:
                # Cleanup expired cache entries
                self.cache.cleanup_expired()
                
                # Sleep for 60 seconds
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                
    @contextmanager
    def cached_query(self, cache_key: str, ttl: Optional[int] = None):
        """Context manager for cached queries."""
        # Check cache first
        if self.caching_enabled:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                # Record cache hit in profiling
                query_hash = hashlib.md5(cache_key.encode()).hexdigest()
                with self._profile_lock:
                    if query_hash in self._profiles:
                        self._profiles[query_hash].record_cache_hit()
                yield cached_result
                return
                
        # Cache miss - execute query
        result = None
        try:
            yield None  # Caller will set result
        finally:
            # Cache the result if set
            if self.caching_enabled and result is not None:
                self.cache.set(cache_key, result, ttl)
                # Record cache miss
                query_hash = hashlib.md5(cache_key.encode()).hexdigest()
                with self._profile_lock:
                    if query_hash in self._profiles:
                        self._profiles[query_hash].record_cache_miss()
                        
    def get_slow_queries(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Get queries slower than threshold (in seconds)."""
        slow_queries = []
        
        with self._profile_lock:
            for profile in self._profiles.values():
                if profile.avg_time > threshold:
                    slow_queries.append(profile.to_dict())
                    
        # Sort by average time descending
        slow_queries.sort(key=lambda x: x['avg_time'], reverse=True)
        return slow_queries
        
    def get_most_frequent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently executed queries."""
        queries = []
        
        with self._profile_lock:
            for profile in self._profiles.values():
                queries.append(profile.to_dict())
                
        # Sort by execution count descending
        queries.sort(key=lambda x: x['execution_count'], reverse=True)
        return queries[:limit]
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'profiling': {
                'enabled': self.profiling_enabled,
                'total_queries': len(self._profiles),
                'slow_queries': len(self.get_slow_queries()),
                'top_queries': self.get_most_frequent_queries(5)
            },
            'caching': {
                'enabled': self.caching_enabled,
                'cache_size': self.cache.size,
                'hit_rate': round(self.cache.hit_rate, 2),
                'max_size': self.cache.max_size
            },
            'connection_pool': self.pool_optimizer.get_pool_status(),
            'indexes': self.index_optimizer.get_index_statistics()
        }
        
        # Add optimization suggestions
        report['suggestions'] = self._generate_suggestions()
        
        return report
        
    def _generate_suggestions(self) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on current metrics."""
        suggestions = []
        
        # Check slow queries
        slow_queries = self.get_slow_queries()
        if len(slow_queries) > 5:
            suggestions.append({
                'type': 'slow_queries',
                'severity': 'high',
                'message': f"Found {len(slow_queries)} slow queries",
                'action': 'Review and optimize these queries or add appropriate indexes'
            })
            
        # Check cache hit rate
        if self.cache.hit_rate < 50:
            suggestions.append({
                'type': 'cache_hit_rate',
                'severity': 'medium',
                'message': f"Low cache hit rate: {self.cache.hit_rate:.1f}%",
                'action': 'Consider increasing cache size or TTL for frequently accessed data'
            })
            
        # Check connection pool
        pool_status = self.pool_optimizer.get_pool_status()
        if pool_status.get('avg_wait_time', 0) > 0.1:
            suggestions.append({
                'type': 'connection_pool',
                'severity': 'high',
                'message': f"High connection wait time: {pool_status['avg_wait_time']:.3f}s",
                'action': 'Increase connection pool size'
            })
            
        # Check for missing indexes
        index_suggestions = self.index_optimizer.analyze_query_patterns(self._profiles)
        if index_suggestions:
            suggestions.append({
                'type': 'missing_indexes',
                'severity': 'medium',
                'message': f"Found {len(index_suggestions)} potential index optimizations",
                'action': 'Create suggested indexes to improve query performance',
                'details': index_suggestions
            })
            
        return suggestions
        
    def optimize(self) -> Dict[str, Any]:
        """Run automatic optimization based on current metrics."""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'optimizations': []
        }
        
        # 1. Create suggested indexes
        index_suggestions = self.index_optimizer.analyze_query_patterns(self._profiles)
        for suggestion in index_suggestions[:3]:  # Limit to 3 indexes at a time
            success = self.index_optimizer.create_index(
                suggestion['table'],
                suggestion['columns'],
                suggestion['name']
            )
            results['optimizations'].append({
                'type': 'index_creation',
                'success': success,
                'details': suggestion
            })
            
        # 2. Optimize connection pool
        pool_recommendations = self.pool_optimizer.optimize_pool_size()
        if pool_recommendations['recommendations']:
            results['optimizations'].append({
                'type': 'connection_pool',
                'recommendations': pool_recommendations['recommendations']
            })
            
        # 3. Clear cache if hit rate is too low
        if self.cache.hit_rate < 20:
            self.cache.clear()
            results['optimizations'].append({
                'type': 'cache_clear',
                'reason': 'Low hit rate',
                'previous_hit_rate': self.cache.hit_rate
            })
            
        return results
        
    def reset_profiling(self):
        """Reset all profiling data."""
        with self._profile_lock:
            self._profiles.clear()
        self.logger.info("Reset profiling data")
        
    def export_profiles(self, filepath: str):
        """Export profiling data to file."""
        try:
            profiles_data = []
            with self._profile_lock:
                for profile in self._profiles.values():
                    profiles_data.append(profile.to_dict())
                    
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': datetime.utcnow().isoformat(),
                    'profiles': profiles_data
                }, f, indent=2)
                
            self.logger.info(f"Exported {len(profiles_data)} profiles to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export profiles: {e}")
            raise


# Decorator for automatic query caching
def cached_query(cache_key_prefix: str, ttl: Optional[int] = None):
    """Decorator for caching query results."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key
            cache_key = f"{cache_key_prefix}:{str(args)}:{str(kwargs)}"
            
            # Get optimizer from self (assumes it's a method on a class with optimizer)
            if hasattr(self, 'performance_optimizer'):
                optimizer = self.performance_optimizer
                
                # Try cache first
                if optimizer.caching_enabled:
                    cached = optimizer.cache.get(cache_key)
                    if cached is not None:
                        return cached
                        
                # Execute query
                result = func(self, *args, **kwargs)
                
                # Cache result
                if optimizer.caching_enabled and result is not None:
                    optimizer.cache.set(cache_key, result, ttl)
                    
                return result
            else:
                # No optimizer, just execute normally
                return func(self, *args, **kwargs)
                
        return wrapper
    return decorator


# Example usage functions
def example_usage():
    """Example of how to use the performance optimizer."""
    from sqlalchemy import create_engine
    
    # Create engine
    engine = create_engine('sqlite:///example.db')
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer(
        engine,
        enable_profiling=True,
        enable_caching=True,
        cache_size=1000,
        cache_ttl=300
    )
    
    # Example 1: Using cached queries
    with optimizer.cached_query('model_stats:model123', ttl=600) as cached:
        if cached is None:
            # Execute query
            result = {'data': 'query result'}
            # Cache will be set automatically
        else:
            result = cached
            
    # Example 2: Batch operations
    records = [{'name': f'record{i}'} for i in range(10000)]
    optimizer.batch_optimizer.batch_insert(AnalysisResult, records, batch_size=500)
    
    # Example 3: Get performance report
    report = optimizer.get_performance_report()
    print(json.dumps(report, indent=2))
    
    # Example 4: Run automatic optimizations
    optimization_results = optimizer.optimize()
    print(json.dumps(optimization_results, indent=2))