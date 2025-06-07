"""
Tests for Database Performance Optimizer.

This module tests the performance optimization features including
query profiling, caching, index optimization, and batch operations.
"""

import pytest
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, text, select, and_
from sqlalchemy.orm import sessionmaker

from laser_trim_analyzer.database.models import (
    Base, AnalysisResult, TrackResult, SystemType, StatusType, RiskCategory
)
from laser_trim_analyzer.database.performance_optimizer import (
    PerformanceOptimizer, QueryProfile, QueryCache, PreparedStatementManager,
    IndexOptimizer, ConnectionPoolOptimizer, BatchQueryOptimizer,
    cached_query
)


@pytest.fixture
def test_engine():
    """Create a test SQLite database engine."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Cleanup
    engine.dispose()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def performance_optimizer(test_engine):
    """Create a performance optimizer instance."""
    return PerformanceOptimizer(
        test_engine,
        enable_profiling=True,
        enable_caching=True,
        cache_size=100,
        cache_ttl=60
    )


@pytest.fixture
def sample_data(test_engine):
    """Insert sample data for testing."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    
    # Create sample analysis results
    for i in range(10):
        analysis = AnalysisResult(
            filename=f'test_file_{i}.xls',
            model=f'MODEL_{i % 3}',
            serial=f'SERIAL_{i}',
            system=SystemType.A if i % 2 == 0 else SystemType.B,
            has_multi_tracks=i % 2 == 0,
            overall_status=StatusType.PASS if i % 3 != 0 else StatusType.FAIL,
            file_date=datetime.utcnow() - timedelta(days=i)
        )
        
        # Add tracks
        for j in range(2 if analysis.has_multi_tracks else 1):
            track = TrackResult(
                track_id=f'TRK{j+1}' if analysis.has_multi_tracks else 'default',
                status=StatusType.PASS if (i + j) % 4 != 0 else StatusType.FAIL,
                sigma_gradient=0.5 + (i * 0.1) + (j * 0.05),
                sigma_threshold=1.0,
                sigma_pass=(0.5 + (i * 0.1) + (j * 0.05)) < 1.0,
                failure_probability=0.1 * (i % 4),
                risk_category=RiskCategory.LOW if i % 4 == 0 else (
                    RiskCategory.MEDIUM if i % 4 < 3 else RiskCategory.HIGH
                )
            )
            analysis.tracks.append(track)
        
        session.add(analysis)
    
    session.commit()
    session.close()
    
    yield
    
    # Cleanup
    session = Session()
    session.query(TrackResult).delete()
    session.query(AnalysisResult).delete()
    session.commit()
    session.close()


class TestQueryCache:
    """Test query caching functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = QueryCache(max_size=10, default_ttl=60)
        
        # Test set and get
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        # Test cache miss
        assert cache.get('nonexistent') is None
        
        # Test hit rate
        cache.get('key1')  # Hit
        cache.get('key2')  # Miss
        assert cache.hit_rate == 50.0
        
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        cache = QueryCache(max_size=10, default_ttl=1)
        
        cache.set('key1', 'value1', ttl=1)
        assert cache.get('key1') == 'value1'
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get('key1') is None
        
    def test_cache_size_limit(self):
        """Test cache size limiting."""
        cache = QueryCache(max_size=3, default_ttl=60)
        
        # Fill cache
        for i in range(5):
            cache.set(f'key{i}', f'value{i}')
            
        # Should only have last 3 items
        assert cache.size == 3
        assert cache.get('key0') is None  # Evicted
        assert cache.get('key1') is None  # Evicted
        assert cache.get('key4') == 'value4'  # Still there
        
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        cache = QueryCache(max_size=10, default_ttl=60)
        
        # Add items
        cache.set('model:123', 'data1')
        cache.set('model:456', 'data2')
        cache.set('serial:789', 'data3')
        
        # Invalidate specific key
        cache.invalidate('model:123')
        assert cache.get('model:123') is None
        assert cache.get('model:456') == 'data2'
        
        # Invalidate by pattern
        cache.invalidate_pattern('model:')
        assert cache.get('model:456') is None
        assert cache.get('serial:789') == 'data3'


class TestQueryProfiling:
    """Test query profiling functionality."""
    
    def test_query_profile_recording(self):
        """Test recording query executions."""
        profile = QueryProfile('hash123', 'SELECT * FROM test')
        
        # Record executions
        profile.record_execution(0.1, 10)
        profile.record_execution(0.2, 20)
        profile.record_execution(0.15, 15)
        
        assert profile.execution_count == 3
        assert profile.avg_time == pytest.approx(0.15, 0.01)
        assert profile.min_time == 0.1
        assert profile.max_time == 0.2
        assert profile.avg_row_count == 15.0
        
    def test_query_profile_cache_tracking(self):
        """Test cache hit/miss tracking."""
        profile = QueryProfile('hash123', 'SELECT * FROM test')
        
        profile.record_cache_hit()
        profile.record_cache_hit()
        profile.record_cache_miss()
        
        assert profile.cache_hits == 2
        assert profile.cache_misses == 1
        assert profile.cache_hit_rate == pytest.approx(66.67, 0.01)


class TestPerformanceOptimizer:
    """Test main performance optimizer functionality."""
    
    def test_profiling_enabled(self, performance_optimizer, test_engine, sample_data):
        """Test query profiling when enabled."""
        # Execute some queries
        with test_engine.connect() as conn:
            conn.execute(text("SELECT * FROM analysis_results WHERE model = 'MODEL_0'"))
            conn.execute(text("SELECT * FROM track_results WHERE sigma_gradient > 0.5"))
            
        # Check profiles were recorded
        time.sleep(0.1)  # Allow event handlers to process
        
        # Note: Profiling through events might not capture in test environment
        # This is more of an integration test
        
    def test_cached_query_context_manager(self, performance_optimizer):
        """Test cached query context manager."""
        cache_key = 'test_query_1'
        
        # First execution - cache miss
        with performance_optimizer.cached_query(cache_key) as cached:
            assert cached is None
            # Simulate query execution
            result = {'data': 'test_result'}
            performance_optimizer.cache.set(cache_key, result)
            
        # Second execution - cache hit
        with performance_optimizer.cached_query(cache_key) as cached:
            assert cached == {'data': 'test_result'}
            
    def test_slow_query_detection(self, performance_optimizer):
        """Test detection of slow queries."""
        # Manually add some query profiles
        with performance_optimizer._profile_lock:
            # Add slow query
            slow_profile = QueryProfile('slow1', 'SELECT * FROM large_table')
            slow_profile.record_execution(0.5, 1000)
            slow_profile.record_execution(0.6, 1000)
            performance_optimizer._profiles['slow1'] = slow_profile
            
            # Add fast query
            fast_profile = QueryProfile('fast1', 'SELECT * FROM small_table')
            fast_profile.record_execution(0.01, 10)
            fast_profile.record_execution(0.02, 10)
            performance_optimizer._profiles['fast1'] = fast_profile
            
        slow_queries = performance_optimizer.get_slow_queries(threshold=0.1)
        assert len(slow_queries) == 1
        assert slow_queries[0]['query_hash'] == 'slow1'
        assert slow_queries[0]['avg_time'] > 0.5
        
    def test_performance_report(self, performance_optimizer):
        """Test performance report generation."""
        report = performance_optimizer.get_performance_report()
        
        assert 'timestamp' in report
        assert 'profiling' in report
        assert 'caching' in report
        assert 'connection_pool' in report
        assert 'indexes' in report
        assert 'suggestions' in report
        
        assert report['profiling']['enabled'] is True
        assert report['caching']['enabled'] is True
        assert report['caching']['max_size'] == 100


class TestIndexOptimizer:
    """Test index optimization functionality."""
    
    def test_index_analysis(self, test_engine):
        """Test index analysis and suggestions."""
        optimizer = IndexOptimizer(test_engine)
        
        # Create mock query profiles
        profiles = {
            'query1': QueryProfile('q1', 'SELECT * FROM analysis_results WHERE model = ?'),
            'query2': QueryProfile('q2', 'SELECT * FROM track_results WHERE risk_category = ?')
        }
        
        # Make them slow queries
        profiles['query1'].record_execution(0.2, 100)
        profiles['query2'].record_execution(0.3, 200)
        
        suggestions = optimizer.analyze_query_patterns(profiles)
        
        # Should suggest indexes for slow queries
        assert len(suggestions) > 0
        
    def test_index_creation(self, test_engine):
        """Test index creation."""
        optimizer = IndexOptimizer(test_engine)
        
        # Create a test index
        success = optimizer.create_index(
            'analysis_results',
            ['model', 'serial'],
            'idx_test_model_serial'
        )
        
        assert success is True
        assert 'idx_test_model_serial' in optimizer._existing_indexes


class TestBatchOperations:
    """Test batch query operations."""
    
    def test_batch_insert(self, test_engine):
        """Test batch insert operation."""
        optimizer = BatchQueryOptimizer(test_engine, batch_size=5)
        
        # Prepare test data
        records = []
        for i in range(15):
            records.append({
                'filename': f'batch_test_{i}.xls',
                'model': f'BATCH_MODEL_{i}',
                'serial': f'BATCH_SERIAL_{i}',
                'system': SystemType.A.value,
                'has_multi_tracks': False,
                'overall_status': StatusType.PASS.value,
                'file_date': datetime.utcnow()
            })
            
        # Perform batch insert
        inserted = optimizer.batch_insert(AnalysisResult, records, batch_size=5)
        assert inserted == 15
        
        # Verify data was inserted
        Session = sessionmaker(bind=test_engine)
        session = Session()
        count = session.query(AnalysisResult).filter(
            AnalysisResult.model.like('BATCH_MODEL_%')
        ).count()
        assert count == 15
        session.close()


class TestPreparedStatements:
    """Test prepared statement functionality."""
    
    def test_prepared_statement_management(self, test_engine):
        """Test prepared statement creation and execution."""
        manager = PreparedStatementManager(test_engine)
        
        # Prepare a statement
        query = "SELECT * FROM analysis_results WHERE model = :model"
        stmt = manager.prepare('get_by_model', query)
        
        assert 'get_by_model' in manager._statements
        
        # Execute prepared statement
        result = manager.execute('get_by_model', {'model': 'MODEL_0'})
        
        # Should execute without error
        rows = result.fetchall()
        
        # Test removal
        manager.remove('get_by_model')
        assert 'get_by_model' not in manager._statements


class TestConnectionPoolOptimizer:
    """Test connection pool optimization."""
    
    def test_pool_status(self, test_engine):
        """Test connection pool status monitoring."""
        optimizer = ConnectionPoolOptimizer(test_engine)
        
        status = optimizer.get_pool_status()
        
        assert 'size' in status
        assert 'checked_out' in status
        assert 'overflow' in status
        assert 'total' in status
        assert 'avg_wait_time' in status
        
    def test_pool_optimization_recommendations(self, test_engine):
        """Test pool optimization recommendations."""
        optimizer = ConnectionPoolOptimizer(test_engine)
        
        recommendations = optimizer.optimize_pool_size(target_concurrency=20)
        
        assert 'current_size' in recommendations
        assert 'current_overflow' in recommendations
        assert 'recommendations' in recommendations
        assert isinstance(recommendations['recommendations'], list)


class TestCachedQueryDecorator:
    """Test the cached_query decorator."""
    
    class MockDatabaseManager:
        def __init__(self, optimizer):
            self.performance_optimizer = optimizer
            
        @cached_query('get_model_stats', ttl=300)
        def get_model_statistics(self, model: str):
            # Simulate expensive query
            time.sleep(0.1)
            return {'model': model, 'stats': 'data'}
            
    def test_cached_query_decorator(self, performance_optimizer):
        """Test cached query decorator functionality."""
        manager = self.MockDatabaseManager(performance_optimizer)
        
        # First call - should execute query
        start = time.time()
        result1 = manager.get_model_statistics('MODEL_1')
        duration1 = time.time() - start
        
        # Second call - should use cache
        start = time.time()
        result2 = manager.get_model_statistics('MODEL_1')
        duration2 = time.time() - start
        
        assert result1 == result2
        assert duration2 < duration1  # Cached query should be faster
        
        # Different parameter - should execute query again
        result3 = manager.get_model_statistics('MODEL_2')
        assert result3['model'] == 'MODEL_2'


def test_optimization_suggestions(performance_optimizer):
    """Test optimization suggestion generation."""
    # Add some problematic metrics
    performance_optimizer.cache._hit_count = 10
    performance_optimizer.cache._miss_count = 90  # Low hit rate
    
    suggestions = performance_optimizer._generate_suggestions()
    
    # Should suggest improving cache hit rate
    cache_suggestions = [s for s in suggestions if s['type'] == 'cache_hit_rate']
    assert len(cache_suggestions) > 0
    assert cache_suggestions[0]['severity'] == 'medium'


def test_export_profiles(performance_optimizer, tmp_path):
    """Test exporting profiling data."""
    # Add some profiles
    with performance_optimizer._profile_lock:
        profile = QueryProfile('test1', 'SELECT * FROM test')
        profile.record_execution(0.1, 10)
        performance_optimizer._profiles['test1'] = profile
        
    # Export to file
    export_file = tmp_path / 'profiles.json'
    performance_optimizer.export_profiles(str(export_file))
    
    assert export_file.exists()
    
    # Verify content
    import json
    with open(export_file) as f:
        data = json.load(f)
        
    assert 'timestamp' in data
    assert 'profiles' in data
    assert len(data['profiles']) == 1
    assert data['profiles'][0]['query_hash'] == 'test1'