"""
Unit tests for the cache manager system.
"""

import time
import tempfile
import shutil
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.laser_trim_analyzer.core.cache_manager import (
    CacheManager, MemoryCache, FileCache, CacheStrategy,
    get_cache_manager, initialize_cache_manager
)
from src.laser_trim_analyzer.core.exceptions import CacheError


class TestMemoryCache:
    """Test in-memory cache backend."""
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = MemoryCache(max_size=1024*1024, max_entries=10)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test non-existent key
        assert cache.get("nonexistent") is None
        
        # Test exists
        assert cache.exists("key1")
        assert not cache.exists("nonexistent")
        
        # Test delete
        assert cache.delete("key1")
        assert not cache.exists("key1")
        assert not cache.delete("key1")  # Already deleted
        
        # Test clear
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.clear()
        assert not cache.exists("key2")
        assert not cache.exists("key3")
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = MemoryCache(default_ttl=0.1)  # 100ms TTL
        
        cache.set("expiring", "value")
        assert cache.get("expiring") == "value"
        
        time.sleep(0.2)  # Wait for expiration
        assert cache.get("expiring") is None
        assert not cache.exists("expiring")
    
    def test_lru_eviction(self):
        """Test LRU eviction strategy."""
        cache = MemoryCache(max_entries=3, strategy=CacheStrategy.LRU)
        
        # Fill cache
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        
        # Access 'a' to make it recently used
        cache.get("a")
        
        # Add new item, should evict 'b' (least recently used)
        cache.set("d", "4")
        
        assert cache.exists("a")
        assert not cache.exists("b")  # Evicted
        assert cache.exists("c")
        assert cache.exists("d")
    
    def test_lfu_eviction(self):
        """Test LFU eviction strategy."""
        cache = MemoryCache(max_entries=3, strategy=CacheStrategy.LFU)
        
        # Fill cache with different access counts
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        
        # Access items different number of times
        cache.get("a")  # 1 access
        cache.get("b")  # 1 access
        cache.get("b")  # 2 accesses
        cache.get("c")  # 1 access
        cache.get("c")  # 2 accesses
        cache.get("c")  # 3 accesses
        
        # Add new item, should evict 'a' (least frequently used)
        cache.set("d", "4")
        
        assert not cache.exists("a")  # Evicted (1 access)
        assert cache.exists("b")  # 2 accesses
        assert cache.exists("c")  # 3 accesses
        assert cache.exists("d")
    
    def test_size_based_eviction(self):
        """Test size-based eviction."""
        cache = MemoryCache(max_size=100, strategy=CacheStrategy.SIZE)
        
        # Add items of different sizes
        cache.set("small", "x")
        cache.set("medium", "x" * 10)
        cache.set("large", "x" * 50)
        
        # Add item that requires eviction
        cache.set("new", "x" * 40)
        
        # Large item should be evicted first
        assert cache.exists("small")
        assert cache.exists("medium")
        assert not cache.exists("large")  # Evicted due to size
        assert cache.exists("new")


class TestFileCache:
    """Test file-based cache backend."""
    
    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_basic_operations(self):
        """Test basic file cache operations."""
        cache = FileCache(self.cache_dir)
        
        # Test set and get
        cache.set("key1", {"data": "value1"})
        assert cache.get("key1") == {"data": "value1"}
        
        # Test persistence (create new instance)
        cache2 = FileCache(self.cache_dir)
        assert cache2.get("key1") == {"data": "value1"}
        
        # Test delete
        assert cache.delete("key1")
        assert not cache.exists("key1")
        
        # Test clear
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.clear()
        assert not cache.exists("key2")
        assert not cache.exists("key3")
    
    def test_complex_objects(self):
        """Test caching complex objects."""
        cache = FileCache(self.cache_dir)
        
        # Test numpy array
        arr = np.array([1, 2, 3, 4, 5])
        cache.set("numpy", arr)
        retrieved = cache.get("numpy")
        assert np.array_equal(retrieved, arr)
        
        # Test pandas DataFrame
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cache.set("pandas", df)
        retrieved = cache.get("pandas")
        pd.testing.assert_frame_equal(retrieved, df)
    
    def test_size_limit(self):
        """Test file cache size limits."""
        cache = FileCache(self.cache_dir, max_size=1000)  # 1KB limit
        
        # Add items until size limit
        for i in range(10):
            cache.set(f"key{i}", "x" * 100)  # 100 bytes each
        
        # Check that old items were evicted
        total_size = cache.size()
        assert total_size <= 1000


class TestCacheManager:
    """Test the unified cache manager."""
    
    def setup_method(self):
        """Create temporary directory and initialize cache manager."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.manager = CacheManager(
            memory_cache_size=1024*1024,
            file_cache_size=1024*1024,
            cache_dir=self.cache_dir,
            monitor_memory=False  # Disable for tests
        )
    
    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
    
    def test_different_cache_types(self):
        """Test using different cache types."""
        # General cache
        self.manager.set("key1", "value1", cache_type='general')
        assert self.manager.get("key1", cache_type='general') == "value1"
        
        # File content cache
        self.manager.set("file1", b"content", cache_type='file_content')
        assert self.manager.get("file1", cache_type='file_content') == b"content"
        
        # Model predictions cache
        self.manager.set("pred1", [0.1, 0.9], cache_type='model_predictions')
        assert self.manager.get("pred1", cache_type='model_predictions') == [0.1, 0.9]
        
        # Config cache
        self.manager.set("config1", {"key": "value"}, cache_type='config')
        assert self.manager.get("config1", cache_type='config') == {"key": "value"}
        
        # Analysis results cache (file-based)
        self.manager.set("analysis1", {"result": 42}, cache_type='analysis_results')
        assert self.manager.get("analysis1", cache_type='analysis_results') == {"result": 42}
    
    def test_file_content_caching(self):
        """Test file content caching methods."""
        file_path = Path("/tmp/test_file.txt")
        content = {"data": [1, 2, 3]}
        
        self.manager.cache_file_content(file_path, content)
        retrieved = self.manager.get_file_content(file_path)
        assert retrieved == content
    
    def test_model_prediction_caching(self):
        """Test model prediction caching methods."""
        model_name = "test_model"
        input_hash = "abc123"
        prediction = np.array([0.1, 0.2, 0.7])
        
        self.manager.cache_model_prediction(model_name, input_hash, prediction)
        retrieved = self.manager.get_model_prediction(model_name, input_hash)
        assert np.array_equal(retrieved, prediction)
    
    def test_analysis_result_caching(self):
        """Test analysis result caching methods."""
        analysis_id = "analysis_123"
        result = {
            "sigma": 0.05,
            "linearity": 0.98,
            "tracks": [{"id": 1, "value": 100}]
        }
        
        self.manager.cache_analysis_result(analysis_id, result)
        retrieved = self.manager.get_analysis_result(analysis_id)
        assert retrieved == result
    
    def test_pattern_invalidation(self):
        """Test pattern-based cache invalidation."""
        # Add multiple entries
        self.manager.set("test:1", "value1")
        self.manager.set("test:2", "value2")
        self.manager.set("other:1", "value3")
        
        # Invalidate pattern
        count = self.manager.invalidate_pattern("test:*")
        assert count == 2
        
        # Check results
        assert not self.manager.exists("test:1")
        assert not self.manager.exists("test:2")
        assert self.manager.exists("other:1")
    
    def test_cache_decorator(self):
        """Test cache decorator functionality."""
        call_count = 0
        
        @self.manager.decorator(cache_type='general', ttl=1.0)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call - should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call - should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Not incremented
        
        # Different arguments - should execute function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
        
        # Wait for TTL expiration
        time.sleep(1.1)
        result4 = expensive_function(1, 2)
        assert result4 == 3
        assert call_count == 3  # Executed again after expiration
    
    def test_get_stats(self):
        """Test cache statistics."""
        # Add some data
        self.manager.set("key1", "value1", cache_type='general')
        self.manager.set("key2", "value2", cache_type='general')
        self.manager.set("file1", b"content", cache_type='file_content')
        
        stats = self.manager.get_stats()
        
        # Check structure
        assert 'general' in stats
        assert 'file_content' in stats
        assert 'analysis_results' in stats
        
        # Check general cache stats
        general_stats = stats['general']
        assert general_stats['entries'] == 2
        assert general_stats['type'] == 'MemoryCache'
        assert 'size_bytes' in general_stats
        assert 'strategy' in general_stats
    
    def test_clear_operations(self):
        """Test clearing caches."""
        # Add data to different caches
        self.manager.set("key1", "value1", cache_type='general')
        self.manager.set("key2", "value2", cache_type='file_content')
        
        # Clear specific cache
        self.manager.clear('general')
        assert not self.manager.exists("key1", cache_type='general')
        assert self.manager.exists("key2", cache_type='file_content')
        
        # Clear all caches
        self.manager.clear()
        assert not self.manager.exists("key2", cache_type='file_content')


class TestGlobalCacheManager:
    """Test global cache manager functions."""
    
    def test_singleton_behavior(self):
        """Test that get_cache_manager returns singleton."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        assert manager1 is manager2
    
    def test_initialization(self):
        """Test custom initialization."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = initialize_cache_manager(
                memory_cache_size=2048,
                cache_dir=Path(temp_dir) / "custom_cache"
            )
            
            # Test it works
            manager.set("test", "value")
            assert manager.get("test") == "value"
            
            # Test it's the same as get_cache_manager
            assert get_cache_manager() is manager
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])