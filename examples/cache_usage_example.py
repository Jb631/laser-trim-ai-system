"""
Example usage of the cache system in the laser trim analyzer.

This script demonstrates various caching features and best practices.
"""

import asyncio
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.laser_trim_analyzer.core.cache_manager import get_cache_manager, initialize_cache_manager
from src.laser_trim_analyzer.core.cache_config import (
    CacheConfig, CachePreset, setup_cache_from_preset
)
from src.laser_trim_analyzer.core.cached_processor import (
    CachedFileProcessor, CachedBatchProcessor
)
from src.laser_trim_analyzer.core.config import Config


def example_basic_caching():
    """Demonstrate basic cache operations."""
    print("=== Basic Cache Operations ===")
    
    # Initialize cache manager
    cache = get_cache_manager()
    
    # Store and retrieve simple data
    cache.set("user_preference", {"theme": "dark", "language": "en"})
    pref = cache.get("user_preference")
    print(f"Retrieved preferences: {pref}")
    
    # Cache with TTL
    cache.set("temporary_data", "This expires in 5 seconds", ttl=5)
    print(f"Temporary data: {cache.get('temporary_data')}")
    
    time.sleep(6)
    print(f"After 6 seconds: {cache.get('temporary_data')}")  # None
    
    # Different cache types
    cache.set("config_item", {"api_key": "secret"}, cache_type='config')
    cache.set("analysis_123", {"result": 42}, cache_type='analysis_results')
    
    print(f"\nCache statistics:")
    stats = cache.get_stats()
    for name, info in stats.items():
        if isinstance(info, dict) and 'entries' in info:
            print(f"  {name}: {info['entries']} entries, {info.get('size_mb', 0):.2f} MB")


def example_file_content_caching():
    """Demonstrate file content caching."""
    print("\n=== File Content Caching ===")
    
    cache = get_cache_manager()
    
    # Simulate file content
    file_path = Path("/tmp/test_file.xlsx")
    file_data = pd.DataFrame({
        'track_id': range(1, 101),
        'resistance': np.random.normal(1000, 50, 100),
        'tolerance': np.random.normal(0.05, 0.01, 100)
    })
    
    # Cache file content
    cache.cache_file_content(file_path, file_data)
    
    # Retrieve cached content
    cached_data = cache.get_file_content(file_path)
    if cached_data is not None:
        print(f"Retrieved cached data with shape: {cached_data.shape}")
    
    # Clear file cache
    cache.invalidate_pattern(f"file:{file_path}:*", 'file_content')
    print("File cache cleared")


def example_ml_prediction_caching():
    """Demonstrate ML prediction caching."""
    print("\n=== ML Prediction Caching ===")
    
    cache = get_cache_manager()
    
    # Simulate ML input data
    input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_hash = "abc123"  # In practice, compute from input_data
    
    # Check if prediction exists
    cached_pred = cache.get_model_prediction("quality_model", input_hash)
    
    if cached_pred is None:
        print("No cached prediction, computing...")
        # Simulate ML prediction
        prediction = np.array([0.95, 0.87])
        
        # Cache the prediction
        cache.cache_model_prediction("quality_model", input_hash, prediction)
        print(f"Cached prediction: {prediction}")
    else:
        print(f"Using cached prediction: {cached_pred}")


def example_decorator_usage():
    """Demonstrate cache decorator usage."""
    print("\n=== Cache Decorator Usage ===")
    
    cache = get_cache_manager()
    
    @cache.decorator(cache_type='general', ttl=60)
    def expensive_calculation(x, y):
        """Simulate expensive calculation."""
        print(f"Computing {x} ^ {y}...")
        time.sleep(1)  # Simulate work
        return x ** y
    
    # First call - takes 1 second
    start = time.time()
    result1 = expensive_calculation(2, 10)
    print(f"First call: {result1} (took {time.time() - start:.2f}s)")
    
    # Second call - returns instantly from cache
    start = time.time()
    result2 = expensive_calculation(2, 10)
    print(f"Second call: {result2} (took {time.time() - start:.2f}s)")
    
    # Different arguments - computes again
    start = time.time()
    result3 = expensive_calculation(3, 10)
    print(f"Third call: {result3} (took {time.time() - start:.2f}s)")


async def example_cached_processor():
    """Demonstrate cached file processor."""
    print("\n=== Cached File Processor ===")
    
    # Setup cache with performance preset
    setup_cache_from_preset(CachePreset.PERFORMANCE)
    
    # Create config and processor
    config = Config()
    output_dir = Path("./output")
    processor = CachedFileProcessor(config, output_dir)
    
    # Simulate processing a file
    test_file = Path("test_files/sample.xlsx")
    
    if test_file.exists():
        # First processing - will cache results
        print("Processing file for the first time...")
        start = time.time()
        result1 = await processor.process_file(test_file)
        print(f"Processing took {time.time() - start:.2f}s")
        
        # Second processing - from cache
        print("\nProcessing same file again...")
        start = time.time()
        result2 = await processor.process_file(test_file)
        print(f"Processing took {time.time() - start:.2f}s (from cache)")
        
        # Get cache statistics
        stats = processor.get_cache_stats()
        print(f"\nCache performance: {stats['hit_rate']:.1%} hit rate")
    else:
        print(f"Test file {test_file} not found")


def example_batch_caching():
    """Demonstrate batch processing with caching."""
    print("\n=== Batch Processing with Caching ===")
    
    config = Config()
    output_dir = Path("./output")
    batch_processor = CachedBatchProcessor(config, output_dir)
    
    # Simulate batch of files
    test_files = [
        Path(f"test_files/file_{i}.xlsx")
        for i in range(1, 6)
    ]
    
    # Preload cache for faster processing
    print("Preloading cache...")
    preloaded = batch_processor.preload_cache(test_files[:3])
    print(f"Preloaded {preloaded} files")
    
    # Process batch
    async def process_batch():
        def progress_callback(current, total, message):
            print(f"[{current}/{total}] {message}")
        
        results = await batch_processor.process_batch(
            test_files,
            progress_callback=progress_callback
        )
        
        success_count = sum(1 for r in results if r.error is None)
        print(f"\nProcessed {success_count}/{len(test_files)} files successfully")
    
    # Run if files exist
    if any(f.exists() for f in test_files):
        asyncio.run(process_batch())
    else:
        print("Test files not found")


def example_cache_management():
    """Demonstrate cache management operations."""
    print("\n=== Cache Management ===")
    
    cache = get_cache_manager()
    
    # Add some test data
    for i in range(10):
        cache.set(f"test:{i}", f"value_{i}")
        cache.set(f"temp:{i}", f"temp_value_{i}", ttl=300)
    
    # Pattern invalidation
    cleared = cache.invalidate_pattern("test:*", 'general')
    print(f"Cleared {cleared} entries matching 'test:*'")
    
    # Get detailed statistics
    stats = cache.get_stats()
    print("\nDetailed cache statistics:")
    for cache_type, info in stats.items():
        if isinstance(info, dict) and 'type' in info:
            print(f"\n{cache_type}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    # Clear specific cache type
    cache.clear('general')
    print("\nCleared general cache")
    
    # Clear all caches
    cache.clear()
    print("Cleared all caches")


def example_advanced_features():
    """Demonstrate advanced cache features."""
    print("\n=== Advanced Cache Features ===")
    
    # Custom cache configuration
    cache_config = CacheConfig(
        memory_cache_size_mb=500,
        file_cache_size_mb=2000,
        general_strategy=CacheStrategy.LFU,
        enable_memory_monitor=True,
        memory_threshold=0.9
    )
    
    # Initialize with custom config
    from src.laser_trim_analyzer.core.cache_config import setup_cache
    setup_cache(cache_config)
    
    cache = get_cache_manager()
    
    # Test memory-aware caching
    print("Testing memory-aware caching...")
    
    # Add large data
    large_data = np.random.rand(1000, 1000)  # ~8MB
    cache.set("large_data", large_data)
    
    # Check memory usage
    stats = cache.get_stats()
    if 'system_memory' in stats:
        mem = stats['system_memory']
        print(f"System memory usage: {mem['percent_used']:.1f}%")
    
    # Test cache eviction
    print("\nTesting cache eviction...")
    
    # Fill cache to trigger eviction
    for i in range(100):
        data = np.random.rand(100, 100)
        cache.set(f"evict_test:{i}", data)
    
    # Check how many entries remain
    stats = cache.get_stats()
    general_stats = stats.get('general', {})
    print(f"Entries after eviction: {general_stats.get('entries', 0)}")


def main():
    """Run all examples."""
    examples = [
        example_basic_caching,
        example_file_content_caching,
        example_ml_prediction_caching,
        example_decorator_usage,
        example_cache_management,
        example_advanced_features
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
        print("\n" + "="*50 + "\n")
    
    # Run async examples
    print("Running async examples...")
    asyncio.run(example_cached_processor())
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    # Initialize cache system
    initialize_cache_manager()
    
    # Run examples
    main()