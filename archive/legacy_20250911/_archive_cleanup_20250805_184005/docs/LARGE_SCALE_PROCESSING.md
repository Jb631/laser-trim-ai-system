# Large-Scale Processing Guide

This guide covers how to efficiently process thousands of files with the Laser Trim Analyzer.

## Overview

The Laser Trim Analyzer is optimized to handle large-scale batch processing with:

- **Intelligent Memory Management**: Automatic garbage collection and cache clearing
- **Chunked Processing**: Files processed in optimized batches
- **Performance Monitoring**: Real-time memory and speed tracking
- **Crash Recovery**: Resume from specific files
- **Database Optimization**: Bulk inserts and batch commits
- **Automatic Scaling**: Optimizations kick in for large batches

## Quick Start

### 1. Scan Directory First
Before processing thousands of files, scan the directory to understand requirements:

```bash
python -m laser_trim_analyzer scan "C:\path\to\your\files"
```

This will show you:
- Total file count and size
- Estimated processing time
- Memory requirements
- Optimization recommendations

### 2. Process Large Directories

**For Standard Processing (< 500 files):**
```bash
python -m laser_trim_analyzer batch "C:\path\to\your\files" --output "C:\results"
```

**For Large Batches (500+ files):**
```bash
python -m laser_trim_analyzer batch "C:\path\to\your\files" \
    --output "C:\results" \
    --high-performance \
    --disable-plots \
    --max-workers 8
```

**For Very Large Batches (1000+ files):**
```bash
python -m laser_trim_analyzer batch "C:\path\to\your\files" \
    --output "C:\results" \
    --high-performance \
    --disable-plots \
    --max-workers 12 \
    --memory-limit 4 \
    --batch-size 2000
```

## Configuration Optimizations

### For Large-Scale Processing

Create a `config/large_scale.yaml` file:

```yaml
processing:
  # Increased limits for large batches
  max_batch_size: 2000
  memory_limit_mb: 4096  # 4GB
  max_workers: 12
  max_concurrent_files: 100
  
  # Performance optimizations
  high_performance_mode: true
  disable_plots_large_batch: 500
  enable_streaming_processing: true
  generate_plots: false  # Disable for speed
  
  # Memory management
  gc_interval: 50  # Garbage collect every 50 files
  clear_cache_interval: 500
  
  # Database optimizations
  enable_bulk_insert: true
  database_batch_size: 200
  batch_commit_interval: 100

database:
  enabled: true
  pool_size: 10  # Increased connection pool
```

Then use it:
```bash
python -m laser_trim_analyzer batch "C:\path\to\files" --config-file config/large_scale.yaml
```

## Performance Tuning

### Memory Management

**Current Default**: 2GB memory limit, 50 concurrent files
**For Large Batches**: 4-8GB memory limit, 100+ concurrent files

```yaml
processing:
  memory_limit_mb: 8192  # 8GB
  max_concurrent_files: 150
  gc_interval: 25  # More frequent cleanup
```

### CPU Optimization

**Conservative**: 4 workers (default)
**Balanced**: 8 workers (recommended for most systems)
**Aggressive**: 12-16 workers (high-end systems only)

```bash
# Detect your CPU cores
python -c "import psutil; print(f'CPU cores: {psutil.cpu_count()}')"

# Use 75% of cores for processing
python -m laser_trim_analyzer batch "path" --max-workers 12
```

### Storage Optimization

**For Thousands of Files**:
- Use SSD storage for input and output
- Disable plot generation: `--disable-plots`
- Consider separate output drive
- Enable compression for database storage

## Monitoring and Recovery

### Progress Monitoring

The batch processor provides real-time monitoring:

```
🔄 Progress: [████████████████████░░░░░░░░░░] 75.3%
📝 Status: Processing file_2847.xlsx...

Processing Statistics:
├─ Files Processed: 2,847
├─ Failed Files: 23
├─ Total Files: 3,782
├─ Speed: 4.2 files/sec
├─ Elapsed Time: 11.3 minutes
├─ Est. Remaining: 3.7 minutes
├─ Memory Usage: 1,247.3 MB
└─ Peak Memory: 1,534.8 MB
```

### Crash Recovery

If processing is interrupted, resume from where you left off:

```bash
python -m laser_trim_analyzer batch "C:\path\to\files" \
    --resume-from "file_2847.xlsx" \
    --output "C:\results"
```

### Error Handling

Failed files are logged with detailed error messages:
- Processing continues with remaining files
- Failed files are reported in final summary
- Individual file errors don't stop batch processing

## Best Practices

### 1. Pre-Processing Validation

```bash
# Check file accessibility and format
python -m laser_trim_analyzer scan "C:\path\to\files" --include-subdirs
```

### 2. Gradual Scaling

Start with smaller batches to test configuration:

```bash
# Test with 100 files first
python -m laser_trim_analyzer batch "C:\test_subset" --max-workers 4

# Then scale up
python -m laser_trim_analyzer batch "C:\full_directory" --high-performance
```

### 3. Monitor System Resources

```bash
# Watch memory usage during processing
# Windows: Task Manager or Resource Monitor
# Use --memory-limit to prevent system issues
```

### 4. Database Management

For thousands of files, database performance is critical:

```yaml
database:
  enabled: true
  pool_size: 20
  # Use faster database for very large batches
  path: "D:\\fast_ssd\\analysis.db"
```

### 5. Network Storage Considerations

**Avoid**: Network drives for large batches
**Recommended**: Local SSD storage
**Alternative**: Copy files locally first

```bash
# Copy files locally for better performance
robocopy "\\network\path" "C:\local_temp" *.xlsx /S
python -m laser_trim_analyzer batch "C:\local_temp" --output "C:\results"
```

## Troubleshooting

### Memory Issues

**Symptoms**: System slowing down, out of memory errors
**Solutions**:
```bash
# Reduce concurrent files
--max-concurrent-files 25

# Lower memory limit
--memory-limit 2

# More frequent cleanup
--config-file config/low_memory.yaml
```

### Performance Issues

**Symptoms**: Very slow processing (< 1 file/sec)
**Solutions**:
```bash
# Enable high-performance mode
--high-performance

# Disable plots
--disable-plots

# Increase workers (if CPU allows)
--max-workers 8

# Check storage speed
# Move to SSD if using HDD
```

### Database Lock Issues

**Symptoms**: "Database locked" errors
**Solutions**:
```yaml
database:
  pool_size: 1  # Reduce to single connection
  enable_bulk_insert: false  # Disable bulk operations
```

### File Access Issues

**Symptoms**: "Permission denied" or "File in use" errors
**Solutions**:
- Close Excel/other applications using files
- Run as administrator if needed
- Check file permissions
- Use `--skip-patterns` to exclude problematic files

## Performance Benchmarks

### Typical Performance

| File Count | System | Time | Speed | Memory |
|------------|--------|------|-------|--------|
| 100 files | Standard | 2-5 min | 0.5-2 files/sec | 500MB |
| 500 files | Standard | 10-25 min | 1-3 files/sec | 1GB |
| 1000 files | High-Perf | 15-30 min | 2-4 files/sec | 2GB |
| 5000 files | High-Perf | 60-120 min | 3-5 files/sec | 4GB |

### Optimization Impact

| Optimization | Speed Improvement | Memory Reduction |
|--------------|-------------------|------------------|
| High-Performance Mode | +50-100% | -20% |
| Disable Plots | +30-50% | -30% |
| SSD Storage | +25-40% | 0% |
| Increased Workers | +20-80% | +10-30% |

## Advanced Configuration

### Custom File Filtering

```python
# In Python code, you can add custom filters
def custom_filter(file_path):
    # Only process files from last 30 days
    import time
    file_age = time.time() - file_path.stat().st_mtime
    return file_age < (30 * 24 * 3600)  # 30 days

# Use with large-scale processor
from laser_trim_analyzer.core.large_scale_processor import LargeScaleProcessor
processor = LargeScaleProcessor(config)
results = await processor.process_large_directory(
    directory=Path("C:/files"),
    file_filter=custom_filter
)
```

### Custom Progress Tracking

```python
def detailed_progress_callback(message, progress, stats):
    # Custom logging or monitoring
    print(f"Progress: {progress*100:.1f}% - {message}")
    
    # Log to file
    with open("processing_log.txt", "a") as f:
        f.write(f"{time.time()}: {progress:.3f} - {message}\n")
    
    # Send to monitoring system
    # send_metrics_to_dashboard(stats)
```

## Summary

Processing thousands of files is definitely supported and optimized. Key points:

✅ **Automatic Optimizations**: System detects large batches and optimizes automatically  
✅ **Memory Management**: Intelligent cleanup prevents memory issues  
✅ **Performance Scaling**: Up to 16 workers, configurable batch sizes  
✅ **Progress Tracking**: Real-time monitoring and ETA  
✅ **Crash Recovery**: Resume from interruptions  
✅ **Database Optimization**: Bulk operations for better performance  

The system is designed to handle enterprise-scale processing efficiently while maintaining reliability and providing detailed progress feedback. 