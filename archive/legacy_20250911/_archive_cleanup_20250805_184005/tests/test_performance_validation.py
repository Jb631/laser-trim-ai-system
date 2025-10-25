"""
Performance Validation Test Suite

Validates and benchmarks:
- Large batch processing performance
- Memory usage optimization
- UI responsiveness under load
- File processing throughput
- System resource management
"""

import pytest
import asyncio
import tempfile
import shutil
import time
import psutil
import gc
from pathlib import Path
from unittest.mock import Mock, patch
import tkinter as tk
from tkinter import ttk

from laser_trim_analyzer.gui.pages.analysis_page import AnalysisPage
from laser_trim_analyzer.core.config import Config, ProcessingConfig
from laser_trim_analyzer.core.processor import LaserTrimProcessor


class TestPerformanceValidation:
    """Comprehensive performance validation tests."""

    @pytest.fixture
    def root_window(self):
        """Create root tkinter window."""
        root = tk.Tk()
        root.withdraw()
        yield root
        root.destroy()

    @pytest.fixture
    def test_config(self):
        """Performance-optimized test configuration."""
        return Config(
            debug=False,  # Disable debug for accurate performance testing
            processing=ProcessingConfig(
                max_workers=4,
                generate_plots=False,
                cache_enabled=True,
                max_batch_size=1000,
                memory_limit_mb=1024.0,
                high_performance_mode=True
            )
        )

    @pytest.fixture
    def mock_main_window(self, test_config):
        """Create mock main window with performance config."""
        mock_window = Mock()
        mock_window.config = test_config
        mock_window.db_manager = None
        # ML is required - create a mock ML predictor
        mock_ml_predictor = Mock()
        mock_ml_predictor.predict = Mock(return_value=0.1)  # Low failure risk
        mock_ml_predictor.is_initialized = True
        mock_window.ml_predictor = mock_ml_predictor
        mock_window.logger = Mock()
        return mock_window

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def measure_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def test_large_batch_file_loading_performance(self, root_window, mock_main_window):
        """Test performance with very large file batches."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Performance benchmarks for different batch sizes
        batch_sizes = [100, 500, 1000, 2000]
        results = {}
        
        for batch_size in batch_sizes:
            # Clean up before each test
            gc.collect()
            initial_memory = self.measure_memory_usage()
            
            # Create file list
            files = [Path(f"perf_test_{i}.xlsx") for i in range(batch_size)]
            
            # Measure loading time
            start_time = time.time()
            page._add_files(files)
            load_time = time.time() - start_time
            
            # Measure memory usage
            final_memory = self.measure_memory_usage()
            memory_increase = final_memory - initial_memory
            
            results[batch_size] = {
                'load_time': load_time,
                'memory_increase': memory_increase,
                'files_per_second': batch_size / load_time if load_time > 0 else float('inf')
            }
            
            # Clean up for next iteration
            page._clear_files()
            gc.collect()
        
        # Validate performance requirements
        for batch_size, metrics in results.items():
            # Loading should be fast (< 5 seconds for any batch size)
            assert metrics['load_time'] < 5.0, \
                f"Batch {batch_size}: Load time {metrics['load_time']:.2f}s too slow"
            
            # Memory usage should be reasonable (< 1MB per 100 files)
            max_memory = batch_size / 100 * 1.0  # 1MB per 100 files
            assert metrics['memory_increase'] < max_memory, \
                f"Batch {batch_size}: Memory usage {metrics['memory_increase']:.2f}MB too high"
            
            # Throughput should be reasonable (> 100 files/second)
            assert metrics['files_per_second'] > 100, \
                f"Batch {batch_size}: Throughput {metrics['files_per_second']:.2f} files/s too low"

    def test_tree_view_vs_widget_performance(self, root_window, mock_main_window):
        """Compare performance between tree view and widget modes."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Test data
        file_count = 500
        files = [Path(f"compare_{i}.xlsx") for i in range(file_count)]
        
        # Test widget mode (force it for files < 200)
        gc.collect()
        widget_start_memory = self.measure_memory_usage()
        widget_start_time = time.time()
        
        with patch.object(page, '_switch_to_tree_view_mode'):
            page._add_files(files[:150])  # Under threshold
        
        widget_time = time.time() - widget_start_time
        widget_memory = self.measure_memory_usage() - widget_start_memory
        
        page._clear_files()
        gc.collect()
        
        # Test tree view mode (force it for files > 200)
        tree_start_memory = self.measure_memory_usage()
        tree_start_time = time.time()
        
        page._add_files(files)  # Over threshold, should use tree view
        
        tree_time = time.time() - tree_start_time
        tree_memory = self.measure_memory_usage() - tree_start_memory
        
        # Tree view should be more efficient
        assert tree_time < widget_time * 2, \
            f"Tree view ({tree_time:.2f}s) not significantly faster than widgets ({widget_time:.2f}s)"
        
        # Tree view should use less memory per file
        widget_memory_per_file = widget_memory / 150
        tree_memory_per_file = tree_memory / file_count
        
        assert tree_memory_per_file < widget_memory_per_file, \
            f"Tree view memory per file ({tree_memory_per_file:.4f}MB) not better than widgets ({widget_memory_per_file:.4f}MB)"

    def test_ui_responsiveness_under_load(self, root_window, mock_main_window):
        """Test UI remains responsive under heavy load."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Track UI operations
        ui_operations = []
        original_after = page.after
        
        def track_ui_operations(delay, func, *args):
            ui_operations.append({
                'timestamp': time.time(),
                'delay': delay,
                'function': func.__name__ if hasattr(func, '__name__') else str(func)
            })
            return original_after(delay, func, *args)
        
        page.after = track_ui_operations
        
        # Start heavy file loading
        large_files = [Path(f"load_test_{i}.xlsx") for i in range(1000)]
        
        start_time = time.time()
        page._add_files(large_files)
        
        # Simulate user interactions during loading
        interaction_times = []
        for i in range(10):
            interaction_start = time.time()
            
            # Simulate UI update
            page._update_stats()
            
            interaction_time = time.time() - interaction_start
            interaction_times.append(interaction_time)
            
            time.sleep(0.1)  # Wait between interactions
        
        total_time = time.time() - start_time
        
        # Validate responsiveness
        max_interaction_time = max(interaction_times)
        avg_interaction_time = sum(interaction_times) / len(interaction_times)
        
        # UI interactions should complete quickly
        assert max_interaction_time < 0.1, \
            f"UI interaction too slow: {max_interaction_time:.3f}s"
        
        assert avg_interaction_time < 0.05, \
            f"Average UI interaction too slow: {avg_interaction_time:.3f}s"
        
        # Should have scheduled non-blocking operations
        assert len(ui_operations) > 0, "No non-blocking UI operations detected"
        
        # UI delays should be reasonable
        delays = [op['delay'] for op in ui_operations]
        assert all(delay <= 10 for delay in delays), \
            "Some UI delays too long for responsiveness"

    def test_memory_leak_detection(self, root_window, mock_main_window):
        """Test for memory leaks during repeated operations."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Baseline memory
        gc.collect()
        baseline_memory = self.measure_memory_usage()
        
        memory_samples = []
        
        # Perform repeated file operations
        for iteration in range(10):
            # Add files
            files = [Path(f"leak_test_{iteration}_{i}.xlsx") for i in range(100)]
            page._add_files(files)
            
            # Update file statuses
            for i, file in enumerate(files[:10]):
                page._update_file_status(str(file), 'Processing')
            
            # Clear files
            page._clear_files()
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory
            current_memory = self.measure_memory_usage()
            memory_increase = current_memory - baseline_memory
            memory_samples.append(memory_increase)
        
        # Check for memory leaks
        initial_increase = memory_samples[0]
        final_increase = memory_samples[-1]
        
        # Memory should not grow significantly over iterations
        memory_growth = final_increase - initial_increase
        assert memory_growth < 10.0, \
            f"Potential memory leak detected: {memory_growth:.2f}MB growth over iterations"
        
        # Memory variance should be low (stable)
        import statistics
        memory_variance = statistics.variance(memory_samples[1:])  # Skip first sample
        assert memory_variance < 5.0, \
            f"Memory usage too variable: {memory_variance:.2f}MB variance"

    def test_file_processing_throughput(self, temp_dir):
        """Test file processing throughput under optimal conditions."""
        # Create test configuration optimized for throughput
        config = Config(
            processing=ProcessingConfig(
                max_workers=4,
                generate_plots=False,
                cache_enabled=True,
                high_performance_mode=True,
                concurrent_batch_size=20,
                memory_limit_mb=2048.0
            )
        )
        
        # Create processor
        processor = LaserTrimProcessor(config=config)
        
        # Create test files (simplified for performance testing)
        test_files = []
        for i in range(50):  # Manageable number for actual file creation
            file_path = temp_dir / f"throughput_test_{i}.xlsx"
            # Create minimal Excel file for testing
            import pandas as pd
            
            # Create very simple test data
            df = pd.DataFrame({
                'A': range(10),
                'B': range(10),
                'G': [0.01] * 10,  # Error
                'H': range(10),    # Position
                'I': [0.05] * 10,  # Upper limit
                'J': [-0.05] * 10  # Lower limit
            })
            
            with pd.ExcelWriter(file_path) as writer:
                df.to_excel(writer, sheet_name='SEC1 TRK1 0', index=False)
            
            test_files.append(file_path)
        
        # Measure processing throughput
        start_time = time.time()
        processed_count = 0
        
        for file_path in test_files:
            try:
                # Simple processing without full analysis
                result = asyncio.run(processor.process_file(file_path, temp_dir / "output"))
                if result:
                    processed_count += 1
            except Exception:
                pass  # Skip failed files for throughput test
        
        total_time = time.time() - start_time
        throughput = processed_count / total_time if total_time > 0 else 0
        
        # Validate throughput requirements
        assert throughput > 2.0, \
            f"Processing throughput too low: {throughput:.2f} files/second"
        
        assert processed_count >= len(test_files) * 0.8, \
            f"Too many processing failures: {processed_count}/{len(test_files)} succeeded"

    def test_concurrent_operations_performance(self, root_window, mock_main_window):
        """Test performance under concurrent UI and processing operations."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Add large batch of files
        files = [Path(f"concurrent_{i}.xlsx") for i in range(500)]
        page._add_files(files)
        
        # Measure concurrent operations
        operation_times = []
        
        for i in range(20):
            start_time = time.time()
            
            # Simulate concurrent operations
            page._update_stats()
            page._update_file_status(str(files[i % len(files)]), 'Processing')
            
            if i % 5 == 0:
                # Occasional heavy operation
                page._update_ui_state()
            
            operation_time = time.time() - start_time
            operation_times.append(operation_time)
        
        # Validate concurrent performance
        max_operation_time = max(operation_times)
        avg_operation_time = sum(operation_times) / len(operation_times)
        
        assert max_operation_time < 0.05, \
            f"Concurrent operation too slow: {max_operation_time:.3f}s"
        
        assert avg_operation_time < 0.02, \
            f"Average concurrent operation too slow: {avg_operation_time:.3f}s"

    def test_system_resource_limits(self, root_window, mock_main_window):
        """Test system stays within resource limits under stress."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        initial_cpu_percent = process.cpu_percent()
        
        # Stress test with maximum file load
        max_files = [Path(f"stress_{i}.xlsx") for i in range(3000)]
        
        start_time = time.time()
        page._add_files(max_files)
        
        # Monitor for a period
        monitoring_duration = 2.0
        memory_peaks = []
        cpu_peaks = []
        
        monitor_start = time.time()
        while time.time() - monitor_start < monitoring_duration:
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            memory_peaks.append(memory_mb)
            cpu_peaks.append(cpu_percent)
            
            time.sleep(0.1)
        
        load_time = time.time() - start_time
        
        # Validate resource usage
        max_memory = max(memory_peaks)
        avg_cpu = sum(cpu_peaks) / len(cpu_peaks) if cpu_peaks else 0
        
        memory_increase = max_memory - initial_memory
        
        # Should not use excessive memory (< 500MB increase)
        assert memory_increase < 500, \
            f"Memory usage too high: {memory_increase:.2f}MB increase"
        
        # Should not consume excessive CPU (< 80% average)
        assert avg_cpu < 80, \
            f"CPU usage too high: {avg_cpu:.1f}% average"
        
        # Should complete within reasonable time
        assert load_time < 10.0, \
            f"Stress test took too long: {load_time:.2f}s"

    def test_performance_regression_detection(self, root_window, mock_main_window):
        """Detect performance regressions compared to baseline."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Baseline performance targets (these should be maintained)
        performance_targets = {
            'file_loading_100': 0.5,      # 100 files in 0.5s
            'file_loading_500': 1.5,      # 500 files in 1.5s
            'file_loading_1000': 3.0,     # 1000 files in 3.0s
            'status_update_100': 0.1,     # 100 status updates in 0.1s
            'memory_per_file': 0.01,      # 0.01MB per file maximum
        }
        
        # Test file loading performance
        for file_count in [100, 500, 1000]:
            files = [Path(f"regression_{i}.xlsx") for i in range(file_count)]
            
            gc.collect()
            start_memory = self.measure_memory_usage()
            start_time = time.time()
            
            page._add_files(files)
            
            load_time = time.time() - start_time
            memory_increase = self.measure_memory_usage() - start_memory
            
            # Check against targets
            target_key = f'file_loading_{file_count}'
            assert load_time <= performance_targets[target_key], \
                f"Regression detected: {target_key} took {load_time:.2f}s, target {performance_targets[target_key]}s"
            
            memory_per_file = memory_increase / file_count
            assert memory_per_file <= performance_targets['memory_per_file'], \
                f"Memory regression: {memory_per_file:.4f}MB per file, target {performance_targets['memory_per_file']}MB"
            
            page._clear_files()
            gc.collect()
        
        # Test status update performance
        files = [Path(f"status_test_{i}.xlsx") for i in range(100)]
        page._add_files(files)
        
        start_time = time.time()
        for file in files:
            page._update_file_status(str(file), 'Processing')
        update_time = time.time() - start_time
        
        assert update_time <= performance_targets['status_update_100'], \
            f"Status update regression: {update_time:.2f}s, target {performance_targets['status_update_100']}s"


if __name__ == "__main__":
    # Run performance tests with detailed output
    pytest.main([__file__, "-v", "--tb=short", "-s"]) 