"""
Performance Benchmark Script for Laser Trim Analyzer

Measures processing performance for different batch sizes (100/500/1000 files).
Records: time, memory usage, CPU usage, throughput.

Usage:
    python scripts/benchmark_processing.py --files 100
    python scripts/benchmark_processing.py --files 500
    python scripts/benchmark_processing.py --files 1000
"""

import argparse
import asyncio
import time
import psutil
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.core.processor import LaserTrimProcessor


def find_test_files(test_dir: Path, limit: int) -> List[Path]:
    """Find Excel test files up to the specified limit."""
    print(f"Searching for test files in: {test_dir}")

    # Find all Excel files
    excel_files = []
    for ext in ["*.xls", "*.xlsx"]:
        excel_files.extend(test_dir.rglob(ext))

    # Sort for consistency
    excel_files.sort()

    # Limit to requested count
    files = excel_files[:limit]

    print(f"Found {len(excel_files)} total files, using first {len(files)} for benchmark")
    return files


def measure_memory() -> Dict[str, float]:
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return {
        "rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size in MB
        "vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
    }


def measure_cpu() -> float:
    """Get current CPU usage percentage."""
    process = psutil.Process(os.getpid())
    return process.cpu_percent(interval=0.1)


async def run_benchmark(file_count: int, test_dir: Path, config: Config) -> Dict[str, Any]:
    """Run processing benchmark for specified number of files."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Processing {file_count} files")
    print(f"{'='*60}\n")

    # Find test files
    files = find_test_files(test_dir, file_count)

    if len(files) < file_count:
        print(f"WARNING: Only found {len(files)} files, requested {file_count}")

    # Measure initial state
    mem_before = measure_memory()
    print(f"Memory before: {mem_before['rss_mb']:.1f} MB (RSS)")

    # Create processor
    processor = LaserTrimProcessor(config)

    # Warm-up: process first file (not included in benchmark)
    if files:
        print("Warming up (processing 1 file)...")
        try:
            await processor.process_file(str(files[0]))
        except Exception as e:
            print(f"Warm-up failed: {e}")

    # Wait a moment for system to stabilize
    await asyncio.sleep(1)

    # Reset memory measurement after warm-up
    mem_before = measure_memory()

    # Start benchmark
    print(f"\nStarting benchmark for {len(files)} files...")
    start_time = time.time()
    start_cpu = measure_cpu()

    # Process files
    successful = 0
    failed = 0
    errors = []

    for i, file_path in enumerate(files, 1):
        try:
            # Process file
            result = await processor.process_file(str(file_path))

            if result and result.status == "success":
                successful += 1
            else:
                failed += 1
                errors.append(f"{file_path.name}: No result or failed status")

        except Exception as e:
            failed += 1
            errors.append(f"{file_path.name}: {str(e)}")

        # Progress update every 10%
        if i % max(1, len(files) // 10) == 0:
            progress = (i / len(files)) * 100
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"Progress: {progress:5.1f}% ({i}/{len(files)}) - "
                  f"{rate:.2f} files/sec - {elapsed:.1f}s elapsed")

    # End benchmark
    end_time = time.time()
    end_cpu = measure_cpu()
    mem_after = measure_memory()

    # Calculate metrics
    total_time = end_time - start_time
    throughput = len(files) / total_time if total_time > 0 else 0
    time_per_file_ms = (total_time / len(files)) * 1000 if files else 0

    mem_peak_mb = mem_after["rss_mb"]
    mem_increase_mb = mem_after["rss_mb"] - mem_before["rss_mb"]

    # Results
    results = {
        "file_count": len(files),
        "successful": successful,
        "failed": failed,
        "total_time_seconds": round(total_time, 2),
        "throughput_files_per_second": round(throughput, 2),
        "time_per_file_ms": round(time_per_file_ms, 1),
        "memory_before_mb": round(mem_before["rss_mb"], 1),
        "memory_after_mb": round(mem_after["rss_mb"], 1),
        "memory_peak_mb": round(mem_peak_mb, 1),
        "memory_increase_mb": round(mem_increase_mb, 1),
        "cpu_percent": round((start_cpu + end_cpu) / 2, 1),
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS: {file_count} files")
    print(f"{'='*60}")
    print(f"Success Rate:  {successful}/{len(files)} files ({successful/len(files)*100:.1f}%)")
    print(f"Total Time:    {total_time:.2f} seconds")
    print(f"Throughput:    {throughput:.2f} files/second")
    print(f"Per File:      {time_per_file_ms:.1f} ms/file")
    print(f"Memory Peak:   {mem_peak_mb:.1f} MB")
    print(f"Memory Change: +{mem_increase_mb:.1f} MB")
    print(f"CPU Usage:     ~{results['cpu_percent']}%")
    print(f"{'='*60}\n")

    if errors and failed > 0:
        print(f"Failed files: {failed}")
        if len(errors) <= 5:
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"  (showing first 5 of {len(errors)} errors)")
            for error in errors[:5]:
                print(f"  - {error}")

    return results


async def main_async(args):
    """Async main function."""
    # Setup
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}")
        return 1

    # Load config (development mode)
    os.environ["LTA_ENV"] = "development"
    config = Config()

    print(f"Benchmark Configuration:")
    print(f"  Files:    {args.files}")
    print(f"  Test Dir: {test_dir}")
    print(f"  Mode:     {os.environ.get('LTA_ENV', 'development')}")

    # Run benchmark
    results = await run_benchmark(args.files, test_dir, config)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Return appropriate exit code
    return 0 if results["failed"] == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Benchmark processing performance")
    parser.add_argument("--files", type=int, default=100,
                        help="Number of files to process (100, 500, 1000)")
    parser.add_argument("--test-dir", type=str, default="test_files",
                        help="Directory containing test files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Run async main
    exit_code = asyncio.run(main_async(args))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
