# laser_trim_analyzer/core/utils.py
"""
Utility functions for the refactored processor.
"""
import asyncio
from typing import List, TypeVar, Callable, Optional
import logging
from pathlib import Path

T = TypeVar('T')


async def process_concurrently(
        items: List[T],
        processor: Callable[[T], asyncio.Future],
        max_concurrent: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None
) -> List:
    """
    Process items concurrently with a limit on concurrent operations.

    Args:
        items: List of items to process
        processor: Async function to process each item
        max_concurrent: Maximum concurrent operations
        progress_callback: Optional progress callback

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(item, index):
        async with semaphore:
            result = await processor(item)
            if progress_callback:
                progress_callback(index + 1, len(items))
            return result

    tasks = [
        process_with_semaphore(item, i)
        for i, item in enumerate(items)
    ]

    return await asyncio.gather(*tasks, return_exceptions=True)


def setup_logging(output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        output_dir: Directory for log files
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger('laser_trim_analyzer')
    logger.setLevel(level)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # File handler
    log_file = output_dir / 'laser_trim_analysis.log'
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# laser_trim_analyzer/core/example_usage.py
"""
Example of how to use the refactored processor.
"""
import asyncio
from pathlib import Path

from .processor import LaserTrimProcessor
from .implementations import (
    ExcelFileReader,
    StandardDataExtractor,
    StandardMetricsCalculator,
    ExcelResultsFormatter
)
from .utils import setup_logging


async def main():
    """Example usage of the refactored processor."""
    # Setup
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    logger = setup_logging(output_dir)

    # Create components
    file_reader = ExcelFileReader()
    data_extractor = StandardDataExtractor(logger)
    metrics_calculator = StandardMetricsCalculator(logger)
    results_formatter = ExcelResultsFormatter(logger)

    # Create processor
    processor = LaserTrimProcessor(
        file_reader=file_reader,
        data_extractor=data_extractor,
        metrics_calculator=metrics_calculator,
        results_formatter=results_formatter,
        logger=logger,
        max_workers=4,
        generate_plots=True
    )

    # Process folder
    input_folder = Path("data/laser_trim_files")

    def progress_callback(current: int, total: int, filename: str):
        print(f"Processing {current}/{total}: {filename}")

    # Run processing
    results = await processor.process_folder(input_folder, progress_callback)

    # Generate outputs
    if results:
        # Excel report
        df = results_formatter.format_for_excel(results)
        excel_path = output_dir / "analysis_results.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"Excel report saved to: {excel_path}")

        # HTML report
        html_content = results_formatter.format_for_html(results)
        html_path = output_dir / "analysis_report.html"
        html_path.write_text(html_content)
        print(f"HTML report saved to: {html_path}")

        # Database format (for example)
        db_records = results_formatter.format_for_database(results)
        print(f"Prepared {len(db_records)} records for database storage")

    print(f"\nProcessing complete! Analyzed {len(results)} files.")


if __name__ == "__main__":
    asyncio.run(main())