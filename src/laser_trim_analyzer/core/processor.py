# laser_trim_analyzer/core/processor.py
"""
Main processor implementation with clean architecture.
"""
import asyncio
from typing import List, Optional, Dict, Any
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

from .interfaces import (
    FileResult, TrackResult, SystemType, Status,
    FileReader, DataExtractor, MetricsCalculator, ResultsFormatter
)
from .strategies import SystemStrategy, SystemAStrategy, SystemBStrategy


class LaserTrimProcessor:
    """
    Main processor class with clean separation of concerns.

    This class orchestrates the analysis process but delegates
    specific tasks to specialized components.
    """

    def __init__(
            self,
            file_reader: FileReader,
            data_extractor: DataExtractor,
            metrics_calculator: MetricsCalculator,
            results_formatter: ResultsFormatter,
            logger: Optional[logging.Logger] = None,
            max_workers: int = 4,
            generate_plots: bool = True
    ):
        self.file_reader = file_reader
        self.data_extractor = data_extractor
        self.metrics_calculator = metrics_calculator
        self.results_formatter = results_formatter
        self.logger = logger or logging.getLogger(__name__)
        self.max_workers = max_workers
        self.generate_plots = generate_plots

        # Strategy pattern for different systems
        self.strategies: Dict[SystemType, SystemStrategy] = {
            SystemType.SYSTEM_A: SystemAStrategy(data_extractor, metrics_calculator),
            SystemType.SYSTEM_B: SystemBStrategy(data_extractor, metrics_calculator)
        }

    async def process_folder(
            self,
            folder_path: Path,
            progress_callback: Optional[callable] = None
    ) -> List[FileResult]:
        """
        Process all Excel files in a folder asynchronously.

        Args:
            folder_path: Path to folder containing Excel files
            progress_callback: Optional callback for progress updates

        Returns:
            List of processing results
        """
        folder_path = Path(folder_path)
        excel_files = list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls"))

        if not excel_files:
            self.logger.warning(f"No Excel files found in {folder_path}")
            return []

        self.logger.info(f"Found {len(excel_files)} Excel files to process")

        # Process files concurrently
        tasks = []
        for i, file_path in enumerate(excel_files):
            if progress_callback:
                progress_callback(i, len(excel_files), file_path.name)

            task = self.process_file(file_path)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Processing error: {result}")
            elif result:
                valid_results.append(result)

        return valid_results

    async def process_file(self, filepath: Path) -> Optional[FileResult]:
        """
        Process a single file with proper error handling.

        Args:
            filepath: Path to Excel file

        Returns:
            Processing results or None if failed
        """
        start_time = time.time()
        self.logger.info(f"Processing file: {filepath.name}")

        try:
            # Read file structure
            file_info = await self.file_reader.read_file(str(filepath))

            # Detect system type
            system_type = self._detect_system(file_info)
            strategy = self.strategies[system_type]

            # Extract metadata
            model, serial = self._parse_filename(filepath.name)

            # Process using appropriate strategy
            tracks = await strategy.process_file(
                filepath=str(filepath),
                file_info=file_info,
                model=model,
                generate_plots=self.generate_plots
            )

            if not tracks:
                self.logger.error(f"No valid tracks found in {filepath.name}")
                return None

            # Determine overall status
            overall_status = self._determine_overall_status(tracks)

            # Create file result
            result = FileResult(
                filename=filepath.name,
                filepath=str(filepath),
                model=model,
                serial=serial,
                system=system_type,
                overall_status=overall_status,
                tracks=tracks,
                is_multi_track=len(tracks) > 1,
                file_date=datetime.fromtimestamp(filepath.stat().st_mtime),
                processing_time=time.time() - start_time,
                output_directory=str(filepath.parent)
            )

            self.logger.info(
                f"Completed {filepath.name}: {overall_status.value} "
                f"({len(tracks)} tracks, {result.processing_time:.2f}s)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error processing {filepath.name}: {e}", exc_info=True)
            return None

    def _detect_system(self, file_info: Dict[str, Any]) -> SystemType:
        """Detect system type from file structure."""
        sheet_names = file_info.get("sheet_names", [])

        # Check for System A patterns
        if any("SEC1 TRK1" in sheet for sheet in sheet_names):
            return SystemType.SYSTEM_A

        # Check for System B patterns
        if "test" in sheet_names and "Lin Error" in sheet_names:
            return SystemType.SYSTEM_B

        # Default based on filename patterns
        filename = file_info.get("filename", "")
        if filename.startswith(("8340", "834")):
            return SystemType.SYSTEM_B
        elif filename.startswith(("68", "78", "85")):
            return SystemType.SYSTEM_A

        # Default to System A
        self.logger.warning(f"Could not detect system type, defaulting to System A")
        return SystemType.SYSTEM_A

    def _parse_filename(self, filename: str) -> tuple[str, str]:
        """Extract model and serial from filename."""
        parts = filename.split('_')

        if len(parts) >= 2:
            model = parts[0]
            serial = ''.join(c for c in parts[1] if c.isalnum())
            return model, serial

        return "Unknown", "Unknown"

    def _determine_overall_status(self, tracks: Dict[str, TrackResult]) -> Status:
        """Determine overall file status from track statuses."""
        if not tracks:
            return Status.ERROR

        statuses = [track.status for track in tracks.values()]

        if any(s == Status.FAIL for s in statuses):
            return Status.FAIL
        elif any(s == Status.WARNING for s in statuses):
            return Status.WARNING
        elif all(s == Status.PASS for s in statuses):
            return Status.PASS
        else:
            return Status.UNKNOWN

