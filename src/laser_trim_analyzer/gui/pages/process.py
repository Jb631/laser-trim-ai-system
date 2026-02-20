"""
Process Page - Import and process files.

Handles single file and batch processing with incremental mode.
Wired to the actual Processor for real analysis.
"""

import customtkinter as ctk
import logging
import os
import time
from pathlib import Path
from tkinter import filedialog
from typing import Optional, List
from datetime import datetime

from laser_trim_analyzer.utils.threads import get_thread_manager

from laser_trim_analyzer.core.processor import Processor
from laser_trim_analyzer.core.models import AnalysisResult, ProcessingStatus, AnalysisStatus
from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.export import export_batch_results, generate_batch_export_filename, ExcelExportError

logger = logging.getLogger(__name__)


class ProcessPage(ctk.CTkFrame):
    """
    Process page for importing and processing files.

    Features:
    - File and folder selection
    - Incremental mode toggle
    - Progress bar with real-time updates
    - Database save option
    - Processing results summary
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.selected_files: List[Path] = []
        self.processor: Optional[Processor] = None
        self.is_processing = False
        # Keep lightweight result summaries instead of full results to prevent memory issues
        # Full results are saved to database immediately, not kept in memory
        self.results: List[AnalysisResult] = []
        self._result_count = 0
        self._pass_count = 0
        self._warning_count = 0
        self._fail_count = 0
        self._error_count = 0
        self._anomaly_count = 0
        self._last_ui_update = 0  # Throttle UI updates
        self._processing_start_time = 0  # Track batch start time for ETA calculation
        self._date_filter: Optional[datetime] = None  # Date filter for file selection
        self._is_scanning = False  # Track folder scan state for cancellation

        self._create_ui()

    def _create_ui(self):
        """Create the process page UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Header
        header = ctk.CTkLabel(
            self,
            text="Process Files",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Options frame
        options_frame = ctk.CTkFrame(self)
        options_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))

        # Incremental mode toggle
        self.incremental_var = ctk.BooleanVar(value=True)
        incremental_check = ctk.CTkCheckBox(
            options_frame,
            text="Incremental Mode (only process new files)",
            variable=self.incremental_var,
            command=self._on_incremental_toggle
        )
        incremental_check.pack(side="left", padx=15, pady=15)

        # Save to database toggle
        self.save_db_var = ctk.BooleanVar(value=True)
        save_db_check = ctk.CTkCheckBox(
            options_frame,
            text="Save to Database",
            variable=self.save_db_var
        )
        save_db_check.pack(side="left", padx=15, pady=15)

        # Date filter (packed to right side of options frame)
        date_filter_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
        date_filter_frame.pack(side="right", padx=15, pady=15)

        self.clear_date_btn = ctk.CTkButton(
            date_filter_frame,
            text="X",
            width=28,
            command=self._clear_date_filter,
            fg_color="gray",
            hover_color="darkgray"
        )
        self.clear_date_btn.pack(side="right")

        self.date_filter_var = ctk.StringVar(value="")
        self.date_filter_entry = ctk.CTkEntry(
            date_filter_frame,
            textvariable=self.date_filter_var,
            placeholder_text="MM/DD/YYYY",
            width=110
        )
        self.date_filter_entry.pack(side="right", padx=(0, 5))

        date_label = ctk.CTkLabel(
            date_filter_frame,
            text="Process files from:",
            font=ctk.CTkFont(size=12)
        )
        date_label.pack(side="right", padx=(0, 5))

        # File selection area
        file_frame = ctk.CTkFrame(self)
        file_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))

        select_file_btn = ctk.CTkButton(
            file_frame,
            text="Select File(s)",
            command=self._select_files
        )
        select_file_btn.pack(side="left", padx=15, pady=15)

        select_folder_btn = ctk.CTkButton(
            file_frame,
            text="Select Folder",
            command=self._select_folder
        )
        select_folder_btn.pack(side="left", padx=5, pady=15)

        self.file_count_label = ctk.CTkLabel(
            file_frame,
            text="No files selected",
            text_color="gray"
        )
        self.file_count_label.pack(side="left", padx=15, pady=15)

        # Process button
        self.process_btn = ctk.CTkButton(
            file_frame,
            text="Start Processing",
            command=self._start_processing,
            state="disabled",
            fg_color="green",
            hover_color="darkgreen"
        )
        self.process_btn.pack(side="right", padx=15, pady=15)

        # Cancel button
        self.cancel_btn = ctk.CTkButton(
            file_frame,
            text="Cancel",
            command=self._cancel_processing,
            state="disabled",
            fg_color="red",
            hover_color="darkred"
        )
        self.cancel_btn.pack(side="right", padx=5, pady=15)

        # Export button
        self.export_btn = ctk.CTkButton(
            file_frame,
            text="📄 Export",
            command=self._export_results,
            state="disabled",
            width=100
        )
        self.export_btn.pack(side="right", padx=5, pady=15)

        # Content frame - scrollable for smaller screens
        content = ctk.CTkScrollableFrame(self)
        content.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 20))
        content.grid_columnconfigure(0, weight=1, uniform="col")  # File list
        content.grid_columnconfigure(1, weight=1, uniform="col")  # Results
        content.grid_rowconfigure(0, weight=0)  # Progress bar row
        content.grid_rowconfigure(1, weight=1, minsize=300)  # Main content row with min height

        # Progress frame
        progress_frame = ctk.CTkFrame(content)
        progress_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        self.progress_label = ctk.CTkLabel(
            progress_frame,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack(padx=15, pady=(10, 5), anchor="w")

        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=15, pady=(0, 10))
        self.progress_bar.set(0)

        # File list area
        list_frame = ctk.CTkFrame(content)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        list_label = ctk.CTkLabel(
            list_frame,
            text="Selected Files",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        list_label.pack(padx=15, pady=(15, 5), anchor="w")

        self.file_list = ctk.CTkTextbox(list_frame, height=150)
        self.file_list.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.file_list.configure(state="disabled")

        # Results area
        results_frame = ctk.CTkFrame(content)
        results_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        results_label = ctk.CTkLabel(
            results_frame,
            text="Processing Results",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        results_label.pack(padx=15, pady=(15, 5), anchor="w")

        self.results_text = ctk.CTkTextbox(results_frame, height=150)
        self.results_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.results_text.configure(state="disabled")
        self._update_results("Ready to process files.\n\nSelect files and click 'Start Processing' to begin.")

    def _on_incremental_toggle(self):
        """Called when incremental mode checkbox is toggled."""
        if not self.selected_files:
            return

        if self.incremental_var.get():
            # Incremental mode turned ON - check how many files need processing
            self._check_incremental_count()
        else:
            # Incremental mode turned OFF - show total count
            self._update_file_count_label()

    def _parse_date_filter(self) -> Optional[datetime]:
        """Parse the date filter entry value."""
        date_str = self.date_filter_var.get().strip()
        if not date_str:
            return None

        for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%m/%d/%y']:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Invalid date filter format: {date_str}")
        return None

    def _apply_date_filter(self, files: List[Path]) -> List[Path]:
        """Filter files by modification date. Uses os.stat for speed."""
        if self._date_filter is None:
            return files

        cutoff = self._date_filter
        filtered = []
        for f in files:
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime >= cutoff:
                    filtered.append(f)
            except OSError:
                filtered.append(f)  # Include if we can't stat

        return filtered

    def _clear_date_filter(self):
        """Clear the date filter and refresh file count."""
        self.date_filter_var.set("")
        self._date_filter = None
        if self.selected_files:
            self._update_file_count_label()
            if self.incremental_var.get():
                self._check_incremental_count()

    def _select_files(self):
        """Open file dialog to select files."""
        files = filedialog.askopenfilenames(
            title="Select Excel Files",
            filetypes=[("Excel files", "*.xls *.xlsx"), ("All files", "*.*")]
        )
        if files:
            self._date_filter = self._parse_date_filter()
            selected = [Path(f) for f in files]
            if self._date_filter:
                selected = self._apply_date_filter(selected)
            self.selected_files = selected
            self._update_file_list()
            # Check incremental count if mode is on
            if self.incremental_var.get():
                self._check_incremental_count()

    def _select_folder(self):
        """Open folder dialog to select a folder (crawls all subfolders)."""
        folder = filedialog.askdirectory(title="Select Folder with Excel Files")
        if folder:
            folder_path = Path(folder)

            # Parse date filter on main thread (UI variable access)
            self._date_filter = self._parse_date_filter()
            self._is_scanning = True

            # Show loading indicator
            date_info = f" (from {self._date_filter.strftime('%m/%d/%Y')})" if self._date_filter else ""
            self.file_count_label.configure(text=f"Scanning folder{date_info}...")
            self.file_list.configure(state="normal")
            self.file_list.delete("1.0", "end")
            self.file_list.insert("end", f"Scanning {folder_path.name}...\nThis may take a moment for large folders.")
            self.file_list.configure(state="disabled")
            self.process_btn.configure(state="disabled")
            self.cancel_btn.configure(state="normal")

            # Capture date filter for background thread (avoid accessing UI vars from thread)
            # Convert to timestamp once for fast comparison (avoids datetime objects per file)
            date_cutoff_ts = self._date_filter.timestamp() if self._date_filter else None

            # Run file discovery in background thread to avoid UI freeze
            def discover_files():
                try:
                    all_files = []
                    last_progress_time = time.monotonic()

                    # Single recursive walk using os.scandir for efficiency:
                    # 1. Walks the tree ONCE (old code did two rglob passes)
                    # 2. Matches both .xls and .xlsx in one pass
                    # 3. Date filter uses DirEntry.stat() which is cached on Windows
                    #    (no extra network round-trip per file on network drives)
                    # 4. Reports progress to UI periodically
                    dirs_to_scan = [str(folder_path)]
                    while dirs_to_scan:
                        if not self._is_scanning:
                            self.after(0, self._on_folder_scan_cancelled)
                            return

                        current_dir = dirs_to_scan.pop()
                        try:
                            with os.scandir(current_dir) as entries:
                                for entry in entries:
                                    if not self._is_scanning:
                                        self.after(0, self._on_folder_scan_cancelled)
                                        return

                                    try:
                                        if entry.is_dir(follow_symlinks=False):
                                            dirs_to_scan.append(entry.path)
                                        elif entry.is_file(follow_symlinks=False):
                                            name_lower = entry.name.lower()
                                            if name_lower.endswith('.xls') or name_lower.endswith('.xlsx'):
                                                # Date filter using DirEntry.stat() - cached on Windows
                                                if date_cutoff_ts:
                                                    try:
                                                        if entry.stat().st_mtime < date_cutoff_ts:
                                                            continue
                                                    except OSError:
                                                        pass  # Include if we can't stat
                                                all_files.append(Path(entry.path))
                                    except OSError:
                                        continue
                        except (PermissionError, OSError):
                            continue

                        # Progress update every ~1 second
                        now = time.monotonic()
                        if now - last_progress_time >= 1.0:
                            last_progress_time = now
                            count = len(all_files)
                            remaining = len(dirs_to_scan)
                            self.after(0, lambda c=count, r=remaining:
                                       self.file_count_label.configure(
                                           text=f"Scanning... {c:,} files found ({r:,} folders queued)"))

                    # Sort by path for consistent ordering (groups files by subfolder)
                    all_files.sort(key=lambda f: (f.parent, f.name))

                    self._is_scanning = False
                    # Update UI on main thread
                    self.after(0, lambda: self._on_folder_scanned(all_files))
                except Exception as e:
                    self._is_scanning = False
                    logger.error(f"Error scanning folder: {e}")
                    self.after(0, lambda: self._on_folder_scan_error(str(e)))

            get_thread_manager().start_thread(target=discover_files, name="folder-scan")

    def _on_folder_scanned(self, files: list):
        """Called when folder scan completes - updates UI with found files."""
        self.selected_files = files
        self.cancel_btn.configure(state="disabled")
        self._update_file_list()

        # If incremental mode is ON, check how many files are already processed
        if self.incremental_var.get() and len(files) > 0:
            self._check_incremental_count()

    def _check_incremental_count(self):
        """Check how many files are already processed (for incremental mode display)."""
        if not self.selected_files:
            return

        # Show checking status
        total = len(self.selected_files)
        self.file_count_label.configure(text=f"Checking {total:,} files against database...")

        def check_files():
            try:
                # Create temporary processor, clear cache after to free memory
                processor = Processor(use_ml=False)
                unprocessed, already_processed = processor.get_unprocessed_count(
                    self.selected_files, clear_cache=True
                )
                self.after(0, lambda: self._on_incremental_count_ready(unprocessed, already_processed))
            except Exception as e:
                logger.error(f"Error checking incremental count: {e}")
                # Fall back to showing total count
                self.after(0, lambda: self._update_file_count_label())

        get_thread_manager().start_thread(target=check_files, name="incremental-check")

    def _on_incremental_count_ready(self, unprocessed: int, already_processed: int):
        """Called when incremental count check completes."""
        total = len(self.selected_files)
        unique_folders = len(set(f.parent for f in self.selected_files))
        folder_info = f" in {unique_folders} folder(s)" if unique_folders > 1 else ""
        date_info = f" from {self._date_filter.strftime('%m/%d/%Y')}" if self._date_filter else ""

        if already_processed > 0:
            self.file_count_label.configure(
                text=f"{unprocessed} new files to process ({already_processed} already in DB){date_info}{folder_info}"
            )
        else:
            self.file_count_label.configure(
                text=f"{total} files selected{date_info}{folder_info}"
            )

    def _update_file_count_label(self):
        """Update file count label with basic info (no incremental check)."""
        if not self.selected_files:
            self.file_count_label.configure(text="No files selected")
            return

        total = len(self.selected_files)
        unique_folders = len(set(f.parent for f in self.selected_files))
        folder_info = f" in {unique_folders} folder(s)" if unique_folders > 1 else ""
        date_info = f" from {self._date_filter.strftime('%m/%d/%Y')}" if self._date_filter else ""
        self.file_count_label.configure(text=f"{total} files selected{date_info}{folder_info}")

    def _on_folder_scan_error(self, error: str):
        """Called when folder scan fails."""
        self.file_list.configure(state="normal")
        self.file_list.delete("1.0", "end")
        self.file_list.insert("end", f"Error scanning folder: {error}")
        self.file_list.configure(state="disabled")
        self.file_count_label.configure(text="Scan failed")
        self.cancel_btn.configure(state="disabled")

    def _on_folder_scan_cancelled(self):
        """Called when folder scan is cancelled by user."""
        self.file_list.configure(state="normal")
        self.file_list.delete("1.0", "end")
        self.file_list.insert("end", "Scan cancelled.")
        self.file_list.configure(state="disabled")
        self.file_count_label.configure(text="Scan cancelled")
        self.cancel_btn.configure(state="disabled")

    def _update_file_list(self):
        """Update the file list display."""
        self.file_list.configure(state="normal")
        self.file_list.delete("1.0", "end")

        if self.selected_files:
            # Group files by parent folder for better display
            current_folder = None
            display_count = 0
            max_display = 100  # Show more files since we now have subfolders

            for f in self.selected_files:
                if display_count >= max_display:
                    break

                # Show folder header when it changes
                if f.parent != current_folder:
                    current_folder = f.parent
                    folder_name = current_folder.name
                    if display_count > 0:
                        self.file_list.insert("end", "\n")
                    self.file_list.insert("end", f"📁 {folder_name}/\n")

                self.file_list.insert("end", f"   {f.name}\n")
                display_count += 1

            if len(self.selected_files) > max_display:
                self.file_list.insert("end", f"\n... and {len(self.selected_files) - max_display} more files")

            # Count unique folders
            unique_folders = len(set(f.parent for f in self.selected_files))
            folder_info = f" in {unique_folders} folder(s)" if unique_folders > 1 else ""
            self.file_count_label.configure(text=f"{len(self.selected_files)} files selected{folder_info}")
            self.process_btn.configure(state="normal")
        else:
            self.file_list.insert("end", "No files selected")
            self.file_count_label.configure(text="No files selected")
            self.process_btn.configure(state="disabled")

        self.file_list.configure(state="disabled")

    def _update_results(self, text: str):
        """Update the results display."""
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", text)
        self.results_text.configure(state="disabled")

    def _append_result(self, text: str):
        """Append to results display."""
        self.results_text.configure(state="normal")
        self.results_text.insert("end", text)
        self.results_text.see("end")
        self.results_text.configure(state="disabled")

    def _format_eta(self, seconds: float) -> str:
        """Format seconds into human-readable ETA string."""
        if seconds < 0 or seconds > 86400 * 30:  # Cap at 30 days
            return "calculating..."

        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days}d {hours}h"

    def _start_processing(self):
        """Start processing selected files in a background thread."""
        if not self.selected_files or self.is_processing:
            return

        self.is_processing = True
        # Reset counters - don't accumulate full results in memory for large batches
        self.results = []
        self._result_count = 0
        self._pass_count = 0
        self._warning_count = 0
        self._fail_count = 0
        self._error_count = 0
        self._anomaly_count = 0
        self._last_ui_update = 0

        # Update UI state
        self.process_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Initializing processor...")

        self._update_results(
            f"Processing {len(self.selected_files)} files...\n"
            f"Incremental mode: {'ON' if self.incremental_var.get() else 'OFF'}\n"
            f"Save to database: {'ON' if self.save_db_var.get() else 'OFF'}\n"
            f"Started: {datetime.now().strftime('%H:%M:%S')}\n"
            f"{'-' * 40}\n"
        )

        # Capture UI state on main thread for thread safety
        self._save_to_db = self.save_db_var.get()

        # Start processing in background thread (tracked for graceful shutdown)
        get_thread_manager().start_thread(target=self._process_files, name="file-processing")

    def _process_files(self):
        """Process files in background thread."""
        import gc

        try:
            # Initialize processor
            self.processor = Processor()

            # Record start time for ETA calculation
            self._processing_start_time = time.time()

            # Determine if this is a large batch (affects memory strategy)
            is_large_batch = len(self.selected_files) > 100
            total_files = len(self.selected_files)

            # Process callback with throttling for large batches
            def on_progress(status: ProcessingStatus):
                # Throttle UI updates for large batches (max 10 updates/sec)
                current_time = time.time()
                if is_large_batch and (current_time - self._last_ui_update) < 0.1:
                    # Skip UI update but still count results
                    if status.status == "completed" and status.result:
                        return  # Will be counted in main loop
                    return

                self._last_ui_update = current_time
                # Schedule UI update on main thread
                self.after(0, lambda s=status: self._on_progress_update(s))

            # Process batch
            incremental = self.incremental_var.get()
            gc_interval = 100  # Force GC every 100 files for large batches

            for i, result in enumerate(self.processor.process_batch(
                self.selected_files,
                progress_callback=on_progress,
                incremental=incremental
            )):
                if not self.is_processing:
                    break  # User cancelled

                # Update counters
                self._result_count += 1
                if result.overall_status == AnalysisStatus.PASS:
                    self._pass_count += 1
                elif result.overall_status == AnalysisStatus.WARNING:
                    self._warning_count += 1
                elif result.overall_status == AnalysisStatus.FAIL:
                    self._fail_count += 1
                else:
                    self._error_count += 1

                # Count anomalies (trim failures with linear slope pattern)
                if any(getattr(t, 'is_anomaly', False) for t in result.tracks):
                    self._anomaly_count += 1

                # Save to database if enabled (captured on main thread)
                if self._save_to_db:
                    try:
                        db = get_database()
                        db.save_analysis(result)
                    except Exception as e:
                        logger.error(f"Failed to save to database: {e}")

                # For small batches, keep results for export
                # For large batches, only keep last 50 to prevent memory issues
                if is_large_batch:
                    if len(self.results) >= 50:
                        self.results.pop(0)  # Remove oldest
                    self.results.append(result)

                    # Periodic garbage collection for large batches
                    if (i + 1) % gc_interval == 0:
                        gc.collect()
                        logger.debug(f"GC after {i + 1} files processed")
                else:
                    self.results.append(result)

            self.after(0, self._on_processing_complete)

        except Exception as e:
            logger.exception(f"Processing error: {e}")
            self.after(0, lambda: self._on_processing_error(str(e)))

    def _on_progress_update(self, status: ProcessingStatus):
        """Handle progress update (called on main thread)."""
        if not self.is_processing:
            return

        # Update progress bar
        self.progress_bar.set(status.progress_percent / 100)

        # Handle scanning status (incremental mode file checking)
        if status.status == "scanning":
            self.progress_label.configure(text=status.message)
            self._append_result(f"[INFO] {status.message}\n")
            return

        # Calculate ETA from elapsed time and completed count
        total_files = len(self.selected_files)
        completed = self._result_count
        elapsed = time.time() - self._processing_start_time

        if elapsed > 2 and completed > 0:  # Wait 2 seconds for stable rate
            rate = completed / elapsed
            remaining = (total_files - completed) / rate if rate > 0 else 0
            eta_str = self._format_eta(remaining)
            rate_str = f"{rate:.1f}/sec"

            self.progress_label.configure(
                text=f"Processing: {completed:,} / {total_files:,} ({status.progress_percent:.1f}%) | ~{eta_str} remaining | {rate_str}"
            )
        else:
            # Early in processing, show simple progress
            self.progress_label.configure(
                text=f"Processing: {status.filename} ({status.progress_percent:.0f}%)"
            )

        # Update results
        if status.status == "completed" and status.result:
            result = status.result
            # Map status to display string
            if result.overall_status == AnalysisStatus.PASS:
                status_str = "PASS"
            elif result.overall_status == AnalysisStatus.WARNING:
                status_str = "WARN"
            elif result.overall_status == AnalysisStatus.FAIL:
                status_str = "FAIL"
            else:
                status_str = "ERROR"
            self._append_result(
                f"[{status_str}] {status.filename} - "
                f"{len(result.tracks)} track(s), "
                f"{result.processing_time:.2f}s\n"
            )
        elif status.status == "skipped":
            self._append_result(f"[SKIP] {status.filename} - Already processed\n")
        elif status.status == "failed":
            self._append_result(f"[ERROR] {status.filename} - {status.message}\n")

    def _on_processing_complete(self):
        """Handle processing completion."""
        self.is_processing = False

        # Update UI state
        self.process_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.export_btn.configure(state="normal" if self.results else "disabled")
        self.progress_bar.set(1)
        self.progress_label.configure(text="Processing complete!")

        # Use counters for summary (self.results may be truncated for large batches)
        total = self._result_count
        passed = self._pass_count
        warnings = self._warning_count
        failed = self._fail_count
        errors = self._error_count
        anomalies = self._anomaly_count
        pass_rate = (passed / total * 100) if total > 0 else 0

        # Note about large batch export limitations
        is_large_batch = len(self.selected_files) > 100
        export_note = ""
        if is_large_batch and self.results:
            export_note = f"\n  Note: Export contains last {len(self.results)} files only (use Analyze page for full data)\n"

        # Anomaly note if any detected
        anomaly_note = ""
        if anomalies > 0:
            anomaly_note = f"  Anomalies: {anomalies} (trim failures excluded from stats)\n"

        # Append summary
        self._append_result(
            f"\n{'-' * 40}\n"
            f"SUMMARY:\n"
            f"  Total processed: {total}\n"
            f"  Passed: {passed}\n"
            f"  Warnings: {warnings} (partial pass)\n"
            f"  Failed: {failed}\n"
            f"  Errors: {errors}\n"
            f"{anomaly_note}"
            f"  Pass rate: {pass_rate:.1f}%{export_note}"
            f"Completed: {datetime.now().strftime('%H:%M:%S')}\n"
        )

        logger.info(f"Processing complete: {total} files, {passed} passed, {warnings} warnings, {failed} failed, {anomalies} anomalies")

    def _on_processing_error(self, error: str):
        """Handle processing error."""
        self.is_processing = False

        self.process_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.progress_label.configure(text="Processing failed!")

        self._append_result(f"\n[ERROR] Processing failed: {error}\n")

        logger.error(f"Processing failed: {error}")

    def _cancel_processing(self):
        """Cancel processing or folder scanning."""
        if self.is_processing:
            self.is_processing = False
            self.cancel_btn.configure(state="disabled")
            self.progress_label.configure(text="Cancelling...")
            self._append_result("\n[CANCELLED] Processing cancelled by user.\n")
        elif self._is_scanning:
            self._is_scanning = False  # Signal background thread to stop

    def _export_results(self):
        """Export batch results to Excel."""
        if not self.results:
            return

        # Generate default filename
        default_name = generate_batch_export_filename(self.results)

        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Export Batch Results to Excel",
            defaultextension=".xlsx",
            initialfile=default_name,
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if not file_path:
            return  # User cancelled

        try:
            self.progress_label.configure(text="Exporting...")
            output_path = export_batch_results(self.results, file_path)
            self.progress_label.configure(text=f"Exported: {output_path.name}")
            self._append_result(f"\n[EXPORT] Saved to: {output_path}\n")
            logger.info(f"Exported batch results to: {output_path}")
        except ExcelExportError as e:
            self.progress_label.configure(text=f"Export failed")
            self._append_result(f"\n[ERROR] Export failed: {e}\n")
            logger.error(f"Batch export failed: {e}")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Process page shown")
