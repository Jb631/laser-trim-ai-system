"""
Process Page - Import and process files.

Handles single file and batch processing with incremental mode.
Wired to the actual Processor for real analysis.
"""

import threading
import customtkinter as ctk
import logging
from pathlib import Path
from tkinter import filedialog
from typing import Optional, List
from datetime import datetime

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
        self._last_ui_update = 0  # Throttle UI updates

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
            variable=self.incremental_var
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

    def _select_files(self):
        """Open file dialog to select files."""
        files = filedialog.askopenfilenames(
            title="Select Excel Files",
            filetypes=[("Excel files", "*.xls *.xlsx"), ("All files", "*.*")]
        )
        if files:
            self.selected_files = [Path(f) for f in files]
            self._update_file_list()

    def _select_folder(self):
        """Open folder dialog to select a folder (crawls all subfolders)."""
        folder = filedialog.askdirectory(title="Select Folder with Excel Files")
        if folder:
            folder_path = Path(folder)
            # Find all Excel files recursively in folder and all subfolders
            # Using rglob for recursive search - supports master folder with model subfolders
            xls_files = list(folder_path.rglob("*.xls"))
            xlsx_files = list(folder_path.rglob("*.xlsx"))
            self.selected_files = xls_files + xlsx_files
            # Sort by path for consistent ordering (groups files by subfolder)
            self.selected_files.sort(key=lambda f: (f.parent, f.name))
            self._update_file_list()

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

        # Start processing in background thread
        thread = threading.Thread(target=self._process_files, daemon=True)
        thread.start()

    def _process_files(self):
        """Process files in background thread."""
        import time
        import gc

        try:
            # Initialize processor
            self.processor = Processor()

            # Determine if this is a large batch (affects memory strategy)
            is_large_batch = len(self.selected_files) > 100

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

                # Save to database if enabled
                if self.save_db_var.get():
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

        # Update label
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
        pass_rate = (passed / total * 100) if total > 0 else 0

        # Note about large batch export limitations
        is_large_batch = len(self.selected_files) > 100
        export_note = ""
        if is_large_batch and self.results:
            export_note = f"\n  Note: Export contains last {len(self.results)} files only (use Analyze page for full data)\n"

        # Append summary
        self._append_result(
            f"\n{'-' * 40}\n"
            f"SUMMARY:\n"
            f"  Total processed: {total}\n"
            f"  Passed: {passed}\n"
            f"  Warnings: {warnings} (partial pass)\n"
            f"  Failed: {failed}\n"
            f"  Errors: {errors}\n"
            f"  Pass rate: {pass_rate:.1f}%{export_note}"
            f"Completed: {datetime.now().strftime('%H:%M:%S')}\n"
        )

        logger.info(f"Processing complete: {total} files, {passed} passed, {warnings} warnings, {failed} failed")

    def _on_processing_error(self, error: str):
        """Handle processing error."""
        self.is_processing = False

        self.process_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.progress_label.configure(text="Processing failed!")

        self._append_result(f"\n[ERROR] Processing failed: {error}\n")

        logger.error(f"Processing failed: {error}")

    def _cancel_processing(self):
        """Cancel processing."""
        if self.is_processing:
            self.is_processing = False
            self.progress_label.configure(text="Cancelling...")
            self._append_result("\n[CANCELLED] Processing cancelled by user.\n")

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
