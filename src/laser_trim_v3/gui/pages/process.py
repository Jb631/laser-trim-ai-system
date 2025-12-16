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

from laser_trim_v3.core.processor import Processor
from laser_trim_v3.core.models import AnalysisResult, ProcessingStatus, AnalysisStatus
from laser_trim_v3.database import get_database
from laser_trim_v3.export import export_batch_results, generate_batch_export_filename, ExcelExportError

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
        self.results: List[AnalysisResult] = []

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

        # Content frame
        content = ctk.CTkFrame(self)
        content.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 20))
        content.grid_columnconfigure(0, weight=1, uniform="col")  # File list
        content.grid_columnconfigure(1, weight=1, uniform="col")  # Results
        content.grid_rowconfigure(0, weight=0)  # Progress bar row
        content.grid_rowconfigure(1, weight=1)  # Main content row

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
        """Open folder dialog to select a folder."""
        folder = filedialog.askdirectory(title="Select Folder with Excel Files")
        if folder:
            folder_path = Path(folder)
            # Find all Excel files in folder
            self.selected_files = list(folder_path.glob("*.xls")) + list(folder_path.glob("*.xlsx"))
            self._update_file_list()

    def _update_file_list(self):
        """Update the file list display."""
        self.file_list.configure(state="normal")
        self.file_list.delete("1.0", "end")

        if self.selected_files:
            for f in self.selected_files[:50]:  # Show first 50
                self.file_list.insert("end", f"{f.name}\n")
            if len(self.selected_files) > 50:
                self.file_list.insert("end", f"\n... and {len(self.selected_files) - 50} more files")

            self.file_count_label.configure(text=f"{len(self.selected_files)} files selected")
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
        self.results = []

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
        try:
            # Initialize processor
            self.processor = Processor()

            # Process callback
            def on_progress(status: ProcessingStatus):
                # Schedule UI update on main thread
                self.after(0, lambda s=status: self._on_progress_update(s))

            # Process batch
            incremental = self.incremental_var.get()

            for result in self.processor.process_batch(
                self.selected_files,
                progress_callback=on_progress,
                incremental=incremental
            ):
                if not self.is_processing:
                    break  # User cancelled

                self.results.append(result)

                # Save to database if enabled
                if self.save_db_var.get():
                    try:
                        db = get_database()
                        db.save_analysis(result)
                    except Exception as e:
                        logger.error(f"Failed to save to database: {e}")

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

        # Update label
        self.progress_label.configure(
            text=f"Processing: {status.filename} ({status.progress_percent:.0f}%)"
        )

        # Update results
        if status.status == "completed" and status.result:
            result = status.result
            status_str = "PASS" if result.overall_status == AnalysisStatus.PASS else "FAIL"
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

        # Calculate summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r.overall_status == AnalysisStatus.PASS)
        warnings = sum(1 for r in self.results if r.overall_status == AnalysisStatus.WARNING)
        failed = sum(1 for r in self.results if r.overall_status == AnalysisStatus.FAIL)
        errors = sum(1 for r in self.results if r.overall_status == AnalysisStatus.ERROR)
        pass_rate = (passed / total * 100) if total > 0 else 0

        # Append summary
        self._append_result(
            f"\n{'-' * 40}\n"
            f"SUMMARY:\n"
            f"  Total processed: {total}\n"
            f"  Passed: {passed}\n"
            f"  Warnings: {warnings} (partial pass)\n"
            f"  Failed: {failed}\n"
            f"  Errors: {errors}\n"
            f"  Pass rate: {pass_rate:.1f}%\n"
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
