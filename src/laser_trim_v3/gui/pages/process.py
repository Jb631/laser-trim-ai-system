"""
Process Page - Import and process files.

Handles single file and batch processing with incremental mode.
"""

import customtkinter as ctk
import logging
from pathlib import Path
from tkinter import filedialog

logger = logging.getLogger(__name__)


class ProcessPage(ctk.CTkFrame):
    """
    Process page for importing and processing files.

    Features:
    - File drop zone (drag & drop)
    - Folder selector (batch processing)
    - Incremental mode toggle
    - Progress bar with file count
    - Recent results summary
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.selected_files: list[Path] = []

        self._create_ui()

    def _create_ui(self):
        """Create the process page UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

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

        # Generate plots toggle
        self.plots_var = ctk.BooleanVar(value=True)
        plots_check = ctk.CTkCheckBox(
            options_frame,
            text="Generate Plots",
            variable=self.plots_var
        )
        plots_check.pack(side="left", padx=15, pady=15)

        # Content frame
        content = ctk.CTkFrame(self)
        content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        content.grid_columnconfigure((0, 1), weight=1)
        content.grid_rowconfigure(1, weight=1)

        # File selection area
        file_frame = ctk.CTkFrame(content)
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

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

        # File list area
        list_frame = ctk.CTkFrame(content)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        list_label = ctk.CTkLabel(
            list_frame,
            text="Selected Files",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        list_label.pack(padx=15, pady=(15, 5), anchor="w")

        self.file_list = ctk.CTkTextbox(list_frame, height=200)
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

        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.results_text.configure(state="disabled")
        self._update_results("No processing results yet.\n\nSelect files and click 'Start Processing' to begin.")

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

    def _start_processing(self):
        """Start processing selected files."""
        if not self.selected_files:
            return

        # TODO: Implement actual processing
        self._update_results(
            f"Processing {len(self.selected_files)} files...\n\n"
            f"Incremental mode: {'ON' if self.incremental_var.get() else 'OFF'}\n"
            f"Generate plots: {'ON' if self.plots_var.get() else 'OFF'}\n\n"
            "Processing not yet implemented in v3."
        )

        logger.info(f"Would process {len(self.selected_files)} files")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Process page shown")
