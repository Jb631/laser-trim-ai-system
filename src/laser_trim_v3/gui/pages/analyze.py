"""
Analyze Page - View and compare results.

Supports single file analysis, track comparison, and final line comparison.
"""

import customtkinter as ctk
import logging

logger = logging.getLogger(__name__)


class AnalyzePage(ctk.CTkFrame):
    """
    Analyze page for viewing and comparing results.

    Sub-modes:
    - Single File: Detailed analysis with charts
    - Track Compare: Side-by-side TA vs TB
    - Final Line: Compare to final test data
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        self._create_ui()

    def _create_ui(self):
        """Create the analyze page UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkLabel(
            self,
            text="Analyze Results",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Tab view for different analysis modes
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

        # Create tabs
        self.tabview.add("Single File")
        self.tabview.add("Track Compare")
        self.tabview.add("Final Line")

        # Single File tab content
        self._create_single_file_tab()

        # Track Compare tab content
        self._create_track_compare_tab()

        # Final Line tab content
        self._create_final_line_tab()

    def _create_single_file_tab(self):
        """Create the single file analysis tab."""
        tab = self.tabview.tab("Single File")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # File selection
        select_frame = ctk.CTkFrame(tab)
        select_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        select_btn = ctk.CTkButton(
            select_frame,
            text="Select File",
            command=self._select_single_file
        )
        select_btn.pack(side="left", padx=15, pady=15)

        self.single_file_label = ctk.CTkLabel(
            select_frame,
            text="No file selected",
            text_color="gray"
        )
        self.single_file_label.pack(side="left", padx=15, pady=15)

        # Results area
        results_frame = ctk.CTkFrame(tab)
        results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)

        placeholder = ctk.CTkLabel(
            results_frame,
            text="Select a file to analyze\n\nResults will appear here with:\n- PASS/FAIL status\n- Sigma gradient analysis\n- Linearity analysis\n- Error vs Position chart",
            text_color="gray",
            justify="center"
        )
        placeholder.grid(row=0, column=0, padx=20, pady=20)

    def _create_track_compare_tab(self):
        """Create the track comparison tab."""
        tab = self.tabview.tab("Track Compare")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        placeholder = ctk.CTkLabel(
            tab,
            text="Track Comparison\n\nCompare TA vs TB tracks on the same unit.\n\nNot yet implemented in v3.",
            text_color="gray",
            justify="center"
        )
        placeholder.grid(row=0, column=0, padx=20, pady=20)

    def _create_final_line_tab(self):
        """Create the final line comparison tab."""
        tab = self.tabview.tab("Final Line")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        placeholder = ctk.CTkLabel(
            tab,
            text="Final Line Comparison\n\nCompare trim data to final test data.\n\nNot yet implemented in v3.",
            text_color="gray",
            justify="center"
        )
        placeholder.grid(row=0, column=0, padx=20, pady=20)

    def _select_single_file(self):
        """Select a single file for analysis."""
        from tkinter import filedialog
        file = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xls *.xlsx"), ("All files", "*.*")]
        )
        if file:
            self.single_file_label.configure(text=file.split("/")[-1])
            # TODO: Implement actual analysis
            logger.info(f"Would analyze file: {file}")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Analyze page shown")
