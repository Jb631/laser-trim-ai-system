"""
Analyze Page - View and compare results.

Supports single file analysis, track comparison, and final line comparison.
Wired to the actual Processor and ChartWidget for real analysis.
"""

import threading
import customtkinter as ctk
import logging
from pathlib import Path
from tkinter import filedialog
from typing import Optional, List

from laser_trim_v3.core.processor import Processor
from laser_trim_v3.core.models import AnalysisResult, AnalysisStatus, TrackData
from laser_trim_v3.gui.widgets.chart import ChartWidget, ChartStyle
from laser_trim_v3.database import get_database
from laser_trim_v3.export import export_single_result, generate_export_filename, ExcelExportError

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
        self.current_result: Optional[AnalysisResult] = None
        self.processor: Optional[Processor] = None

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
        tab.grid_columnconfigure(1, weight=2)
        tab.grid_rowconfigure(1, weight=1)

        # File selection row
        select_frame = ctk.CTkFrame(tab)
        select_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

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

        # Analyze button
        self.analyze_btn = ctk.CTkButton(
            select_frame,
            text="Analyze",
            command=self._analyze_file,
            state="disabled",
            fg_color="green",
            hover_color="darkgreen"
        )
        self.analyze_btn.pack(side="left", padx=15, pady=15)

        # Save to database checkbox
        self.save_db_var = ctk.BooleanVar(value=True)
        save_db_check = ctk.CTkCheckBox(
            select_frame,
            text="Save to Database",
            variable=self.save_db_var
        )
        save_db_check.pack(side="left", padx=15, pady=15)

        # Status label
        self.status_label = ctk.CTkLabel(
            select_frame,
            text="",
            text_color="gray"
        )
        self.status_label.pack(side="right", padx=15, pady=15)

        # Export button
        self.export_btn = ctk.CTkButton(
            select_frame,
            text="📄 Export",
            command=self._export_result,
            state="disabled",
            width=100
        )
        self.export_btn.pack(side="right", padx=5, pady=15)

        # Results panel (left side)
        results_frame = ctk.CTkFrame(tab)
        results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(1, weight=1)

        results_label = ctk.CTkLabel(
            results_frame,
            text="Analysis Results",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        results_label.pack(padx=15, pady=(15, 5), anchor="w")

        # Status banner
        self.status_banner = ctk.CTkFrame(results_frame, height=60)
        self.status_banner.pack(fill="x", padx=15, pady=5)

        self.status_text = ctk.CTkLabel(
            self.status_banner,
            text="READY",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="gray"
        )
        self.status_text.pack(expand=True, pady=10)

        # Metrics display
        self.metrics_text = ctk.CTkTextbox(results_frame, height=300, font=ctk.CTkFont(size=12))
        self.metrics_text.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.metrics_text.configure(state="disabled")
        self._update_metrics("Select a file and click 'Analyze' to begin.")

        # Track selector (for multi-track files)
        track_select_frame = ctk.CTkFrame(results_frame)
        track_select_frame.pack(fill="x", padx=15, pady=(0, 15))

        ctk.CTkLabel(track_select_frame, text="Track:").pack(side="left", padx=5)
        self.track_selector = ctk.CTkComboBox(
            track_select_frame,
            values=["All Tracks"],
            command=self._on_track_selected,
            state="disabled"
        )
        self.track_selector.pack(side="left", padx=5, fill="x", expand=True)

        # Chart panel (right side)
        chart_frame = ctk.CTkFrame(tab)
        chart_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=(0, 10))

        chart_label = ctk.CTkLabel(
            chart_frame,
            text="Error vs Position Chart",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        chart_label.pack(padx=15, pady=(15, 5), anchor="w")

        # Chart widget
        self.chart = ChartWidget(
            chart_frame,
            style=ChartStyle(figure_size=(8, 5), dpi=100)
        )
        self.chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.chart.show_placeholder("Select a file to analyze")

        # Store selected file path
        self.selected_file: Optional[Path] = None

    def _create_track_compare_tab(self):
        """Create the track comparison tab."""
        tab = self.tabview.tab("Track Compare")
        tab.grid_columnconfigure((0, 1), weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # File selection
        select_frame = ctk.CTkFrame(tab)
        select_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        select_btn = ctk.CTkButton(
            select_frame,
            text="Select File for Comparison",
            command=self._select_compare_file
        )
        select_btn.pack(side="left", padx=15, pady=15)

        self.compare_file_label = ctk.CTkLabel(
            select_frame,
            text="No file selected",
            text_color="gray"
        )
        self.compare_file_label.pack(side="left", padx=15, pady=15)

        # Track A chart
        chart_a_frame = ctk.CTkFrame(tab)
        chart_a_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        ctk.CTkLabel(
            chart_a_frame,
            text="Track A",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(padx=15, pady=(15, 5), anchor="w")

        self.chart_a = ChartWidget(
            chart_a_frame,
            style=ChartStyle(figure_size=(6, 4), dpi=100)
        )
        self.chart_a.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.chart_a.show_placeholder("Track A data")

        # Track B chart
        chart_b_frame = ctk.CTkFrame(tab)
        chart_b_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=(0, 10))

        ctk.CTkLabel(
            chart_b_frame,
            text="Track B",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(padx=15, pady=(15, 5), anchor="w")

        self.chart_b = ChartWidget(
            chart_b_frame,
            style=ChartStyle(figure_size=(6, 4), dpi=100)
        )
        self.chart_b.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.chart_b.show_placeholder("Track B data")

        self.compare_file: Optional[Path] = None

    def _create_final_line_tab(self):
        """Create the final line comparison tab."""
        tab = self.tabview.tab("Final Line")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        placeholder = ctk.CTkLabel(
            tab,
            text="Final Line Comparison\n\nCompare trim data to final test data.\n\nComing soon in v3.",
            text_color="gray",
            justify="center"
        )
        placeholder.grid(row=0, column=0, padx=20, pady=20)

    def _select_single_file(self):
        """Select a single file for analysis."""
        file = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xls *.xlsx"), ("All files", "*.*")]
        )
        if file:
            self.selected_file = Path(file)
            self.single_file_label.configure(text=self.selected_file.name)
            self.analyze_btn.configure(state="normal")
            self.status_label.configure(text="Ready to analyze")
            logger.info(f"Selected file: {file}")

    def _select_compare_file(self):
        """Select a file for track comparison."""
        file = filedialog.askopenfilename(
            title="Select Excel File for Comparison",
            filetypes=[("Excel files", "*.xls *.xlsx"), ("All files", "*.*")]
        )
        if file:
            self.compare_file = Path(file)
            self.compare_file_label.configure(text=self.compare_file.name)
            self._analyze_for_comparison()

    def _analyze_file(self):
        """Analyze the selected file in a background thread."""
        if not self.selected_file:
            return

        # Update UI
        self.analyze_btn.configure(state="disabled")
        self.status_label.configure(text="Analyzing...")
        self.status_text.configure(text="ANALYZING...", text_color="orange")

        # Run analysis in background thread
        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _run_analysis(self):
        """Run analysis in background thread."""
        try:
            # Initialize processor if needed
            if self.processor is None:
                self.processor = Processor()

            # Process file
            result = self.processor.process_file(self.selected_file)
            self.current_result = result

            # Save to database if enabled
            if self.save_db_var.get():
                try:
                    db = get_database()
                    db.save_analysis(result)
                    logger.info(f"Saved analysis to database: {result.metadata.filename}")
                except Exception as e:
                    logger.error(f"Failed to save to database: {e}")

            # Update UI on main thread
            self.after(0, lambda: self._display_result(result))

        except Exception as e:
            logger.exception(f"Analysis error: {e}")
            self.after(0, lambda: self._on_analysis_error(str(e)))

    def _display_result(self, result: AnalysisResult):
        """Display analysis results."""
        self.analyze_btn.configure(state="normal")
        self.export_btn.configure(state="normal")

        # Update status banner
        if result.overall_status == AnalysisStatus.PASS:
            self.status_text.configure(text="PASS", text_color="#27ae60")
            self.status_banner.configure(fg_color="#1e4d2b")
        elif result.overall_status == AnalysisStatus.FAIL:
            self.status_text.configure(text="FAIL", text_color="#e74c3c")
            self.status_banner.configure(fg_color="#4d1e1e")
        elif result.overall_status == AnalysisStatus.WARNING:
            self.status_text.configure(text="WARNING", text_color="#f39c12")
            self.status_banner.configure(fg_color="#4d3d1e")
        else:
            self.status_text.configure(text="ERROR", text_color="#e74c3c")
            self.status_banner.configure(fg_color="#4d1e1e")

        self.status_label.configure(
            text=f"Completed in {result.processing_time:.2f}s"
        )

        # Update metrics display
        self._display_metrics(result)

        # Update track selector
        if result.tracks:
            track_values = ["All Tracks"] + [f"Track {t.track_id}" for t in result.tracks]
            self.track_selector.configure(values=track_values, state="normal")
            self.track_selector.set("All Tracks")

        # Display chart for first track
        if result.tracks:
            self._display_track_chart(result.tracks[0])

        logger.info(f"Analysis complete: {result.overall_status.value}")

    def _display_metrics(self, result: AnalysisResult):
        """Display analysis metrics."""
        lines = []

        # File info
        lines.append("═══ FILE INFORMATION ═══")
        lines.append(f"Filename: {result.metadata.filename}")
        lines.append(f"Model: {result.metadata.model}")
        lines.append(f"Serial: {result.metadata.serial}")
        lines.append(f"System: {result.metadata.system.value}")
        lines.append(f"Date: {result.metadata.file_date.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # Overall status
        lines.append("═══ OVERALL STATUS ═══")
        lines.append(f"Status: {result.overall_status.value}")
        lines.append(f"Tracks: {len(result.tracks)}")
        lines.append(f"Processing Time: {result.processing_time:.2f}s")
        lines.append("")

        # Track details
        for track in result.tracks:
            lines.append(f"═══ TRACK {track.track_id} ═══")
            lines.append(f"Status: {track.status.value}")
            lines.append("")
            lines.append("Sigma Analysis:")
            lines.append(f"  Gradient: {track.sigma_gradient:.6f}")
            lines.append(f"  Threshold: {track.sigma_threshold:.6f}")
            lines.append(f"  Pass: {'✓' if track.sigma_pass else '✗'}")
            lines.append("")
            lines.append("Linearity Analysis:")
            lines.append(f"  Optimal Offset: {track.optimal_offset:.6f}")
            lines.append(f"  Max Error: {track.linearity_error:.6f}")
            lines.append(f"  Fail Points: {track.linearity_fail_points}")
            lines.append(f"  Pass: {'✓' if track.linearity_pass else '✗'}")
            lines.append("")
            lines.append("Risk Assessment:")
            lines.append(f"  Probability: {track.failure_probability:.1%}" if track.failure_probability is not None else "  Probability: N/A")
            lines.append(f"  Category: {track.risk_category.value}")
            lines.append("")

            if track.unit_length:
                lines.append(f"Unit Length: {track.unit_length}")
            if track.travel_length:
                lines.append(f"Travel Length: {track.travel_length}")
            if track.untrimmed_resistance:
                lines.append(f"Untrimmed R: {track.untrimmed_resistance}")
            if track.trimmed_resistance:
                lines.append(f"Trimmed R: {track.trimmed_resistance}")
            lines.append("")

        self._update_metrics("\n".join(lines))

    def _display_track_chart(self, track: TrackData):
        """Display chart for a single track."""
        if not track.position_data or not track.error_data:
            self.chart.show_placeholder("No data available for chart")
            return

        # Get fail point indices
        fail_indices = None
        if track.linearity_fail_points > 0 and track.upper_limits and track.lower_limits:
            fail_indices = []
            shifted_errors = [e + track.optimal_offset for e in track.error_data]
            for i, e in enumerate(shifted_errors):
                if i < len(track.upper_limits) and i < len(track.lower_limits):
                    if e > track.upper_limits[i] or e < track.lower_limits[i]:
                        fail_indices.append(i)

        status_str = "PASS" if track.status == AnalysisStatus.PASS else track.status.value
        title = f"Track {track.track_id} - {status_str}"

        self.chart.plot_error_vs_position(
            positions=track.position_data,
            trimmed_errors=track.error_data,
            upper_limits=track.upper_limits,
            lower_limits=track.lower_limits,
            untrimmed_positions=track.untrimmed_positions,
            untrimmed_errors=track.untrimmed_errors,
            offset=track.optimal_offset,
            title=title,
            fail_points=fail_indices
        )

    def _on_track_selected(self, selection: str):
        """Handle track selection change."""
        if not self.current_result or not self.current_result.tracks:
            return

        if selection == "All Tracks":
            # Show first track
            self._display_track_chart(self.current_result.tracks[0])
        else:
            # Find selected track
            track_id = selection.replace("Track ", "")
            for track in self.current_result.tracks:
                if track.track_id == track_id:
                    self._display_track_chart(track)
                    break

    def _on_analysis_error(self, error: str):
        """Handle analysis error."""
        self.analyze_btn.configure(state="normal")
        self.status_text.configure(text="ERROR", text_color="#e74c3c")
        self.status_banner.configure(fg_color="#4d1e1e")
        self.status_label.configure(text="Analysis failed")
        self._update_metrics(f"Analysis failed:\n\n{error}")
        self.chart.show_placeholder(f"Error: {error}")

    def _update_metrics(self, text: str):
        """Update metrics display."""
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("end", text)
        self.metrics_text.configure(state="disabled")

    def _analyze_for_comparison(self):
        """Analyze file for track comparison."""
        if not self.compare_file:
            return

        def run():
            try:
                if self.processor is None:
                    self.processor = Processor()

                result = self.processor.process_file(self.compare_file)
                self.after(0, lambda: self._display_comparison(result))
            except Exception as e:
                logger.error(f"Comparison analysis error: {e}")
                self.after(0, lambda: self._show_comparison_error(str(e)))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _display_comparison(self, result: AnalysisResult):
        """Display track comparison."""
        tracks = result.tracks

        if len(tracks) >= 1:
            track_a = tracks[0]
            if track_a.position_data and track_a.error_data:
                self.chart_a.plot_error_vs_position(
                    positions=track_a.position_data,
                    trimmed_errors=track_a.error_data,
                    upper_limits=track_a.upper_limits,
                    lower_limits=track_a.lower_limits,
                    offset=track_a.optimal_offset,
                    title=f"Track A - {'PASS' if track_a.sigma_pass and track_a.linearity_pass else 'FAIL'}"
                )
            else:
                self.chart_a.show_placeholder("No data for Track A")
        else:
            self.chart_a.show_placeholder("No Track A found")

        if len(tracks) >= 2:
            track_b = tracks[1]
            if track_b.position_data and track_b.error_data:
                self.chart_b.plot_error_vs_position(
                    positions=track_b.position_data,
                    trimmed_errors=track_b.error_data,
                    upper_limits=track_b.upper_limits,
                    lower_limits=track_b.lower_limits,
                    offset=track_b.optimal_offset,
                    title=f"Track B - {'PASS' if track_b.sigma_pass and track_b.linearity_pass else 'FAIL'}"
                )
            else:
                self.chart_b.show_placeholder("No data for Track B")
        else:
            self.chart_b.show_placeholder("No Track B found")

    def _show_comparison_error(self, error: str):
        """Show comparison error."""
        self.chart_a.show_placeholder(f"Error: {error}")
        self.chart_b.show_placeholder(f"Error: {error}")

    def _export_result(self):
        """Export the current analysis result to Excel."""
        if not self.current_result:
            return

        # Generate default filename
        default_name = generate_export_filename(self.current_result)

        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Export Analysis to Excel",
            defaultextension=".xlsx",
            initialfile=default_name,
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if not file_path:
            return  # User cancelled

        try:
            self.status_label.configure(text="Exporting...")
            output_path = export_single_result(self.current_result, file_path)
            self.status_label.configure(text=f"Exported: {output_path.name}")
            logger.info(f"Exported analysis to: {output_path}")
        except ExcelExportError as e:
            self.status_label.configure(text=f"Export failed: {str(e)[:30]}")
            logger.error(f"Export failed: {e}")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Analyze page shown")

    def load_result(self, result: AnalysisResult):
        """Load an analysis result from another page (e.g., Process page)."""
        self.current_result = result
        self.single_file_label.configure(text=result.metadata.filename)
        self.tabview.set("Single File")
        self._display_result(result)
