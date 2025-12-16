"""
Analyze Page - Browse and view analysis results from database.

This page is for reviewing already-processed files.
Use the Process Files page to process new files.
"""

import threading
import customtkinter as ctk
import logging
from pathlib import Path
from tkinter import filedialog
from typing import Optional, List, Dict, Any

from laser_trim_v3.core.models import AnalysisResult, AnalysisStatus, TrackData
from laser_trim_v3.core.processor import Processor
from laser_trim_v3.gui.widgets.chart import ChartWidget, ChartStyle
from laser_trim_v3.database import get_database
from laser_trim_v3.export import (
    export_single_result, export_batch_results,
    generate_export_filename, generate_batch_export_filename, ExcelExportError
)
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalyzePage(ctk.CTkFrame):
    """
    Analyze page for browsing and viewing results from database.

    Features:
    - Browse recent analyses from database
    - Filter by model, date, and status
    - View detailed metrics and charts
    - Export individual results
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.current_result: Optional[AnalysisResult] = None
        self.recent_analyses: List[AnalysisResult] = []

        self._create_ui()

    def _create_ui(self):
        """Create the analyze page UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)

        header = ctk.CTkLabel(
            header_frame,
            text="Analyze Results",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            header_frame,
            text="Browse and analyze results from the database",
            text_color="gray",
            font=ctk.CTkFont(size=12)
        )
        subtitle.grid(row=1, column=0, sticky="w")

        # Filter controls row
        filter_frame = ctk.CTkFrame(self)
        filter_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))

        # Model filter
        ctk.CTkLabel(filter_frame, text="Model:").pack(side="left", padx=(15, 5), pady=15)
        self.model_filter = ctk.CTkOptionMenu(
            filter_frame,
            values=["All Models"],
            command=self._on_filter_change,
            width=150
        )
        self.model_filter.pack(side="left", padx=5, pady=15)

        # Days filter
        ctk.CTkLabel(filter_frame, text="Period:").pack(side="left", padx=(20, 5), pady=15)
        self.days_filter = ctk.CTkOptionMenu(
            filter_frame,
            values=["Today", "Last 7 Days", "Last 30 Days", "All Time"],
            command=self._on_filter_change,
            width=120
        )
        self.days_filter.set("Last 7 Days")
        self.days_filter.pack(side="left", padx=5, pady=15)

        # Status filter
        ctk.CTkLabel(filter_frame, text="Status:").pack(side="left", padx=(20, 5), pady=15)
        self.status_filter = ctk.CTkOptionMenu(
            filter_frame,
            values=["All", "Pass", "Fail"],
            command=self._on_filter_change,
            width=100
        )
        self.status_filter.pack(side="left", padx=5, pady=15)

        # Refresh button
        refresh_btn = ctk.CTkButton(
            filter_frame,
            text="⟳ Refresh",
            command=self._load_recent_analyses,
            width=100
        )
        refresh_btn.pack(side="right", padx=15, pady=15)

        # Results count label
        self.count_label = ctk.CTkLabel(
            filter_frame,
            text="",
            text_color="gray"
        )
        self.count_label.pack(side="right", padx=10, pady=15)

        # Main content area
        content = ctk.CTkFrame(self)
        content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        content.grid_columnconfigure(0, weight=1, minsize=300)  # List panel
        content.grid_columnconfigure(1, weight=2)  # Details panel
        content.grid_rowconfigure(0, weight=1)

        # Left panel - analysis list
        list_frame = ctk.CTkFrame(content)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        list_frame.grid_rowconfigure(1, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        list_label = ctk.CTkLabel(
            list_frame,
            text="Recent Analyses",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        list_label.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        # Scrollable list of analyses
        self.analysis_list_frame = ctk.CTkScrollableFrame(list_frame)
        self.analysis_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.analysis_list_frame.grid_columnconfigure(0, weight=1)

        # Right panel - details view
        details_frame = ctk.CTkFrame(content)
        details_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        details_frame.grid_rowconfigure(2, weight=1)  # Row 2 = tabview with chart
        details_frame.grid_columnconfigure(0, weight=1)

        # Status banner
        self.status_banner = ctk.CTkFrame(details_frame, height=60)
        self.status_banner.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))

        self.status_text = ctk.CTkLabel(
            self.status_banner,
            text="Select an analysis",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="gray"
        )
        self.status_text.pack(expand=True, pady=15)

        # Action buttons row
        actions_frame = ctk.CTkFrame(details_frame, fg_color="transparent")
        actions_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=5)

        # Process file button (analyze new file)
        self.process_file_btn = ctk.CTkButton(
            actions_frame,
            text="📂 Analyze File",
            command=self._process_single_file,
            width=130,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.process_file_btn.pack(side="left", padx=5)

        # Re-analyze button (re-process selected file from original path)
        self.reanalyze_btn = ctk.CTkButton(
            actions_frame,
            text="🔄 Re-analyze",
            command=self._reanalyze_current,
            state="disabled",
            width=110,
            fg_color="orange",
            hover_color="darkorange"
        )
        self.reanalyze_btn.pack(side="left", padx=5)

        # Export chart button
        self.export_chart_btn = ctk.CTkButton(
            actions_frame,
            text="📊 Export Chart",
            command=self._export_chart,
            state="disabled",
            width=120
        )
        self.export_chart_btn.pack(side="right", padx=5)

        # Export model button (all SNs for selected model)
        self.export_model_btn = ctk.CTkButton(
            actions_frame,
            text="📋 Export Model",
            command=self._export_model_results,
            state="disabled",
            width=130
        )
        self.export_model_btn.pack(side="right", padx=5)

        # Export single result button
        self.export_btn = ctk.CTkButton(
            actions_frame,
            text="📄 Export SN",
            command=self._export_result,
            state="disabled",
            width=100
        )
        self.export_btn.pack(side="right", padx=5)

        # Track selector
        ctk.CTkLabel(actions_frame, text="Track:").pack(side="left", padx=(0, 5))
        self.track_selector = ctk.CTkComboBox(
            actions_frame,
            values=["All Tracks"],
            command=self._on_track_selected,
            state="disabled",
            width=120
        )
        self.track_selector.pack(side="left", padx=5)

        # Details tabview (Chart / Metrics)
        self.details_tabview = ctk.CTkTabview(details_frame)
        self.details_tabview.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 5))

        self.details_tabview.add("Chart")
        self.details_tabview.add("Metrics")
        self.details_tabview.add("File Info")

        # Chart tab - responsive sizing
        chart_tab = self.details_tabview.tab("Chart")
        chart_tab.grid_columnconfigure(0, weight=1)
        chart_tab.grid_rowconfigure(0, weight=1)

        # Start with small figure, let dynamic resizing handle actual size
        self.chart = ChartWidget(
            chart_tab,
            style=ChartStyle(figure_size=(6, 4), dpi=100)
        )
        self.chart.pack(fill="both", expand=True)
        self.chart.show_placeholder("Select an analysis to view chart")

        # Metrics tab
        metrics_tab = self.details_tabview.tab("Metrics")
        self.metrics_text = ctk.CTkTextbox(metrics_tab, font=ctk.CTkFont(size=11, family="Consolas"))
        self.metrics_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.metrics_text.configure(state="disabled")
        self._update_metrics("Select an analysis from the list to view details.")

        # File Info tab
        info_tab = self.details_tabview.tab("File Info")
        self.info_text = ctk.CTkTextbox(info_tab, font=ctk.CTkFont(size=11, family="Consolas"))
        self.info_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.info_text.configure(state="disabled")
        self._update_info("Select an analysis from the list.")

        # Database info footer
        db_info_frame = ctk.CTkFrame(details_frame, fg_color="transparent")
        db_info_frame.grid(row=3, column=0, sticky="sew", padx=15, pady=(5, 10))

        self.db_info_label = ctk.CTkLabel(
            db_info_frame,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=10)
        )
        self.db_info_label.pack(side="left")

    # =========================================================================
    # Data Loading
    # =========================================================================

    def _load_recent_analyses(self):
        """Load recent analyses from database."""
        self.count_label.configure(text="Loading...")

        thread = threading.Thread(target=self._fetch_analyses, daemon=True)
        thread.start()

    def _fetch_analyses(self):
        """Fetch analyses in background thread."""
        try:
            db = get_database()

            # Get filter values
            model = self.model_filter.get()
            days_str = self.days_filter.get()
            status = self.status_filter.get()

            # Parse days
            days_map = {
                "Today": 1,
                "Last 7 Days": 7,
                "Last 30 Days": 30,
                "All Time": 3650,
            }
            days_back = days_map.get(days_str, 7)

            # Get historical data
            analyses = db.get_historical_data(
                model=None if model == "All Models" else model,
                days_back=days_back,
                limit=100
            )

            # Filter by status if needed
            if status == "Pass":
                analyses = [a for a in analyses if a.overall_status == AnalysisStatus.PASS]
            elif status == "Fail":
                analyses = [a for a in analyses if a.overall_status == AnalysisStatus.FAIL]

            # Get models list for dropdown
            models = db.get_models_list()

            # Get database info
            db_path = db.get_database_path()
            record_count = db.get_record_count()

            # Update UI on main thread
            self.after(0, lambda: self._display_analyses(analyses, models, db_path, record_count))

        except Exception as e:
            logger.error(f"Failed to fetch analyses: {e}")
            self.after(0, lambda: self._show_fetch_error(str(e)))

    def _display_analyses(self, analyses: List[AnalysisResult], models: List[str],
                          db_path: Path, record_count: Dict[str, int]):
        """Display fetched analyses."""
        self.recent_analyses = analyses

        # Update model dropdown
        model_values = ["All Models"] + models
        current = self.model_filter.get()
        self.model_filter.configure(values=model_values)
        if current in model_values:
            self.model_filter.set(current)

        # Update count
        self.count_label.configure(text=f"{len(analyses)} results")

        # Update database info
        self.db_info_label.configure(
            text=f"Database: {db_path.name} | {record_count.get('analyses', 0)} analyses, {record_count.get('tracks', 0)} tracks"
        )

        # Clear existing list
        for widget in self.analysis_list_frame.winfo_children():
            widget.destroy()

        if not analyses:
            no_data_label = ctk.CTkLabel(
                self.analysis_list_frame,
                text="No analyses found.\n\nUse 'Process Files' to\nprocess some files first!",
                text_color="gray",
                justify="center"
            )
            no_data_label.pack(pady=40)
            return

        # Create list items
        for i, analysis in enumerate(analyses):
            self._create_analysis_list_item(analysis, i)

    def _create_analysis_list_item(self, analysis: AnalysisResult, index: int):
        """Create a clickable item for an analysis."""
        # Determine status color
        if analysis.overall_status == AnalysisStatus.PASS:
            status_color = "#27ae60"
            status_text = "PASS"
        elif analysis.overall_status == AnalysisStatus.FAIL:
            status_color = "#e74c3c"
            status_text = "FAIL"
        else:
            status_color = "#f39c12"
            status_text = analysis.overall_status.value.upper()

        item_frame = ctk.CTkFrame(self.analysis_list_frame, cursor="hand2")
        item_frame.pack(fill="x", pady=2)

        # Status indicator
        status_indicator = ctk.CTkLabel(
            item_frame,
            text=status_text,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=status_color,
            width=40
        )
        status_indicator.pack(side="left", padx=(10, 5), pady=8)

        # Info section
        info_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        filename_label = ctk.CTkLabel(
            info_frame,
            text=analysis.metadata.filename[:35] + ("..." if len(analysis.metadata.filename) > 35 else ""),
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        filename_label.pack(anchor="w")

        details_text = f"{analysis.metadata.model} | {analysis.metadata.system.value}"
        if analysis.metadata.file_date:
            details_text += f" | {analysis.metadata.file_date.strftime('%m/%d %H:%M')}"

        details_label = ctk.CTkLabel(
            info_frame,
            text=details_text,
            font=ctk.CTkFont(size=10),
            text_color="gray",
            anchor="w"
        )
        details_label.pack(anchor="w")

        # Bind click event
        def on_click(event, a=analysis):
            self._show_analysis_details(a)

        item_frame.bind("<Button-1>", on_click)
        status_indicator.bind("<Button-1>", on_click)
        info_frame.bind("<Button-1>", on_click)
        filename_label.bind("<Button-1>", on_click)
        details_label.bind("<Button-1>", on_click)

    # =========================================================================
    # Details Display
    # =========================================================================

    def _show_analysis_details(self, analysis: AnalysisResult):
        """Show details for selected analysis."""
        self.current_result = analysis

        # Update status banner
        if analysis.overall_status == AnalysisStatus.PASS:
            self.status_text.configure(text="PASS", text_color="#27ae60")
            self.status_banner.configure(fg_color="#1e4d2b")
        elif analysis.overall_status == AnalysisStatus.FAIL:
            self.status_text.configure(text="FAIL", text_color="#e74c3c")
            self.status_banner.configure(fg_color="#4d1e1e")
        else:
            self.status_text.configure(text=analysis.overall_status.value.upper(), text_color="#f39c12")
            self.status_banner.configure(fg_color="#4d3d1e")

        # Enable export buttons
        self.export_btn.configure(state="normal")
        self.export_chart_btn.configure(state="normal")
        self.export_model_btn.configure(state="normal")
        self.reanalyze_btn.configure(state="normal")

        # Update track selector
        if analysis.tracks:
            # Add "Compare All" option if multiple tracks exist
            track_values = [f"Track {t.track_id}" for t in analysis.tracks]
            if len(analysis.tracks) > 1:
                track_values.insert(0, "📊 Compare All Tracks")
            self.track_selector.configure(values=track_values, state="normal")
            self.track_selector.set(track_values[0])

            # Show appropriate chart
            if len(analysis.tracks) > 1:
                # If multiple tracks, show comparison by default
                self._display_comparison_chart(analysis.tracks)
            else:
                # Single track
                self._display_track_chart(analysis.tracks[0])

        # Update metrics and info
        self._display_metrics(analysis)
        self._display_file_info(analysis)

    def _display_track_chart(self, track: TrackData):
        """Display chart for a track."""
        if not track.position_data or not track.error_data:
            self.chart.show_placeholder("No chart data available\n\n(Position/error data not stored)")
            return

        # Use stored spec limits (position-dependent) from database
        upper_limits = track.upper_limits
        lower_limits = track.lower_limits

        # If limits not stored (old data), fall back to flat calculation
        if not upper_limits or not lower_limits:
            if track.linearity_spec and track.linearity_spec > 0:
                upper_limits = [track.linearity_spec] * len(track.position_data)
                lower_limits = [-track.linearity_spec] * len(track.position_data)

        # ALWAYS recalculate fail point indices (old DB data may have incorrect fail_points=0)
        fail_indices = []
        actual_fail_count = 0
        if upper_limits and lower_limits:
            shifted_errors = [e + track.optimal_offset for e in track.error_data]
            for i, e in enumerate(shifted_errors):
                if i < len(upper_limits) and i < len(lower_limits):
                    if e > upper_limits[i] or e < lower_limits[i]:
                        fail_indices.append(i)
                        actual_fail_count += 1

        # Determine actual status based on recalculated fail points
        if actual_fail_count > 0:
            status_str = "FAIL (Lin)"
        elif not track.sigma_pass:
            status_str = "FAIL (Sigma)"
        elif track.status == AnalysisStatus.PASS:
            status_str = "PASS"
        else:
            status_str = track.status.value
        title = f"Track {track.track_id} - {status_str}"

        self.chart.plot_error_vs_position(
            positions=track.position_data,
            trimmed_errors=track.error_data,
            upper_limits=upper_limits,
            lower_limits=lower_limits,
            untrimmed_positions=track.untrimmed_positions,
            untrimmed_errors=track.untrimmed_errors,
            offset=track.optimal_offset,
            title=title,
            fail_points=fail_indices
        )

    def _display_metrics(self, analysis: AnalysisResult):
        """Display analysis metrics."""
        lines = []

        # Overall status
        lines.append("═══════════════════════════════════════")
        lines.append(f"  OVERALL STATUS: {analysis.overall_status.value.upper()}")
        lines.append("═══════════════════════════════════════")
        lines.append("")

        # Track details
        for track in analysis.tracks:
            lines.append(f"━━━ TRACK {track.track_id} ━━━")
            lines.append(f"  Status: {track.status.value}")
            lines.append("")

            # Sigma Analysis
            lines.append("  SIGMA ANALYSIS:")
            lines.append(f"    Gradient:  {track.sigma_gradient:.6f}")
            lines.append(f"    Threshold: {track.sigma_threshold:.6f}")
            margin = track.sigma_threshold - track.sigma_gradient
            lines.append(f"    Margin:    {margin:.6f}")
            lines.append(f"    Result:    {'✓ PASS' if track.sigma_pass else '✗ FAIL'}")
            lines.append("")

            # Linearity Analysis
            lines.append("  LINEARITY ANALYSIS:")

            # Recalculate fail points using actual spec limits
            actual_upper = track.upper_limits
            actual_lower = track.lower_limits

            # Fall back to linearity_spec if limits not stored
            if not actual_upper or not actual_lower:
                if track.linearity_spec and track.linearity_spec > 0:
                    actual_upper = [track.linearity_spec] * len(track.error_data) if track.error_data else []
                    actual_lower = [-track.linearity_spec] * len(track.error_data) if track.error_data else []

            # Calculate actual fail points
            actual_fail_count = 0
            if actual_upper and actual_lower and track.error_data:
                shifted_errors = [e + track.optimal_offset for e in track.error_data]
                for i, e in enumerate(shifted_errors):
                    if i < len(actual_upper) and i < len(actual_lower):
                        if e > actual_upper[i] or e < actual_lower[i]:
                            actual_fail_count += 1

            # Show spec limits
            if actual_upper and actual_lower:
                max_upper = max(actual_upper)
                min_lower = min(actual_lower)
                lines.append(f"    Spec:       ±{track.linearity_spec:.6f}")
                if max_upper != track.linearity_spec or min_lower != -track.linearity_spec:
                    lines.append(f"    (Limits vary: +{max_upper:.6f} / {min_lower:.6f})")
            else:
                lines.append(f"    Spec:       ±{track.linearity_spec:.6f}")

            lines.append(f"    Max Error:  {track.linearity_error:.6f}")
            lines.append(f"    Offset:     {track.optimal_offset:.6f}")

            # Show both stored and recalculated fail points if they differ
            if actual_fail_count != track.linearity_fail_points:
                lines.append(f"    Fail Pts:   {actual_fail_count} (DB stored: {track.linearity_fail_points})")
            else:
                lines.append(f"    Fail Pts:   {track.linearity_fail_points}")

            # Show actual result based on recalculation
            actual_pass = actual_fail_count == 0
            if actual_pass != track.linearity_pass:
                lines.append(f"    Result:     {'✓ PASS' if actual_pass else '✗ FAIL'} (DB: {'PASS' if track.linearity_pass else 'FAIL'})")
            else:
                lines.append(f"    Result:     {'✓ PASS' if track.linearity_pass else '✗ FAIL'}")
            lines.append("")

            # Risk Assessment
            lines.append("  RISK ASSESSMENT:")
            if track.failure_probability is not None:
                lines.append(f"    Probability: {track.failure_probability:.1%}")
            else:
                lines.append(f"    Probability: N/A")
            lines.append(f"    Category:    {track.risk_category.value}")
            lines.append("")

            # Unit Properties (if available)
            if track.unit_length or track.untrimmed_resistance or track.trimmed_resistance:
                lines.append("  UNIT PROPERTIES:")
                if track.unit_length:
                    lines.append(f"    Unit Length: {track.unit_length:.4f}")
                if track.untrimmed_resistance:
                    lines.append(f"    Untrimmed R: {track.untrimmed_resistance:.2f}")
                if track.trimmed_resistance:
                    lines.append(f"    Trimmed R:   {track.trimmed_resistance:.2f}")
                lines.append("")

        self._update_metrics("\n".join(lines))

    def _display_file_info(self, analysis: AnalysisResult):
        """Display file information."""
        lines = []

        lines.append("═══════════════════════════════════════")
        lines.append("  FILE INFORMATION")
        lines.append("═══════════════════════════════════════")
        lines.append("")

        lines.append(f"  Filename:    {analysis.metadata.filename}")
        lines.append(f"  Model:       {analysis.metadata.model}")
        lines.append(f"  Serial:      {analysis.metadata.serial}")
        lines.append(f"  System:      {analysis.metadata.system.value}")
        lines.append("")

        lines.append(f"  File Date:   {analysis.metadata.file_date.strftime('%Y-%m-%d %H:%M:%S')}")
        if analysis.metadata.test_date:
            lines.append(f"  Test Date:   {analysis.metadata.test_date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        lines.append(f"  Multi-Track: {'Yes' if analysis.metadata.has_multi_tracks else 'No'}")
        lines.append(f"  Track Count: {len(analysis.tracks)}")
        lines.append("")

        lines.append("───────────────────────────────────────")
        lines.append("  PROCESSING INFO")
        lines.append("───────────────────────────────────────")
        lines.append("")
        lines.append(f"  Processing Time: {analysis.processing_time:.2f}s")
        lines.append(f"  File Path: {analysis.metadata.file_path}")

        self._update_info("\n".join(lines))

    def _on_track_selected(self, selection: str):
        """Handle track selection change."""
        if not self.current_result or not self.current_result.tracks:
            return

        # Check if "Compare All" was selected
        if "Compare All" in selection:
            self._display_comparison_chart(self.current_result.tracks)
            return

        # Individual track selection
        track_id = selection.replace("Track ", "")
        for track in self.current_result.tracks:
            if track.track_id == track_id:
                self._display_track_chart(track)
                break

    def _display_comparison_chart(self, tracks: List[TrackData]):
        """Display comparison chart overlaying all tracks."""
        if not tracks or len(tracks) < 2:
            self._display_track_chart(tracks[0] if tracks else None)
            return

        # Use ChartWidget's comparison method
        self.chart.plot_track_comparison(tracks)

    def _on_filter_change(self, *args):
        """Handle filter change."""
        self._load_recent_analyses()

    def _show_fetch_error(self, error: str):
        """Show fetch error."""
        self.count_label.configure(text="Error")

        for widget in self.analysis_list_frame.winfo_children():
            widget.destroy()

        error_label = ctk.CTkLabel(
            self.analysis_list_frame,
            text=f"Error loading data:\n{error}",
            text_color="red"
        )
        error_label.pack(pady=20)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _update_metrics(self, text: str):
        """Update metrics display."""
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("end", text)
        self.metrics_text.configure(state="disabled")

    def _update_info(self, text: str):
        """Update info display."""
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("end", text)
        self.info_text.configure(state="disabled")

    def _process_single_file(self):
        """Process a single file and display results."""
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Excel File to Analyze",
            filetypes=[("Excel files", "*.xls *.xlsx"), ("All files", "*.*")]
        )

        if not file_path:
            return  # User cancelled

        # Update UI to show processing
        self.status_text.configure(text="Processing...", text_color="orange")
        self.status_banner.configure(fg_color="orange")
        self.process_file_btn.configure(state="disabled")
        self.update_idletasks()

        # Process in background thread
        def process():
            try:
                processor = Processor()
                result = processor.process_file(Path(file_path))

                # Save to database
                try:
                    db = get_database()
                    db.save_analysis(result)
                except Exception as e:
                    logger.error(f"Failed to save to database: {e}")

                # Update UI on main thread
                self.after(0, lambda: self._on_single_file_processed(result))

            except Exception as e:
                logger.exception(f"Processing error: {e}")
                self.after(0, lambda: self._on_single_file_error(str(e)))

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _on_single_file_processed(self, result: AnalysisResult):
        """Handle single file processing completion."""
        self.process_file_btn.configure(state="normal")
        self.current_result = result

        # Show the result
        self._show_analysis_details(result)

        # Refresh the list to include the new result
        self._load_recent_analyses()

        logger.info(f"Processed single file: {result.metadata.filename}")

    def _on_single_file_error(self, error: str):
        """Handle single file processing error."""
        self.process_file_btn.configure(state="normal")
        self.reanalyze_btn.configure(state="normal")
        self.status_text.configure(text=f"Error: {error}", text_color="red")
        self.status_banner.configure(fg_color="red")
        logger.error(f"Single file processing failed: {error}")

    def _reanalyze_current(self):
        """Re-analyze the current file from its original path (updates DB with corrected values)."""
        if not self.current_result:
            return

        file_path = self.current_result.metadata.file_path
        if not file_path or not Path(file_path).exists():
            self.status_text.configure(
                text=f"Original file not found: {file_path}",
                text_color="red"
            )
            self.status_banner.configure(fg_color="red")
            logger.error(f"Cannot re-analyze - file not found: {file_path}")
            return

        logger.info(f"Re-analyzing file: {file_path}")
        self.reanalyze_btn.configure(state="disabled")
        self.status_text.configure(text="Re-analyzing...", text_color="#3498db")
        self.status_banner.configure(fg_color="#1e3d4d")

        # Process in background thread
        def process():
            try:
                processor = Processor()
                result = processor.process_file(Path(file_path))

                # Save to database (will update existing record)
                try:
                    db = get_database()
                    db.save_analysis(result)
                    logger.info(f"Re-analyzed and updated DB: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to save to database: {e}")

                # Update UI on main thread
                self.after(0, lambda: self._on_reanalyze_complete(result))

            except Exception as e:
                logger.exception(f"Re-analysis error: {e}")
                self.after(0, lambda: self._on_reanalyze_error(str(e)))

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _on_reanalyze_complete(self, result: AnalysisResult):
        """Handle re-analysis completion."""
        self.reanalyze_btn.configure(state="normal")
        self.current_result = result

        # Show the updated result
        self._show_analysis_details(result)

        # Refresh the list
        self._load_recent_analyses()

        logger.info(f"Re-analysis complete: {result.metadata.filename}")

    def _on_reanalyze_error(self, error: str):
        """Handle re-analysis error."""
        self.reanalyze_btn.configure(state="normal")
        self.status_text.configure(text=f"Re-analysis error: {error}", text_color="red")
        self.status_banner.configure(fg_color="red")
        logger.error(f"Re-analysis failed: {error}")

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
            output_path = export_single_result(self.current_result, file_path)
            logger.info(f"Exported analysis to: {output_path}")
        except ExcelExportError as e:
            logger.error(f"Export failed: {e}")

    def _export_model_results(self):
        """Export all analysis results for the selected model to Excel."""
        if not self.current_result:
            return

        model = self.current_result.metadata.model

        try:
            # Get all analyses for this model from database
            db = get_database()
            model_results = db.get_historical_data(model=model, days_back=36500, limit=10000)

            if not model_results:
                logger.warning(f"No results found for model: {model}")
                return

            # Generate default filename
            default_name = generate_batch_export_filename(model_results, prefix=f"model_{model}")

            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                title=f"Export All {model} Results to Excel",
                defaultextension=".xlsx",
                initialfile=default_name,
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )

            if not file_path:
                return  # User cancelled

            output_path = export_batch_results(model_results, file_path)
            logger.info(f"Exported {len(model_results)} results for model {model} to: {output_path}")

        except Exception as e:
            logger.error(f"Model export failed: {e}")

    def _export_chart(self):
        """Export the current chart to an image file with analysis info like V2."""
        if not self.current_result:
            return

        # Generate default filename
        model = self.current_result.metadata.model
        serial = self.current_result.metadata.serial
        default_name = f"{model}_{serial}_chart.png"

        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Export Chart",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("JPEG Image", "*.jpg"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return  # User cancelled

        try:
            # Export comprehensive chart with analysis info (like V2)
            self._export_comprehensive_chart(file_path)
            logger.info(f"Exported chart to: {file_path}")
        except Exception as e:
            logger.error(f"Chart export failed: {e}")

    def _export_comprehensive_chart(self, output_path: str):
        """
        Export chart with comprehensive analysis info, matching V2 format.

        Creates a multi-panel figure with:
        - Main error vs position chart
        - Analysis metrics summary
        - Status display
        - Unit information
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Rectangle
        import numpy as np

        # Use light mode for export (matches Excel export style)
        plt.style.use('default')

        result = self.current_result
        if not result or not result.tracks:
            raise ValueError("No analysis data to export")

        # Get the currently selected track
        track_name = self.track_selector.get()
        track = None
        for t in result.tracks:
            if f"Track {t.track_id}" == track_name:
                track = t
                break

        if not track:
            track = result.tracks[0]

        # Create figure with subplots (larger for detailed export) - LIGHT mode
        fig = plt.figure(figsize=(14, 10), dpi=150, facecolor='white')

        # Title - black text for light mode
        title = f'Laser Trim Analysis - {result.metadata.model} / {result.metadata.serial}'
        fig.suptitle(title, fontsize=16, fontweight='bold', color='black')

        # Grid spec for layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3,
                              left=0.08, right=0.95, top=0.92, bottom=0.08)

        # ===== Main error plot (top 2/3) =====
        ax_main = fig.add_subplot(gs[0:2, :])
        self._plot_error_vs_position_export(ax_main, track)

        # ===== Unit Info (bottom left) =====
        ax_info = fig.add_subplot(gs[2, 0])
        self._plot_unit_info(ax_info, result, track)

        # ===== Metrics summary (bottom middle) =====
        ax_metrics = fig.add_subplot(gs[2, 1])
        self._plot_metrics_summary(ax_metrics, track)

        # ===== Status display (bottom right) =====
        ax_status = fig.add_subplot(gs[2, 2])
        self._plot_status_display(ax_status, track, result)

        # Save
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def _plot_error_vs_position_export(self, ax, track: TrackData):
        """Plot error vs position for export (light mode)."""
        import numpy as np

        # Colors for light mode export
        QA_COLORS = {
            'pass': '#27ae60',
            'fail': '#e74c3c',
            'warning': '#f39c12',
            'trimmed': '#27ae60',
            'untrimmed': '#3498db',
            'spec_limit': '#e74c3c',
        }

        positions = np.array(track.position_data)
        errors = np.array(track.error_data)

        # Apply offset
        offset = track.optimal_offset
        errors_shifted = errors + offset

        # Plot untrimmed data if available
        if track.untrimmed_positions and track.untrimmed_errors:
            ax.plot(track.untrimmed_positions, track.untrimmed_errors,
                   'b--', linewidth=1.5, label='Untrimmed',
                   color=QA_COLORS['untrimmed'], alpha=0.6)

        # Plot trimmed/shifted data
        ax.plot(positions, errors_shifted, 'g-', linewidth=2,
               label='Trimmed (Offset Applied)', color=QA_COLORS['trimmed'])

        # Get spec limits - use stored limits or calculate from linearity_spec
        upper_limits = track.upper_limits
        lower_limits = track.lower_limits

        # Fallback: if limits not stored, use flat linearity_spec
        if (not upper_limits or not lower_limits or len(upper_limits) != len(positions)):
            if track.linearity_spec and track.linearity_spec > 0:
                upper_limits = [track.linearity_spec] * len(positions)
                lower_limits = [-track.linearity_spec] * len(positions)
                logger.debug(f"Using flat spec limits: +/-{track.linearity_spec}")

        # Plot spec limits
        if upper_limits and lower_limits and len(upper_limits) == len(positions):
            ax.plot(positions, upper_limits, 'r--', linewidth=1.5,
                   label=f'Spec Limits (+/-{track.linearity_spec:.4f})', color=QA_COLORS['spec_limit'])
            ax.plot(positions, lower_limits, 'r--', linewidth=1.5,
                   color=QA_COLORS['spec_limit'])
            ax.fill_between(positions, lower_limits, upper_limits,
                           alpha=0.15, color=QA_COLORS['spec_limit'])

        # Find and mark fail points
        fail_indices = []
        if upper_limits and lower_limits:
            for i, e in enumerate(errors_shifted):
                if i < len(upper_limits) and i < len(lower_limits):
                    if e > upper_limits[i] or e < lower_limits[i]:
                        fail_indices.append(i)

        if fail_indices:
            fail_pos = [positions[i] for i in fail_indices]
            fail_err = [errors_shifted[i] for i in fail_indices]
            ax.scatter(fail_pos, fail_err, color=QA_COLORS['fail'],
                      s=100, marker='x', linewidth=3,
                      label=f'Fail Points ({len(fail_indices)})', zorder=5)

        # Zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # Labels
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Error (Volts)', fontsize=12)

        status_str = "PASS" if track.linearity_pass and track.sigma_pass else "FAIL"
        ax.set_title(f'Track {track.track_id} - {status_str}', fontsize=14, fontweight='bold')

        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add offset info
        if offset != 0:
            ax.text(0.02, 0.98, f'Optimal Offset: {offset:.6f} V',
                   transform=ax.transAxes, fontsize=10, va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    def _plot_unit_info(self, ax, result: AnalysisResult, track: TrackData):
        """Plot unit information panel (light mode)."""
        ax.axis('off')
        ax.set_facecolor('white')

        # File/unit info
        info_lines = [
            f"Model: {result.metadata.model}",
            f"Serial: {result.metadata.serial}",
            f"System: {result.metadata.system.value}",
            f"",
            f"Track: {track.track_id}",
            f"Travel Length: {track.travel_length:.3f}" if track.travel_length else "Travel Length: N/A",
            f"Unit Length: {track.unit_length}" if track.unit_length else "Unit Length: N/A",
        ]

        # Trim date
        trim_date = result.metadata.file_date or result.metadata.test_date
        if trim_date:
            info_lines.append(f"Trim Date: {trim_date.strftime('%m/%d/%Y')}")

        # Resistance values
        if track.untrimmed_resistance:
            info_lines.append(f"Untrimmed R: {track.untrimmed_resistance}")
        if track.trimmed_resistance:
            info_lines.append(f"Trimmed R: {track.trimmed_resistance}")

        y_pos = 0.95
        ax.text(0.05, 0.98, "Unit Information", fontsize=11, fontweight='bold',
               transform=ax.transAxes, va='top', color='black')

        for line in info_lines:
            y_pos -= 0.09
            ax.text(0.05, y_pos, line, fontsize=10, transform=ax.transAxes, va='top', color='black')

    def _plot_metrics_summary(self, ax, track: TrackData):
        """Plot analysis metrics summary (light mode)."""
        ax.axis('off')
        ax.set_facecolor('white')

        metrics = [
            f"Sigma Gradient: {track.sigma_gradient:.6f}",
            f"Sigma Threshold: {track.sigma_threshold:.6f}",
            f"Sigma Pass: {'✓ YES' if track.sigma_pass else '✗ NO'}",
            "",
            f"Optimal Offset: {track.optimal_offset:.6f}",
            f"Linearity Error: {track.linearity_error:.6f}",
            f"Fail Points: {track.linearity_fail_points}",
            f"Linearity Pass: {'✓ YES' if track.linearity_pass else '✗ NO'}",
            "",
            f"Risk Category: {track.risk_category.value}",
            f"Failure Prob: {track.failure_probability:.1%}" if track.failure_probability else "Failure Prob: N/A"
        ]

        y_pos = 0.95
        ax.text(0.05, 0.98, "Analysis Metrics", fontsize=11, fontweight='bold',
               transform=ax.transAxes, va='top', color='black')

        for metric in metrics:
            y_pos -= 0.085
            color = 'black'
            if '✗' in metric:
                color = '#e74c3c'
            elif '✓' in metric:
                color = '#27ae60'

            ax.text(0.05, y_pos, metric, fontsize=10, transform=ax.transAxes,
                   va='top', color=color)

    def _plot_status_display(self, ax, track: TrackData, result: AnalysisResult):
        """Plot status display panel (light mode)."""
        from matplotlib.patches import Rectangle

        ax.axis('off')
        ax.set_facecolor('white')

        # Determine overall status
        if track.sigma_pass and track.linearity_pass:
            status = "PASS"
            color = '#27ae60'
        else:
            status = "FAIL"
            color = '#e74c3c'

        # Draw status box
        rect = Rectangle((0.1, 0.6), 0.8, 0.25,
                         linewidth=3, edgecolor=color,
                         facecolor='white', alpha=0.9)
        ax.add_patch(rect)

        ax.text(0.5, 0.72, f'STATUS: {status}', ha='center', va='center',
               fontsize=16, color=color, fontweight='bold', transform=ax.transAxes)

        # Add failure details
        details = []
        if not track.sigma_pass:
            details.append("Sigma: FAIL")
        if not track.linearity_pass:
            details.append(f"Linearity: FAIL ({track.linearity_fail_points} pts)")

        if details:
            ax.text(0.5, 0.45, "\n".join(details), ha='center', va='center',
                   fontsize=11, color=color, transform=ax.transAxes)

        # Analysis timestamp
        ax.text(0.5, 0.15, f"Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
               ha='center', va='center', fontsize=9, color='gray',
               style='italic', transform=ax.transAxes)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # =========================================================================
    # Page Lifecycle
    # =========================================================================

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Analyze page shown")
        # Load recent analyses when page is shown
        self._load_recent_analyses()

    def load_result(self, result: AnalysisResult):
        """Load an analysis result from another page (e.g., Process page)."""
        self.current_result = result
        self._show_analysis_details(result)
