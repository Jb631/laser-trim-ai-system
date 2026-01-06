"""
Export Page - Select units from database and export charts.

Allows users to:
- Filter by model, date range, and serial number
- Select single or multiple units
- Export charts as PNG files or combined PDF
"""

import customtkinter as ctk
import logging
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional, List, Dict, Set, Any, TYPE_CHECKING
from datetime import datetime, timedelta

from laser_trim_analyzer.core.models import AnalysisResult, AnalysisStatus, TrackData
from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.utils.threads import get_thread_manager
from laser_trim_analyzer.gui.widgets.scrollable_combobox import ScrollableComboBox

if TYPE_CHECKING:
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle

logger = logging.getLogger(__name__)


class ExportPage(ctk.CTkFrame):
    """
    Export page for selecting units and exporting charts.

    Features:
    - Filter by model, date range, serial number
    - Multi-select with checkboxes
    - Export to PNG or combined PDF
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.search_results: List[AnalysisResult] = []
        self.selected_ids: Set[int] = set()  # Track selected by database ID
        self._checkbox_vars: Dict[int, ctk.BooleanVar] = {}  # Checkbox state by ID

        self._create_ui()

    def _create_ui(self):
        """Create the export page UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        header = ctk.CTkLabel(
            header_frame,
            text="Export Charts",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            header_frame,
            text="Select units from the database to export charts",
            text_color="gray",
            font=ctk.CTkFont(size=12)
        )
        subtitle.pack(anchor="w")

        # Filter controls
        filter_frame = ctk.CTkFrame(self)
        filter_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))

        # Model filter
        ctk.CTkLabel(filter_frame, text="Model:").pack(side="left", padx=(15, 5), pady=15)
        self.model_filter = ScrollableComboBox(
            filter_frame,
            values=["All Models"],
            width=150,
            dropdown_height=300,
        )
        self.model_filter.set("All Models")
        self.model_filter.pack(side="left", padx=5, pady=15)

        # Date range filter
        ctk.CTkLabel(filter_frame, text="Trim Date:").pack(side="left", padx=(20, 5), pady=15)
        self.date_filter = ctk.CTkOptionMenu(
            filter_frame,
            values=["Today", "Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
            width=120
        )
        self.date_filter.set("Last 30 Days")
        self.date_filter.pack(side="left", padx=5, pady=15)

        # Serial filter
        ctk.CTkLabel(filter_frame, text="Serial:").pack(side="left", padx=(20, 5), pady=15)
        self.serial_filter = ctk.CTkEntry(
            filter_frame,
            placeholder_text="Partial match...",
            width=120
        )
        self.serial_filter.pack(side="left", padx=5, pady=15)

        # Search button
        search_btn = ctk.CTkButton(
            filter_frame,
            text="Search",
            command=self._search,
            width=100
        )
        search_btn.pack(side="left", padx=15, pady=15)

        # Results count
        self.count_label = ctk.CTkLabel(
            filter_frame,
            text="",
            text_color="gray"
        )
        self.count_label.pack(side="right", padx=15, pady=15)

        # Main content area
        content = ctk.CTkFrame(self)
        content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 10))
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)

        # Selection controls
        selection_frame = ctk.CTkFrame(content, fg_color="transparent")
        selection_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        self.results_label = ctk.CTkLabel(
            selection_frame,
            text="Results",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.results_label.pack(side="left")

        # Clear All button
        clear_btn = ctk.CTkButton(
            selection_frame,
            text="Clear All",
            command=self._clear_selection,
            width=80,
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30")
        )
        clear_btn.pack(side="right", padx=5)

        # Select All button
        select_all_btn = ctk.CTkButton(
            selection_frame,
            text="Select All",
            command=self._select_all,
            width=80,
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30")
        )
        select_all_btn.pack(side="right", padx=5)

        # Scrollable results list
        self.results_frame = ctk.CTkScrollableFrame(content)
        self.results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.results_frame.grid_columnconfigure(0, weight=1)

        # Bottom panel with export options
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 20))
        bottom_frame.grid_columnconfigure(1, weight=1)

        # Selection count
        self.selection_label = ctk.CTkLabel(
            bottom_frame,
            text="Selected: 0 units",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.selection_label.grid(row=0, column=0, padx=15, pady=15, sticky="w")

        # Export format selection
        format_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        format_frame.grid(row=0, column=1, pady=15)

        ctk.CTkLabel(format_frame, text="Export as:").pack(side="left", padx=(0, 10))

        self.export_format = ctk.StringVar(value="png")
        png_radio = ctk.CTkRadioButton(
            format_frame,
            text="Individual PNGs",
            variable=self.export_format,
            value="png"
        )
        png_radio.pack(side="left", padx=5)

        pdf_radio = ctk.CTkRadioButton(
            format_frame,
            text="Combined PDF",
            variable=self.export_format,
            value="pdf"
        )
        pdf_radio.pack(side="left", padx=5)

        # Export button
        self.export_btn = ctk.CTkButton(
            bottom_frame,
            text="Export Charts",
            command=self._export_charts,
            width=140,
            state="disabled"
        )
        self.export_btn.grid(row=0, column=2, padx=15, pady=15, sticky="e")

    def _search(self):
        """Search for units matching filter criteria."""
        self.count_label.configure(text="Searching...")

        # Clear previous results
        self._clear_results()

        get_thread_manager().start_thread(target=self._fetch_results, name="export-fetch-results")

    def _fetch_results(self):
        """Fetch results in background thread."""
        try:
            db = get_database()

            # Get filter values
            model = self.model_filter.get()
            date_str = self.date_filter.get()
            serial = self.serial_filter.get().strip()

            # Parse date range
            date_from = None
            date_to = datetime.now()

            days_map = {
                "Today": 1,
                "Last 7 Days": 7,
                "Last 30 Days": 30,
                "Last 90 Days": 90,
                "All Time": None,
            }
            days_back = days_map.get(date_str)
            if days_back:
                date_from = datetime.now() - timedelta(days=days_back)

            # Search
            results = db.search_for_export(
                model=None if model == "All Models" else model,
                serial=serial if serial else None,
                date_from=date_from,
                date_to=date_to,
                limit=500
            )

            # Update model dropdown with available models
            try:
                models = db.get_models_list()
            except Exception:
                models = []

            # Update UI on main thread
            self.after(0, lambda: self._display_results(results, models))

        except Exception as e:
            logger.error(f"Search failed: {e}")
            self.after(0, lambda: self._show_error(str(e)))

    def _display_results(self, results: List[AnalysisResult], models: List[str]):
        """Display search results."""
        self.search_results = results
        self.selected_ids.clear()
        self._checkbox_vars.clear()

        # Update model dropdown
        model_values = ["All Models"] + models
        current = self.model_filter.get()
        self.model_filter.configure(values=model_values)
        if current in model_values:
            self.model_filter.set(current)

        # Update count
        self.count_label.configure(text=f"{len(results)} results")
        self.results_label.configure(text=f"Results ({len(results)})")

        # Clear existing items
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        if not results:
            no_data_label = ctk.CTkLabel(
                self.results_frame,
                text="No results found.\n\nTry adjusting your filters.",
                text_color="gray",
                justify="center"
            )
            no_data_label.pack(pady=40)
            self._update_selection_count()
            return

        # Create result items with checkboxes
        for i, result in enumerate(results):
            self._create_result_item(result, i)

        self._update_selection_count()

    def _create_result_item(self, result: AnalysisResult, index: int):
        """Create a result item with checkbox."""
        # Use a unique ID (we'll use the index as fallback if no DB id)
        item_id = index

        # Status color
        if result.overall_status == AnalysisStatus.PASS:
            status_color = "#27ae60"
            status_text = "PASS"
        elif result.overall_status == AnalysisStatus.FAIL:
            status_color = "#e74c3c"
            status_text = "FAIL"
        else:
            status_color = "#f39c12"
            status_text = result.overall_status.value.upper()

        item_frame = ctk.CTkFrame(self.results_frame)
        item_frame.pack(fill="x", pady=2, padx=5)

        # Checkbox
        var = ctk.BooleanVar(value=False)
        self._checkbox_vars[item_id] = var

        checkbox = ctk.CTkCheckBox(
            item_frame,
            text="",
            variable=var,
            command=lambda idx=item_id: self._on_checkbox_toggle(idx),
            width=30
        )
        checkbox.pack(side="left", padx=(10, 5), pady=8)

        # Status indicator
        status_label = ctk.CTkLabel(
            item_frame,
            text=status_text,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=status_color,
            width=50
        )
        status_label.pack(side="left", padx=5, pady=8)

        # Info section
        info_text = f"{result.metadata.model}  |  SN: {result.metadata.serial}"
        if result.metadata.file_date:
            info_text += f"  |  {result.metadata.file_date.strftime('%m/%d/%Y')}"

        info_label = ctk.CTkLabel(
            item_frame,
            text=info_text,
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        info_label.pack(side="left", fill="x", expand=True, padx=5, pady=8)

        # Click on row to toggle checkbox
        def on_row_click(event, idx=item_id):
            var = self._checkbox_vars.get(idx)
            if var:
                var.set(not var.get())
                self._on_checkbox_toggle(idx)

        item_frame.bind("<Button-1>", on_row_click)
        info_label.bind("<Button-1>", on_row_click)
        status_label.bind("<Button-1>", on_row_click)

    def _on_checkbox_toggle(self, item_id: int):
        """Handle checkbox toggle."""
        var = self._checkbox_vars.get(item_id)
        if var:
            if var.get():
                self.selected_ids.add(item_id)
            else:
                self.selected_ids.discard(item_id)
        self._update_selection_count()

    def _select_all(self):
        """Select all results."""
        for item_id, var in self._checkbox_vars.items():
            var.set(True)
            self.selected_ids.add(item_id)
        self._update_selection_count()

    def _clear_selection(self):
        """Clear all selections."""
        for var in self._checkbox_vars.values():
            var.set(False)
        self.selected_ids.clear()
        self._update_selection_count()

    def _update_selection_count(self):
        """Update the selection count label."""
        count = len(self.selected_ids)
        self.selection_label.configure(text=f"Selected: {count} unit{'s' if count != 1 else ''}")

        # Enable/disable export button
        if count > 0:
            self.export_btn.configure(state="normal")
        else:
            self.export_btn.configure(state="disabled")

    def _clear_results(self):
        """Clear results list."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.search_results.clear()
        self.selected_ids.clear()
        self._checkbox_vars.clear()
        self._update_selection_count()

    def _show_error(self, error: str):
        """Show error message."""
        self.count_label.configure(text="Error")
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        error_label = ctk.CTkLabel(
            self.results_frame,
            text=f"Error: {error}",
            text_color="red"
        )
        error_label.pack(pady=20)

    def _export_charts(self):
        """Export charts for selected units."""
        if not self.selected_ids:
            return

        # Get selected results
        selected_results = [
            self.search_results[idx]
            for idx in sorted(self.selected_ids)
            if idx < len(self.search_results)
        ]

        if not selected_results:
            return

        export_format = self.export_format.get()

        if export_format == "pdf":
            self._export_as_pdf(selected_results)
        else:
            self._export_as_pngs(selected_results)

    def _export_as_pngs(self, results: List[AnalysisResult]):
        """Export charts as individual PNG files."""
        # Ask for output directory
        output_dir = filedialog.askdirectory(
            title="Select Output Directory for Charts"
        )

        if not output_dir:
            return

        output_path = Path(output_dir)

        # Show progress
        self.export_btn.configure(state="disabled", text="Exporting...")

        def do_export():
            try:
                exported = []
                for i, result in enumerate(results):
                    try:
                        # Generate filename
                        filename = f"{result.metadata.model}_{result.metadata.serial}_chart.png"
                        filepath = output_path / filename

                        # Handle duplicates
                        counter = 1
                        while filepath.exists():
                            filename = f"{result.metadata.model}_{result.metadata.serial}_chart_{counter}.png"
                            filepath = output_path / filename
                            counter += 1

                        # Export chart
                        self._export_single_chart(result, filepath)
                        exported.append(filepath)

                    except Exception as e:
                        logger.error(f"Failed to export {result.metadata.filename}: {e}")

                self.after(0, lambda: self._export_complete(len(exported), "PNG files"))

            except Exception as e:
                logger.error(f"Export failed: {e}")
                self.after(0, lambda: self._export_error(str(e)))

        get_thread_manager().start_thread(target=do_export, name="export-png")

    def _export_as_pdf(self, results: List[AnalysisResult]):
        """Export charts as combined PDF."""
        # Ask for output file
        default_name = f"charts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_file = filedialog.asksaveasfilename(
            title="Save PDF",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if not output_file:
            return

        # Show progress
        self.export_btn.configure(state="disabled", text="Exporting...")

        def do_export():
            try:
                self._export_multi_page_pdf(results, Path(output_file))
                self.after(0, lambda: self._export_complete(len(results), "PDF"))

            except Exception as e:
                logger.error(f"PDF export failed: {e}")
                self.after(0, lambda: self._export_error(str(e)))

        get_thread_manager().start_thread(target=do_export, name="export-pdf")

    def _export_single_chart(self, result: AnalysisResult, output_path: Path):
        """Export a comprehensive chart to file (matching Analyze page format)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np

        # Use light mode for export
        plt.style.use('default')

        if not result.tracks:
            raise ValueError("No track data to export")

        track = result.tracks[0]

        # RECALCULATE fail points using actual spec limits
        actual_fail_count = 0
        upper_limits = track.upper_limits
        lower_limits = track.lower_limits

        # Fallback to flat linearity_spec if limits not stored
        if not upper_limits or not lower_limits:
            if track.linearity_spec and track.linearity_spec > 0 and track.error_data:
                upper_limits = [track.linearity_spec] * len(track.error_data)
                lower_limits = [-track.linearity_spec] * len(track.error_data)

        if upper_limits and lower_limits and track.error_data:
            shifted_errors = [e + track.optimal_offset for e in track.error_data]
            for i, e in enumerate(shifted_errors):
                if i < len(upper_limits) and i < len(lower_limits):
                    if upper_limits[i] is not None and lower_limits[i] is not None:
                        if e > upper_limits[i] or e < lower_limits[i]:
                            actual_fail_count += 1

        corrected_values = {
            'fail_points': actual_fail_count,
            'linearity_pass': actual_fail_count == 0,
        }

        # Create figure with subplots (larger for detailed export)
        fig = plt.figure(figsize=(14, 10), dpi=150, facecolor='white')

        # Title (positioned at y=0.97)
        title = f'Laser Trim Analysis - {result.metadata.model} / {result.metadata.serial}'
        fig.suptitle(title, fontsize=16, fontweight='bold', color='black', y=0.97)

        # Add filename on separate line below title (y=0.94)
        fig.text(0.95, 0.94, f"File: {result.metadata.filename}",
                fontsize=9, ha='right', va='top', color='gray', style='italic')

        # Grid spec for layout (top adjusted down for title spacing)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3,
                              left=0.08, right=0.95, top=0.90, bottom=0.08)

        # Main error plot (top 2/3)
        ax_main = fig.add_subplot(gs[0:2, :])
        self._plot_error_vs_position_export(ax_main, track)

        # Unit Info (bottom left)
        ax_info = fig.add_subplot(gs[2, 0])
        self._plot_unit_info(ax_info, result, track)

        # Metrics summary (bottom middle)
        ax_metrics = fig.add_subplot(gs[2, 1])
        self._plot_metrics_summary(ax_metrics, track, corrected_values)

        # Status display (bottom right)
        ax_status = fig.add_subplot(gs[2, 2])
        self._plot_status_display(ax_status, track, result, corrected_values)

        # Save
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def _export_multi_page_pdf(self, results: List[AnalysisResult], output_path: Path):
        """Export multiple comprehensive charts to a single PDF."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.patches import Rectangle
        import numpy as np

        plt.style.use('default')

        with PdfPages(output_path) as pdf:
            for result in results:
                if not result.tracks:
                    continue

                track = result.tracks[0]

                # RECALCULATE fail points using actual spec limits
                actual_fail_count = 0
                upper_limits = track.upper_limits
                lower_limits = track.lower_limits

                if not upper_limits or not lower_limits:
                    if track.linearity_spec and track.linearity_spec > 0 and track.error_data:
                        upper_limits = [track.linearity_spec] * len(track.error_data)
                        lower_limits = [-track.linearity_spec] * len(track.error_data)

                if upper_limits and lower_limits and track.error_data:
                    shifted_errors = [e + track.optimal_offset for e in track.error_data]
                    for i, e in enumerate(shifted_errors):
                        if i < len(upper_limits) and i < len(lower_limits):
                            if upper_limits[i] is not None and lower_limits[i] is not None:
                                if e > upper_limits[i] or e < lower_limits[i]:
                                    actual_fail_count += 1

                corrected_values = {
                    'fail_points': actual_fail_count,
                    'linearity_pass': actual_fail_count == 0,
                }

                # Create figure with subplots (letter size for PDF)
                fig = plt.figure(figsize=(11, 8.5), facecolor='white')

                # Title (positioned at y=0.96)
                title = f'Laser Trim Analysis - {result.metadata.model} / {result.metadata.serial}'
                fig.suptitle(title, fontsize=14, fontweight='bold', color='black', y=0.96)

                # Add filename on separate line below title (y=0.93)
                fig.text(0.95, 0.93, f"File: {result.metadata.filename}",
                        fontsize=8, ha='right', va='top', color='gray', style='italic')

                # Grid spec for layout (top adjusted down for title spacing)
                gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3,
                                      left=0.08, right=0.95, top=0.88, bottom=0.08)

                # Main error plot (top 2/3)
                ax_main = fig.add_subplot(gs[0:2, :])
                self._plot_error_vs_position_export(ax_main, track)

                # Unit Info (bottom left)
                ax_info = fig.add_subplot(gs[2, 0])
                self._plot_unit_info(ax_info, result, track)

                # Metrics summary (bottom middle)
                ax_metrics = fig.add_subplot(gs[2, 1])
                self._plot_metrics_summary(ax_metrics, track, corrected_values)

                # Status display (bottom right)
                ax_status = fig.add_subplot(gs[2, 2])
                self._plot_status_display(ax_status, track, result, corrected_values)

                # Add to PDF
                pdf.savefig(fig, facecolor='white')
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

        if not track.position_data or not track.error_data:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return

        positions = np.array(track.position_data)
        errors = np.array(track.error_data)

        # Apply offset
        offset = track.optimal_offset
        errors_shifted = errors + offset

        # Plot untrimmed data if available
        if track.untrimmed_positions and track.untrimmed_errors:
            ax.plot(track.untrimmed_positions, track.untrimmed_errors,
                   '--', linewidth=1.5, label='Untrimmed',
                   color=QA_COLORS['untrimmed'], alpha=0.6)

        # Plot trimmed/shifted data
        ax.plot(positions, errors_shifted, '-', linewidth=2,
               label='Trimmed (Offset Applied)', color=QA_COLORS['trimmed'])

        # Get spec limits - use stored limits or calculate from linearity_spec
        upper_limits = track.upper_limits
        lower_limits = track.lower_limits

        # Fallback: if limits not stored, use flat linearity_spec
        if (not upper_limits or not lower_limits or len(upper_limits) != len(positions)):
            if track.linearity_spec and track.linearity_spec > 0:
                upper_limits = [track.linearity_spec] * len(positions)
                lower_limits = [-track.linearity_spec] * len(positions)

        # Plot spec limits (handle None = no limit at that position)
        if upper_limits and lower_limits and len(upper_limits) == len(positions):
            # Convert None to NaN for matplotlib (creates gaps in the line)
            upper_plot = np.array([u if u is not None else np.nan for u in upper_limits])
            lower_plot = np.array([l if l is not None else np.nan for l in lower_limits])

            ax.plot(positions, upper_plot, '--', linewidth=1.5,
                   label=f'Spec Limits (+/-{track.linearity_spec:.4f})', color=QA_COLORS['spec_limit'])
            ax.plot(positions, lower_plot, '--', linewidth=1.5,
                   color=QA_COLORS['spec_limit'])
            # Fill only where limits are defined
            ax.fill_between(positions, lower_plot, upper_plot,
                           alpha=0.15, color=QA_COLORS['spec_limit'],
                           where=~np.isnan(upper_plot) & ~np.isnan(lower_plot))

        # Find and mark fail points (skip positions with None = no spec)
        fail_indices = []
        if upper_limits and lower_limits:
            for i, e in enumerate(errors_shifted):
                if i < len(upper_limits) and i < len(lower_limits):
                    if upper_limits[i] is not None and lower_limits[i] is not None:
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

        # Use recalculated lin_pass from fail_indices, not stored value
        lin_pass = len(fail_indices) == 0
        sigma_pass = track.sigma_pass
        if lin_pass and sigma_pass:
            status_str = "PASS"
        elif lin_pass and not sigma_pass:
            status_str = "WARNING (Sigma Fail)"
        elif not lin_pass and sigma_pass:
            status_str = "WARNING (Lin Fail)"
        else:
            status_str = "FAIL"
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

    def _plot_metrics_summary(self, ax, track: TrackData, corrected_values: Dict[str, Any] = None):
        """Plot analysis metrics summary (light mode)."""
        ax.axis('off')
        ax.set_facecolor('white')

        # Use corrected values if provided (recalculated from actual spec limits)
        fail_points = corrected_values['fail_points'] if corrected_values else track.linearity_fail_points
        linearity_pass = corrected_values['linearity_pass'] if corrected_values else track.linearity_pass

        metrics = [
            f"Sigma Gradient: {track.sigma_gradient:.6f}",
            f"Sigma Threshold: {track.sigma_threshold:.6f}",
            f"Sigma Pass: {'YES' if track.sigma_pass else 'NO'}",
            "",
            f"Optimal Offset: {track.optimal_offset:.6f}",
            f"Linearity Error: {track.linearity_error:.6f}",
            f"Fail Points: {fail_points}",
            f"Linearity Pass: {'YES' if linearity_pass else 'NO'}",
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
            if 'NO' in metric and 'Pass:' in metric:
                color = '#e74c3c'
            elif 'YES' in metric:
                color = '#27ae60'

            ax.text(0.05, y_pos, metric, fontsize=10, transform=ax.transAxes,
                   va='top', color=color)

    def _plot_status_display(self, ax, track: TrackData, result: AnalysisResult, corrected_values: Dict[str, Any] = None):
        """Plot status display panel (light mode)."""
        from matplotlib.patches import Rectangle

        ax.axis('off')
        ax.set_facecolor('white')

        # Use corrected values if provided (recalculated from actual spec limits)
        fail_points = corrected_values['fail_points'] if corrected_values else track.linearity_fail_points
        linearity_pass = corrected_values['linearity_pass'] if corrected_values else track.linearity_pass

        # Determine overall status using corrected linearity pass
        # WARNING when one passes and one fails
        if track.sigma_pass and linearity_pass:
            status = "PASS"
            color = '#27ae60'
        elif track.sigma_pass or linearity_pass:
            status = "WARNING"
            color = '#f39c12'
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
        if not linearity_pass:
            details.append(f"Linearity: FAIL ({fail_points} pts)")

        if details:
            ax.text(0.5, 0.45, "\n".join(details), ha='center', va='center',
                   fontsize=11, color=color, transform=ax.transAxes)

        # Analysis timestamp
        ax.text(0.5, 0.15, f"Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
               ha='center', va='center', fontsize=9, color='gray',
               style='italic', transform=ax.transAxes)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def _export_complete(self, count: int, format_name: str):
        """Handle export completion."""
        self.export_btn.configure(state="normal", text="Export Charts")
        messagebox.showinfo(
            "Export Complete",
            f"Successfully exported {count} chart{'s' if count != 1 else ''} as {format_name}."
        )

    def _export_error(self, error: str):
        """Handle export error."""
        self.export_btn.configure(state="normal", text="Export Charts")
        messagebox.showerror("Export Error", f"Failed to export charts:\n{error}")

    def on_show(self):
        """Called when page is shown."""
        # Load models list
        try:
            db = get_database()
            models = db.get_models_list()
            model_values = ["All Models"] + models
            self.model_filter.configure(values=model_values)
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
