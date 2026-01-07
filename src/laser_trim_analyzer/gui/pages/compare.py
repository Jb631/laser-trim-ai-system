"""
Compare Page - Final Test vs Trim comparison with overlay charts.

This page allows viewing Final Test results and comparing them
with linked Trim results using overlay charts.
"""

import customtkinter as ctk
import logging
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional, List, Dict, Any, Set, TYPE_CHECKING

from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.utils.threads import get_thread_manager
from laser_trim_analyzer.gui.widgets.scrollable_combobox import ScrollableComboBox

# Lazy import for ChartWidget
if TYPE_CHECKING:
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget

logger = logging.getLogger(__name__)


class ComparePage(ctk.CTkFrame):
    """
    Compare page for Final Test vs Trim comparison.

    Features:
    - Browse Final Test results from database
    - Filter by model and date
    - View comparison charts overlaying Trim and Final Test data
    - Show match confidence and days since trim
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.current_comparison: Optional[Dict[str, Any]] = None
        self.comparison_pairs: List[Dict[str, Any]] = []

        # Pagination
        self._page_size = 20
        self._current_page = 0
        self._total_pages = 1

        # Selection tracking for export
        self._selected_ids: Set[int] = set()
        self._checkbox_vars: Dict[int, ctk.BooleanVar] = {}

        # Lazy chart initialization
        self._chart_initialized = False
        self.chart = None

        self._create_ui()

    def _create_ui(self):
        """Create the compare page UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)

        header = ctk.CTkLabel(
            header_frame,
            text="Final Test Analysis",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            header_frame,
            text="Browse and analyze Final Test results",
            text_color="gray",
            font=ctk.CTkFont(size=12)
        )
        subtitle.grid(row=1, column=0, sticky="w")

        # Filter controls
        filter_frame = ctk.CTkFrame(self)
        filter_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))

        # Model filter
        ctk.CTkLabel(filter_frame, text="Model:").pack(side="left", padx=(15, 5), pady=15)
        self.model_filter = ScrollableComboBox(
            filter_frame,
            values=["All Models"],
            command=self._on_filter_change,
            width=150,
            dropdown_height=300,
        )
        self.model_filter.set("All Models")
        self.model_filter.pack(side="left", padx=5, pady=15)

        # Date range filter (filters by test date)
        ctk.CTkLabel(filter_frame, text="From:").pack(side="left", padx=(20, 5), pady=15)
        self.date_from = ctk.CTkEntry(
            filter_frame,
            placeholder_text="MM/DD/YYYY",
            width=90
        )
        self.date_from.pack(side="left", padx=2, pady=15)
        self.date_from.bind("<Return>", lambda e: self._load_comparisons())
        # Set default to 30 days ago
        default_from = (datetime.now() - timedelta(days=30)).strftime("%m/%d/%Y")
        self.date_from.insert(0, default_from)

        ctk.CTkLabel(filter_frame, text="To:").pack(side="left", padx=(10, 5), pady=15)
        self.date_to = ctk.CTkEntry(
            filter_frame,
            placeholder_text="MM/DD/YYYY",
            width=90
        )
        self.date_to.pack(side="left", padx=2, pady=15)
        self.date_to.bind("<Return>", lambda e: self._load_comparisons())
        # Set default to today
        self.date_to.insert(0, datetime.now().strftime("%m/%d/%Y"))

        # Status filter
        ctk.CTkLabel(filter_frame, text="Status:").pack(side="left", padx=(15, 5), pady=15)
        self.status_filter = ctk.CTkOptionMenu(
            filter_frame,
            values=["All", "Pass", "Fail"],
            command=self._on_filter_change,
            width=80
        )
        self.status_filter.set("All")
        self.status_filter.pack(side="left", padx=5, pady=15)

        # Serial filter (partial match)
        ctk.CTkLabel(filter_frame, text="Serial:").pack(side="left", padx=(15, 5), pady=15)
        self.serial_filter = ctk.CTkEntry(
            filter_frame,
            placeholder_text="Search...",
            width=80
        )
        self.serial_filter.pack(side="left", padx=5, pady=15)
        self.serial_filter.bind("<Return>", lambda e: self._load_comparisons())

        # Linked only filter
        self.linked_only_var = ctk.BooleanVar(value=False)
        linked_check = ctk.CTkCheckBox(
            filter_frame,
            text="Linked Only",
            variable=self.linked_only_var,
            command=self._on_filter_change
        )
        linked_check.pack(side="left", padx=10, pady=15)

        # Refresh button
        refresh_btn = ctk.CTkButton(
            filter_frame,
            text="Refresh",
            command=self._load_comparisons,
            width=100
        )
        refresh_btn.pack(side="right", padx=15, pady=15)

        # Export Selected button
        self.export_selected_btn = ctk.CTkButton(
            filter_frame,
            text="Export Selected",
            command=self._export_selected,
            width=120,
            state="disabled"
        )
        self.export_selected_btn.pack(side="right", padx=5, pady=15)

        # Export Current button
        self.export_current_btn = ctk.CTkButton(
            filter_frame,
            text="Export Chart",
            command=self._export_current,
            width=100,
            state="disabled"
        )
        self.export_current_btn.pack(side="right", padx=5, pady=15)

        # Fix Missing Tracks button
        self.fix_tracks_btn = ctk.CTkButton(
            filter_frame,
            text="Fix Missing Tracks",
            command=self._fix_missing_tracks,
            width=140,
            fg_color="#f39c12",
            hover_color="#e67e22"
        )
        self.fix_tracks_btn.pack(side="right", padx=5, pady=15)

        # Results count
        self.count_label = ctk.CTkLabel(
            filter_frame,
            text="",
            text_color="gray"
        )
        self.count_label.pack(side="right", padx=10, pady=15)

        # Main content - split view
        content = ctk.CTkFrame(self)
        content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        content.grid_columnconfigure(0, weight=1, minsize=300)
        content.grid_columnconfigure(1, weight=3)
        content.grid_rowconfigure(0, weight=1)

        # Left panel - comparison list
        list_frame = ctk.CTkFrame(content)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        list_frame.grid_rowconfigure(1, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        # Header with label and select controls
        list_header = ctk.CTkFrame(list_frame, fg_color="transparent")
        list_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        list_header.grid_columnconfigure(0, weight=1)

        list_label = ctk.CTkLabel(
            list_header,
            text="Final Test Results",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        list_label.grid(row=0, column=0, padx=5, sticky="w")

        # Select All / None buttons
        select_frame = ctk.CTkFrame(list_header, fg_color="transparent")
        select_frame.grid(row=0, column=1, sticky="e")

        select_all_btn = ctk.CTkButton(
            select_frame,
            text="All",
            width=40,
            height=24,
            font=ctk.CTkFont(size=11),
            command=self._select_all
        )
        select_all_btn.pack(side="left", padx=2)

        select_none_btn = ctk.CTkButton(
            select_frame,
            text="None",
            width=45,
            height=24,
            font=ctk.CTkFont(size=11),
            command=self._select_none
        )
        select_none_btn.pack(side="left", padx=2)

        # Scrollable list
        self.list_frame = ctk.CTkScrollableFrame(list_frame)
        self.list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 5))

        # Pagination controls
        pagination_frame = ctk.CTkFrame(list_frame, fg_color="transparent")
        pagination_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        self.prev_btn = ctk.CTkButton(
            pagination_frame,
            text="<",
            width=40,
            command=self._prev_page,
            state="disabled"
        )
        self.prev_btn.pack(side="left", padx=5)

        self.page_label = ctk.CTkLabel(pagination_frame, text="Page 1/1")
        self.page_label.pack(side="left", padx=10)

        self.next_btn = ctk.CTkButton(
            pagination_frame,
            text=">",
            width=40,
            command=self._next_page,
            state="disabled"
        )
        self.next_btn.pack(side="left", padx=5)

        # Right panel - comparison details
        details_frame = ctk.CTkFrame(content)
        details_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        details_frame.grid_rowconfigure(1, weight=1)
        details_frame.grid_columnconfigure(0, weight=1)

        details_label = ctk.CTkLabel(
            details_frame,
            text="Comparison Details",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        details_label.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        # Details content area
        self.details_content = ctk.CTkFrame(details_frame)
        self.details_content.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.details_content.grid_rowconfigure(1, weight=1)
        self.details_content.grid_columnconfigure(0, weight=1)

        # Info panel at top
        self.info_frame = ctk.CTkFrame(self.details_content)
        self.info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.info_label = ctk.CTkLabel(
            self.info_frame,
            text="Select a Final Test result to view comparison",
            text_color="gray",
            justify="left"
        )
        self.info_label.pack(padx=15, pady=15, anchor="w")

        # Chart area
        self.chart_frame = ctk.CTkFrame(self.details_content)
        self.chart_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Placeholder for chart
        self.chart_placeholder = ctk.CTkLabel(
            self.chart_frame,
            text="Comparison chart will appear here",
            text_color="gray"
        )
        self.chart_placeholder.pack(expand=True, fill="both", padx=20, pady=20)

    def _ensure_chart_initialized(self):
        """Lazily initialize the chart widget."""
        if self._chart_initialized:
            return

        try:
            from laser_trim_analyzer.gui.widgets.chart import ChartWidget

            # Remove placeholder
            self.chart_placeholder.destroy()

            # Create chart
            self.chart = ChartWidget(self.chart_frame)
            self.chart.pack(expand=True, fill="both", padx=5, pady=5)

            self._chart_initialized = True
            logger.info("Compare page chart initialized")

        except Exception as e:
            logger.error(f"Failed to initialize chart: {e}")

    def _load_comparisons(self):
        """Load Final Test comparison pairs from database."""
        self.count_label.configure(text="Loading...")

        def fetch():
            try:
                db = get_database()

                # Get filter values
                model = self.model_filter.get()
                if model == "All Models":
                    model = None

                # Parse date range
                date_from = None
                date_to = None

                from_str = self.date_from.get().strip()
                to_str = self.date_to.get().strip()

                # Parse From date
                if from_str:
                    try:
                        date_from = datetime.strptime(from_str, "%m/%d/%Y")
                    except ValueError:
                        try:
                            date_from = datetime.strptime(from_str, "%m/%d/%y")
                        except ValueError:
                            logger.warning(f"Invalid From date format: {from_str}")

                # Parse To date
                if to_str:
                    try:
                        date_to = datetime.strptime(to_str, "%m/%d/%Y")
                    except ValueError:
                        try:
                            date_to = datetime.strptime(to_str, "%m/%d/%y")
                        except ValueError:
                            logger.warning(f"Invalid To date format: {to_str}")

                # Get other filters
                serial = self.serial_filter.get().strip()
                status = self.status_filter.get()
                linked_only = self.linked_only_var.get()

                # Get comparison pairs using new search method
                pairs = db.search_final_tests(
                    model=model,
                    serial=serial if serial else None,
                    date_from=date_from,
                    date_to=date_to,
                    status=status,
                    linked_only=linked_only,
                    limit=500
                )

                # Get models for filter dropdown
                models = db.get_final_test_models_list()

                self.after(0, lambda: self._display_comparisons(pairs, models))

            except Exception as e:
                logger.exception(f"Error loading comparisons: {e}")
                self.after(0, lambda: self._show_error(str(e)))

        get_thread_manager().start_thread(target=fetch, name="fetch-comparisons")

    def _display_comparisons(self, pairs: List[Dict], models: List[str]):
        """Display comparison pairs in the list."""
        self.comparison_pairs = pairs

        # Update model filter
        all_models = ["All Models"] + sorted(models)
        current = self.model_filter.get()
        self.model_filter.configure(values=all_models)
        if current in all_models:
            self.model_filter.set(current)

        # Update count
        self.count_label.configure(text=f"{len(pairs)} results")

        # Calculate pagination
        self._total_pages = max(1, (len(pairs) + self._page_size - 1) // self._page_size)
        self._current_page = min(self._current_page, self._total_pages - 1)

        self._display_current_page()

    def _display_current_page(self):
        """Display current page of comparison pairs."""
        # Clear existing items
        for widget in self.list_frame.winfo_children():
            widget.destroy()

        if not self.comparison_pairs:
            no_data = ctk.CTkLabel(
                self.list_frame,
                text="No Final Test results found.\n\nProcess Final Test files first.",
                text_color="gray",
                justify="center"
            )
            no_data.pack(expand=True, pady=50)
            self._update_pagination()
            return

        # Get current page items
        start_idx = self._current_page * self._page_size
        end_idx = min(start_idx + self._page_size, len(self.comparison_pairs))
        page_items = self.comparison_pairs[start_idx:end_idx]

        # Create list items
        for i, pair in enumerate(page_items):
            self._create_list_item(pair, start_idx + i)

        self._update_pagination()

    def _create_list_item(self, pair: Dict, index: int):
        """Create a list item for a comparison pair."""
        ft_id = pair.get("final_test_id")

        item_frame = ctk.CTkFrame(self.list_frame)
        item_frame.pack(fill="x", padx=5, pady=2)

        # Checkbox for selection
        var = ctk.BooleanVar(value=ft_id in self._selected_ids)
        self._checkbox_vars[ft_id] = var

        checkbox = ctk.CTkCheckBox(
            item_frame,
            text="",
            width=20,
            variable=var,
            command=lambda: self._on_checkbox_toggle(ft_id, var)
        )
        checkbox.pack(side="left", padx=(5, 0), pady=10)

        # Status indicator
        status = pair.get("final_test_status", "UNKNOWN")
        status_colors = {
            "Pass": "#2ecc71",
            "Fail": "#e74c3c",
            "Warning": "#f39c12",
        }
        status_color = status_colors.get(status, "gray")

        status_label = ctk.CTkLabel(
            item_frame,
            text="",
            width=8,
            fg_color=status_color,
            corner_radius=4
        )
        status_label.pack(side="left", padx=(5, 5), pady=10)

        # Info
        info_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        # Model/Serial
        model = pair.get("model", "Unknown")
        serial = pair.get("serial", "Unknown")
        title = f"{model} - SN:{serial}"

        title_label = ctk.CTkLabel(
            info_frame,
            text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        title_label.pack(fill="x")

        # Date and link status
        test_date = pair.get("final_test_date")
        if test_date:
            if isinstance(test_date, datetime):
                date_str = test_date.strftime("%Y-%m-%d")
            else:
                date_str = str(test_date)[:10]
        else:
            date_str = "Unknown date"

        is_linked = pair.get("is_linked", False)
        days_since = pair.get("days_since_trim")
        if is_linked and days_since is not None:
            link_text = f"Linked ({days_since}d)"
            link_color = "#2ecc71"
        elif is_linked:
            link_text = "Linked"
            link_color = "#2ecc71"
        else:
            link_text = "No match"
            link_color = "gray"

        details_label = ctk.CTkLabel(
            info_frame,
            text=f"{date_str}  |  {link_text}",
            font=ctk.CTkFont(size=10),
            text_color=link_color,
            anchor="w"
        )
        details_label.pack(fill="x")

        # Click handler
        def on_click(event, p=pair):
            self._show_comparison(p)

        item_frame.bind("<Button-1>", on_click)
        for child in item_frame.winfo_children():
            child.bind("<Button-1>", on_click)
        for child in info_frame.winfo_children():
            child.bind("<Button-1>", on_click)

        # Hover effect
        def on_enter(event):
            item_frame.configure(fg_color=("gray80", "gray25"))

        def on_leave(event):
            item_frame.configure(fg_color=("gray90", "gray17"))

        item_frame.bind("<Enter>", on_enter)
        item_frame.bind("<Leave>", on_leave)

    def _show_comparison(self, pair: Dict):
        """Show comparison details for selected pair."""
        self.current_comparison = pair

        # Enable export current button
        self.export_current_btn.configure(state="normal")

        # Update info panel
        model = pair.get("model", "Unknown")
        serial = pair.get("serial", "Unknown")
        status = pair.get("final_test_status", "Unknown")
        linearity_pass = pair.get("linearity_pass")

        test_date = pair.get("final_test_date")
        if test_date and isinstance(test_date, datetime):
            date_str = test_date.strftime("%Y-%m-%d %H:%M")
        else:
            date_str = str(test_date) if test_date else "Unknown"

        is_linked = pair.get("is_linked", False)
        days_since = pair.get("days_since_trim")
        confidence = pair.get("match_confidence")

        info_text = f"""Model: {model}
Serial: {serial}
Test Date: {date_str}
Status: {status}
Linearity: {"PASS" if linearity_pass else "FAIL" if linearity_pass is False else "N/A"}

"""
        if is_linked:
            linked_trim = pair.get("linked_trim", {})
            trim_file = linked_trim.get("filename", "Unknown") if linked_trim else "Unknown"
            info_text += f"""Linked Trim: {trim_file}
Days Since Trim: {days_since if days_since else 'N/A'}
Match Confidence: {confidence*100:.0f}% if confidence else 'N/A'"""
        else:
            info_text += "No linked trim result found"

        self.info_label.configure(text=info_text)

        # Load full comparison data and display chart
        self._load_comparison_chart(pair.get("final_test_id"))

    def _load_comparison_chart(self, final_test_id: int):
        """Load full comparison data and display chart."""
        if not final_test_id:
            return

        def fetch():
            try:
                db = get_database()
                data = db.get_comparison_data(final_test_id)
                self.after(0, lambda: self._display_comparison_chart(data))
            except Exception as e:
                logger.exception(f"Error loading comparison data: {e}")

        get_thread_manager().start_thread(target=fetch, name="load-comparison-chart")

    def _display_comparison_chart(self, data: Optional[Dict]):
        """Display the comparison overlay chart."""
        if not data:
            logger.warning("Compare chart: No data received")
            return

        self._ensure_chart_initialized()
        if not self.chart:
            logger.warning("Compare chart: Chart not initialized")
            return

        final_test = data.get("final_test", {})
        trim = data.get("trim")

        # Debug logging
        logger.info(f"Compare chart: final_test keys={list(final_test.keys())}")
        logger.info(f"Compare chart: tracks count={len(final_test.get('tracks', []))}")

        # Get track data
        ft_tracks = final_test.get("tracks", [])
        if not ft_tracks:
            # Show message when Final Test has no track data
            self._ensure_chart_initialized()
            if self.chart:
                self.chart.clear()
                ax = self.chart.figure.add_subplot(111)
                self.chart._style_axis(ax)
                ax.text(0.5, 0.5, "No track data for this Final Test\n\n"
                       "Re-process the file to fix this.",
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                self.chart.figure.tight_layout()
                self.chart.canvas.draw()
            return

        ft_track = ft_tracks[0]  # Use first track
        # Support both "positions" and "electrical_angles" (new format)
        ft_positions = ft_track.get("positions") or ft_track.get("electrical_angles", [])
        ft_errors = ft_track.get("errors", [])

        # Get spec limits from Final Test track (preferred) or Trim track
        upper_limits = ft_track.get("upper_limits", [])
        lower_limits = ft_track.get("lower_limits", [])
        spec_positions = ft_positions  # Use Final Test positions for spec limits

        # Prepare chart data
        chart_data = {
            "final_test": {
                "positions": ft_positions,
                "errors": ft_errors,
                "label": f"Final Test ({final_test.get('filename', 'Unknown')[:20]})",
            },
            "upper_limits": upper_limits,
            "lower_limits": lower_limits,
            "spec_positions": spec_positions,
        }

        if trim:
            trim_tracks = trim.get("tracks", [])
            if trim_tracks:
                trim_track = trim_tracks[0]
                trim_positions = trim_track.get("positions", [])
                trim_errors = trim_track.get("errors", [])
                trim_offset = trim_track.get("offset", 0) or trim_track.get("optimal_offset", 0)

                chart_data["trim"] = {
                    "positions": trim_positions,
                    "errors": trim_errors,
                    "offset": trim_offset,
                    "label": f"Trim ({trim.get('filename', 'Unknown')[:20]})",
                }

                # If Final Test doesn't have spec limits, try to get from Trim
                if not upper_limits or not lower_limits:
                    trim_upper = trim_track.get("upper_limits", [])
                    trim_lower = trim_track.get("lower_limits", [])
                    if trim_upper and trim_lower:
                        chart_data["upper_limits"] = trim_upper
                        chart_data["lower_limits"] = trim_lower
                        chart_data["spec_positions"] = trim_positions
            else:
                # Trim record exists but has no track data
                chart_data["trim_missing_data"] = True

        # Plot comparison
        self._plot_comparison(chart_data)

    def _plot_comparison(self, chart_data: Dict):
        """Plot the comparison overlay chart with spec limits."""
        import numpy as np
        from laser_trim_analyzer.gui.widgets.chart import COLORS

        if not self.chart:
            return

        self.chart.clear()

        ax = self.chart.figure.add_subplot(111)

        # Apply dark mode styling
        self.chart._style_axis(ax)

        # Get data
        ft_data = chart_data.get("final_test", {})
        trim_data = chart_data.get("trim", {})
        upper_limits = chart_data.get("upper_limits")
        lower_limits = chart_data.get("lower_limits")
        spec_positions = chart_data.get("spec_positions")

        has_ft_data = bool(ft_data.get("positions") and ft_data.get("errors"))
        has_trim_data = bool(trim_data.get("positions") and trim_data.get("errors"))

        # Show message if no data
        if not has_ft_data and not has_trim_data:
            ax.text(0.5, 0.5, "No track data available",
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            self.chart.figure.tight_layout()
            self.chart.canvas.draw()
            return

        # Plot Trim data first (if available) - so Final Test appears on top
        if has_trim_data:
            trim_errors = trim_data["errors"]
            offset = trim_data.get("offset", 0)
            if offset:
                trim_errors = [e + offset for e in trim_errors]

            ax.plot(
                trim_data["positions"],
                trim_errors,
                color='#3498db',  # Blue for Trim (laser trim data)
                linewidth=1.5,
                label=trim_data.get("label", "Trim"),
                alpha=0.9
            )

        # Plot Final Test data
        if has_ft_data:
            ax.plot(
                ft_data["positions"],
                ft_data["errors"],
                color='#27ae60',  # Green for Final Test (post-assembly)
                linewidth=1.5,
                label=ft_data.get("label", "Final Test"),
                alpha=0.9
            )

        # Plot specification limits (matching Analyze page style)
        if upper_limits and lower_limits and spec_positions:
            # Convert None to NaN for matplotlib (creates gaps)
            upper_plot = np.array([u if u is not None else np.nan for u in upper_limits])
            lower_plot = np.array([l if l is not None else np.nan for l in lower_limits])
            pos_array = np.array(spec_positions[:len(upper_limits)])

            ax.plot(pos_array, upper_plot, color=COLORS.get('spec_limit', '#e74c3c'),
                   linestyle='--', linewidth=1, alpha=0.8, label='Spec Limits')
            ax.plot(pos_array, lower_plot, color=COLORS.get('spec_limit', '#e74c3c'),
                   linestyle='--', linewidth=1, alpha=0.8)

            # Fill between limits
            ax.fill_between(pos_array, lower_plot, upper_plot,
                           alpha=0.1, color=COLORS.get('spec_limit', '#e74c3c'),
                           where=~np.isnan(upper_plot) & ~np.isnan(lower_plot))

            # Find and mark fail points for Final Test data
            if has_ft_data:
                fail_indices = []
                ft_positions = ft_data["positions"]
                ft_errors = ft_data["errors"]

                for i, e in enumerate(ft_errors):
                    if i < len(upper_limits) and i < len(lower_limits):
                        if upper_limits[i] is not None and lower_limits[i] is not None:
                            if e > upper_limits[i] or e < lower_limits[i]:
                                fail_indices.append(i)

                if fail_indices:
                    fail_pos = [ft_positions[i] for i in fail_indices]
                    fail_err = [ft_errors[i] for i in fail_indices]
                    ax.scatter(fail_pos, fail_err, color='#e74c3c',
                              s=100, marker='x', linewidth=3,
                              label=f'Fail Points ({len(fail_indices)})', zorder=5)

        # Show note if trim data is missing for linked unit
        if has_ft_data and not has_trim_data and chart_data.get("trim_missing_data"):
            ax.text(0.02, 0.98, "Note: Linked Trim has no track data (re-process to fix)",
                   ha='left', va='top', transform=ax.transAxes,
                   fontsize=9, color='#f39c12', style='italic')

        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        # Build title
        if has_trim_data:
            title = "Trim vs Final Test Comparison"
        else:
            title = "Final Test Linearity"

        # Labels and legend (matching Analyze page style)
        ax.set_xlabel('Position', fontsize=self.chart.style.font_size)
        ax.set_ylabel('Error', fontsize=self.chart.style.font_size)
        ax.set_title(title, fontsize=self.chart.style.title_size)
        ax.legend(loc='best', fontsize=self.chart.style.font_size - 2)
        ax.grid(True, alpha=0.3, color=COLORS.get('grid', 'gray'))

        self.chart.figure.tight_layout()
        self.chart.canvas.draw()

    def _update_pagination(self):
        """Update pagination controls."""
        self.page_label.configure(text=f"Page {self._current_page + 1}/{self._total_pages}")
        self.prev_btn.configure(state="normal" if self._current_page > 0 else "disabled")
        self.next_btn.configure(state="normal" if self._current_page < self._total_pages - 1 else "disabled")

    def _prev_page(self):
        """Go to previous page."""
        if self._current_page > 0:
            self._current_page -= 1
            self._display_current_page()

    def _next_page(self):
        """Go to next page."""
        if self._current_page < self._total_pages - 1:
            self._current_page += 1
            self._display_current_page()

    def _on_filter_change(self, *args):
        """Handle filter changes."""
        self._current_page = 0
        self._load_comparisons()

    def _fix_missing_tracks(self):
        """Re-parse Final Test and Trim files that have missing track data."""
        from pathlib import Path
        from laser_trim_analyzer.core.final_test_parser import FinalTestParser
        from laser_trim_analyzer.core.processor import AnalysisProcessor

        self.fix_tracks_btn.configure(state="disabled", text="Fixing...")
        self.count_label.configure(text="Finding records with missing tracks...")

        def fix():
            try:
                db = get_database()
                ft_parser = FinalTestParser()
                trim_processor = AnalysisProcessor()

                fixed_ft = 0
                failed_ft = 0
                fixed_trim = 0
                failed_trim = 0

                # Phase 1: Fix Final Test tracks
                ft_missing = db.get_final_tests_missing_tracks()
                total_ft = len(ft_missing)

                for i, record in enumerate(ft_missing):
                    self.after(0, lambda i=i, t=total_ft: self.count_label.configure(
                        text=f"Fixing Final Test {i+1}/{t}..."
                    ))

                    file_path = Path(record["file_path"]) if record.get("file_path") else None
                    if not file_path or not file_path.exists():
                        model = record.get("model", "")
                        filename = record.get("filename", "")
                        alt_path = Path(f"test_files/Final Test files/{model}/{filename}")
                        if alt_path.exists():
                            file_path = alt_path
                        else:
                            logger.warning(f"Final Test file not found: {record.get('file_path')}")
                            failed_ft += 1
                            continue

                    try:
                        result = ft_parser.parse_file(file_path)
                        tracks = result.get("tracks", [])
                        if tracks and db.update_final_test_tracks(record["id"], tracks):
                            fixed_ft += 1
                        else:
                            failed_ft += 1
                    except Exception as e:
                        logger.error(f"Error fixing Final Test {record.get('filename')}: {e}")
                        failed_ft += 1

                # Phase 2: Fix Trim tracks (linked ones only)
                # Note: Some "Trim" records may actually point to Final Test files
                # if the original data was imported incorrectly. We'll try both parsers.
                trim_missing = db.get_trim_records_missing_tracks(linked_only=True)
                total_trim = len(trim_missing)

                for i, record in enumerate(trim_missing):
                    self.after(0, lambda i=i, t=total_trim: self.count_label.configure(
                        text=f"Fixing Trim {i+1}/{t}..."
                    ))

                    file_path = Path(record["file_path"]) if record.get("file_path") else None
                    if not file_path or not file_path.exists():
                        model = record.get("model", "")
                        filename = record.get("filename", "")
                        # Try common locations
                        for base in ["test_files/System A test files", "test_files/System B test files",
                                     "test_files/Final Test files"]:
                            alt_path = Path(base) / model / filename
                            if alt_path.exists():
                                file_path = alt_path
                                break
                        else:
                            logger.warning(f"Trim file not found: {record.get('file_path')}")
                            failed_trim += 1
                            continue

                    try:
                        # First try parsing as Final Test (some "Trim" records point to FT files)
                        ft_result = ft_parser.parse_file(file_path)
                        ft_tracks = ft_result.get("tracks", [])

                        if ft_tracks:
                            # Convert FT track format to TrackResult format
                            track_data = db.update_trim_tracks_from_final_test(record["id"], ft_tracks)
                            if track_data:
                                fixed_trim += 1
                                continue

                        # Fall back to Trim processor
                        analysis = trim_processor.process_file(file_path)
                        if analysis and analysis.tracks:
                            if db.update_trim_tracks(record["id"], analysis.tracks):
                                fixed_trim += 1
                            else:
                                failed_trim += 1
                        else:
                            logger.warning(f"No tracks from re-processing {record.get('filename')}")
                            failed_trim += 1
                    except Exception as e:
                        logger.error(f"Error fixing Trim {record.get('filename')}: {e}")
                        failed_trim += 1

                # Report results
                total_fixed = fixed_ft + fixed_trim
                total_failed = failed_ft + failed_trim
                self.after(0, lambda: self._fix_complete(
                    total_fixed, total_failed,
                    details=f"FT:{fixed_ft}/{fixed_ft+failed_ft}, Trim:{fixed_trim}/{fixed_trim+failed_trim}"
                ))

            except Exception as e:
                logger.exception(f"Error in fix missing tracks: {e}")
                self.after(0, lambda: self._fix_complete(0, 0, str(e)))

        get_thread_manager().start_thread(target=fix, name="fix-missing-tracks")

    def _fix_complete(self, fixed: int, failed: int, error: str = None, details: str = None):
        """Called when fix operation completes."""
        self.fix_tracks_btn.configure(state="normal", text="Fix Missing Tracks")

        if error:
            self.count_label.configure(text=f"Error: {error}")
        elif fixed == 0 and failed == 0:
            self.count_label.configure(text="No records need fixing")
        else:
            msg = f"Fixed {fixed}, failed {failed}"
            if details:
                msg += f" ({details})"
            self.count_label.configure(text=msg)

        # Reload data
        self._load_comparisons()

    def _show_error(self, error: str):
        """Show error message."""
        self.count_label.configure(text="Error loading data")
        logger.error(f"Compare page error: {error}")

    def on_show(self):
        """Called when page becomes visible."""
        self._load_comparisons()

    def on_hide(self):
        """Called when page becomes hidden - cleanup to free memory."""
        import matplotlib.pyplot as plt

        # Clear large data lists
        self.comparison_pairs = []
        self.current_comparison = None

        # Clear chart to free matplotlib resources
        if self.chart and hasattr(self.chart, 'figure'):
            try:
                self.chart.clear()
            except Exception as e:
                logger.debug(f"Chart cleanup warning: {e}")

    # ===== Selection Methods =====

    def _on_checkbox_toggle(self, ft_id: int, var: ctk.BooleanVar):
        """Handle checkbox toggle for selection."""
        if var.get():
            self._selected_ids.add(ft_id)
        else:
            self._selected_ids.discard(ft_id)
        self._update_export_button_state()

    def _select_all(self):
        """Select all items on current page."""
        for pair in self.comparison_pairs:
            ft_id = pair.get("final_test_id")
            if ft_id:
                self._selected_ids.add(ft_id)
                if ft_id in self._checkbox_vars:
                    self._checkbox_vars[ft_id].set(True)
        self._update_export_button_state()

    def _select_none(self):
        """Deselect all items."""
        self._selected_ids.clear()
        for var in self._checkbox_vars.values():
            var.set(False)
        self._update_export_button_state()

    def _update_export_button_state(self):
        """Update Export Selected button state based on selection count."""
        count = len(self._selected_ids)
        if count > 0:
            self.export_selected_btn.configure(
                state="normal",
                text=f"Export Selected ({count})"
            )
        else:
            self.export_selected_btn.configure(
                state="disabled",
                text="Export Selected"
            )

    # ===== Export Methods =====

    def _export_current(self):
        """Export the currently displayed comparison chart."""
        if not self.current_comparison:
            return

        ft_id = self.current_comparison.get("final_test_id")
        model = self.current_comparison.get("model", "Unknown")
        serial = self.current_comparison.get("serial", "Unknown")

        default_name = f"{model}-SN{serial}-compare.png"

        file_path = filedialog.asksaveasfilename(
            title="Export Comparison Chart",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.export_current_btn.configure(state="disabled", text="Exporting...")

        def do_export():
            try:
                db = get_database()
                data = db.get_comparison_data(ft_id)
                if data:
                    self._export_comparison_chart(data, Path(file_path))
                    self.after(0, lambda: self._export_complete(1, "chart"))
                else:
                    self.after(0, lambda: self._export_error("No data found"))
            except Exception as e:
                logger.exception(f"Export failed: {e}")
                self.after(0, lambda: self._export_error(str(e)))

        get_thread_manager().start_thread(target=do_export, name="export-comparison")

    def _export_selected(self):
        """Export selected comparison charts as PDF."""
        if not self._selected_ids:
            return

        count = len(self._selected_ids)

        # Build a meaningful default filename
        if count == 1:
            # Single selection - use model/SN format
            ft_id = list(self._selected_ids)[0]
            pair = next((p for p in self.comparison_pairs if p.get("final_test_id") == ft_id), None)
            if pair:
                model = pair.get("model", "Unknown")
                serial = pair.get("serial", "Unknown")
                default_name = f"{model}-SN{serial}-compare.pdf"
            else:
                default_name = f"compare-export.pdf"
        else:
            # Multiple selections - use model name if all same model
            models = set()
            for ft_id in self._selected_ids:
                pair = next((p for p in self.comparison_pairs if p.get("final_test_id") == ft_id), None)
                if pair:
                    models.add(pair.get("model", "Unknown"))

            if len(models) == 1:
                default_name = f"{models.pop()}-compare-{count}units.pdf"
            else:
                default_name = f"compare-export-{count}units.pdf"

        file_path = filedialog.asksaveasfilename(
            title=f"Export {count} Comparison Charts",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[
                ("PDF Document", "*.pdf"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.export_selected_btn.configure(state="disabled", text="Exporting...")

        def do_export():
            try:
                db = get_database()
                comparisons = []

                for ft_id in self._selected_ids:
                    data = db.get_comparison_data(ft_id)
                    if data:
                        comparisons.append(data)

                if comparisons:
                    self._export_multi_comparison_pdf(comparisons, Path(file_path))
                    self.after(0, lambda: self._export_complete(len(comparisons), "PDF"))
                else:
                    self.after(0, lambda: self._export_error("No data found"))
            except Exception as e:
                logger.exception(f"Export failed: {e}")
                self.after(0, lambda: self._export_error(str(e)))

        get_thread_manager().start_thread(target=do_export, name="export-comparisons-pdf")

    def _export_comparison_chart(self, data: Dict, output_path: Path):
        """Export a single comparison chart to file."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np

        plt.style.use('default')

        final_test = data.get("final_test", {})
        trim = data.get("trim")

        # RECALCULATE linearity status from actual data (don't trust stored value)
        corrected_values = self._calculate_linearity_status(data)

        # Create figure
        fig = plt.figure(figsize=(14, 10), dpi=150, facecolor='white')

        model = final_test.get("model", "Unknown")
        serial = final_test.get("serial", "Unknown")
        title = f'Final Test Comparison - {model} / SN:{serial}'
        fig.suptitle(title, fontsize=16, fontweight='bold', color='black', y=0.97)

        # Grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3,
                              left=0.08, right=0.95, top=0.92, bottom=0.08)

        # Main comparison chart (top 2/3)
        ax_main = fig.add_subplot(gs[0:2, :])
        self._plot_comparison_export(ax_main, data, corrected_values)

        # Unit Info (bottom left)
        ax_ft_info = fig.add_subplot(gs[2, 0])
        self._plot_final_test_info(ax_ft_info, data)

        # Comparison Metrics (bottom middle)
        ax_metrics = fig.add_subplot(gs[2, 1])
        self._plot_comparison_metrics(ax_metrics, data, corrected_values)

        # Status Display (bottom right)
        ax_status = fig.add_subplot(gs[2, 2])
        self._plot_comparison_status(ax_status, data, corrected_values)

        # Save
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def _export_multi_comparison_pdf(self, comparisons: List[Dict], output_path: Path):
        """Export multiple comparison charts to a single PDF."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import numpy as np

        plt.style.use('default')

        with PdfPages(output_path) as pdf:
            for data in comparisons:
                final_test = data.get("final_test", {})
                trim = data.get("trim")

                # RECALCULATE linearity status from actual data
                corrected_values = self._calculate_linearity_status(data)

                # Create figure (letter size for PDF)
                fig = plt.figure(figsize=(11, 8.5), facecolor='white')

                model = final_test.get("model", "Unknown")
                serial = final_test.get("serial", "Unknown")
                title = f'Final Test Comparison - {model} / SN:{serial}'
                fig.suptitle(title, fontsize=14, fontweight='bold', color='black', y=0.96)

                gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3,
                                      left=0.08, right=0.95, top=0.90, bottom=0.08)

                ax_main = fig.add_subplot(gs[0:2, :])
                self._plot_comparison_export(ax_main, data, corrected_values)

                ax_ft_info = fig.add_subplot(gs[2, 0])
                self._plot_final_test_info(ax_ft_info, data)

                ax_metrics = fig.add_subplot(gs[2, 1])
                self._plot_comparison_metrics(ax_metrics, data, corrected_values)

                ax_status = fig.add_subplot(gs[2, 2])
                self._plot_comparison_status(ax_status, data, corrected_values)

                pdf.savefig(fig, facecolor='white')
                plt.close(fig)

    def _calculate_linearity_status(self, data: Dict) -> Dict[str, Any]:
        """Calculate linearity status from actual track data using displayed spec limits."""
        final_test = data.get("final_test", {})
        trim = data.get("trim")
        ft_tracks = final_test.get("tracks", [])

        if not ft_tracks:
            return {'fail_points': 0, 'linearity_pass': None, 'trim_fail_points': 0, 'trim_linearity_pass': None}

        ft_track = ft_tracks[0]
        ft_errors = ft_track.get("errors", [])
        upper_limits = ft_track.get("upper_limits", [])
        lower_limits = ft_track.get("lower_limits", [])

        if not ft_errors or not upper_limits or not lower_limits:
            return {'fail_points': 0, 'linearity_pass': None, 'trim_fail_points': 0, 'trim_linearity_pass': None}

        # Count Final Test fail points
        fail_count = 0
        for i, e in enumerate(ft_errors):
            if i < len(upper_limits) and i < len(lower_limits):
                if upper_limits[i] is not None and lower_limits[i] is not None:
                    if e > upper_limits[i] or e < lower_limits[i]:
                        fail_count += 1

        # Calculate Trim linearity against the SAME spec limits (Final Test spec)
        # Need to interpolate spec limits at trim positions since they may differ
        import numpy as np

        trim_fail_count = 0
        trim_linearity_pass = None
        if trim:
            trim_tracks = trim.get("tracks", [])
            if trim_tracks:
                trim_track = trim_tracks[0]
                trim_errors = trim_track.get("errors", [])
                trim_positions = trim_track.get("positions", [])
                offset = trim_track.get("optimal_offset", 0) or 0

                # Apply offset to trim errors (same as chart display)
                if offset and trim_errors:
                    trim_errors = [e + offset for e in trim_errors]

                # Get FT positions for interpolation
                ft_positions = ft_track.get("positions", [])

                # Check trim against Final Test spec limits using interpolation
                if trim_errors and trim_positions and ft_positions and len(ft_positions) == len(upper_limits):
                    # Interpolate spec limits at trim positions
                    try:
                        upper_interp = np.interp(trim_positions, ft_positions,
                                                  [u if u is not None else np.nan for u in upper_limits])
                        lower_interp = np.interp(trim_positions, ft_positions,
                                                  [l if l is not None else np.nan for l in lower_limits])

                        for i, e in enumerate(trim_errors):
                            if not np.isnan(upper_interp[i]) and not np.isnan(lower_interp[i]):
                                if e > upper_interp[i] or e < lower_interp[i]:
                                    trim_fail_count += 1
                    except Exception:
                        # Fallback to index-based comparison
                        for i, e in enumerate(trim_errors):
                            if i < len(upper_limits) and i < len(lower_limits):
                                if upper_limits[i] is not None and lower_limits[i] is not None:
                                    if e > upper_limits[i] or e < lower_limits[i]:
                                        trim_fail_count += 1

                    trim_linearity_pass = trim_fail_count == 0

        return {
            'fail_points': fail_count,
            'linearity_pass': fail_count == 0,
            'trim_fail_points': trim_fail_count,
            'trim_linearity_pass': trim_linearity_pass,
        }

    def _plot_comparison_export(self, ax, data: Dict, corrected_values: Dict = None):
        """Plot comparison chart for export (light mode)."""
        import numpy as np

        COLORS = {
            'final_test': '#27ae60',  # Green
            'trim': '#3498db',  # Blue
            'spec_limit': '#e74c3c',  # Red
            'fail': '#e74c3c',
        }

        final_test = data.get("final_test", {})
        trim = data.get("trim")

        ft_tracks = final_test.get("tracks", [])
        has_ft_data = bool(ft_tracks and ft_tracks[0].get("positions") and ft_tracks[0].get("errors"))

        trim_tracks = trim.get("tracks", []) if trim else []
        has_trim_data = bool(trim_tracks and trim_tracks[0].get("positions") and trim_tracks[0].get("errors"))

        if not has_ft_data and not has_trim_data:
            ax.text(0.5, 0.5, "No track data available",
                   ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return

        # Plot Trim data first (if available)
        if has_trim_data:
            trim_track = trim_tracks[0]
            trim_positions = trim_track.get("positions", [])
            trim_errors = trim_track.get("errors", [])
            offset = trim_track.get("optimal_offset", 0) or trim_track.get("offset", 0)

            if offset:
                trim_errors = [e + offset for e in trim_errors]

            ax.plot(trim_positions, trim_errors, color=COLORS['trim'],
                   linewidth=1.5, label=f"Trim ({trim.get('filename', 'Unknown')[:25]})", alpha=0.9)

        # Plot Final Test data
        if has_ft_data:
            ft_track = ft_tracks[0]
            ft_positions = ft_track.get("positions", [])
            ft_errors = ft_track.get("errors", [])

            ax.plot(ft_positions, ft_errors, color=COLORS['final_test'],
                   linewidth=1.5, label=f"Final Test ({final_test.get('filename', 'Unknown')[:25]})", alpha=0.9)

            # Get spec limits
            upper_limits = ft_track.get("upper_limits", [])
            lower_limits = ft_track.get("lower_limits", [])

            if upper_limits and lower_limits:
                upper_plot = np.array([u if u is not None else np.nan for u in upper_limits])
                lower_plot = np.array([l if l is not None else np.nan for l in lower_limits])
                pos_array = np.array(ft_positions[:len(upper_limits)])

                ax.plot(pos_array, upper_plot, color=COLORS['spec_limit'],
                       linestyle='--', linewidth=1, alpha=0.8, label='Spec Limits')
                ax.plot(pos_array, lower_plot, color=COLORS['spec_limit'],
                       linestyle='--', linewidth=1, alpha=0.8)

                ax.fill_between(pos_array, lower_plot, upper_plot,
                               alpha=0.1, color=COLORS['spec_limit'],
                               where=~np.isnan(upper_plot) & ~np.isnan(lower_plot))

                # Mark fail points (use corrected values if available)
                fail_count = corrected_values.get('fail_points', 0) if corrected_values else 0
                if fail_count > 0:
                    fail_indices = []
                    for i, e in enumerate(ft_errors):
                        if i < len(upper_limits) and i < len(lower_limits):
                            if upper_limits[i] is not None and lower_limits[i] is not None:
                                if e > upper_limits[i] or e < lower_limits[i]:
                                    fail_indices.append(i)

                    if fail_indices:
                        fail_pos = [ft_positions[i] for i in fail_indices]
                        fail_err = [ft_errors[i] for i in fail_indices]
                        ax.scatter(fail_pos, fail_err, color=COLORS['fail'],
                                  s=100, marker='x', linewidth=3,
                                  label=f'Fail Points ({len(fail_indices)})', zorder=5)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)

        # Use corrected linearity status for title
        if corrected_values and corrected_values.get('linearity_pass') is not None:
            lin_pass = corrected_values['linearity_pass']
            if lin_pass:
                status = "PASS"
            else:
                status = f"FAIL ({corrected_values['fail_points']} pts)"
        else:
            status = "Linked Comparison" if has_trim_data else "Final Test Only"

        ax.set_title(f'Track Comparison - {status}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_final_test_info(self, ax, data: Dict):
        """Plot Final Test information panel with source files."""
        ax.axis('off')
        ax.set_facecolor('white')

        final_test = data.get("final_test", {})
        trim = data.get("trim")

        test_date = final_test.get("test_date")
        if test_date and isinstance(test_date, datetime):
            date_str = test_date.strftime('%m/%d/%Y')
        else:
            date_str = str(test_date)[:10] if test_date else "N/A"

        # Truncate long filenames
        ft_filename = final_test.get('filename', 'Unknown')
        if len(ft_filename) > 35:
            ft_filename = ft_filename[:32] + "..."
        trim_filename = trim.get('filename', 'N/A') if trim else 'N/A'
        if len(trim_filename) > 35:
            trim_filename = trim_filename[:32] + "..."

        info_lines = [
            f"Model: {final_test.get('model', 'Unknown')}",
            f"Serial: {final_test.get('serial', 'Unknown')}",
            f"Test Date: {date_str}",
            "",
            "Source Files:",
            f"  FT: {ft_filename}",
            f"  Trim: {trim_filename}",
        ]

        y_pos = 0.95
        ax.text(0.05, 0.98, "Unit Info", fontsize=11, fontweight='bold',
               transform=ax.transAxes, va='top', color='black')

        for line in info_lines:
            y_pos -= 0.11
            ax.text(0.05, y_pos, line, fontsize=9, transform=ax.transAxes, va='top', color='black')

    def _plot_comparison_metrics(self, ax, data: Dict, corrected_values: Dict = None):
        """Plot comparison metrics panel."""
        ax.axis('off')
        ax.set_facecolor('white')

        final_test = data.get("final_test", {})
        trim = data.get("trim")
        days_since = data.get("days_since_trim")
        confidence = data.get("match_confidence")

        metrics = [
            f"Days Since Trim: {days_since}" if days_since is not None else "Days Since Trim: N/A",
            f"Match Confidence: {confidence*100:.0f}%" if confidence else "Match Confidence: N/A",
            "",
        ]

        # Final Test results - use corrected values if available
        ft_tracks = final_test.get("tracks", [])
        if ft_tracks:
            if corrected_values and corrected_values.get('linearity_pass') is not None:
                linearity_pass = corrected_values['linearity_pass']
                fail_pts = corrected_values.get('fail_points', 0)
                metrics.extend([
                    "Final Test Results:",
                    f"  Linearity: {'PASS' if linearity_pass else f'FAIL ({fail_pts} pts)'}",
                ])
            else:
                linearity_pass = final_test.get("linearity_pass")
                metrics.extend([
                    "Final Test Results:",
                    f"  Linearity: {'PASS' if linearity_pass else 'FAIL' if linearity_pass is False else 'N/A'}",
                ])

        # Trim info - use recalculated value against Final Test spec limits
        if trim:
            if corrected_values and corrected_values.get('trim_linearity_pass') is not None:
                trim_lin_pass = corrected_values['trim_linearity_pass']
                trim_fail_pts = corrected_values.get('trim_fail_points', 0)
                metrics.extend([
                    "",
                    "Trim Results (vs FT spec):",
                    f"  Linearity: {'PASS' if trim_lin_pass else f'FAIL ({trim_fail_pts} pts)'}",
                ])
            else:
                trim_tracks = trim.get("tracks", [])
                if trim_tracks:
                    trim_track = trim_tracks[0]
                    lin_pass = trim_track.get('linearity_pass')
                    metrics.extend([
                        "",
                        "Trim Results:",
                        f"  Linearity: {'PASS' if lin_pass else 'FAIL' if lin_pass is False else 'N/A'}",
                    ])

        y_pos = 0.95
        ax.text(0.05, 0.98, "Comparison Metrics", fontsize=11, fontweight='bold',
               transform=ax.transAxes, va='top', color='black')

        for metric in metrics:
            y_pos -= 0.09
            color = 'black'
            if 'FAIL' in metric:
                color = '#e74c3c'
            elif 'PASS' in metric:
                color = '#27ae60'
            ax.text(0.05, y_pos, metric, fontsize=10, transform=ax.transAxes, va='top', color=color)

    def _plot_comparison_status(self, ax, data: Dict, corrected_values: Dict = None):
        """Plot status display panel."""
        from matplotlib.patches import Rectangle

        ax.axis('off')
        ax.set_facecolor('white')

        final_test = data.get("final_test", {})

        # Use corrected linearity_pass if available
        if corrected_values and corrected_values.get('linearity_pass') is not None:
            linearity_pass = corrected_values['linearity_pass']
        else:
            linearity_pass = final_test.get("linearity_pass")

        # Determine color from actual linearity status
        if linearity_pass:
            status = "PASS"
            color = '#27ae60'
        elif linearity_pass is False:
            status = "FAIL"
            color = '#e74c3c'
        else:
            status = "UNKNOWN"
            color = 'gray'

        # Draw status box
        rect = Rectangle((0.1, 0.55), 0.8, 0.30,
                         linewidth=3, edgecolor=color,
                         facecolor='white', alpha=0.9)
        ax.add_patch(rect)

        ax.text(0.5, 0.70, f'STATUS: {status}', ha='center', va='center',
               fontsize=16, color=color, fontweight='bold', transform=ax.transAxes)

        # Show fail details if applicable
        if corrected_values and not linearity_pass and corrected_values.get('fail_points', 0) > 0:
            ax.text(0.5, 0.48, f"({corrected_values['fail_points']} fail points)",
                   ha='center', va='center', fontsize=10, color=color, transform=ax.transAxes)

        # Linked status
        trim = data.get("trim")
        if trim:
            ax.text(0.5, 0.35, "Linked to Trim", ha='center', va='center',
                   fontsize=11, color='#27ae60', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.35, "No Linked Trim", ha='center', va='center',
                   fontsize=11, color='gray', transform=ax.transAxes)

        # Export timestamp
        ax.text(0.5, 0.15, f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
               ha='center', va='center', fontsize=9, color='gray',
               style='italic', transform=ax.transAxes)

    def _export_complete(self, count: int, format_name: str):
        """Called when export completes successfully."""
        self.export_current_btn.configure(state="normal", text="Export Chart")
        self._update_export_button_state()
        messagebox.showinfo(
            "Export Complete",
            f"Successfully exported {count} comparison{'s' if count != 1 else ''} as {format_name}."
        )

    def _export_error(self, error: str):
        """Called when export fails."""
        self.export_current_btn.configure(state="normal", text="Export Chart")
        self._update_export_button_state()
        messagebox.showerror("Export Error", f"Failed to export:\n{error}")
