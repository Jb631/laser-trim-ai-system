"""
Compare Page - Final Test vs Trim comparison with overlay charts.

This page allows viewing Final Test results and comparing them
with linked Trim results using overlay charts.
"""

import threading
import customtkinter as ctk
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from laser_trim_analyzer.database import get_database
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
            text="Final Test Comparison",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            header_frame,
            text="Compare Final Test results with linked Trim data",
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

        # Days filter
        ctk.CTkLabel(filter_frame, text="Period:").pack(side="left", padx=(20, 5), pady=15)
        self.days_filter = ctk.CTkOptionMenu(
            filter_frame,
            values=["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
            command=self._on_filter_change,
            width=120
        )
        self.days_filter.set("Last 30 Days")
        self.days_filter.pack(side="left", padx=5, pady=15)

        # Linked only filter
        self.linked_only_var = ctk.BooleanVar(value=False)
        linked_check = ctk.CTkCheckBox(
            filter_frame,
            text="Linked Only",
            variable=self.linked_only_var,
            command=self._on_filter_change
        )
        linked_check.pack(side="left", padx=20, pady=15)

        # Refresh button
        refresh_btn = ctk.CTkButton(
            filter_frame,
            text="Refresh",
            command=self._load_comparisons,
            width=100
        )
        refresh_btn.pack(side="right", padx=15, pady=15)

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

        list_label = ctk.CTkLabel(
            list_frame,
            text="Final Test Results",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        list_label.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

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

                days_text = self.days_filter.get()
                days_map = {
                    "Last 7 Days": 7,
                    "Last 30 Days": 30,
                    "Last 90 Days": 90,
                    "All Time": 3650,
                }
                days = days_map.get(days_text, 30)

                linked_only = self.linked_only_var.get()

                # Get comparison pairs
                pairs = db.get_comparison_pairs(
                    model=model,
                    days_back=days,
                    linked_only=linked_only
                )

                # Get models for filter dropdown
                models = db.get_final_test_models_list()

                self.after(0, lambda: self._display_comparisons(pairs, models))

            except Exception as e:
                logger.exception(f"Error loading comparisons: {e}")
                self.after(0, lambda: self._show_error(str(e)))

        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()

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
        item_frame = ctk.CTkFrame(self.list_frame)
        item_frame.pack(fill="x", padx=5, pady=2)

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
        status_label.pack(side="left", padx=(10, 5), pady=10)

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

        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()

    def _display_comparison_chart(self, data: Optional[Dict]):
        """Display the comparison overlay chart."""
        if not data:
            return

        self._ensure_chart_initialized()
        if not self.chart:
            return

        final_test = data.get("final_test", {})
        trim = data.get("trim")

        # Get track data
        ft_tracks = final_test.get("tracks", [])
        if not ft_tracks:
            self.chart.clear()
            return

        ft_track = ft_tracks[0]  # Use first track
        # Support both "positions" and "electrical_angles" (new format)
        ft_positions = ft_track.get("positions") or ft_track.get("electrical_angles", [])
        ft_errors = ft_track.get("errors", [])

        # Prepare chart data
        chart_data = {
            "final_test": {
                "positions": ft_positions,
                "errors": ft_errors,
                "label": f"Final Test ({final_test.get('filename', 'Unknown')[:20]})",
            }
        }

        if trim:
            trim_tracks = trim.get("tracks", [])
            if trim_tracks:
                trim_track = trim_tracks[0]
                chart_data["trim"] = {
                    "positions": trim_track.get("positions", []),
                    "errors": trim_track.get("errors", []),
                    "label": f"Trim ({trim.get('filename', 'Unknown')[:20]})",
                }

        # Plot comparison
        self._plot_comparison(chart_data)

    def _plot_comparison(self, chart_data: Dict):
        """Plot the comparison overlay chart."""
        if not self.chart:
            return

        self.chart.clear()

        ax = self.chart.figure.add_subplot(111)

        # Plot Final Test data
        ft_data = chart_data.get("final_test", {})
        if ft_data.get("positions") and ft_data.get("errors"):
            ax.plot(
                ft_data["positions"],
                ft_data["errors"],
                'b-',
                linewidth=1.5,
                label=ft_data.get("label", "Final Test"),
                alpha=0.8
            )

        # Plot Trim data if available
        trim_data = chart_data.get("trim", {})
        if trim_data.get("positions") and trim_data.get("errors"):
            ax.plot(
                trim_data["positions"],
                trim_data["errors"],
                'g-',
                linewidth=1.5,
                label=trim_data.get("label", "Trim"),
                alpha=0.8
            )

        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Labels and legend
        ax.set_xlabel("Position")
        ax.set_ylabel("Linearity Error")
        ax.set_title("Trim vs Final Test Comparison")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        self.chart.draw()

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

    def _show_error(self, error: str):
        """Show error message."""
        self.count_label.configure(text="Error loading data")
        logger.error(f"Compare page error: {error}")

    def on_show(self):
        """Called when page becomes visible."""
        self._load_comparisons()
