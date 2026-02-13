"""
Analyze Page - Browse and view analysis results from database.

This page is for reviewing already-processed files only.
All file processing is done through the Process Files page.
"""

import customtkinter as ctk
import logging
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from laser_trim_analyzer.core.models import AnalysisResult, AnalysisStatus, TrackData
from laser_trim_analyzer.core.processor import Processor
from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.config import get_config
from laser_trim_analyzer.utils.threads import get_thread_manager
from laser_trim_analyzer.export import (
    export_single_result, export_batch_results,
    generate_export_filename, generate_batch_export_filename, ExcelExportError
)
from laser_trim_analyzer.gui.widgets.scrollable_combobox import ScrollableComboBox
from datetime import datetime, timedelta

# Lazy import for ChartWidget - defer matplotlib loading until first use
if TYPE_CHECKING:
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle

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

        # Pagination settings
        self._page_size = 20
        self._current_page = 0
        self._total_pages = 1

        # Lazy chart initialization
        self._chart_initialized = False
        self.chart = None

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

        # Model filter - use ScrollableComboBox for dropdown with many models
        ctk.CTkLabel(filter_frame, text="Model:").pack(side="left", padx=(15, 5), pady=15)
        self.model_filter = ScrollableComboBox(
            filter_frame,
            values=["All Models"],
            command=self._on_filter_change,
            width=150,
            dropdown_height=300,  # Scrollable dropdown
        )
        self.model_filter.set("All Models")
        self.model_filter.pack(side="left", padx=5, pady=15)

        # Active Only filter - hide inactive models from dropdown
        self.active_only_var = ctk.BooleanVar(value=True)
        self.active_only_check = ctk.CTkCheckBox(
            filter_frame,
            text="Active Only",
            variable=self.active_only_var,
            command=self._on_filter_change,
            width=100,
            checkbox_width=18,
            checkbox_height=18,
        )
        self.active_only_check.pack(side="left", padx=(5, 5), pady=15)

        # Date range filter (filters by trim date, not processing date)
        ctk.CTkLabel(filter_frame, text="From:").pack(side="left", padx=(20, 5), pady=15)
        self.date_from = ctk.CTkEntry(
            filter_frame,
            placeholder_text="MM/DD/YYYY",
            width=90
        )
        self.date_from.pack(side="left", padx=2, pady=15)
        self.date_from.bind("<Return>", lambda e: self._load_recent_analyses())
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
        self.date_to.bind("<Return>", lambda e: self._load_recent_analyses())
        # Set default to today
        self.date_to.insert(0, datetime.now().strftime("%m/%d/%Y"))

        # Status filter
        ctk.CTkLabel(filter_frame, text="Status:").pack(side="left", padx=(20, 5), pady=15)
        self.status_filter = ctk.CTkOptionMenu(
            filter_frame,
            values=["All", "Pass", "Warning", "Fail"],
            command=self._on_filter_change,
            width=100
        )
        self.status_filter.pack(side="left", padx=5, pady=15)

        # Serial filter (partial match)
        ctk.CTkLabel(filter_frame, text="Serial:").pack(side="left", padx=(20, 5), pady=15)
        self.serial_filter = ctk.CTkEntry(
            filter_frame,
            placeholder_text="Search...",
            width=100
        )
        self.serial_filter.pack(side="left", padx=5, pady=15)
        self.serial_filter.bind("<Return>", lambda e: self._load_recent_analyses())

        # Refresh button
        refresh_btn = ctk.CTkButton(
            filter_frame,
            text="⟳ Refresh",
            command=self._load_recent_analyses,
            width=80
        )
        refresh_btn.pack(side="right", padx=15, pady=15)

        # Results count label
        self.count_label = ctk.CTkLabel(
            filter_frame,
            text="",
            text_color="gray"
        )
        self.count_label.pack(side="right", padx=10, pady=15)

        # Main content area - use regular frame for proper resizing
        content = ctk.CTkFrame(self)
        content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        # Configure inner frame for proper layout
        content.grid_columnconfigure(0, weight=1, minsize=250)  # List panel
        content.grid_columnconfigure(1, weight=3)  # Details panel - more space
        content.grid_rowconfigure(0, weight=1)  # Allow full resizing

        # Left panel - analysis list
        list_frame = ctk.CTkFrame(content)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        list_frame.grid_rowconfigure(1, weight=1)
        list_frame.grid_rowconfigure(2, weight=0)  # Pagination row
        list_frame.grid_columnconfigure(0, weight=1)

        list_label = ctk.CTkLabel(
            list_frame,
            text="Recent Analyses",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        list_label.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        # Scrollable list of analyses
        self.analysis_list_frame = ctk.CTkScrollableFrame(list_frame)
        self.analysis_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 5))
        self.analysis_list_frame.grid_columnconfigure(0, weight=1)

        # Pagination controls
        pagination_frame = ctk.CTkFrame(list_frame, fg_color="transparent")
        pagination_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        self._prev_btn = ctk.CTkButton(
            pagination_frame,
            text="<",
            width=30,
            command=self._prev_page,
            state="disabled"
        )
        self._prev_btn.pack(side="left", padx=2)

        self._page_label = ctk.CTkLabel(
            pagination_frame,
            text="Page 1/1",
            font=ctk.CTkFont(size=11)
        )
        self._page_label.pack(side="left", padx=10, expand=True)

        self._next_btn = ctk.CTkButton(
            pagination_frame,
            text=">",
            width=30,
            command=self._next_page,
            state="disabled"
        )
        self._next_btn.pack(side="right", padx=2)

        # Right panel - details view
        details_frame = ctk.CTkFrame(content)
        details_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        details_frame.grid_rowconfigure(0, weight=0)  # Status banner - fixed
        details_frame.grid_rowconfigure(1, weight=0)  # Action buttons - fixed
        details_frame.grid_rowconfigure(2, weight=1)  # Tabview with chart - fully expandable
        details_frame.grid_rowconfigure(3, weight=0)  # DB info footer - fixed
        details_frame.grid_columnconfigure(0, weight=1)

        # Status banner - dynamic height based on content, but capped
        self.status_banner = ctk.CTkFrame(details_frame)
        self.status_banner.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))
        # Prevent banner from pushing other rows around during resize
        self.status_banner.grid_propagate(True)

        self.status_text = ctk.CTkLabel(
            self.status_banner,
            text="Select an analysis",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="gray",
            wraplength=500  # Wrap long text to prevent horizontal overflow
        )
        self.status_text.pack(pady=12, padx=15)

        # Action buttons row
        actions_frame = ctk.CTkFrame(details_frame, fg_color="transparent")
        actions_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=5)

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

        # Delete button (remove from database)
        self.delete_btn = ctk.CTkButton(
            actions_frame,
            text="🗑 Delete",
            command=self._delete_current,
            state="disabled",
            width=90,
            fg_color="#8B0000",  # Dark red
            hover_color="#B22222"  # Lighter red on hover
        )
        self.delete_btn.pack(side="left", padx=5)

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
        self._chart_tab = self.details_tabview.tab("Chart")
        self._chart_tab.grid_columnconfigure(0, weight=1)
        self._chart_tab.grid_rowconfigure(0, weight=1)

        # Placeholder until chart is needed - ChartWidget created lazily
        self._chart_placeholder = ctk.CTkLabel(
            self._chart_tab,
            text="Select an analysis to view chart",
            text_color="gray"
        )
        self._chart_placeholder.pack(fill="both", expand=True)

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

    def _ensure_chart_initialized(self):
        """Lazily initialize ChartWidget on first use - defers matplotlib loading."""
        if self._chart_initialized:
            return

        # Import matplotlib-dependent ChartWidget only when needed
        from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle

        # Remove placeholder
        if self._chart_placeholder:
            self._chart_placeholder.destroy()
            self._chart_placeholder = None

        # Create actual chart widget
        self.chart = ChartWidget(
            self._chart_tab,
            style=ChartStyle(figure_size=(6, 4), dpi=100)
        )
        self.chart.pack(fill="both", expand=True)
        self.chart.show_placeholder("Select an analysis to view chart")

        self._chart_initialized = True
        logger.debug("AnalyzePage ChartWidget initialized (matplotlib loaded)")

    # =========================================================================
    # Mousewheel Support
    # =========================================================================

    def _bind_mousewheel_scroll(self, combobox):
        """
        Bind mousewheel events to CTkComboBox dropdown.

        CTkComboBox creates a dropdown window dynamically. We need to:
        1. Bind to the combobox itself for scrolling when dropdown is closed
        2. Hook into dropdown creation to enable mousewheel scrolling in the list
        """
        def on_mousewheel_closed(event):
            """Scroll through values when dropdown is closed."""
            values = combobox.cget("values")
            if not values:
                return "break"

            current = combobox.get()
            try:
                current_idx = list(values).index(current)
            except ValueError:
                current_idx = 0

            # Scroll direction (Windows: event.delta, Linux: event.num)
            if hasattr(event, 'delta'):
                direction = -1 if event.delta > 0 else 1
            else:
                direction = -1 if event.num == 4 else 1

            new_idx = current_idx + direction
            if 0 <= new_idx < len(values):
                combobox.set(values[new_idx])
                command = combobox.cget("command")
                if command:
                    command(values[new_idx])
            return "break"

        # Bind for scrolling when dropdown is closed (hover over combobox)
        combobox.bind("<MouseWheel>", on_mousewheel_closed)
        combobox.bind("<Button-4>", on_mousewheel_closed)
        combobox.bind("<Button-5>", on_mousewheel_closed)

        # Patch the dropdown open method to add mousewheel scrolling
        if hasattr(combobox, '_open_dropdown_menu'):
            original_open = combobox._open_dropdown_menu

            def patched_open(*args, **kwargs):
                result = original_open(*args, **kwargs)
                # After dropdown opens, bind mousewheel to it
                combobox.after(10, lambda: self._enable_dropdown_scroll(combobox))
                return result

            combobox._open_dropdown_menu = patched_open

    def _enable_dropdown_scroll(self, combobox):
        """Enable mousewheel scrolling on the open dropdown."""
        try:
            # CTkComboBox stores dropdown in _dropdown_menu
            if hasattr(combobox, '_dropdown_menu') and combobox._dropdown_menu:
                dropdown = combobox._dropdown_menu

                # Find the scrollable frame's canvas - CTkScrollableDropdownFrame structure
                canvas = None

                # Try different attribute names used by CTkScrollableDropdownFrame
                if hasattr(dropdown, '_scrollable_frame'):
                    sf = dropdown._scrollable_frame
                    if hasattr(sf, '_parent_canvas'):
                        canvas = sf._parent_canvas
                    elif hasattr(sf, '_canvas'):
                        canvas = sf._canvas
                elif hasattr(dropdown, '_canvas'):
                    canvas = dropdown._canvas

                # Search children if not found
                if not canvas:
                    def find_canvas(widget):
                        for child in widget.winfo_children():
                            child_type = str(type(child)).lower()
                            if 'canvas' in child_type:
                                return child
                            found = find_canvas(child)
                            if found:
                                return found
                        return None
                    canvas = find_canvas(dropdown)

                if canvas:
                    def scroll_dropdown(event):
                        """Handle mousewheel in dropdown."""
                        try:
                            # Scroll direction
                            delta = -1 * (event.delta // 120) if event.delta else (-1 if event.num == 4 else 1)
                            canvas.yview_scroll(delta, "units")
                        except Exception:
                            pass
                        return "break"

                    # Bind globally on the dropdown toplevel window
                    dropdown.bind_all("<MouseWheel>", scroll_dropdown)
                    dropdown.bind_all("<Button-4>", scroll_dropdown)
                    dropdown.bind_all("<Button-5>", scroll_dropdown)

                    # Clean up when dropdown closes
                    def on_destroy(event):
                        try:
                            dropdown.unbind_all("<MouseWheel>")
                            dropdown.unbind_all("<Button-4>")
                            dropdown.unbind_all("<Button-5>")
                        except Exception:
                            pass

                    dropdown.bind("<Destroy>", on_destroy)
        except Exception as e:
            logger.debug(f"Could not enable dropdown scroll: {e}")

    # =========================================================================
    # Data Loading
    # =========================================================================

    def _load_recent_analyses(self):
        """Load recent analyses from database."""
        self.count_label.configure(text="Loading...")

        get_thread_manager().start_thread(target=self._fetch_analyses, name="fetch-analyses")

    def _fetch_analyses(self):
        """Fetch analyses in background thread."""
        try:
            db = get_database()

            # Get filter values
            model = self.model_filter.get()
            # Strip (inactive) suffix if present
            if " (inactive)" in model:
                model = model.replace(" (inactive)", "")
            status = self.status_filter.get()
            serial = self.serial_filter.get().strip()

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
                    # Include the entire end date
                    date_to = date_to.replace(hour=23, minute=59, second=59)
                except ValueError:
                    try:
                        date_to = datetime.strptime(to_str, "%m/%d/%y")
                        date_to = date_to.replace(hour=23, minute=59, second=59)
                    except ValueError:
                        logger.warning(f"Invalid To date format: {to_str}")

            # Use search_for_export which supports date range filtering
            try:
                analyses = db.search_for_export(
                    model=None if model == "All Models" else model,
                    serial=serial if serial else None,
                    date_from=date_from,
                    date_to=date_to,
                    limit=500
                )
            except Exception as e:
                logger.error(f"Failed to get historical data: {e}")
                analyses = []

            # Filter by status if needed
            if status == "Pass":
                analyses = [a for a in analyses if a.overall_status == AnalysisStatus.PASS]
            elif status == "Warning":
                analyses = [a for a in analyses if a.overall_status == AnalysisStatus.WARNING]
            elif status == "Fail":
                analyses = [a for a in analyses if a.overall_status in (AnalysisStatus.FAIL, AnalysisStatus.ERROR)]

            # Limit results after filtering
            analyses = analyses[:100]

            # Get models list for dropdown - separate try to ensure dropdown works even if data fails
            try:
                config = get_config()
                prioritized = db.get_models_list_prioritized(
                    mps_models=config.active_models.mps_models,
                    recent_days=config.active_models.recent_days
                )
                # Build model list with inactive suffix
                active_only = self.active_only_var.get()
                models = []
                for m in prioritized:
                    if m['status'] == 'inactive':
                        if not active_only:
                            models.append(f"{m['model']} (inactive)")
                    else:
                        models.append(m['model'])
            except Exception as e:
                logger.error(f"Failed to get models list: {e}")
                models = []

            # Get database info
            try:
                db_path = db.get_database_path()
                record_count = db.get_record_count()
            except Exception as e:
                logger.error(f"Failed to get database info: {e}")
                db_path = db.database_path
                record_count = {"analyses": 0, "tracks": 0}

            # Update UI on main thread
            self.after(0, lambda: self._display_analyses(analyses, models, db_path, record_count))

        except Exception as e:
            logger.error(f"Failed to fetch analyses: {e}")
            import traceback
            traceback.print_exc()
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
        else:
            self.model_filter.set("All Models")

        # Update count
        self.count_label.configure(text=f"{len(analyses)} results")

        # Update database info
        self.db_info_label.configure(
            text=f"Database: {db_path.name} | {record_count.get('analyses', 0)} analyses, {record_count.get('tracks', 0)} tracks"
        )

        # Calculate pagination
        self._total_pages = max(1, (len(analyses) + self._page_size - 1) // self._page_size)
        self._current_page = min(self._current_page, self._total_pages - 1)

        # Update pagination controls
        self._update_pagination_controls()

        # Display current page
        self._display_current_page()

    def _display_current_page(self):
        """Display only the items for the current page."""
        # Clear existing list
        for widget in self.analysis_list_frame.winfo_children():
            widget.destroy()

        if not self.recent_analyses:
            no_data_label = ctk.CTkLabel(
                self.analysis_list_frame,
                text="No analyses found.\n\nUse 'Process Files' to\nprocess some files first!",
                text_color="gray",
                justify="center"
            )
            no_data_label.pack(pady=40)
            return

        # Get items for current page
        start_idx = self._current_page * self._page_size
        end_idx = start_idx + self._page_size
        page_items = self.recent_analyses[start_idx:end_idx]

        # Create list items for current page only
        for i, analysis in enumerate(page_items):
            self._create_analysis_list_item(analysis, start_idx + i)

    def _update_pagination_controls(self):
        """Update pagination button states and label."""
        self._page_label.configure(text=f"Page {self._current_page + 1}/{self._total_pages}")

        # Enable/disable prev button
        if self._current_page > 0:
            self._prev_btn.configure(state="normal")
        else:
            self._prev_btn.configure(state="disabled")

        # Enable/disable next button
        if self._current_page < self._total_pages - 1:
            self._next_btn.configure(state="normal")
        else:
            self._next_btn.configure(state="disabled")

    def _prev_page(self):
        """Go to previous page."""
        if self._current_page > 0:
            self._current_page -= 1
            self._update_pagination_controls()
            self._display_current_page()

    def _next_page(self):
        """Go to next page."""
        if self._current_page < self._total_pages - 1:
            self._current_page += 1
            self._update_pagination_controls()
            self._display_current_page()

    def _create_analysis_list_item(self, analysis: AnalysisResult, index: int):
        """Create a clickable item for an analysis."""
        # Check if any track is flagged as anomaly
        has_anomaly = any(t.is_anomaly for t in analysis.tracks if hasattr(t, 'is_anomaly'))

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

        # Anomaly indicator (if any track is anomalous)
        if has_anomaly:
            anomaly_indicator = ctk.CTkLabel(
                item_frame,
                text="!",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color="#9b59b6",  # Purple for anomaly
                width=15
            )
            anomaly_indicator.pack(side="left", padx=(0, 2), pady=8)

        # Info section
        info_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        # Line 1: Model and Serial (most important info)
        model_serial = f"{analysis.metadata.model} / {analysis.metadata.serial}"
        model_serial_label = ctk.CTkLabel(
            info_frame,
            text=model_serial,
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w"
        )
        model_serial_label.pack(anchor="w")

        # Line 2: Date (compact format)
        if analysis.metadata.file_date:
            date_text = analysis.metadata.file_date.strftime('%m/%d/%y')
        else:
            date_text = "No date"

        date_label = ctk.CTkLabel(
            info_frame,
            text=date_text,
            font=ctk.CTkFont(size=10),
            text_color="gray",
            anchor="w"
        )
        date_label.pack(anchor="w")

        # Bind click event
        def on_click(event, a=analysis):
            self._show_analysis_details(a)

        item_frame.bind("<Button-1>", on_click)
        status_indicator.bind("<Button-1>", on_click)
        info_frame.bind("<Button-1>", on_click)
        model_serial_label.bind("<Button-1>", on_click)
        date_label.bind("<Button-1>", on_click)
        # Bind anomaly indicator if it exists
        if has_anomaly:
            anomaly_indicator.bind("<Button-1>", on_click)

    # =========================================================================
    # Details Display
    # =========================================================================

    def _show_analysis_details(self, analysis: AnalysisResult):
        """Show details for selected analysis."""
        self.current_result = analysis

        # Update status banner - reset font to normal size
        normal_font = ctk.CTkFont(size=24, weight="bold")
        if analysis.overall_status == AnalysisStatus.PASS:
            self.status_text.configure(text="PASS", text_color="#27ae60", font=normal_font)
            self.status_banner.configure(fg_color="#1e4d2b")
        elif analysis.overall_status == AnalysisStatus.FAIL:
            self.status_text.configure(text="FAIL", text_color="#e74c3c", font=normal_font)
            self.status_banner.configure(fg_color="#4d1e1e")
        else:
            self.status_text.configure(text=analysis.overall_status.value.upper(), text_color="#f39c12", font=normal_font)
            self.status_banner.configure(fg_color="#4d3d1e")

        # Enable export buttons
        self.export_btn.configure(state="normal")
        self.export_chart_btn.configure(state="normal")
        self.export_model_btn.configure(state="normal")
        self.reanalyze_btn.configure(state="normal")
        self.delete_btn.configure(state="normal")

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
        # Ensure chart is initialized before use
        self._ensure_chart_initialized()

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
        # Skip positions where limit is None (no spec = unlimited at that position)
        fail_indices = []
        actual_fail_count = 0
        if upper_limits and lower_limits:
            shifted_errors = [e + track.optimal_offset for e in track.error_data]
            for i, e in enumerate(shifted_errors):
                if i < len(upper_limits) and i < len(lower_limits):
                    # Only check positions that have spec limits (not None)
                    if upper_limits[i] is not None and lower_limits[i] is not None:
                        if e > upper_limits[i] or e < lower_limits[i]:
                            fail_indices.append(i)
                            actual_fail_count += 1

        # Determine actual status based on recalculated fail points
        # Use WARNING when one passes and one fails
        lin_pass = actual_fail_count == 0
        sigma_pass = track.sigma_pass

        if lin_pass and sigma_pass:
            status_str = "PASS"
        elif lin_pass and not sigma_pass:
            status_str = "WARNING (Sigma Fail)"
        elif not lin_pass and sigma_pass:
            status_str = "WARNING (Lin Fail)"
        else:
            status_str = "FAIL"

        # Build title with model and SN for identification
        model = self.current_result.metadata.model if self.current_result else "?"
        serial = self.current_result.metadata.serial if self.current_result else "?"

        # Get trim date for display
        trim_date = None
        if self.current_result and self.current_result.metadata.file_date:
            trim_date = self.current_result.metadata.file_date.strftime('%m/%d/%Y')

        # Add anomaly indicator if detected
        if track.is_anomaly:
            title = f"{model} - Track {track.track_id} - {status_str} [ANOMALY]"
        else:
            title = f"{model} - Track {track.track_id} - {status_str}"

        self.chart.plot_error_vs_position(
            positions=track.position_data,
            trimmed_errors=track.error_data,
            upper_limits=upper_limits,
            lower_limits=lower_limits,
            untrimmed_positions=track.untrimmed_positions,
            untrimmed_errors=track.untrimmed_errors,
            offset=track.optimal_offset,
            title=title,
            fail_points=fail_indices,
            serial_number=serial,
            trim_improvement_percent=track.trim_improvement_percent,
            trim_date=trim_date
        )

    def _display_metrics(self, analysis: AnalysisResult):
        """Display analysis metrics."""
        try:
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

                # Calculate actual fail points (skip None = no limit at that position)
                actual_fail_count = 0
                if actual_upper and actual_lower and track.error_data:
                    shifted_errors = [e + track.optimal_offset for e in track.error_data]
                    for i, e in enumerate(shifted_errors):
                        if i < len(actual_upper) and i < len(actual_lower):
                            # Skip positions with no spec limit (None = unlimited)
                            if actual_upper[i] is not None and actual_lower[i] is not None:
                                if e > actual_upper[i] or e < actual_lower[i]:
                                    actual_fail_count += 1

                # Show spec limits (filter out None values for display)
                lines.append(f"    Spec:       ±{track.linearity_spec:.6f}")
                if actual_upper and actual_lower:
                    # Filter out None values for max/min calculation
                    valid_upper = [u for u in actual_upper if u is not None]
                    valid_lower = [l for l in actual_lower if l is not None]
                    if valid_upper and valid_lower:
                        # Count positions with no spec
                        none_count = sum(1 for u in actual_upper if u is None)
                        if none_count > 0:
                            lines.append(f"    (No spec at {none_count} positions)")

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

                # Anomaly Detection (if flagged)
                if track.is_anomaly:
                    lines.append("  ⚠️  ANOMALY DETECTED:")
                    lines.append(f"    Reason: {track.anomaly_reason or 'Unknown'}")
                    lines.append("    Note: This unit likely has a trim failure")
                    lines.append("          and may skew statistical analysis.")
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
                    if track.resistance_change is not None:
                        pct = track.resistance_change_percent
                        pct_str = f" ({pct:+.1f}%)" if pct is not None else ""
                        lines.append(f"    R Change:    {track.resistance_change:+.2f}{pct_str}")
                    lines.append("")

                # Trim Effectiveness (if calculated)
                if track.trim_improvement_percent is not None or track.untrimmed_rms_error is not None:
                    lines.append("  TRIM EFFECTIVENESS:")
                    if track.untrimmed_rms_error is not None:
                        lines.append(f"    Untrimmed RMS Error: {track.untrimmed_rms_error:.4f}")
                    if track.trimmed_rms_error is not None:
                        lines.append(f"    Trimmed RMS Error:   {track.trimmed_rms_error:.4f}")
                    if track.trim_improvement_percent is not None:
                        lines.append(f"    Improvement:         {track.trim_improvement_percent:.1f}%")
                    if track.max_error_reduction_percent is not None:
                        lines.append(f"    Max Error Reduction: {track.max_error_reduction_percent:.1f}%")
                    lines.append("")

            self._update_metrics("\n".join(lines))
        except Exception as e:
            logger.error(f"Error displaying metrics: {e}")
            self._update_metrics(f"Error loading metrics: {e}")

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

        lines.append(f"  Trim Date:   {analysis.metadata.file_date.strftime('%Y-%m-%d %H:%M:%S')}")
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

        # Ensure chart is initialized before use
        self._ensure_chart_initialized()

        # Build title with model/SN
        model = self.current_result.metadata.model if self.current_result else "?"
        serial = self.current_result.metadata.serial if self.current_result else "?"

        # Get trim date for display
        trim_date = None
        if self.current_result and self.current_result.metadata.file_date:
            trim_date = self.current_result.metadata.file_date.strftime('%m/%d/%Y')

        # Use ChartWidget's comparison method with model/SN title
        self.chart.plot_track_comparison(
            tracks,
            title=f"{model} - Track Comparison",
            serial_number=serial,
            trim_date=trim_date
        )

    def _on_filter_change(self, *args):
        """Handle filter change."""
        self._current_page = 0  # Reset to first page on filter change
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

    def _reanalyze_current(self):
        """Re-analyze the current file from its original path (updates DB with corrected values)."""
        if not self.current_result:
            return

        file_path = self.current_result.metadata.file_path
        if not file_path or not Path(file_path).exists():
            # Show file not found error - use smaller font for longer error text
            display_path = str(file_path)[:50] + "..." if len(str(file_path)) > 50 else str(file_path)
            self.status_text.configure(
                text=f"File not found:\n{display_path}",
                text_color="white",
                font=ctk.CTkFont(size=14, weight="bold")  # Smaller for error
            )
            self.status_banner.configure(fg_color="#8B0000")  # Dark red
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

        get_thread_manager().start_thread(target=process, name="reanalyze")

    def _on_reanalyze_complete(self, result: AnalysisResult):
        """Handle re-analysis completion."""
        self.reanalyze_btn.configure(state="normal")
        self.current_result = result

        # Show the updated result (updates chart, metrics, etc.)
        self._show_analysis_details(result)

        # Override status banner to show re-analysis success with status
        status_str = result.overall_status.value.upper()
        self.status_text.configure(
            text=f"✓ Re-analyzed: {status_str}",
            text_color="#27ae60",
            font=ctk.CTkFont(size=24, weight="bold")  # Reset to normal font
        )
        self.status_banner.configure(fg_color="#1e4d2b")

        # Refresh the list to show updated entry
        self._load_recent_analyses()

        logger.info(f"Re-analysis complete: {result.metadata.filename} - {status_str}")

    def _on_reanalyze_error(self, error: str):
        """Handle re-analysis error."""
        self.reanalyze_btn.configure(state="normal")
        # Truncate long error messages for display
        display_error = error[:80] + "..." if len(error) > 80 else error
        self.status_text.configure(
            text=f"Re-analysis error:\n{display_error}",
            text_color="white",
            font=ctk.CTkFont(size=14, weight="bold")  # Smaller for error
        )
        self.status_banner.configure(fg_color="#8B0000")  # Dark red, not bright red
        logger.error(f"Re-analysis failed: {error}")

    def _delete_current(self):
        """Delete the current analysis from the database."""
        if not self.current_result:
            return

        filename = self.current_result.metadata.filename
        model = self.current_result.metadata.model
        serial = self.current_result.metadata.serial

        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete this record?\n\n"
            f"Model: {model}\n"
            f"Serial: {serial}\n"
            f"File: {filename}\n\n"
            f"This action cannot be undone.",
            icon="warning"
        )

        if not confirm:
            return

        logger.info(f"Deleting analysis: {filename}")
        self.delete_btn.configure(state="disabled")
        self.status_text.configure(text="Deleting...", text_color="#f39c12")
        self.status_banner.configure(fg_color="#4d3d1e")

        # Delete in background thread
        def delete():
            try:
                db = get_database()
                success = db.delete_analysis_by_filename(filename)

                # Update UI on main thread
                self.after(0, lambda: self._on_delete_complete(success, filename))

            except Exception as e:
                logger.exception(f"Delete error: {e}")
                self.after(0, lambda: self._on_delete_error(str(e)))

        get_thread_manager().start_thread(target=delete, name="delete-analysis")

    def _on_delete_complete(self, success: bool, filename: str):
        """Handle deletion completion."""
        self.delete_btn.configure(state="normal")

        if success:
            # Clear current result
            self.current_result = None

            # Reset status banner
            self.status_text.configure(
                text="Record Deleted",
                text_color="#27ae60",
                font=ctk.CTkFont(size=24, weight="bold")
            )
            self.status_banner.configure(fg_color="#1e4d2b")

            # Disable buttons that need a selection
            self.export_btn.configure(state="disabled")
            self.export_chart_btn.configure(state="disabled")
            self.export_model_btn.configure(state="disabled")
            self.reanalyze_btn.configure(state="disabled")
            self.delete_btn.configure(state="disabled")
            self.track_selector.configure(state="disabled")

            # Show placeholder chart
            if self.chart:
                self.chart.show_placeholder("Select an analysis to view chart")

            # Clear metrics and info
            self._update_metrics("Record deleted. Select another analysis from the list.")
            self._update_info("Select an analysis from the list.")

            # Refresh the list
            self._load_recent_analyses()

            logger.info(f"Successfully deleted: {filename}")
        else:
            self.status_text.configure(
                text="Delete failed - record not found",
                text_color="white",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            self.status_banner.configure(fg_color="#8B0000")
            logger.warning(f"Delete failed - record not found: {filename}")

    def _on_delete_error(self, error: str):
        """Handle deletion error."""
        self.delete_btn.configure(state="normal")
        display_error = error[:80] + "..." if len(error) > 80 else error
        self.status_text.configure(
            text=f"Delete error:\n{display_error}",
            text_color="white",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_banner.configure(fg_color="#8B0000")
        logger.error(f"Delete failed: {error}")

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

        # RECALCULATE fail points using actual spec limits (DB may have incorrect values)
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
                    # Skip positions with no spec limit (None = unlimited)
                    if upper_limits[i] is not None and lower_limits[i] is not None:
                        if e > upper_limits[i] or e < lower_limits[i]:
                            actual_fail_count += 1

        # Create corrected values dict for export panels
        corrected_values = {
            'fail_points': actual_fail_count,
            'linearity_pass': actual_fail_count == 0,
        }

        # Create figure with subplots (larger for detailed export) - LIGHT mode
        fig = plt.figure(figsize=(14, 10), dpi=150, facecolor='white')

        # Title - black text for light mode (positioned at y=0.97)
        title = f'Laser Trim Analysis - {result.metadata.model} / {result.metadata.serial}'
        fig.suptitle(title, fontsize=16, fontweight='bold', color='black', y=0.97)

        # Add filename on separate line below title (y=0.94)
        fig.text(0.95, 0.94, f"File: {result.metadata.filename}",
                fontsize=9, ha='right', va='top', color='gray', style='italic')

        # Grid spec for layout (top adjusted down for title spacing)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3,
                              left=0.08, right=0.95, top=0.90, bottom=0.08)

        # ===== Main error plot (top 2/3) =====
        ax_main = fig.add_subplot(gs[0:2, :])
        self._plot_error_vs_position_export(ax_main, track)

        # ===== Unit Info (bottom left) =====
        ax_info = fig.add_subplot(gs[2, 0])
        self._plot_unit_info(ax_info, result, track)

        # ===== Metrics summary (bottom middle) - use corrected values =====
        ax_metrics = fig.add_subplot(gs[2, 1])
        self._plot_metrics_summary(ax_metrics, track, corrected_values)

        # ===== Status display (bottom right) - use corrected values =====
        ax_status = fig.add_subplot(gs[2, 2])
        self._plot_status_display(ax_status, track, result, corrected_values)

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

        # Plot spec limits (handle None = no limit at that position)
        if upper_limits and lower_limits and len(upper_limits) == len(positions):
            # Convert None to NaN for matplotlib (creates gaps in the line)
            upper_plot = np.array([u if u is not None else np.nan for u in upper_limits])
            lower_plot = np.array([l if l is not None else np.nan for l in lower_limits])

            ax.plot(positions, upper_plot, 'r--', linewidth=1.5,
                   label=f'Spec Limits (+/-{track.linearity_spec:.4f})', color=QA_COLORS['spec_limit'])
            ax.plot(positions, lower_plot, 'r--', linewidth=1.5,
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
        if track.resistance_change is not None:
            pct = track.resistance_change_percent
            pct_str = f" ({pct:+.1f}%)" if pct is not None else ""
            info_lines.append(f"R Change: {track.resistance_change:+.2f}{pct_str}")
        if track.trim_improvement_percent is not None:
            info_lines.append(f"Trim Improvement: {track.trim_improvement_percent:.1f}%")

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
            f"Sigma Pass: {'✓ YES' if track.sigma_pass else '✗ NO'}",
            "",
            f"Optimal Offset: {track.optimal_offset:.6f}",
            f"Linearity Error: {track.linearity_error:.6f}",
            f"Fail Points: {fail_points}",
            f"Linearity Pass: {'✓ YES' if linearity_pass else '✗ NO'}",
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

    # =========================================================================
    # Page Lifecycle
    # =========================================================================

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Analyze page shown")
        # Load recent analyses when page is shown
        self._load_recent_analyses()

    def on_hide(self):
        """Called when page becomes hidden - cleanup to free memory."""
        # Clear large data lists (will be reloaded on next on_show)
        self.recent_analyses = []
        self.current_result = None

        # Clear chart to free matplotlib resources
        if self.chart and hasattr(self.chart, 'figure'):
            try:
                self.chart.clear()
            except Exception as e:
                logger.debug(f"Chart cleanup warning: {e}")

    def load_result(self, result: AnalysisResult):
        """Load an analysis result from another page (e.g., Process page)."""
        self.current_result = result
        self._show_analysis_details(result)
