"""
Smoothness Page - Output Smoothness test results with charts.

Displays Output Smoothness test data, links to trim results,
and shows smoothness vs position charts.
"""

import customtkinter as ctk
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.utils.threads import get_thread_manager
from laser_trim_analyzer.gui.widgets.scrollable_combobox import ScrollableComboBox

if TYPE_CHECKING:
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget

logger = logging.getLogger(__name__)


class SmoothnessPage(ctk.CTkFrame):
    """Smoothness page for Output Smoothness test results."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.current_result: Optional[Dict[str, Any]] = None
        self.results_list: List[Dict[str, Any]] = []
        self._page_size = 20
        self._current_page = 0
        self._total_pages = 1
        self._chart: Optional['ChartWidget'] = None
        self._create_layout()

    def _create_layout(self):
        """Create the page layout."""
        # Row 0: top bar, Row 1: stats panel, Row 2: content (stretches)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Row 0 — Top bar with filters
        bar = ctk.CTkFrame(self)
        bar.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))

        ctk.CTkLabel(bar, text="Output Smoothness",
                    font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=10)

        ctk.CTkLabel(bar, text="Model:").pack(side="left", padx=(20, 5))
        self.model_dropdown = ScrollableComboBox(
            bar,
            values=["All Models"],
            command=lambda _: self._load_results(),
            width=120
        )
        self.model_dropdown.set("All Models")
        self.model_dropdown.pack(side="left", padx=5)

        ctk.CTkButton(bar, text="Refresh", width=80,
                     command=self._load_results).pack(side="left", padx=10)

        ctk.CTkButton(bar, text="Export to Excel", width=100,
                     command=self._export_to_excel).pack(side="left", padx=5)

        self.stats_label = ctk.CTkLabel(bar, text="", text_color="gray")
        self.stats_label.pack(side="right", padx=10)

        # Row 1 — Stats panel (comparison table or single-model card)
        stats_panel = ctk.CTkFrame(self)
        stats_panel.grid(row=1, column=0, sticky="ew", padx=5, pady=(5, 0))
        stats_panel.grid_columnconfigure(0, weight=1)

        self.stats_table_frame = ctk.CTkScrollableFrame(stats_panel, height=180)
        self.stats_table_frame.grid(row=0, column=0, sticky="ew")
        self.stats_table_frame.grid_columnconfigure(0, weight=1)

        self.stats_card_frame = ctk.CTkFrame(stats_panel)
        self.stats_card_frame.grid(row=0, column=0, sticky="ew")
        # Start with both hidden; _display_model_stats will show the right one
        self.stats_table_frame.grid_remove()
        self.stats_card_frame.grid_remove()

        # Row 2 — Main content
        content = ctk.CTkFrame(self)
        content.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=2)

        # Results list (left)
        list_frame = ctk.CTkFrame(content)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        self.results_scroll = ctk.CTkScrollableFrame(list_frame)
        self.results_scroll.grid(row=0, column=0, sticky="nsew")
        self.results_scroll.grid_columnconfigure(0, weight=1)

        page_frame = ctk.CTkFrame(list_frame)
        page_frame.grid(row=1, column=0, sticky="ew", pady=5)
        self.prev_btn = ctk.CTkButton(page_frame, text="<", width=30, command=self._prev_page)
        self.prev_btn.pack(side="left", padx=5)
        self.page_label = ctk.CTkLabel(page_frame, text="Page 1/1")
        self.page_label.pack(side="left", padx=5)
        self.next_btn = ctk.CTkButton(page_frame, text=">", width=30, command=self._next_page)
        self.next_btn.pack(side="left", padx=5)

        # Detail panel (right)
        detail = ctk.CTkFrame(content)
        detail.grid(row=0, column=1, sticky="nsew")
        detail.grid_rowconfigure(1, weight=1)
        detail.grid_columnconfigure(0, weight=1)

        self.info_text = ctk.CTkTextbox(detail, height=120)
        self.info_text.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.chart_frame = ctk.CTkFrame(detail)
        self.chart_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def _get_chart(self) -> 'ChartWidget':
        """Lazy-load the chart widget."""
        if self._chart is None:
            from laser_trim_analyzer.gui.widgets.chart import ChartWidget
            self._chart = ChartWidget(self.chart_frame)
            self._chart.pack(fill="both", expand=True)
        return self._chart

    def on_show(self):
        """Called when page becomes visible."""
        self._load_models()
        self._load_results()

    def on_hide(self):
        """Called when page is hidden."""
        if self._chart:
            self._chart.clear()

    def _load_models(self):
        """Load model list for dropdown (background thread)."""
        from laser_trim_analyzer.utils.threads import get_thread_manager

        def _fetch():
            try:
                db = get_database()
                models = db.get_models_list()
                self.after(0, lambda: self.model_dropdown.configure(
                    values=["All Models"] + models
                ) if self.winfo_exists() else None)
            except Exception as e:
                logger.warning(f"Could not load models: {e}")

        get_thread_manager().start_thread(target=_fetch, name="smoothness-models")

    def _load_results(self):
        """Load smoothness results from database in background."""
        # Capture filter on main thread
        model = self.model_dropdown.get()
        if model == "All Models":
            model = None

        def _do_load():
            try:
                db = get_database()
                results = db.search_smoothness_results(model=model, limit=500)
                stats = db.get_smoothness_stats()
                model_stats = db.get_smoothness_stats_by_model(model=model)
                return results, stats, model_stats
            except Exception as e:
                logger.error(f"Error loading smoothness results: {e}")
                return [], {}, []

        def _on_done(load_result):
            results, stats, model_stats = load_result
            def _apply():
                if not self.winfo_exists():
                    return
                self.results_list = results
                self._current_page = 0
                self._total_pages = max(1, (len(results) + self._page_size - 1) // self._page_size)
                self._display_results()
                self._display_stats(stats)
                self._display_model_stats(model_stats, is_single=(model is not None))
            self.after(0, _apply)

        get_thread_manager().start_thread(
            target=lambda: _on_done(_do_load()),
            name="smoothness-load"
        )

    def _display_results(self):
        """Display current page of results."""
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        start = self._current_page * self._page_size
        end = start + self._page_size
        page_results = self.results_list[start:end]

        if not page_results:
            msg = ("No smoothness data found.\n\n"
                   "To populate this page, process Output Smoothness\n"
                   "test files via the Process Files page.")
            ctk.CTkLabel(self.results_scroll, text=msg,
                        text_color="gray", justify="center").grid(row=0, column=0, pady=20)
            return

        for i, result in enumerate(page_results):
            self._create_result_row(i, result)

        self.page_label.configure(text=f"Page {self._current_page + 1}/{self._total_pages}")

    def _create_result_row(self, row: int, result: Dict[str, Any]):
        """Create a result row."""
        status = result.get("overall_status", "UNKNOWN")
        is_pass = status.upper() == "PASS" if status else False

        frame = ctk.CTkFrame(self.results_scroll, cursor="hand2")
        frame.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        frame.grid_columnconfigure(1, weight=1)

        color = "#27ae60" if is_pass else "#e74c3c"
        ctk.CTkLabel(frame, text="  ", width=8,
                    fg_color=color, corner_radius=2).grid(row=0, column=0, padx=(2, 5), pady=2)

        text = f"{result['model']} SN{result['serial']}"
        if result.get("element_label"):
            text += f" ({result['element_label']})"
        ctk.CTkLabel(frame, text=text, font=ctk.CTkFont(size=12)).grid(
            row=0, column=1, sticky="w", padx=2)

        fd = result.get("file_date")
        date_str = fd.strftime("%Y-%m-%d") if hasattr(fd, 'strftime') else str(fd)[:10] if fd else ""
        ctk.CTkLabel(frame, text=date_str, text_color="gray",
                    font=ctk.CTkFont(size=11)).grid(row=0, column=2, padx=5)

        frame.bind("<Button-1>", lambda e, r=result: self._on_result_selected(r))
        for child in frame.winfo_children():
            child.bind("<Button-1>", lambda e, r=result: self._on_result_selected(r))

    def _on_result_selected(self, result: Dict[str, Any]):
        """Handle result selection."""
        result_id = result["id"]

        def _do_load():
            try:
                db = get_database()
                return db.get_smoothness_result(result_id)
            except Exception as e:
                logger.error(f"Error loading smoothness detail: {e}")
                return None

        def _on_done(detail):
            if detail:
                self.after(0, lambda: self._display_detail(detail))

        get_thread_manager().start_thread(
            target=lambda: _on_done(_do_load()),
            name="smoothness-detail"
        )

    def _display_detail(self, detail: Dict[str, Any]):
        """Display selected result detail and chart."""
        if not self.winfo_exists():
            return
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")

        lines = [f"Model: {detail['model']}    Serial: {detail['serial']}"]
        if detail.get("element_label"):
            lines.append(f"Element: {detail['element_label']}")
        lines.append(f"Status: {detail['overall_status']}")
        if detail.get("smoothness_spec"):
            lines.append(f"Spec Limit: {detail['smoothness_spec']}")
        if detail.get("max_smoothness_value") is not None:
            lines.append(f"Max Smoothness: {detail['max_smoothness_value']:.4f}")
        if detail.get("linked_trim_id"):
            lines.append(f"Linked Trim ID: {detail['linked_trim_id']}")
            if detail.get("match_confidence"):
                lines.append(f"Match: {detail.get('match_method', 'unknown')} ({detail['match_confidence']:.0%})")

        self.info_text.insert("1.0", "\n".join(lines))
        self.info_text.configure(state="disabled")

        # Plot chart - if there's no track data, show a helpful placeholder
        # explaining that the file needs to be re-imported. (Records imported
        # before the smoothness_tracks fix are missing the position/value
        # arrays - the result row exists but the chart can't render.)
        chart = self._get_chart()
        tracks = detail.get("tracks") or []
        track = tracks[0] if tracks else None
        positions = (track or {}).get("positions") or []
        values = (track or {}).get("smoothness_values") or []

        if positions and values:
            chart.plot_smoothness(
                positions=positions,
                smoothness_values=values,
                spec_limit=(track or {}).get("smoothness_spec") or detail.get("smoothness_spec"),
                title=f"Output Smoothness - {detail['model']} SN{detail['serial']}",
                serial_number=detail["serial"],
                test_date=detail["test_date"].strftime("%Y-%m-%d") if detail.get("test_date") else None,
                element_label=detail.get("element_label"),
            )
        else:
            chart.show_placeholder(
                "No track data stored for this record.\n\n"
                "This file was imported before the smoothness fix.\n"
                "Re-process the source file via Process Files to populate the chart."
            )

    def _display_model_stats(self, model_stats: List[Dict[str, Any]], is_single: bool = False):
        """Show per-model stats panel (table or single-model card)."""
        # Clear both frames
        for child in self.stats_table_frame.winfo_children():
            child.destroy()
        for child in self.stats_card_frame.winfo_children():
            child.destroy()

        if not model_stats:
            self.stats_table_frame.grid_remove()
            self.stats_card_frame.grid_remove()
            return

        if is_single:
            self.stats_table_frame.grid_remove()
            self.stats_card_frame.grid()
            self._build_single_model_card(model_stats[0])
        else:
            self.stats_card_frame.grid_remove()
            self.stats_table_frame.grid()
            self._build_comparison_table(model_stats)

    def _build_comparison_table(self, model_stats: List[Dict[str, Any]]):
        """Build the all-models comparison table inside stats_table_frame."""
        frame = self.stats_table_frame
        frame.grid_columnconfigure(0, weight=1)

        headers = ["Model", "Count", "Pass Rate", "Avg Max", "Worst Case", "Spec Limit", "Margin"]
        header_font = ctk.CTkFont(size=11, weight="bold")
        header_color = "#aaaaaa"

        # Header row
        hdr = ctk.CTkFrame(frame)
        hdr.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 0))
        for ci in range(len(headers)):
            hdr.grid_columnconfigure(ci, weight=1)
        for ci, h in enumerate(headers):
            ctk.CTkLabel(hdr, text=h, font=header_font, text_color=header_color).grid(
                row=0, column=ci, sticky="w", padx=6, pady=2)

        # Data rows
        for ri, s in enumerate(model_stats, start=1):
            row_frame = ctk.CTkFrame(frame)
            row_frame.grid(row=ri, column=0, sticky="ew", padx=2, pady=1)
            for ci in range(len(headers)):
                row_frame.grid_columnconfigure(ci, weight=1)

            # Model name — clickable
            model_lbl = ctk.CTkLabel(
                row_frame, text=s.get("model", ""),
                text_color="#3498db", cursor="hand2",
                font=ctk.CTkFont(size=12))
            model_lbl.grid(row=0, column=0, sticky="w", padx=6, pady=2)
            model_name = s.get("model", "")
            model_lbl.bind("<Button-1>", lambda e, m=model_name: self._select_model(m))

            # Count
            ctk.CTkLabel(row_frame, text=str(s.get("count", 0)),
                        font=ctk.CTkFont(size=12)).grid(
                row=0, column=1, sticky="w", padx=6, pady=2)

            # Pass rate — color-coded
            pr = s.get("pass_rate")
            pr_text = f"{pr:.0f}%" if pr is not None else "N/A"
            if pr is not None:
                if pr < 80:
                    pr_color = "#e74c3c"
                elif pr < 95:
                    pr_color = "#f39c12"
                else:
                    pr_color = "#27ae60"
            else:
                pr_color = "gray"
            ctk.CTkLabel(row_frame, text=pr_text, text_color=pr_color,
                        font=ctk.CTkFont(size=12, weight="bold")).grid(
                row=0, column=2, sticky="w", padx=6, pady=2)

            # Avg max smoothness
            avg = s.get("avg_max_smoothness")
            avg_text = f"{avg:.4f}" if avg is not None else "N/A"
            ctk.CTkLabel(row_frame, text=avg_text,
                        font=ctk.CTkFont(size=12)).grid(
                row=0, column=3, sticky="w", padx=6, pady=2)

            # Worst case
            worst = s.get("worst_case")
            worst_text = f"{worst:.4f}" if worst is not None else "N/A"
            ctk.CTkLabel(row_frame, text=worst_text,
                        font=ctk.CTkFont(size=12)).grid(
                row=0, column=4, sticky="w", padx=6, pady=2)

            # Spec limit
            spec = s.get("spec_limit")
            spec_text = f"{spec:.4f}" if spec is not None else "N/A"
            ctk.CTkLabel(row_frame, text=spec_text,
                        font=ctk.CTkFont(size=12)).grid(
                row=0, column=5, sticky="w", padx=6, pady=2)

            # Margin — color-coded
            margin = s.get("margin")
            if margin is not None:
                margin_text = f"{margin:.4f}"
                if margin < 0:
                    margin_color = "#e74c3c"
                elif spec is not None and spec > 0 and margin < 0.20 * spec:
                    margin_color = "#f39c12"
                else:
                    margin_color = "#27ae60"
            else:
                margin_text = "N/A"
                margin_color = "gray"
            ctk.CTkLabel(row_frame, text=margin_text, text_color=margin_color,
                        font=ctk.CTkFont(size=12)).grid(
                row=0, column=6, sticky="w", padx=6, pady=2)

    def _build_single_model_card(self, s: Dict[str, Any]):
        """Build a horizontal stats card for a single model."""
        frame = self.stats_card_frame

        items = [
            ("Pass Rate", s.get("pass_rate"), "%", True),
            ("Avg Max Smoothness", s.get("avg_max_smoothness"), "", False),
            ("Worst Case", s.get("worst_case"), "", False),
            ("Spec Limit", s.get("spec_limit"), "", False),
            ("Margin", s.get("margin"), "", False),
        ]

        for label_text, value, suffix, is_pass_rate in items:
            item_frame = ctk.CTkFrame(frame)
            item_frame.pack(side="left", padx=12, pady=8)

            ctk.CTkLabel(item_frame, text=label_text,
                        font=ctk.CTkFont(size=10), text_color="gray").pack(anchor="w")

            if is_pass_rate and value is not None:
                val_text = f"{value:.0f}{suffix}"
                if value < 80:
                    val_color = "#e74c3c"
                elif value < 95:
                    val_color = "#f39c12"
                else:
                    val_color = "#27ae60"
            elif label_text == "Margin":
                if value is not None:
                    val_text = f"{value:.4f}"
                    spec = s.get("spec_limit")
                    if value < 0:
                        val_color = "#e74c3c"
                    elif spec is not None and spec > 0 and value < 0.20 * spec:
                        val_color = "#f39c12"
                    else:
                        val_color = "#27ae60"
                else:
                    val_text = "N/A"
                    val_color = "gray"
            elif value is not None:
                val_text = f"{value:.4f}{suffix}"
                val_color = None  # default
            else:
                val_text = "N/A"
                val_color = "gray"

            lbl = ctk.CTkLabel(item_frame, text=val_text,
                              font=ctk.CTkFont(size=14, weight="bold"))
            if val_color:
                lbl.configure(text_color=val_color)
            lbl.pack(anchor="w")

    def _select_model(self, model: str):
        """Select a model in the dropdown and reload results."""
        self.model_dropdown.set(model)
        self._load_results()

    def _export_to_excel(self):
        """Export smoothness data to Excel. Implemented in Task 3."""
        pass

    def _display_stats(self, stats: Dict[str, Any]):
        """Display summary stats."""
        if not stats or stats.get("total", 0) == 0:
            self.stats_label.configure(text="No smoothness data imported yet")
            return
        text = (
            f"Total: {stats['total']}  |  "
            f"Pass: {stats['pass_rate']:.0f}%  |  "
            f"Linked: {stats['link_rate']:.0f}%"
        )
        self.stats_label.configure(text=text)

    def _prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._display_results()

    def _next_page(self):
        if self._current_page < self._total_pages - 1:
            self._current_page += 1
            self._display_results()
