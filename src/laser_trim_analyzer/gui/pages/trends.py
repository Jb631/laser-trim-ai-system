"""
Trends Page - Historical analysis and ML insights.

Redesigned to show:
- "All Models" view: Summary of best/worst active models, models needing attention
- Specific model view: SPC scatter chart with threshold, rolling average, detailed stats

Features:
- Focus on models with recent activity (last 90 days)
- Adjustable rolling average window (7/14/30/60 days)
- Alert criteria: <80% pass rate, trending worse, high variance
- Best/worst performing models at a glance
"""

import customtkinter as ctk
import math
import tkinter as tk
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

from laser_trim_analyzer.utils.threads import get_thread_manager

import numpy as np

from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.config import get_config
from laser_trim_analyzer.gui.widgets.scrollable_combobox import ScrollableComboBox

# Lazy import for ChartWidget - defer matplotlib loading until first use
if TYPE_CHECKING:
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle

logger = logging.getLogger(__name__)

# Module-level flag to track if ChartWidget has been imported
_chart_module_loaded = False
_ChartWidget = None
_ChartStyle = None


def _ensure_chart_module():
    """Lazily load ChartWidget module - defers matplotlib loading."""
    global _chart_module_loaded, _ChartWidget, _ChartStyle
    if not _chart_module_loaded:
        from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle
        _ChartWidget = ChartWidget
        _ChartStyle = ChartStyle
        _chart_module_loaded = True
        logger.debug("ChartWidget module loaded (matplotlib initialized)")
    return _ChartWidget, _ChartStyle


# ============================================================================
# Drift-dashboard helpers (module-level for testability)
# ============================================================================

# Drift-panel visual constants — shared across _draw_smoothed_panel and
# _draw_sigma_panel so palette changes are a single-point edit.
_BASELINE_COLOR = "#888"
_BASELINE_CUTOFF_COLOR = "#fd7e14"
_REF_LINEWIDTH = 0.7
_MEAN_LINEWIDTH = 1.8

# Sigma-panel specific colors
_SIGMA_LIMIT_COLOR = "#dc3545"      # red dashed UCL/LCL
_SIGMA_IN_CONTROL_DOT = "#7ed99e"  # green dot
_SIGMA_OOC_DOT = "#ff4040"         # red dot for out-of-control points
_SIGMA_NO_BASELINE_DOT = "#888888" # gray when there's no baseline
_SIGMA_OVERLAY_COLOR = "#ffffff"   # smoothed mean overlay
_SIGMA_CENTER_COLOR = "#666"
_SIGMA_CENTER_LINEWIDTH = 0.5

# Z-score thresholds for status pill color in process panels.
_DRIFT_Z_WARN = 2.0      # orange "↑ DRIFT" if |z| >= this
_DRIFT_Z_ALARM = 3.0     # red "↑ OOC" if |z| >= this

# Retrim-rate panel pill thresholds (panel-specific).
_RETRIM_RISING_FLOOR_PCT = 10.0   # "rising" requires recent rate >= this %
_RETRIM_RISING_RATIO = 2.0        # "rising" requires recent >= ratio × baseline
_RETRIM_OOC_PCT = 15.0            # "OOC" if recent rate >= this %
_RETRIM_Y_MIN_UPPER = 20.0        # retrim panel y-axis upper bound floor

# Status colors used for chart line/band fills (process + retrim panels).
# Differ from the sigma-dot colors; lighter so they read well behind data.
_STATUS_STABLE_COLOR = "#7ed99e"
_STATUS_WARN_COLOR = "#ffb060"
_STATUS_ALARM_COLOR = "#ff8080"


def _compute_buckets(
    series: List[Tuple[str, float]],
    n_per_bucket: int = 50,
) -> List[Dict[str, Any]]:
    """Group a time-ordered (iso_date, value) series into adaptive buckets.

    A new bucket is emitted every `n_per_bucket` rows. The trailing partial
    bucket is kept as its own bucket if it holds at least 5 rows; otherwise
    it folds into the previous bucket (so a tail of a few stragglers doesn't
    distort the chart with a tiny low-confidence point). If there's no
    previous bucket to fold into, the tail is rendered as-is.

    Returns a list of dicts with keys:
        bucket_index, n, mean, stddev, se, min_date, max_date

    `stddev` and `se` are 0.0 for n=1 (avoids division-by-zero).
    """
    if not series:
        return []

    raw_buckets: List[List[Tuple[str, float]]] = []
    cur: List[Tuple[str, float]] = []
    for entry in series:
        cur.append(entry)
        if len(cur) >= n_per_bucket:
            raw_buckets.append(cur)
            cur = []

    if cur:
        if len(cur) >= 5 or not raw_buckets:
            raw_buckets.append(cur)
        else:
            raw_buckets[-1].extend(cur)

    out: List[Dict[str, Any]] = []
    for idx, rows in enumerate(raw_buckets):
        values = [v for _, v in rows]
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            var = sum((x - mean) ** 2 for x in values) / (n - 1)
            stddev = var ** 0.5
            se = stddev / math.sqrt(n)
        else:
            stddev = 0.0
            se = 0.0
        out.append({
            "bucket_index": idx,
            "n": n,
            "mean": mean,
            "stddev": stddev,
            "se": se,
            "min_date": rows[0][0],
            "max_date": rows[-1][0],
        })

    return out


def _cutoff_bucket_index(
    series: List[Tuple[str, float]],
    baseline_cutoff_iso: Optional[str],
    n_per_bucket: int = 50,
) -> Optional[int]:
    """Find the bucket index that marks the baseline → recent boundary.

    Walks the series until the first row whose date is on or after the
    cutoff, then returns that row's bucket index (using the same
    `i // n_per_bucket` formula as _compute_buckets).

    Returns None when there's no series, no cutoff supplied, or the
    cutoff is past the end of the window.
    """
    if not baseline_cutoff_iso or not series:
        return None
    cutoff = datetime.fromisoformat(baseline_cutoff_iso)
    for i, (iso, _) in enumerate(series):
        if datetime.fromisoformat(iso) >= cutoff:
            return i // n_per_bucket
    return None


def _draw_smoothed_panel(
    ax,
    buckets: List[Dict[str, Any]],
    baseline_mean: Optional[float],
    baseline_cutoff_bucket_index: Optional[int],
    color: str,
) -> None:
    """Draw a process-metric panel: smoothed mean + confidence band.

    Adds, in order:
      - a horizontal gray dashed line at baseline_mean (if not None)
      - a vertical orange dashed line at baseline_cutoff_bucket_index (if not None)
      - the +/-1 SE confidence band (filled, alpha 0.15; when >= 2 buckets)
      - the smoothed mean line, drawn on top of the band (always, when >= 2 buckets)
      - a single dot at the bucket value when there's exactly one bucket
        (single-bucket case)

    The caller is responsible for setting the title, subtitle, x/y limits,
    and ticks. This helper is intentionally pure with respect to those —
    it only draws series content into the Axes.
    """
    if not buckets:
        return

    # Reference lines first so they end up under the series
    if baseline_mean is not None:
        ax.axhline(baseline_mean, color=_BASELINE_COLOR, linestyle="--", linewidth=_REF_LINEWIDTH)
    if baseline_cutoff_bucket_index is not None:
        ax.axvline(
            baseline_cutoff_bucket_index,
            color=_BASELINE_CUTOFF_COLOR, linestyle="--", linewidth=_REF_LINEWIDTH,
        )

    xs = [b["bucket_index"] for b in buckets]
    means = [b["mean"] for b in buckets]

    if len(buckets) == 1:
        # Single-bucket case: dot only, no line / no band.
        ax.scatter([xs[0]], [means[0]], color=color, s=40, zorder=3)
        return

    # >= 2 buckets: smoothed line + confidence band
    ses = [b["se"] for b in buckets]
    upper = [m + s for m, s in zip(means, ses)]
    lower = [m - s for m, s in zip(means, ses)]

    ax.fill_between(xs, lower, upper, color=color, alpha=0.15, linewidth=0)
    ax.plot(xs, means, color=color, linewidth=_MEAN_LINEWIDTH)


def _draw_sigma_panel(
    ax,
    sigma_series: List[Tuple[str, float]],
    detector,
    baseline_cutoff_bucket_index: Optional[int],
    n_per_bucket: int = 50,
) -> int:
    """Draw the hybrid SPC sigma panel: individual unit dots (green = in-control,
    red = out-of-control by Western Electric rule 1) + smoothed white mean
    overlay + UCL/LCL/center reference lines.

    Returns the count of out-of-control unit dots in the entire window (for the
    header pill).
    """
    if not sigma_series:
        return 0

    has_baseline = bool(getattr(detector, "has_baseline", False))
    if has_baseline:
        lcl, center, ucl = detector.get_control_limits()
    else:
        lcl, center, ucl = None, None, None

    # Slight per-dot horizontal jitter so coincident sigma values at the same
    # bucket index don't completely overlap. Deterministic (index-based) so
    # rendering is reproducible.
    xs: List[float] = []
    ys: List[float] = []
    colors: List[str] = []
    violations = 0
    for i, (_, value) in enumerate(sigma_series):
        bucket_idx = i // n_per_bucket
        # Divisor 2.5 keeps max |jitter| below 0.2, well inside the bucket width.
        jitter = ((i % n_per_bucket) - n_per_bucket / 2) / (n_per_bucket * 2.5)
        xs.append(bucket_idx + jitter)
        ys.append(value)
        if has_baseline and (
            (lcl is not None and value < lcl) or (ucl is not None and value > ucl)
        ):
            colors.append(_SIGMA_OOC_DOT)
            violations += 1
        else:
            colors.append(_SIGMA_IN_CONTROL_DOT if has_baseline else _SIGMA_NO_BASELINE_DOT)

    # Reference lines before dots so they end up under the series.
    if has_baseline:
        if ucl is not None:
            ax.axhline(ucl, color=_SIGMA_LIMIT_COLOR, linestyle="--", linewidth=_REF_LINEWIDTH)
        if lcl is not None:
            ax.axhline(lcl, color=_SIGMA_LIMIT_COLOR, linestyle="--", linewidth=_REF_LINEWIDTH)
        if center is not None:
            ax.axhline(center, color=_SIGMA_CENTER_COLOR, linewidth=_SIGMA_CENTER_LINEWIDTH)
    if baseline_cutoff_bucket_index is not None:
        ax.axvline(
            baseline_cutoff_bucket_index,
            color=_BASELINE_CUTOFF_COLOR, linestyle="--", linewidth=_REF_LINEWIDTH,
        )

    ax.scatter(xs, ys, c=colors, s=10, alpha=0.6, linewidths=0, zorder=2)

    # Smoothed mean overlay on top of the dots — only when baseline exists
    # (no point smoothing into nonexistent control bands).
    if has_baseline:
        buckets = _compute_buckets(sigma_series, n_per_bucket=n_per_bucket)
        if len(buckets) >= 2:
            bxs = [b["bucket_index"] for b in buckets]
            means = [b["mean"] for b in buckets]
            ax.plot(bxs, means, color=_SIGMA_OVERLAY_COLOR, linewidth=_MEAN_LINEWIDTH, zorder=3)

    return violations


class _Sparkline(tk.Canvas):
    """Tiny inline line chart drawn on a tk.Canvas. Used inside table rows.

    Pure tkinter (no matplotlib) so per-row rendering is cheap.
    """

    def __init__(self, parent, width=80, height=14, **kwargs):
        # tk.Canvas defaults to the system color (white on macOS, light gray
        # on Windows). On the dark CTk theme that paints a bright rectangle
        # inside every table row. Pick the bg from the current appearance
        # mode so the sparkline blends into its parent CTkFrame.
        bg = kwargs.pop("bg", None)
        if bg is None:
            try:
                mode = ctk.get_appearance_mode()
            except Exception:
                mode = "Dark"
            bg = "#2b2b2b" if mode == "Dark" else "#ebebeb"
        super().__init__(
            parent, width=width, height=height,
            highlightthickness=0, bd=0, bg=bg, **kwargs,
        )
        self._canvas_w = width
        self._canvas_h = height

    def draw(self, values: list, color: str = "#7ed99e") -> None:
        self.delete("all")
        if not values or len(values) < 2:
            return
        v_min = min(values)
        v_max = max(values)
        span = v_max - v_min if v_max != v_min else 1.0
        n = len(values)
        # Map values to canvas coordinates with 1-px top/bottom margin
        h = self._canvas_h - 2
        pts = []
        for i, v in enumerate(values):
            x = i * (self._canvas_w - 1) / (n - 1)
            # invert so higher values plot toward the top
            y = 1 + h - (v - v_min) / span * h
            pts.extend((x, y))
        self.create_line(*pts, fill=color, width=1.2, smooth=False)


class _SortableTable(ctk.CTkScrollableFrame):
    """Grid layout with click-to-sort headers and click-to-select rows.

    Columns is a list of (key, label, render) tuples:
      - key: string used for sort ordering (rows are dicts keyed by this)
      - label: header text
      - render: callable(parent, row_dict) -> tk widget (or None to render
                row_dict[key] as plain text via a CTkLabel).

    Rows are dicts. Pass row_click=fn to be notified on row selection;
    the callback receives the row dict.
    """

    def __init__(
        self,
        parent,
        columns,
        rows,
        row_click=None,
        default_sort_key=None,
        default_sort_reverse=False,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self._columns = columns
        self._rows = list(rows)
        self._row_click = row_click
        self._sort_key = default_sort_key
        self._sort_reverse = default_sort_reverse
        self._build()

    def _build(self):
        for w in self.winfo_children():
            w.destroy()
        # Header row
        for col_idx, (key, label, _) in enumerate(self._columns):
            arrow = ""
            if key == self._sort_key:
                arrow = " ↓" if self._sort_reverse else " ↑"
            btn = ctk.CTkButton(
                self,
                text=f"{label}{arrow}",
                anchor="w",
                fg_color="transparent",
                hover_color=("gray85", "gray25"),
                text_color=("gray20", "gray80"),
                font=ctk.CTkFont(size=10, weight="bold"),
                height=22,
                command=lambda k=key: self._on_sort(k),
            )
            btn.grid(row=0, column=col_idx, sticky="ew", padx=2, pady=(2, 4))

        # Data rows
        sorted_rows = self._sorted_rows()
        for row_idx, row in enumerate(sorted_rows, start=1):
            for col_idx, (key, _, render) in enumerate(self._columns):
                if render is None:
                    val = row.get(key)
                    text = "" if val is None else str(val)
                    cell = ctk.CTkLabel(
                        self, text=text, anchor="w",
                        font=ctk.CTkFont(size=10),
                    )
                else:
                    cell = render(self, row)
                cell.grid(row=row_idx, column=col_idx, sticky="ew", padx=4, pady=1)
                if self._row_click is not None:
                    cell.bind(
                        "<Button-1>",
                        lambda _e, r=row: self._row_click(r),
                    )

        # Stretch all columns evenly
        for col_idx in range(len(self._columns)):
            self.grid_columnconfigure(col_idx, weight=1)

    def _sorted_rows(self):
        if self._sort_key is None:
            return self._rows

        def keyfn(r):
            v = r.get(self._sort_key)
            if v is None:
                # Push None to the end regardless of direction
                return (1, 0)
            if isinstance(v, (int, float)):
                return (0, v)
            return (0, str(v))

        return sorted(self._rows, key=keyfn, reverse=self._sort_reverse)

    def _on_sort(self, key):
        if self._sort_key == key:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_key = key
            self._sort_reverse = False
        self._build()

    def update_rows(self, rows):
        """Replace the row data and re-render."""
        self._rows = list(rows)
        self._build()


class TrendsPage(ctk.CTkFrame):
    """
    Trends page for historical analysis.

    Two modes:
    1. Summary Mode ("All Models") - Shows best/worst models, alerts
    2. Detail Mode (specific model) - Shows SPC scatter, rolling average, stats
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.selected_model: str = "All Models"
        self.selected_days: int = 3650  # Default to All Time
        self.rolling_window: int = 30
        self.chart_timeline_days: int = 0  # 0 = all data
        self.active_models_data: List[Dict[str, Any]] = []
        self.model_trend_data: Optional[Dict[str, Any]] = None

        # Track chart widgets for proper cleanup
        self._chart_widgets: List["ChartWidget"] = []

        # Lazy chart initialization flags
        self._summary_charts_initialized = False
        self._detail_charts_initialized = False

        # Generation counter — bumped on every navigation/filter change.
        # Background loads capture this value at launch and the after()
        # callbacks discard themselves if a newer load has superseded
        # them. Without this, a stale _load_summary_data callback fires
        # AFTER _create_detail_view destroyed _alerts_frame and tries to
        # create a ChartWidget on the dead frame, raising
        # _tkinter.TclError "bad window path name". Same pattern as
        # ComparePage._load_generation.
        self._load_generation = 0

        # Drift tab sub-view selector. Set when the user clicks the toggle
        # inside the Drift tab; consumed by render guards in
        # _render_drift_timeline and _render_process_drift_table to discard a
        # stale render that fires after the user has flipped to the other
        # sub-view.
        self._drift_subtab: str = "ML Drift"

        # Global model filter for the Drift tab. None / "All models" → all-models view.
        # Set when the user picks a model from the new dropdown in _create_drift_view.
        self._drift_filter_model: Optional[str] = None

        # Which physical metric is shown on the Process Drift sub-tab.
        self._process_drift_metric: str = "untrimmed_resistance"

        self._create_ui()

    def _create_ui(self):
        """Create the trends page UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header with trend type selector
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)

        ctk.CTkLabel(
            header_frame,
            text="Trends & ML Insights",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(side="left")

        self._trend_type = ctk.CTkSegmentedButton(
            header_frame,
            values=["Standard", "Drift", "Trim Difficulty"],
            command=self._on_trend_type_changed,
        )
        self._trend_type.set("Standard")
        self._trend_type.pack(side="right", padx=10)

        # Controls frame
        controls = ctk.CTkFrame(self)
        controls.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))

        # Model selector - use ScrollableComboBox for many models
        model_label = ctk.CTkLabel(controls, text="Model:")
        model_label.pack(side="left", padx=(15, 5), pady=15)

        self.model_dropdown = ScrollableComboBox(
            controls,
            values=["All Models"],
            command=self._on_model_change,
            width=150,
            dropdown_height=300,  # Scrollable dropdown
        )
        self.model_dropdown.set("All Models")
        self.model_dropdown.pack(side="left", padx=5, pady=15)

        # Active Only filter - hide inactive models from dropdown
        self.active_only_var = ctk.BooleanVar(value=True)
        self.active_only_check = ctk.CTkCheckBox(
            controls,
            text="Active Only",
            variable=self.active_only_var,
            command=self._on_active_filter_change,
            width=100,
            checkbox_width=18,
            checkbox_height=18,
        )
        self.active_only_check.pack(side="left", padx=(5, 5), pady=15)

        # Minimum sample size filter
        min_samples_label = ctk.CTkLabel(controls, text="Min samples:")
        min_samples_label.pack(side="left", padx=(10, 2), pady=15)
        self.min_samples_entry = ctk.CTkEntry(controls, width=40, justify="center")
        self.min_samples_entry.pack(side="left", padx=2, pady=15)
        self.min_samples_entry.insert(0, "20")

        # Failure rate threshold filter
        fail_rate_label = ctk.CTkLabel(controls, text="Min fail%:")
        fail_rate_label.pack(side="left", padx=(10, 2), pady=15)
        self.fail_rate_entry = ctk.CTkEntry(controls, width=40, justify="center")
        self.fail_rate_entry.pack(side="left", padx=2, pady=15)
        self.fail_rate_entry.insert(0, "0")

        # Date range for active models consideration (by trim date)
        date_label = ctk.CTkLabel(controls, text="Trim Date:")
        date_label.pack(side="left", padx=(10, 5), pady=15)

        self.date_dropdown = ctk.CTkOptionMenu(
            controls,
            values=["Last 30 Days", "Last 90 Days", "Last Year", "All Time"],
            command=self._on_date_change
        )
        self.date_dropdown.set("All Time")
        self.date_dropdown.pack(side="left", padx=5, pady=15)

        # Element type filter
        et_label = ctk.CTkLabel(controls, text="Element:")
        et_label.pack(side="left", padx=(10, 2), pady=15)
        self._element_filter = ctk.CTkComboBox(
            controls,
            values=["All"],
            command=self._on_spec_filter_change,
            width=100,
            font=ctk.CTkFont(size=11)
        )
        self._element_filter.set("All")
        self._element_filter.pack(side="left", padx=2, pady=15)

        # Product class filter
        pc_label = ctk.CTkLabel(controls, text="Class:")
        pc_label.pack(side="left", padx=(5, 2), pady=15)
        self._class_filter = ctk.CTkComboBox(
            controls,
            values=["All"],
            command=self._on_spec_filter_change,
            width=100,
            font=ctk.CTkFont(size=11)
        )
        self._class_filter.set("All")
        self._class_filter.pack(side="left", padx=2, pady=15)

        # Rolling average window (only shown in detail mode — hidden by default)
        self.rolling_label = ctk.CTkLabel(controls, text="Rolling Avg:")
        self.rolling_dropdown = ctk.CTkOptionMenu(
            controls,
            values=["7 Days", "14 Days", "30 Days", "60 Days"],
            command=self._on_rolling_change
        )
        self.rolling_dropdown.set("30 Days")
        # Not packed until detail mode is activated

        # Refresh button
        refresh_btn = ctk.CTkButton(
            controls,
            text="Refresh",
            command=self._refresh_data,
            width=80
        )
        refresh_btn.pack(side="right", padx=5, pady=15)

        # Export PDF button
        export_pdf_btn = ctk.CTkButton(
            controls,
            text="Export PDF",
            command=self._export_summary_pdf,
            width=90
        )
        export_pdf_btn.pack(side="right", padx=5, pady=15)

        # Status label
        self.status_label = ctk.CTkLabel(
            controls,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=10)
        )
        self.status_label.pack(side="right", padx=15, pady=15)

        # Main content area - will be dynamically updated
        self.content = ctk.CTkScrollableFrame(self)
        self.content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.content.grid_columnconfigure(0, weight=1)

        # Create placeholder content
        self._create_summary_view()

    def _cleanup_charts(self):
        """Properly destroy chart widgets to free matplotlib resources."""
        import matplotlib.pyplot as plt

        for chart in self._chart_widgets:
            try:
                # Explicitly close the figure before destroying widget
                if hasattr(chart, 'figure') and chart.figure:
                    plt.close(chart.figure)
                chart.destroy()
            except Exception as e:
                logger.debug(f"Chart cleanup warning: {e}")
        self._chart_widgets.clear()

        # Reset initialization flags so charts get recreated on next show
        self._summary_charts_initialized = False
        self._detail_charts_initialized = False

        # Null all chart references so re-initialization guards work correctly
        self.linearity_chart = None
        self.scatter_chart = None
        self.dist_chart = None
        self.alerts_chart = None
        self.trending_chart = None
        self.best_chart = None
        self.recent_issues_chart = None
        self.heatmap_chart = None
        # Clear any stale data
        self.active_models_data = []
        self.model_trend_data = None

    def _create_summary_view(self):
        """Create the summary view (All Models mode).

        Five sections, top to bottom:
        1. Active Models Summary stats row
        2. Focus This Week (5 native widget rows)
        3. Failure Severity bar chart
        4. Cost Impact horizontal bars
        5. ML Status one-liner
        """
        # Clean up existing charts first (frees matplotlib figures)
        self._cleanup_charts()
        self._summary_charts_initialized = False

        # Clear existing content
        for widget in self.content.winfo_children():
            widget.destroy()

        # Row weights: stats compact, focus compact, severity grows,
        # cost grows tallest (bar chart per model), ML compact.
        self.content.grid_rowconfigure(0, weight=0)  # Stats
        self.content.grid_rowconfigure(1, weight=0, minsize=200)  # Focus
        self.content.grid_rowconfigure(2, weight=1, minsize=240)  # Severity
        self.content.grid_rowconfigure(3, weight=1, minsize=320)  # Cost
        self.content.grid_rowconfigure(4, weight=0)  # ML

        # ---- Section 1: Active Models Summary stats ----
        stats_frame = ctk.CTkFrame(self.content)
        stats_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        stats_label = ctk.CTkLabel(
            stats_frame,
            text="Active Models Summary",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        stats_label.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="w", columnspan=9)

        self.summary_stat_labels = {}
        stat_names = [
            ("active_models", "Active Models"),
            ("total_samples", "Total Samples"),
            ("avg_linearity_rate", "Linearity Pass"),
            ("avg_sigma_rate", "Sigma Pass"),
            ("avg_pass_rate", "Combined Pass"),
            ("models_at_risk", "Needs Attention"),
            ("best_model", "Best (Linearity)"),
            ("worst_model", "Worst (Linearity)"),
            ("top_anomaly", "Top Anomaly Model"),
        ]
        for idx, (key, label) in enumerate(stat_names):
            stat_col = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_col.grid(row=1, column=idx, padx=15, pady=(0, 15), sticky="w")
            ctk.CTkLabel(stat_col, text=label, text_color="gray",
                         font=ctk.CTkFont(size=11)).pack(anchor="w")
            value_label = ctk.CTkLabel(stat_col, text="--",
                                        font=ctk.CTkFont(size=14, weight="bold"))
            value_label.pack(anchor="w")
            self.summary_stat_labels[key] = value_label

        # ---- Section 2: Focus This Week ----
        self._create_focus_section()

        # ---- Section 3: Failure Severity ----
        self._create_failure_severity_chart()

        # ---- Section 4: Cost Impact ----
        self._create_cost_impact_chart()

        # ---- Section 5: ML Status ----
        ml_frame = ctk.CTkFrame(self.content)
        ml_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(5, 10))

        ml_header = ctk.CTkFrame(ml_frame, fg_color="transparent")
        ml_header.pack(fill="x", padx=15, pady=(15, 5))
        ctk.CTkLabel(
            ml_header, text="ML Insights",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left")
        self._ml_view_all_btn = ctk.CTkButton(
            ml_header, text="View All Details",
            command=self._show_ml_details_dialog,
            width=100, height=24, font=ctk.CTkFont(size=11)
        )
        self._ml_view_all_btn.pack(side="right", padx=5)

        self.ml_text = ctk.CTkTextbox(ml_frame, height=80)
        self.ml_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.ml_text.configure(state="disabled")
        self._cached_alert_models = None
        self._cached_ml_insights = None
        self._update_ml_summary(None)

        # New 5-section layout creates severity_chart and cost_chart eagerly
        # inside _create_failure_severity_chart / _create_cost_impact_chart,
        # so there is no lazy phase. Mark initialized=True so on_show stops
        # tearing down and rebuilding the whole summary on every page visit.
        self._summary_charts_initialized = True

    def _create_focus_section(self):
        """Native-widget Focus This Week section.

        Replaces the previous matplotlib `ax.text` blob. Five labeled rows
        rendered as CTkFrames inside a parent CTkFrame, each row carrying
        a primary line (rank · model · fail rate) and a secondary line
        (recommendation, or "—" when none). Color-coded per row by fail
        rate; row heights stay uniform across all 5 entries so the visual
        rhythm is stable regardless of recommendation length.
        """
        focus_frame = ctk.CTkFrame(self.content)
        focus_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        ctk.CTkLabel(
            focus_frame, text="Focus This Week",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 5))

        self._focus_rows_frame = ctk.CTkFrame(focus_frame, fg_color="transparent")
        self._focus_rows_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        # Initial placeholder; replaced by _update_focus_section.
        self._focus_placeholder = ctk.CTkLabel(
            self._focus_rows_frame,
            text="Loading priority models…",
            text_color="gray",
        )
        self._focus_placeholder.pack(pady=10)

    def _update_focus_section(self, priorities):
        """Render the top 5 priority models as native widget rows."""
        if not hasattr(self, "_focus_rows_frame"):
            return
        # Wipe previous rows (including the loading placeholder)
        for w in self._focus_rows_frame.winfo_children():
            w.destroy()

        if not priorities:
            ctk.CTkLabel(
                self._focus_rows_frame,
                text="No priority models in this window",
                text_color="gray",
            ).pack(pady=10)
            return

        for i, p in enumerate(priorities[:5], start=1):
            row = ctk.CTkFrame(self._focus_rows_frame)
            row.pack(fill="x", pady=2)

            fail_rate = 100.0 - float(p.get("linearity_pass_rate", 100))
            if fail_rate >= 30:
                rate_color = "#dc3545"
            elif fail_rate >= 15:
                rate_color = "#fd7e14"
            elif fail_rate >= 5:
                rate_color = "#f1c40f"
            else:
                rate_color = "white"

            top_line = ctk.CTkFrame(row, fg_color="transparent")
            top_line.pack(fill="x", padx=10, pady=(6, 0))
            ctk.CTkLabel(
                top_line, text=f"#{i}",
                font=ctk.CTkFont(size=12, weight="bold"),
                width=30,
            ).pack(side="left")
            ctk.CTkLabel(
                top_line, text=p.get("model", "?"),
                font=ctk.CTkFont(size=12, weight="bold"),
                width=120, anchor="w",
            ).pack(side="left", padx=(5, 10))
            ctk.CTkLabel(
                top_line, text=f"{fail_rate:.1f}% fail",
                font=ctk.CTkFont(size=11),
                text_color=rate_color,
                width=80, anchor="w",
            ).pack(side="left")
            ctk.CTkLabel(
                top_line,
                text=(
                    f"{p.get('failed_units', 0)} fails / "
                    f"{p.get('total_tracks', 0)} tracks · "
                    f"{p.get('near_miss_count', 0)} near-miss"
                ),
                font=ctk.CTkFont(size=11),
                text_color="gray",
                anchor="w",
            ).pack(side="left", padx=(15, 0))

            rec = p.get("recommendation") or "—"
            rec_line = ctk.CTkLabel(
                row,
                text=f"   → {rec}",
                font=ctk.CTkFont(size=11),
                text_color="#cccccc" if rec != "—" else "gray",
                anchor="w",
                justify="left",
            )
            rec_line.pack(fill="x", padx=10, pady=(0, 6))

    def _create_failure_severity_chart(self):
        """Bar chart of failing tracks bucketed by fail-point count."""
        ChartWidget, ChartStyle = _ensure_chart_module()
        sev_frame = ctk.CTkFrame(self.content)
        sev_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        # Section header is static; the active date window is rendered in
        # the matplotlib axis title inside _update_failure_severity_chart so
        # it stays in sync with the user's date-filter selection.
        ctk.CTkLabel(
            sev_frame, text="Failure Severity",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 5))
        self._severity_chart_frame = sev_frame
        self.severity_chart = ChartWidget(
            sev_frame, style=ChartStyle(figure_size=(10, 2.6), dpi=100)
        )
        self.severity_chart.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.severity_chart.show_placeholder("Loading failure severity…")
        self._chart_widgets.append(self.severity_chart)

    def _update_failure_severity_chart(self, near_miss):
        """Render Failure Severity buckets onto severity_chart."""
        if not getattr(self, "severity_chart", None):
            return
        chart = self.severity_chart
        chart.clear()
        fig = chart.figure
        ax = fig.add_subplot(111)
        chart._style_axis(ax)

        total_failing = (near_miss or {}).get("total_failing", 0)
        if total_failing == 0:
            self._draw_empty_state(ax, "No failing tracks in this window")
        else:
            ax.set_title(
                f"last {self.selected_days}d  ·  {total_failing} failing tracks",
                loc="left", fontsize=10, color="#aaaaaa",
            )
            buckets = near_miss["distribution"]
            labels = [
                "1-3 pts\n(near-miss)", "4-10 pts",
                "11-50 pts", "50+ pts\n(hard-fail)",
            ]
            values = [
                buckets.get("1-3 points", 0), buckets.get("4-10 points", 0),
                buckets.get("11-50 points", 0), buckets.get("50+ points", 0),
            ]
            colors = ["#198754", "#fd7e14", "#dc3545", "#6f42c1"]
            bars = ax.bar(labels, values, color=colors,
                          edgecolor="#1a1a1a", linewidth=0.5)
            for bar, v in zip(bars, values):
                pct = (v / total_failing) * 100 if total_failing else 0
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v}\n({pct:.0f}%)", ha="center", va="bottom",
                        fontsize=9, color="#dddddd")
            ax.set_ylabel("Failing tracks")
            ax.set_ylim(0, max(values) * 1.25 if max(values) else 1)
            ax.grid(True, axis="y", alpha=0.2)
        try:
            fig.tight_layout()
        except Exception:
            pass
        chart.canvas.draw_idle()

    def _create_cost_impact_chart(self):
        ChartWidget, ChartStyle = _ensure_chart_module()
        cost_frame = ctk.CTkFrame(self.content)
        cost_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        ctk.CTkLabel(
            cost_frame, text="Cost Impact",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 5))
        self._cost_chart_frame = cost_frame
        self.cost_chart = ChartWidget(
            cost_frame, style=ChartStyle(figure_size=(10, 4.0), dpi=100)
        )
        self.cost_chart.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.cost_chart.show_placeholder("Loading cost impact…")
        self._chart_widgets.append(self.cost_chart)

    def _update_cost_impact_chart(self, priorities, pricing, cost_ratio):
        """Render top-15 horizontal bars of estimated scrap cost."""
        if not getattr(self, "cost_chart", None):
            return
        chart = self.cost_chart
        chart.clear()
        fig = chart.figure
        ax = fig.add_subplot(111)
        chart._style_axis(ax)

        with_price = [
            (p, pricing.get(p["model"]))
            for p in (priorities or [])
            if pricing.get(p["model"])
        ]
        if not with_price:
            self._draw_empty_state(
                ax,
                "No model pricing configured.\n"
                "Add prices in Settings → Active Models to see\n"
                "estimated scrap cost per model.",
            )
            chart.canvas.draw_idle()
            return

        cost_rows = [
            {
                "model": p["model"],
                "failed": p["failed_units"],
                "cost": p["failed_units"] * price * cost_ratio,
                "near_miss": p.get("near_miss_count", 0),
            }
            for p, price in with_price
            if p["failed_units"] > 0
        ]
        if not cost_rows:
            # Pricing is configured but no failures in this window — show
            # an explicit message instead of a blank chart with the
            # nonsensical title "Top 0 models ($0 total)".
            self._draw_empty_state(
                ax,
                "No failing units with configured pricing in this window.",
            )
            chart.canvas.draw_idle()
            return
        cost_rows.sort(key=lambda r: r["cost"], reverse=True)
        cost_rows = list(reversed(cost_rows[:15]))
        models = [r["model"] for r in cost_rows]
        costs = [r["cost"] for r in cost_rows]

        def _color(r):
            if r["failed"] == 0:
                return "#888888"
            ratio = r["near_miss"] / r["failed"]
            if ratio >= 0.5:
                return "#198754"
            if ratio >= 0.25:
                return "#fd7e14"
            return "#dc3545"

        colors = [_color(r) for r in cost_rows]
        y_pos = list(range(len(models)))
        ax.barh(y_pos, costs, color=colors, edgecolor="#1a1a1a", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=9)
        for i, r in enumerate(cost_rows):
            nm_pct = (r["near_miss"] / r["failed"]) * 100 if r["failed"] else 0
            ax.text(r["cost"] * 1.01, i,
                    f"${r['cost']:,.0f} · {r['failed']} fails · "
                    f"{nm_pct:.0f}% near-miss",
                    va="center", fontsize=8, color="#cccccc")

        ax.set_xlabel(
            f"Est. scrap cost ($, last {self.selected_days}d, "
            f"cost_ratio={cost_ratio:.2f})"
        )
        total_cost = sum(costs)
        ax.set_title(
            f"Top {len(cost_rows)} models  (${total_cost:,.0f} total)",
            loc="left", fontsize=11, fontweight="bold", color="#ffffff",
        )
        ax.grid(True, axis="x", alpha=0.2)
        ax.set_xlim(0, max(costs) * 1.6 if costs else 1)
        try:
            fig.tight_layout()
        except Exception:
            pass
        chart.canvas.draw_idle()

    def _ensure_summary_charts_initialized(self):
        """Lazily initialize summary view charts - defers matplotlib loading."""
        if self._summary_charts_initialized:
            return

        # Bail out if the summary frames have been destroyed (e.g. user
        # navigated to detail mode while a stale callback was queued).
        # Creating a ChartWidget on a destroyed parent raises
        # _tkinter.TclError "bad window path name".
        if (
            getattr(self, "_alerts_frame", None) is None
            or not self._alerts_frame.winfo_exists()
        ):
            logger.debug("Summary frames missing; skipping chart init")
            return

        ChartWidget, ChartStyle = _ensure_chart_module()

        # Create alerts chart
        if self._alerts_placeholder and self._alerts_placeholder.winfo_exists():
            self._alerts_placeholder.destroy()
        self.alerts_chart = ChartWidget(
            self._alerts_frame,
            style=ChartStyle(figure_size=(10, 3), dpi=100)
        )
        self._chart_widgets.append(self.alerts_chart)
        self.alerts_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.alerts_chart.show_placeholder("Loading models requiring attention...")

        # Create best chart
        if self._best_placeholder and self._best_placeholder.winfo_exists():
            self._best_placeholder.destroy()
        self.best_chart = ChartWidget(
            self._best_frame,
            style=ChartStyle(figure_size=(5, 3), dpi=100)
        )
        self._chart_widgets.append(self.best_chart)
        self.best_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.best_chart.show_placeholder("Loading best models...")

        # Create recent issues chart (replaces worst chart)
        if self._recent_issues_placeholder and self._recent_issues_placeholder.winfo_exists():
            self._recent_issues_placeholder.destroy()
        self.recent_issues_chart = ChartWidget(
            self._recent_issues_frame,
            style=ChartStyle(figure_size=(5, 3), dpi=100)
        )
        self._chart_widgets.append(self.recent_issues_chart)
        self.recent_issues_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.recent_issues_chart.show_placeholder("Loading recent issues...")

        # Create trending worse chart
        if self._trending_placeholder and self._trending_placeholder.winfo_exists():
            self._trending_placeholder.destroy()
        self.trending_chart = ChartWidget(
            self._trending_frame,
            style=ChartStyle(figure_size=(5, 3), dpi=100)
        )
        self._chart_widgets.append(self.trending_chart)
        self.trending_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.trending_chart.show_placeholder("Loading trending data...")

        # Create heatmap chart
        if self._heatmap_placeholder and self._heatmap_placeholder.winfo_exists():
            self._heatmap_placeholder.destroy()
        self.heatmap_chart = ChartWidget(
            self._heatmap_frame,
            style=ChartStyle(figure_size=(10, 4), dpi=100)
        )
        self._chart_widgets.append(self.heatmap_chart)
        self.heatmap_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.heatmap_chart.show_placeholder("Loading heat map data...")

        self._summary_charts_initialized = True
        logger.debug("Summary charts initialized (matplotlib loaded)")

    def _create_detail_view(self):
        """Create the detail view (specific model mode)."""
        # Clean up existing charts first (frees matplotlib figures)
        self._cleanup_charts()
        self._detail_charts_initialized = False

        # Reset timeline filter to show all data when switching models
        self.chart_timeline_days = 0

        # Clear existing content
        for widget in self.content.winfo_children():
            widget.destroy()

        self.content.grid_rowconfigure(0, weight=0)  # Stats row - compact
        self.content.grid_rowconfigure(1, weight=1, minsize=250)  # Sigma scatter
        self.content.grid_rowconfigure(2, weight=1, minsize=250)  # Linearity scatter - NEW
        self.content.grid_rowconfigure(3, weight=1, minsize=200)  # Distribution
        self.content.grid_rowconfigure(4, weight=0)  # Alerts/ML - compact

        # Model stats at top
        stats_frame = ctk.CTkFrame(self.content)
        stats_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        stats_label = ctk.CTkLabel(
            stats_frame,
            text=f"Model: {self.selected_model}",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        stats_label.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="w", columnspan=6)

        # Stats in a horizontal row
        self.detail_stat_labels = {}
        stat_names = [
            ("total_samples", "Total Samples"),
            ("anomalies", "Anomalies"),
            ("linearity_pass_rate", "Linearity Pass"),
            ("sigma_pass_rate", "Sigma Pass"),
            ("overall_pass_rate", "Overall Pass"),
            ("sigma_cpk", "Sigma Cpk"),
            ("near_miss", "Near-Miss"),
            ("avg_trim_improvement", "Avg Trim Imp."),
            ("trend", "Trend"),
        ]

        for idx, (key, label) in enumerate(stat_names):
            stat_col = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_col.grid(row=1, column=idx, padx=15, pady=(0, 15), sticky="w")

            ctk.CTkLabel(stat_col, text=label, text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w")
            value_label = ctk.CTkLabel(stat_col, text="--", font=ctk.CTkFont(size=14, weight="bold"))
            value_label.pack(anchor="w")
            self.detail_stat_labels[key] = value_label

        # Main SPC scatter chart - placeholder
        self._scatter_frame = ctk.CTkFrame(self.content)
        self._scatter_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Header row with title and timeline filter
        scatter_header = ctk.CTkFrame(self._scatter_frame, fg_color="transparent")
        scatter_header.pack(fill="x", padx=15, pady=(15, 5))

        chart_label = ctk.CTkLabel(
            scatter_header,
            text=f"Sigma Gradient Trend ({self.rolling_window}-Day Rolling Average)",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        chart_label.pack(side="left", anchor="w")

        # Timeline filter for zooming into specific date ranges
        timeline_frame = ctk.CTkFrame(scatter_header, fg_color="transparent")
        timeline_frame.pack(side="right", anchor="e")

        timeline_label = ctk.CTkLabel(timeline_frame, text="Chart Range:", font=ctk.CTkFont(size=11))
        timeline_label.pack(side="left", padx=(0, 5))

        self.timeline_dropdown = ctk.CTkOptionMenu(
            timeline_frame,
            values=["All Data", "Recent 7 Days", "Recent 14 Days", "Recent 30 Days", "Recent 60 Days"],
            command=self._on_timeline_change,
            width=130,
        )
        self.timeline_dropdown.set("All Data")
        self.timeline_dropdown.pack(side="left")

        export_pdf_btn = ctk.CTkButton(
            timeline_frame,
            text="Export PDF",
            command=self._export_detail_pdf,
            width=100,
            height=28
        )
        export_pdf_btn.pack(side="right", padx=(10, 0))

        self._scatter_placeholder = ctk.CTkLabel(
            self._scatter_frame,
            text="Loading trend data...",
            text_color="gray"
        )
        self._scatter_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.scatter_chart = None

        # Linearity scatter chart - placeholder
        self._linearity_frame = ctk.CTkFrame(self.content)
        self._linearity_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Header row
        linearity_header = ctk.CTkFrame(self._linearity_frame, fg_color="transparent")
        linearity_header.pack(fill="x", padx=15, pady=(15, 5))

        linearity_label = ctk.CTkLabel(
            linearity_header,
            text="Linearity Pass Rate Trend",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        linearity_label.pack(side="left", anchor="w")

        self._linearity_placeholder = ctk.CTkLabel(
            self._linearity_frame,
            text="Loading linearity trend data...",
            text_color="gray"
        )
        self._linearity_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.linearity_chart = None

        # Distribution chart - placeholder
        self._dist_frame = ctk.CTkFrame(self.content)
        self._dist_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)

        dist_label = ctk.CTkLabel(
            self._dist_frame,
            text="Sigma Distribution",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        dist_label.pack(padx=15, pady=(15, 5), anchor="w")

        self._dist_placeholder = ctk.CTkLabel(
            self._dist_frame,
            text="Loading distribution...",
            text_color="gray"
        )
        self._dist_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.dist_chart = None

        # Bottom row: Alerts and ML side by side
        bottom_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        bottom_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(5, 10))
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)

        # Alerts section
        alerts_frame = ctk.CTkFrame(bottom_frame)
        alerts_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        alerts_label = ctk.CTkLabel(
            alerts_frame,
            text="Model Alerts",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        alerts_label.pack(padx=15, pady=(15, 10), anchor="w")

        self.detail_alerts_text = ctk.CTkTextbox(alerts_frame, height=100)
        self.detail_alerts_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.detail_alerts_text.configure(state="disabled")

        # ML recommendations section
        ml_frame = ctk.CTkFrame(bottom_frame)
        ml_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        ml_label = ctk.CTkLabel(
            ml_frame,
            text="ML Recommendations",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        ml_label.pack(padx=15, pady=(15, 10), anchor="w")

        self.detail_ml_text = ctk.CTkTextbox(ml_frame, height=100)
        self.detail_ml_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.detail_ml_text.configure(state="disabled")

    def _ensure_detail_charts_initialized(self):
        """Lazily initialize detail view charts - defers matplotlib loading."""
        if self._detail_charts_initialized:
            return

        ChartWidget, ChartStyle = _ensure_chart_module()

        # Create scatter chart
        if self._scatter_placeholder and self._scatter_placeholder.winfo_exists():
            self._scatter_placeholder.destroy()
        self.scatter_chart = ChartWidget(
            self._scatter_frame,
            style=ChartStyle(figure_size=(10, 4), dpi=100)
        )
        self._chart_widgets.append(self.scatter_chart)
        self.scatter_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.scatter_chart.show_placeholder("Loading trend data...")

        # Create linearity chart
        if self._linearity_placeholder and self._linearity_placeholder.winfo_exists():
            self._linearity_placeholder.destroy()
        if not self.linearity_chart:
            self.linearity_chart = ChartWidget(
                self._linearity_frame,
                style=ChartStyle(figure_size=(10, 4), dpi=100)
            )
            self._chart_widgets.append(self.linearity_chart)
            self.linearity_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
            self.linearity_chart.show_placeholder("Loading linearity trend data...")

        # Create distribution chart
        if self._dist_placeholder and self._dist_placeholder.winfo_exists():
            self._dist_placeholder.destroy()
        self.dist_chart = ChartWidget(
            self._dist_frame,
            style=ChartStyle(figure_size=(10, 2.5), dpi=100)
        )
        self._chart_widgets.append(self.dist_chart)
        self.dist_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.dist_chart.show_placeholder("Loading distribution...")

        self._detail_charts_initialized = True
        logger.debug("Detail charts initialized (matplotlib loaded)")

    def _bind_mousewheel_scroll(self, combobox):
        """Bind mousewheel events to CTkComboBox dropdown."""
        def on_mousewheel_closed(event):
            values = combobox.cget("values")
            if not values:
                return "break"

            current = combobox.get()
            try:
                current_idx = list(values).index(current)
            except ValueError:
                current_idx = 0

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

        combobox.bind("<MouseWheel>", on_mousewheel_closed)
        combobox.bind("<Button-4>", on_mousewheel_closed)
        combobox.bind("<Button-5>", on_mousewheel_closed)

        if hasattr(combobox, '_open_dropdown_menu'):
            original_open = combobox._open_dropdown_menu

            def patched_open(*args, **kwargs):
                result = original_open(*args, **kwargs)
                combobox.after(10, lambda: self._enable_dropdown_scroll(combobox))
                return result

            combobox._open_dropdown_menu = patched_open

    def _enable_dropdown_scroll(self, combobox):
        """Enable mousewheel scrolling on the open dropdown."""
        try:
            if hasattr(combobox, '_dropdown_menu') and combobox._dropdown_menu:
                dropdown = combobox._dropdown_menu
                canvas = None

                if hasattr(dropdown, '_scrollable_frame'):
                    sf = dropdown._scrollable_frame
                    if hasattr(sf, '_parent_canvas'):
                        canvas = sf._parent_canvas
                    elif hasattr(sf, '_canvas'):
                        canvas = sf._canvas
                elif hasattr(dropdown, '_canvas'):
                    canvas = dropdown._canvas

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
                        try:
                            delta = -1 * (event.delta // 120) if event.delta else (-1 if event.num == 4 else 1)
                            canvas.yview_scroll(delta, "units")
                        except Exception:
                            pass
                        return "break"

                    dropdown.bind("<MouseWheel>", scroll_dropdown)
                    dropdown.bind("<Button-4>", scroll_dropdown)
                    dropdown.bind("<Button-5>", scroll_dropdown)

                    def on_destroy(event):
                        try:
                            dropdown.unbind("<MouseWheel>")
                            dropdown.unbind("<Button-4>")
                            dropdown.unbind("<Button-5>")
                        except Exception:
                            pass

                    dropdown.bind("<Destroy>", on_destroy)
        except Exception as e:
            logger.debug(f"Could not enable dropdown scroll: {e}")

    def _on_model_change(self, model: str):
        """Handle model selection change."""
        self.selected_model = model
        logger.debug(f"Model changed to: {model}")

        # Show/hide rolling avg controls (only useful in detail mode)
        if model == "All Models":
            self.rolling_label.pack_forget()
            self.rolling_dropdown.pack_forget()
            self._create_summary_view()
        else:
            self.rolling_label.pack(side="left", padx=(10, 2), pady=15)
            self.rolling_dropdown.pack(side="left", padx=2, pady=15)
            self._create_detail_view()

        self._refresh_data()

    def _on_active_filter_change(self):
        """Handle Active Only checkbox change - refresh to update model list."""
        self._refresh_data()

    def _filter_model_list(
        self,
        prioritized_models: List[Dict],
        priority_data: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Build filtered model list from prioritized models.

        Applies: active_only, min_samples, min_fail_rate, element_type,
        and product_class filters.
        MPS models are always included regardless of filters.
        Uses cached filter values captured on main thread (tkinter is not thread-safe).
        """
        active_only = getattr(self, '_cached_active_only', True)
        min_samples = getattr(self, '_cached_min_samples', 0)
        min_fail_rate = getattr(self, '_cached_min_fail_rate', 0)

        # Element type / product class filter from cached values
        et = getattr(self, '_cached_element_filter', "All")
        pc = getattr(self, '_cached_class_filter', "All")
        spec_filter_models = None
        if (et and et != "All") or (pc and pc != "All"):
            try:
                db = get_database()
                all_specs = db.get_all_model_specs()
                spec_filter_models = set()
                for s in all_specs:
                    if et and et != "All" and s.get("element_type") != et:
                        continue
                    if pc and pc != "All" and s.get("product_class") != pc:
                        continue
                    spec_filter_models.add(s["model"])
            except Exception:
                spec_filter_models = None

        mps_set = set(self.app.config.active_models.mps_models)

        # Build fail rate lookup from priority data if available
        fail_rates = {}
        if priority_data:
            for p in priority_data:
                rate = 100 - p.get("linearity_pass_rate", 100)
                fail_rates[p.get("model", "")] = rate

        model_names = ["All Models"]
        for m in prioritized_models:
            model = m['model']
            status = m['status']
            count = m.get('count', 0)
            is_mps = model in mps_set

            # Active-only filter
            if status == 'inactive' and active_only:
                continue

            # Element type / product class filter
            if spec_filter_models is not None and model not in spec_filter_models:
                continue

            # Min samples filter (MPS models bypass)
            if not is_mps and min_samples > 0 and count < min_samples:
                continue

            # Min fail rate filter (MPS models bypass)
            if not is_mps and min_fail_rate > 0 and model in fail_rates:
                if fail_rates[model] < min_fail_rate:
                    continue

            # Suffix for inactive models
            if status == 'inactive':
                model_names.append(f"{model} (inactive)")
            else:
                model_names.append(model)

        return model_names

    def _on_date_change(self, date_range: str):
        """Handle date range change."""
        days_map = {
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "Last Year": 365,
            "All Time": 3650,
        }
        self.selected_days = days_map.get(date_range, 3650)
        logger.debug(f"Date range changed to: {date_range} ({self.selected_days} days)")
        # The Drift tab is fed by its own queries, not _refresh_data's
        # summary/detail loaders, so route the refresh based on which
        # top-level tab is active.
        if self._trend_type.get() == "Drift":
            self._refresh_drift_view()
        else:
            self._refresh_data()

    def _on_rolling_change(self, rolling: str):
        """Handle rolling average window change."""
        rolling_map = {
            "7 Days": 7,
            "14 Days": 14,
            "30 Days": 30,
            "60 Days": 60,
        }
        self.rolling_window = rolling_map.get(rolling, 30)
        logger.debug(f"Rolling window changed to: {rolling} ({self.rolling_window} days)")
        self._refresh_data()

    def _on_timeline_change(self, timeline: str):
        """Handle chart timeline filter change (zooms into the scatter chart)."""
        timeline_map = {
            "All Data": 0,
            "Recent 7 Days": 7,
            "Recent 14 Days": 14,
            "Recent 30 Days": 30,
            "Recent 60 Days": 60,
        }
        self.chart_timeline_days = timeline_map.get(timeline, 0)
        logger.debug(f"Chart timeline changed to: {timeline} ({self.chart_timeline_days} days)")
        # Re-render charts with filtered data (no need to reload from DB)
        if self.model_trend_data:
            self._update_scatter_chart_with_filter()
            self._update_linearity_chart_with_filter()

    def _update_scatter_chart_with_filter(self):
        """Update the scatter chart with current timeline filter applied."""
        if not self.model_trend_data or not self.model_trend_data.get("data_points"):
            return

        if not self.scatter_chart:
            return

        data_points = self.model_trend_data["data_points"]
        threshold = self.model_trend_data.get("threshold")

        # Apply timeline filter - use RELATIVE to data, not absolute calendar dates
        # This way "Last 7 Days" means "most recent 7 days of data that exists"
        if self.chart_timeline_days > 0 and data_points:
            # Find the most recent date in the data
            def get_date(d):
                try:
                    return d["date"] if isinstance(d["date"], datetime) else datetime.strptime(str(d["date"])[:10], "%Y-%m-%d")
                except (ValueError, TypeError):
                    return None

            valid_dates = [dt for dt in (get_date(d) for d in data_points) if dt is not None]
            if not valid_dates:
                return
            most_recent = max(valid_dates)
            cutoff_date = most_recent - timedelta(days=self.chart_timeline_days)
            filtered_points = [d for d in data_points if (get_date(d) or datetime.min) >= cutoff_date]
        else:
            filtered_points = data_points

        # Exclude anomalies from chart - they are shown as a count in the stats
        # Anomalies (trim failures with linear slope) would skew the visual trend
        normal_points = [d for d in filtered_points if not d.get("is_anomaly", False)]

        if not normal_points:
            anomaly_count = len([d for d in filtered_points if d.get("is_anomaly", False)])
            if anomaly_count > 0:
                self.scatter_chart.show_placeholder(f"All {anomaly_count} samples are anomalies\n(excluded from chart)")
            else:
                self.scatter_chart.show_placeholder(f"No data in selected range (last {self.chart_timeline_days} days)")
            return

        # Extract values for plotting (normal samples only) - include year in date format
        # Build aligned arrays - only include points with valid sigma
        valid_points = [d for d in normal_points if d["sigma_gradient"] is not None]
        dates = [d["date"].strftime("%m/%d/%y") if hasattr(d["date"], 'strftime') else str(d["date"])[:8] for d in valid_points]
        sigma_values = [d["sigma_gradient"] for d in valid_points]
        pass_flags = [d.get("sigma_pass", False) for d in valid_points]

        # Calculate rolling average for filtered data (normal samples only)
        rolling_vals = None
        window = min(self.rolling_window, len(sigma_values))
        if window > 1 and sigma_values:
            rolling_vals = []
            for i in range(len(sigma_values)):
                start = max(0, i - window + 1)
                window_vals = sigma_values[start:i+1]
                rolling_vals.append(np.mean(window_vals))

        # Determine title suffix based on filter
        filter_suffix = f" (Last {self.chart_timeline_days} Days)" if self.chart_timeline_days > 0 else ""

        # Plot without anomaly_flags since we've already filtered them out
        self.scatter_chart.plot_sigma_scatter(
            dates=dates,
            sigma_values=sigma_values,
            pass_flags=pass_flags,
            threshold=threshold,
            rolling_avg=rolling_vals,
            title=f"Sigma Gradient Trend - {self.selected_model}{filter_suffix}",
            ylabel="Sigma Gradient",
        )

    def _update_linearity_chart_with_filter(self):
        """Update linearity chart with aggregated pass rate by day."""
        if not self.model_trend_data or not self.linearity_chart:
            return

        # Get linearity pass rates by day from database
        linearity_pass_rates = self.model_trend_data.get("linearity_pass_rates_by_day", [])

        if not linearity_pass_rates:
            self.linearity_chart.show_placeholder("No linearity data available")
            return

        # Apply timeline filter - use RELATIVE to data, not absolute calendar dates
        if self.chart_timeline_days > 0 and linearity_pass_rates:
            # Find the most recent date in the data
            def _parse_date(date_str):
                """Parse date string, handling both date-only and datetime formats."""
                try:
                    return datetime.strptime(date_str[:10], "%Y-%m-%d")
                except (ValueError, TypeError):
                    return None

            most_recent_str = max(pr["date"] for pr in linearity_pass_rates)
            most_recent = _parse_date(most_recent_str)
            if most_recent:
                cutoff_date = most_recent - timedelta(days=self.chart_timeline_days)
                linearity_pass_rates = [
                    pr for pr in linearity_pass_rates
                    if (_parse_date(pr["date"]) or datetime.min) >= cutoff_date
                ]

        if len(linearity_pass_rates) < 2:
            self.linearity_chart.show_placeholder("Insufficient linearity data for selected range")
            return

        # Extract dates and pass rates
        dates = [pr["date"] for pr in linearity_pass_rates]
        pass_rates = [pr["pass_rate"] for pr in linearity_pass_rates]
        totals = [pr["total"] for pr in linearity_pass_rates]

        # Format dates for display
        display_dates = [datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%y") for d in dates]

        # Calculate rolling average of pass rates
        rolling_avg = None
        window_size = min(self.rolling_window, len(pass_rates))
        if window_size > 1 and pass_rates:
            rolling_avg = []
            for i in range(len(pass_rates)):
                start_idx = max(0, i - window_size + 1)
                # Weighted average by totals
                window_rates = pass_rates[start_idx:i + 1]
                window_totals = totals[start_idx:i + 1]
                total_weight = sum(window_totals)
                if total_weight > 0:
                    weighted_avg = sum(r * t for r, t in zip(window_rates, window_totals)) / total_weight
                else:
                    weighted_avg = 0.0
                rolling_avg.append(weighted_avg)

        # Determine title suffix based on filter
        filter_suffix = f" (Last {self.chart_timeline_days} Days)" if self.chart_timeline_days > 0 else ""

        # Plot as P-chart with binomial control limits
        self.linearity_chart.plot_pchart(
            dates=display_dates,
            pass_rates=pass_rates,
            sample_sizes=totals,
            title=f"Linearity P-Chart - {self.selected_model}{filter_suffix}",
            ylabel="Pass Rate (%)",
        )

        # Overlay unit-yield line on the P-chart so the operator can see
        # the divergence — track-level pass rate (sensitive to retrims)
        # vs unit-level yield (sensitive to whether anything actually
        # shipped). Gated by the enable_unit_yield_view feature flag so
        # the new view can be hidden without removing the data.
        try:
            cfg = get_config()
            show_unit_yield = bool(getattr(
                cfg.active_models, "enable_unit_yield_view", True
            ))
        except Exception:
            show_unit_yield = True

        if show_unit_yield:
            model_for_query = self.selected_model.replace(" (inactive)", "")
            try:
                db = get_database()
                unit_rows = db.get_unit_yield_trend(
                    model=model_for_query, days_back=self.selected_days
                )
            except Exception as e:
                logger.debug(f"unit yield trend unavailable: {e}")
                unit_rows = []
            if unit_rows:
                # Align unit-yield to the same date axis the P-chart uses.
                yield_by_date = {r["date"]: r["yield_pct"] for r in unit_rows}
                unit_yield_series = [yield_by_date.get(d) for d in dates]
                self.linearity_chart.overlay_line(
                    series=unit_yield_series,
                    label="Unit Yield (sections all-pass)",
                    color="#2c7fb8",
                    linestyle="--",
                )

    def _refresh_data(self):
        """Refresh data from database."""
        self.status_label.configure(text="Loading...")
        # Bump generation so any in-flight summary/detail load discards
        # its result when the after() callback fires (the user has
        # already moved on, possibly destroying the target frames).
        self._load_generation += 1
        gen = self._load_generation
        # Capture UI filter values on main thread (tkinter is not thread-safe)
        self._cached_active_only = self.active_only_var.get()
        try:
            self._cached_min_samples = int(self.min_samples_entry.get())
        except (ValueError, AttributeError):
            self._cached_min_samples = 0
        try:
            self._cached_min_fail_rate = float(self.fail_rate_entry.get())
        except (ValueError, AttributeError):
            self._cached_min_fail_rate = 0
        self._cached_element_filter = self._element_filter.get() if hasattr(self, '_element_filter') else "All"
        self._cached_class_filter = self._class_filter.get() if hasattr(self, '_class_filter') else "All"
        get_thread_manager().start_thread(
            target=lambda g=gen: self._load_data(g), name="trends-load-data"
        )

    def _load_data(self, gen: int):
        """Load data in background thread; gen lets stale callbacks bail."""
        try:
            db = get_database()

            if self.selected_model == "All Models":
                # Summary mode
                self._load_summary_data(db, gen)
            else:
                # Detail mode
                self._load_detail_data(db, gen)

        except Exception as e:
            logger.error(f"Failed to load trend data: {e}", exc_info=True)
            self.after(0, lambda: self._show_error(str(e)))

    def _load_summary_data(self, db, gen: int = 0):
        """Load data for summary mode."""
        # Get active models config for MPS prioritization
        config = get_config()
        mps_models = config.active_models.mps_models
        recent_days = config.active_models.recent_days

        # Get active models summary
        active_models = db.get_active_models_summary(
            days_back=self.selected_days,
            min_samples=5
        )

        # Get models requiring attention (filter to 20+ samples, linearity-focused)
        alert_models = db.get_models_requiring_attention(
            days_back=self.selected_days,
            min_samples=20,  # Increased to filter out low-data models
            pass_rate_threshold=80.0,
            trend_threshold=10.0,
            rolling_days=self.rolling_window,
            metric="linearity"
        )

        # Get trending worse models
        trending_worse = db.get_trending_worse_models(
            days_back=self.selected_days,
            min_samples=20,
            trend_threshold=10.0,
            rolling_days=self.rolling_window
        )

        # Get impact-ranked prioritization (linearity-focused)
        try:
            priority_models = db.get_linearity_prioritization(days_back=self.selected_days, min_samples=10)
        except Exception as e:
            logger.debug(f"Could not load prioritization: {e}")
            priority_models = []

        # Get prioritized model list for dropdown (MPS first, then active, then inactive)
        prioritized_models = db.get_models_list_prioritized(
            mps_models=mps_models,
            recent_days=recent_days
        )

        # Build model names list with filtering
        model_names = self._filter_model_list(prioritized_models, priority_data=priority_models)

        # Get heatmap data
        try:
            heatmap_data = db.get_heatmap_data(days_back=self.selected_days)
        except Exception as e:
            logger.debug(f"Could not load heatmap data: {e}")
            heatmap_data = None

        # Load ML insights on background thread (disk I/O for MLManager.load_all)
        ml_insights = self._get_ml_summary_insights()

        # Per-model anomaly rates — surfaced as a stat tile so persistent
        # setup issues are visible at the model level (per-track is_anomaly
        # was visible per-unit but never rolled up before).
        try:
            anomaly_rows = db.get_anomaly_rate_by_model(
                days_back=self.selected_days, min_samples=10
            )
        except Exception as e:
            logger.debug(f"Could not load anomaly rates: {e}")
            anomaly_rows = []

        # Failure Severity + Cost Impact need near_miss summary and pricing
        try:
            near_miss = db.get_near_miss_summary(days_back=self.selected_days)
        except Exception as e:
            logger.debug(f"Could not load near-miss summary: {e}")
            near_miss = {}
        cfg = get_config()
        pricing = dict(cfg.active_models.model_prices or {})
        cost_ratio = float(getattr(cfg.active_models, "cost_ratio", 0.5))

        # Update UI on main thread; capture gen so we can discard stale loads
        self.after(0, lambda g=gen: self._update_summary_display_if_current(
            g, active_models, alert_models, model_names, trending_worse,
            mps_models=mps_models, recent_days=recent_days,
            priority_models=priority_models, heatmap_data=heatmap_data,
            ml_insights=ml_insights, anomaly_rows=anomaly_rows,
            near_miss=near_miss, pricing=pricing, cost_ratio=cost_ratio,
        ))

    def _update_summary_display_if_current(self, gen: int, *args, **kwargs):
        """Apply summary load result only if no newer load has superseded it.

        Without this guard the callback runs even after _create_detail_view
        has destroyed the summary frames, raising TclError when we try to
        update widgets that no longer exist.
        """
        if gen != self._load_generation:
            return
        # Defensive: also confirm the page itself is still alive AND a
        # frame from the new 5-section summary layout still exists.
        # Pick _focus_rows_frame because it is one of the first widgets
        # _create_summary_view builds; if it has been destroyed, the
        # whole summary view is gone (user navigated to detail mode).
        if not self.winfo_exists():
            return
        focus_frame = getattr(self, "_focus_rows_frame", None)
        if focus_frame is None or not focus_frame.winfo_exists():
            return
        self._update_summary_display(*args, **kwargs)

    def _load_detail_data(self, db, gen: int = 0):
        """Load data for detail mode."""
        # Get active models config for MPS prioritization
        config = get_config()
        mps_models = config.active_models.mps_models
        recent_days = config.active_models.recent_days

        # Get detailed trend data for this model
        # Strip (inactive) suffix if present
        clean_model = self.selected_model
        if " (inactive)" in clean_model:
            clean_model = clean_model.replace(" (inactive)", "")

        trend_data = db.get_model_trend_data(
            model=clean_model,
            days_back=self.selected_days,
            rolling_window=self.rolling_window
        )

        # Get alerts for this model (linearity-focused)
        alert_models = db.get_models_requiring_attention(
            days_back=self.selected_days,
            min_samples=5,
            pass_rate_threshold=80.0,
            trend_threshold=10.0,
            rolling_days=self.rolling_window,
            metric="linearity"
        )
        model_alerts = next((a for a in alert_models if a["model"] == clean_model), None)

        # Get ML recommendations
        ml_recommendations = self._get_ml_recommendations(trend_data)

        # Get prioritized model list for dropdown (MPS first, then active, then inactive)
        prioritized_models = db.get_models_list_prioritized(
            mps_models=mps_models,
            recent_days=recent_days
        )

        # Build model names list with filtering
        model_names = self._filter_model_list(prioritized_models)

        # Get the model's analysis-level stats (for consistent pass rate with alerts)
        active_models = db.get_active_models_summary(self.selected_days, 5)
        model_stats = next((m for m in active_models if m["model"] == clean_model), None)

        # Get linearity margin analysis and prioritization data for this model
        try:
            margin_data = db.get_linearity_margin_analysis(clean_model, days_back=self.selected_days)
        except Exception as e:
            logger.debug(f"Could not load margin analysis: {e}")
            margin_data = {}
        model_priority = None
        try:
            prio_list = db.get_linearity_prioritization(days_back=self.selected_days, min_samples=5)
            model_priority = next((m for m in prio_list if m["model"] == clean_model), None)
        except Exception as e:
            logger.debug(f"Could not load prioritization: {e}")

        # Get Cpk
        try:
            cpk_data = db.get_model_cpk(clean_model, days_back=self.selected_days)
        except Exception as e:
            logger.debug(f"Could not load Cpk: {e}")
            cpk_data = None

        # Update UI on main thread; gen lets us discard stale results if
        # the user switched models or back to All Models in the meantime.
        self.after(0, lambda g=gen: self._update_detail_display_if_current(
            g, trend_data, model_alerts, ml_recommendations, model_names, model_stats,
            margin_data=margin_data, model_priority=model_priority, cpk_data=cpk_data
        ))

    def _update_detail_display_if_current(self, gen: int, *args, **kwargs):
        """Apply detail load result only if not superseded by a newer load."""
        if gen != self._load_generation:
            return
        if not self.winfo_exists():
            return
        # Confirm the detail frames the renderer will write into still exist.
        # _create_summary_view destroys these; running anyway crashes Tk.
        scatter_frame = getattr(self, "_scatter_frame", None)
        if scatter_frame is not None and not scatter_frame.winfo_exists():
            return
        self._update_detail_display(*args, **kwargs)

    def _get_ml_recommendations(self, trend_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get ML recommendations for current model using per-model ML system."""
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.ml import get_shared_ml_manager

            db = get_database()
            # Shared cached MLManager — avoids reloading 134 predictor pickles
            # on every navigation. Settings → Train invalidates the cache.
            ml_manager = get_shared_ml_manager(db)

            # Get threshold from per-model optimizer (strip inactive suffix)
            clean_model_name = self.selected_model.replace(" (inactive)", "")
            threshold = ml_manager.get_threshold(clean_model_name)

            if threshold is not None:
                optimizer = ml_manager.threshold_optimizers.get(clean_model_name)
                profiler = ml_manager.profilers.get(clean_model_name)
                detector = ml_manager.drift_detectors.get(clean_model_name)

                result = {
                    "recommended_threshold": threshold,
                    "confidence": optimizer.confidence if optimizer else 0.5,
                    "method": optimizer.method if optimizer else "formula",
                    "basis": f"{optimizer.n_samples if optimizer else 0} samples",
                }

                # Add drift info if available
                if detector and detector.has_baseline:
                    result["drift_status"] = "Drifting" if detector.is_drifting else "Stable"
                    result["drift_direction"] = detector.drift_direction.value if detector.drift_direction else None

                # Add profile insights if available
                if profiler and profiler.profile:
                    result["pass_rate"] = profiler.profile.pass_rate
                    result["difficulty"] = profiler.profile.difficulty_score
                    result["insights"] = profiler.get_insights()[:3]  # Top 3 insights

                return result

            # No fallback - new per-model system is the only source
            # Train models in Settings to get ML recommendations

        except Exception as e:
            logger.debug(f"ML recommendations not available: {e}")

        return None

    def _update_summary_display(
        self,
        active_models: List[Dict[str, Any]],
        alert_models: List[Dict[str, Any]],
        model_names: List[str],
        trending_worse: Optional[List[Dict[str, Any]]] = None,
        mps_models: Optional[List[str]] = None,
        recent_days: int = 90,
        priority_models: Optional[List[Dict[str, Any]]] = None,
        heatmap_data: Optional[Dict[str, Any]] = None,
        ml_insights: Optional[Dict[str, Any]] = None,
        anomaly_rows: Optional[List[Dict[str, Any]]] = None,
        near_miss: Optional[Dict[str, Any]] = None,
        pricing: Optional[Dict[str, float]] = None,
        cost_ratio: float = 0.5,
    ):
        """Update summary display with loaded data.

        Five sections, all driven by data already fetched in
        _load_summary_data: stats row, Focus This Week (priority_models),
        Failure Severity (near_miss), Cost Impact (priority_models +
        pricing + cost_ratio), ML Status.
        """
        if not self.winfo_exists():
            return
        # Update model dropdown
        current_model = self.model_dropdown.get()
        self.model_dropdown.configure(values=model_names)
        if current_model in model_names:
            self.model_dropdown.set(current_model)
        else:
            self.model_dropdown.set("All Models")

        self.active_models_data = active_models

        # Stats row
        if not active_models:
            self._reset_summary_stats()
        else:
            total_models = len(active_models)
            # get_active_models_summary returns rows keyed by "total" (not
            # "total_samples"). Using the wrong key silently produced 0 for
            # the Total Samples stat tile.
            total_samples = sum(m.get("total", 0) for m in active_models)

            # Volume-weighted fleet rates (NOT an unweighted mean of per-model %),
            # so a high-volume bad lot isn't diluted by many tiny healthy models.
            def _weighted(key):
                num = sum(m.get(key, 0) * m.get("total", 0) for m in active_models)
                return (num / total_samples) if total_samples else 0

            avg_pass_rate = _weighted("pass_rate")
            avg_sigma_rate = _weighted("sigma_pass_rate")
            avg_linearity_rate = _weighted("linearity_pass_rate")
            models_at_risk = sum(
                1 for m in active_models if m.get("linearity_pass_rate", 100) < 80
            )
            sorted_by_rate = sorted(
                active_models, key=lambda x: x.get("linearity_pass_rate", 0),
                reverse=True
            )
            best_model = sorted_by_rate[0]["model"] if sorted_by_rate else "--"
            worst_model = sorted_by_rate[-1]["model"] if sorted_by_rate else "--"
            best_rate = sorted_by_rate[0].get("linearity_pass_rate", 0) if sorted_by_rate else 0
            worst_rate = sorted_by_rate[-1].get("linearity_pass_rate", 0) if sorted_by_rate else 0

            self.summary_stat_labels["active_models"].configure(text=str(total_models))
            self.summary_stat_labels["total_samples"].configure(text=f"{total_samples:,}")
            self.summary_stat_labels["avg_pass_rate"].configure(
                text=f"{avg_pass_rate:.1f}%",
                text_color="#27ae60" if avg_pass_rate >= 90
                           else "#f39c12" if avg_pass_rate >= 80 else "#e74c3c",
            )
            self.summary_stat_labels["avg_sigma_rate"].configure(
                text=f"{avg_sigma_rate:.1f}%",
                text_color="#27ae60" if avg_sigma_rate >= 90
                           else "#f39c12" if avg_sigma_rate >= 80 else "#e74c3c",
            )
            self.summary_stat_labels["avg_linearity_rate"].configure(
                text=f"{avg_linearity_rate:.1f}%",
                text_color="#27ae60" if avg_linearity_rate >= 90
                           else "#f39c12" if avg_linearity_rate >= 80 else "#e74c3c",
            )
            self.summary_stat_labels["models_at_risk"].configure(
                text=str(models_at_risk),
                text_color="#e74c3c" if models_at_risk > 0 else "#27ae60",
            )
            self.summary_stat_labels["best_model"].configure(
                text=f"{best_model} ({best_rate:.0f}%)", text_color="#27ae60"
            )
            self.summary_stat_labels["worst_model"].configure(
                text=f"{worst_model} ({worst_rate:.0f}%)",
                text_color="#e74c3c" if worst_rate < 80 else "#f39c12",
            )
            top_anom = (anomaly_rows or [None])[0] if anomaly_rows else None
            anomaly_label = self.summary_stat_labels.get("top_anomaly")
            if anomaly_label is not None:
                if top_anom and top_anom["anomaly_count"] > 0:
                    rate = top_anom["anomaly_rate"]
                    color = ("#e74c3c" if rate >= 15
                             else "#f39c12" if rate >= 5 else "white")
                    anomaly_label.configure(
                        text=f"{top_anom['model']} "
                             f"({top_anom['anomaly_count']}, {rate:.0f}%)",
                        text_color=color,
                    )
                else:
                    anomaly_label.configure(text="None", text_color="#27ae60")

        # Focus This Week
        self._update_focus_section(priority_models or [])

        # Failure Severity chart
        self._update_failure_severity_chart(near_miss or {})

        # Cost Impact chart
        self._update_cost_impact_chart(
            priority_models or [], pricing or {}, cost_ratio
        )

        # ML Status (existing helper)
        self._update_ml_summary(alert_models, ml_insights=ml_insights)
        self._cached_alert_models = alert_models
        self._cached_ml_insights = ml_insights

    def _update_heatmap(self, heatmap_data: Optional[Dict[str, Any]]):
        """Update the heat map chart with model x week pass rate data."""
        try:
            if not self.heatmap_chart:
                return

            if not heatmap_data or not heatmap_data.get("models"):
                self.heatmap_chart.show_placeholder("No data for heat map")
                return

            self.heatmap_chart.plot_heatmap(
                models=heatmap_data["models"],
                periods=heatmap_data["periods"],
                values=heatmap_data["values"],
            )
        except Exception as e:
            logger.debug(f"Heatmap update error: {e}")
            if self.heatmap_chart:
                self.heatmap_chart.show_placeholder("Error loading heat map")

    def _update_low_data_list(self, low_data_models: List[Dict[str, Any]]):
        """Update the low data models list section."""
        # Clear existing items
        for widget in self._low_data_list.winfo_children():
            widget.destroy()

        if not low_data_models:
            placeholder = ctk.CTkLabel(
                self._low_data_list,
                text="No models with insufficient data",
                text_color="gray",
                font=ctk.CTkFont(size=10)
            )
            placeholder.pack(padx=10, pady=20)
            return

        # Sort by sample count (fewest first)
        sorted_models = sorted(low_data_models, key=lambda x: x["total"])

        for model_data in sorted_models[:10]:  # Show max 10
            model_name = model_data["model"]
            samples = model_data["total"]
            last_date = model_data.get("last_date")

            # Format: "Model: X samples (last: date)"
            date_str = last_date.strftime("%m/%d") if last_date else "N/A"
            text = f"{model_name}: {samples} samples (last: {date_str})"

            label = ctk.CTkLabel(
                self._low_data_list,
                text=text,
                font=ctk.CTkFont(size=10),
                text_color="#f39c12"  # Orange for warning
            )
            label.pack(padx=10, pady=2, anchor="w")

    def _update_detail_display(
        self,
        trend_data: Dict[str, Any],
        model_alerts: Optional[Dict[str, Any]],
        ml_recommendations: Optional[Dict[str, Any]],
        model_names: List[str],
        model_stats: Optional[Dict[str, Any]] = None,
        margin_data: Optional[Dict[str, Any]] = None,
        model_priority: Optional[Dict[str, Any]] = None,
        cpk_data: Optional[Dict[str, Any]] = None,
    ):
        """Update detail display with loaded data."""
        if not self.winfo_exists():
            return
        # Ensure charts are initialized before use (lazy matplotlib loading)
        self._ensure_detail_charts_initialized()

        # Update model dropdown
        current_model = self.model_dropdown.get()
        self.model_dropdown.configure(values=model_names)
        if current_model in model_names:
            self.model_dropdown.set(current_model)
        else:
            # Current model was filtered out (e.g. Active Only) — switch to summary
            self.model_dropdown.set("All Models")
            self.selected_model = "All Models"
            self.after(50, lambda: self._on_model_change("All Models"))
            return

        self.model_trend_data = trend_data

        if not trend_data or not trend_data.get("data_points"):
            self._reset_detail_stats()
            self.scatter_chart.show_placeholder("No data for this model in selected period")
            self.dist_chart.show_placeholder("No distribution data")
            self.status_label.configure(text="No data")
            return

        # Extract data
        data_points = trend_data["data_points"]
        rolling_averages = trend_data.get("rolling_averages", [])
        threshold = trend_data.get("threshold")
        total_samples = len(data_points)

        # Count and filter anomalies from statistics calculations
        # Anomalies (trim failures with linear slope) would skew averages
        anomaly_count = sum(1 for d in data_points if d.get("is_anomaly", False))
        normal_points = [d for d in data_points if not d.get("is_anomaly", False)]
        normal_sample_count = len(normal_points)

        # Calculate stats from NORMAL samples only (excludes anomalies)
        sigma_values = [d["sigma_gradient"] for d in normal_points if d["sigma_gradient"] is not None]

        # Sigma pass rate (track-level). Exclude points with no sigma result
        # (UNTRIMMED -> sigma_pass is None) from BOTH numerator and denominator,
        # matching the DB-side denominator; otherwise the rate is deflated.
        sigma_gradeable = [d for d in normal_points if d.get("sigma_pass") is not None]
        sigma_pass_count = sum(1 for d in sigma_gradeable if d.get("sigma_pass"))
        sigma_pass_rate = (sigma_pass_count / len(sigma_gradeable) * 100) if sigma_gradeable else 0

        # Overall pass rate - use model_stats from get_active_models_summary for consistency with alerts
        # This counts analysis-level pass (both sigma AND linearity must pass for all tracks)
        if model_stats:
            overall_pass_rate = model_stats.get("pass_rate", 0)
            linearity_pass_rate = model_stats.get("linearity_pass_rate", 0)
            total_analyses = model_stats.get("total", total_samples)
        else:
            # Fallback: count from track data (may differ from analysis-level count)
            overall_pass_count = sum(1 for d in data_points if str(d.get("status", "")).upper() == "PASS")
            overall_pass_rate = (overall_pass_count / total_samples * 100) if total_samples > 0 else 0
            linearity_pass_rate = 0  # Can't calculate without model_stats
            total_analyses = total_samples

        avg_sigma = np.mean(sigma_values) if sigma_values else 0
        std_sigma = np.std(sigma_values, ddof=1) if len(sigma_values) > 1 else 0

        # Trend direction
        if len(sigma_values) >= 6:
            first_half = np.mean(sigma_values[:len(sigma_values)//2])
            second_half = np.mean(sigma_values[len(sigma_values)//2:])
            if second_half > first_half * 1.1:
                trend = "Increasing"
                trend_color = "#e74c3c"
            elif second_half < first_half * 0.9:
                trend = "Decreasing"
                trend_color = "#27ae60"
            else:
                trend = "Stable"
                trend_color = "#3498db"
        else:
            trend = "Insufficient Data"
            trend_color = "gray"

        # Status based on overall pass rate
        if overall_pass_rate >= 95:
            status = "Excellent"
            status_color = "#27ae60"
        elif overall_pass_rate >= 80:
            status = "Good"
            status_color = "#3498db"
        elif overall_pass_rate >= 70:
            status = "Warning"
            status_color = "#f39c12"
        else:
            status = "Critical"
            status_color = "#e74c3c"

        # Update stat labels
        # Use analysis count from model_stats if available, otherwise track count
        display_count = total_analyses if model_stats else total_samples
        self.detail_stat_labels["total_samples"].configure(text=f"{display_count:,}")
        self.detail_stat_labels["anomalies"].configure(
            text=f"{anomaly_count}",
            text_color="#9b59b6" if anomaly_count > 0 else "gray"
        )
        self.detail_stat_labels["sigma_pass_rate"].configure(
            text=f"{sigma_pass_rate:.1f}%",
            text_color="#27ae60" if sigma_pass_rate >= 90 else "#f39c12" if sigma_pass_rate >= 80 else "#e74c3c"
        )
        self.detail_stat_labels["linearity_pass_rate"].configure(
            text=f"{linearity_pass_rate:.1f}%",
            text_color="#27ae60" if linearity_pass_rate >= 90 else "#f39c12" if linearity_pass_rate >= 80 else "#e74c3c"
        )
        self.detail_stat_labels["overall_pass_rate"].configure(
            text=f"{overall_pass_rate:.1f}%",
            text_color="#27ae60" if overall_pass_rate >= 90 else "#f39c12" if overall_pass_rate >= 80 else "#e74c3c"
        )
        # Near-miss and trim effectiveness from prioritization data
        near_miss = model_priority.get("near_miss_count", 0) if model_priority else 0
        avg_trim_imp = model_priority.get("avg_trim_improvement", None) if model_priority else None
        self.detail_stat_labels["near_miss"].configure(
            text=f"{near_miss}",
            text_color="#f39c12" if near_miss > 0 else "gray"
        )
        self.detail_stat_labels["avg_trim_improvement"].configure(
            text=f"{avg_trim_imp:.1f}%" if avg_trim_imp is not None else "N/A",
            text_color="white"
        )
        self.detail_stat_labels["trend"].configure(text=trend, text_color=trend_color)

        # Cpk
        if cpk_data and cpk_data.get("sigma_cpk") is not None:
            cpk_val = cpk_data["sigma_cpk"]
            cpk_color = cpk_data.get("sigma_cpk_color", "gray")
            self.detail_stat_labels["sigma_cpk"].configure(
                text=f"{cpk_val:.2f}",
                text_color=cpk_color
            )
        else:
            self.detail_stat_labels["sigma_cpk"].configure(text="N/A", text_color="gray")

        # Update scatter chart using the filter method (applies timeline filter)
        self._update_scatter_chart_with_filter()

        # Update linearity chart
        self._update_linearity_chart_with_filter()

        # Update distribution - only show histogram for 20+ samples
        # For small sample sizes, the trend chart above shows everything needed
        if len(sigma_values) >= 20:
            self.dist_chart.master.grid()  # Show the distribution frame
            self.dist_chart.plot_histogram(
                values=sigma_values,
                bins=min(30, len(sigma_values) // 3 + 1),
                title="Linearity Error Distribution",
                xlabel="Linearity Error Rate",
                spec_limit=threshold
            )
        else:
            # Hide distribution chart for small sample sizes
            self.dist_chart.master.grid_remove()

        # Update alerts text
        self._update_detail_alerts(model_alerts)

        # Update ML text
        self._update_detail_ml(ml_recommendations)

        # Update status
        self.status_label.configure(text=f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    def _reset_summary_stats(self):
        """Reset summary statistics to default values."""
        for key in self.summary_stat_labels:
            self.summary_stat_labels[key].configure(text="--", text_color="white")
        # impact_text was removed in favour of the Priorities-tab pointer
        # button — nothing to reset here. Left as comment for grep history.

    def _reset_detail_stats(self):
        """Reset detail statistics to default values."""
        for key in self.detail_stat_labels:
            self.detail_stat_labels[key].configure(text="--", text_color="white")

    def _update_ml_summary(self, alert_models: Optional[List[Dict[str, Any]]], ml_insights: Optional[Dict[str, Any]] = None):  # noqa: E501
        """ML insights summary on the Standard view.

        The drift portion that used to render here duplicated the Drift
        tab — now we just show a one-line trained-model count and let
        the user click 'View All Details' for the full breakdown.
        """
        # Cache for the details dialog
        self._cached_alert_models = alert_models
        # Use pre-loaded insights (from background thread) or fall back to cached
        self._cached_ml_insights = ml_insights if ml_insights is not None else self._cached_ml_insights

        self.ml_text.configure(state="normal")
        self.ml_text.delete("1.0", "end")

        ml_insights = self._cached_ml_insights
        has_content = False

        if ml_insights:
            has_content = True
            trained = ml_insights.get("trained_models", 0)
            drifting = ml_insights.get("drifting_models", [])

            # Single status line — full drift list lives on the Drift tab,
            # listing model names again here was redundant per UX audit.
            if trained > 0:
                drift_note = (
                    f"{len(drifting)} drifting (see Drift tab)"
                    if drifting else "no drift"
                )
                self.ml_text.insert(
                    "end",
                    f"ML Status: {trained} models trained · {drift_note}\n\n",
                )

        # Show alert summary if any - show more items now
        if alert_models:
            has_content = True
            self.ml_text.insert("end", f"Models Requiring Attention ({len(alert_models)}):\n")
            for alert_model in alert_models[:8]:  # Show top 8 in summary
                model_name = alert_model.get("model", "Unknown")
                pass_rate = alert_model.get("pass_rate", 0)
                alerts = alert_model.get("alerts", [])

                # Get alert types for this model
                alert_types = [al["type"] for al in alerts]

                if "LOW_PASS_RATE" in alert_types:
                    self.ml_text.insert("end", f"  ⚠ {model_name}: {pass_rate:.1f}% pass rate\n")
                elif "TRENDING_WORSE" in alert_types:
                    self.ml_text.insert("end", f"  ↓ {model_name}: trending worse\n")
                elif "HIGH_VARIANCE" in alert_types:
                    self.ml_text.insert("end", f"  ~ {model_name}: high variance\n")
                else:
                    self.ml_text.insert("end", f"  • {model_name}\n")

            if len(alert_models) > 8:
                self.ml_text.insert("end", f"  ... click 'View All Details' for {len(alert_models) - 8} more\n")

        if not has_content:
            self.ml_text.insert("end", "All models performing well.\n")
            self.ml_text.insert("end", "Train models in Settings for ML insights.")

        self.ml_text.configure(state="disabled")

    def _get_ml_summary_insights(self) -> Optional[Dict[str, Any]]:
        """Get ML insights for summary view."""
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.ml import get_shared_ml_manager

            db = get_database()
            ml_manager = get_shared_ml_manager(db)

            if not ml_manager.profilers:
                return None

            result = {
                "trained_models": len(ml_manager.profilers),
                "difficulty_ranking": [],
                "drifting_models": [],
            }

            # Get difficulty ranking (hardest first)
            for model, profiler in ml_manager.profilers.items():
                if profiler.profile:
                    result["difficulty_ranking"].append(
                        (model, profiler.profile.difficulty_score)
                    )
            result["difficulty_ranking"].sort(key=lambda x: -x[1])  # Descending

            # Get drifting models
            for model, detector in ml_manager.drift_detectors.items():
                if detector.has_baseline and detector.is_drifting:
                    direction = detector.drift_direction.value if detector.drift_direction else "unknown"
                    result["drifting_models"].append((model, direction))

            return result

        except Exception as e:
            logger.debug(f"Could not get ML summary insights: {e}")
            return None

    def _show_ml_details_dialog(self):
        """Show a dialog with full ML insights details."""
        from tkinter import Toplevel

        dialog = ctk.CTkToplevel(self)
        dialog.title("ML Insights - Full Details")
        dialog.geometry("700x600")
        dialog.transient(self)
        dialog.grab_set()

        # Main container with scrollable text
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        title_label = ctk.CTkLabel(
            main_frame,
            text="ML Insights - Full Details",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=(10, 15))

        # Scrollable text area
        text_widget = ctk.CTkTextbox(main_frame, width=650, height=450)
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)

        # Use cached data (loaded on background thread during summary refresh)
        ml_insights = getattr(self, '_cached_ml_insights', None)
        alert_models = getattr(self, '_cached_alert_models', None)

        # Section 1: ML Status Overview
        text_widget.insert("end", "=" * 60 + "\n")
        text_widget.insert("end", "ML SYSTEM STATUS\n")
        text_widget.insert("end", "=" * 60 + "\n\n")

        if ml_insights:
            trained = ml_insights.get("trained_models", 0)
            text_widget.insert("end", f"Trained Models: {trained}\n\n")

            # Drift status
            drifting = ml_insights.get("drifting_models", [])
            if drifting:
                text_widget.insert("end", f"DRIFT DETECTED in {len(drifting)} model(s):\n")
                for model, direction in drifting:
                    text_widget.insert("end", f"  ⚠ {model}: drifting {direction}\n")
                text_widget.insert("end", "\n")
            else:
                text_widget.insert("end", "Drift Status: All models stable\n\n")

            # Difficulty ranking
            difficulty = ml_insights.get("difficulty_ranking", [])
            if difficulty:
                text_widget.insert("end", "Model Difficulty Ranking (hardest first):\n")
                for rank, (model, score) in enumerate(difficulty, 1):
                    label = "Easy" if score < 0.3 else "Medium" if score < 0.6 else "Hard"
                    bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                    text_widget.insert("end", f"  {rank:2}. {model:<15} {bar} {label} ({score:.2f})\n")
                text_widget.insert("end", "\n")
        else:
            text_widget.insert("end", "No ML models trained.\n")
            text_widget.insert("end", "Go to Settings → Train Models to enable ML insights.\n\n")

        # Section 2: Models Requiring Attention
        text_widget.insert("end", "=" * 60 + "\n")
        text_widget.insert("end", "MODELS REQUIRING ATTENTION\n")
        text_widget.insert("end", "=" * 60 + "\n\n")

        if alert_models:
            text_widget.insert("end", f"Total: {len(alert_models)} models with alerts\n\n")

            for i, alert_model in enumerate(alert_models, 1):
                model_name = alert_model.get("model", "Unknown")
                pass_rate = alert_model.get("pass_rate", 0)
                total = alert_model.get("total", 0)
                alerts = alert_model.get("alerts", [])

                text_widget.insert("end", f"{i}. {model_name}\n")
                text_widget.insert("end", f"   Pass Rate: {pass_rate:.1f}%  |  Samples: {total}\n")

                if alerts:
                    text_widget.insert("end", "   Alerts:\n")
                    for alert in alerts:
                        alert_type = alert.get("type", "Unknown")
                        severity = alert.get("severity", "Medium")
                        message = alert.get("message", "")
                        icon = "!!" if severity == "High" else "!" if severity == "Medium" else "·"
                        text_widget.insert("end", f"     {icon} [{severity}] {alert_type}\n")
                        if message:
                            text_widget.insert("end", f"        {message}\n")
                text_widget.insert("end", "\n")
        else:
            text_widget.insert("end", "No models currently require attention.\n")
            text_widget.insert("end", "All models are performing within acceptable parameters.\n")

        # Section 3: Quick Tips
        text_widget.insert("end", "\n" + "=" * 60 + "\n")
        text_widget.insert("end", "QUICK TIPS\n")
        text_widget.insert("end", "=" * 60 + "\n\n")
        text_widget.insert("end", "• LOW_PASS_RATE: Model has <50% pass rate - investigate process\n")
        text_widget.insert("end", "• TRENDING_WORSE: Quality declining over time - check for drift\n")
        text_widget.insert("end", "• HIGH_VARIANCE: Inconsistent results - check equipment calibration\n")
        text_widget.insert("end", "• DRIFT (up): Sigma values increasing - process degrading\n")
        text_widget.insert("end", "• DRIFT (down): Sigma values decreasing - process improving\n")

        text_widget.configure(state="disabled")

        # Close button
        close_btn = ctk.CTkButton(
            main_frame,
            text="Close",
            command=dialog.destroy,
            width=100
        )
        close_btn.pack(pady=(10, 10))

        # Center dialog on parent
        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def _update_detail_alerts(self, model_alerts: Optional[Dict[str, Any]]):
        """Update detail alerts text."""
        self.detail_alerts_text.configure(state="normal")
        self.detail_alerts_text.delete("1.0", "end")

        if not model_alerts:
            self.detail_alerts_text.insert("end", "No alerts for this model.\n\n")
            self.detail_alerts_text.insert("end", "Model is performing within acceptable parameters.")
        else:
            alerts = model_alerts.get("alerts", [])
            self.detail_alerts_text.insert("end", f"{len(alerts)} Alert(s) Detected:\n\n")

            for alert in alerts:
                severity = alert.get("severity", "Medium")
                icon = "!!" if severity == "High" else "!"
                self.detail_alerts_text.insert("end", f"{icon} [{severity}] {alert.get('message', 'Unknown')}\n")

        self.detail_alerts_text.configure(state="disabled")

    def _update_detail_ml(self, ml_recommendations: Optional[Dict[str, Any]]):
        """Update detail ML recommendations text."""
        self.detail_ml_text.configure(state="normal")
        self.detail_ml_text.delete("1.0", "end")

        if ml_recommendations is None:
            self.detail_ml_text.insert("end", "ML recommendations not available.\n\n")
            self.detail_ml_text.insert("end", "Train models in Settings to see recommendations.")
        else:
            threshold = ml_recommendations.get("recommended_threshold", 0)
            confidence = ml_recommendations.get("confidence", 0)
            basis = ml_recommendations.get("basis", "historical data")
            method = ml_recommendations.get("method", "unknown")
            is_legacy = ml_recommendations.get("legacy", False)

            # Threshold section — explain in plain language
            self.detail_ml_text.insert("end", f"Recommended Threshold: {threshold:.6f}\n")
            self.detail_ml_text.insert("end", f"  Method: {method}, Confidence: {confidence:.0%}\n")
            self.detail_ml_text.insert("end", f"  Based on: {basis}\n")
            # Add plain-English interpretation
            if confidence >= 0.7:
                self.detail_ml_text.insert("end", f"  → High confidence — consider using this threshold\n")
            elif confidence >= 0.4:
                self.detail_ml_text.insert("end", f"  → Moderate confidence — review before applying\n")
            else:
                self.detail_ml_text.insert("end", f"  → Low confidence — more data needed\n")

            if is_legacy:
                self.detail_ml_text.insert("end", "  (Using legacy optimizer - retrain for per-model ML)\n")

            # Drift status section (new ML system only)
            drift_status = ml_recommendations.get("drift_status")
            if drift_status:
                drift_direction = ml_recommendations.get("drift_direction", "")
                drift_text = f"{drift_status}"
                if drift_direction:
                    drift_text += f" ({drift_direction})"
                status_indicator = "⚠" if drift_status == "Drifting" else "✓"
                self.detail_ml_text.insert("end", f"\nDrift Status: {status_indicator} {drift_text}\n")

            # Profile insights section (new ML system only)
            pass_rate = ml_recommendations.get("pass_rate")
            difficulty = ml_recommendations.get("difficulty")
            if pass_rate is not None or difficulty is not None:
                self.detail_ml_text.insert("end", "\nModel Profile:\n")
                if pass_rate is not None:
                    self.detail_ml_text.insert("end", f"  Pass Rate: {pass_rate * 100:.1f}%\n")
                if difficulty is not None:
                    diff_label = "Easy" if difficulty < 0.3 else "Medium" if difficulty < 0.6 else "Hard"
                    self.detail_ml_text.insert("end", f"  Difficulty: {diff_label} ({difficulty:.2f})\n")

            # Top insights (new ML system only)
            insights = ml_recommendations.get("insights", [])
            if insights:
                self.detail_ml_text.insert("end", "\nInsights:\n")
                for insight in insights[:3]:
                    msg = insight.message if hasattr(insight, 'message') else str(insight)
                    self.detail_ml_text.insert("end", f"  • {msg}\n")

        self.detail_ml_text.configure(state="disabled")

    def _show_error(self, error: str):
        """Show error state."""
        if not self.winfo_exists():
            return
        self.status_label.configure(text="Error loading data")
        if getattr(self, 'alerts_chart', None):
            self.alerts_chart.show_placeholder(f"Error: {error}")
        if getattr(self, 'scatter_chart', None):
            self.scatter_chart.show_placeholder(f"Error: {error}")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Trends page shown")
        # Populate spec filter dropdowns
        self._populate_spec_filters()
        # Recreate view if charts were cleaned up
        if not self._summary_charts_initialized and not self._detail_charts_initialized:
            if self.selected_model == "All Models":
                self._create_summary_view()
            else:
                self._create_detail_view()
        self._refresh_data()

    def _on_trend_type_changed(self, value: str):
        """Handle trend type selector change.

        Three top-level views; Drift internally toggles between ML and
        Process via the sub-button on its tab. Bumping _load_generation
        on non-Standard branches discards any in-flight summary load.
        """
        if value != "Standard":
            self._load_generation += 1

        if value == "Standard":
            if self.selected_model == "All Models":
                self._create_summary_view()
            else:
                self._create_detail_view()
            self._refresh_data()
        elif value == "Drift":
            # Reset the sub-toggle so re-entering the Drift tab after a
            # previous "Process Drift" sub-selection doesn't trip the
            # render guard (which would leave the tab blank).
            self._drift_subtab = "ML Drift"
            if hasattr(self, "_drift_subtab_button"):
                try:
                    self._drift_subtab_button.set("ML Drift")
                except Exception:
                    pass
            self._refresh_drift_view()
        elif value == "Trim Difficulty":
            self._show_trim_difficulty()
        # Any legacy stored selection ("Priorities", "Comparative", etc.)
        # is silently ignored — the segmented button no longer surfaces
        # those values, so the only way to reach this branch is a stale
        # config restoration on app launch. Treat as no-op.

    def show_drift_tab(self):
        """Public hook used by the Dashboard's Drift Alerts card to jump
        straight to the Drift timeline view on this page."""
        try:
            self._trend_type.set("Drift")
        except Exception:
            pass
        self._refresh_drift_view()

    def _refresh_drift_view(self):
        """Re-render whichever Drift sub-view is currently active.

        Used by callers (date-range change, tab entry, dashboard hook)
        that need the Drift tab to refresh without having to know which
        of single-model / Process Drift / ML Drift is on screen.
        """
        if self._drift_filter_model:
            self._show_single_model_drift()
        elif self._drift_subtab == "Process Drift":
            self._show_process_drift()
        else:
            self._show_drift_timeline()

    def _create_dedicated_chart_view(self, title: str) -> "ChartWidget":
        """Create a dedicated full-width chart view for non-Standard tabs.

        Replaces the summary view with a single titled chart, avoiding the
        Standard tab's stats/headers bleeding into other tab views.
        """
        self._cleanup_charts()

        for widget in self.content.winfo_children():
            widget.destroy()

        ChartWidget, ChartStyle = _ensure_chart_module()

        frame = ctk.CTkFrame(self.content)
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text=title,
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        chart = ChartWidget(
            frame,
            style=ChartStyle(figure_size=(12, 6), dpi=100)
        )
        chart.grid(row=1, column=0, sticky="nsew", padx=15, pady=(5, 15))
        self._chart_widgets.append(chart)

        self.content.grid_rowconfigure(0, weight=1)

        return chart

    def _create_drift_view(self) -> "ChartWidget":
        """Build the Drift tab: header row (model filter + ML/Process toggle)
        on top, chart frame below. Returns the chart widget for All-models
        sub-views; the single-model view replaces the chart frame contents.
        """
        for widget in self.content.winfo_children():
            widget.destroy()
        self._cleanup_charts()

        self.content.grid_rowconfigure(0, weight=0)
        self.content.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.content, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        # Model filter dropdown
        ctk.CTkLabel(
            header, text="Model:",
            font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(0, 4))
        try:
            db = get_database()
            model_options = ["All models"] + db.get_models_with_sigma_data(
                days_back=self.selected_days
            )
        except Exception as e:
            logger.debug(f"Could not populate drift model filter: {e}")
            model_options = ["All models"]
        self._drift_model_filter = ctk.CTkComboBox(
            header,
            values=model_options,
            command=self._on_drift_model_filter_changed,
            width=160,
        )
        self._drift_model_filter.set(self._drift_filter_model or "All models")
        self._drift_model_filter.pack(side="left", padx=(0, 14))

        # ML/Process sub-tab toggle — hidden when a specific model is selected.
        self._drift_subtab_button = ctk.CTkSegmentedButton(
            header,
            values=["ML Drift", "Process Drift"],
            command=self._on_drift_subtab_changed,
        )
        self._drift_subtab_button.set(self._drift_subtab)
        if self._drift_filter_model:
            self._drift_subtab_button.pack_forget()
        else:
            self._drift_subtab_button.pack(side="left")

        chart_frame = ctk.CTkFrame(self.content)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self._drift_content_frame = chart_frame

        ChartWidget, ChartStyle = _ensure_chart_module()
        chart = ChartWidget(
            chart_frame,
            style=ChartStyle(figure_size=(12, 6), dpi=100),
        )
        chart.pack(fill="both", expand=True, padx=15, pady=15)
        self._chart_widgets.append(chart)
        return chart

    def _on_drift_subtab_changed(self, value: str):
        """Switch between ML Drift and Process Drift sub-views."""
        if value == self._drift_subtab:
            return
        self._drift_subtab = value
        # Bump the load generation so any in-flight render of the
        # previous sub-view discards itself when its after() callback
        # fires — same protection pattern the top-level tab toggle uses.
        self._load_generation += 1
        if value == "ML Drift":
            self._show_drift_timeline()
        else:
            self._show_process_drift()

    def _on_drift_model_filter_changed(self, value: str):
        """Switch between All-models view and single-model dashboard."""
        new = None if value == "All models" else value
        if new == self._drift_filter_model:
            return
        self._drift_filter_model = new
        self._load_generation += 1
        # Render: pickup whichever sub-view is current (ignored when single
        # model; the single-model dashboard is rendered directly)
        if new is None:
            if self._drift_subtab == "ML Drift":
                self._show_drift_timeline()
            else:
                self._show_process_drift()
        else:
            self._show_single_model_drift()

    def _show_single_model_drift(self):
        """Single-model investigation dashboard.

        Layout:
          Header pills (model · status · days drifting · score · units)
          2×2 chart grid:
            [ Sigma Drift           | Untrimmed Resistance ]
            [ Electrical Angle      | Retrim Rate          ]
        """
        if not self._drift_filter_model:
            return
        model = self._drift_filter_model
        self.status_label.configure(text=f"Loading drift dashboard for {model}...")
        selected_days = self.selected_days
        # Capture generation so a stale render doesn't rebuild the content
        # frame after the user has rapidly clicked away from this model and
        # come back to a different one. Matches the pattern used by the
        # other drift renderers.
        self._load_generation += 1
        gen = self._load_generation

        def _load():
            try:
                from laser_trim_analyzer.ml import get_shared_ml_manager
                db = get_database()
                data = db.get_model_drift_dashboard(
                    model=model, days_back=selected_days
                )
                ml_manager = get_shared_ml_manager(db)
                detector = ml_manager.drift_detectors.get(model)
                self.after(0, lambda g=gen: self._render_single_model_drift(
                    data, detector, g
                ))
            except Exception as e:
                logger.error(f"Single-model drift error: {e}", exc_info=True)
                self.after(0, lambda exc=e: self.status_label.configure(
                    text=f"Drift dashboard error: {exc}"))

        get_thread_manager().start_thread(
            target=_load, name="single-model-drift",
        )

    def _render_single_model_drift(self, data: Dict[str, Any], detector, gen: int = 0):
        if not self.winfo_exists():
            return
        if gen != self._load_generation:
            return  # Superseded by a newer load (rapid model switch).
        if self._trend_type.get() != "Drift":
            return
        if self._drift_filter_model != data.get("model"):
            return
        try:
            self._create_drift_view()
            for w in self._drift_content_frame.winfo_children():
                w.destroy()

            model = data["model"]
            unit_count = data["unit_count"]

            # Header pill bar
            pills = ctk.CTkFrame(self._drift_content_frame, fg_color="transparent")
            pills.pack(fill="x", padx=10, pady=(8, 4))

            # Model name big
            ctk.CTkLabel(
                pills, text=model,
                font=ctk.CTkFont(size=18, weight="bold"),
            ).pack(side="left", padx=(0, 12))

            # Status badge
            is_drifting = bool(detector and detector.is_drifting)
            direction = (
                detector.drift_direction.value
                if detector and detector.drift_direction
                else None
            )
            if not detector or not detector.has_baseline:
                badge_text, badge_color = "○ no baseline", "gray"
            elif is_drifting and direction == "up":
                badge_text, badge_color = "↑ DRIFTING", "#ff8080"
            elif is_drifting:
                badge_text, badge_color = "↓ DRIFTING", "#ffb060"
            else:
                badge_text, badge_color = "✓ stable", "#7ed99e"
            ctk.CTkLabel(
                pills, text=badge_text, text_color=badge_color,
                font=ctk.CTkFont(size=11, weight="bold"),
            ).pack(side="left", padx=(0, 12))

            # Days drifting / score / units pills
            def _pill(text):
                f = ctk.CTkFrame(pills, fg_color=("gray85", "gray25"), corner_radius=4)
                f.pack(side="left", padx=4)
                ctk.CTkLabel(
                    f, text=text, font=ctk.CTkFont(size=10),
                ).pack(padx=8, pady=2)

            if detector and detector.has_baseline:
                cusum_value = max(detector.cusum_pos, detector.cusum_neg)
                _pill(f"score {cusum_value:.1f} / {detector.cusum_h:.1f}")
            _pill(f"{unit_count:,} units")

            # 2×2 chart grid
            ChartWidget, ChartStyle = _ensure_chart_module()
            chart = ChartWidget(
                self._drift_content_frame,
                style=ChartStyle(figure_size=(12, 7), dpi=100),
            )
            chart.pack(fill="both", expand=True, padx=10, pady=(8, 10))
            self._chart_widgets.append(chart)
            chart.clear()

            fig = chart.figure
            gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.25)

            # Compute the bucket index that marks the baseline → recent boundary.
            # Used as the x-coordinate for the orange vertical line on each panel.
            baseline_cutoff_iso = data.get("baseline_cutoff_date")

            process = data.get("process", {})

            # ---- Top-left: Sigma drift (hybrid SPC) ----
            ax_sigma = fig.add_subplot(gs[0, 0])
            chart._style_axis(ax_sigma)
            sigma_pts = data.get("sigma_series", [])

            if not sigma_pts:
                self._draw_empty_state(ax_sigma, "No data in window")
            elif not (detector and detector.has_baseline):
                self._draw_empty_state(
                    ax_sigma,
                    f"Need ≥30 baseline samples to enable SPC; "
                    f"have {len(sigma_pts)}. Train this model in Settings.",
                )
            else:
                violations = _draw_sigma_panel(
                    ax_sigma,
                    sigma_series=sigma_pts,
                    detector=detector,
                    baseline_cutoff_bucket_index=_cutoff_bucket_index(sigma_pts, baseline_cutoff_iso),
                )
                if violations > 0:
                    title_status = f"↑ OOC · {violations} violations"
                elif detector.is_drifting:
                    direction = "↑" if (
                        detector.drift_direction
                        and detector.drift_direction.value == "up"
                    ) else "↓"
                    title_status = f"{direction} DRIFT"
                else:
                    title_status = "✓ stable"
                ax_sigma.set_title(
                    f"Sigma Drift  ·  {title_status}",
                    loc="left", fontsize=11, fontweight="bold",
                )
            ax_sigma.tick_params(axis="x", rotation=0, labelsize=8)

            # ---- Top-right + bottom-left: continuous process panels ----
            metric_axes = (
                ("untrimmed_resistance", gs[0, 1], "Untrimmed Resistance"),
                ("measured_electrical_angle", gs[1, 0], "Electrical Angle"),
            )
            meta = self._db_metric_meta()
            for metric, slot, title in metric_axes:
                ax = fig.add_subplot(slot)
                chart._style_axis(ax)
                panel = process.get(metric, {})
                series = panel.get("series", [])
                if not series:
                    self._draw_empty_state(ax, f"No {title.lower()} data in window")
                    ax.tick_params(axis="x", rotation=0, labelsize=8)
                    continue

                buckets = _compute_buckets(series, n_per_bucket=50)
                z = panel.get("z_score") or 0.0
                if abs(z) >= _DRIFT_Z_ALARM:
                    color = _STATUS_ALARM_COLOR
                    pill = ("↑ OOC" if z > 0 else "↓ OOC")
                elif abs(z) >= _DRIFT_Z_WARN:
                    color = _STATUS_WARN_COLOR
                    pill = ("↑ DRIFT" if z > 0 else "↓ DRIFT")
                else:
                    color = _STATUS_STABLE_COLOR
                    pill = "✓ stable"

                _draw_smoothed_panel(
                    ax,
                    buckets=buckets,
                    baseline_mean=panel.get("baseline_mean"),
                    baseline_cutoff_bucket_index=_cutoff_bucket_index(series, baseline_cutoff_iso),
                    color=color,
                )

                fmt = meta.get(metric, {}).get("fmt", "{:.2f}")
                unit = meta.get(metric, {}).get("unit", "")
                base_mean = panel.get("baseline_mean")
                base_s = fmt.format(base_mean) if base_mean is not None else "—"
                rec_s = (
                    fmt.format(panel["recent_mean"])
                    if panel.get("recent_mean") is not None else "—"
                )
                pct = panel.get("delta_pct")
                pct_s = f"({pct:+.1f}%)" if pct is not None else ""
                z_s = f"z={z:+.1f}" if panel.get("z_score") is not None else ""
                ax.set_title(
                    f"{title}  ·  {pill}  ·  "
                    f"{base_s} → {rec_s} {unit} {pct_s}  {z_s}",
                    loc="left", fontsize=10, fontweight="bold",
                )
                ax.tick_params(axis="x", rotation=0, labelsize=8)

            # ---- Bottom-right: Retrim Rate ----
            ax_retrim = fig.add_subplot(gs[1, 1])
            chart._style_axis(ax_retrim)
            retrim_series = process.get("retrim_rate_series", [])

            if not retrim_series:
                self._draw_empty_state(
                    ax_retrim,
                    "trim_pass_count not captured for this window's rows. "
                    "Re-parse to populate.",
                )
            else:
                # Convert 0/1 → bucket retrim rate %.
                buckets = _compute_buckets(retrim_series, n_per_bucket=50)
                # Mean of 0/1 values * 100 = percentage. SE scales the same way.
                for b in buckets:
                    b["mean"] = b["mean"] * 100.0
                    b["stddev"] = b["stddev"] * 100.0
                    b["se"] = b["se"] * 100.0

                # baseline retrim rate from the rows before baseline_cutoff
                baseline_rate = None
                recent_rate = None
                if baseline_cutoff_iso:
                    cutoff = datetime.fromisoformat(baseline_cutoff_iso)
                    base_vals = [v for iso, v in retrim_series
                                 if datetime.fromisoformat(iso) < cutoff]
                    recent_vals = [v for iso, v in retrim_series
                                   if datetime.fromisoformat(iso) >= cutoff]
                    if base_vals:
                        baseline_rate = sum(base_vals) / len(base_vals) * 100.0
                    if recent_vals:
                        recent_rate = sum(recent_vals) / len(recent_vals) * 100.0

                # Status pill: rising if recent ≥ 2× baseline AND recent ≥ 10%
                if (baseline_rate is not None and recent_rate is not None
                        and recent_rate >= _RETRIM_RISING_FLOOR_PCT
                        and recent_rate >= _RETRIM_RISING_RATIO * max(baseline_rate, 1.0)):
                    pill = "↑ rising"
                    color = _STATUS_WARN_COLOR
                elif recent_rate is not None and recent_rate >= _RETRIM_OOC_PCT:
                    pill = "↑ OOC"
                    color = _STATUS_ALARM_COLOR
                else:
                    pill = "✓ stable"
                    color = _STATUS_STABLE_COLOR

                _draw_smoothed_panel(
                    ax_retrim,
                    buckets=buckets,
                    baseline_mean=baseline_rate,
                    baseline_cutoff_bucket_index=_cutoff_bucket_index(retrim_series, baseline_cutoff_iso),
                    color=color,
                )
                # Y-axis: 0% lower bound; upper bound max(20%, 1.5 × peak).
                peak = max((b["mean"] for b in buckets), default=0.0)
                ax_retrim.set_ylim(0.0, max(_RETRIM_Y_MIN_UPPER, 1.5 * peak))

                base_s = f"{baseline_rate:.1f}%" if baseline_rate is not None else "—"
                rec_s = f"{recent_rate:.1f}%" if recent_rate is not None else "—"
                delta_pp = (
                    f"({recent_rate - baseline_rate:+.1f} pp)"
                    if baseline_rate is not None and recent_rate is not None
                    else ""
                )
                ax_retrim.set_title(
                    f"Retrim Rate  ·  {pill}  ·  "
                    f"{base_s} → {rec_s}  {delta_pp}",
                    loc="left", fontsize=10, fontweight="bold",
                )
            ax_retrim.tick_params(axis="x", rotation=0, labelsize=8)

            try:
                fig.tight_layout()
            except Exception:
                pass
            chart.canvas.draw_idle()
            self.status_label.configure(text=f"Drift dashboard · {model}")
        except Exception as e:
            logger.error(f"Single-model drift render error: {e}", exc_info=True)
            self.status_label.configure(text=f"Drift dashboard error: {e}")

    def _draw_empty_state(self, ax, message: str) -> None:
        """Render a visible empty-state message centered on an Axes.

        Used when a chart can't be produced (no data, missing spec, etc.) so
        the user sees an explanation instead of a silent blank page.
        """
        ax.clear()
        ax.text(
            0.5, 0.5, message,
            ha='center', va='center',
            transform=ax.transAxes,
            fontsize=11,
            color='#adb5bd',
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _show_yield_trend(self):
        """Show overall yield trend across all models."""
        self.status_label.configure(text="Loading yield trend...")
        selected_days = self.selected_days
        # Pick aggregation period to keep ~10–60 buckets on the chart.
        # Weekly buckets over 10 years = 520 ticks, unreadable; monthly
        # buckets over 90 days = 3 ticks, also useless. Heuristic:
        #   ≤ 90 d  → daily
        #   ≤ 730 d → weekly
        #   > 730 d → monthly
        if selected_days <= 90:
            period = "day"
        elif selected_days <= 730:
            period = "week"
        else:
            period = "month"

        def _load():
            try:
                db = get_database()
                data = db.get_yield_trend(days_back=selected_days, period=period)
                self.after(0, lambda d=data, p=period: self._render_yield_trend(d, p))
            except Exception as e:
                logger.error(f"Yield trend error: {e}", exc_info=True)
                self.after(0, lambda exc=e: self.status_label.configure(
                    text=f"Yield trend error: {exc}"))

        get_thread_manager().start_thread(target=_load, name="yield-trend")

    def _render_yield_trend(self, data, period: str = "week"):
        """Render yield trend chart on the main thread."""
        if not self.winfo_exists():
            return
        try:
            chart = self._create_dedicated_chart_view("Yield Trend")
            fig = chart.figure
            fig.clear()
            ax = fig.add_subplot(111)
            chart._style_axis(ax)

            if not data:
                self._draw_empty_state(
                    ax,
                    "No yield data in the last 180 days.\n\n"
                    "Process some files to populate yield history.",
                )
                chart.canvas.draw_idle()
                self.status_label.configure(text="No yield data")
                return

            periods = [d["period"] for d in data]
            rates = [d["pass_rate"] for d in data]

            if len(periods) < 2:
                # Single bucket = noise, not a trend. Tell the user the
                # date filter / aggregation produced no usable trend
                # rather than drawing a misleading single point.
                self._draw_empty_state(
                    ax,
                    "Not enough data to draw a yield trend.\n\n"
                    "Try a wider date range, or check that\n"
                    "files have been processed in this window.",
                )
                chart.canvas.draw_idle()
                self.status_label.configure(text="Yield trend: insufficient data")
                return

            x = list(range(len(periods)))
            target = 80.0
            # Fill between the curve and the target so colour shows the gap
            # to spec, not absolute area-under-the-curve which exaggerated
            # 90% yield as a giant blue block (per usability review).
            above = [max(r, target) for r in rates]
            below = [min(r, target) for r in rates]
            ax.fill_between(x, target, above, color="#198754", alpha=0.25,
                            label="Above target")
            ax.fill_between(x, below, target, color="#dc3545", alpha=0.25,
                            label="Below target")
            ax.plot(x, rates, marker="o", markersize=3,
                    color="#0d6efd", linewidth=1.5, label="Yield")

            # Rolling-average overlay smooths out weekly/daily noise so
            # the underlying trend is visible at long date ranges. Window
            # is ~10% of the visible periods, capped 4–12 buckets.
            if len(rates) >= 6:
                w = max(4, min(12, len(rates) // 10))
                rolling = []
                for i in range(len(rates)):
                    lo = max(0, i - w + 1)
                    window_vals = [v for v in rates[lo:i + 1] if v is not None]
                    rolling.append(
                        sum(window_vals) / len(window_vals) if window_vals else None
                    )
                ax.plot(x, rolling, color="#fd7e14", linewidth=2.0,
                        label=f"Rolling avg ({w}{period[0]})")

            ax.axhline(y=target, color="#198754", linestyle="--",
                       alpha=0.7, label="Target (80%)")

            step = max(1, len(periods) // 10)
            ax.set_xticks(range(0, len(periods), step))
            ax.set_xticklabels(
                [periods[i] for i in range(0, len(periods), step)],
                rotation=45, fontsize=8,
            )
            ax.set_ylabel("Yield (%)")
            granularity = {"day": "daily", "week": "weekly",
                           "month": "monthly"}.get(period, period)
            ax.set_title(
                f"Overall Yield Trend — last {self.selected_days}d, "
                f"{granularity} buckets ({len(periods)} points)"
            )
            # Auto-scale y-axis around the data with breathing room rather
            # than 0–105% which wasted half the chart on empty space.
            data_min = min(r for r in rates if r is not None)
            data_max = max(r for r in rates if r is not None)
            pad = max(2.0, (data_max - data_min) * 0.1)
            ax.set_ylim(max(0, data_min - pad), min(105, data_max + pad))
            ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
            ax.grid(True, axis="y", alpha=0.2)
            try:
                fig.tight_layout()
            except Exception:
                pass
            chart.canvas.draw_idle()
            self.status_label.configure(
                text=f"Yield trend: {len(periods)} {granularity} buckets"
            )
        except Exception as e:
            logger.error(f"Yield trend error: {e}")
            self.status_label.configure(text=f"Yield trend error: {e}")

    def _show_drift_timeline(self):
        """Show ML Drift sortable model table (All-models view)."""
        if self._drift_filter_model is not None:
            # Single-model view takes over; don't render the table.
            return
        self.status_label.configure(text="Loading drift table...")
        selected_days = self.selected_days

        def _load():
            try:
                from laser_trim_analyzer.ml import get_shared_ml_manager
                db = get_database()
                state_by_model = db.get_drift_state_for_models(
                    days_back=selected_days
                )
                ml_manager = get_shared_ml_manager(db)
                # Combine DB state + in-memory CUSUM score per detector
                rows = []
                for model, state in state_by_model.items():
                    detector = ml_manager.drift_detectors.get(model)
                    has_baseline = detector is not None and detector.has_baseline
                    if detector is not None:
                        cusum_value = max(detector.cusum_pos, detector.cusum_neg)
                        cusum_h = detector.cusum_h
                    else:
                        cusum_value = None
                        cusum_h = None
                    days_drifting = None
                    if state["is_drifting"] and state["drift_start_date"]:
                        days_drifting = (
                            datetime.now() - state["drift_start_date"]
                        ).days
                    rows.append({
                        "model": model,
                        "has_baseline": has_baseline,
                        "is_drifting": state["is_drifting"],
                        "direction": state["direction"],
                        "drift_score": cusum_value,
                        "drift_threshold": cusum_h,
                        "drift_start_date": state["drift_start_date"],
                        "days_drifting": days_drifting,
                        "sigma_series": state["sigma_series"],
                    })
                self.after(0, lambda: self._render_drift_timeline(rows))
            except Exception as e:
                logger.error(f"Drift table error: {e}", exc_info=True)
                self.after(0, lambda: self.status_label.configure(
                    text=f"Drift error: {e}"))

        get_thread_manager().start_thread(target=_load, name="drift-table")

    def _render_drift_timeline(self, rows):
        """Render the ML Drift sortable model table on the main thread."""
        if not self.winfo_exists():
            return
        if (
            self._trend_type.get() != "Drift"
            or self._drift_subtab != "ML Drift"
            or self._drift_filter_model is not None
        ):
            return
        try:
            self._create_drift_view()  # rebuild header + content frame
            for w in self._drift_content_frame.winfo_children():
                w.destroy()

            if not rows:
                lbl = ctk.CTkLabel(
                    self._drift_content_frame,
                    text="No drift data yet — process more files or train ML in Settings.",
                    font=ctk.CTkFont(size=11),
                    text_color="gray",
                )
                lbl.pack(expand=True, padx=20, pady=20)
                self.status_label.configure(text="No drift data")
                return

            def render_status(parent, row):
                if not row["has_baseline"]:
                    return ctk.CTkLabel(
                        parent, text="○ no baseline",
                        text_color="gray",
                        font=ctk.CTkFont(size=10),
                    )
                if row["is_drifting"]:
                    if row["direction"] == "up":
                        return ctk.CTkLabel(
                            parent, text="↑ DRIFTING",
                            text_color="#ff8080",
                            font=ctk.CTkFont(size=10, weight="bold"),
                        )
                    return ctk.CTkLabel(
                        parent, text="↓ DRIFTING",
                        text_color="#ffb060",
                        font=ctk.CTkFont(size=10, weight="bold"),
                    )
                return ctk.CTkLabel(
                    parent, text="✓ stable",
                    text_color="#7ed99e",
                    font=ctk.CTkFont(size=10),
                )

            def render_score(parent, row):
                if row["drift_score"] is None:
                    text = "—"
                    color = "gray"
                else:
                    text = f"{row['drift_score']:.1f} / {row['drift_threshold']:.1f}"
                    color = (
                        "#ff8080" if row["is_drifting"] and row["direction"] == "up"
                        else "#ffb060" if row["is_drifting"]
                        else "#7ed99e"
                    )
                return ctk.CTkLabel(
                    parent, text=text, text_color=color,
                    font=ctk.CTkFont(size=10),
                )

            def render_last_event(parent, row):
                d = row["drift_start_date"]
                text = d.strftime("%Y-%m-%d") if (d and row["is_drifting"]) else "—"
                return ctk.CTkLabel(
                    parent, text=text,
                    font=ctk.CTkFont(size=10),
                )

            def render_days(parent, row):
                d = row["days_drifting"]
                text = f"{d}d" if d is not None else "—"
                return ctk.CTkLabel(
                    parent, text=text,
                    font=ctk.CTkFont(size=10),
                )

            def render_spark(parent, row):
                s = _Sparkline(parent, width=80, height=14)
                values = [v for _, v in row["sigma_series"]]
                color = (
                    "#ff8080" if row["is_drifting"] and row["direction"] == "up"
                    else "#ffb060" if row["is_drifting"]
                    else "#7ed99e"
                )
                s.draw(values, color=color)
                return s

            columns = [
                ("model", "Model", None),
                ("status_sort", "Status", render_status),
                ("score_sort", "Drift score", render_score),
                ("drift_start_date", "Last event", render_last_event),
                ("days_drifting", "Days drifting", render_days),
                ("model", "Sigma trend", render_spark),
            ]
            # Add sort-key fields
            for r in rows:
                # Drifting first (0), stable (1), no-baseline (2)
                if not r["has_baseline"]:
                    r["status_sort"] = 2
                elif r["is_drifting"]:
                    r["status_sort"] = 0
                else:
                    r["status_sort"] = 1
                r["score_sort"] = (
                    r["drift_score"] if r["drift_score"] is not None else -1
                )

            table = _SortableTable(
                self._drift_content_frame,
                columns=columns,
                rows=rows,
                row_click=lambda r: self._on_drift_row_click(r["model"]),
                default_sort_key="status_sort",
                default_sort_reverse=False,
            )
            table.pack(fill="both", expand=True, padx=10, pady=10)
            drifting_count = sum(1 for r in rows if r["is_drifting"])
            self.status_label.configure(
                text=f"Drift: {drifting_count} drifting / {len(rows)} models"
            )
        except Exception as e:
            logger.error(f"Drift table render error: {e}", exc_info=True)
            self.status_label.configure(text=f"Drift render error: {e}")

    def _on_drift_row_click(self, model: str):
        """Drill into a model's drift dashboard from a table row click."""
        self._drift_filter_model = model
        if hasattr(self, "_drift_model_filter"):
            self._drift_model_filter.set(model)
        self._show_single_model_drift()

    def _show_process_drift(self):
        """Show Process Drift table for the active metric (All-models view)."""
        if self._drift_filter_model is not None:
            return
        self.status_label.configure(text="Loading process drift...")
        selected_days = self.selected_days
        baseline_days = max(selected_days, 60)
        recent_days = max(7, min(28, baseline_days // 7))
        metric = self._process_drift_metric

        def _load():
            try:
                db = get_database()
                rows = db.get_process_drift_table(
                    metric=metric,
                    baseline_days=baseline_days,
                    recent_days=recent_days,
                )
                self.after(0, lambda: self._render_process_drift_table(rows))
            except Exception as e:
                logger.error(f"Process drift error: {e}", exc_info=True)
                self.after(0, lambda exc=e: self.status_label.configure(
                    text=f"Process drift error: {exc}"))

        get_thread_manager().start_thread(target=_load, name="process-drift-table")

    def _render_process_drift_table(self, rows):
        if not self.winfo_exists():
            return
        if (
            self._trend_type.get() != "Drift"
            or self._drift_subtab != "Process Drift"
            or self._drift_filter_model is not None
        ):
            return
        try:
            self._create_drift_view()
            for w in self._drift_content_frame.winfo_children():
                w.destroy()

            # Metric tab strip
            tabs = ctk.CTkFrame(self._drift_content_frame, fg_color="transparent")
            tabs.pack(fill="x", padx=10, pady=(8, 0))
            metric_options = [
                ("untrimmed_resistance", "Untrimmed Resistance"),
                ("measured_electrical_angle", "Electrical Angle"),
                ("trim_pass_count", "Trim Passes"),
            ]
            for mkey, mlabel in metric_options:
                is_active = mkey == self._process_drift_metric
                btn = ctk.CTkButton(
                    tabs,
                    text=mlabel,
                    width=130,
                    height=24,
                    fg_color=("#0d6efd" if is_active else ("gray85", "gray25")),
                    hover_color=("#3b8eff" if is_active else ("gray75", "gray35")),
                    text_color=("white" if is_active else None),
                    font=ctk.CTkFont(size=10, weight=("bold" if is_active else "normal")),
                    command=lambda k=mkey: self._on_process_metric_changed(k),
                )
                btn.pack(side="left", padx=(0, 4))

            if not rows:
                lbl = ctk.CTkLabel(
                    self._drift_content_frame,
                    text="No models meet the baseline+recent thresholds for this metric.",
                    font=ctk.CTkFont(size=11),
                    text_color="gray",
                )
                lbl.pack(expand=True, padx=20, pady=20)
                self.status_label.configure(text="Process drift: no rows")
                return

            meta = self._db_metric_meta().get(self._process_drift_metric, {})
            unit = meta.get("unit", "")
            fmt = meta.get("fmt", "{:.2f}")

            def render_baseline(parent, row):
                txt = f"{fmt.format(row['baseline_mean'])} {unit}".strip()
                return ctk.CTkLabel(parent, text=txt, font=ctk.CTkFont(size=10))

            def render_recent(parent, row):
                txt = f"{fmt.format(row['recent_mean'])} {unit}".strip()
                return ctk.CTkLabel(parent, text=txt, font=ctk.CTkFont(size=10))

            def render_delta_pct(parent, row):
                v = row["delta_pct"]
                color = (
                    "#ff8080" if abs(row["z_score"]) >= 3.0
                    else "#ffb060" if abs(row["z_score"]) >= 2.0
                    else "#7ed99e"
                )
                return ctk.CTkLabel(
                    parent, text=f"{v:+.1f}%",
                    text_color=color,
                    font=ctk.CTkFont(size=10),
                )

            def render_z(parent, row):
                z = row["z_score"]
                color = (
                    "#ff8080" if abs(z) >= 3.0
                    else "#ffb060" if abs(z) >= 2.0
                    else "#7ed99e"
                )
                return ctk.CTkLabel(
                    parent, text=f"{z:+.1f}",
                    text_color=color,
                    font=ctk.CTkFont(size=10),
                )

            def render_trend(parent, row):
                s = _Sparkline(parent, width=80, height=14)
                values = [v for _, v in row["series"]]
                color = (
                    "#ff8080" if abs(row["z_score"]) >= 3.0
                    else "#ffb060" if abs(row["z_score"]) >= 2.0
                    else "#7ed99e"
                )
                s.draw(values, color=color)
                return s

            columns = [
                ("model", "Model", None),
                ("baseline_mean", "Baseline", render_baseline),
                ("recent_mean", "Recent", render_recent),
                ("delta_pct", "Δ%", render_delta_pct),
                ("z_score", "z", render_z),
                ("model", "Trend", render_trend),
            ]
            # Sort by |z| desc — but SortableTable sorts on raw values, so
            # add an absolute-z field for default sort.
            for r in rows:
                r["abs_z"] = abs(r["z_score"])
            columns.insert(0, ("abs_z", "|z|", None))

            table = _SortableTable(
                self._drift_content_frame,
                columns=columns,
                rows=rows,
                row_click=lambda r: self._on_drift_row_click(r["model"]),
                default_sort_key="abs_z",
                default_sort_reverse=True,
            )
            table.pack(fill="both", expand=True, padx=10, pady=10)
            drifting_count = sum(1 for r in rows if r["is_drifting"])
            self.status_label.configure(
                text=f"Process drift ({self._process_drift_metric}): "
                     f"{drifting_count} drifting / {len(rows)} models"
            )
        except Exception as e:
            logger.error(f"Process drift render error: {e}", exc_info=True)
            self.status_label.configure(text=f"Process drift render error: {e}")

    def _on_process_metric_changed(self, metric: str):
        if metric == self._process_drift_metric:
            return
        self._process_drift_metric = metric
        self._show_process_drift()

    @staticmethod
    def _db_metric_meta():
        """Cached lookup for axis-unit metadata (kept here so the GUI
        doesn't have to import the DB module's private constant)."""
        from laser_trim_analyzer.database.manager import DatabaseManager
        return DatabaseManager._PROCESS_DRIFT_METRICS

    def _show_trim_difficulty(self):
        """Show models ranked by how many laser-trim passes the equipment
        runs per unit on average. Higher = harder unit, more retrim work."""
        self.status_label.configure(text="Loading trim difficulty...")
        selected_days = self.selected_days

        def _load():
            try:
                db = get_database()
                rows = db.get_trim_difficulty_by_model(
                    days_back=selected_days, min_units=5, limit=25
                )
                self.after(0, lambda: self._render_trim_difficulty(rows))
            except Exception as e:
                logger.error(f"Trim difficulty error: {e}", exc_info=True)
                self.after(0, lambda exc=e: self.status_label.configure(
                    text=f"Trim difficulty error: {exc}"))

        get_thread_manager().start_thread(target=_load, name="trim-difficulty")

    def _render_trim_difficulty(self, rows):
        """Horizontal bar chart of avg trim passes per model, worst at top.
        Bars are colored by retrim rate (red = many units needed retrimming).
        Each bar is annotated with N units and max passes seen."""
        if not self.winfo_exists():
            return
        if self._trend_type.get() != "Trim Difficulty":
            return  # User switched tabs; don't destroy the new tab's frames
        try:
            chart = self._create_dedicated_chart_view("Trim Difficulty by Model")
            chart.clear()
            fig = chart.figure
            ax = fig.add_subplot(111)
            chart._style_axis(ax)

            if not rows:
                self._draw_empty_state(
                    ax,
                    "No trim difficulty data yet.\n\n"
                    "trim_pass_count is captured at parse time —\n"
                    "process new files (or reanalyze existing ones)\n"
                    "to populate this view.",
                )
                chart.canvas.draw_idle()
                self.status_label.configure(text="No trim difficulty data")
                return

            # Reverse so highest avg sits at the top of the horizontal bar chart.
            rows_top_first = list(reversed(rows))
            models = [r["model"] for r in rows_top_first]
            avgs = [r["avg_passes"] for r in rows_top_first]
            retrim_rates = [r["retrim_rate"] for r in rows_top_first]

            # Color bars by retrim rate. 0% retrim = green, 50%+ = red.
            def _color(rate):
                if rate < 10:
                    return "#27ae60"   # green: easy
                if rate < 25:
                    return "#f1c40f"   # yellow: occasional retrim
                if rate < 50:
                    return "#e67e22"   # orange: frequent retrim
                return "#e74c3c"       # red: most units retrimmed

            colors = [_color(r) for r in retrim_rates]

            target_height = max(4.0, min(14.0, 1.5 + 0.35 * len(models)))
            fig.set_size_inches(fig.get_size_inches()[0], target_height, forward=True)

            y_pos = list(range(len(models)))
            ax.barh(y_pos, avgs, color=colors, edgecolor="#1a1a1a", linewidth=0.5)

            # Annotate each bar with sample size, max passes, retrim rate,
            # and (when available) average max-error-reduction. The last
            # field distinguishes models where extra trim passes are
            # genuinely fixing outcomes (high Δ) from models where extra
            # passes don't help (low Δ — process root-cause issue).
            for i, r in enumerate(rows_top_first):
                aer = r.get("avg_error_reduction")
                aer_part = f" · avg Δ {aer:.0f}%" if aer is not None else ""
                ax.text(
                    avgs[i] + 0.05,
                    i,
                    f"{r['count']} units · max {r['max_passes']} · "
                    f"retrim {r['retrim_rate']:.0f}%{aer_part}",
                    va="center",
                    fontsize=8,
                    color="#cccccc",
                )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(models, fontsize=9)
            ax.set_xlabel("Avg trim passes per unit")
            ax.set_title(
                f"Trim Difficulty by Model — last {self.selected_days} days "
                f"(top {len(models)} hardest)"
            )
            ax.axvline(x=1.0, linestyle="--", color="#888", linewidth=0.8, alpha=0.6)
            ax.grid(True, axis="x", alpha=0.2)
            try:
                fig.tight_layout()
            except Exception:
                pass
            chart.canvas.draw_idle()
            self.status_label.configure(
                text=f"Trim difficulty: {len(rows)} models ranked"
            )
        except Exception as e:
            logger.error(f"Trim difficulty render error: {e}", exc_info=True)
            self.status_label.configure(text=f"Trim difficulty error: {e}")

    def _populate_spec_filters(self):
        """Populate element type and product class filter dropdowns.

        DB queries run on a background thread to avoid freezing the UI during
        page navigation; widget updates are marshaled back to the main thread
        via self.after() because tkinter is not thread-safe.
        """
        def _load_spec_values():
            try:
                db = get_database()
                etypes = ["All"] + db.get_distinct_element_types()
                pclasses = ["All"] + db.get_distinct_product_classes()
                self.after(0, lambda: self._apply_spec_filter_values(etypes, pclasses))
            except Exception as e:
                logger.debug(f"Could not populate spec filters: {e}")

        get_thread_manager().start_thread(
            target=_load_spec_values,
            name="trends-populate-spec-filters",
        )

    def _apply_spec_filter_values(self, etypes, pclasses):
        """Main-thread callback to apply dropdown values."""
        try:
            self._element_filter.configure(values=etypes)
            self._class_filter.configure(values=pclasses)
        except Exception as e:
            logger.debug(f"Could not apply spec filter values: {e}")

    def _on_spec_filter_change(self, _=None):
        """Handle element type or product class filter change."""
        self._refresh_data()

    def on_hide(self):
        """Called when page becomes hidden."""
        # Don't destroy charts on hide — recreating 10 ChartWidgets on every
        # page visit is the #1 performance bottleneck. Instead, keep them alive
        # and just clear stale data. Charts are only destroyed on model switch
        # (via _create_summary_view / _create_detail_view) which already calls
        # _cleanup_charts internally.
        self.active_models_data = []
        self.model_trend_data = None

    def _export_summary_pdf(self):
        """Export summary view to PDF."""
        from tkinter import filedialog, messagebox
        from laser_trim_analyzer.export.trends_pdf import export_trends_summary_pdf

        if not self.active_models_data:
            logger.warning("No summary data to export")
            messagebox.showwarning("No Data", "No summary data available to export.")
            return

        # File dialog
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"trends_summary_{timestamp}.pdf"

        file_path = filedialog.asksaveasfilename(
            title="Export Summary to PDF",
            defaultextension=".pdf",
            initialfile=default_name,
            initialdir=getattr(self.app.config, 'export_path', None),
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            export_trends_summary_pdf(self.active_models_data, Path(file_path))
            logger.info(f"Exported summary PDF to: {file_path}")
            messagebox.showinfo("Export Complete", f"PDF exported successfully to:\n{file_path}")
        except Exception as e:
            logger.error(f"Failed to export summary PDF: {e}", exc_info=True)
            messagebox.showerror("Export Failed", f"Failed to export PDF:\n{str(e)}")

    def _export_detail_pdf(self):
        """Export detail view to PDF."""
        from tkinter import filedialog, messagebox
        from laser_trim_analyzer.export.trends_pdf import export_trends_detail_pdf

        if not self.model_trend_data or not self.selected_model:
            logger.warning("No model data to export")
            messagebox.showwarning("No Data", "No model data available to export.")
            return

        # Gather stats from UI labels
        model_stats = {
            key: label.cget("text")
            for key, label in self.detail_stat_labels.items()
        }

        # File dialog
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_safe = self.selected_model.replace(" ", "_")
        default_name = f"trends_{model_safe}_{timestamp}.pdf"

        file_path = filedialog.asksaveasfilename(
            title="Export Model Trends to PDF",
            defaultextension=".pdf",
            initialfile=default_name,
            initialdir=getattr(self.app.config, 'export_path', None),
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            export_trends_detail_pdf(
                self.selected_model,
                self.model_trend_data,
                model_stats,
                Path(file_path)
            )
            logger.info(f"Exported detail PDF to: {file_path}")
            messagebox.showinfo("Export Complete", f"PDF exported successfully to:\n{file_path}")
        except Exception as e:
            logger.error(f"Failed to export detail PDF: {e}", exc_info=True)
            messagebox.showerror("Export Failed", f"Failed to export PDF:\n{str(e)}")
