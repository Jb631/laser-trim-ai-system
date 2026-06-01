# Spec 3c — Model Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Model placeholder page with per-model investigation view: focus chart on top, 8-pill glance row, 3 tabs (Drift Metrics / Smoothness / Units). Header: model search dropdown + time-window picker + Export Evidence Pack button. Optional collapsible Predictor panel.

**Architecture:** Largest sub-spec. Reuses Spec 3a's chrome + Spec 2's drift API + existing `gui/widgets/chart.py`. Composition:
- `ModelPage` (PageBase)
- `MetricPillRow` widget — 8 clickable pills
- `FocusChart` widget — wraps the existing ChartWidget with baseline/recent overlays
- `DriftMetricsTab` / `SmoothnessTab` / `UnitsTab` widgets
- `PredictorPanel` (collapsible)

**Tech Stack:** Python 3.x, customtkinter, matplotlib (via existing ChartWidget), pytest. Depends on Specs 1, 2, 3a, 3b.

**Target branch:** `V6` only. Latest commit before starting: the Spec 3b final commit.

**Spec reference:** `docs/superpowers/specs/2026-05-30-spec3-ui-shell-design.md` (Sub-spec 3c section).

---

## File Structure

**Files created:**
- `src/laser_trim_analyzer/gui/v6/widgets/metric_pill_row.py` — pill bar
- `src/laser_trim_analyzer/gui/v6/widgets/focus_chart.py` — wraps ChartWidget
- `src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py`
- `src/laser_trim_analyzer/gui/v6/widgets/smoothness_tab.py`
- `src/laser_trim_analyzer/gui/v6/widgets/units_tab.py`
- `src/laser_trim_analyzer/gui/v6/widgets/predictor_panel.py`
- `src/laser_trim_analyzer/gui/v6/widgets/tab_view.py` — wraps CTkTabview with theme styling
- `src/laser_trim_analyzer/gui/v6/pages/model_page.py`
- `tests/test_spec3c_model.py`

**Files modified:**
- `src/laser_trim_analyzer/ml/manager.py` — extend routing API: `consume_model_route_with_focus` for focus-metric hint
- `src/laser_trim_analyzer/gui/v6/app.py` — replace ModelPlaceholder with real `ModelPage`

---

## Task 1: Focus-metric routing hint extension

Extend `V6App.set_model_route` to optionally include a focus_metric. Add `consume_model_route_full()` that returns `(model, focus_metric)`.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/v6/app.py`
- Test: `tests/test_spec3c_model.py` (CREATE)

- [ ] **Step 1: Create test file with routing tests**

Create `tests/test_spec3c_model.py`:

```python
"""Spec 3c — Model page."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def tk_root():
    import customtkinter as ctk
    root = ctk.CTk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Task 1: Routing hint with focus metric
# ---------------------------------------------------------------------------


def test_consume_model_route_full_returns_tuple(tmp_path):
    """set_model_route(model, focus) → consume_model_route_full returns the pair."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "route.db"
    app = V6App(cfg)
    try:
        app.withdraw()
        app.set_model_route("M1", "linearity_error")
        assert app.consume_model_route_full() == ("M1", "linearity_error")
        # Already consumed
        assert app.consume_model_route_full() == (None, None)
    finally:
        app.destroy()


def test_consume_model_route_full_without_focus(tmp_path):
    """When focus is not set, the tuple's second slot is None."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "route2.db"
    app = V6App(cfg)
    try:
        app.withdraw()
        app.set_model_route("M2")
        assert app.consume_model_route_full() == ("M2", None)
    finally:
        app.destroy()
```

- [ ] **Step 2: Add the method to V6App**

In `src/laser_trim_analyzer/gui/v6/app.py`, in the `V6App` class, after `consume_model_route`:

```python
    def consume_model_route_full(self) -> Tuple[Optional[str], Optional[str]]:
        """Pop the full routing hint as (model, focus_metric).

        Used by the Model page on show.  Either component may be None.
        """
        if self._model_route is None:
            return (None, None)
        result = self._model_route
        self._model_route = None
        return result
```

Add `from typing import Tuple` to the imports if not already there.

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_spec3c_model.py -v -k consume_model_route
```

Expected: 2 PASS.

```bash
git add src/laser_trim_analyzer/gui/v6/app.py tests/test_spec3c_model.py
git commit -m "feat(spec3c): consume_model_route_full returns (model, focus)"
```

---

## Task 2: ThemedTabView wrapper

CustomTkinter's `CTkTabview` doesn't honor our ThemeManager colors. Wrap it with a thin class that applies the theme.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/tab_view.py`
- Test: `tests/test_spec3c_model.py` (APPEND)

- [ ] **Step 1: Append the tab view tests**

```python
# ---------------------------------------------------------------------------
# Task 2: ThemedTabView
# ---------------------------------------------------------------------------


def test_themed_tab_view_holds_tabs(tk_root):
    """ThemedTabView.add('name') returns a frame the caller fills."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.tab_view import ThemedTabView

    tv = ThemedTabView(tk_root, theme=ThemeManager())
    drift_frame = tv.add("Drift Metrics")
    smoothness_frame = tv.add("Smoothness")
    assert drift_frame is not None
    assert smoothness_frame is not None


def test_themed_tab_view_set_selects_tab(tk_root):
    """set('name') makes the named tab visible."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.tab_view import ThemedTabView

    tv = ThemedTabView(tk_root, theme=ThemeManager())
    tv.add("A")
    tv.add("B")
    tv.set("B")
    assert tv.get() == "B"
```

- [ ] **Step 2: Create the wrapper**

Create `src/laser_trim_analyzer/gui/v6/widgets/tab_view.py`:

```python
"""Spec 3c — ThemedTabView.

Thin wrapper around CTkTabview that applies V6 theme colors and font.
"""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class ThemedTabView(ctk.CTkTabview):
    """CTkTabview with V6 theme tokens applied."""

    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(
            master,
            fg_color=theme.SURFACE,
            segmented_button_fg_color=theme.CARD,
            segmented_button_selected_color=theme.ACCENT,
            segmented_button_selected_hover_color=theme.ACCENT_HOVER,
            segmented_button_unselected_color=theme.CARD,
            segmented_button_unselected_hover_color=theme.ELEVATED,
            text_color=theme.TEXT_PRIMARY,
            corner_radius=theme.RADIUS_MD,
            **kwargs,
        )
        self.theme = theme
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_spec3c_model.py -v -k themed_tab_view
```

Expected: 2 PASS.

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/tab_view.py tests/test_spec3c_model.py
git commit -m "feat(spec3c): ThemedTabView wrapper for CTkTabview"
```

---

## Task 3: MetricPillRow

Horizontal row of 8 clickable pills, one per watched metric. Each pill shows: metric name + one-line summary + tier color. Click → emits `on_pill_click(metric_name)`. Selected pill has accent border.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/metric_pill_row.py`
- Test: `tests/test_spec3c_model.py` (APPEND)

- [ ] **Step 1: Append pill tests**

```python
# ---------------------------------------------------------------------------
# Task 3: MetricPillRow
# ---------------------------------------------------------------------------


def _build_status(model="M1"):
    """Helper: build a ModelDriftStatus with all 8 metrics in Stable."""
    from datetime import datetime
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, MetricStatus, ModelDriftStatus, WATCHED_METRICS,
    )

    per_metric = {
        m: MetricStatus(
            metric=m, tier=DriftTier.STABLE, alert_type=None,
            magnitude=0.0, baseline_mean=0.0, baseline_std=1.0,
            recent_mean=0.0, recent_count=0, is_trained=True,
        ) for m in WATCHED_METRICS
    }
    return ModelDriftStatus(
        model=model, overall_tier=DriftTier.STABLE,
        worst_metric=None, worst_alert_type=None,
        per_metric=per_metric, last_processed=datetime.now(),
    )


def test_metric_pill_row_renders_eight_pills(tk_root):
    """One pill per WATCHED_METRICS entry."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import (
        MetricPillRow,
    )
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS

    row = MetricPillRow(
        tk_root, theme=ThemeManager(), on_pill_click=lambda m: None,
    )
    row.set_status(_build_status())
    assert len(row._pills) == len(WATCHED_METRICS)


def test_metric_pill_row_click_emits_metric_name(tk_root):
    """Clicking a pill emits on_pill_click(metric)."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import (
        MetricPillRow,
    )

    received: list[str] = []
    row = MetricPillRow(
        tk_root, theme=ThemeManager(),
        on_pill_click=lambda m: received.append(m),
    )
    row.set_status(_build_status())
    row._pills["sigma_gradient"]._on_click()
    assert received == ["sigma_gradient"]


def test_metric_pill_row_set_selected_highlights_pill(tk_root):
    """set_selected('name') marks that pill as selected."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import (
        MetricPillRow,
    )

    row = MetricPillRow(
        tk_root, theme=ThemeManager(), on_pill_click=lambda m: None,
    )
    row.set_status(_build_status())
    row.set_selected("linearity_error")
    assert row._selected_metric == "linearity_error"
```

- [ ] **Step 2: Create MetricPillRow**

Create `src/laser_trim_analyzer/gui/v6/widgets/metric_pill_row.py`:

```python
"""Spec 3c — MetricPillRow."""
from typing import Callable, Dict, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import (
    DriftTier, ModelDriftStatus, WATCHED_METRICS,
)


class MetricPillRow(ctk.CTkFrame):
    """Horizontal row of 8 clickable metric pills."""

    def __init__(
        self,
        master,
        theme: ThemeManager,
        on_pill_click: Callable[[str], None],
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_pill_click = on_pill_click
        self._pills: Dict[str, "_Pill"] = {}
        self._selected_metric: Optional[str] = None

        for metric in WATCHED_METRICS:
            pill = _Pill(
                self, metric=metric, theme=theme,
                on_click=self._handle_click,
            )
            pill.pack(side="left", padx=(0, theme.SPACE_SM))
            self._pills[metric] = pill

    def set_status(self, status: ModelDriftStatus) -> None:
        """Update each pill from its MetricStatus."""
        for metric, pill in self._pills.items():
            metric_status = status.per_metric.get(metric)
            if metric_status is not None:
                pill.set_metric_status(metric_status)

    def set_selected(self, metric: str) -> None:
        """Visually mark the named pill as selected; clear others."""
        if self._selected_metric == metric:
            return
        if self._selected_metric and self._selected_metric in self._pills:
            self._pills[self._selected_metric].set_selected(False)
        if metric in self._pills:
            self._pills[metric].set_selected(True)
            self._selected_metric = metric

    def _handle_click(self, metric: str) -> None:
        self._on_pill_click(metric)


class _Pill(ctk.CTkFrame):
    """One metric pill."""

    def __init__(
        self,
        master,
        metric: str,
        theme: ThemeManager,
        on_click: Callable[[str], None],
    ):
        super().__init__(
            master, fg_color=theme.CARD,
            corner_radius=theme.RADIUS_MD,
            border_width=2, border_color=theme.CARD,
        )
        self.metric = metric
        self.theme = theme
        self._on_click_external = on_click
        self._selected = False

        self._name_label = ctk.CTkLabel(
            self, text=metric,
            font=(theme.FONT_FAMILY[0], theme.SIZE_CAPTION, "bold"),
            text_color=theme.TEXT_PRIMARY,
        )
        self._name_label.pack(
            side="top", padx=theme.SPACE_SM, pady=(theme.SPACE_SM, 0),
        )
        self._summary_label = ctk.CTkLabel(
            self, text="—",
            font=(theme.FONT_FAMILY[0], theme.SIZE_CAPTION),
            text_color=theme.TEXT_SECONDARY,
        )
        self._summary_label.pack(
            side="top", padx=theme.SPACE_SM, pady=(0, theme.SPACE_SM),
        )

        for w in (self, self._name_label, self._summary_label):
            w.bind("<Button-1>", lambda e: self._on_click())

    def set_metric_status(self, status) -> None:
        """Recolor + relabel from a MetricStatus."""
        bg, fg = self.theme.tier_color(status.tier)
        self.configure(fg_color=bg)
        if not self._selected:
            self.configure(border_color=bg)
        if status.tier == DriftTier.STABLE:
            text = "OK"
        else:
            text = f"{status.magnitude:+.1f}σ"
        self._summary_label.configure(text=text, text_color=fg)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        if selected:
            self.configure(border_color=self.theme.ACCENT)
        else:
            # restore to tier bg color
            self.configure(border_color=self.cget("fg_color"))

    def _on_click(self) -> None:
        self._on_click_external(self.metric)
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_spec3c_model.py -v -k metric_pill_row
```

Expected: 3 PASS.

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/metric_pill_row.py tests/test_spec3c_model.py
git commit -m "feat(spec3c): MetricPillRow with 8 clickable tier-colored pills"
```

---

## Task 4: FocusChart widget

Wraps the existing `gui/widgets/chart.py` `ChartWidget`. Renders the selected metric's time series with baseline mean line + ±2σ band + recent-batch highlight.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/focus_chart.py`
- Test: `tests/test_spec3c_model.py` (APPEND)

- [ ] **Step 1: Append focus chart test**

```python
# ---------------------------------------------------------------------------
# Task 4: FocusChart
# ---------------------------------------------------------------------------


def test_focus_chart_set_series_does_not_crash(tk_root):
    """set_series() with synthetic data renders without raising."""
    from datetime import datetime, timedelta
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    chart = FocusChart(tk_root, theme=ThemeManager())
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(20, 0, -1)]
    values = [0.01 + 0.0001 * i for i in range(20)]
    # No raise expected
    chart.set_series(
        metric="sigma_gradient",
        dates=dates,
        values=values,
        baseline_mean=0.011,
        baseline_std=0.0005,
    )
```

- [ ] **Step 2: Create FocusChart**

Create `src/laser_trim_analyzer/gui/v6/widgets/focus_chart.py`:

```python
"""Spec 3c — FocusChart.

One large time-series chart for the currently-selected metric.  Overlays:
baseline mean line, baseline ± 2σ band, recent-batch highlight.

Wraps the existing gui/widgets/chart.py.ChartWidget to inherit its
matplotlib infrastructure.  This file owns the styling decisions
specific to drift visualization.
"""
from datetime import datetime
from typing import List, Optional

import customtkinter as ctk
import matplotlib
matplotlib.use("Agg")  # safe default; CTk creates its own backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class FocusChart(ctk.CTkFrame):
    """Time-series chart for one drift metric."""

    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(
            master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD,
            **kwargs,
        )
        self.theme = theme

        # Figure + canvas
        self._fig = Figure(figsize=(8, 3), dpi=96, facecolor=theme.CARD)
        self._ax = self._fig.add_subplot(111)
        self._style_axes()

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(
            fill="both", expand=True,
            padx=theme.SPACE_SM, pady=theme.SPACE_SM,
        )

    def _style_axes(self) -> None:
        ax = self._ax
        ax.set_facecolor(self.theme.CARD)
        ax.spines["bottom"].set_color(self.theme.TEXT_SECONDARY)
        ax.spines["left"].set_color(self.theme.TEXT_SECONDARY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=self.theme.TEXT_SECONDARY, labelsize=9)
        ax.xaxis.label.set_color(self.theme.TEXT_SECONDARY)
        ax.yaxis.label.set_color(self.theme.TEXT_SECONDARY)
        ax.title.set_color(self.theme.TEXT_PRIMARY)

    def set_series(
        self,
        metric: str,
        dates: List[datetime],
        values: List[float],
        baseline_mean: Optional[float] = None,
        baseline_std: Optional[float] = None,
        recent_batch_start: Optional[datetime] = None,
    ) -> None:
        """Replace the chart contents with a new series."""
        ax = self._ax
        ax.clear()
        self._style_axes()

        if not dates or not values:
            ax.text(
                0.5, 0.5, "No data",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.theme.TEXT_SECONDARY,
            )
            self._canvas.draw_idle()
            return

        # Main series
        ax.plot(
            dates, values, marker="o", markersize=3,
            linestyle="-", linewidth=1,
            color=self.theme.ACCENT,
            label=metric,
        )

        # Baseline overlays
        if baseline_mean is not None:
            ax.axhline(
                baseline_mean, color=self.theme.TEXT_SECONDARY,
                linestyle="--", linewidth=1, label="Baseline",
            )
            if baseline_std is not None:
                ax.axhspan(
                    baseline_mean - 2 * baseline_std,
                    baseline_mean + 2 * baseline_std,
                    color=self.theme.TEXT_SECONDARY,
                    alpha=0.12,
                )

        # Recent-batch highlight
        if recent_batch_start is not None:
            ax.axvspan(
                recent_batch_start, dates[-1],
                color=self.theme.ACCENT, alpha=0.10,
            )

        ax.set_title(metric)
        self._fig.tight_layout()
        self._canvas.draw_idle()
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_spec3c_model.py -v -k focus_chart
```

Expected: 1 PASS.

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/focus_chart.py tests/test_spec3c_model.py
git commit -m "feat(spec3c): FocusChart with baseline + recent-batch overlays"
```

---

## Task 5: DriftMetricsTab + SmoothnessTab + UnitsTab + PredictorPanel

Four tab-content widgets. Per the design spec:

- **DriftMetricsTab** — table view of all 8 metrics: metric / tier / alert type / baseline / recent / Δσ. Row selection mirrors the pill row.
- **SmoothnessTab** — embeds a small FocusChart-style chart of `max_smoothness_value` over time + a list of recent smoothness tests.
- **UnitsTab** — sortable table of recent units in the time window. Click row → opens existing per-unit chart in a modal. "Export to Excel" button.
- **PredictorPanel** — collapsible. Default collapsed. Shows existing per-unit predictor output marked "diagnostic — not part of daily flow."

**Files:**
- Create: 4 files in `gui/v6/widgets/`
- Test: `tests/test_spec3c_model.py` (APPEND)

For each widget, follow the pattern below (replacing names + content):

### DriftMetricsTab

- [ ] **Step 1: Append test**

```python
# ---------------------------------------------------------------------------
# Task 5a: DriftMetricsTab
# ---------------------------------------------------------------------------


def test_drift_metrics_tab_renders_row_per_metric(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import (
        DriftMetricsTab,
    )
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS

    tab = DriftMetricsTab(
        tk_root, theme=ThemeManager(), on_metric_select=lambda m: None,
    )
    tab.set_status(_build_status())
    assert set(tab._rows.keys()) == set(WATCHED_METRICS)


def test_drift_metrics_tab_row_click_emits_metric(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import (
        DriftMetricsTab,
    )

    received: list[str] = []
    tab = DriftMetricsTab(
        tk_root, theme=ThemeManager(),
        on_metric_select=lambda m: received.append(m),
    )
    tab.set_status(_build_status())
    tab._rows["sigma_gradient"]._on_click()
    assert received == ["sigma_gradient"]
```

- [ ] **Step 2: Create the widget**

Create `src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py`:

```python
"""Spec 3c — DriftMetricsTab: table of all 8 metrics."""
from typing import Callable, Dict

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import (
    AlertType, DriftTier, ModelDriftStatus, WATCHED_METRICS,
)


_COLUMNS = ["Metric", "Tier", "Alert", "Baseline", "Recent", "Δσ"]


class DriftMetricsTab(ctk.CTkScrollableFrame):
    """Tabular view of per-metric drift status."""

    def __init__(
        self,
        master,
        theme: ThemeManager,
        on_metric_select: Callable[[str], None],
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_metric_select = on_metric_select
        self._rows: Dict[str, "_MetricRow"] = {}

        # Header row
        header = ctk.CTkFrame(self, fg_color=theme.CARD)
        header.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        for col in _COLUMNS:
            label = ctk.CTkLabel(
                header, text=col,
                font=(theme.FONT_FAMILY[0], theme.SIZE_CAPTION, "bold"),
                text_color=theme.TEXT_SECONDARY,
            )
            label.pack(
                side="left", expand=True, fill="x",
                padx=theme.SPACE_SM, pady=theme.SPACE_XS,
            )

    def set_status(self, status: ModelDriftStatus) -> None:
        # Clear existing
        for row in self._rows.values():
            row.destroy()
        self._rows.clear()
        # Build a row per metric
        for metric in WATCHED_METRICS:
            ms = status.per_metric.get(metric)
            if ms is None:
                continue
            row = _MetricRow(
                self, metric_status=ms, theme=self.theme,
                on_click=self._on_metric_select,
            )
            row.pack(side="top", fill="x", pady=1)
            self._rows[metric] = row


class _MetricRow(ctk.CTkFrame):
    """One row in the drift metrics table."""

    def __init__(self, master, metric_status, theme: ThemeManager,
                 on_click: Callable[[str], None]):
        bg, fg = theme.tier_color(metric_status.tier)
        super().__init__(master, fg_color=bg)
        self.theme = theme
        self.metric = metric_status.metric
        self._on_click_external = on_click

        cells = [
            metric_status.metric,
            metric_status.tier.name.title(),
            (metric_status.alert_type.value
             if metric_status.alert_type else "—"),
            f"{metric_status.baseline_mean:.4g} ± {metric_status.baseline_std:.4g}",
            (f"{metric_status.recent_mean:.4g}"
             if metric_status.recent_mean is not None else "—"),
            f"{metric_status.magnitude:+.2f}",
        ]
        for cell_text in cells:
            label = ctk.CTkLabel(
                self, text=cell_text,
                font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
                text_color=theme.TEXT_PRIMARY,
            )
            label.pack(
                side="left", expand=True, fill="x",
                padx=theme.SPACE_SM, pady=theme.SPACE_XS,
            )
            label.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self) -> None:
        self._on_click_external(self.metric)
```

### SmoothnessTab

- [ ] **Step 3: Append test**

```python
# ---------------------------------------------------------------------------
# Task 5b: SmoothnessTab
# ---------------------------------------------------------------------------


def test_smoothness_tab_renders_with_no_data(tk_root):
    """With no smoothness records, shows empty state without crashing."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import (
        SmoothnessTab,
    )

    tab = SmoothnessTab(tk_root, theme=ThemeManager())
    tab.set_records([])
    # No raise expected
```

- [ ] **Step 4: Create the widget**

Create `src/laser_trim_analyzer/gui/v6/widgets/smoothness_tab.py`:

```python
"""Spec 3c — SmoothnessTab.

Per-model smoothness summary: max_smoothness_value trend chart + recent
test list.  Reads SmoothnessResult records directly.
"""
from typing import List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart


class SmoothnessTab(ctk.CTkFrame):
    """Smoothness summary for one model."""

    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme

        self._chart = FocusChart(self, theme=theme)
        self._chart.pack(side="top", fill="x", pady=(0, theme.SPACE_MD))

        self._list_frame = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
        )
        self._list_frame.pack(side="top", fill="both", expand=True)

        self._empty_label = ctk.CTkLabel(
            self._list_frame,
            text="No smoothness records for this model.",
            font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
            text_color=theme.TEXT_SECONDARY,
        )

    def set_records(self, records: List) -> None:
        """records is a list of SmoothnessResult ORM rows (or duck-typed)."""
        # Clear list rows
        for child in self._list_frame.winfo_children():
            child.destroy()

        if not records:
            self._empty_label = ctk.CTkLabel(
                self._list_frame,
                text="No smoothness records for this model.",
                font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_BODY),
                text_color=self.theme.TEXT_SECONDARY,
            )
            self._empty_label.pack(pady=self.theme.SPACE_LG)
            self._chart.set_series(
                metric="max_smoothness_value", dates=[], values=[],
            )
            return

        # Update chart with the max_smoothness_value series
        dates = [r.file_date for r in records if r.file_date is not None]
        values = [
            r.max_smoothness_value for r in records
            if r.max_smoothness_value is not None
        ]
        self._chart.set_series(
            metric="max_smoothness_value",
            dates=dates, values=values,
        )

        # Build a row per record (latest first)
        for r in sorted(records, key=lambda r: r.file_date or 0, reverse=True):
            row = ctk.CTkFrame(self._list_frame, fg_color=self.theme.CARD)
            row.pack(side="top", fill="x", pady=1)
            text = (
                f"{r.serial or '—'} · "
                f"{(r.file_date.strftime('%Y-%m-%d') if r.file_date else '—')} · "
                f"max={r.max_smoothness_value or '—'} · "
                f"avg={r.avg_smoothness_value or '—'}"
            )
            label = ctk.CTkLabel(
                row, text=text,
                font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_CAPTION),
                text_color=self.theme.TEXT_PRIMARY,
            )
            label.pack(side="left", padx=self.theme.SPACE_SM, pady=self.theme.SPACE_XS)
```

### UnitsTab

- [ ] **Step 5: Append test**

```python
# ---------------------------------------------------------------------------
# Task 5c: UnitsTab
# ---------------------------------------------------------------------------


def test_units_tab_renders_row_per_unit(tk_root):
    """Each unit record becomes one row in the table."""
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab

    units = [
        {"serial": f"sn{i}", "file_date": datetime.now(),
         "overall_status": "PASS", "sigma_gradient": 0.01,
         "linearity_error": 0.005}
        for i in range(3)
    ]
    tab = UnitsTab(
        tk_root, theme=ThemeManager(),
        on_unit_click=lambda u: None,
        on_export=lambda: None,
    )
    tab.set_units(units)
    assert len(tab._rows) == 3
```

- [ ] **Step 6: Create the widget**

Create `src/laser_trim_analyzer/gui/v6/widgets/units_tab.py`:

```python
"""Spec 3c — UnitsTab.

Sortable table of recent units for the selected model.  Click a row →
open per-unit chart modal.  'Export to Excel' button at top.
"""
from typing import Callable, Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


_COLUMNS = [
    ("serial", "Serial"),
    ("file_date", "Date"),
    ("overall_status", "Status"),
    ("sigma_gradient", "Sigma"),
    ("linearity_error", "Lin Err"),
]


class UnitsTab(ctk.CTkFrame):
    """Sortable table of recent units."""

    def __init__(
        self,
        master,
        theme: ThemeManager,
        on_unit_click: Callable[[dict], None],
        on_export: Callable[[], None],
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_unit_click = on_unit_click
        self._on_export = on_export
        self._rows: List["_UnitRow"] = []
        self._units: List[dict] = []
        self._sort_key: str = "file_date"
        self._sort_reverse: bool = True

        # Export button at top
        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        export_btn = ctk.CTkButton(
            button_row, text="Export to Excel",
            fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
            text_color=theme.TEXT_INVERSE,
            command=self._on_export,
            corner_radius=theme.RADIUS_SM,
        )
        export_btn.pack(side="right")

        # Header
        header = ctk.CTkFrame(self, fg_color=theme.CARD)
        header.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        for key, label_text in _COLUMNS:
            label = ctk.CTkLabel(
                header, text=label_text,
                font=(theme.FONT_FAMILY[0], theme.SIZE_CAPTION, "bold"),
                text_color=theme.TEXT_SECONDARY,
            )
            label.pack(
                side="left", expand=True, fill="x",
                padx=theme.SPACE_SM, pady=theme.SPACE_XS,
            )
            label.bind("<Button-1>", lambda e, k=key: self._sort_by(k))

        # Scrollable rows
        self._list_frame = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
        )
        self._list_frame.pack(side="top", fill="both", expand=True)

    def set_units(self, units: List[dict]) -> None:
        self._units = list(units)
        self._render()

    def _sort_by(self, key: str) -> None:
        if self._sort_key == key:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_key = key
            self._sort_reverse = False
        self._render()

    def _render(self) -> None:
        for row in self._rows:
            row.destroy()
        self._rows.clear()
        sorted_units = sorted(
            self._units,
            key=lambda u: (u.get(self._sort_key) is None,
                           u.get(self._sort_key)),
            reverse=self._sort_reverse,
        )
        for unit in sorted_units:
            row = _UnitRow(
                self._list_frame, unit=unit, theme=self.theme,
                on_click=self._on_unit_click,
            )
            row.pack(side="top", fill="x", pady=1)
            self._rows.append(row)


class _UnitRow(ctk.CTkFrame):
    def __init__(self, master, unit: dict, theme: ThemeManager,
                 on_click: Callable[[dict], None]):
        super().__init__(master, fg_color=theme.SURFACE)
        self.unit = unit
        self._on_click_external = on_click

        for key, _ in _COLUMNS:
            value = unit.get(key)
            if hasattr(value, "strftime"):
                text = value.strftime("%Y-%m-%d")
            elif isinstance(value, float):
                text = f"{value:.4g}"
            else:
                text = str(value) if value is not None else "—"
            label = ctk.CTkLabel(
                self, text=text,
                font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
                text_color=theme.TEXT_PRIMARY,
            )
            label.pack(
                side="left", expand=True, fill="x",
                padx=theme.SPACE_SM, pady=theme.SPACE_XS,
            )
            label.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self) -> None:
        self._on_click_external(self.unit)
```

### PredictorPanel

- [ ] **Step 7: Append test**

```python
# ---------------------------------------------------------------------------
# Task 5d: PredictorPanel
# ---------------------------------------------------------------------------


def test_predictor_panel_starts_collapsed(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import (
        PredictorPanel,
    )

    panel = PredictorPanel(tk_root, theme=ThemeManager())
    assert panel._expanded is False


def test_predictor_panel_toggle_changes_state(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import (
        PredictorPanel,
    )

    panel = PredictorPanel(tk_root, theme=ThemeManager())
    panel.toggle()
    assert panel._expanded is True
    panel.toggle()
    assert panel._expanded is False
```

- [ ] **Step 8: Create the widget**

Create `src/laser_trim_analyzer/gui/v6/widgets/predictor_panel.py`:

```python
"""Spec 3c — PredictorPanel.

Demoted per-unit predictor output.  Default collapsed.  Marked
'diagnostic — not part of daily flow'.
"""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class PredictorPanel(ctk.CTkFrame):
    """Collapsible panel for the legacy per-unit predictor."""

    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(
            master, fg_color=theme.CARD,
            corner_radius=theme.RADIUS_MD, **kwargs,
        )
        self.theme = theme
        self._expanded = False

        # Header: title + toggle button
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(side="top", fill="x")
        title = ctk.CTkLabel(
            header, text="Predictor (diagnostic — not part of daily flow)",
            font=(theme.FONT_FAMILY[0], theme.SIZE_CAPTION, "bold"),
            text_color=theme.TEXT_SECONDARY,
        )
        title.pack(side="left", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
        self._toggle_btn = ctk.CTkButton(
            header, text="Show", width=60,
            fg_color=theme.CARD, hover_color=theme.ELEVATED,
            text_color=theme.TEXT_SECONDARY,
            command=self.toggle,
            corner_radius=theme.RADIUS_SM,
        )
        self._toggle_btn.pack(side="right", padx=theme.SPACE_SM, pady=theme.SPACE_XS)

        # Body (created but not packed until expanded)
        self._body = ctk.CTkFrame(self, fg_color="transparent")
        self._body_label = ctk.CTkLabel(
            self._body,
            text="No predictor data loaded.",
            font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
            text_color=theme.TEXT_SECONDARY,
        )
        self._body_label.pack(padx=theme.SPACE_SM, pady=theme.SPACE_SM)

    def toggle(self) -> None:
        if self._expanded:
            self._body.pack_forget()
            self._toggle_btn.configure(text="Show")
        else:
            self._body.pack(side="top", fill="x")
            self._toggle_btn.configure(text="Hide")
        self._expanded = not self._expanded

    def set_predictor_output(self, text: str) -> None:
        self._body_label.configure(text=text)
```

- [ ] **Step 9: Run + commit all widgets at once**

```bash
pytest tests/test_spec3c_model.py -v
```

Expected: all PASS so far.

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py src/laser_trim_analyzer/gui/v6/widgets/smoothness_tab.py src/laser_trim_analyzer/gui/v6/widgets/units_tab.py src/laser_trim_analyzer/gui/v6/widgets/predictor_panel.py tests/test_spec3c_model.py
git commit -m "feat(spec3c): DriftMetricsTab + SmoothnessTab + UnitsTab + PredictorPanel"
```

---

## Task 6: ModelPage composition

Assembles all the above into the ModelPage. Wires `on_show` to consume the routing hint and load data.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/pages/model_page.py`
- Modify: `src/laser_trim_analyzer/gui/v6/app.py` — register real ModelPage
- Test: `tests/test_spec3c_model.py` (APPEND)

- [ ] **Step 1: Append integration test**

```python
# ---------------------------------------------------------------------------
# Task 6: ModelPage composition
# ---------------------------------------------------------------------------


def test_model_page_consumes_routing_hint_on_show(tmp_path):
    """When Triage routes to Model with a model name, on_show populates the page."""
    from datetime import datetime
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SystemType as DBSystemType,
        StatusType as DBStatusType,
    )
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "model.db"
    app = V6App(cfg)
    try:
        app.withdraw()
        with app.db.session() as s:
            ar = DBAR(
                filename="x.xls", file_path="/fake/x.xls", file_hash="hx",
                model="ROUTED-MODEL", serial="sn1",
                system=DBSystemType.A, file_date=datetime.now(),
                timestamp=datetime.now(),
                overall_status=DBStatusType.PASS,
                has_multi_tracks=False, processing_time=0.1,
            )
            s.add(ar)
            s.commit()
        app.set_model_route("ROUTED-MODEL", "sigma_gradient")
        app.show_page("model")
        # The model page should have stashed this model
        model_page = app.page_container.get_page("model")
        assert model_page._current_model == "ROUTED-MODEL"
    finally:
        app.destroy()
```

- [ ] **Step 2: Create ModelPage**

Create `src/laser_trim_analyzer/gui/v6/pages/model_page.py`:

```python
"""Spec 3c — ModelPage.

Per-model investigation view.  Composes the focus chart, pill row,
tabs, and predictor panel.  on_show() consumes the routing hint from
V6App and loads data for the selected model.
"""
import threading
from datetime import datetime, timedelta
from typing import List, Optional

import customtkinter as ctk

from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAR, SmoothnessResult as DBSR, TrackResult as DBTR,
)
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import (
    DriftMetricsTab,
)
from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
from laser_trim_analyzer.gui.v6.widgets.tab_view import ThemedTabView
from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
from laser_trim_analyzer.ml.manager import get_model_drift_status


_WINDOW_DAYS = {"30d": 30, "90d": 90, "365d": 365, "All": None}


class ModelPage(PageBase):
    """Per-model investigation page."""
    page_title = "Model"

    def __init__(self, master, theme, app):
        self._app = app
        self._current_model: Optional[str] = None
        self._current_metric: str = "sigma_gradient"
        self._window_choice: str = "90d"
        super().__init__(master, theme=theme)

    def header_actions(self):
        # Time-window dropdown
        self._window_var = ctk.StringVar(value=self._window_choice)
        window_menu = ctk.CTkOptionMenu(
            self,
            values=list(_WINDOW_DAYS.keys()),
            variable=self._window_var,
            command=self._on_window_change,
            fg_color=self.theme.CARD,
            button_color=self.theme.ACCENT,
            button_hover_color=self.theme.ACCENT_HOVER,
            text_color=self.theme.TEXT_PRIMARY,
            corner_radius=self.theme.RADIUS_SM,
        )
        # Export button
        export_btn = ctk.CTkButton(
            self, text="Export evidence pack",
            command=self._on_export,
            fg_color=self.theme.ACCENT,
            hover_color=self.theme.ACCENT_HOVER,
            text_color=self.theme.TEXT_INVERSE,
            corner_radius=self.theme.RADIUS_SM,
        )
        return [window_menu, export_btn]

    def build_content(self, parent):
        # Pill row at top
        self._pill_row = MetricPillRow(
            parent, theme=self.theme, on_pill_click=self._on_pill_click,
        )
        self._pill_row.pack(side="top", fill="x", pady=(0, self.theme.SPACE_MD))

        # Focus chart (large)
        self._focus_chart = FocusChart(parent, theme=self.theme)
        self._focus_chart.pack(side="top", fill="x", pady=(0, self.theme.SPACE_MD))

        # Tabs
        self._tab_view = ThemedTabView(parent, theme=self.theme)
        self._tab_view.pack(side="top", fill="both", expand=True)
        drift_frame = self._tab_view.add("Drift Metrics")
        smooth_frame = self._tab_view.add("Smoothness")
        units_frame = self._tab_view.add("Units")

        self._drift_tab = DriftMetricsTab(
            drift_frame, theme=self.theme,
            on_metric_select=self._on_pill_click,
        )
        self._drift_tab.pack(fill="both", expand=True)

        self._smoothness_tab = SmoothnessTab(smooth_frame, theme=self.theme)
        self._smoothness_tab.pack(fill="both", expand=True)

        self._units_tab = UnitsTab(
            units_frame, theme=self.theme,
            on_unit_click=self._on_unit_click,
            on_export=self._on_export,
        )
        self._units_tab.pack(fill="both", expand=True)

        # Predictor panel below tabs
        self._predictor = PredictorPanel(parent, theme=self.theme)
        self._predictor.pack(side="top", fill="x", pady=(self.theme.SPACE_MD, 0))

    def on_show(self):
        # Consume routing hint if present
        model, focus = self._app.consume_model_route_full()
        if model:
            self._current_model = model
        if focus and focus in WATCHED_METRICS:
            self._current_metric = focus
        if self._current_model:
            self._pill_row.set_selected(self._current_metric)
            threading.Thread(target=self._reload_sync, daemon=True).start()

    def _reload_sync(self) -> None:
        if not self._current_model:
            return
        try:
            status = get_model_drift_status(self._app.db, self._current_model)
            series_dates, series_values, baseline = self._load_focus_series(
                self._current_model, self._current_metric
            )
            units = self._load_units(self._current_model)
            smoothness_records = self._load_smoothness(self._current_model)
        except Exception:
            status = None
            series_dates, series_values, baseline = [], [], (None, None)
            units = []
            smoothness_records = []

        def update_ui():
            if status:
                self._pill_row.set_status(status)
                self._drift_tab.set_status(status)
            mean, std = baseline
            self._focus_chart.set_series(
                metric=self._current_metric,
                dates=series_dates, values=series_values,
                baseline_mean=mean, baseline_std=std,
            )
            self._units_tab.set_units(units)
            self._smoothness_tab.set_records(smoothness_records)

        self.after(0, update_ui)

    # ---- Data-loading helpers --------------------------------------------

    def _window_cutoff(self) -> Optional[datetime]:
        days = _WINDOW_DAYS.get(self._window_choice)
        if days is None:
            return None
        return datetime.now() - timedelta(days=days)

    def _load_focus_series(self, model: str, metric: str):
        """Return (dates, values, (baseline_mean, baseline_std))."""
        from laser_trim_analyzer.database.models import ModelMetricState

        cutoff = self._window_cutoff()
        with self._app.db.session() as s:
            col = getattr(DBTR, metric, None)
            if metric == "max_smoothness_value":
                q = s.query(DBSR.file_date, DBSR.max_smoothness_value).filter(
                    DBSR.model == model,
                    DBSR.max_smoothness_value.isnot(None),
                )
                if cutoff:
                    q = q.filter(DBSR.file_date >= cutoff)
                rows = q.order_by(DBSR.file_date).all()
            elif col is None:
                rows = []
            else:
                q = (
                    s.query(DBAR.file_date, col)
                    .join(DBTR, DBTR.analysis_id == DBAR.id)
                    .filter(DBAR.model == model, col.isnot(None))
                )
                if cutoff:
                    q = q.filter(DBAR.file_date >= cutoff)
                rows = q.order_by(DBAR.file_date).all()
            dates = [r[0] for r in rows if r[0] is not None]
            values = [r[1] for r in rows if r[1] is not None]

            # Baseline
            ms = (
                s.query(ModelMetricState)
                .filter(
                    ModelMetricState.model == model,
                    ModelMetricState.metric == metric,
                )
                .first()
            )
            if ms:
                return dates, values, (ms.baseline_mean, ms.baseline_std)
            return dates, values, (None, None)

    def _load_units(self, model: str) -> List[dict]:
        cutoff = self._window_cutoff()
        with self._app.db.session() as s:
            q = (
                s.query(
                    DBAR.serial, DBAR.file_date, DBAR.overall_status,
                    DBTR.sigma_gradient,
                    DBTR.final_linearity_error_shifted.label("linearity_error"),
                )
                .join(DBTR, DBTR.analysis_id == DBAR.id)
                .filter(DBAR.model == model)
            )
            if cutoff:
                q = q.filter(DBAR.file_date >= cutoff)
            rows = q.order_by(DBAR.file_date.desc()).limit(200).all()
            return [
                {
                    "serial": r.serial,
                    "file_date": r.file_date,
                    "overall_status": r.overall_status.value
                    if hasattr(r.overall_status, "value") else str(r.overall_status),
                    "sigma_gradient": r.sigma_gradient,
                    "linearity_error": r.linearity_error,
                }
                for r in rows
            ]

    def _load_smoothness(self, model: str) -> List:
        cutoff = self._window_cutoff()
        with self._app.db.session() as s:
            q = s.query(DBSR).filter(DBSR.model == model)
            if cutoff:
                q = q.filter(DBSR.file_date >= cutoff)
            return q.order_by(DBSR.file_date.desc()).limit(200).all()

    # ---- Event handlers --------------------------------------------------

    def _on_pill_click(self, metric: str) -> None:
        self._current_metric = metric
        self._pill_row.set_selected(metric)
        threading.Thread(target=self._reload_sync, daemon=True).start()

    def _on_window_change(self, choice: str) -> None:
        self._window_choice = choice
        threading.Thread(target=self._reload_sync, daemon=True).start()

    def _on_unit_click(self, unit: dict) -> None:
        # For 3c: just log.  The per-unit modal port is a 3e-or-later concern
        # if the existing V5 modal logic transfers cleanly.
        pass

    def _on_export(self) -> None:
        # Spec 3c does not implement evidence-pack export; the button hook
        # is here so Spec 3d / future work can wire it up.
        pass
```

- [ ] **Step 3: Replace ModelPlaceholder in V6App**

In `gui/v6/app.py`, in `_build_pages()`, after the Triage construction:

```python
        from laser_trim_analyzer.gui.v6.pages.model_page import ModelPage
        model_page = ModelPage(self.page_container, theme=self.theme, app=self)
        self.page_container.add_page("model", model_page)
```

Remove the `("model", "Model", "3c")` tuple from the placeholder loop.

- [ ] **Step 4: Run + regression**

```bash
pytest tests/test_spec3c_model.py -v
pytest tests/test_spec1_untrimmed_sigma.py tests/test_log_derived_bugfixes_2026_05_30.py tests/test_5_8_2026_bugfixes.py tests/test_spec2_multi_metric_drift.py tests/test_spec3a_shell.py tests/test_spec3b_triage.py tests/test_spec3c_model.py 2>&1 | tail -5
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/app.py src/laser_trim_analyzer/gui/v6/pages/model_page.py tests/test_spec3c_model.py
git commit -m "feat(spec3c): real ModelPage with focus chart + pills + 3 tabs

Composes MetricPillRow + FocusChart + ThemedTabView (Drift Metrics /
Smoothness / Units) + PredictorPanel.  on_show consumes routing hint
from V6App, kicks off background load.  Pill click swaps focus chart;
window dropdown rescopes all data sources.

Evidence export button is wired but the action is a no-op placeholder
until a future spec implements the export."
```

---

## Out-of-scope reminders

- **Do not** implement evidence-pack Excel export (future spec).
- **Do not** port the per-unit chart modal (3e or later).
- **Do not** add cross-model comparison views (anchor doc: per-model mission).
- **Do not** wire predictor data loading — the panel just shows "No predictor data loaded" for v1.
