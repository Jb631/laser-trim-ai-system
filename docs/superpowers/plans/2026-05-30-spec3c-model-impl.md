# Spec 3c — Model Page Implementation Plan (rewritten 2026-06-01)

> **READ FIRST:** `docs/superpowers/plans/2026-06-01-spec3-rewrite-foundations.md`.
> This is the largest sub-spec and the mission's "show evidence" payoff. It implements foundations §3
> (QA rules Q1, Q4, Q5, Q6, Q7, Q8, Q9), §4.3 (shared metric column/label maps), and decision D4
> (ship the model selector, Copy summary, per-unit chart modal, and evidence export here — not "a future
> spec"). Shared fixtures in `tests/conftest.py`.

**Goal:** Replace the Model placeholder with the per-model investigation view: a model selector +
time-window picker + **Copy summary** + **Export evidence pack** in the header; an 8-pill glance row; a
large SPC focus chart; and three tabs (Drift Metrics / Smoothness / Units). The Units tab opens the
**per-unit pre/post-trim chart** (an explicitly-kept V5 feature). A demoted, lazily-loaded Predictor
panel sits below.

**Target branch:** `V6`. Start at the Spec 3b final commit.

**Fixes applied (foundations §6):** Q4 (focus chart uses `TRACK_METRIC_COLUMNS` so `linearity_error`
plots `final_linearity_error_shifted`, the column the detector trained on), Q5 (pair-then-unzip), I8
(materialize ORM → dicts before session close), I3 (generation token kills stale reloads), I9
(`safe_after`), C5/C6 (header actions built with the actions parent), Q1 (per-unit chart shows every
point + marks all fail points), Q6/Q7 (σ-honest, Rule-1 SPC overlays), S6 (empty state + model
selector), S7 (predictor lazy + graceful), D4 (Copy summary + evidence Excel + per-unit modal ship now).

---

## File Structure

**Created:** `gui/v6/widgets/{tab_view,metric_pill_row,focus_chart,drift_metrics_tab,smoothness_tab,
units_tab,unit_chart_modal,predictor_panel}.py`, `gui/v6/pages/model_page.py`,
`src/laser_trim_analyzer/export/evidence.py`, `tests/test_spec3c_model.py`.

**Modified:** `ml/drift_training.py` (public `TRACK_METRIC_COLUMNS` alias), `gui/v6/app.py`
(`consume_model_route_full` + register real `ModelPage`).

---

## Task 1: Routing-with-focus + public metric column map

- [ ] **Step 1:** Create `tests/test_spec3c_model.py`:

```python
"""Spec 3c — Model page. Foundations §3/§4.3. Fixtures in tests/conftest.py."""

# ---- Task 1: routing + column map ----------------------------------------

def test_consume_model_route_full(make_app):
    app = make_app()
    app.set_model_route("M1", "linearity_error")
    assert app.consume_model_route_full() == ("M1", "linearity_error")
    assert app.consume_model_route_full() == (None, None)


def test_consume_model_route_full_without_focus(make_app):
    app = make_app()
    app.set_model_route("M2")
    assert app.consume_model_route_full() == ("M2", None)


def test_track_metric_columns_public_and_linearity_maps_to_shifted():
    from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
    from laser_trim_analyzer.database.models import TrackResult as DBTR
    # Q4: the detector trains linearity_error on final_linearity_error_shifted; the UI must match.
    assert TRACK_METRIC_COLUMNS["linearity_error"] is DBTR.final_linearity_error_shifted
    assert "max_smoothness_value" not in TRACK_METRIC_COLUMNS  # lives on SmoothnessResult
```

- [ ] **Step 2:** In `ml/drift_training.py`, just after the `_TRACK_METRIC_COLUMNS = {...}` definition,
  add the public alias:

```python
# Public alias so the Spec 3 UI charts the SAME column the detector trained on
# (prevents the linearity_error/final_linearity_error_shifted mismatch). Read-only.
TRACK_METRIC_COLUMNS = _TRACK_METRIC_COLUMNS
```

- [ ] **Step 3:** In `gui/v6/app.py`, add to `V6App` (keep `consume_model_route` from 3b):

```python
    def consume_model_route_full(self):
        """Pop (model, focus_metric). Either may be None. Used by the Model page."""
        if self._model_route is None:
            return (None, None)
        route = self._model_route
        self._model_route = None
        return route
```

- [ ] **Step 4:** Run `-k "route_full or track_metric_columns"` → PASS. Commit
  `feat(spec3c): consume_model_route_full + public TRACK_METRIC_COLUMNS`.

---

## Task 2: ThemedTabView

- [ ] **Step 1:** Append:

```python
# ---- Task 2: ThemedTabView ------------------------------------------------

def test_themed_tab_view(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.tab_view import ThemedTabView
    tv = ThemedTabView(tk_root, theme=ThemeManager())
    assert tv.add("Drift Metrics") is not None
    tv.add("Units"); tv.set("Units")
    assert tv.get() == "Units"
```

- [ ] **Step 2:** Create `widgets/tab_view.py` (unchanged from the original draft — correct):

```python
"""Spec 3c — ThemedTabView: CTkTabview with V6 theme tokens."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class ThemedTabView(ctk.CTkTabview):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color=theme.SURFACE, segmented_button_fg_color=theme.CARD,
                         segmented_button_selected_color=theme.ACCENT,
                         segmented_button_selected_hover_color=theme.ACCENT_HOVER,
                         segmented_button_unselected_color=theme.CARD,
                         segmented_button_unselected_hover_color=theme.ELEVATED,
                         text_color=theme.TEXT_PRIMARY, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
```

- [ ] **Step 3:** Run → PASS. Commit `feat(spec3c): ThemedTabView`.

---

## Task 3: MetricPillRow (8 pills, readable labels, σ-honest)

- [ ] **Step 1:** Append:

```python
# ---- Task 3: MetricPillRow ------------------------------------------------

def _status(model="M1", **tiers):
    """Build a ModelDriftStatus; pass metric=Tier kwargs to override specific metrics."""
    from datetime import datetime
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, MetricStatus, ModelDriftStatus, WATCHED_METRICS)
    per = {}
    for m in WATCHED_METRICS:
        tier = tiers.get(m, DriftTier.STABLE)
        per[m] = MetricStatus(metric=m, tier=tier, alert_type=None,
                              magnitude=0.0 if tier == DriftTier.STABLE else 3.1,
                              baseline_mean=0.01, baseline_std=0.001,
                              recent_mean=0.012, recent_count=5, is_trained=True)
    return ModelDriftStatus(model=model, overall_tier=DriftTier.STABLE, worst_metric=None,
                            worst_alert_type=None, per_metric=per, last_processed=datetime.now())


def test_pill_row_has_eight_pills(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
    row = MetricPillRow(tk_root, theme=ThemeManager(), on_pill_click=lambda _: None)
    row.set_status(_status())
    assert set(row._pills) == set(WATCHED_METRICS)


def test_pill_shows_readable_label(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
    row = MetricPillRow(tk_root, theme=ThemeManager(), on_pill_click=lambda _: None)
    row.set_status(_status())
    assert row._pills["untrimmed_resistance"]._name_label.cget("text") == "Untrimmed resistance"


def test_pill_click_and_select(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
    got = []
    row = MetricPillRow(tk_root, theme=ThemeManager(), on_pill_click=got.append)
    row.set_status(_status())
    row._pills["sigma_gradient"]._on_click()
    assert got == ["sigma_gradient"]
    row.set_selected("linearity_error")
    assert row._selected_metric == "linearity_error"
```

- [ ] **Step 2:** Create `widgets/metric_pill_row.py` (labels via `metric_label`; summary text σ-honest):

```python
"""Spec 3c — MetricPillRow: 8 clickable, tier-colored metric pills."""
from typing import Callable, Dict, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import DriftTier, ModelDriftStatus, WATCHED_METRICS, metric_label


class MetricPillRow(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_pill_click: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme; self._cb = on_pill_click
        self._pills: Dict[str, _Pill] = {}
        self._selected_metric: Optional[str] = None
        for m in WATCHED_METRICS:
            p = _Pill(self, metric=m, theme=theme, on_click=self._cb)
            p.pack(side="left", padx=(0, theme.SPACE_SM), pady=theme.SPACE_XS)
            self._pills[m] = p

    def set_status(self, status: ModelDriftStatus) -> None:
        for m, pill in self._pills.items():
            ms = status.per_metric.get(m)
            if ms is not None:
                pill.set_metric_status(ms)

    def set_selected(self, metric: str) -> None:
        if self._selected_metric == metric:
            return
        if self._selected_metric and self._selected_metric in self._pills:
            self._pills[self._selected_metric].set_selected(False)
        if metric in self._pills:
            self._pills[metric].set_selected(True)
            self._selected_metric = metric


class _Pill(ctk.CTkFrame):
    def __init__(self, master, metric, theme: ThemeManager, on_click):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD,
                         border_width=2, border_color=theme.CARD)
        self.metric = metric; self.theme = theme; self._cb = on_click; self._selected = False
        self._name_label = ctk.CTkLabel(self, text=metric_label(metric),
                                        font=theme.font(theme.SIZE_CAPTION, "bold"),
                                        text_color=theme.TEXT_PRIMARY)
        self._name_label.pack(side="top", padx=theme.SPACE_SM, pady=(theme.SPACE_SM, 0))
        self._summary_label = ctk.CTkLabel(self, text="—", font=theme.font(theme.SIZE_CAPTION),
                                           text_color=theme.TEXT_SECONDARY)
        self._summary_label.pack(side="top", padx=theme.SPACE_SM, pady=(0, theme.SPACE_SM))
        for w in (self, self._name_label, self._summary_label):
            w.bind("<Button-1>", lambda e: self._on_click())

    def set_metric_status(self, ms) -> None:
        bg, fg = self.theme.tier_color(ms.tier)
        self.configure(fg_color=bg)
        if not self._selected:
            self.configure(border_color=bg)
        if not ms.is_trained:
            text = "untrained"
        elif ms.tier == DriftTier.STABLE:
            text = "OK"
        else:
            text = f"{ms.magnitude:+.1f}σ"      # Q6: σ beyond the tier threshold
        self._summary_label.configure(text=text, text_color=fg)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.configure(border_color=self.theme.ACCENT if selected else self.cget("fg_color"))

    def _on_click(self):
        self._cb(self.metric)
```

- [ ] **Step 3:** Run `-k pill` → PASS. Commit `feat(spec3c): MetricPillRow (readable labels, σ summary)`.

---

## Task 4: FocusChart — Rule-1 SPC overlays, leak-safe

Overlays (Q7): baseline mean (dashed), ±2σ warning band, ±3σ control limits, recent-window highlight;
points beyond ±3σ drawn in the OOC color. Pairs (date,value) before plotting (Q5). Closes its Figure on
destroy (matplotlib leak guard).

- [ ] **Step 1:** Append:

```python
# ---- Task 4: FocusChart ---------------------------------------------------

def test_focus_chart_set_series_no_crash(tk_root):
    from datetime import datetime, timedelta
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    chart = FocusChart(tk_root, theme=ThemeManager())
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(20, 0, -1)]
    values = [0.01 + 0.0001 * i for i in range(20)]
    chart.set_series(metric="sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.011, baseline_std=0.0005)


def test_focus_chart_empty_no_crash(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_series(metric="linearity_error", dates=[], values=[])  # empty state, no raise
```

- [ ] **Step 2:** Create `widgets/focus_chart.py`:

```python
"""Spec 3c — FocusChart: one metric's time series with Rule-1 SPC overlays (Q7)."""
from datetime import datetime
from typing import List, Optional

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import metric_label


class FocusChart(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
        self._fig = Figure(figsize=(8, 3), dpi=96, facecolor=theme.CARD)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True,
                                          padx=theme.SPACE_SM, pady=theme.SPACE_SM)
        self.bind("<Destroy>", self._on_destroy)
        self._style()

    def _style(self):
        ax, t = self._ax, self.theme
        ax.set_facecolor(t.CARD)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(t.TEXT_SECONDARY)
        ax.tick_params(colors=t.TEXT_SECONDARY, labelsize=9)
        ax.title.set_color(t.TEXT_PRIMARY)

    def set_series(self, metric: str, dates: List[datetime], values: List[float],
                   baseline_mean: Optional[float] = None, baseline_std: Optional[float] = None,
                   recent_batch_start: Optional[datetime] = None) -> None:
        ax, t = self._ax, self.theme
        ax.clear(); self._style()
        ax.set_title(metric_label(metric))
        if not dates or not values:
            ax.text(0.5, 0.5, "No measurements for this metric in the selected window.",
                    transform=ax.transAxes, ha="center", va="center", color=t.TEXT_SECONDARY)
            self._canvas.draw_idle(); return

        # SPC overlays (Rule 1 only).
        if baseline_mean is not None:
            ax.axhline(baseline_mean, color=t.TEXT_SECONDARY, ls="--", lw=1, label="Baseline mean")
            if baseline_std:
                ax.axhspan(baseline_mean - 2 * baseline_std, baseline_mean + 2 * baseline_std,
                           color=t.TIER_WARNING, alpha=0.08)
                for k in (3, -3):
                    ax.axhline(baseline_mean + k * baseline_std, color=t.TIER_OOC, ls=":", lw=1,
                               label="±3σ control limit" if k == 3 else None)
        if recent_batch_start is not None:
            ax.axvspan(recent_batch_start, dates[-1], color=t.ACCENT, alpha=0.10)

        ax.plot(dates, values, marker="o", ms=3, lw=1, color=t.ACCENT, label=metric_label(metric))
        # Mark out-of-3σ points in the OOC color (Rule 1 violations).
        if baseline_mean is not None and baseline_std:
            ucl, lcl = baseline_mean + 3 * baseline_std, baseline_mean - 3 * baseline_std
            ox = [d for d, v in zip(dates, values) if v > ucl or v < lcl]
            oy = [v for v in values if v > ucl or v < lcl]
            if ox:
                ax.scatter(ox, oy, color=t.TIER_OOC, s=30, zorder=5)
        ax.legend(loc="best", fontsize=8, facecolor=t.CARD, edgecolor=t.BORDER, labelcolor=t.TEXT_SECONDARY)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _on_destroy(self, _evt=None):
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        except Exception:
            pass
```

- [ ] **Step 3:** Run `-k focus_chart` → PASS. Commit `feat(spec3c): FocusChart (Rule-1 SPC overlays,
  leak-safe)`.

---

## Task 5: DriftMetricsTab (readable labels + Δσ)

- [ ] **Step 1:** Append:

```python
# ---- Task 5: DriftMetricsTab ----------------------------------------------

def test_drift_tab_row_per_metric_and_click(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
    got = []
    tab = DriftMetricsTab(tk_root, theme=ThemeManager(), on_metric_select=got.append)
    tab.set_status(_status())
    assert set(tab._rows) == set(WATCHED_METRICS)
    tab._rows["sigma_gradient"]._on_click()
    assert got == ["sigma_gradient"]
```

- [ ] **Step 2:** Create `widgets/drift_metrics_tab.py` — same structure as the original but cells use
  `metric_label`, show baseline mean ± std, recent mean, and Δσ (= magnitude), and the alert type:

```python
"""Spec 3c — DriftMetricsTab: table of all 8 metrics for the model."""
from typing import Callable, Dict

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import ModelDriftStatus, WATCHED_METRICS, metric_label

_COLUMNS = ["Metric", "Tier", "Alert", "Baseline (mean±std)", "Recent", "Δσ"]


class DriftMetricsTab(ctk.CTkScrollableFrame):
    def __init__(self, master, theme: ThemeManager, on_metric_select: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme; self._cb = on_metric_select
        self._rows: Dict[str, _MetricRow] = {}
        header = ctk.CTkFrame(self, fg_color=theme.CARD)
        header.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        for col in _COLUMNS:
            ctk.CTkLabel(header, text=col, font=theme.font(theme.SIZE_CAPTION, "bold"),
                         text_color=theme.TEXT_SECONDARY)\
                .pack(side="left", expand=True, fill="x", padx=theme.SPACE_SM, pady=theme.SPACE_XS)

    def set_status(self, status: ModelDriftStatus) -> None:
        for r in self._rows.values():
            r.destroy()
        self._rows.clear()
        for m in WATCHED_METRICS:
            ms = status.per_metric.get(m)
            if ms is None:
                continue
            row = _MetricRow(self, ms=ms, theme=self.theme, on_click=self._cb)
            row.pack(side="top", fill="x", pady=1)
            self._rows[m] = row


class _MetricRow(ctk.CTkFrame):
    def __init__(self, master, ms, theme: ThemeManager, on_click):
        bg, _ = theme.tier_color(ms.tier)
        super().__init__(master, fg_color=bg)
        self.metric = ms.metric; self._cb = on_click
        recent = f"{ms.recent_mean:.4g}" if ms.recent_mean is not None else "—"
        cells = [metric_label(ms.metric), ms.tier.name.replace("_", " ").title(),
                 ms.alert_type.value if ms.alert_type else "—",
                 f"{ms.baseline_mean:.4g} ± {ms.baseline_std:.4g}", recent, f"{ms.magnitude:+.2f}"]
        for txt in cells:
            lbl = ctk.CTkLabel(self, text=txt, font=theme.font(theme.SIZE_BODY),
                               text_color=theme.TEXT_PRIMARY)
            lbl.pack(side="left", expand=True, fill="x", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            lbl.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        self._cb(self.metric)
```

- [ ] **Step 3:** Run `-k drift_tab` → PASS. Commit `feat(spec3c): DriftMetricsTab`.

---

## Task 6: SmoothnessTab — consumes dicts (DetachedInstanceError fix)

The Model page passes a list of **plain dicts** (materialized in-session, see Task 10), never ORM rows.

- [ ] **Step 1:** Append:

```python
# ---- Task 6: SmoothnessTab ------------------------------------------------

def test_smoothness_tab_empty(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
    SmoothnessTab(tk_root, theme=ThemeManager()).set_records([])  # no raise


def test_smoothness_tab_with_records(tk_root):
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
    tab = SmoothnessTab(tk_root, theme=ThemeManager())
    tab.set_records([{"serial": "sn1", "file_date": datetime.now(),
                      "max_smoothness_value": 0.4, "avg_smoothness_value": 0.2}])
    assert len(tab._rows) == 1
```

- [ ] **Step 2:** Create `widgets/smoothness_tab.py` (consumes dicts; pairs date/value via Q5):

```python
"""Spec 3c — SmoothnessTab: max_smoothness_value trend + recent test list (dict-driven)."""
from typing import Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart


class SmoothnessTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._chart = FocusChart(self, theme=theme)
        self._chart.pack(side="top", fill="x", pady=(0, theme.SPACE_MD))
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._rows: List[ctk.CTkFrame] = []

    def set_records(self, records: List[Dict]) -> None:
        for r in self._rows:
            r.destroy()
        self._rows.clear()
        t = self.theme
        # Q5: pair date+value together.
        pairs = sorted(((r["file_date"], r["max_smoothness_value"]) for r in records
                        if r.get("file_date") is not None and r.get("max_smoothness_value") is not None),
                       key=lambda p: p[0])
        self._chart.set_series(metric="max_smoothness_value",
                               dates=[p[0] for p in pairs], values=[p[1] for p in pairs])
        if not records:
            lbl = ctk.CTkLabel(self._list, text="No smoothness records for this model.",
                               font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY)
            lbl.pack(pady=t.SPACE_LG); self._rows.append(lbl); return
        for r in sorted(records, key=lambda r: r.get("file_date") or 0, reverse=True):
            row = ctk.CTkFrame(self._list, fg_color=t.CARD); row.pack(side="top", fill="x", pady=1)
            when = r["file_date"].strftime("%Y-%m-%d") if r.get("file_date") else "—"
            txt = (f"{r.get('serial') or '—'} · {when} · "
                   f"max={r.get('max_smoothness_value', '—')} · avg={r.get('avg_smoothness_value', '—')}")
            ctk.CTkLabel(row, text=txt, font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_PRIMARY)\
                .pack(side="left", padx=t.SPACE_SM, pady=t.SPACE_XS)
            self._rows.append(row)
```

- [ ] **Step 3:** Run `-k smoothness_tab` → PASS. Commit `feat(spec3c): SmoothnessTab (dict-driven, paired
  series)`.

---

## Task 7: UnitsTab + UnitChartModal (kept per-unit drill-down, Q1/Q2)

UnitsTab consumes dicts (including `analysis_id` so the modal can load tracks). Row click opens the
per-unit pre/post-trim chart (reuses `gui/widgets/chart.py`). Q2: never dedupe repeated serials.

- [ ] **Step 1:** Append:

```python
# ---- Task 7: UnitsTab + UnitChartModal -----------------------------------

def test_units_tab_row_per_unit_keeps_duplicate_serials(tk_root):
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
    tab = UnitsTab(tk_root, theme=ThemeManager(), on_unit_click=lambda u: None, on_export=lambda: None)
    # Same serial twice = two valid trims (Q2). Both rows must appear.
    units = [{"analysis_id": 1, "serial": "sn1", "file_date": datetime.now(),
              "overall_status": "Pass", "sigma_gradient": 0.01, "linearity_error": 0.004},
             {"analysis_id": 2, "serial": "sn1", "file_date": datetime.now(),
              "overall_status": "Fail", "sigma_gradient": 0.02, "linearity_error": 0.05}]
    tab.set_units(units)
    assert len(tab._rows) == 2


def test_unit_chart_modal_marks_fail_points(tk_root):
    """Q1: every point shown; out-of-limit points become fail_points."""
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import compute_fail_points
    err = [0.0, 0.5, -0.2, 0.9]
    upper = [0.4, 0.4, 0.4, 0.4]; lower = [-0.4, -0.4, -0.4, -0.4]
    assert compute_fail_points(err, upper, lower) == [1, 3]  # 0.5>0.4 and 0.9>0.4
```

- [ ] **Step 2:** Create `widgets/units_tab.py` (dict-driven, sortable; export button):

```python
"""Spec 3c — UnitsTab: recent units for the model (dict rows). Click → per-unit chart modal."""
from typing import Callable, Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_COLUMNS = [("serial", "Serial"), ("file_date", "Date"), ("overall_status", "Status"),
            ("sigma_gradient", "Sigma"), ("linearity_error", "Lin Err")]


class UnitsTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_unit_click: Callable[[dict], None],
                 on_export: Callable[[], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme; self._on_unit_click = on_unit_click; self._on_export = on_export
        self._units: List[dict] = []; self._rows: List[_UnitRow] = []
        self._sort_key = "file_date"; self._sort_rev = True
        bar = ctk.CTkFrame(self, fg_color="transparent"); bar.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        ctk.CTkButton(bar, text="Export to Excel", fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                      text_color=theme.TEXT_INVERSE, command=self._on_export,
                      corner_radius=theme.RADIUS_SM).pack(side="right")
        header = ctk.CTkFrame(self, fg_color=theme.CARD); header.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        for key, lbl in _COLUMNS:
            h = ctk.CTkLabel(header, text=lbl, font=theme.font(theme.SIZE_CAPTION, "bold"),
                             text_color=theme.TEXT_SECONDARY)
            h.pack(side="left", expand=True, fill="x", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            h.bind("<Button-1>", lambda e, k=key: self._sort_by(k))
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._cap = ctk.CTkLabel(self, text="", font=theme.font(theme.SIZE_CAPTION),
                                 text_color=theme.TEXT_SECONDARY)
        self._cap.pack(side="top", fill="x")

    def set_units(self, units: List[dict]) -> None:
        self._units = list(units); self._render()

    def _sort_by(self, key):
        self._sort_rev = not self._sort_rev if self._sort_key == key else False
        self._sort_key = key; self._render()

    def _render(self):
        for r in self._rows:
            r.destroy()
        self._rows.clear()
        ordered = sorted(self._units,
                         key=lambda u: (u.get(self._sort_key) is None, u.get(self._sort_key)),
                         reverse=self._sort_rev)
        for u in ordered:
            row = _UnitRow(self._list, unit=u, theme=self.theme, on_click=self._on_unit_click)
            row.pack(side="top", fill="x", pady=1); self._rows.append(row)


class _UnitRow(ctk.CTkFrame):
    def __init__(self, master, unit: dict, theme: ThemeManager, on_click):
        super().__init__(master, fg_color=theme.SURFACE)
        self.unit = unit; self._cb = on_click
        for key, _ in _COLUMNS:
            v = unit.get(key)
            txt = (v.strftime("%Y-%m-%d") if hasattr(v, "strftime")
                   else f"{v:.4g}" if isinstance(v, float) else str(v) if v is not None else "—")
            lbl = ctk.CTkLabel(self, text=txt, font=theme.font(theme.SIZE_BODY),
                               text_color=theme.TEXT_PRIMARY)
            lbl.pack(side="left", expand=True, fill="x", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            lbl.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        self._cb(self.unit)
```

- [ ] **Step 3:** Create `widgets/unit_chart_modal.py` — reuses `ChartWidget.plot_error_vs_position`.
  `compute_fail_points` is a pure function (Q1). The modal loads the unit's first track arrays
  (materialized in-session) and plots; multi-track units note "track 1 of N" (full track picker is a
  later enhancement, disclosed — Q10):

```python
"""Spec 3c — UnitChartModal: per-unit pre/post-trim chart (kept V5 drill-down). Q1: show every point."""
from typing import List, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


def compute_fail_points(errors, upper_limits, lower_limits) -> List[int]:
    """Indices where the post-trim error violates the per-point spec band (zero-tolerance, Q1)."""
    out = []
    for i, e in enumerate(errors or []):
        if e is None:
            continue
        up = upper_limits[i] if upper_limits and i < len(upper_limits) else None
        lo = lower_limits[i] if lower_limits and i < len(lower_limits) else None
        if (up is not None and e > up) or (lo is not None and e < lo):
            out.append(i)
    return out


def load_unit_track(db, analysis_id: int) -> Optional[dict]:
    """Materialize the first track's arrays for an analysis INSIDE the session (I8-safe)."""
    from laser_trim_analyzer.database.models import TrackResult as DBTR
    with db.session() as s:
        tracks = s.query(DBTR).filter(DBTR.analysis_id == analysis_id).order_by(DBTR.track_id).all()
        if not tracks:
            return None
        tr = tracks[0]
        return {
            "track_id": tr.track_id, "n_tracks": len(tracks),
            "position_data": list(tr.position_data or []),
            "error_data": list(tr.error_data or []),
            "upper_limits": list(tr.upper_limits or []),
            "lower_limits": list(tr.lower_limits or []),
            "untrimmed_positions": list(tr.untrimmed_positions or []),
            "untrimmed_errors": list(tr.untrimmed_errors or []),
        }


class UnitChartModal(ctk.CTkToplevel):
    def __init__(self, master, theme: ThemeManager, db, unit: dict):
        super().__init__(master)
        self.theme = theme
        self.title(f"Unit {unit.get('serial', '')} — {unit.get('file_date', '')}")
        self.geometry("900x560"); self.configure(fg_color=theme.SURFACE); self.transient(master)
        from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle
        chart = ChartWidget(self, style=ChartStyle(figure_size=(9, 5)))
        chart.pack(fill="both", expand=True, padx=theme.SPACE_MD, pady=theme.SPACE_MD)
        data = load_unit_track(db, unit.get("analysis_id"))
        if not data or not data["position_data"]:
            chart.show_placeholder("No stored measurement arrays for this unit.")
            return
        fp = compute_fail_points(data["error_data"], data["upper_limits"], data["lower_limits"])
        title = f"Unit {unit.get('serial', '')}"
        if data["n_tracks"] > 1:
            title += f" — track {data['track_id']} of {data['n_tracks']} (showing track 1)"
        chart.plot_error_vs_position(
            positions=data["position_data"], trimmed_errors=data["error_data"],
            upper_limits=data["upper_limits"] or None, lower_limits=data["lower_limits"] or None,
            untrimmed_positions=data["untrimmed_positions"] or None,
            untrimmed_errors=data["untrimmed_errors"] or None,
            fail_points=fp, title=title, serial_number=str(unit.get("serial", "")))
```

- [ ] **Step 4:** Run `-k "units_tab or unit_chart"` → PASS. Commit `feat(spec3c): UnitsTab + per-unit
  chart modal (Q1 fail-point marking, Q2 keeps repeats)`.

---

## Task 8: PredictorPanel (lazy, labeled, graceful — fixes the always-empty panel)

- [ ] **Step 1:** Append:

```python
# ---- Task 8: PredictorPanel -----------------------------------------------

def test_predictor_panel_collapsed_then_toggles(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
    p = PredictorPanel(tk_root, theme=ThemeManager(), load_fn=lambda model: "Risk: LOW (demo)")
    assert p._expanded is False
    p.set_model("8340-1")
    p.toggle()                 # expand → triggers lazy load
    assert p._expanded is True
    assert "Risk" in p._body_label.cget("text")


def test_predictor_panel_load_failure_is_graceful(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
    def boom(model): raise RuntimeError("no predictor")
    p = PredictorPanel(tk_root, theme=ThemeManager(), load_fn=boom)
    p.set_model("X"); p.toggle()
    assert "No predictor" in p._body_label.cget("text")
```

- [ ] **Step 2:** Create `widgets/predictor_panel.py` (lazy load via injected `load_fn`; default
  load_fn reads the shared ML manager, best-effort):

```python
"""Spec 3c — PredictorPanel: demoted per-unit predictor. Lazy, clearly diagnostic, graceful."""
from typing import Callable, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


def _default_load(db):
    def _load(model: str) -> str:
        # Best-effort read of whatever the existing predictor exposes for the model.
        try:
            from laser_trim_analyzer.ml import get_shared_ml_manager
            mgr = get_shared_ml_manager(db)
            pred = getattr(mgr, "predictors", {}).get(model)
            if pred is None:
                raise LookupError("no trained predictor for this model")
            return f"Predictor loaded for {model} (diagnostic — not part of daily flow)."
        except Exception as exc:
            raise exc
    return _load


class PredictorPanel(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, load_fn: Optional[Callable[[str], str]] = None,
                 db=None, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
        self._load_fn = load_fn or (_default_load(db) if db is not None else None)
        self._model: Optional[str] = None
        self._expanded = False
        header = ctk.CTkFrame(self, fg_color="transparent"); header.pack(side="top", fill="x")
        ctk.CTkLabel(header, text="Predictor (diagnostic — not part of daily flow)",
                     font=theme.font(theme.SIZE_CAPTION, "bold"), text_color=theme.TEXT_SECONDARY)\
            .pack(side="left", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
        self._toggle_btn = ctk.CTkButton(header, text="Show", width=60, fg_color=theme.CARD,
                                         hover_color=theme.ELEVATED, text_color=theme.TEXT_SECONDARY,
                                         command=self.toggle, corner_radius=theme.RADIUS_SM)
        self._toggle_btn.pack(side="right", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
        self._body = ctk.CTkFrame(self, fg_color="transparent")
        self._body_label = ctk.CTkLabel(self._body, text="", font=theme.font(theme.SIZE_BODY),
                                        text_color=theme.TEXT_SECONDARY, wraplength=700, justify="left")
        self._body_label.pack(padx=theme.SPACE_SM, pady=theme.SPACE_SM)

    def set_model(self, model: str) -> None:
        self._model = model
        if self._expanded:
            self._load()

    def toggle(self) -> None:
        if self._expanded:
            self._body.pack_forget(); self._toggle_btn.configure(text="Show")
        else:
            self._body.pack(side="top", fill="x"); self._toggle_btn.configure(text="Hide"); self._load()
        self._expanded = not self._expanded

    def _load(self) -> None:
        if not self._model or self._load_fn is None:
            self._body_label.configure(text="No predictor available."); return
        try:
            self._body_label.configure(text=self._load_fn(self._model))
        except Exception:
            self._body_label.configure(
                text=f"No predictor for {self._model}. Train it in Settings → ML Training.")
```

- [ ] **Step 3:** Run `-k predictor` → PASS. Commit `feat(spec3c): PredictorPanel (lazy, graceful)`.

---

## Task 9: Evidence export (Copy summary + Excel pack) — the mission payoff (Q8)

- [ ] **Step 1:** Append:

```python
# ---- Task 9: evidence export ----------------------------------------------

def test_build_summary_text_has_evidence_metrics():
    from laser_trim_analyzer.export.evidence import build_summary_text
    txt = build_summary_text("8340-1", _status())
    assert "8340-1" in txt
    # Q8: the three evidence metrics James hands engineers must be present, readable.
    for label in ("Untrimmed resistance", "Linearity error", "Electrical angle"):
        assert label in txt


def test_export_evidence_pack_writes_xlsx(tmp_path):
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    from laser_trim_analyzer.export.evidence import export_evidence_pack
    db = DatabaseManager(tmp_path / "ev.db")
    with db.session() as s:
        ar = DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="8340-1", serial="sn1",
                  system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                  overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1)
        s.add(ar); s.flush()
        # TrackResult.status is NOT NULL — set it on any committed row.
        s.add(DBTR(analysis_id=ar.id, track_id="TRK1", status=StatusType.PASS,
                   sigma_gradient=0.01, final_linearity_error_shifted=0.004))
        s.commit()
    out = tmp_path / "pack.xlsx"
    export_evidence_pack(db, "8340-1", out, window_days=365)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2:** Create `src/laser_trim_analyzer/export/evidence.py` — self-contained, traceable
  (Q8). `build_summary_text` is paste-ready for chat/email; `export_evidence_pack` writes a two-sheet
  xlsx (per-metric baseline-vs-recent + the units list):

```python
"""Spec 3c — Evidence export. The daily 'what I hand engineers' payoff (foundations Q8)."""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS, metric_label


def build_summary_text(model: str, status) -> str:
    """Paste-ready text: model + per-metric baseline-vs-recent + alert. Q8 traceable."""
    lines = [f"Drift summary — model {model}",
             f"Overall: {status.overall_tier.name.replace('_', ' ').title()}"
             + (f" (worst: {metric_label(status.worst_metric)})" if status.worst_metric else ""), ""]
    for m in WATCHED_METRICS:
        ms = status.per_metric.get(m)
        if ms is None:
            continue
        recent = f"{ms.recent_mean:.4g}" if ms.recent_mean is not None else "n/a"
        tier = ms.tier.name.replace("_", " ").title()
        lines.append(f"- {metric_label(m)}: baseline {ms.baseline_mean:.4g} ± {ms.baseline_std:.4g}, "
                     f"recent {recent}, Δ {ms.magnitude:+.2f}σ [{tier}]")
    return "\n".join(lines)


def export_evidence_pack(db, model: str, out_path, window_days: Optional[int] = 365) -> Path:
    """Write a traceable evidence workbook for `model`. Sheets: Metrics, Units."""
    import pandas as pd
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR)
    from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
    from laser_trim_analyzer.ml.manager import get_model_drift_status

    status = get_model_drift_status(db, model)
    metric_rows = []
    for m in WATCHED_METRICS:
        ms = status.per_metric.get(m)
        if ms is None:
            continue
        metric_rows.append({"Metric": metric_label(m), "Tier": ms.tier.name,
                            "Alert": ms.alert_type.value if ms.alert_type else "",
                            "Baseline mean": ms.baseline_mean, "Baseline std": ms.baseline_std,
                            "Recent mean": ms.recent_mean, "Delta_sigma": ms.magnitude})

    cutoff = datetime.now() - timedelta(days=window_days) if window_days else None
    with db.session() as s:
        q = (s.query(DBAR.serial, DBAR.file_date, DBAR.overall_status, DBTR.sigma_gradient,
                     DBTR.final_linearity_error_shifted, DBTR.untrimmed_resistance,
                     DBTR.measured_electrical_angle)
             .join(DBTR, DBTR.analysis_id == DBAR.id).filter(DBAR.model == model))
        if cutoff is not None:
            q = q.filter(DBAR.file_date >= cutoff)
        unit_rows = [{"Serial": r[0],
                      "Date": r[1], "Status": getattr(r[2], "value", str(r[2])),
                      "Sigma gradient": r[3], "Linearity error": r[4],
                      "Untrimmed resistance": r[5], "Electrical angle": r[6]}
                     for r in q.order_by(DBAR.file_date.desc()).all()]

    out_path = Path(out_path)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        pd.DataFrame(metric_rows).to_excel(xl, sheet_name="Metrics", index=False)
        pd.DataFrame(unit_rows).to_excel(xl, sheet_name="Units", index=False)
    return out_path
```

- [ ] **Step 3:** Run `-k evidence` → PASS. Commit `feat(spec3c): evidence export (Copy summary text +
  xlsx pack)`.

---

## Task 10: ModelPage composition (selector, empty state, generation token)

Header (built into the actions parent): **model selector** (CTkComboBox of known models), time-window
dropdown, **Copy summary**, **Export evidence pack**. Empty state when no model selected (S6/Q9).
All loads use the generation token (I3) and `safe_after` (I9). All DB reads materialize to dicts (I8).

- [ ] **Step 1:** Append:

```python
# ---- Task 10: ModelPage ---------------------------------------------------

def test_model_page_consumes_route_on_show(make_app):
    from datetime import datetime
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SystemType, StatusType
    app = make_app()
    with app.db.session() as s:
        s.add(DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="ROUTED-MODEL",
                   serial="sn1", system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                   overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1))
        s.commit()
    app.set_model_route("ROUTED-MODEL", "linearity_error")
    app.show_page("model")
    page = app.page_container.get_page("model")
    assert page._current_model == "ROUTED-MODEL"
    assert page._current_metric == "linearity_error"


def test_model_page_empty_state_when_no_model(make_app):
    app = make_app()
    app.show_page("model")        # no route set
    page = app.page_container.get_page("model")
    assert page._current_model is None
    assert page._empty_label.winfo_ismapped() or page._empty_label.winfo_exists()


def test_model_page_focus_series_uses_shifted_linearity(make_app):
    """Q4: requesting linearity_error reads final_linearity_error_shifted."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    app = make_app()
    with app.db.session() as s:
        ar = DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="QM", serial="sn1",
                  system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                  overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1)
        s.add(ar); s.flush()
        # TrackResult.status is NOT NULL — set it on any committed row.
        s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                   final_linearity_error_shifted=0.0042))
        s.commit()
    page = app.page_container.get_page("model")
    dates, values, baseline = page._load_focus_series("QM", "linearity_error")
    assert values == [0.0042]
```

- [ ] **Step 2:** Create `pages/model_page.py`:

```python
"""Spec 3c — ModelPage: per-model investigation (selector + window + Copy/Export header,
8 pills, SPC focus chart, 3 tabs, demoted predictor). Foundations §3 + D4."""
import threading
from datetime import datetime, timedelta
from typing import List, Optional

import customtkinter as ctk

from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAR, ModelMetricState, SmoothnessResult as DBSR, TrackResult as DBTR)
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab
from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
from laser_trim_analyzer.gui.v6.widgets.tab_view import ThemedTabView
from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import UnitChartModal
from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
from laser_trim_analyzer.ml.manager import get_model_drift_status, list_known_models

_WINDOW_DAYS = {"30d": 30, "90d": 90, "365d": 365, "All": None}


class ModelPage(PageBase):
    page_title = "Model"

    def __init__(self, master, *, theme, app, page_title="Model"):
        self._current_model: Optional[str] = None
        self._current_metric: str = "sigma_gradient"
        self._window_choice: str = "90d"
        self._reload_gen = 0
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    # ---- header (built INTO the actions parent — no reparenting) ----
    def header_actions(self, parent):
        t = self.theme
        self._model_selector = ctk.CTkComboBox(parent, values=[], width=200,
                                                command=self._on_model_selected, fg_color=t.CARD,
                                                border_color=t.BORDER, button_color=t.ACCENT,
                                                button_hover_color=t.ACCENT_HOVER, text_color=t.TEXT_PRIMARY)
        self._model_selector.set("Select model…")
        self._model_selector.pack(side="left", padx=(0, t.SPACE_SM))
        self._window_menu = ctk.CTkOptionMenu(parent, values=list(_WINDOW_DAYS), width=80,
                                              command=self._on_window_change, fg_color=t.CARD,
                                              button_color=t.ACCENT, button_hover_color=t.ACCENT_HOVER,
                                              text_color=t.TEXT_PRIMARY)
        self._window_menu.set(self._window_choice)
        self._window_menu.pack(side="left", padx=(0, t.SPACE_SM))
        ctk.CTkButton(parent, text="Copy summary", command=self._on_copy_summary, fg_color=t.CARD,
                      hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY, corner_radius=t.RADIUS_SM)\
            .pack(side="left", padx=(0, t.SPACE_SM))
        ctk.CTkButton(parent, text="Export evidence pack", command=self._on_export, fg_color=t.ACCENT,
                      hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM)\
            .pack(side="left")

    def build_content(self, parent):
        t = self.theme
        self._empty_label = ctk.CTkLabel(
            parent, text="Pick a model above, or click one from Triage, to see its drift profile.",
            font=t.font(t.SIZE_HEADING), text_color=t.TEXT_SECONDARY)
        self._body = ctk.CTkFrame(parent, fg_color="transparent")
        self._pill_row = MetricPillRow(self._body, theme=t, on_pill_click=self._on_pill_click)
        self._pill_row.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        self._focus_chart = FocusChart(self._body, theme=t)
        self._focus_chart.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        self._tabs = ThemedTabView(self._body, theme=t)
        self._tabs.pack(side="top", fill="both", expand=True)
        self._drift_tab = DriftMetricsTab(self._tabs.add("Drift Metrics"), theme=t,
                                          on_metric_select=self._on_pill_click)
        self._drift_tab.pack(fill="both", expand=True)
        self._smoothness_tab = SmoothnessTab(self._tabs.add("Smoothness"), theme=t)
        self._smoothness_tab.pack(fill="both", expand=True)
        self._units_tab = UnitsTab(self._tabs.add("Units"), theme=t,
                                   on_unit_click=self._on_unit_click, on_export=self._on_export)
        self._units_tab.pack(fill="both", expand=True)
        self._predictor = PredictorPanel(self._body, theme=t, db=self.app.db)
        self._predictor.pack(side="top", fill="x", pady=(t.SPACE_MD, 0))
        self._show_empty()

    # ---- visibility ----
    def _show_empty(self):
        self._body.pack_forget()
        self._empty_label.pack(expand=True)

    def _show_body(self):
        self._empty_label.pack_forget()
        self._body.pack(fill="both", expand=True)

    # ---- lifecycle ----
    def on_show(self):
        model, focus = self.app.consume_model_route_full()
        if model:
            self._current_model = model
        if focus and focus in WATCHED_METRICS:
            self._current_metric = focus
        # Refresh the selector's model list each show.
        threading.Thread(target=self._refresh_selector_values, daemon=True).start()
        if self._current_model:
            self._show_body()
            self._pill_row.set_selected(self._current_metric)
            self._predictor.set_model(self._current_model)
            self._reload()
        else:
            self._show_empty()

    def _refresh_selector_values(self):
        try:
            models = [m.model for m in list_known_models(self.app.db)]
        except Exception:
            models = []
        def apply():
            self._model_selector.configure(values=models)
            if self._current_model:
                self._model_selector.set(self._current_model)
        self.safe_after(apply)

    # ---- reload with generation token (I3) ----
    def _reload(self):
        if not self._current_model:
            return
        self._reload_gen += 1
        gen = self._reload_gen
        model, metric = self._current_model, self._current_metric
        def work():
            try:
                status = get_model_drift_status(self.app.db, model)
                dates, values, baseline = self._load_focus_series(model, metric)
                units = self._load_units(model)
                smoothness = self._load_smoothness(model)
            except Exception:
                status, dates, values, baseline, units, smoothness = None, [], [], (None, None), [], []
            def apply():
                if gen != self._reload_gen:
                    return  # a newer reload superseded this one
                if status:
                    self._pill_row.set_status(status)
                    self._drift_tab.set_status(status)
                self._focus_chart.set_series(metric=metric, dates=dates, values=values,
                                             baseline_mean=baseline[0], baseline_std=baseline[1])
                self._units_tab.set_units(units)
                self._smoothness_tab.set_records(smoothness)
            self.safe_after(apply)
        threading.Thread(target=work, daemon=True).start()

    # ---- loaders (all materialize to plain values inside the session — I8) ----
    def _window_cutoff(self) -> Optional[datetime]:
        days = _WINDOW_DAYS.get(self._window_choice)
        return None if days is None else datetime.now() - timedelta(days=days)

    def _load_focus_series(self, model, metric):
        cutoff = self._window_cutoff()
        with self.app.db.session() as s:
            if metric == "max_smoothness_value":
                q = s.query(DBSR.file_date, DBSR.max_smoothness_value).filter(
                    DBSR.model == model, DBSR.max_smoothness_value.isnot(None))
                if cutoff:
                    q = q.filter(DBSR.file_date >= cutoff)
                rows = q.order_by(DBSR.file_date).all()
            elif metric in TRACK_METRIC_COLUMNS:
                col = TRACK_METRIC_COLUMNS[metric]      # Q4: SAME column the detector trained on
                q = (s.query(DBAR.file_date, col).join(DBTR, DBTR.analysis_id == DBAR.id)
                     .filter(DBAR.model == model, col.isnot(None)))
                if cutoff:
                    q = q.filter(DBAR.file_date >= cutoff)
                rows = q.order_by(DBAR.file_date).all()
            else:
                rows = []
            pairs = [(r[0], r[1]) for r in rows if r[0] is not None and r[1] is not None]  # Q5
            ms = s.query(ModelMetricState).filter_by(model=model, metric=metric).first()
            baseline = (ms.baseline_mean, ms.baseline_std) if ms else (None, None)
        return [p[0] for p in pairs], [p[1] for p in pairs], baseline

    def _load_units(self, model) -> List[dict]:
        cutoff = self._window_cutoff()
        with self.app.db.session() as s:
            q = (s.query(DBAR.id, DBAR.serial, DBAR.file_date, DBAR.overall_status,
                         DBTR.sigma_gradient, DBTR.final_linearity_error_shifted)
                 .join(DBTR, DBTR.analysis_id == DBAR.id).filter(DBAR.model == model))
            if cutoff:
                q = q.filter(DBAR.file_date >= cutoff)
            rows = q.order_by(DBAR.file_date.desc()).limit(200).all()
            return [{"analysis_id": r[0], "serial": r[1], "file_date": r[2],
                     "overall_status": getattr(r[3], "value", str(r[3])),
                     "sigma_gradient": r[4], "linearity_error": r[5]} for r in rows]

    def _load_smoothness(self, model) -> List[dict]:
        cutoff = self._window_cutoff()
        with self.app.db.session() as s:
            q = s.query(DBSR).filter(DBSR.model == model)
            if cutoff:
                q = q.filter(DBSR.file_date >= cutoff)
            rows = q.order_by(DBSR.file_date.desc()).limit(200).all()
            return [{"serial": r.serial, "file_date": r.file_date,
                     "max_smoothness_value": r.max_smoothness_value,
                     "avg_smoothness_value": r.avg_smoothness_value,
                     "overall_status": getattr(r.overall_status, "value", str(r.overall_status))}
                    for r in rows]

    # ---- events ----
    def _on_model_selected(self, model):
        if model and model != "Select model…":
            self._current_model = model
            self._show_body()
            self._predictor.set_model(model)
            self._reload()

    def _on_pill_click(self, metric):
        self._current_metric = metric
        self._pill_row.set_selected(metric)
        self._reload()

    def _on_window_change(self, choice):
        self._window_choice = choice
        self._reload()

    def _on_unit_click(self, unit):
        UnitChartModal(self, theme=self.theme, db=self.app.db, unit=unit)

    def _on_copy_summary(self):
        if not self._current_model:
            return
        from laser_trim_analyzer.export.evidence import build_summary_text
        try:
            status = get_model_drift_status(self.app.db, self._current_model)
            text = build_summary_text(self._current_model, status)
            self.clipboard_clear(); self.clipboard_append(text)
        except Exception:
            pass

    def _on_export(self):
        if not self._current_model:
            return
        from tkinter import filedialog
        from laser_trim_analyzer.export.evidence import export_evidence_pack
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            initialfile=f"evidence_{self._current_model}.xlsx",
                                            filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        days = _WINDOW_DAYS.get(self._window_choice)
        threading.Thread(
            target=lambda: export_evidence_pack(self.app.db, self._current_model, path, days),
            daemon=True).start()
```

- [ ] **Step 3:** Register the real ModelPage in `gui/v6/app.py` `_build_pages()` (replace the `model`
  placeholder; keep process/settings placeholders):

```python
        from laser_trim_analyzer.gui.v6.pages.model_page import ModelPage
        self.page_container.add_page(
            "model", ModelPage(self.page_container, theme=self.theme, app=self, page_title="Model"))
```
Remove the `("model", "Model", "3c")` tuple from the placeholder loop.

- [ ] **Step 4:** Run `pytest tests/test_spec3c_model.py -v` → all PASS. Regression sweep → 0 fail.

- [ ] **Step 5:** Commit `feat(spec3c): real ModelPage (selector, empty state, gen-token reloads,
  Q4 focus series, per-unit modal, Copy/Export)`.

---

## Out of scope (3c)
- Per-metric sensitivity overrides (Settings, single global preset only). Editing exclude_points (lives
  in Settings → Per-model specs, 3d). Multi-track per-unit picker (disclosed "showing track 1"); full
  picker is a later enhancement. Real-time refresh while processing (explicit re-show is the refresh).
</content>
