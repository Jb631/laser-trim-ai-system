# Dashboard "Production Health" + Model-page fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a yield-focused **Dashboard** landing page (trim + final-test pass-rate panels with a per-day trend and a clickable lowest-yield-models list that routes to the Model page) and fix three Model-page defects (open on the worst metric, real UI-computed "Recent" values, grid-aligned Drift Metrics table).

**Architecture:** Pure windowed-yield helpers in `core/yield_stats.py` (testable against a tmp DB, no Tk) feed three new V6 widgets (`MiniTrendChart`, `YieldPanel`, `WorstModelsList`) composed by `DashboardPage` (a `PageBase` subclass). Dashboard becomes the startup page. The Model fixes are surgical edits to `pages/model_page.py` and `widgets/drift_metrics_tab.py`. Everything follows the foundations doc contracts (PageBase, V6App, theme tokens, threading §2.4, QA §3) and the established 3b/3c/3d test patterns (`tests/conftest.py` `tk_root` + `make_app`).

**Tech Stack:** Python 3.14, customtkinter 5.2.2, matplotlib (TkAgg), SQLAlchemy 2.0, pytest. Run tests with `.venv/bin/python -m pytest`.

---

## Conventions for every task

- Branch is `V6`. Tests run with `.venv/bin/python -m pytest <path> -p no:warnings -q`.
- Matplotlib widgets MUST name their canvas attribute `self.canvas`, never `self._canvas` (CTkFrame reserves `_canvas` for its background canvas and delegates `.bind()` to it — using `_canvas` breaks construction).
- `WATCHED_METRICS` is the live set in `ml/drift_types.py` (8 metrics incl. `composite_trim_risk_score`; `sigma_gradient` is NOT in it). Trust the code, not older docs.
- New widget tests use the `tk_root` fixture; page/app tests use `make_app`; never both in one test.

---

## File Structure

**Created:**
- `src/laser_trim_analyzer/core/yield_stats.py` — pure windowed yield + worst-model aggregation.
- `src/laser_trim_analyzer/gui/v6/widgets/mini_trend_chart.py` — compact matplotlib line (date → %).
- `src/laser_trim_analyzer/gui/v6/widgets/yield_panel.py` — one yield panel (%, counts, total, trend).
- `src/laser_trim_analyzer/gui/v6/widgets/worst_models_list.py` — ranked clickable model rows.
- `src/laser_trim_analyzer/gui/v6/pages/dashboard_page.py` — the page.
- `tests/test_dashboard.py` — all Dashboard tests.

**Modified:**
- `src/laser_trim_analyzer/gui/v6/sidebar.py` — add `("dashboard", "Dashboard")` to `ITEMS`.
- `src/laser_trim_analyzer/gui/v6/app.py` — register `DashboardPage`; open on it.
- `tests/test_spec3a_shell.py` — update the `Sidebar.ITEMS` assertion.
- `src/laser_trim_analyzer/gui/v6/pages/model_page.py` — worst-metric default + UI recent means.
- `src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py` — grid layout + recent overrides.
- `tests/test_spec3c_model.py` — extend for the Model fixes.

---

## Task 1: `compute_yield` pure helper

**Files:**
- Create: `src/laser_trim_analyzer/core/yield_stats.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_dashboard.py`:

```python
"""Dashboard (Production Health) + helpers. Fixtures in tests/conftest.py."""
from datetime import datetime, timedelta

import pytest

from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAR, FinalTestResult as DBFT, SystemType, StatusType)


def _add_ar(s, model, status, when):
    s.add(DBAR(filename=f"{model}-{status.name}-{when.microsecond}.xls",
               file_path="/f/x.xls", file_hash=f"h{model}{status.name}{when.microsecond}{when.second}",
               model=model, serial="sn1", system=SystemType.A, file_date=when, timestamp=when,
               overall_status=status, has_multi_tracks=False, processing_time=0.1))


def _add_ft(s, model, status, when):
    s.add(DBFT(filename=f"ft-{model}-{status.name}-{when.microsecond}.xls", file_path="/f/ft.xls",
               file_hash=f"f{model}{status.name}{when.microsecond}{when.second}", model=model, serial="sn1",
               file_date=when, test_date=when, timestamp=when, overall_status=status))


def test_compute_yield_empty(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    y = compute_yield(DatabaseManager(tmp_path / "e.db"), DBAR, None)
    assert y["total"] == 0 and y["pass_rate"] is None and y["trend"] == []


def test_compute_yield_buckets_and_rate(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    db = DatabaseManager(tmp_path / "y.db")
    now = datetime.now()
    with db.session() as s:
        for _ in range(3):
            _add_ar(s, "M", StatusType.PASS, now)
        _add_ar(s, "M", StatusType.WARNING, now)
        _add_ar(s, "M", StatusType.FAIL, now)
        _add_ar(s, "M", StatusType.UNTRIMMED, now)   # excluded from rate
        s.commit()
    y = compute_yield(db, DBAR, None)
    assert (y["passed"], y["warnings"], y["failed"], y["untrimmed"]) == (3, 1, 1, 1)
    assert y["gradeable"] == 5
    assert y["pass_rate"] == pytest.approx(60.0)     # 3 / (3+1+1)


def test_compute_yield_windowed(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    db = DatabaseManager(tmp_path / "w.db")
    now = datetime.now()
    with db.session() as s:
        _add_ar(s, "M", StatusType.PASS, now)
        _add_ar(s, "M", StatusType.PASS, now - timedelta(days=200))   # outside 90d
        s.commit()
    assert compute_yield(db, DBAR, now - timedelta(days=90))["total"] == 1


def test_compute_yield_on_final_test(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    db = DatabaseManager(tmp_path / "ft.db")
    now = datetime.now()
    with db.session() as s:
        _add_ft(s, "M", StatusType.PASS, now)
        _add_ft(s, "M", StatusType.FAIL, now)
        s.commit()
    y = compute_yield(db, DBFT, None)
    assert y["passed"] == 1 and y["failed"] == 1 and y["pass_rate"] == pytest.approx(50.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k compute_yield -p no:warnings -q`
Expected: FAIL (`ModuleNotFoundError: ...core.yield_stats`).

- [ ] **Step 3: Implement** — create `src/laser_trim_analyzer/core/yield_stats.py`:

```python
"""Windowed production-yield aggregation for the Dashboard.

Pure helpers: given a DatabaseManager and an ORM model class that has
`overall_status` + `file_date` (analysis_results or final_test_results), return
status-bucket counts, a pass-rate, and a per-day pass-rate trend. No Tk.
"""
from datetime import datetime
from typing import Optional

# overall_status (StatusType) name -> yield bucket.
_BUCKET = {
    "PASS": "passed", "WARNING": "warnings", "FAIL": "failed",
    "ERROR": "errors", "PROCESSING_FAILED": "errors", "UNTRIMMED": "untrimmed",
}


def _bucket(status) -> str:
    name = getattr(status, "name", None) or str(status)
    return _BUCKET.get(str(name).upper(), "errors")


def compute_yield(db, model_cls, cutoff: Optional[datetime]) -> dict:
    """Yield over `model_cls` rows with file_date >= cutoff (cutoff None = all time).

    pass_rate = passed / (passed + warnings + failed) * 100, i.e. a WARNING counts
    as not-a-clean-pass and ERROR/UNTRIMMED are excluded from the denominator.
    Returns counts, gradeable, total, pass_rate (None if no gradeable rows), and
    trend = [{"date": "YYYY-MM-DD", "pass_rate": float}] ascending by day.
    """
    counts = {"passed": 0, "warnings": 0, "failed": 0, "errors": 0, "untrimmed": 0}
    per_day: dict = {}
    with db.session() as s:
        q = s.query(model_cls.file_date, model_cls.overall_status)
        if cutoff is not None:
            q = q.filter(model_cls.file_date >= cutoff)
        rows = q.all()
    for file_date, status in rows:
        b = _bucket(status)
        counts[b] += 1
        if b in ("passed", "warnings", "failed") and file_date is not None:
            slot = per_day.setdefault(file_date.strftime("%Y-%m-%d"), [0, 0])
            slot[0] += 1
            if b == "passed":
                slot[1] += 1
    gradeable = counts["passed"] + counts["warnings"] + counts["failed"]
    pass_rate = (counts["passed"] / gradeable * 100.0) if gradeable else None
    trend = [{"date": d, "pass_rate": (p / t * 100.0 if t else 0.0)}
             for d, (t, p) in sorted(per_day.items())]
    return {**counts, "gradeable": gradeable, "total": sum(counts.values()),
            "pass_rate": pass_rate, "trend": trend}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k compute_yield -p no:warnings -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/core/yield_stats.py tests/test_dashboard.py
git commit -m "feat(dashboard): compute_yield (windowed status buckets + pass-rate + per-day trend)"
```

---

## Task 2: `worst_models_by_yield` pure helper

**Files:**
- Modify: `src/laser_trim_analyzer/core/yield_stats.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Append the failing tests** to `tests/test_dashboard.py`:

```python
def test_worst_models_ranks_and_min_units(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import worst_models_by_yield
    db = DatabaseManager(tmp_path / "wm.db")
    now = datetime.now()
    with db.session() as s:
        # GOOD: 5 pass -> 100%
        for _ in range(5):
            _add_ar(s, "GOOD", StatusType.PASS, now)
        # BAD: 3 pass + 2 fail -> 60%
        for _ in range(3):
            _add_ar(s, "BAD", StatusType.PASS, now)
        for _ in range(2):
            _add_ar(s, "BAD", StatusType.FAIL, now)
        # TINY: 1 fail -> below min_units, excluded
        _add_ar(s, "TINY", StatusType.FAIL, now)
        s.commit()
    rows, total = worst_models_by_yield(db, None, min_units=5, limit=10)
    assert [r["model"] for r in rows] == ["BAD", "GOOD"]   # worst first
    assert total == 2                                       # TINY excluded by min_units
    assert rows[0]["units"] == 5 and rows[0]["trim_rate"] == pytest.approx(60.0)


def test_worst_models_cap_disclosed(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import worst_models_by_yield
    db = DatabaseManager(tmp_path / "cap.db")
    now = datetime.now()
    with db.session() as s:
        for i in range(12):
            for _ in range(5):
                _add_ar(s, f"M{i:02d}", StatusType.PASS, now)
        s.commit()
    rows, total = worst_models_by_yield(db, None, min_units=5, limit=10)
    assert len(rows) == 10 and total == 12


def test_worst_models_joins_ft_rate(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import worst_models_by_yield
    db = DatabaseManager(tmp_path / "j.db")
    now = datetime.now()
    with db.session() as s:
        for _ in range(5):
            _add_ar(s, "M", StatusType.PASS, now)
        _add_ft(s, "M", StatusType.PASS, now)
        _add_ft(s, "M", StatusType.FAIL, now)     # FT 50%
        s.commit()
    rows, _ = worst_models_by_yield(db, None, min_units=5, limit=10)
    assert rows[0]["ft_rate"] == pytest.approx(50.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k worst_models -p no:warnings -q`
Expected: FAIL (`cannot import name 'worst_models_by_yield'`).

- [ ] **Step 3: Implement** — append to `src/laser_trim_analyzer/core/yield_stats.py`:

```python
def worst_models_by_yield(db, cutoff: Optional[datetime], min_units: int = 5, limit: int = 10):
    """Rank models by trim yield (worst first) over the window.

    'units' = gradeable trim records (pass+warning+fail) for the model; only models
    with units >= min_units are ranked (so a 1-unit 0% model can't dominate). Each
    row: {model, units, trim_rate, ft_rate}. trim_rate/ft_rate are % or None.
    Returns (rows[:limit], total_qualifying) so the caller can disclose the cap.
    """
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, FinalTestResult as DBFT)

    def _by_model(model_cls):
        acc: dict = {}
        with db.session() as s:
            q = s.query(model_cls.model, model_cls.overall_status)
            if cutoff is not None:
                q = q.filter(model_cls.file_date >= cutoff)
            rows = q.all()
        for model, status in rows:
            if not model:
                continue
            b = _bucket(status)
            if b not in ("passed", "warnings", "failed"):
                continue
            slot = acc.setdefault(model, [0, 0])   # [gradeable, passed]
            slot[0] += 1
            if b == "passed":
                slot[1] += 1
        return acc

    trim = _by_model(DBAR)
    ft = _by_model(DBFT)

    rows = []
    for model, (gradeable, passed) in trim.items():
        if gradeable < min_units:
            continue
        ftv = ft.get(model)
        rows.append({
            "model": model,
            "units": gradeable,
            "trim_rate": (passed / gradeable * 100.0) if gradeable else None,
            "ft_rate": (ftv[1] / ftv[0] * 100.0) if ftv and ftv[0] else None,
        })
    rows.sort(key=lambda r: (r["trim_rate"] is None, r["trim_rate"] if r["trim_rate"] is not None else 0.0))
    return rows[:limit], len(rows)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k worst_models -p no:warnings -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/core/yield_stats.py tests/test_dashboard.py
git commit -m "feat(dashboard): worst_models_by_yield (ranked, min-units guard, FT join, disclosed cap)"
```

---

## Task 3: `MiniTrendChart` widget

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/mini_trend_chart.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Append the failing tests**:

```python
def test_mini_trend_chart_set_points_no_crash(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.mini_trend_chart import MiniTrendChart
    c = MiniTrendChart(tk_root, theme=ThemeManager())
    c.set_points([("2026-05-01", 90.0), ("2026-05-02", 80.0), ("2026-05-03", 95.0)])


def test_mini_trend_chart_empty_no_crash(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.mini_trend_chart import MiniTrendChart
    MiniTrendChart(tk_root, theme=ThemeManager()).set_points([])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k mini_trend -p no:warnings -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** — create `src/laser_trim_analyzer/gui/v6/widgets/mini_trend_chart.py`:

```python
"""Dashboard — MiniTrendChart: a compact pass-rate-over-time line (no SPC overlays)."""
from typing import List, Tuple

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class MiniTrendChart(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_SM, **kwargs)
        self.theme = theme
        self._fig = Figure(figsize=(3.2, 1.0), dpi=96, facecolor=theme.CARD)
        self._ax = self._fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self._fig, master=self)   # NOTE: `canvas`, not `_canvas`
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.bind("<Destroy>", self._on_destroy)

    def set_points(self, points: List[Tuple[str, float]]) -> None:
        ax, t = self._ax, self.theme
        ax.clear()
        ax.set_facecolor(t.CARD)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        if points:
            ys = [p[1] for p in points]
            ax.plot(range(len(ys)), ys, lw=1.5, color=t.ACCENT)
            ax.set_ylim(0, 100)
        else:
            ax.text(0.5, 0.5, "no trend", transform=ax.transAxes, ha="center", va="center",
                    color=t.TEXT_DISABLED, fontsize=8)
        self._fig.tight_layout(pad=0.2)
        self.canvas.draw_idle()

    def _on_destroy(self, _evt=None):
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        except Exception:
            pass
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k mini_trend -p no:warnings -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/mini_trend_chart.py tests/test_dashboard.py
git commit -m "feat(dashboard): MiniTrendChart (compact pass-rate line, leak-safe)"
```

---

## Task 4: `YieldPanel` widget

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/yield_panel.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Append the failing tests**:

```python
def _labels_text(widget):
    import customtkinter as ctk
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
            out.append(c.cget("text"))
        out.extend(_labels_text(c))
    return out


def test_yield_panel_renders_rate_and_counts(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.yield_panel import YieldPanel
    p = YieldPanel(tk_root, theme=ThemeManager(), title="Trim analysis yield")
    p.set_yield({"passed": 3, "warnings": 1, "failed": 1, "errors": 0, "untrimmed": 1,
                 "gradeable": 5, "total": 6, "pass_rate": 60.0, "trend": []},
                total_label="6 units")
    txt = " | ".join(_labels_text(p))
    assert "Trim analysis yield" in txt
    assert "60" in txt                # headline %
    assert "3" in txt and "1" in txt  # pass / warn-fail counts
    assert "6 units" in txt


def test_yield_panel_empty_state(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.yield_panel import YieldPanel
    p = YieldPanel(tk_root, theme=ThemeManager(), title="Final-test yield")
    p.set_yield({"passed": 0, "warnings": 0, "failed": 0, "errors": 0, "untrimmed": 0,
                 "gradeable": 0, "total": 0, "pass_rate": None, "trend": []},
                total_label="0 matched")
    assert "—" in " ".join(_labels_text(p))   # no fabricated 0%
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k yield_panel -p no:warnings -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** — create `src/laser_trim_analyzer/gui/v6/widgets/yield_panel.py`:

```python
"""Dashboard — YieldPanel: headline pass-rate %, Pass/Warn/Fail counts, total, trend."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.mini_trend_chart import MiniTrendChart


class YieldPanel(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, title: str, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
        t = theme
        ctk.CTkLabel(self, text=title, font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w")\
            .pack(side="top", fill="x", padx=t.SPACE_MD, pady=(t.SPACE_MD, 0))
        self._rate = ctk.CTkLabel(self, text="—", font=t.font(t.SIZE_DISPLAY, "bold"),
                                  text_color=t.TEXT_PRIMARY, anchor="w")
        self._rate.pack(side="top", fill="x", padx=t.SPACE_MD)
        self._counts = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_BODY),
                                    text_color=t.TEXT_SECONDARY, anchor="w")
        self._counts.pack(side="top", fill="x", padx=t.SPACE_MD)
        self._total = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_CAPTION),
                                   text_color=t.TEXT_SECONDARY, anchor="w")
        self._total.pack(side="top", fill="x", padx=t.SPACE_MD)
        self._trend = MiniTrendChart(self, theme=t)
        self._trend.pack(side="top", fill="x", padx=t.SPACE_MD, pady=(t.SPACE_SM, t.SPACE_MD))

    def set_yield(self, stats: dict, total_label: str) -> None:
        rate = stats.get("pass_rate")
        self._rate.configure(text=f"{rate:.1f}% pass" if rate is not None else "—")
        self._counts.configure(
            text=f"Pass {stats.get('passed', 0)} · Warn {stats.get('warnings', 0)} · "
                 f"Fail {stats.get('failed', 0)}")
        self._total.configure(text=total_label)
        self._trend.set_points([(p["date"], p["pass_rate"]) for p in stats.get("trend", [])])
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k yield_panel -p no:warnings -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/yield_panel.py tests/test_dashboard.py
git commit -m "feat(dashboard): YieldPanel (headline %, Pass/Warn/Fail, total, trend)"
```

---

## Task 5: `WorstModelsList` widget

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/worst_models_list.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Append the failing tests**:

```python
def test_worst_models_list_rows_and_click(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
    got = []
    w = WorstModelsList(tk_root, theme=ThemeManager(), on_row_click=got.append)
    w.set_rows([{"model": "BAD", "units": 5, "trim_rate": 60.0, "ft_rate": 48.0},
                {"model": "OK", "units": 9, "trim_rate": 95.0, "ft_rate": None}], total=2)
    assert len(w._rows) == 2
    w._rows[0]._on_click()
    assert got == ["BAD"]


def test_worst_models_list_discloses_cap(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
    w = WorstModelsList(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    w.set_rows([{"model": f"M{i}", "units": 5, "trim_rate": 50.0, "ft_rate": None}
                for i in range(10)], total=25)
    assert "10 of 25" in w._cap.cget("text")


def test_worst_models_list_empty(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
    w = WorstModelsList(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    w.set_rows([], total=0)
    assert w._rows == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k worst_models_list -p no:warnings -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** — create `src/laser_trim_analyzer/gui/v6/widgets/worst_models_list.py`:

```python
"""Dashboard — WorstModelsList: ranked, clickable model rows (model/units/trim%/FT%)."""
from typing import Callable, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_COLS = [("model", "Model"), ("units", "Units"), ("trim_rate", "Trim %"), ("ft_rate", "FT %")]


def _fmt(key, value) -> str:
    if value is None:
        return "—"
    if key in ("trim_rate", "ft_rate"):
        return f"{value:.0f}%"
    return str(value)


class WorstModelsList(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_row_click: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_row_click
        self._rows: List["_WorstRow"] = []
        t = theme
        ctk.CTkLabel(self, text="Lowest-yield models", font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        header = ctk.CTkFrame(self, fg_color=t.CARD)
        header.pack(side="top", fill="x")
        header.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="wm")
        for i, (_key, label) in enumerate(_COLS):
            ctk.CTkLabel(header, text=label, font=t.font(t.SIZE_CAPTION, "bold"),
                         text_color=t.TEXT_SECONDARY, anchor="w")\
                .grid(row=0, column=i, sticky="ew", padx=t.SPACE_SM, pady=t.SPACE_XS)
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._cap = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_CAPTION),
                                 text_color=t.TEXT_SECONDARY, anchor="w")
        self._cap.pack(side="top", fill="x")
        self._empty = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_BODY),
                                   text_color=t.TEXT_SECONDARY, anchor="w")

    def set_rows(self, rows: List[dict], total: int) -> None:
        for r in self._rows:
            r.destroy()
        self._rows.clear()
        t = self.theme
        if not rows:
            self._cap.configure(text="")
            self._empty.configure(text="No models with enough recent data to rank.")
            self._empty.pack(side="top", fill="x", pady=t.SPACE_MD)
            return
        self._empty.pack_forget()
        for row in rows:
            r = _WorstRow(self._list, row=row, theme=t, on_click=self._cb)
            r.pack(side="top", fill="x", pady=1)
            self._rows.append(r)
        self._cap.configure(text=f"Showing {len(rows)} of {total} (min 5 units, worst first)."
                            if total > len(rows) else f"{total} models (min 5 units, worst first).")


class _WorstRow(ctk.CTkFrame):
    def __init__(self, master, row: dict, theme: ThemeManager, on_click: Callable[[str], None]):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=theme.RADIUS_SM)
        self.row = row
        self._cb = on_click
        self.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="wm")
        for i, (key, _label) in enumerate(_COLS):
            lbl = ctk.CTkLabel(self, text=_fmt(key, row.get(key)), font=theme.font(theme.SIZE_BODY),
                               text_color=theme.TEXT_PRIMARY, anchor="w")
            lbl.grid(row=0, column=i, sticky="ew", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            lbl.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        self._cb(self.row["model"])
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k worst_models_list -p no:warnings -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/worst_models_list.py tests/test_dashboard.py
git commit -m "feat(dashboard): WorstModelsList (ranked clickable rows, disclosed cap, empty state)"
```

---

## Task 6: `DashboardPage` + landing + sidebar

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/pages/dashboard_page.py`
- Modify: `src/laser_trim_analyzer/gui/v6/sidebar.py` (ITEMS), `src/laser_trim_analyzer/gui/v6/app.py` (register + startup)
- Modify test: `tests/test_spec3a_shell.py` (ITEMS assertion)
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Append the failing tests** to `tests/test_dashboard.py`:

```python
def test_dashboard_is_landing(make_app):
    app = make_app()
    assert app.page_container.current_page == "dashboard"
    assert app.page_container.get_page("dashboard") is not None


def test_dashboard_reload_now_populates(make_app):
    app = make_app()
    now = datetime.now()
    with app.db.session() as s:
        for _ in range(5):
            _add_ar(s, "DASH", StatusType.PASS, now)
        _add_ar(s, "DASH", StatusType.FAIL, now)
        s.commit()
    page = app.page_container.get_page("dashboard")
    page.reload_now()
    # trim panel shows a rate; worst-models has DASH (5 pass + 1 fail = 6 gradeable >= 5)
    assert any("83" in x or "%" in x for x in _labels_text(page._trim_panel))
    assert any(r.row["model"] == "DASH" for r in page._worst._rows)


def test_dashboard_row_click_routes_to_model(make_app):
    app = make_app()
    page = app.page_container.get_page("dashboard")
    page._on_model_click("ROUTED")
    assert app.page_container.current_page == "model"
    assert app._model_route == ("ROUTED", None)
```

- [ ] **Step 2: Update the 3a sidebar assertion** in `tests/test_spec3a_shell.py` — change the `Sidebar.ITEMS` expectation to include Dashboard first:

```python
    assert Sidebar.ITEMS == [("dashboard", "Dashboard"), ("triage", "Triage"),
                             ("process", "Process"), ("model", "Model"), ("settings", "Settings")]
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k dashboard tests/test_spec3a_shell.py -k "sidebar or dashboard" -p no:warnings -q`
Expected: FAIL (no `dashboard` page; ITEMS mismatch).

- [ ] **Step 4: Add the sidebar item** — edit `src/laser_trim_analyzer/gui/v6/sidebar.py`:

```python
    ITEMS: List[Tuple[str, str]] = [
        ("dashboard", "Dashboard"), ("triage", "Triage"), ("process", "Process"),
        ("model", "Model"), ("settings", "Settings"),
    ]
```

- [ ] **Step 5: Create the page** — `src/laser_trim_analyzer/gui/v6/pages/dashboard_page.py`:

```python
"""Dashboard — Production Health landing: trim + final-test yield panels with trend,
and a clickable lowest-yield-models list that routes to the Model page."""
import threading
from datetime import datetime, timedelta

import customtkinter as ctk

from laser_trim_analyzer.core.yield_stats import compute_yield, worst_models_by_yield
from laser_trim_analyzer.database.models import AnalysisResult as DBAR, FinalTestResult as DBFT
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
from laser_trim_analyzer.gui.v6.widgets.yield_panel import YieldPanel

_WINDOW_DAYS = {"30d": 30, "90d": 90, "365d": 365, "All": 36500}


class DashboardPage(PageBase):
    page_title = "Dashboard"

    def __init__(self, master, *, theme, app, page_title="Dashboard"):
        self._window_choice = "90d"
        self._reload_gen = 0
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    def header_actions(self, parent):
        t = self.theme
        self._window_menu = ctk.CTkOptionMenu(parent, values=list(_WINDOW_DAYS), width=90,
                                              command=self._on_window_change, fg_color=t.CARD,
                                              button_color=t.ACCENT, button_hover_color=t.ACCENT_HOVER,
                                              text_color=t.TEXT_PRIMARY)
        self._window_menu.set(self._window_choice)
        self._window_menu.pack(side="left")

    def build_content(self, parent):
        t = self.theme
        panels = ctk.CTkFrame(parent, fg_color="transparent")
        panels.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        panels.grid_columnconfigure((0, 1), weight=1, uniform="yp")
        self._trim_panel = YieldPanel(panels, theme=t, title="Trim analysis yield")
        self._trim_panel.grid(row=0, column=0, sticky="ew", padx=(0, t.SPACE_SM))
        self._ft_panel = YieldPanel(panels, theme=t, title="Final-test yield")
        self._ft_panel.grid(row=0, column=1, sticky="ew", padx=(t.SPACE_SM, 0))
        self._worst = WorstModelsList(parent, theme=t, on_row_click=self._on_model_click)
        self._worst.pack(side="top", fill="both", expand=True)

    # ---- lifecycle ----
    def on_show(self):
        threading.Thread(target=self._reload_threaded, daemon=True).start()

    def _cutoff(self):
        return datetime.now() - timedelta(days=_WINDOW_DAYS.get(self._window_choice, 90))

    def reload_now(self):
        """Synchronous reload + apply (test path / main-thread apply)."""
        self._apply(*self._query())

    def _reload_threaded(self):
        self._reload_gen += 1
        gen = self._reload_gen
        data = self._query()
        self.safe_after(lambda: self._apply(*data) if gen == self._reload_gen else None)

    def _query(self):
        cutoff = self._cutoff()
        try:
            trim = compute_yield(self.app.db, DBAR, cutoff)
            ft = compute_yield(self.app.db, DBFT, cutoff)
            worst, total = worst_models_by_yield(self.app.db, cutoff)
        except Exception:
            trim = ft = {"passed": 0, "warnings": 0, "failed": 0, "errors": 0, "untrimmed": 0,
                         "gradeable": 0, "total": 0, "pass_rate": None, "trend": []}
            worst, total = [], 0
        return trim, ft, worst, total

    def _apply(self, trim, ft, worst, total):
        self._trim_panel.set_yield(trim, total_label=f"{trim['total']} units")
        self._ft_panel.set_yield(ft, total_label=f"{ft['total']} matched")
        self._worst.set_rows(worst, total)

    # ---- events ----
    def _on_window_change(self, choice):
        self._window_choice = choice
        self.on_show()

    def _on_model_click(self, model):
        self.app.set_model_route(model)
        self.app.show_page("model")
```

- [ ] **Step 6: Register + make it the landing** — in `src/laser_trim_analyzer/gui/v6/app.py` `_build_pages`, add the import and the page BEFORE triage, and change the startup page. Add to the import group and the `add_page` calls:

```python
        from laser_trim_analyzer.gui.v6.pages.dashboard_page import DashboardPage
        from laser_trim_analyzer.gui.v6.pages.triage_page import TriagePage
        from laser_trim_analyzer.gui.v6.pages.model_page import ModelPage
        from laser_trim_analyzer.gui.v6.pages.settings_page import SettingsPage
        from laser_trim_analyzer.gui.v6.pages.process_page import ProcessPage
        self.page_container.add_page(
            "dashboard",
            DashboardPage(self.page_container, theme=self.theme, app=self, page_title="Dashboard"),
        )
        self.page_container.add_page(
            "triage",
            TriagePage(self.page_container, theme=self.theme, app=self, page_title="Triage"),
        )
        # ... model, settings, process unchanged ...
```

And in `__init__`, change the startup call from `self.show_page("triage")` to:

```python
        self.show_page("dashboard")
```

- [ ] **Step 7: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py tests/test_spec3a_shell.py -p no:warnings -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/pages/dashboard_page.py src/laser_trim_analyzer/gui/v6/sidebar.py src/laser_trim_analyzer/gui/v6/app.py tests/test_dashboard.py tests/test_spec3a_shell.py
git commit -m "feat(dashboard): DashboardPage (yield panels + worst-models) as the V6 landing"
```

---

## Task 7: Model page opens on the worst/flagged metric

**Files:**
- Modify: `src/laser_trim_analyzer/gui/v6/pages/model_page.py`
- Test: `tests/test_spec3c_model.py`

- [ ] **Step 1: Append the failing tests** to `tests/test_spec3c_model.py`:

```python
# ---- Dashboard-round Model fixes ------------------------------------------

def test_resolve_focus_metric_prefers_worst_when_not_user_picked():
    from laser_trim_analyzer.gui.v6.pages.model_page import ModelPage
    from laser_trim_analyzer.ml.drift_types import DriftTier, ModelDriftStatus
    status = ModelDriftStatus(model="M", overall_tier=DriftTier.OUT_OF_CONTROL,
                              worst_metric="trim_pass_count", worst_alert_type=None, per_metric={})
    # not user-picked -> worst metric wins
    assert ModelPage._resolve_focus_metric(status, False, "untrimmed_sigma_gradient") == "trim_pass_count"
    # user picked -> keep their choice
    assert ModelPage._resolve_focus_metric(status, True, "linearity_error") == "linearity_error"
    # no worst (all stable) -> keep current fallback
    stable = ModelDriftStatus(model="M", overall_tier=DriftTier.STABLE, worst_metric=None,
                              worst_alert_type=None, per_metric={})
    assert ModelPage._resolve_focus_metric(stable, False, "untrimmed_sigma_gradient") == "untrimmed_sigma_gradient"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_spec3c_model.py -k resolve_focus_metric -p no:warnings -q`
Expected: FAIL (`AttributeError: _resolve_focus_metric`).

- [ ] **Step 3: Implement** — in `src/laser_trim_analyzer/gui/v6/pages/model_page.py`:

(a) In `__init__`, add a user-pick flag after `self._current_metric = _DEFAULT_METRIC`:

```python
        self._user_picked_metric = False
```

(b) Add the pure resolver as a staticmethod (next to `_bucket_for_status`-style helpers, e.g. just after `__init__`):

```python
    @staticmethod
    def _resolve_focus_metric(status, user_picked, current):
        """Pick the metric to focus: the user's explicit pick wins; otherwise the
        model's worst flagged metric; otherwise the current fallback."""
        if user_picked:
            return current
        if status is not None and status.worst_metric and status.worst_metric in WATCHED_METRICS:
            return status.worst_metric
        return current
```

(c) In `on_show`, when a routing focus IS supplied treat it as a user pick; when arriving without focus, reset the flag so auto-selection can run. Replace the focus block:

```python
        model, focus = self.app.consume_model_route_full()
        if model:
            self._current_model = model
            self._user_picked_metric = False
        if focus and focus in WATCHED_METRICS:
            self._current_metric = focus
            self._user_picked_metric = True
```

(d) In `_on_model_selected`, reset the flag so the new model auto-focuses its worst metric. After `self._current_model = model` add:

```python
            self._user_picked_metric = False
```

(e) In `_on_pill_click`, mark an explicit pick. At the top of the method add:

```python
        self._user_picked_metric = True
```

(f) In `_reload`'s worker, resolve the metric from the loaded status before loading the series. Replace the body of `work()` so it loads status first, resolves the metric, then loads that metric's series:

```python
        def work():
            try:
                status = get_model_drift_status(self.app.db, model)
                chosen = self._resolve_focus_metric(status, self._user_picked_metric, metric)
                dates, values, baseline = self._load_focus_series(model, chosen)
                units = self._load_units(model)
                smoothness = self._load_smoothness(model)
            except Exception:
                status, chosen = None, metric
                dates, values, baseline, units, smoothness = [], [], (None, None), [], []
            def apply():
                if gen != self._reload_gen:
                    return
                self._current_metric = chosen
                if status:
                    self._pill_row.set_status(status)
                    self._drift_tab.set_status(status)
                self._pill_row.set_selected(chosen)
                self._focus_chart.set_series(metric=chosen, dates=dates, values=values,
                                             baseline_mean=baseline[0], baseline_std=baseline[1])
                self._units_tab.set_units(units)
                self._smoothness_tab.set_records(smoothness)
            self.safe_after(apply)
```

(Keep the `self._reload_gen += 1; gen = self._reload_gen; model, metric = ...` preamble as-is above `work`.)

- [ ] **Step 4: Run to verify pass** (and that 3c didn't regress)

Run: `.venv/bin/python -m pytest tests/test_spec3c_model.py -p no:warnings -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/pages/model_page.py tests/test_spec3c_model.py
git commit -m "fix(model): open on the worst/flagged metric unless the user picked one"
```

---

## Task 8: Model page — UI-computed "Recent" values

**Files:**
- Modify: `src/laser_trim_analyzer/gui/v6/pages/model_page.py`, `src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py`
- Test: `tests/test_spec3c_model.py`

- [ ] **Step 1: Append the failing test**:

```python
def test_model_recent_means_computed_from_data(make_app):
    """Recent column comes from the actual recent-window data, not the detector
    (which never persists a recent mean)."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    app = make_app()
    with app.db.session() as s:
        for val in (0.0040, 0.0044):
            ar = DBAR(filename=f"r{val}.xls", file_path="/f/x.xls", file_hash=f"hr{val}",
                      model="RM", serial="sn1", system=SystemType.A, file_date=datetime.now(),
                      timestamp=datetime.now(), overall_status=StatusType.PASS,
                      has_multi_tracks=False, processing_time=0.1)
            s.add(ar); s.flush()
            s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                       final_linearity_error_shifted=val))
        s.commit()
    page = app.page_container.get_page("model")
    means = page._recent_means("RM")
    assert means["linearity_error"] == pytest.approx(0.0042)   # mean(0.0040, 0.0044)
```

(Ensure `import pytest` is present at the top of `tests/test_spec3c_model.py`; add it if missing.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_spec3c_model.py -k recent_means -p no:warnings -q`
Expected: FAIL (`AttributeError: _recent_means`).

- [ ] **Step 3: Implement**

(a) In `src/laser_trim_analyzer/gui/v6/pages/model_page.py`, add a recent-window constant near `_WINDOW_DAYS`:

```python
_RECENT_DAYS = 30   # "recent" window for the baseline-vs-recent comparison
```

(b) Add the `_recent_means` loader (next to `_load_focus_series`), reusing the same column map and the smoothness special-case, averaging each metric over the last `_RECENT_DAYS`:

```python
    def _recent_means(self, model) -> dict:
        """Mean of each watched metric over the last _RECENT_DAYS, computed from the
        actual data (the detector does not persist a recent mean). Metric -> float|None."""
        from sqlalchemy import func
        cutoff = datetime.now() - timedelta(days=_RECENT_DAYS)
        out = {}
        with self.app.db.session() as s:
            for metric in WATCHED_METRICS:
                if metric == "max_smoothness_value":
                    val = (s.query(func.avg(DBSR.max_smoothness_value))
                           .filter(DBSR.model == model, DBSR.max_smoothness_value.isnot(None),
                                   DBSR.file_date >= cutoff).scalar())
                elif metric in TRACK_METRIC_COLUMNS:
                    col = TRACK_METRIC_COLUMNS[metric]
                    val = (s.query(func.avg(col)).join(DBAR, DBTR.analysis_id == DBAR.id)
                           .filter(DBAR.model == model, col.isnot(None),
                                   DBAR.file_date >= cutoff).scalar())
                else:
                    val = None
                out[metric] = float(val) if val is not None else None
        return out
```

(c) In `_reload`'s `work()`, also compute recent means and pass them to the drift tab. Add `recent = self._recent_means(model)` into the `try` (and `recent = {}` in the `except`), then in `apply()` change the drift-tab line to:

```python
                if status:
                    self._pill_row.set_status(status)
                    self._drift_tab.set_status(status, recent_means=recent)
```

(d) In `src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py`, make `set_status` accept optional recent overrides and pass each metric's value to its row. Change the signature and the row construction:

```python
    def set_status(self, status: ModelDriftStatus, recent_means: dict = None) -> None:
        recent_means = recent_means or {}
        for r in self._rows.values():
            r.destroy()
        self._rows.clear()
        for m in WATCHED_METRICS:
            ms = status.per_metric.get(m)
            if ms is None:
                continue
            row = _MetricRow(self, ms=ms, theme=self.theme, on_click=self._cb,
                             recent_override=recent_means.get(m))
            row.pack(side="top", fill="x", pady=1)
            self._rows[m] = row
```

And in `_MetricRow.__init__`, accept `recent_override` and prefer it for the Recent cell:

```python
    def __init__(self, master, ms, theme: ThemeManager, on_click, recent_override=None):
        bg, _ = theme.tier_color(ms.tier)
        super().__init__(master, fg_color=bg)
        self.metric = ms.metric
        self._cb = on_click
        recent_val = recent_override if recent_override is not None else ms.recent_mean
        recent = f"{recent_val:.4g}" if recent_val is not None else "—"
        cells = [metric_label(ms.metric), ms.tier.name.replace("_", " ").title(),
                 ms.alert_type.value if ms.alert_type else "—",
                 f"{ms.baseline_mean:.4g} ± {ms.baseline_std:.4g}", recent, f"{ms.magnitude:+.2f}"]
        # ... existing cell-rendering loop unchanged ...
```

- [ ] **Step 4: Run to verify pass** (and 3c not regressed)

Run: `.venv/bin/python -m pytest tests/test_spec3c_model.py -p no:warnings -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/pages/model_page.py src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py tests/test_spec3c_model.py
git commit -m "fix(model): compute real Recent values UI-side (detector never persists them)"
```

---

## Task 9: Drift Metrics table — grid alignment

**Files:**
- Modify: `src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py`
- Test: `tests/test_spec3c_model.py` (existing `test_drift_tab_row_per_metric_and_click` must still pass)

- [ ] **Step 1: Add an alignment-structure test** to `tests/test_spec3c_model.py`:

```python
def test_drift_tab_uses_grid_columns(tk_root):
    """Header and rows share the same weighted grid columns, so they line up."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab, _COLUMNS
    tab = DriftMetricsTab(tk_root, theme=ThemeManager(), on_metric_select=lambda _: None)
    tab.set_status(_status())
    row = tab._rows["linearity_error"]
    # one gridded cell per column, and the column weights are configured
    assert len(row.grid_slaves()) == len(_COLUMNS)
    assert row.grid_columnconfigure(0)["weight"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_spec3c_model.py -k drift_tab_uses_grid -p no:warnings -q`
Expected: FAIL (rows currently use `pack`, so `grid_slaves()` is empty / weight 0).

- [ ] **Step 3: Implement** — rewrite the layout in `src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py` to use grid with shared weighted columns. Replace the header build (in `DriftMetricsTab.__init__`) and the `_MetricRow` cell loop:

Header (configure columns + grid the labels):

```python
        header = ctk.CTkFrame(self, fg_color=theme.CARD)
        header.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        for i in range(len(_COLUMNS)):
            header.grid_columnconfigure(i, weight=1, uniform="dm")
        for i, col in enumerate(_COLUMNS):
            ctk.CTkLabel(header, text=col, font=theme.font(theme.SIZE_CAPTION, "bold"),
                         text_color=theme.TEXT_SECONDARY, anchor="w")\
                .grid(row=0, column=i, sticky="ew", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
```

`_MetricRow` (grid the cells with the same weighted/uniform columns, left-aligned):

```python
        for i in range(len(cells)):
            self.grid_columnconfigure(i, weight=1, uniform="dm")
        for i, txt in enumerate(cells):
            lbl = ctk.CTkLabel(self, text=txt, font=theme.font(theme.SIZE_BODY),
                               text_color=theme.TEXT_PRIMARY, anchor="w")
            lbl.grid(row=0, column=i, sticky="ew", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            lbl.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())
```

(The `cells = [...]` list and `recent_override` handling from Task 8 stay; only the rendering switches from `pack` to `grid`. `DriftMetricsTab` remains a `CTkScrollableFrame` and rows are still `pack`ed into it — only the *cells within* the header and each row move to grid, which is what aligns the columns since both use the same `uniform="dm"` weighted config.)

- [ ] **Step 4: Run to verify pass** (full 3c file)

Run: `.venv/bin/python -m pytest tests/test_spec3c_model.py -p no:warnings -q`
Expected: PASS (incl. the existing `test_drift_tab_row_per_metric_and_click`).

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py tests/test_spec3c_model.py
git commit -m "fix(model): grid-align the Drift Metrics table (shared weighted columns)"
```

---

## Task 10: Full regression + manual-smoke checkpoint

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -p no:warnings -q`
Expected: 0 failed, 0 errors. (Baseline 279 + new dashboard/model tests.)

- [ ] **Step 2: Import smoke**

Run: `.venv/bin/python -c "from laser_trim_analyzer.gui.v6.app import V6App; from laser_trim_analyzer.gui.v6.pages.dashboard_page import DashboardPage; print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Hand back to James** for a manual `--v6` smoke: app opens on **Dashboard**; yield panels + trend render; lowest-yield list click → Model; Model now opens on the flagged metric with a populated **Recent** column and an aligned table. Then `superpowers:finishing-a-development-branch` (keep-as-is; Graduation still gated).

---

## Self-review notes (addressed)

- **Spec coverage:** yield panels (T4/T6), trend (T3/T4), both sources (T1 parameterized + T6 wires DBAR/DBFT), worst-models clickable→Model (T5/T6), landing (T6), pass-rate definition (T1), Q9 empty states (T4/T5/T6), Q10 cap (T5), Model worst-metric (T7), UI Recent (T8), grid table (T9). All covered.
- **Deviation from spec §1.5:** the spec floated reusing `get_dashboard_stats`; it has no Warning breakout and a different denominator, so it can't produce the Pass/Warn/Fail panel. The plan uses one purpose-built `compute_yield` for both panels instead — same intent, cleaner, DRY across trim+FT. Recorded here.
- **Type consistency:** `compute_yield`/`worst_models_by_yield` dict keys are used identically in the widgets and page; `_resolve_focus_metric`, `_recent_means`, and `set_status(recent_means=...)` signatures match across tasks.
