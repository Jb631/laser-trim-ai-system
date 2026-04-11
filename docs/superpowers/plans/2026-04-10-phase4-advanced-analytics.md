# Phase 4: Advanced Analytics & Visualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the app into an "Open and Know" analytics platform with dashboard overhaul (attention cards, overall health, process health), model scorecards, enhanced trends (Cpk/yield/drift timelines), Cpk/Ppk process capability analysis, failure pattern analysis, spec-aware ML enhancements, and executive reporting.

**Architecture:** New `cpk.py` module for Cpk/Ppk calculations. New `scorecard.py` page for per-model summaries. Dashboard restructured into sections (attention, health, process). Trends page gains comparative and capability views. ML predictor gains spec-aware features. Excel export gains executive summary sheet. All new queries added to existing `manager.py`.

**Tech Stack:** SQLAlchemy 2.0, customtkinter, matplotlib (lazy via ChartWidget), numpy/scipy (for Cpk), existing ML stack (scikit-learn RandomForest, CUSUM/EWMA drift), openpyxl (Excel export)

**Prerequisites:** Phases 1-3 complete — model_specs table exists, spec-aware analysis engine works, smoothness analysis available.

**Dev Environment Notes:**
- No pytest installed — use `python3 -c "import ast; ast.parse(open('file').read())"` for syntax checks
- No pydantic — use dataclasses or plain dicts
- Runtime: `python3` (not `python`)
- Working directory: `/Users/jb631/projects/laser-trim-ai-system-v5/`
- Branch: `v5-upgrade`

---

### Task 1: Create Cpk/Ppk Calculation Module

**Files:**
- Create: `src/laser_trim_analyzer/core/cpk.py`

- [ ] **Step 1: Create the Cpk/Ppk calculation module**

Create `src/laser_trim_analyzer/core/cpk.py` with functions for process capability analysis. This module calculates Cpk (within-subgroup capability) and Ppk (overall performance) using linearity deviation data against spec limits from model_specs.

```python
"""
Cpk/Ppk process capability analysis for linearity data.

Cpk uses within-subgroup variation (short-term capability).
Ppk uses overall variation (long-term performance).
Both require a spec limit — sourced from model_specs.linearity_spec_pct.

For linearity, the spec is symmetric: ±spec_pct, so:
  USL = +spec_pct, LSL = -spec_pct
  Cpk = min((USL - mean) / (3*sigma), (mean - LSL) / (3*sigma))
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CpkResult:
    """Result of a Cpk/Ppk calculation."""
    cpk: Optional[float] = None
    ppk: Optional[float] = None
    cp: Optional[float] = None
    pp: Optional[float] = None
    mean: Optional[float] = None
    std_within: Optional[float] = None
    std_overall: Optional[float] = None
    usl: Optional[float] = None
    lsl: Optional[float] = None
    n_samples: int = 0
    rating: str = "Unknown"  # "Excellent", "Capable", "Marginal", "Incapable"

    def to_dict(self):
        return {
            "cpk": self.cpk,
            "ppk": self.ppk,
            "cp": self.cp,
            "pp": self.pp,
            "mean": self.mean,
            "std_within": self.std_within,
            "std_overall": self.std_overall,
            "usl": self.usl,
            "lsl": self.lsl,
            "n_samples": self.n_samples,
            "rating": self.rating,
        }


def rate_cpk(cpk_value: Optional[float]) -> str:
    """Rate a Cpk value according to industry standards."""
    if cpk_value is None:
        return "Unknown"
    if cpk_value >= 1.67:
        return "Excellent"
    elif cpk_value >= 1.33:
        return "Capable"
    elif cpk_value >= 1.0:
        return "Marginal"
    else:
        return "Incapable"


def calculate_cpk(
    deviations: List[float],
    spec_limit_pct: float,
    subgroup_size: int = 5,
) -> CpkResult:
    """
    Calculate Cpk and Ppk for a set of linearity deviation values.

    Args:
        deviations: List of linearity deviation values (as % of applied voltage).
                    These are the raw deviation measurements, not absolute values.
        spec_limit_pct: The spec limit as a percentage (e.g., 0.5 for ±0.5%).
                        Symmetric spec: USL = +spec_limit_pct, LSL = -spec_limit_pct.
        subgroup_size: Size of rational subgroups for within-group sigma estimation.
                       Default 5 (typical for SPC).

    Returns:
        CpkResult with Cpk, Ppk, Cp, Pp, and supporting statistics.
    """
    import numpy as np

    result = CpkResult()

    if not deviations or len(deviations) < 2:
        logger.warning("Cpk calculation requires at least 2 data points")
        return result

    data = np.array(deviations, dtype=float)
    data = data[~np.isnan(data)]

    if len(data) < 2:
        return result

    result.n_samples = len(data)
    result.usl = spec_limit_pct
    result.lsl = -spec_limit_pct
    result.mean = float(np.mean(data))

    # Overall standard deviation (for Ppk)
    result.std_overall = float(np.std(data, ddof=1))

    # Within-subgroup standard deviation (for Cpk)
    # Use moving range method if subgroup_size == 1, else use pooled std
    if subgroup_size <= 1 or len(data) < subgroup_size * 2:
        # Moving range method (MR-bar / d2)
        moving_ranges = np.abs(np.diff(data))
        mr_bar = np.mean(moving_ranges)
        d2 = 1.128  # d2 constant for n=2
        result.std_within = float(mr_bar / d2) if mr_bar > 0 else result.std_overall
    else:
        # Pooled standard deviation from subgroups
        n_subgroups = len(data) // subgroup_size
        subgroups = [
            data[i * subgroup_size : (i + 1) * subgroup_size]
            for i in range(n_subgroups)
        ]
        # R-bar / d2 method
        ranges = [np.max(sg) - np.min(sg) for sg in subgroups]
        r_bar = np.mean(ranges)
        # d2 constants for subgroup sizes 2-10
        d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
                    6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
        d2 = d2_table.get(subgroup_size, 2.326)
        result.std_within = float(r_bar / d2) if r_bar > 0 else result.std_overall

    # Calculate capability indices
    usl = result.usl
    lsl = result.lsl
    mean = result.mean

    if result.std_within and result.std_within > 0:
        result.cp = float((usl - lsl) / (6 * result.std_within))
        cpu = (usl - mean) / (3 * result.std_within)
        cpl = (mean - lsl) / (3 * result.std_within)
        result.cpk = float(min(cpu, cpl))

    if result.std_overall and result.std_overall > 0:
        result.pp = float((usl - lsl) / (6 * result.std_overall))
        ppu = (usl - mean) / (3 * result.std_overall)
        ppl = (mean - lsl) / (3 * result.std_overall)
        result.ppk = float(min(ppu, ppl))

    result.rating = rate_cpk(result.cpk)

    logger.debug(
        f"Cpk={result.cpk:.3f}, Ppk={result.ppk:.3f}, "
        f"n={result.n_samples}, rating={result.rating}"
        if result.cpk is not None else f"Cpk calculation failed, n={result.n_samples}"
    )

    return result


def calculate_cpk_trend(
    deviations_by_period: List[Tuple[str, List[float]]],
    spec_limit_pct: float,
) -> List[dict]:
    """
    Calculate Cpk for each time period to show capability trend.

    Args:
        deviations_by_period: List of (period_label, deviations) tuples.
                              period_label is e.g. "2026-03" or "2026-W14".
        spec_limit_pct: The spec limit percentage.

    Returns:
        List of dicts with keys: period, cpk, ppk, n_samples, rating
    """
    results = []
    for period_label, devs in deviations_by_period:
        r = calculate_cpk(devs, spec_limit_pct, subgroup_size=1)
        results.append({
            "period": period_label,
            "cpk": r.cpk,
            "ppk": r.ppk,
            "n_samples": r.n_samples,
            "rating": r.rating,
        })
    return results
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/cpk.py').read()); print('OK')"
```

- [ ] **Step 3: Smoke test the calculation**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.core.cpk import calculate_cpk
import random
# Simulate linearity deviations centered near 0 with some spread
data = [random.gauss(0.02, 0.1) for _ in range(100)]
result = calculate_cpk(data, spec_limit_pct=0.5)
print(f'Cpk={result.cpk:.3f}, Ppk={result.ppk:.3f}, rating={result.rating}, n={result.n_samples}')
print(f'Mean={result.mean:.4f}, Sigma_w={result.std_within:.4f}, Sigma_o={result.std_overall:.4f}')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/core/cpk.py
git commit -m "feat: add Cpk/Ppk process capability calculation module"
```

---

### Task 2: Add Cpk Database Queries

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`

- [ ] **Step 1: Add query to get linearity deviations for Cpk calculation**

Add these methods to `DatabaseManager`. They pull the raw deviation data needed for Cpk calculations, grouped by model and optionally by time period.

```python
def get_linearity_deviations_for_cpk(
    self, model: str, days_back: int = 90
) -> List[float]:
    """
    Get raw linearity deviation values for a model, for Cpk calculation.

    Returns the max_linearity_deviation values from analysis results
    (the worst-point deviation per unit, as a percentage).
    """
    with self.session() as session:
        cutoff = datetime.now() - timedelta(days=days_back)
        results = session.query(
            DBAnalysisResult.max_linearity_deviation
        ).filter(
            DBAnalysisResult.model == model,
            DBAnalysisResult.file_date >= cutoff,
            DBAnalysisResult.max_linearity_deviation.isnot(None),
        ).all()
        return [r[0] for r in results]


def get_cpk_by_model(self, days_back: int = 90) -> List[Dict[str, Any]]:
    """
    Calculate Cpk for each model that has a linearity spec defined.

    Returns list of dicts: model, cpk, ppk, rating, n_samples, spec_pct
    """
    from laser_trim_analyzer.core.cpk import calculate_cpk

    with self.session() as session:
        # Get all models with specs
        specs = session.query(ModelSpec).filter(
            ModelSpec.linearity_spec_pct.isnot(None)
        ).all()

    results = []
    for spec in specs:
        devs = self.get_linearity_deviations_for_cpk(spec.model, days_back)
        if len(devs) < 10:  # Need meaningful sample size
            continue
        cpk_result = calculate_cpk(devs, spec.linearity_spec_pct)
        results.append({
            "model": spec.model,
            "cpk": cpk_result.cpk,
            "ppk": cpk_result.ppk,
            "rating": cpk_result.rating,
            "n_samples": cpk_result.n_samples,
            "spec_pct": spec.linearity_spec_pct,
            "mean": cpk_result.mean,
        })

    # Sort by Cpk ascending (worst first)
    results.sort(key=lambda x: x["cpk"] if x["cpk"] is not None else 999)
    return results


def get_cpk_trend_for_model(
    self, model: str, spec_limit_pct: float,
    days_back: int = 180, period: str = "month"
) -> List[Dict[str, Any]]:
    """
    Get Cpk trend over time for a specific model.

    Args:
        model: Model name
        spec_limit_pct: Spec limit from model_specs
        days_back: How far back to look
        period: "week" or "month"

    Returns:
        List of dicts with: period, cpk, ppk, n_samples, rating
    """
    from laser_trim_analyzer.core.cpk import calculate_cpk_trend

    with self.session() as session:
        cutoff = datetime.now() - timedelta(days=days_back)

        if period == "week":
            period_expr = func.strftime('%Y-W%W', DBAnalysisResult.file_date)
        else:
            period_expr = func.strftime('%Y-%m', DBAnalysisResult.file_date)

        results = session.query(
            period_expr.label("period"),
            DBAnalysisResult.max_linearity_deviation,
        ).filter(
            DBAnalysisResult.model == model,
            DBAnalysisResult.file_date >= cutoff,
            DBAnalysisResult.max_linearity_deviation.isnot(None),
        ).order_by(period_expr).all()

    # Group by period
    from collections import defaultdict
    period_data = defaultdict(list)
    for r in results:
        period_data[r.period].append(r.max_linearity_deviation)

    deviations_by_period = sorted(period_data.items())
    return calculate_cpk_trend(deviations_by_period, spec_limit_pct)
```

- [ ] **Step 2: Add failure pattern analysis queries**

Add queries to support the failure pattern heatmap and mode categorization:

```python
def get_failure_position_data(
    self, model: Optional[str] = None, days_back: int = 90
) -> List[Dict[str, Any]]:
    """
    Get position-based failure data for heatmap visualization.

    Returns which measurement points (positions along the element) fail most
    frequently, enabling identification of position-dependent failure patterns.
    """
    with self.session() as session:
        cutoff = datetime.now() - timedelta(days=days_back)

        query = session.query(
            DBAnalysisResult.id,
            DBAnalysisResult.model,
            DBAnalysisResult.max_linearity_deviation,
            DBAnalysisResult.max_linearity_position,
            DBAnalysisResult.overall_status,
        ).filter(
            DBAnalysisResult.file_date >= cutoff,
            DBAnalysisResult.max_linearity_deviation.isnot(None),
        )

        if model:
            query = query.filter(DBAnalysisResult.model == model)

        results = query.all()

        return [
            {
                "id": r.id,
                "model": r.model,
                "max_deviation": r.max_linearity_deviation,
                "max_position": r.max_linearity_position,
                "status": str(r.overall_status),
            }
            for r in results
        ]


def get_failure_mode_summary(
    self, days_back: int = 90
) -> List[Dict[str, Any]]:
    """
    Categorize failures by mode: linearity, sigma, resistance, etc.

    Returns counts per failure mode to identify the dominant failure types.
    """
    with self.session() as session:
        cutoff = datetime.now() - timedelta(days=days_back)

        # Get all non-pass results
        results = session.query(
            DBAnalysisResult.overall_status,
            DBAnalysisResult.linearity_status,
            DBAnalysisResult.sigma_status,
            DBAnalysisResult.model,
        ).filter(
            DBAnalysisResult.file_date >= cutoff,
            DBAnalysisResult.overall_status != DBStatusType.PASS,
        ).all()

        # Categorize
        modes = {
            "Linearity Fail": 0,
            "Sigma Fail": 0,
            "Both Fail": 0,
            "Other": 0,
        }
        for r in results:
            lin_fail = r.linearity_status and str(r.linearity_status) != "PASS"
            sig_fail = r.sigma_status and str(r.sigma_status) != "PASS"
            if lin_fail and sig_fail:
                modes["Both Fail"] += 1
            elif lin_fail:
                modes["Linearity Fail"] += 1
            elif sig_fail:
                modes["Sigma Fail"] += 1
            else:
                modes["Other"] += 1

        return [
            {"mode": mode, "count": count}
            for mode, count in modes.items()
            if count > 0
        ]
```

- [ ] **Step 3: Add model scorecard summary query**

```python
def get_model_scorecard_data(
    self, model: str, days_back: int = 90
) -> Dict[str, Any]:
    """
    Get comprehensive scorecard data for a single model.

    Combines: pass rate, Cpk, drift status, ML confidence, failure patterns,
    volume, trend direction, and spec info.
    """
    with self.session() as session:
        cutoff = datetime.now() - timedelta(days=days_back)

        # Basic stats
        total = session.query(func.count(DBAnalysisResult.id)).filter(
            DBAnalysisResult.model == model,
            DBAnalysisResult.file_date >= cutoff,
        ).scalar() or 0

        passed = session.query(func.count(DBAnalysisResult.id)).filter(
            DBAnalysisResult.model == model,
            DBAnalysisResult.file_date >= cutoff,
            DBAnalysisResult.overall_status == DBStatusType.PASS,
        ).scalar() or 0

        pass_rate = (passed / total * 100) if total > 0 else 0

        # Average linearity deviation
        avg_dev = session.query(
            func.avg(DBAnalysisResult.max_linearity_deviation)
        ).filter(
            DBAnalysisResult.model == model,
            DBAnalysisResult.file_date >= cutoff,
            DBAnalysisResult.max_linearity_deviation.isnot(None),
        ).scalar()

        # Get model spec
        spec = session.query(ModelSpec).filter(
            ModelSpec.model == model
        ).first()

    # Cpk if spec available
    cpk_data = None
    if spec and spec.linearity_spec_pct:
        from laser_trim_analyzer.core.cpk import calculate_cpk
        devs = self.get_linearity_deviations_for_cpk(model, days_back)
        if len(devs) >= 10:
            cpk_result = calculate_cpk(devs, spec.linearity_spec_pct)
            cpk_data = cpk_result.to_dict()

    # Drift status from ML state
    drift_status = None
    try:
        ml_state = self.get_model_ml_state(model)
        if ml_state:
            drift_status = ml_state.get("drift_status")
    except Exception:
        pass

    return {
        "model": model,
        "total": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "avg_deviation": avg_dev,
        "cpk": cpk_data,
        "drift_status": drift_status,
        "spec": {
            "element_type": spec.element_type if spec else None,
            "product_class": spec.product_class if spec else None,
            "linearity_type": spec.linearity_type if spec else None,
            "linearity_spec_pct": spec.linearity_spec_pct if spec else None,
        } if spec else None,
    }
```

- [ ] **Step 4: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py
git commit -m "feat: add Cpk, failure pattern, and scorecard queries to DatabaseManager"
```

---

### Task 3: Dashboard Overhaul — "Open and Know" Design

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/dashboard.py`

This is the largest single task. The dashboard is restructured into three visual sections:

1. **Attention Cards** (top) — color-coded cards that highlight models/issues needing immediate action
2. **Overall Health** (middle) — key metrics: overall pass rate, worst models, yield trend sparkline
3. **Process Health** (bottom) — Cpk summary, drift alerts, failure mode breakdown

The existing dashboard content is preserved but reorganized. New content is added around it.

- [ ] **Step 1: Add attention card system**

At the top of the dashboard, add a horizontal row of "attention cards" — colored frames that highlight what needs action right now. Each card shows a model or issue with a severity color (red = critical, orange = warning, green = healthy).

Read the current `dashboard.py` to identify the layout structure (grid rows/columns), then add the attention card row as the first visual element.

The attention card data comes from a new helper method:

```python
def _get_attention_items(self, stats: dict) -> List[dict]:
    """
    Build list of attention items from dashboard stats.

    Each item: {title, subtitle, color, severity, value}
    Severity: "critical" (red), "warning" (orange), "info" (blue), "good" (green)
    """
    items = []

    # Models with worst pass rates
    try:
        from laser_trim_analyzer.database import get_database
        db = get_database()

        # Get models sorted by pass rate (worst first)
        model_stats = stats.get("model_performance", [])
        for m in model_stats[:3]:
            if m.get("pass_rate", 100) < 70:
                items.append({
                    "title": m["model"],
                    "subtitle": f"{m['pass_rate']:.0f}% pass rate",
                    "value": f"{m.get('total', 0)} units",
                    "severity": "critical" if m["pass_rate"] < 50 else "warning",
                })

        # Drift alerts
        # Check ML drift states for active models
        try:
            drift_models = db.get_models_with_drift_alert()
            for dm in drift_models[:2]:
                items.append({
                    "title": f"{dm['model']} Drift",
                    "subtitle": dm.get("drift_type", "Process shift detected"),
                    "value": "Action needed",
                    "severity": "warning",
                })
        except Exception:
            pass  # Method may not exist yet

        # Low Cpk models
        try:
            cpk_data = db.get_cpk_by_model(days_back=30)
            for c in cpk_data[:2]:
                if c["cpk"] is not None and c["cpk"] < 1.0:
                    items.append({
                        "title": f"{c['model']}",
                        "subtitle": f"Cpk = {c['cpk']:.2f} ({c['rating']})",
                        "value": f"Spec: {c['spec_pct']:.1f}%",
                        "severity": "critical" if c["cpk"] < 0.67 else "warning",
                    })
        except Exception:
            pass

    except Exception:
        pass

    # If nothing needs attention, show a "healthy" card
    if not items:
        items.append({
            "title": "All Clear",
            "subtitle": "No models need immediate attention",
            "value": "",
            "severity": "good",
        })

    return items
```

Create the card widgets:

```python
def _create_attention_section(self, parent):
    """Create the attention cards row at the top of the dashboard."""
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))
    frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

    self.attention_frame = frame
    return frame


def _update_attention_cards(self, items: List[dict]):
    """Update attention card display with current items."""
    # Clear existing cards
    for widget in self.attention_frame.winfo_children():
        widget.destroy()

    SEVERITY_COLORS = {
        "critical": ("#dc3545", "#fff"),
        "warning": ("#fd7e14", "#fff"),
        "info": ("#0d6efd", "#fff"),
        "good": ("#198754", "#fff"),
    }

    for i, item in enumerate(items[:4]):  # Max 4 cards
        bg, fg = SEVERITY_COLORS.get(item["severity"], ("#6c757d", "#fff"))

        card = ctk.CTkFrame(self.attention_frame, fg_color=bg, corner_radius=8)
        card.grid(row=0, column=i, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(
            card, text=item["title"],
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=fg,
        ).pack(padx=10, pady=(8, 2), anchor="w")

        ctk.CTkLabel(
            card, text=item["subtitle"],
            font=ctk.CTkFont(size=11),
            text_color=fg,
        ).pack(padx=10, pady=0, anchor="w")

        if item.get("value"):
            ctk.CTkLabel(
                card, text=item["value"],
                font=ctk.CTkFont(size=10),
                text_color=fg,
            ).pack(padx=10, pady=(0, 8), anchor="w")
        else:
            # Add bottom padding
            ctk.CTkLabel(card, text="", height=4).pack()
```

- [ ] **Step 2: Add overall health section**

Below the attention cards, add an "Overall Health" section with:
- Large pass rate percentage (colored by threshold)
- Total units processed count
- Yield trend sparkline (last 30 days as a small inline chart)
- Top 5 worst models mini-table

```python
def _create_health_section(self, parent, row: int):
    """Create the overall health metrics section."""
    frame = ctk.CTkFrame(parent)
    frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
    frame.grid_columnconfigure((0, 1, 2), weight=1)

    ctk.CTkLabel(
        frame, text="Overall Health",
        font=ctk.CTkFont(size=16, weight="bold"),
    ).grid(row=0, column=0, columnspan=3, padx=15, pady=(10, 5), sticky="w")

    # Pass rate (large)
    self.health_pass_rate = ctk.CTkLabel(
        frame, text="--.--%",
        font=ctk.CTkFont(size=36, weight="bold"),
    )
    self.health_pass_rate.grid(row=1, column=0, padx=15, pady=5)

    ctk.CTkLabel(
        frame, text="Pass Rate",
        font=ctk.CTkFont(size=11), text_color="gray",
    ).grid(row=2, column=0, padx=15, pady=(0, 10))

    # Total units
    self.health_total = ctk.CTkLabel(
        frame, text="---",
        font=ctk.CTkFont(size=24, weight="bold"),
    )
    self.health_total.grid(row=1, column=1, padx=15, pady=5)

    ctk.CTkLabel(
        frame, text="Units Processed",
        font=ctk.CTkFont(size=11), text_color="gray",
    ).grid(row=2, column=1, padx=15, pady=(0, 10))

    # Yield trend placeholder (small chart area)
    self.health_trend_frame = ctk.CTkFrame(frame, width=200, height=80)
    self.health_trend_frame.grid(row=1, column=2, rowspan=2, padx=15, pady=10)

    self.health_frame = frame
    return frame
```

- [ ] **Step 3: Add process health section**

Below overall health, add the process health section with:
- Cpk summary (how many models are Excellent/Capable/Marginal/Incapable)
- Failure mode breakdown (pie chart: linearity vs sigma vs both)
- Active drift alerts count

```python
def _create_process_section(self, parent, row: int):
    """Create the process health section with Cpk and failure patterns."""
    frame = ctk.CTkFrame(parent)
    frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
    frame.grid_columnconfigure((0, 1), weight=1)

    ctk.CTkLabel(
        frame, text="Process Health",
        font=ctk.CTkFont(size=16, weight="bold"),
    ).grid(row=0, column=0, columnspan=2, padx=15, pady=(10, 5), sticky="w")

    # Cpk summary frame (left)
    cpk_frame = ctk.CTkFrame(frame)
    cpk_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    ctk.CTkLabel(
        cpk_frame, text="Process Capability (Cpk)",
        font=ctk.CTkFont(size=13, weight="bold"),
    ).pack(padx=10, pady=(10, 5), anchor="w")

    self.cpk_summary_label = ctk.CTkLabel(
        cpk_frame, text="Loading...",
        font=ctk.CTkFont(size=11), justify="left",
    )
    self.cpk_summary_label.pack(padx=10, pady=5, anchor="w")

    # Failure mode frame (right)
    fail_frame = ctk.CTkFrame(frame)
    fail_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    ctk.CTkLabel(
        fail_frame, text="Failure Mode Breakdown",
        font=ctk.CTkFont(size=13, weight="bold"),
    ).pack(padx=10, pady=(10, 5), anchor="w")

    self.failure_mode_label = ctk.CTkLabel(
        fail_frame, text="Loading...",
        font=ctk.CTkFont(size=11), justify="left",
    )
    self.failure_mode_label.pack(padx=10, pady=5, anchor="w")

    self.process_frame = frame
    return frame
```

- [ ] **Step 4: Wire up data loading for new sections**

In the dashboard's `_load_data()` or `on_show()` method, add background data loading for the new sections. After existing stats load completes, fetch attention items, Cpk summary, and failure modes:

```python
def _update_new_sections(self, stats: dict):
    """Update the new dashboard sections with loaded data."""
    # Attention cards
    attention_items = self._get_attention_items(stats)
    self._update_attention_cards(attention_items)

    # Overall health
    pass_rate = stats.get("overall_pass_rate", 0)
    total = stats.get("total_processed", 0)
    color = "#198754" if pass_rate >= 80 else "#fd7e14" if pass_rate >= 60 else "#dc3545"
    self.health_pass_rate.configure(text=f"{pass_rate:.1f}%", text_color=color)
    self.health_total.configure(text=f"{total:,}")

    # Process health - Cpk
    try:
        from laser_trim_analyzer.database import get_database
        db = get_database()
        cpk_data = db.get_cpk_by_model(days_back=30)
        if cpk_data:
            ratings = {"Excellent": 0, "Capable": 0, "Marginal": 0, "Incapable": 0}
            for c in cpk_data:
                r = c.get("rating", "Unknown")
                if r in ratings:
                    ratings[r] += 1
            text = "\n".join(f"  {r}: {n} models" for r, n in ratings.items() if n > 0)
            self.cpk_summary_label.configure(text=text or "No Cpk data available")
        else:
            self.cpk_summary_label.configure(text="No models with specs for Cpk")
    except Exception as e:
        self.cpk_summary_label.configure(text=f"Cpk unavailable: {e}")

    # Process health - Failure modes
    try:
        from laser_trim_analyzer.database import get_database
        db = get_database()
        modes = db.get_failure_mode_summary(days_back=30)
        if modes:
            text = "\n".join(f"  {m['mode']}: {m['count']}" for m in modes)
            self.failure_mode_label.configure(text=text)
        else:
            self.failure_mode_label.configure(text="No failures in period")
    except Exception:
        self.failure_mode_label.configure(text="Failure data unavailable")
```

- [ ] **Step 5: Preserve all existing dashboard content**

The existing dashboard sections (Pareto, P-chart, model performance table, linearity quality card, sigma health card, category breakdown charts, etc.) must remain. Move them below the new sections by adjusting their grid row numbers. The new layout order is:

1. Row 0: Attention cards
2. Row 1: Overall health
3. Row 2: Process health
4. Row 3+: Existing dashboard content (shifted down)

Read the current row assignments in `dashboard.py` and increment each by 3.

- [ ] **Step 6: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/dashboard.py').read()); print('OK')"
```

- [ ] **Step 7: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/dashboard.py
git commit -m "feat: dashboard overhaul with attention cards, health sections, Cpk summary"
```

---

### Task 4: Model Scorecard Page

**Files:**
- Create: `src/laser_trim_analyzer/gui/pages/scorecard.py`
- Modify: `src/laser_trim_analyzer/app.py`
- Modify: `src/laser_trim_analyzer/gui/pages/analyze.py`

The model scorecard is a single-page summary for any model, accessible from the Analyze page or Dashboard. It shows pass rate, Cpk, drift status, ML confidence, failure patterns, and spec info — everything needed to understand a model's health at a glance.

- [ ] **Step 1: Create the scorecard page**

Create `src/laser_trim_analyzer/gui/pages/scorecard.py`. This page is a scrollable single-model summary with sections:

```python
"""
Model Scorecard — single-page summary of a model's health and performance.

Sections:
1. Header: Model name, element type, product class
2. Key Metrics: Pass rate (large), Cpk, volume, trend arrow
3. Specifications: Linearity spec, resistance range, angle, smoothness
4. Cpk Detail: Cpk/Ppk values, histogram of deviations vs spec limits
5. Drift Status: Current CUSUM/EWMA status, last drift event
6. Failure Patterns: Position-based failure heat, mode breakdown
7. ML Confidence: Predictor confidence, feature importances
"""

import logging
from typing import Optional

import customtkinter as ctk

logger = logging.getLogger(__name__)


class ScorecardPage(ctk.CTkFrame):
    """Model scorecard — comprehensive single-model view."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.current_model: Optional[str] = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Model selector at top
        selector_frame = ctk.CTkFrame(self, fg_color="transparent")
        selector_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            selector_frame, text="Model Scorecard",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left", padx=(0, 20))

        self.model_selector = ctk.CTkComboBox(
            selector_frame,
            values=["Select a model..."],
            command=self._on_model_selected,
            width=200,
        )
        self.model_selector.pack(side="left", padx=5)

        ctk.CTkButton(
            selector_frame, text="Refresh", width=80,
            command=self._refresh,
        ).pack(side="left", padx=5)

        # Scrollable content area
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        # Placeholder
        self.placeholder = ctk.CTkLabel(
            self.scroll_frame,
            text="Select a model to view its scorecard",
            font=ctk.CTkFont(size=14),
            text_color="gray",
        )
        self.placeholder.grid(row=0, column=0, pady=50)

    def on_show(self):
        """Called when page becomes visible."""
        self._populate_model_list()
        if self.current_model:
            self._load_scorecard(self.current_model)

    def show_model(self, model: str):
        """Show scorecard for a specific model (called from other pages)."""
        self.current_model = model
        self.model_selector.set(model)
        self._load_scorecard(model)

    def _populate_model_list(self):
        """Load available models into the dropdown."""
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            models = db.get_distinct_models()
            if models:
                self.model_selector.configure(values=models)
        except Exception:
            pass

    def _on_model_selected(self, model: str):
        """Handle model selection from dropdown."""
        if model and model != "Select a model...":
            self.current_model = model
            self._load_scorecard(model)

    def _refresh(self):
        """Refresh current scorecard."""
        if self.current_model:
            self._load_scorecard(self.current_model)

    def _load_scorecard(self, model: str):
        """Load and display scorecard data for a model."""
        # Clear existing content
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            data = db.get_model_scorecard_data(model, days_back=90)
            self._render_scorecard(data)
        except Exception as e:
            logger.error(f"Failed to load scorecard for {model}: {e}")
            ctk.CTkLabel(
                self.scroll_frame,
                text=f"Error loading scorecard: {e}",
                text_color="#dc3545",
            ).grid(row=0, column=0, pady=20)

    def _render_scorecard(self, data: dict):
        """Render the scorecard sections from loaded data."""
        row = 0

        # Section 1: Header
        row = self._render_header(data, row)

        # Section 2: Key Metrics
        row = self._render_key_metrics(data, row)

        # Section 3: Specifications
        if data.get("spec"):
            row = self._render_specs(data, row)

        # Section 4: Cpk Detail
        if data.get("cpk"):
            row = self._render_cpk(data, row)

        # Section 5: Drift Status
        if data.get("drift_status"):
            row = self._render_drift(data, row)

    def _render_header(self, data: dict, row: int) -> int:
        """Render model header with name and classification."""
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text=data["model"],
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, padx=15, pady=(10, 2), sticky="w")

        spec = data.get("spec", {})
        if spec:
            parts = []
            if spec.get("element_type"):
                parts.append(spec["element_type"])
            if spec.get("product_class"):
                parts.append(f"Class {spec['product_class']}")
            if spec.get("linearity_type"):
                parts.append(spec["linearity_type"])
            if parts:
                ctk.CTkLabel(
                    frame, text=" | ".join(parts),
                    font=ctk.CTkFont(size=12), text_color="gray",
                ).grid(row=1, column=0, padx=15, pady=(0, 10), sticky="w")

        return row + 1

    def _render_key_metrics(self, data: dict, row: int) -> int:
        """Render the key metrics row: pass rate, volume, Cpk."""
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=5)
        frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Pass Rate
        pr = data.get("pass_rate", 0)
        pr_color = "#198754" if pr >= 80 else "#fd7e14" if pr >= 60 else "#dc3545"
        self._metric_card(frame, 0, f"{pr:.1f}%", "Pass Rate", pr_color)

        # Volume
        self._metric_card(frame, 1, f"{data.get('total', 0):,}", "Units (90d)", "#0d6efd")

        # Cpk
        cpk = data.get("cpk", {})
        if cpk and cpk.get("cpk") is not None:
            cpk_val = cpk["cpk"]
            cpk_color = "#198754" if cpk_val >= 1.33 else "#fd7e14" if cpk_val >= 1.0 else "#dc3545"
            self._metric_card(frame, 2, f"{cpk_val:.2f}", f"Cpk ({cpk.get('rating', '')})", cpk_color)
        else:
            self._metric_card(frame, 2, "N/A", "Cpk", "#6c757d")

        # Avg Deviation
        avg_dev = data.get("avg_deviation")
        if avg_dev is not None:
            self._metric_card(frame, 3, f"{avg_dev:.3f}%", "Avg Deviation", "#6c757d")
        else:
            self._metric_card(frame, 3, "N/A", "Avg Deviation", "#6c757d")

        return row + 1

    def _metric_card(self, parent, col: int, value: str, label: str, color: str):
        """Create a single metric card within a grid."""
        card = ctk.CTkFrame(parent)
        card.grid(row=0, column=col, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(
            card, text=value,
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=color,
        ).pack(padx=10, pady=(8, 0))

        ctk.CTkLabel(
            card, text=label,
            font=ctk.CTkFont(size=10), text_color="gray",
        ).pack(padx=10, pady=(0, 8))

    def _render_specs(self, data: dict, row: int) -> int:
        """Render the specifications section."""
        spec = data["spec"]
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=5)

        ctk.CTkLabel(
            frame, text="Specifications",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(padx=15, pady=(10, 5), anchor="w")

        details = []
        if spec.get("linearity_spec_pct"):
            details.append(f"Linearity: +/-{spec['linearity_spec_pct']}%")
        if spec.get("linearity_type"):
            details.append(f"Type: {spec['linearity_type']}")

        for d in details:
            ctk.CTkLabel(
                frame, text=f"  {d}",
                font=ctk.CTkFont(size=11),
            ).pack(padx=15, pady=1, anchor="w")

        # Bottom padding
        ctk.CTkLabel(frame, text="", height=5).pack()
        return row + 1

    def _render_cpk(self, data: dict, row: int) -> int:
        """Render Cpk detail section with values and interpretation."""
        cpk = data["cpk"]
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=5)

        ctk.CTkLabel(
            frame, text="Process Capability",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(padx=15, pady=(10, 5), anchor="w")

        lines = [
            f"  Cpk = {cpk['cpk']:.3f}   (within-subgroup)" if cpk.get('cpk') else None,
            f"  Ppk = {cpk['ppk']:.3f}   (overall)" if cpk.get('ppk') else None,
            f"  Cp  = {cpk['cp']:.3f}   (potential)" if cpk.get('cp') else None,
            f"  Mean = {cpk['mean']:.4f}%,  Sigma = {cpk['std_overall']:.4f}%" if cpk.get('mean') is not None else None,
            f"  Samples: {cpk.get('n_samples', 0):,}",
            f"  Spec limits: {cpk.get('lsl', 0):.2f}% to {cpk.get('usl', 0):.2f}%",
        ]
        for line in lines:
            if line:
                ctk.CTkLabel(
                    frame, text=line, font=ctk.CTkFont(size=11),
                ).pack(padx=15, pady=1, anchor="w")

        ctk.CTkLabel(frame, text="", height=5).pack()
        return row + 1

    def _render_drift(self, data: dict, row: int) -> int:
        """Render drift status section."""
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=5)

        ctk.CTkLabel(
            frame, text="Drift Status",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(padx=15, pady=(10, 5), anchor="w")

        status = data.get("drift_status", "Unknown")
        color = "#198754" if status == "stable" else "#fd7e14" if status == "warning" else "#dc3545"

        ctk.CTkLabel(
            frame, text=f"  Status: {status}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=color,
        ).pack(padx=15, pady=5, anchor="w")

        ctk.CTkLabel(frame, text="", height=5).pack()
        return row + 1
```

- [ ] **Step 2: Register scorecard page in app.py**

In `src/laser_trim_analyzer/app.py`:

Add the scorecard page import and registration. It should not appear in the main navigation sidebar (it is accessed via buttons on other pages), so register it as a hidden page:

```python
from laser_trim_analyzer.gui.pages.scorecard import ScorecardPage
self._pages["scorecard"] = ScorecardPage(self.main_frame, self)
```

Add a helper method to the app for navigating to a specific model's scorecard:

```python
def show_model_scorecard(self, model: str):
    """Navigate to the scorecard page for a specific model."""
    self._show_page("scorecard")
    self._pages["scorecard"].show_model(model)
```

- [ ] **Step 3: Add "View Scorecard" button to Analyze page**

In `src/laser_trim_analyzer/gui/pages/analyze.py`, add a button to navigate to the scorecard for the currently viewed model. Place it near the model info section:

```python
ctk.CTkButton(
    info_frame,
    text="View Scorecard",
    width=120,
    command=lambda: self.app.show_model_scorecard(self._current_model),
).pack(side="right", padx=5)
```

Find the appropriate location in the analyze page by reading the current file info / header area. Only show the button when a model is loaded.

- [ ] **Step 4: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/scorecard.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/app.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/analyze.py').read()); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/scorecard.py src/laser_trim_analyzer/app.py src/laser_trim_analyzer/gui/pages/analyze.py
git commit -m "feat: add Model Scorecard page with Cpk, drift, and spec details"
```

---

### Task 5: Enhanced Trends — Comparative, Cpk, Yield, Drift

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py`
- Modify: `src/laser_trim_analyzer/database/manager.py`

The Trends page gains four new chart types that can be selected via tabs or a dropdown:

1. **Comparative Trends** — overlay multiple models on the same chart for pass rate comparison
2. **Cpk Trend** — Cpk over time for a selected model, with capability threshold lines
3. **Yield Trend** — overall yield (all models) over time, with target line
4. **Drift Timeline** — show drift detection events (CUSUM/EWMA) over time

- [ ] **Step 1: Add yield trend query to database manager**

```python
def get_yield_trend(
    self, days_back: int = 180, period: str = "week"
) -> List[Dict[str, Any]]:
    """
    Get overall yield (pass rate) trend across all models.

    Returns list of dicts: period, total, passed, pass_rate
    """
    with self.session() as session:
        cutoff = datetime.now() - timedelta(days=days_back)

        if period == "week":
            period_expr = func.strftime('%Y-W%W', DBAnalysisResult.file_date)
        else:
            period_expr = func.strftime('%Y-%m', DBAnalysisResult.file_date)

        results = session.query(
            period_expr.label("period"),
            func.count(DBAnalysisResult.id).label("total"),
            func.sum(
                case(
                    (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                    else_=0
                )
            ).label("passed"),
        ).filter(
            DBAnalysisResult.file_date >= cutoff,
        ).group_by(period_expr).order_by(period_expr).all()

        return [
            {
                "period": r.period,
                "total": r.total,
                "passed": r.passed,
                "pass_rate": (r.passed / r.total * 100) if r.total > 0 else 0,
            }
            for r in results
        ]


def get_comparative_model_trends(
    self, models: List[str], days_back: int = 90, period: str = "week"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get pass rate trends for multiple models for overlay comparison.

    Returns dict keyed by model name, each value is a list of
    {period, pass_rate, total} dicts.
    """
    result = {}
    for model in models:
        with self.session() as session:
            cutoff = datetime.now() - timedelta(days=days_back)

            if period == "week":
                period_expr = func.strftime('%Y-W%W', DBAnalysisResult.file_date)
            else:
                period_expr = func.strftime('%Y-%m', DBAnalysisResult.file_date)

            rows = session.query(
                period_expr.label("period"),
                func.count(DBAnalysisResult.id).label("total"),
                func.sum(
                    case(
                        (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                        else_=0
                    )
                ).label("passed"),
            ).filter(
                DBAnalysisResult.model == model,
                DBAnalysisResult.file_date >= cutoff,
            ).group_by(period_expr).order_by(period_expr).all()

            result[model] = [
                {
                    "period": r.period,
                    "total": r.total,
                    "pass_rate": (r.passed / r.total * 100) if r.total > 0 else 0,
                }
                for r in rows
            ]
    return result


def get_drift_events_timeline(
    self, days_back: int = 180
) -> List[Dict[str, Any]]:
    """
    Get drift detection events over time for timeline visualization.

    Pulls from ModelMLState records where drift was detected.
    Returns list of {model, date, drift_type, severity} dicts.
    """
    with self.session() as session:
        cutoff = datetime.now() - timedelta(days=days_back)

        # Query ML states that have drift detected
        results = session.query(ModelMLState).filter(
            ModelMLState.updated_at >= cutoff,
            ModelMLState.drift_detected == True,
        ).order_by(ModelMLState.updated_at).all()

        return [
            {
                "model": r.model,
                "date": r.updated_at.isoformat() if r.updated_at else None,
                "drift_type": r.drift_type if hasattr(r, 'drift_type') else "detected",
            }
            for r in results
        ]
```

- [ ] **Step 2: Add trend chart type selector to Trends page**

Read the current `trends.py` to understand its layout, then add a chart type selector (segmented button or tab view) at the top that allows switching between:
- Existing trends (default)
- Comparative (multi-model overlay)
- Cpk Trend
- Yield Trend
- Drift Timeline

```python
# Add to the header area of the Trends page
self.trend_type_selector = ctk.CTkSegmentedButton(
    header_frame,
    values=["Standard", "Comparative", "Cpk Trend", "Yield", "Drift"],
    command=self._on_trend_type_changed,
)
self.trend_type_selector.set("Standard")
self.trend_type_selector.pack(side="left", padx=10)
```

- [ ] **Step 3: Implement comparative trends chart**

When "Comparative" is selected, show a multi-select for models and render overlaid line charts:

```python
def _show_comparative_trends(self):
    """Show comparative pass rate trends for selected models."""
    # Get top 5 models by volume as default selection
    from laser_trim_analyzer.database import get_database
    db = get_database()

    # Use existing model dropdown selection or top models
    models = self._get_selected_models()  # Implement based on current UI
    if not models:
        return

    data = db.get_comparative_model_trends(models, days_back=90)

    # Plot using ChartWidget or direct matplotlib
    fig = self._get_chart_figure()
    ax = fig.add_subplot(111)

    for model, trend in data.items():
        periods = [t["period"] for t in trend]
        rates = [t["pass_rate"] for t in trend]
        ax.plot(periods, rates, marker='o', markersize=4, label=model)

    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Comparative Pass Rate Trends")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    fig.tight_layout()
    self._update_chart_canvas(fig)
```

- [ ] **Step 4: Implement Cpk trend chart**

When "Cpk Trend" is selected, show Cpk over time for a single model with reference lines at 1.0, 1.33, and 1.67:

```python
def _show_cpk_trend(self):
    """Show Cpk trend over time for selected model."""
    from laser_trim_analyzer.database import get_database
    db = get_database()

    model = self._get_current_model()
    if not model:
        return

    spec = db.get_model_spec(model)
    if not spec or not spec.get("linearity_spec_pct"):
        self._show_chart_message("No linearity spec defined for this model")
        return

    trend = db.get_cpk_trend_for_model(
        model, spec["linearity_spec_pct"], days_back=180, period="month"
    )

    fig = self._get_chart_figure()
    ax = fig.add_subplot(111)

    periods = [t["period"] for t in trend]
    cpk_values = [t["cpk"] for t in trend]

    ax.plot(periods, cpk_values, marker='o', color='#0d6efd', linewidth=2, label="Cpk")

    # Reference lines
    ax.axhline(y=1.67, color='#198754', linestyle='--', alpha=0.7, label='Excellent (1.67)')
    ax.axhline(y=1.33, color='#fd7e14', linestyle='--', alpha=0.7, label='Capable (1.33)')
    ax.axhline(y=1.0, color='#dc3545', linestyle='--', alpha=0.7, label='Minimum (1.0)')

    ax.set_ylabel("Cpk")
    ax.set_title(f"Cpk Trend — {model}")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    fig.tight_layout()
    self._update_chart_canvas(fig)
```

- [ ] **Step 5: Implement yield trend chart**

```python
def _show_yield_trend(self):
    """Show overall yield trend across all models."""
    from laser_trim_analyzer.database import get_database
    db = get_database()

    data = db.get_yield_trend(days_back=180, period="week")

    fig = self._get_chart_figure()
    ax = fig.add_subplot(111)

    periods = [d["period"] for d in data]
    rates = [d["pass_rate"] for d in data]

    ax.fill_between(range(len(periods)), rates, alpha=0.3, color='#0d6efd')
    ax.plot(range(len(periods)), rates, marker='o', markersize=3, color='#0d6efd', linewidth=1.5)

    # Target line
    ax.axhline(y=80, color='#198754', linestyle='--', alpha=0.7, label='Target (80%)')

    ax.set_xticks(range(0, len(periods), max(1, len(periods) // 8)))
    ax.set_xticklabels(
        [periods[i] for i in range(0, len(periods), max(1, len(periods) // 8))],
        rotation=45, fontsize=8,
    )
    ax.set_ylabel("Yield (%)")
    ax.set_title("Overall Yield Trend")
    ax.set_ylim(0, 105)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    self._update_chart_canvas(fig)
```

- [ ] **Step 6: Implement drift timeline chart**

```python
def _show_drift_timeline(self):
    """Show drift detection events as a timeline."""
    from laser_trim_analyzer.database import get_database
    db = get_database()

    events = db.get_drift_events_timeline(days_back=180)

    fig = self._get_chart_figure()
    ax = fig.add_subplot(111)

    if not events:
        ax.text(0.5, 0.5, "No drift events detected",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
    else:
        models = list(set(e["model"] for e in events))
        model_y = {m: i for i, m in enumerate(models)}

        for e in events:
            y = model_y[e["model"]]
            x_label = e.get("date", "")[:10]  # Date portion
            ax.scatter(x_label, y, s=80, c='#dc3545', zorder=5, marker='D')

        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    ax.set_title("Drift Detection Timeline")
    fig.tight_layout()
    self._update_chart_canvas(fig)
```

- [ ] **Step 7: Wire up the trend type selector**

```python
def _on_trend_type_changed(self, value: str):
    """Handle trend type selector change."""
    if value == "Standard":
        self._show_standard_trends()  # Existing behavior
    elif value == "Comparative":
        self._show_comparative_trends()
    elif value == "Cpk Trend":
        self._show_cpk_trend()
    elif value == "Yield":
        self._show_yield_trend()
    elif value == "Drift":
        self._show_drift_timeline()
```

- [ ] **Step 8: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/trends.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
```

- [ ] **Step 9: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py src/laser_trim_analyzer/database/manager.py
git commit -m "feat: add comparative, Cpk, yield, and drift trend charts"
```

---

### Task 6: Failure Pattern Analysis

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/dashboard.py`
- Modify: `src/laser_trim_analyzer/gui/pages/scorecard.py`

- [ ] **Step 1: Add failure position heatmap to scorecard**

In the scorecard page, add a `_render_failure_pattern` section that shows a horizontal bar or heatmap of where failures occur along the element (which measurement position). This helps identify position-dependent issues (e.g., failures always at 70-80% of travel = end-of-element issue).

```python
def _render_failure_pattern(self, data: dict, row: int) -> int:
    """Render failure pattern analysis with position heatmap."""
    frame = ctk.CTkFrame(self.scroll_frame)
    frame.grid(row=row, column=0, sticky="ew", pady=5)

    ctk.CTkLabel(
        frame, text="Failure Patterns",
        font=ctk.CTkFont(size=14, weight="bold"),
    ).pack(padx=15, pady=(10, 5), anchor="w")

    try:
        from laser_trim_analyzer.database import get_database
        db = get_database()
        positions = db.get_failure_position_data(
            model=data["model"], days_back=90
        )

        if not positions:
            ctk.CTkLabel(
                frame, text="  No failure position data available",
                font=ctk.CTkFont(size=11), text_color="gray",
            ).pack(padx=15, pady=5, anchor="w")
            ctk.CTkLabel(frame, text="", height=5).pack()
            return row + 1

        # Create a small matplotlib chart for position distribution
        from laser_trim_analyzer.gui.widgets.chart import ChartWidget
        chart = ChartWidget(frame, width=500, height=150)
        chart.pack(padx=15, pady=5, fill="x")

        fig = chart.figure
        ax = fig.add_subplot(111)

        # Extract positions and bin them
        import numpy as np
        pos_values = [p["max_position"] for p in positions
                      if p.get("max_position") is not None]

        if pos_values:
            ax.hist(pos_values, bins=20, color='#dc3545', alpha=0.7,
                    edgecolor='white', linewidth=0.5)
            ax.set_xlabel("Position (%)", fontsize=9)
            ax.set_ylabel("Failure Count", fontsize=9)
            ax.set_title("Failure Position Distribution", fontsize=10)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            chart.draw()
        else:
            ax.text(0.5, 0.5, "No position data",
                    ha='center', va='center', transform=ax.transAxes)

    except Exception as e:
        ctk.CTkLabel(
            frame, text=f"  Pattern analysis unavailable: {e}",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).pack(padx=15, pady=5, anchor="w")

    ctk.CTkLabel(frame, text="", height=5).pack()
    return row + 1
```

Call `_render_failure_pattern` from `_render_scorecard` after the Cpk section.

- [ ] **Step 2: Add failure mode pie chart to dashboard process health**

Replace the text-based failure mode display with a small pie chart in the dashboard's process health section:

```python
def _render_failure_mode_chart(self, parent, modes: List[dict]):
    """Render a pie chart of failure modes in the dashboard."""
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget

    chart = ChartWidget(parent, width=250, height=200)
    chart.pack(padx=10, pady=5)

    fig = chart.figure
    ax = fig.add_subplot(111)

    labels = [m["mode"] for m in modes]
    counts = [m["count"] for m in modes]
    colors = ['#dc3545', '#fd7e14', '#6f42c1', '#6c757d']

    ax.pie(counts, labels=labels, colors=colors[:len(labels)],
           autopct='%1.0f%%', textprops={'fontsize': 8})
    ax.set_title("Failure Modes", fontsize=10)
    fig.tight_layout()
    chart.draw()
```

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/dashboard.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/scorecard.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/dashboard.py src/laser_trim_analyzer/gui/pages/scorecard.py
git commit -m "feat: add failure position heatmap and failure mode pie chart"
```

---

### Task 7: ML Enhancements — Spec-Aware Features and Recommendations

**Files:**
- Modify: `src/laser_trim_analyzer/ml/predictor.py`
- Modify: `src/laser_trim_analyzer/ml/manager.py`

- [ ] **Step 1: Add spec-aware features to the predictor**

In `predictor.py`, enhance the feature extraction to include model spec information when available. This gives the RandomForest model additional context about each unit's engineering constraints.

Read `predictor.py` to find the feature extraction method (likely `_extract_features` or similar). Add these additional features when model specs are available:

```python
# Spec-aware features (add to feature extraction)
def _add_spec_features(self, features: dict, model: str) -> dict:
    """
    Enrich feature vector with model spec data.

    Adds:
    - deviation_to_spec_ratio: how close the deviation is to the spec limit
    - is_tight_spec: 1 if spec < 0.3%, 0 otherwise
    - element_type_encoded: numeric encoding of element type
    """
    try:
        from laser_trim_analyzer.database import get_database
        db = get_database()
        spec = db.get_model_spec(model)

        if spec and spec.get("linearity_spec_pct"):
            max_dev = features.get("max_linearity_deviation", 0)
            spec_pct = spec["linearity_spec_pct"]

            features["deviation_to_spec_ratio"] = (
                abs(max_dev) / spec_pct if spec_pct > 0 else 0
            )
            features["is_tight_spec"] = 1 if spec_pct < 0.3 else 0
            features["spec_margin"] = spec_pct - abs(max_dev)

        if spec and spec.get("element_type"):
            # Simple encoding: hash to a few buckets
            etype = spec["element_type"].lower()
            element_map = {
                "conductive plastic": 1,
                "hybrid": 2,
                "wirewound": 3,
                "cermet": 4,
            }
            features["element_type_code"] = element_map.get(etype, 0)

    except Exception:
        pass  # Specs not available — features stay as-is

    return features
```

Integrate this into the existing feature extraction pipeline by calling `_add_spec_features` after the base features are built.

- [ ] **Step 2: Add adjustment recommendations to ML manager**

In `manager.py`, add a method that generates simple adjustment recommendations based on analysis results and model specs:

```python
def get_adjustment_recommendations(
    self, model: str, recent_results: List[dict]
) -> List[dict]:
    """
    Generate adjustment recommendations based on recent results and specs.

    Looks at failure patterns and suggests:
    - Spec review if Cpk < 1.0 but most failures are near-misses
    - Process adjustment if drift detected
    - Focus areas based on failure position patterns

    Returns list of {recommendation, priority, rationale} dicts.
    """
    recommendations = []

    if not recent_results:
        return recommendations

    # Calculate basic stats
    total = len(recent_results)
    failures = [r for r in recent_results if r.get("status") != "PASS"]
    fail_rate = len(failures) / total * 100 if total > 0 else 0

    # Check for near-misses (failures within 10% of spec)
    try:
        from laser_trim_analyzer.database import get_database
        db = get_database()
        spec = db.get_model_spec(model)

        if spec and spec.get("linearity_spec_pct"):
            spec_pct = spec["linearity_spec_pct"]
            near_misses = [
                r for r in failures
                if r.get("max_deviation") and
                abs(r["max_deviation"]) < spec_pct * 1.1
            ]

            if len(near_misses) > len(failures) * 0.5:
                recommendations.append({
                    "recommendation": "Review spec tightness",
                    "priority": "Medium",
                    "rationale": (
                        f"{len(near_misses)}/{len(failures)} failures are within 10% "
                        f"of the {spec_pct}% spec limit. These are near-misses that "
                        f"might pass with minor process adjustment."
                    ),
                })

        # Check drift
        ml_state = db.get_model_ml_state(model)
        if ml_state and ml_state.get("drift_detected"):
            recommendations.append({
                "recommendation": "Investigate process drift",
                "priority": "High",
                "rationale": (
                    f"Drift detection triggered for {model}. "
                    f"Review recent process changes, material lots, or equipment."
                ),
            })

    except Exception:
        pass

    # High fail rate warning
    if fail_rate > 40:
        recommendations.append({
            "recommendation": "Prioritize for root cause analysis",
            "priority": "High",
            "rationale": (
                f"Fail rate of {fail_rate:.0f}% exceeds 40% threshold. "
                f"This model is consuming significant rework resources."
            ),
        })

    return recommendations
```

- [ ] **Step 3: Add category-level drift detection query**

Add a method to detect drift at the element-type or product-class level (aggregate drift):

```python
def get_category_drift_summary(
    self, category: str = "element_type"
) -> List[Dict[str, Any]]:
    """
    Aggregate drift status by category (element_type or product_class).

    If multiple models in a category show drift, that suggests a
    systematic issue (material change, equipment problem, etc.).

    Returns list of {category, total_models, drifting_models, pct_drifting}.
    """
    from laser_trim_analyzer.database import get_database
    db = get_database()

    with db.session() as session:
        spec_col = ModelSpec.element_type if category == "element_type" else ModelSpec.product_class

        # Get all models with their category
        specs = session.query(
            ModelSpec.model, spec_col.label("category")
        ).filter(
            spec_col.isnot(None)
        ).all()

    # Check drift status for each
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"total": 0, "drifting": 0})

    for spec in specs:
        cat_stats[spec.category]["total"] += 1
        try:
            ml_state = db.get_model_ml_state(spec.model)
            if ml_state and ml_state.get("drift_detected"):
                cat_stats[spec.category]["drifting"] += 1
        except Exception:
            pass

    return [
        {
            "category": cat,
            "total_models": s["total"],
            "drifting_models": s["drifting"],
            "pct_drifting": (s["drifting"] / s["total"] * 100) if s["total"] > 0 else 0,
        }
        for cat, s in sorted(cat_stats.items())
    ]
```

- [ ] **Step 4: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/ml/predictor.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/ml/manager.py').read()); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/ml/predictor.py src/laser_trim_analyzer/ml/manager.py
git commit -m "feat: add spec-aware ML features, adjustment recommendations, category drift"
```

---

### Task 8: Executive Summary Export

**Files:**
- Modify: `src/laser_trim_analyzer/export/excel.py`

- [ ] **Step 1: Add executive summary sheet to Excel export**

Read the current `excel.py` to understand the export structure. Add a new method that creates an "Executive Summary" sheet at the beginning of the workbook with:

- Date range and data scope
- Overall pass rate with color coding
- Cpk summary table (models by capability rating)
- Top 5 worst models with pass rates
- Failure mode breakdown
- Drift alert summary
- Key recommendations

```python
def _create_executive_summary_sheet(self, wb, stats: dict, cpk_data: list, failure_modes: list):
    """
    Create an Executive Summary sheet in the Excel workbook.

    This is the first sheet — designed for management review.
    """
    ws = wb.create_sheet("Executive Summary", 0)

    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    # Styles
    title_font = Font(size=16, bold=True)
    header_font = Font(size=12, bold=True)
    metric_font = Font(size=14, bold=True)
    normal_font = Font(size=10)
    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin'),
    )

    row = 1

    # Title
    ws.cell(row=row, column=1, value="Laser Trim Quality — Executive Summary")
    ws.cell(row=row, column=1).font = title_font
    row += 1

    # Date
    from datetime import datetime
    ws.cell(row=row, column=1, value=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    ws.cell(row=row, column=1).font = normal_font
    row += 2

    # Overall metrics
    ws.cell(row=row, column=1, value="Key Metrics")
    ws.cell(row=row, column=1).font = header_font
    row += 1

    pass_rate = stats.get("overall_pass_rate", 0)
    total = stats.get("total_processed", 0)

    ws.cell(row=row, column=1, value="Overall Pass Rate:")
    ws.cell(row=row, column=2, value=f"{pass_rate:.1f}%")
    ws.cell(row=row, column=2).font = metric_font
    if pass_rate >= 80:
        ws.cell(row=row, column=2).fill = green_fill
    elif pass_rate >= 60:
        ws.cell(row=row, column=2).fill = yellow_fill
    else:
        ws.cell(row=row, column=2).fill = red_fill
    row += 1

    ws.cell(row=row, column=1, value="Total Units Processed:")
    ws.cell(row=row, column=2, value=total)
    row += 2

    # Cpk summary table
    if cpk_data:
        ws.cell(row=row, column=1, value="Process Capability Summary")
        ws.cell(row=row, column=1).font = header_font
        row += 1

        headers = ["Model", "Cpk", "Rating", "Samples", "Spec (%)"]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = Font(bold=True)
            cell.border = thin_border
        row += 1

        for cpk in cpk_data[:15]:  # Top 15 models
            ws.cell(row=row, column=1, value=cpk["model"]).border = thin_border
            cpk_cell = ws.cell(row=row, column=2, value=round(cpk["cpk"], 3) if cpk["cpk"] else "N/A")
            cpk_cell.border = thin_border
            rating_cell = ws.cell(row=row, column=3, value=cpk["rating"])
            rating_cell.border = thin_border
            if cpk["rating"] == "Incapable":
                rating_cell.fill = red_fill
            elif cpk["rating"] == "Marginal":
                rating_cell.fill = yellow_fill
            elif cpk["rating"] in ("Capable", "Excellent"):
                rating_cell.fill = green_fill
            ws.cell(row=row, column=4, value=cpk["n_samples"]).border = thin_border
            ws.cell(row=row, column=5, value=cpk.get("spec_pct", "")).border = thin_border
            row += 1
        row += 1

    # Failure modes
    if failure_modes:
        ws.cell(row=row, column=1, value="Failure Mode Breakdown")
        ws.cell(row=row, column=1).font = header_font
        row += 1
        for fm in failure_modes:
            ws.cell(row=row, column=1, value=fm["mode"])
            ws.cell(row=row, column=2, value=fm["count"])
            row += 1
        row += 1

    # Column widths
    ws.column_dimensions['A'].width = 30
    ws.column_dimensions['B'].width = 15
    ws.column_dimensions['C'].width = 15
    ws.column_dimensions['D'].width = 12
    ws.column_dimensions['E'].width = 12

    return ws
```

- [ ] **Step 2: Add model scorecard export sheet**

Add a method to export individual model scorecards as a sheet:

```python
def _create_scorecard_sheet(self, wb, scorecard_data: dict):
    """Create a scorecard sheet for a single model."""
    model = scorecard_data["model"]
    ws = wb.create_sheet(f"Scorecard — {model[:20]}")

    from openpyxl.styles import Font

    row = 1
    ws.cell(row=row, column=1, value=f"Model Scorecard: {model}")
    ws.cell(row=row, column=1).font = Font(size=14, bold=True)
    row += 2

    # Key metrics
    metrics = [
        ("Pass Rate", f"{scorecard_data.get('pass_rate', 0):.1f}%"),
        ("Total Units (90d)", str(scorecard_data.get("total", 0))),
        ("Avg Deviation", f"{scorecard_data.get('avg_deviation', 0):.4f}%" if scorecard_data.get('avg_deviation') else "N/A"),
    ]

    cpk = scorecard_data.get("cpk", {})
    if cpk and cpk.get("cpk") is not None:
        metrics.append(("Cpk", f"{cpk['cpk']:.3f}"))
        metrics.append(("Cpk Rating", cpk.get("rating", "Unknown")))
        metrics.append(("Ppk", f"{cpk['ppk']:.3f}" if cpk.get('ppk') else "N/A"))

    for label, value in metrics:
        ws.cell(row=row, column=1, value=label)
        ws.cell(row=row, column=2, value=value)
        row += 1

    row += 1

    # Spec info
    spec = scorecard_data.get("spec", {})
    if spec:
        ws.cell(row=row, column=1, value="Specifications")
        ws.cell(row=row, column=1).font = Font(bold=True)
        row += 1
        spec_fields = [
            ("Element Type", spec.get("element_type")),
            ("Product Class", spec.get("product_class")),
            ("Linearity Type", spec.get("linearity_type")),
            ("Linearity Spec", f"±{spec['linearity_spec_pct']}%" if spec.get("linearity_spec_pct") else None),
        ]
        for label, value in spec_fields:
            if value:
                ws.cell(row=row, column=1, value=label)
                ws.cell(row=row, column=2, value=value)
                row += 1

    ws.column_dimensions['A'].width = 20
    ws.column_dimensions['B'].width = 25
```

- [ ] **Step 3: Integrate executive summary into existing export flow**

Find the main export method in `excel.py` (likely `export_to_excel` or similar). After creating the existing sheets, add calls to create the executive summary and optional scorecard sheets:

```python
# In the main export method, after existing sheet creation:
try:
    from laser_trim_analyzer.database import get_database
    db = get_database()
    stats = db.get_dashboard_stats(days_back=30)
    cpk_data = db.get_cpk_by_model(days_back=30)
    failure_modes = db.get_failure_mode_summary(days_back=30)
    self._create_executive_summary_sheet(wb, stats, cpk_data, failure_modes)
except Exception as e:
    logger.warning(f"Could not create executive summary: {e}")
```

- [ ] **Step 4: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/export/excel.py').read()); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/export/excel.py
git commit -m "feat: add executive summary and model scorecard sheets to Excel export"
```

---

### Task 9: Integration Testing and Version Bump

**Files:**
- Modify: `src/laser_trim_analyzer/utils/constants.py` (or wherever APP_VERSION is defined)

- [ ] **Step 1: Full syntax check on all modified and created files**

```bash
python3 -c "
import ast, os
files = [
    'src/laser_trim_analyzer/core/cpk.py',
    'src/laser_trim_analyzer/database/manager.py',
    'src/laser_trim_analyzer/gui/pages/dashboard.py',
    'src/laser_trim_analyzer/gui/pages/scorecard.py',
    'src/laser_trim_analyzer/gui/pages/trends.py',
    'src/laser_trim_analyzer/gui/pages/analyze.py',
    'src/laser_trim_analyzer/ml/predictor.py',
    'src/laser_trim_analyzer/ml/manager.py',
    'src/laser_trim_analyzer/export/excel.py',
    'src/laser_trim_analyzer/app.py',
]
for f in files:
    if os.path.exists(f):
        ast.parse(open(f).read())
        print(f'OK: {f}')
    else:
        print(f'MISSING: {f}')
print('All files checked')
"
```

- [ ] **Step 2: Smoke test Cpk module standalone**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.core.cpk import calculate_cpk, rate_cpk

# Test with known data
import random; random.seed(42)
# Good process: tight distribution well within spec
good = [random.gauss(0, 0.05) for _ in range(200)]
r = calculate_cpk(good, spec_limit_pct=0.5)
print(f'Good process: Cpk={r.cpk:.2f}, rating={r.rating}')
assert r.cpk > 1.5, f'Expected Cpk > 1.5, got {r.cpk}'

# Bad process: wide distribution near spec
bad = [random.gauss(0.2, 0.15) for _ in range(200)]
r2 = calculate_cpk(bad, spec_limit_pct=0.5)
print(f'Bad process: Cpk={r2.cpk:.2f}, rating={r2.rating}')
assert r2.cpk < 1.0, f'Expected Cpk < 1.0, got {r2.cpk}'

print('Cpk tests passed')
"
```

- [ ] **Step 3: Test Cpk rating function**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.core.cpk import rate_cpk
assert rate_cpk(2.0) == 'Excellent'
assert rate_cpk(1.5) == 'Capable'
assert rate_cpk(1.1) == 'Marginal'
assert rate_cpk(0.5) == 'Incapable'
assert rate_cpk(None) == 'Unknown'
print('Rating tests passed')
"
```

- [ ] **Step 4: Bump version**

Update the version to reflect Phase 4 completion. Find the APP_VERSION constant and update it:

```python
APP_VERSION: Final[str] = "5.4.0"
```

This follows semantic versioning: 5.x.0 where x tracks the phase number.

- [ ] **Step 5: Verify version file syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/utils/constants.py').read()); print('OK')"
```

- [ ] **Step 6: Commit and tag**

```bash
git add -A
git commit -m "feat: Phase 4 complete — Advanced Analytics & Visualization v5.4.0

- Cpk/Ppk process capability analysis module
- Dashboard overhaul with attention cards, health sections
- Model Scorecard page with comprehensive model summary
- Enhanced Trends: comparative, Cpk trend, yield, drift timeline
- Failure pattern analysis with position heatmap
- Spec-aware ML features and adjustment recommendations
- Executive summary Excel export with Cpk and scorecard sheets"
git push origin v5-upgrade
```

---

## Summary of Changes

| Task | Files | Description |
|------|-------|-------------|
| 1 | Create `core/cpk.py` | Cpk/Ppk calculation with rating and trend support |
| 2 | Modify `database/manager.py` | Cpk queries, failure patterns, scorecard data, yield trend, comparative trends, drift timeline |
| 3 | Modify `gui/pages/dashboard.py` | Attention cards, overall health, process health sections |
| 4 | Create `gui/pages/scorecard.py`, modify `app.py`, `analyze.py` | Model scorecard page accessible from Analyze |
| 5 | Modify `gui/pages/trends.py`, `database/manager.py` | Comparative, Cpk, yield, drift trend chart types |
| 6 | Modify `gui/pages/dashboard.py`, `gui/pages/scorecard.py` | Failure position heatmap, failure mode pie chart |
| 7 | Modify `ml/predictor.py`, `ml/manager.py` | Spec-aware features, adjustment recommendations, category drift |
| 8 | Modify `export/excel.py` | Executive summary sheet, scorecard export sheet |
| 9 | Modify `utils/constants.py` | Version bump, full syntax check, integration smoke tests |

## Task Dependencies

```
Task 1 (Cpk module) ──┬──> Task 2 (DB queries) ──┬──> Task 3 (Dashboard)
                       │                          ├──> Task 4 (Scorecard)
                       │                          ├──> Task 5 (Trends)
                       │                          └──> Task 6 (Failure patterns)
                       │
                       └──> Task 7 (ML enhancements) ─ independent of Tasks 3-6
                       └──> Task 8 (Export) ─ depends on Task 2
                       
Task 9 (Integration) ──── depends on all above
```

Tasks 3, 4, 5, 6 can be parallelized after Tasks 1 and 2 are complete.
Task 7 can be parallelized with Tasks 3-6.
Task 8 depends on Task 2 but can be parallelized with Tasks 3-7.
