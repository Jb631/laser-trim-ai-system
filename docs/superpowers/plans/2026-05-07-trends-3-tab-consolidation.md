# Trends 3-Tab Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the Trends page from 7 segmented-button tabs to 3 (Standard / Drift / Trim Difficulty), rebuild the Standard summary view around 5 widget rows of priorities content, and replace the matplotlib-text "Focus This Week" with native CTk widgets so it stops rendering as a clipped text blob.

**Architecture:** Pure GUI restructure. No DB schema or method changes. Existing DB methods (`get_linearity_prioritization`, `get_near_miss_summary`, `get_anomaly_rate_by_model`, `get_process_drift_by_model`) feed the same data into a different layout. Drift gets an internal `CTkSegmentedButton` to flip between ML drift and Process drift views.

**Tech Stack:** CustomTkinter for GUI widgets, matplotlib for charts via `ChartWidget` wrapper. No new libraries.

**Source spec:** `docs/superpowers/specs/2026-05-07-trends-3-tab-consolidation-design.md` (commit `7742e03`).

**Branch:** Work directly on `main` (the prior fix branch is already merged). Each task commits independently so any single task can be reverted in isolation.

---

## File Structure

Single file: `src/laser_trim_analyzer/gui/pages/trends.py` (~3700 lines).

The change is destructive (~600 lines deleted) and additive (~250 lines added). Net file size shrinks.

Methods deleted:

- `_show_priorities`, `_render_priorities`
- `_show_comparative_trends`, `_render_comparative_trends`
- `_show_cpk_trend`, `_render_cpk_trend`
- `_update_impact_display` (no-op stub from earlier)

Methods added:

- `_create_focus_section` — builds the 5-row Focus This Week widget
- `_update_focus_section` — refreshes Focus This Week with new priority data
- `_create_failure_severity_chart` / `_update_failure_severity_chart` — chart inside Standard summary
- `_create_cost_impact_chart` / `_update_cost_impact_chart` — chart inside Standard summary
- `_on_drift_subtab_changed` — Drift tab toggle handler

Methods modified:

- `_create_summary_view` — gutted and rebuilt around the 5-section spec
- `_update_summary_display` — gutted and rebuilt to match
- `_render_drift_timeline` and `_render_process_drift` — guard updated to also check `_drift_subtab`
- `_on_trend_type_changed` — branches reduced to 3
- `__init__` — adds `_drift_subtab` state

---

## Task 1: Replace Standard summary view with the 5-section layout

**Goal:** `_create_summary_view` builds: Stats row → Focus This Week (native widgets) → Failure Severity chart → Cost Impact chart → ML Status. `_update_summary_display` populates them. Everything else previously on summary is gone.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py` — `_create_summary_view`, `_update_summary_display`, plus new helper methods
- No tests (UI-only restructure; existing test suite stays green)

This task is large because the existing `_create_summary_view` and `_update_summary_display` are tightly coupled to the widgets they build. Splitting it into smaller commits would leave the app non-functional between commits.

- [ ] **Step 1: Read the current `_create_summary_view` end-to-end**

Run: `grep -n "def _create_summary_view\|def _update_summary_display" src/laser_trim_analyzer/gui/pages/trends.py`

Note the line ranges. Read both methods in full. The current implementation builds (in this order) the stats row, alerts chart frame, impact (Where-to-Focus) frame, models frame (best/recent), row3 frame (trending/low_data), drift section, heatmap section, ML section. All but the stats row and ML section get deleted.

- [ ] **Step 2: Replace `_create_summary_view` body**

Find the method (anchor on the docstring `"""Create the summary view (All Models mode)."""`). Replace the body with this layout. Show the existing stats-row code unchanged at the top, then the 4 new sections, then the ML section unchanged at the bottom.

```python
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

        self._summary_charts_initialized = False  # charts re-init lazily on update
```

Use Edit with `replace_all=False` and a unique anchor — copy the full current body of `_create_summary_view` as `old_string` and the new body above as `new_string`.

- [ ] **Step 3: Add `_create_focus_section` helper**

Insert this method right after `_create_summary_view`. Builds an empty 5-row container; rows are populated in `_update_focus_section` once data arrives.

```python
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
```

- [ ] **Step 4: Add `_update_focus_section` helper**

Renders one row per priority model. Fail-rate color thresholds: ≥30 red, ≥15 orange, ≥5 amber, else default white.

```python
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
```

- [ ] **Step 5: Add `_create_failure_severity_chart` helper**

```python
    def _create_failure_severity_chart(self):
        """Bar chart of failing tracks bucketed by fail-point count."""
        ChartWidget, ChartStyle = _ensure_chart_module()
        sev_frame = ctk.CTkFrame(self.content)
        sev_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        ctk.CTkLabel(
            sev_frame, text="Failure Severity (last %dd)" % self.selected_days,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 5))
        self._severity_chart_frame = sev_frame
        self.severity_chart = ChartWidget(
            sev_frame, style=ChartStyle(figure_size=(10, 2.6), dpi=100)
        )
        self.severity_chart.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.severity_chart.show_placeholder("Loading failure severity…")
        self._chart_widgets.append(self.severity_chart)
```

- [ ] **Step 6: Add `_update_failure_severity_chart` helper**

Reuses the existing matplotlib bucket-bar pattern from the old `_render_priorities` Section 2 — but now the chart is drawn into the dedicated `severity_chart` widget instead of an axis on a multi-section figure.

```python
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
```

- [ ] **Step 7: Add `_create_cost_impact_chart` and `_update_cost_impact_chart` helpers**

Same pattern. The Cost Impact section consumes the existing priorities + pricing flow.

```python
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
```

- [ ] **Step 8: Replace `_update_summary_display` body**

Find the existing method. The current body updates: stats labels, alerts chart, models frame (best/worst), trending, low-data, drift section, heatmap, ML text. Strip everything except: stats labels, ML text. Add: Focus This Week update, Failure Severity update, Cost Impact update, plus a fetch of `near_miss` and pricing data.

Show the new body in full:

```python
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
            total_samples = sum(m.get("total_samples", 0) for m in active_models)
            avg_pass_rate = (
                sum(m.get("pass_rate", 0) for m in active_models) / total_models
            ) if total_models else 0
            avg_sigma_rate = (
                sum(m.get("sigma_pass_rate", 0) for m in active_models) / total_models
            ) if total_models else 0
            avg_linearity_rate = (
                sum(m.get("linearity_pass_rate", 0) for m in active_models)
                / total_models
            ) if total_models else 0
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
```

- [ ] **Step 9: Augment `_load_summary_data` to also fetch near_miss and pricing**

These two pieces of data were already fetched by the now-deleted `_show_priorities` flow; pull them up into the summary load.

Find this in `_load_summary_data` (search for the anomaly-rows fetch added earlier) and add these lines immediately before the `self.after(0, ...)` call:

```python
        # Failure Severity + Cost Impact need near_miss summary and pricing
        try:
            near_miss = db.get_near_miss_summary(days_back=self.selected_days)
        except Exception as e:
            logger.debug(f"Could not load near-miss summary: {e}")
            near_miss = {}
        cfg = get_config()
        pricing = dict(cfg.active_models.model_prices or {})
        cost_ratio = float(getattr(cfg.active_models, "cost_ratio", 0.5))
```

Then update the `self.after(0, lambda g=gen: self._update_summary_display_if_current(...))` call to pass `near_miss=near_miss, pricing=pricing, cost_ratio=cost_ratio` as kwargs.

- [ ] **Step 10: Syntax check**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m py_compile src/laser_trim_analyzer/gui/pages/trends.py && echo OK`

Expected: `OK`.

- [ ] **Step 11: Smoke import**

Run:

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5 && \
python3 -c "import sys; sys.path.insert(0,'src'); \
from laser_trim_analyzer.gui.pages.trends import TrendsPage; \
print('import OK')"
```

Expected: `import OK`.

If you see `AttributeError: 'TrendsPage' object has no attribute '_show_priorities'` (or similar) — that's expected because Task 3 hasn't run yet to remove the segmented-button entries. The error would only fire when the user clicks a non-existent tab. Move on.

- [ ] **Step 12: Run existing tests**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest 2>&1 | tail -3`

Expected: `11 passed`.

- [ ] **Step 13: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "refactor(trends): rebuild Standard summary as 5-section layout

Standard summary view now contains exactly: Active Models Summary
stats, Focus This Week native widget rows, Failure Severity bar
chart, Cost Impact horizontal bars, ML Status one-liner. Heatmap,
alerts chart, best/worst/recent/trending sections, drift summary,
and Where-to-Focus pointer button all removed.

Focus This Week is rebuilt as native CTkLabel rows (the previous
matplotlib ax.text approach clipped at GUI sizes — root cause of
the 'priorities tab is all messed up' report). Each row carries
rank, model, fail rate (color-coded), counts, and recommendation
on a uniform two-line layout regardless of content length.

Failure Severity and Cost Impact lifted from the now-defunct
Priorities tab into dedicated ChartWidget instances on the
summary view; the Priorities tab itself is removed in a later
commit (segmented-button shrink). _load_summary_data now also
fetches get_near_miss_summary and pricing/cost_ratio config so
the new charts have data on first render.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add Drift sub-toggle and update render guards

**Goal:** Drift tab becomes a single tab with a `[ ML Drift | Process Drift ]` segmented button; toggling re-renders the chart area below.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py` — `__init__`, `_render_drift_timeline`, `_render_process_drift`, plus new toggle widget creation

- [ ] **Step 1: Add `_drift_subtab` state to `__init__`**

Find the `__init__` method (around the `_load_generation = 0` line added earlier). Add immediately after:

```python
        # Drift tab sub-view selector. Set when the user clicks the toggle
        # inside the Drift tab; consumed by render guards in
        # _render_drift_timeline and _render_process_drift to discard a
        # stale render that fires after the user has flipped to the other
        # sub-view.
        self._drift_subtab: str = "ML Drift"
```

- [ ] **Step 2: Add `_create_drift_view` helper**

Add a new method immediately after `_create_dedicated_chart_view`. Both `_show_drift_timeline` and `_show_process_drift` will call this helper instead of the generic one — the Drift tab needs the toggle widget on top, which doesn't fit the generic single-chart layout.

```python
    def _create_drift_view(self) -> "ChartWidget":
        """Build the Drift tab: ML/Process toggle on top, chart below.

        Returns the ChartWidget to render into. The toggle's selection
        decides which `_show_*` method should run when the user clicks;
        we still call `_show_drift_timeline()` / `_show_process_drift()`
        from the toggle handler so the render guards (which read
        self._drift_subtab) work uniformly.
        """
        for widget in self.content.winfo_children():
            widget.destroy()
        self._cleanup_charts()

        self.content.grid_rowconfigure(0, weight=0)
        self.content.grid_rowconfigure(1, weight=1)

        toggle_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        toggle_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        ctk.CTkLabel(
            toggle_frame, text="View:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 8))
        self._drift_subtab_button = ctk.CTkSegmentedButton(
            toggle_frame,
            values=["ML Drift", "Process Drift"],
            command=self._on_drift_subtab_changed,
        )
        self._drift_subtab_button.set(self._drift_subtab)
        self._drift_subtab_button.pack(side="left")

        chart_frame = ctk.CTkFrame(self.content)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))

        ChartWidget, ChartStyle = _ensure_chart_module()
        chart = ChartWidget(
            chart_frame,
            style=ChartStyle(figure_size=(12, 6), dpi=100)
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
```

- [ ] **Step 3: Update `_show_drift_timeline` and `_show_process_drift` to use the new view**

Both methods currently call `_create_dedicated_chart_view(title)` to make a single chart. Replace those calls with `_create_drift_view()`. The chart returned has the same shape, so the rest of each method works unchanged.

Find inside `_show_drift_timeline`:
```python
            chart = self._create_dedicated_chart_view("Drift Detection Timeline")
```
Replace with:
```python
            chart = self._create_drift_view()
```

Find inside `_render_process_drift`:
```python
            chart = self._create_dedicated_chart_view("Process Drift")
```
Replace with:
```python
            chart = self._create_drift_view()
```

- [ ] **Step 4: Tighten the render guards on both Drift renderers**

Currently both check `self._trend_type.get() != "Drift" / "Process Drift"`. Replace with a check that includes the sub-toggle.

Find in `_render_drift_timeline`:
```python
        if self._trend_type.get() != "Drift":
            return  # User switched tabs; don't destroy the new tab's frames
```
Replace with:
```python
        if self._trend_type.get() != "Drift" or self._drift_subtab != "ML Drift":
            return  # User switched tab or sub-view; bail before destroying frames
```

Find in `_render_process_drift`:
```python
        if self._trend_type.get() != "Process Drift":
            return  # User switched tabs; don't destroy the new tab's frames
```
Replace with:
```python
        if self._trend_type.get() != "Drift" or self._drift_subtab != "Process Drift":
            return  # User switched tab or sub-view; bail before destroying frames
```

- [ ] **Step 5: Syntax check**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m py_compile src/laser_trim_analyzer/gui/pages/trends.py && echo OK`

Expected: `OK`.

- [ ] **Step 6: Run existing tests**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest 2>&1 | tail -3`

Expected: `11 passed`.

- [ ] **Step 7: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "refactor(trends): Drift tab gets ML/Process sub-toggle

Both ML drift and process drift now live on a single Drift tab with
a CTkSegmentedButton on top to flip between views. The two render
methods are unchanged in body — they reuse a new _create_drift_view
helper that builds the toggle plus a ChartWidget into self.content.
Render guards extended to also check self._drift_subtab so a stale
render queued before a sub-toggle flip discards itself instead of
overwriting the new view's chart frame.

The top-level 'Process Drift' segmented-button entry is removed in
the next commit; until then it still works (calls _show_process_drift
which now goes through the new helper).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Shrink segmented button to 3 entries and simplify dispatcher

**Goal:** Top-level segmented button shows exactly `Standard`, `Drift`, `Trim Difficulty`. `_on_trend_type_changed` only branches into those 3.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py` — segmented-button construction + `_on_trend_type_changed`

- [ ] **Step 1: Replace the segmented-button values**

Find the segmented-button construction (near `_trend_type = ctk.CTkSegmentedButton`). The current `values=` list has 7 entries. Replace.

Find:
```python
            values=["Priorities", "Standard", "Comparative", "Cpk Trend", "Drift", "Process Drift", "Trim Difficulty"],
```
Replace with:
```python
            values=["Standard", "Drift", "Trim Difficulty"],
```

Also find the line after, where the default is set:
```python
        self._trend_type.set("Standard")
```
Confirm it stays `"Standard"`. (If it was `"Priorities"`, change it.)

- [ ] **Step 2: Simplify `_on_trend_type_changed`**

Find the current dispatcher and replace the body with a 3-way branch. Keep the `_load_generation` bump for non-Standard entries since the gen-guard pattern is still in use for Drift's render-guard.

```python
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
            # Reset sub-toggle to ML Drift each time the Drift tab opens.
            self._drift_subtab = "ML Drift"
            self._show_drift_timeline()
        elif value == "Trim Difficulty":
            self._show_trim_difficulty()
        # Any legacy stored selection ("Priorities", "Comparative", etc.)
        # is silently ignored — the segmented button no longer surfaces
        # those values, so the only way to reach this branch is a stale
        # config restoration on app launch. Treat as no-op.
```

- [ ] **Step 3: Syntax check**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m py_compile src/laser_trim_analyzer/gui/pages/trends.py && echo OK`

Expected: `OK`.

- [ ] **Step 4: Smoke import**

Run:

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5 && \
python3 -c "import sys; sys.path.insert(0,'src'); \
from laser_trim_analyzer.gui.pages.trends import TrendsPage; \
print('import OK')"
```

Expected: `import OK`.

- [ ] **Step 5: Run existing tests**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest 2>&1 | tail -3`

Expected: `11 passed`.

- [ ] **Step 6: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "refactor(trends): segmented button now 3 entries (Standard/Drift/Trim Difficulty)

User feedback: 'I'm not sure we need all those tabs.' Trim from 7 to
3 by absorbing Priorities content into Standard summary, merging ML
and Process drift into one Drift tab with a sub-toggle, and
retiring Comparative and Cpk live views (DB methods stay because
the executive Excel export still uses them).

_on_trend_type_changed now branches into 3. Legacy selections
restored from config silently no-op.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Delete the now-unused show/render methods

**Goal:** Remove dead code. After this commit, only the methods reachable from the 3 segmented-button branches remain.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py`

- [ ] **Step 1: Confirm no callers**

Run for each method:

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
grep -n "_show_priorities\|_render_priorities\|_show_comparative_trends\|_render_comparative_trends\|_show_cpk_trend\|_render_cpk_trend\|_update_impact_display" src/laser_trim_analyzer/gui/pages/trends.py
```

Expected: only the `def` lines for each method should appear (no `self.foo()` calls). If anything else shows, stop and inspect — it means an earlier task missed a removal.

- [ ] **Step 2: Delete the methods**

Use Edit to delete each method block. Anchor on the `def methodname(` line and include the full body up to (but not including) the next `def `. For each of the seven methods listed in Step 1, the Edit replaces the full method body with an empty string (or removes the matching `old_string` entirely by setting `new_string` to a single blank line).

Concretely, six methods to delete (`_show_priorities`, `_render_priorities`, `_show_comparative_trends`, `_render_comparative_trends`, `_show_cpk_trend`, `_render_cpk_trend`) plus the no-op stub `_update_impact_display`.

Read the file once to find each method's exact start/end line range. Then for each method run an Edit. The pattern: `old_string` = the entire method including its docstring and body, terminated by the next blank line before the next `def `. `new_string` = empty string.

- [ ] **Step 3: Syntax check**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m py_compile src/laser_trim_analyzer/gui/pages/trends.py && echo OK`

Expected: `OK`.

- [ ] **Step 4: Confirm method count dropped**

Run:

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
grep -c "^    def " src/laser_trim_analyzer/gui/pages/trends.py
```

Expected: dropped by 7 from the previous count (run the same command before Task 4 to compare).

- [ ] **Step 5: Smoke-import all GUI pages**

Run:

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5 && \
python3 -c "import sys; sys.path.insert(0,'src'); \
from laser_trim_analyzer.gui.pages.dashboard import DashboardPage; \
from laser_trim_analyzer.gui.pages.trends import TrendsPage; \
from laser_trim_analyzer.gui.pages.analyze import AnalyzePage; \
from laser_trim_analyzer.gui.pages.compare import ComparePage; \
from laser_trim_analyzer.gui.pages.process import ProcessPage; \
from laser_trim_analyzer.gui.pages.settings import SettingsPage; \
print('all pages import OK')"
```

Expected: `all pages import OK`.

- [ ] **Step 6: Run existing tests**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest 2>&1 | tail -3`

Expected: `11 passed`.

- [ ] **Step 7: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "refactor(trends): delete dead show/render methods after consolidation

After the 3-tab consolidation, six show/render methods plus the
no-op _update_impact_display stub are unreachable. Remove them so
the file shrinks back to the methods that match the actual UI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Push branch**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git push
```

- [ ] **Verify final state**

Run: `git log --oneline -6`

Expected: 4 new commits on `main` from the 4 tasks above plus the earlier review-followup commit and the merge commit.

The user pulls on the work computer next morning, runs the app, opens Trends, sees 3 tabs (Standard / Drift / Trim Difficulty). Standard summary shows the new 5-section layout with native Focus This Week rows that don't clip. Drift has a sub-toggle. Trim Difficulty unchanged.

---

## Notes

- This plan is GUI-only. The DB and ML layers are untouched. If a layout decision proves wrong on the user's screen, only the GUI file needs to change.
- Tests are smoke-level — the existing 11 tests cover the DB methods that aren't being modified. Manual verification (open the app and click each tab) is the real check.
- Render-guard pattern: a `_render_*` callback first checks `self._trend_type.get()`. The Drift renderers also check `self._drift_subtab`. This protects against the cascade where the user switches tabs/views faster than a background fetch completes.
