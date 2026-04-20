"""
Quality Health Page — Operational quality status at a glance.

Answers three questions for every model:
1. Is quality getting better or worse?
2. How bad/good is it right now?
3. What should I do about it?

Designed for non-technical users who need a quick read on quality.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.utils.threads import get_thread_manager

logger = logging.getLogger(__name__)

# ── colour constants ──────────────────────────────────────────────
COLOR_GREEN = "#27ae60"
COLOR_LIGHT_GREEN = "#2ecc71"
COLOR_YELLOW = "#f1c40f"
COLOR_ORANGE = "#f39c12"
COLOR_RED = "#e74c3c"
COLOR_GRAY = "#7f8c8d"
COLOR_WHITE = "#ecf0f1"
COLOR_DARK_BG = "#2b2b2b"
COLOR_CARD_BG = "#333333"

# Status thresholds (linearity pass rate)
THRESHOLD_GOOD = 80.0
THRESHOLD_OK = 65.0
THRESHOLD_WARNING = 50.0
# Below 50% → critical

# Trend thresholds (change in pass rate percentage points, recent vs older half)
TREND_IMPROVING = 3.0   # ≥ +3 pp
TREND_DECLINING = -3.0  # ≤ -3 pp
# Between → stable


class QualityHealthPage(ctk.CTkFrame):
    """Operational quality health dashboard."""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self._model_data: List[Dict[str, Any]] = []
        self._trending_data: List[Dict[str, Any]] = []
        self._active_only = True
        self._last_refresh_time = 0
        self._create_ui()

    # ── UI construction ───────────────────────────────────────────

    def _create_ui(self):
        """Build the page layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header row
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header_frame.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header_frame,
            text="Quality Health",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        # Active-only toggle
        self._active_var = ctk.BooleanVar(value=True)
        active_check = ctk.CTkCheckBox(
            header_frame,
            text="Active Models Only",
            variable=self._active_var,
            command=self._on_filter_change,
            font=ctk.CTkFont(size=12),
        )
        active_check.grid(row=0, column=1, sticky="e", padx=(0, 15))

        refresh_btn = ctk.CTkButton(
            header_frame,
            text="Refresh",
            width=90,
            height=32,
            command=self.refresh,
        )
        refresh_btn.grid(row=0, column=2, sticky="e")

        self._update_label = ctk.CTkLabel(
            header_frame, text="", text_color="gray", font=ctk.CTkFont(size=10)
        )
        self._update_label.grid(row=0, column=3, sticky="e", padx=(10, 0))

        # Scrollable content area
        self._content = ctk.CTkScrollableFrame(self)
        self._content.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self._content.grid_columnconfigure(0, weight=1)

        # Placeholder
        self._placeholder = ctk.CTkLabel(
            self._content,
            text="Loading quality data…",
            text_color="gray",
            font=ctk.CTkFont(size=14),
        )
        self._placeholder.pack(pady=40)

    # ── data loading ──────────────────────────────────────────────

    def refresh(self):
        """Kick off a background data load."""
        import time
        self._last_refresh_time = time.time()
        self._update_label.configure(text="Refreshing…")
        get_thread_manager().start_thread(
            target=self._load_data, name="quality-health-load"
        )

    def on_show(self):
        """Called when the page becomes visible."""
        import time
        # Debounce: skip refresh if last one was < 5 seconds ago
        if time.time() - self._last_refresh_time < 5:
            return
        self.refresh()

    def _load_data(self):
        """Background: pull model stats + trends from the database."""
        try:
            db = get_database()

            # Core data: per-model quality stats with recommendations
            priority = db.get_linearity_prioritization(days_back=365, min_samples=10)

            # Active models (for filter)
            try:
                active_models = db.get_active_models_summary(days_back=365, min_samples=5)
                active_names = {m["model"] for m in active_models}
            except Exception as e:
                logger.warning(f"Could not load active models: {e}")
                active_names = set()

            # Compute linearity trend per model (recent half vs older half)
            trend_map = self._compute_linearity_trends(db, [m["model"] for m in priority])

            # Compute per-model Final Test linearity pass rates
            ft_map = self._compute_ft_pass_rates(db, days_back=365)

            # Enrich priority data with trend info + active status + FT data
            for m in priority:
                name = m["model"]
                m["is_active"] = name in active_names
                trend_info = trend_map.get(name)
                if trend_info:
                    m["recent_pass_rate"] = trend_info["recent_rate"]
                    m["older_pass_rate"] = trend_info["older_rate"]
                    m["decline"] = trend_info["decline"]  # positive = declining
                    if trend_info["decline"] >= abs(TREND_DECLINING):
                        m["trend_direction"] = "declining"
                    elif trend_info["decline"] <= -TREND_IMPROVING:
                        m["trend_direction"] = "improving"
                    else:
                        m["trend_direction"] = "stable"
                else:
                    m["recent_pass_rate"] = None
                    m["older_pass_rate"] = None
                    m["decline"] = 0
                    m["trend_direction"] = "stable"

                # Final Test data — try exact model match first, then base model
                ft_info = ft_map.get(name)
                if not ft_info:
                    # Try normalized model (e.g. trim "8275A" → FT "8275")
                    base = re.sub(r'^(\d+)[A-Za-z]$', r'\1', name)
                    base = re.sub(r'^(\d+(?:-\d+)*)-[A-Za-z]$', r'\1', base)
                    if base != name:
                        ft_info = ft_map.get(base)
                if ft_info:
                    m["ft_pass_rate"] = ft_info["ft_pass_rate"]
                    m["ft_samples"] = ft_info["ft_samples"]
                    # Gap = trim pass rate - FT pass rate
                    # Positive gap = units passing trim but failing FT (escapes)
                    m["trim_ft_gap"] = round(m["linearity_pass_rate"] - ft_info["ft_pass_rate"], 1)
                else:
                    m["ft_pass_rate"] = None
                    m["ft_samples"] = 0
                    m["trim_ft_gap"] = None

            self.after(0, lambda: self._update_display(priority))

        except Exception as e:
            logger.error(f"Quality Health load error: {e}", exc_info=True)
            self.after(0, lambda err=str(e): self._show_error(err))

    def _compute_linearity_trends(self, db, models: List[str]) -> Dict[str, Dict]:
        """
        Compute linearity pass rate change: second half vs first half of data.

        Uses each model's own data range (split at midpoint) so trend
        detection works even when data is sparse or not very recent.

        Returns dict of model -> {recent_rate, older_rate, decline}.
        decline > 0 means quality got worse; decline < 0 means improved.
        """
        from datetime import timedelta
        from sqlalchemy import func, case, literal_column
        from laser_trim_analyzer.database.models import (
            AnalysisResult as DBAnalysisResult,
            TrackResult as DBTrackResult,
        )

        result = {}

        try:
            with db.session() as session:
                # Single query: get date range per model
                date_ranges = (
                    session.query(
                        DBAnalysisResult.model,
                        func.min(DBAnalysisResult.file_date).label("min_date"),
                        func.max(DBAnalysisResult.file_date).label("max_date"),
                    )
                    .filter(
                        DBAnalysisResult.model.in_(models),
                        DBAnalysisResult.file_date.isnot(None),
                    )
                    .group_by(DBAnalysisResult.model)
                    .all()
                )

                # Build midpoints, filter out models with <14 day span
                model_midpoints = {}
                for row in date_ranges:
                    if not row.min_date or not row.max_date:
                        continue
                    span = (row.max_date - row.min_date).days
                    if span < 14:
                        continue
                    model_midpoints[row.model] = row.min_date + timedelta(days=span // 2)

                if not model_midpoints:
                    return result

                # Single query: get counts per model split by older/recent
                # using each model's midpoint via CASE
                mid_models = list(model_midpoints.keys())
                rows = (
                    session.query(
                        DBAnalysisResult.model,
                        DBAnalysisResult.file_date,
                        DBTrackResult.linearity_pass,
                    )
                    .join(DBTrackResult, DBTrackResult.analysis_id == DBAnalysisResult.id)
                    .filter(
                        DBAnalysisResult.model.in_(mid_models),
                        DBAnalysisResult.file_date.isnot(None),
                    )
                    .all()
                )

                # Aggregate in Python — avoids complex per-model CASE in SQL
                from collections import defaultdict
                counts = defaultdict(lambda: {"older_total": 0, "older_passed": 0,
                                               "recent_total": 0, "recent_passed": 0})
                for row in rows:
                    mid = model_midpoints.get(row.model)
                    if mid is None:
                        continue
                    bucket = counts[row.model]
                    if row.file_date < mid:
                        bucket["older_total"] += 1
                        if row.linearity_pass:
                            bucket["older_passed"] += 1
                    else:
                        bucket["recent_total"] += 1
                        if row.linearity_pass:
                            bucket["recent_passed"] += 1

                for model, c in counts.items():
                    if c["recent_total"] < 5 or c["older_total"] < 5:
                        continue
                    recent_rate = c["recent_passed"] / c["recent_total"] * 100
                    older_rate = c["older_passed"] / c["older_total"] * 100
                    decline = older_rate - recent_rate  # positive = got worse

                    result[model] = {
                        "recent_rate": round(recent_rate, 1),
                        "older_rate": round(older_rate, 1),
                        "decline": round(decline, 1),
                    }
        except Exception as e:
            logger.error(f"Error computing linearity trends: {e}")

        return result

    def _compute_ft_pass_rates(self, db, days_back: int = 90) -> Dict[str, Dict]:
        """
        Compute per-model Final Test linearity pass rate.

        Uses same days_back window as trim data so the comparison is
        apples-to-apples on time period.

        Returns dict of model -> {ft_pass_rate, ft_samples}.
        """
        from datetime import timedelta
        from sqlalchemy import func, case
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
            FinalTestTrack as DBFinalTestTrack,
        )

        result = {}
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            with db.session() as session:
                ft_stats = (
                    session.query(
                        DBFinalTestResult.model,
                        func.count(DBFinalTestTrack.id).label('total'),
                        func.sum(
                            case((DBFinalTestTrack.linearity_pass == True, 1), else_=0)
                        ).label('passed'),
                    )
                    .join(DBFinalTestTrack, DBFinalTestResult.id == DBFinalTestTrack.final_test_id)
                    .filter(func.coalesce(DBFinalTestResult.test_date, DBFinalTestResult.file_date) >= cutoff_date)
                    .group_by(DBFinalTestResult.model)
                    .all()
                )

                for row in ft_stats:
                    total = row.total or 0
                    passed = row.passed or 0
                    if total >= 5:  # Need minimum samples
                        result[row.model] = {
                            "ft_pass_rate": round(passed / total * 100, 1),
                            "ft_samples": total,
                        }
        except Exception as e:
            logger.error(f"Error computing FT pass rates: {e}")

        return result

    # ── display ───────────────────────────────────────────────────

    def _on_filter_change(self):
        """Re-render with current filter."""
        self._active_only = self._active_var.get()
        self._render()

    def _update_display(self, priority: List[Dict[str, Any]]):
        """Store data and render."""
        self._model_data = priority
        self._update_label.configure(
            text=f"Updated: {datetime.now().strftime('%H:%M:%S')}"
        )
        self._render()

    def _show_error(self, msg: str):
        self._clear_content()
        lbl = ctk.CTkLabel(
            self._content, text=f"Error loading data: {msg}", text_color=COLOR_RED
        )
        lbl.pack(pady=40)

    def _clear_content(self):
        for w in self._content.winfo_children():
            w.destroy()

    def _render(self):
        """Render the full page from self._model_data."""
        self._clear_content()

        data = self._model_data
        if self._active_only:
            data = [m for m in data if m.get("is_active", True)]

        if not data:
            ctk.CTkLabel(
                self._content,
                text="No model data available. Process some files first.",
                text_color="gray",
                font=ctk.CTkFont(size=14),
            ).pack(pady=40)
            return

        # ── 1. Plant Health Banner ────────────────────────────────
        self._render_plant_banner(data)

        # ── 2. Models Needing Attention (top worst) ───────────────
        attention = [m for m in data if self._get_status(m) in ("critical", "warning")]
        if attention:
            self._render_attention_section(attention[:5])

        # ── 3. All Models Ranked Table ────────────────────────────
        self._render_model_table(data)

    # ── Plant Banner ──────────────────────────────────────────────

    def _render_plant_banner(self, data: List[Dict[str, Any]]):
        """Big status banner at the top."""
        banner = ctk.CTkFrame(self._content, corner_radius=12)
        banner.pack(fill="x", padx=5, pady=(5, 15))

        # Compute aggregates
        total_models = len(data)
        declining = sum(1 for m in data if m.get("trend_direction") == "declining")
        improving = sum(1 for m in data if m.get("trend_direction") == "improving")
        stable = total_models - declining - improving

        avg_pass = sum(m["linearity_pass_rate"] for m in data) / total_models if total_models else 0
        critical = sum(1 for m in data if self._get_status(m) == "critical")
        warning_count = sum(1 for m in data if self._get_status(m) == "warning")

        # Overall status
        if critical > 0:
            status_text = "NEEDS ATTENTION"
            status_color = COLOR_RED
            status_detail = f"{critical} model{'s' if critical != 1 else ''} in critical state"
        elif declining > 0:
            status_text = "WATCH LIST"
            status_color = COLOR_ORANGE
            status_detail = f"{declining} model{'s' if declining != 1 else ''} declining"
        elif warning_count > 0:
            status_text = "FAIR"
            status_color = COLOR_YELLOW
            status_detail = f"{warning_count} model{'s' if warning_count != 1 else ''} below target"
        else:
            status_text = "HEALTHY"
            status_color = COLOR_GREEN
            status_detail = "All models on track"

        # Top row: status text
        status_label = ctk.CTkLabel(
            banner,
            text=f"PLANT QUALITY:  {status_text}",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=status_color,
        )
        status_label.pack(padx=20, pady=(20, 4), anchor="w")

        detail_label = ctk.CTkLabel(
            banner,
            text=status_detail,
            font=ctk.CTkFont(size=13),
            text_color=COLOR_GRAY,
        )
        detail_label.pack(padx=20, pady=(0, 8), anchor="w")

        # Counts row
        counts_frame = ctk.CTkFrame(banner, fg_color="transparent")
        counts_frame.pack(fill="x", padx=20, pady=(0, 6))

        # Declining / Stable / Improving counts
        self._count_badge(counts_frame, f"{declining} Declining", COLOR_RED if declining else COLOR_GRAY)
        self._count_badge(counts_frame, f"{stable} Stable", COLOR_GRAY)
        self._count_badge(counts_frame, f"{improving} Improving", COLOR_GREEN if improving else COLOR_GRAY)

        # Average pass rate
        avg_color = self._rate_color(avg_pass)
        avg_label = ctk.CTkLabel(
            counts_frame,
            text=f"Avg Linearity Pass: {avg_pass:.1f}%",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=avg_color,
        )
        avg_label.pack(side="right", padx=10)

        # Bottom row: total models
        total_label = ctk.CTkLabel(
            banner,
            text=f"{total_models} active model{'s' if total_models != 1 else ''} shown",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_GRAY,
        )
        total_label.pack(padx=20, pady=(0, 15), anchor="w")

    def _count_badge(self, parent, text: str, color: str):
        """Small colored count label."""
        badge = ctk.CTkLabel(
            parent,
            text=f"  {text}  ",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=color,
        )
        badge.pack(side="left", padx=(0, 20))

    # ── Attention Cards ───────────────────────────────────────────

    def _render_attention_section(self, models: List[Dict[str, Any]]):
        """Prominent cards for models needing immediate attention."""
        header = ctk.CTkLabel(
            self._content,
            text="Models Needing Attention",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        header.pack(padx=10, pady=(10, 8), anchor="w")

        for m in models:
            self._render_attention_card(m)

    def _render_attention_card(self, m: Dict[str, Any]):
        """Single attention card for a struggling model."""
        status = self._get_status(m)
        border_color = COLOR_RED if status == "critical" else COLOR_ORANGE

        card = ctk.CTkFrame(self._content, corner_radius=10, border_width=2, border_color=border_color)
        card.pack(fill="x", padx=5, pady=4)

        # Inner layout: 3 columns — status+name | metrics | action
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=15, pady=12)
        inner.grid_columnconfigure(0, weight=0, minsize=200)
        inner.grid_columnconfigure(1, weight=1)
        inner.grid_columnconfigure(2, weight=0, minsize=300)

        # Col 0: Status dot + model name
        status_dot = "●"
        dot_color = COLOR_RED if status == "critical" else COLOR_ORANGE
        name_frame = ctk.CTkFrame(inner, fg_color="transparent")
        name_frame.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            name_frame,
            text=status_dot,
            font=ctk.CTkFont(size=18),
            text_color=dot_color,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkLabel(
            name_frame,
            text=m["model"],
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(side="left")

        # Col 1: Key metrics
        metrics_frame = ctk.CTkFrame(inner, fg_color="transparent")
        metrics_frame.grid(row=0, column=1, sticky="w", padx=20)

        pass_rate = m["linearity_pass_rate"]
        trend_arrow, trend_text, trend_color = self._trend_display(m)

        ctk.CTkLabel(
            metrics_frame,
            text=f"Trim: {pass_rate:.1f}%",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self._rate_color(pass_rate),
        ).pack(side="left", padx=(0, 12))

        # Show FT rate if available
        ft_rate = m.get("ft_pass_rate")
        if ft_rate is not None:
            ctk.CTkLabel(
                metrics_frame,
                text=f"FT: {ft_rate:.1f}%",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=self._rate_color(ft_rate),
            ).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(
            metrics_frame,
            text=f"{trend_arrow} {trend_text}",
            font=ctk.CTkFont(size=13),
            text_color=trend_color,
        ).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(
            metrics_frame,
            text=f"{m['total_tracks']:,} samples",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_GRAY,
        ).pack(side="left")

        # Col 2: Recommendation
        rec = m.get("recommendation", "Review needed")
        ctk.CTkLabel(
            inner,
            text=rec,
            font=ctk.CTkFont(size=12),
            text_color=COLOR_WHITE,
            wraplength=280,
            justify="left",
        ).grid(row=0, column=2, sticky="e")

    # ── All Models Table ──────────────────────────────────────────

    def _render_model_table(self, data: List[Dict[str, Any]]):
        """Ranked table of all models, sorted worst-first."""
        header = ctk.CTkLabel(
            self._content,
            text="All Models — Ranked by Impact",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        header.pack(padx=10, pady=(20, 8), anchor="w")

        # Check if any model has FT data — only show FT columns if data exists
        has_ft_data = any(m.get("ft_pass_rate") is not None for m in data)

        # Table header
        table_frame = ctk.CTkFrame(self._content, corner_radius=8)
        table_frame.pack(fill="x", padx=5, pady=(0, 5))
        col = 0
        table_frame.grid_columnconfigure(col, weight=0, minsize=50);   col += 1  # Status
        table_frame.grid_columnconfigure(col, weight=1, minsize=110);  col += 1  # Model
        table_frame.grid_columnconfigure(col, weight=0, minsize=90);   col += 1  # Trim Pass
        if has_ft_data:
            table_frame.grid_columnconfigure(col, weight=0, minsize=80);  col += 1  # FT Pass
            table_frame.grid_columnconfigure(col, weight=0, minsize=80);  col += 1  # Gap
        table_frame.grid_columnconfigure(col, weight=0, minsize=90);   col += 1  # Trend
        table_frame.grid_columnconfigure(col, weight=0, minsize=80);   col += 1  # Change
        table_frame.grid_columnconfigure(col, weight=0, minsize=70);   col += 1  # Samples
        table_frame.grid_columnconfigure(col, weight=1, minsize=180);  col += 1  # Action

        headers = ["", "Model", "Trim %"]
        if has_ft_data:
            headers += ["FT %", "Gap"]
        headers += ["Trend", "Change", "Samples", "Action"]
        for col, h in enumerate(headers):
            lbl = ctk.CTkLabel(
                table_frame,
                text=h,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=COLOR_GRAY,
            )
            lbl.grid(row=0, column=col, padx=8, pady=(10, 6), sticky="w")

        # Sort: critical first, then warning, then ok, then good — within each group by pass rate ascending
        status_order = {"critical": 0, "warning": 1, "ok": 2, "good": 3}
        sorted_data = sorted(
            data,
            key=lambda m: (
                status_order.get(self._get_status(m), 9),
                m["linearity_pass_rate"],
            ),
        )

        for row_idx, m in enumerate(sorted_data, start=1):
            self._render_table_row(table_frame, row_idx, m, has_ft_data)

        # Bottom padding
        ctk.CTkLabel(table_frame, text="").grid(
            row=len(sorted_data) + 1, column=0, pady=5
        )

    def _render_table_row(self, parent, row: int, m: Dict[str, Any], has_ft: bool = False):
        """One row in the model table."""
        status = self._get_status(m)
        pass_rate = m["linearity_pass_rate"]
        trend_arrow, trend_text, trend_color = self._trend_display(m)

        # Alternate row background
        bg = None  # use default

        # Status dot
        dot_color = {
            "critical": COLOR_RED,
            "warning": COLOR_ORANGE,
            "ok": COLOR_YELLOW,
            "good": COLOR_GREEN,
        }.get(status, COLOR_GRAY)

        # Use dynamic column index so FT columns slot in when present
        c = 0

        # Col: Status dot
        ctk.CTkLabel(
            parent, text="●", text_color=dot_color,
            font=ctk.CTkFont(size=14),
        ).grid(row=row, column=c, padx=8, pady=4, sticky="w"); c += 1

        # Col: Model name
        ctk.CTkLabel(
            parent, text=m["model"],
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=row, column=c, padx=8, pady=4, sticky="w"); c += 1

        # Col: Trim pass rate
        ctk.CTkLabel(
            parent, text=f"{pass_rate:.1f}%",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self._rate_color(pass_rate),
        ).grid(row=row, column=c, padx=8, pady=4, sticky="w"); c += 1

        # Col: FT pass rate + Gap (only if FT data exists)
        if has_ft:
            ft_rate = m.get("ft_pass_rate")
            if ft_rate is not None:
                ft_text = f"{ft_rate:.1f}%"
                ft_color = self._rate_color(ft_rate)
            else:
                ft_text = "—"
                ft_color = COLOR_GRAY

            ctk.CTkLabel(
                parent, text=ft_text,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=ft_color,
            ).grid(row=row, column=c, padx=8, pady=4, sticky="w"); c += 1

            # Gap: trim - FT. Positive = escapes (trim says pass, FT says fail)
            gap = m.get("trim_ft_gap")
            if gap is not None:
                if abs(gap) < 2.0:
                    gap_text = f"{gap:+.1f}"
                    gap_color = COLOR_GRAY
                elif gap > 0:
                    # Trim higher than FT → escapes likely
                    gap_text = f"{gap:+.1f}"
                    gap_color = COLOR_ORANGE
                else:
                    # FT higher than Trim → overkill (trim rejecting good units)
                    gap_text = f"{gap:+.1f}"
                    gap_color = COLOR_LIGHT_GREEN
            else:
                gap_text = "—"
                gap_color = COLOR_GRAY

            ctk.CTkLabel(
                parent, text=gap_text,
                font=ctk.CTkFont(size=12),
                text_color=gap_color,
            ).grid(row=row, column=c, padx=8, pady=4, sticky="w"); c += 1

        # Col: Trend arrow
        ctk.CTkLabel(
            parent, text=f"{trend_arrow} {trend_text}",
            font=ctk.CTkFont(size=12),
            text_color=trend_color,
        ).grid(row=row, column=c, padx=8, pady=4, sticky="w"); c += 1

        # Col: Change
        decline = m.get("decline", 0)
        if decline > 0:
            change_text = f"-{decline:.1f}%"
            change_color = COLOR_RED
        elif decline < 0:
            change_text = f"+{abs(decline):.1f}%"
            change_color = COLOR_GREEN
        else:
            change_text = "—"
            change_color = COLOR_GRAY

        ctk.CTkLabel(
            parent, text=change_text,
            font=ctk.CTkFont(size=12),
            text_color=change_color,
        ).grid(row=row, column=c, padx=8, pady=4, sticky="w"); c += 1

        # Col: Samples
        ctk.CTkLabel(
            parent, text=f"{m['total_tracks']:,}",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_GRAY,
        ).grid(row=row, column=c, padx=8, pady=4, sticky="w"); c += 1

        # Col: Action / recommendation
        rec = m.get("recommendation", "Review needed")
        ctk.CTkLabel(
            parent, text=rec,
            font=ctk.CTkFont(size=11),
            text_color=COLOR_WHITE,
            wraplength=220,
            justify="left",
        ).grid(row=row, column=c, padx=8, pady=4, sticky="w")

    # ── helpers ────────────────────────────────────────────────────

    def _get_status(self, m: Dict[str, Any]) -> str:
        """Classify model health: critical / warning / ok / good."""
        rate = m["linearity_pass_rate"]
        if rate < THRESHOLD_WARNING:
            return "critical"
        elif rate < THRESHOLD_OK:
            return "warning"
        elif rate < THRESHOLD_GOOD:
            return "ok"
        return "good"

    def _rate_color(self, rate: float) -> str:
        """Colour for a pass rate value."""
        if rate >= THRESHOLD_GOOD:
            return COLOR_GREEN
        elif rate >= THRESHOLD_OK:
            return COLOR_LIGHT_GREEN
        elif rate >= THRESHOLD_WARNING:
            return COLOR_ORANGE
        return COLOR_RED

    def _trend_display(self, m: Dict[str, Any]):
        """Return (arrow, text, colour) for trend direction."""
        direction = m.get("trend_direction", "stable")
        decline = m.get("decline", 0)

        if direction == "declining":
            return ("↓", "Declining", COLOR_RED)
        elif direction == "improving":
            return ("↑", "Improving", COLOR_GREEN)
        else:
            return ("→", "Stable", COLOR_GRAY)
