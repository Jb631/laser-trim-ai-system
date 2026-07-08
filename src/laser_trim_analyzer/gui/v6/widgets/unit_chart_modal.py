"""Spec 3c — UnitChartModal: per-unit pre/post-trim chart (kept V5 drill-down). Q1: show every point."""
from typing import List, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


def compute_fail_points(errors, upper_limits, lower_limits,
                        offset: float = 0.0) -> List[int]:
    """Indices where the CORRECTED post-trim error violates the per-point spec
    band (zero-tolerance, Q1).

    `offset` is the stored optimal_offset — grading of record is done on the
    corrected trace. Checking raw errors marked 'fail points' on units whose
    corrected trace is in spec (X markers contradicting 'Linearity Pass: YES',
    found by the 2026-07-07 QA sweep on unit 8074-1/30)."""
    out = []
    off = offset or 0.0
    for i, e in enumerate(errors or []):
        if e is None:
            continue
        e = e + off
        up = upper_limits[i] if upper_limits and i < len(upper_limits) else None
        lo = lower_limits[i] if lower_limits and i < len(lower_limits) else None
        if (up is not None and e > up) or (lo is not None and e < lo):
            out.append(i)
    return out


def load_unit_track(db, analysis_id: int,
                    track_id: Optional[str] = None) -> Optional[dict]:
    """Materialize ONE track's arrays for an analysis INSIDE the session
    (I8-safe). track_id None = first track. Returns all track ids so the
    modal can offer a selector on multi-track units (2026-07-07 — the modal
    used to hard-lock to track 1, hiding the other elements of exactly the
    units most worth investigating)."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR)
    with db.session() as s:
        tracks = s.query(DBTR).filter(DBTR.analysis_id == analysis_id).order_by(DBTR.track_id).all()
        if not tracks:
            return None
        ar = s.query(DBAR.model, DBAR.system).filter(DBAR.id == analysis_id).first()
        model = ar[0] if ar else ""
        system = getattr(ar[1], "value", str(ar[1])) if ar and ar[1] is not None else ""
        track_ids = [t.track_id for t in tracks]
        tr = tracks[0]
        if track_id is not None:
            for t in tracks:
                if t.track_id == track_id:
                    tr = t
                    break
        return {
            "model": model, "system": system,
            "track_id": tr.track_id, "n_tracks": len(tracks),
            "track_ids": track_ids,
            "position_data": list(tr.position_data or []),
            "error_data": list(tr.error_data or []),
            "upper_limits": list(tr.upper_limits or []),
            "lower_limits": list(tr.lower_limits or []),
            "untrimmed_positions": list(tr.untrimmed_positions or []),
            "untrimmed_errors": list(tr.untrimmed_errors or []),
            # Context v5's Analyze chart always showed (v6 parity, 2026-07-07):
            # corrected trace offset + how much the trim improved the unit.
            "optimal_offset": tr.optimal_offset,
            "trim_improvement_percent": tr.trim_improvement_percent,
            # Detail metrics for the print-ready export (export/unit_chart.py).
            "sigma_gradient": tr.sigma_gradient,
            "sigma_threshold": tr.sigma_threshold,
            "sigma_pass": tr.sigma_pass,
            "linearity_error": tr.final_linearity_error_shifted,
            "linearity_pass": tr.linearity_pass,
            "untrimmed_resistance": tr.untrimmed_resistance,
            "trimmed_resistance": getattr(tr, "trimmed_resistance", None),
            "resistance_change_percent": tr.resistance_change_percent,
            "measured_electrical_angle": tr.measured_electrical_angle,
        }


class UnitChartModal(ctk.CTkToplevel):
    def __init__(self, master, theme: ThemeManager, db, unit: dict):
        super().__init__(master)
        self.theme = theme
        self._unit = unit
        status = str(unit.get("overall_status", "") or "").strip()
        date_only = str(unit.get("file_date", "")).split(" ")[0]
        # Day-granularity data: '2026-05-27 00:00:00' is noise; status in the
        # title answers 'is this unit good?' without reading the chart.
        self.title(f"Unit {unit.get('serial', '')} — {date_only}"
                   + (f" — {status.upper()}" if status else ""))
        self.geometry("900x600")
        self.configure(fg_color=theme.SURFACE)
        self.transient(master)
        from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle
        self._chart = chart = ChartWidget(self, style=ChartStyle(figure_size=(9, 5)))
        chart.pack(side="top", fill="both", expand=True, padx=theme.SPACE_MD, pady=theme.SPACE_MD)
        # Action bar: restore the single-unit chart export that V5 had (ChartWidget
        # still exposes save_figure; the V6 modal just never surfaced a button for it).
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(side="bottom", fill="x", padx=theme.SPACE_MD, pady=(0, theme.SPACE_MD))
        self._save_btn = ctk.CTkButton(bar, text="Save chart…", command=self._save_chart,
                                       fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                                       text_color=theme.TEXT_INVERSE, corner_radius=theme.RADIUS_SM)
        self._save_btn.pack(side="right")
        # Track selector (packed only when the unit has multiple tracks).
        self._track_menu = ctk.CTkOptionMenu(
            bar, values=["Track"], width=140, command=self._on_track_change,
            fg_color=theme.CARD, button_color=theme.ACCENT,
            button_hover_color=theme.ACCENT_HOVER, text_color=theme.TEXT_PRIMARY)
        self._db = db

        # Resolve the app's UiDispatcher now (main thread); master may be nested.
        from laser_trim_analyzer.gui.v6.ui_dispatch import resolve_dispatcher
        self._ui = resolve_dispatcher(master)

        # Fetch on a worker: this DB read used to run HERE on the main thread,
        # so opening a unit chart during a batch save / training froze the UI
        # until the DB lock was released.
        chart.show_placeholder("Loading unit data…")
        self._save_btn.configure(state="disabled")

        def work():
            try:
                data = load_unit_track(db, unit.get("analysis_id"))
            except Exception:
                data = None
            self._post(lambda: self._render(data))

        import threading
        threading.Thread(target=work, daemon=True).start()

    def _post(self, fn) -> None:
        def guarded():
            try:
                if self.winfo_exists():
                    fn()
            except Exception:
                pass
        if self._ui is not None:
            self._ui.post(guarded)
        else:
            guarded()  # tests: main-thread, no dispatcher

    def _on_track_change(self, choice: str) -> None:
        """Reload the modal for the selected track (worker, like initial load)."""
        data = getattr(self, "_track_data", None)
        if not data or choice == data.get("track_id"):
            return
        self._chart.show_placeholder(f"Loading {choice}…")

        def work():
            try:
                new = load_unit_track(self._db, self._unit.get("analysis_id"),
                                      track_id=choice)
            except Exception:
                new = None
            self._post(lambda: self._render(new))

        import threading
        threading.Thread(target=work, daemon=True).start()

    def _render(self, data: Optional[dict]) -> None:
        self._track_data = data  # kept for the print-ready export
        unit = self._unit
        # Multi-track: surface the selector with the real track ids.
        if data and data.get("n_tracks", 1) > 1 and data.get("track_ids"):
            self._track_menu.configure(values=data["track_ids"])
            self._track_menu.set(data.get("track_id"))
            self._track_menu.pack(side="left")
        if not data or not data["position_data"]:
            self._chart.show_placeholder(
                "No linearity sweep stored for this unit.\n\n"
                "It has other measurements (e.g. resistance) but no point-by-point error "
                "trace — so it was never fully linearity-evaluated, which is why it reads "
                "as Warning rather than Pass/Fail. Re-process the source file to capture "
                "the sweep.")
            self._save_btn.configure(state="disabled")
            return
        fp = compute_fail_points(data["error_data"], data["upper_limits"], data["lower_limits"],
                                 offset=data.get("optimal_offset") or 0.0)
        title = f"Unit {unit.get('serial', '')}"
        if data["n_tracks"] > 1:
            title += f" — {data['track_id']} of {data['n_tracks']} tracks"
        self._chart.plot_error_vs_position(
            positions=data["position_data"], trimmed_errors=data["error_data"],
            upper_limits=data["upper_limits"] or None, lower_limits=data["lower_limits"] or None,
            untrimmed_positions=data["untrimmed_positions"] or None,
            untrimmed_errors=data["untrimmed_errors"] or None,
            offset=data.get("optimal_offset") or 0.0,
            trim_improvement_percent=data.get("trim_improvement_percent"),
            trim_date=str(unit.get("file_date", "")).split(" ")[0] or None,
            fail_points=fp, title=title, serial_number=str(unit.get("serial", "")))
        self._save_btn.configure(state="normal")

    def _save_chart(self) -> None:
        """Export the print-ready 4-panel unit document (v5 parity).

        The on-screen dark figure is for working; the SAVED chart is what gets
        handed to the team — light background, unit info, metrics, and status
        panels (export/unit_chart.py), like V5's Export Chart produced.
        """
        from tkinter import filedialog
        serial = str(self._unit.get("serial", "unit"))
        date = str(self._unit.get("file_date", "")).split(" ")[0]
        initial = f"unit_{serial}{('_' + date) if date else ''}.png"
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".png", initialfile=initial,
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not path:
            return
        data = getattr(self, "_track_data", None)
        if not data:
            self._chart.save_figure(path)  # nothing loaded — fall back
            return
        try:
            from laser_trim_analyzer.export.unit_chart import build_unit_export_figure
            from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import compute_fail_points
            fp = compute_fail_points(data.get("error_data"), data.get("upper_limits"),
                                     data.get("lower_limits"),
                                     offset=data.get("optimal_offset") or 0.0)
            meta = {"model": data.get("model") or self._unit.get("model", ""),
                    "serial": self._unit.get("serial", ""),
                    "system": data.get("system") or self._unit.get("system", ""),
                    "trim_date": date,
                    "track_id": data.get("track_id"),
                    "n_tracks": data.get("n_tracks", 1)}
            fig = build_unit_export_figure(meta, data, fp)
            fig.savefig(path, facecolor="white", bbox_inches="tight")
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Print-ready export failed; "
                                                  "saving on-screen figure instead")
            self._chart.save_figure(path)
