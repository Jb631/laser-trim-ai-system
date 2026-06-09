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
        self.geometry("900x560")
        self.configure(fg_color=theme.SURFACE)
        self.transient(master)
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
