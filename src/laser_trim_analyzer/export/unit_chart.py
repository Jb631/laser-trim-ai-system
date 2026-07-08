"""Print-ready unit chart export (2026-07-07).

Port of V5 Analyze's 4-panel light-mode export (error plot + unit info +
metrics + status) as a SHARED builder, so the V6 unit modal's "Save chart…"
produces the same handoff-quality document instead of a dark screen dump.

One deliberate change from V5: the status box used OR-logic (sigma_pass OR
linearity_pass -> WARNING), which let a linearity-FAIL show as WARNING.
Domain rule (James, 2026-07-07): linearity is the zero-tolerance customer
requirement and decides accept/reject; sigma only separates PASS from
"pass, watch process". The builder applies that rule.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

_C = {
    "pass": "#27ae60", "fail": "#e74c3c", "warning": "#f39c12",
    "trimmed": "#27ae60", "untrimmed": "#3498db", "spec": "#e74c3c",
}


def _fmt(v, spec=".6f", na="N/A"):
    return format(v, spec) if isinstance(v, (int, float)) else na


def build_unit_export_figure(meta: Dict[str, Any], data: Dict[str, Any],
                             fail_points: Optional[List[int]] = None) -> Figure:
    """Light-mode 4-panel unit document.

    meta: model, serial, system, trim_date (str), track_id, n_tracks.
    data: load_unit_track dict (arrays + stored metrics).
    """
    fail_points = fail_points or []
    fig = Figure(figsize=(11, 8.5), dpi=120, facecolor="white")  # landscape letter
    gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.0],
                          hspace=0.28, wspace=0.18,
                          left=0.07, right=0.97, top=0.90, bottom=0.05)
    ax = fig.add_subplot(gs[0, :])
    ax_info = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_status = fig.add_subplot(gs[1, 2])

    title = f"{meta.get('model', '')} — Unit {meta.get('serial', '')}"
    if meta.get("n_tracks", 1) and meta["n_tracks"] > 1:
        title += f" (track {meta.get('track_id', '?')} of {meta['n_tracks']})"
    fig.suptitle(title, fontsize=14, fontweight="bold", color="black")

    # ---- main error-vs-position plot (light mode) ----
    pos = data.get("position_data") or []
    err = data.get("error_data") or []
    if not pos or not err:
        up, ue = data.get("untrimmed_positions") or [], data.get("untrimmed_errors") or []
        if up and ue:
            ax.plot(up, ue, "--", lw=1.5, color=_C["untrimmed"], alpha=0.8,
                    label="Untrimmed (test sweep — no trim run)")
            ax.legend(loc="lower right", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No measurement data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="black")
    else:
        positions = np.asarray(pos, dtype=float)
        errors = np.asarray(err, dtype=float)
        offset = float(data.get("optimal_offset") or 0.0)
        corrected = errors + offset

        up, ue = data.get("untrimmed_positions") or [], data.get("untrimmed_errors") or []
        if up and ue:
            imp = data.get("trim_improvement_percent")
            lbl = "Untrimmed" + (f" (trim improved {imp:.0f}%)"
                                 if isinstance(imp, (int, float)) else "")
            ax.plot(up, ue, "--", lw=1.4, color=_C["untrimmed"], alpha=0.6, label=lbl)

        corr_lbl = ("Trimmed corrected (no-op)" if abs(offset) < 1e-9
                    else f"Trimmed corrected (offset: {offset:+.6f})")
        ax.plot(positions, corrected, lw=2, color=_C["trimmed"], zorder=3, label=corr_lbl)
        ax.plot(positions, errors, "--", lw=1.1, color=_C["trimmed"], alpha=0.35,
                zorder=2, label="Trimmed (as measured)")

        upper = data.get("upper_limits") or []
        lower = data.get("lower_limits") or []
        if upper and lower:
            u = np.asarray([v if v is not None else np.nan for v in upper], dtype=float)
            lo = np.asarray([v if v is not None else np.nan for v in lower], dtype=float)
            n = min(len(positions), len(u), len(lo))
            ax.plot(positions[:n], u[:n], "--", lw=1, color=_C["spec"], alpha=0.8,
                    label="Spec limits")
            ax.plot(positions[:n], lo[:n], "--", lw=1, color=_C["spec"], alpha=0.8)
            ax.fill_between(positions[:n], lo[:n], u[:n], color=_C["spec"], alpha=0.06,
                            where=~np.isnan(u[:n]) & ~np.isnan(lo[:n]))
        if fail_points:
            fx = [positions[i] for i in fail_points if i < len(positions)]
            fy = [corrected[i] for i in fail_points if i < len(corrected)]
            ax.scatter(fx, fy, marker="x", s=55, color=_C["fail"], zorder=5,
                       linewidths=2, label=f"Fail points ({len(fail_points)})")
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25)
    ax.tick_params(colors="black", labelsize=9)
    ax.set_xlabel("Position", fontsize=11, color="black")
    ax.set_ylabel("Error (Volts)", fontsize=11, color="black")
    for s in ax.spines.values():
        s.set_color("black")

    # ---- unit information panel ----
    ax_info.axis("off")
    lines = [f"Model: {meta.get('model', 'N/A')}",
             f"Serial: {meta.get('serial', 'N/A')}",
             f"System: {meta.get('system', 'N/A')}",
             f"Track: {meta.get('track_id', 'N/A')}",
             f"Trim Date: {meta.get('trim_date', 'N/A')}", ""]
    if data.get("untrimmed_resistance") is not None:
        lines.append(f"Untrimmed R: {_fmt(data['untrimmed_resistance'], '.1f')}")
    if data.get("trimmed_resistance") is not None:
        lines.append(f"Trimmed R: {_fmt(data['trimmed_resistance'], '.1f')}")
    if data.get("resistance_change_percent") is not None:
        lines.append(f"R Change: {data['resistance_change_percent']:+.1f}%")
    if data.get("measured_electrical_angle") is not None:
        lines.append(f"Meas. Elec. Angle: {_fmt(data['measured_electrical_angle'], '.4g')}")
    if data.get("trim_improvement_percent") is not None:
        lines.append(f"Trim Improvement: {data['trim_improvement_percent']:.1f}%")
    ax_info.text(0.02, 0.98, "Unit Information", fontsize=11, fontweight="bold",
                 va="top", transform=ax_info.transAxes, color="black")
    for i, ln in enumerate(lines):
        ax_info.text(0.02, 0.88 - i * 0.075, ln, fontsize=9.5, va="top",
                     transform=ax_info.transAxes, color="black")

    # ---- metrics panel ----
    ax_metrics.axis("off")
    sig_pass = data.get("sigma_pass")
    lin_pass = data.get("linearity_pass")
    rows = [
        f"Sigma Gradient: {_fmt(data.get('sigma_gradient'))}",
        f"Sigma Threshold: {_fmt(data.get('sigma_threshold'))}",
        f"Sigma Watch: {'no' if sig_pass else ('YES' if sig_pass is not None else 'N/A')}",
        "",
        f"Optimal Offset: {_fmt(data.get('optimal_offset'))}",
        f"Linearity Error: {_fmt(data.get('linearity_error'))}",
        f"Fail Points: {len(fail_points)}",
        f"Linearity Pass: {'YES' if lin_pass else ('NO' if lin_pass is not None else 'N/A')}",
    ]
    ax_metrics.text(0.02, 0.98, "Analysis Metrics", fontsize=11, fontweight="bold",
                    va="top", transform=ax_metrics.transAxes, color="black")
    for i, ln in enumerate(rows):
        col = "black"
        if ln.endswith(": NO") or ln.endswith("Watch: YES"):
            col = _C["fail"] if "Linearity" in ln else _C["warning"]
        elif ln.endswith(": YES"):
            col = _C["pass"]
        ax_metrics.text(0.02, 0.88 - i * 0.085, ln, fontsize=9.5, va="top",
                        transform=ax_metrics.transAxes, color=col)

    # ---- status panel (domain rule: linearity decides; sigma = watch flag) ----
    ax_status.axis("off")
    if lin_pass is False:
        status, color, note = "FAIL", _C["fail"], "Linearity out of spec (zero-tolerance)"
    elif lin_pass and sig_pass:
        status, color, note = "PASS", _C["pass"], "Linearity in spec · no sigma watch"
    elif lin_pass:
        status, color, note = "PASS (WATCH)", _C["warning"], "Linearity in spec · sigma drift-watch"
    else:
        status, color, note = "NOT EVALUATED", "#7f8c8d", "No linearity grading stored"
    ax_status.add_patch(Rectangle((0.06, 0.58), 0.88, 0.30, linewidth=3,
                                  edgecolor=color, facecolor="white"))
    ax_status.text(0.5, 0.73, status, ha="center", va="center", fontsize=15,
                   fontweight="bold", color=color, transform=ax_status.transAxes)
    ax_status.text(0.5, 0.44, note, ha="center", va="center", fontsize=9.5,
                   color=color, transform=ax_status.transAxes, wrap=True)
    ax_status.text(0.5, 0.12, f"Exported {datetime.now():%Y-%m-%d %H:%M}",
                   ha="center", va="center", fontsize=8.5, color="gray",
                   style="italic", transform=ax_status.transAxes)
    return fig
