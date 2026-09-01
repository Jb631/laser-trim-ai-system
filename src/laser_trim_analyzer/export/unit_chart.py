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


def corrected_errors(errors, offset=0.0, k=0.0, theory=None):
    """The GRADED trace: `error + theory*k + offset`.

    Single definition of the adjustment, shared by this export, the V6 unit
    modal's fail-point count and its on-screen chart, so all three can never
    disagree about which curve is being judged.

    k (stored as TrackResult.optimal_slope) is the theory ROTATION factor;
    dropping it re-grades the unit on a curve the analyzer never judged and
    invents fail points at the end of travel, where theory is largest —
    the 2026-08-31 "Fail Points: 18 / Linearity Pass: YES" contradiction on
    8415-1 SN 26. Mirrors analyzer._calculate_linearity, including its
    `if theory_volts and optimal_k != 0` guard: no theory column or no k
    means offset-only, exactly as the analyzer graded it.
    """
    errs = list(errors or [])
    off = float(offset or 0.0)
    k = float(k or 0.0)
    if not k or not theory or len(theory) < len(errs):
        return [None if e is None else e + off for e in errs]
    return [None if e is None else e + (theory[i] or 0.0) * k + off
            for i, e in enumerate(errs)]


def build_unit_export_figure(meta: Dict[str, Any], data: Dict[str, Any],
                             fail_points: Optional[List[int]] = None,
                             kind: str = "trim") -> Figure:
    """Light-mode 4-panel unit document.

    meta: model, serial, system, trim_date (str), track_id, n_tracks.
    data: load_unit_track (trim) or load_ft_track (FT) dict.
    kind: "trim" (default) or "ft". FT drops the sigma metrics (there is no
    sigma at final test) and takes its PASS/FAIL from the final-test result.
    """
    fail_points = fail_points or []
    is_ft = kind == "ft"
    fig = Figure(figsize=(11, 8.5), dpi=120, facecolor="white")  # landscape letter
    gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.0],
                          hspace=0.28, wspace=0.18,
                          left=0.07, right=0.97, top=0.90, bottom=0.05)
    ax = fig.add_subplot(gs[0, :])
    ax_info = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_status = fig.add_subplot(gs[1, 2])

    title = f"{meta.get('model', '')} — " + ("Final Test — " if is_ft else "") \
            + f"Unit {meta.get('serial', '')}"
    if meta.get("n_tracks", 1) and meta["n_tracks"] > 1:
        title += f" (track {meta.get('track_id', '?')} of {meta['n_tracks']})"
    fig.suptitle(title, fontsize=14, fontweight="bold", color="black")

    # ---- main error-vs-position plot (light mode) ----
    pos = data.get("position_data") or []
    err = data.get("error_data") or []
    # Only a drawn sweep can be re-graded. With no trace there are no fail
    # markers either, so nothing can contradict the stored verdict and it
    # stands on its own (a resistance-only row must not read as "in spec").
    has_trace = bool(pos and err)
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
        k = float(data.get("optimal_slope") or 0.0)
        corrected = np.asarray(
            corrected_errors(err, offset, k, data.get("theory_data")),
            dtype=float)

        up, ue = data.get("untrimmed_positions") or [], data.get("untrimmed_errors") or []
        if up and ue:
            imp = data.get("trim_improvement_percent")
            lbl = "Untrimmed" + (f" (trim improved {imp:.0f}%)"
                                 if isinstance(imp, (int, float)) else "")
            ax.plot(up, ue, "--", lw=1.4, color=_C["untrimmed"], alpha=0.6, label=lbl)

        _base = "Final test" if is_ft else "Trimmed"
        _rot = k and data.get("theory_data")
        if abs(offset) < 1e-9 and not _rot:
            corr_lbl = f"{_base} (no-op)"
        else:
            corr_lbl = f"{_base} corrected (offset: {offset:+.6f}"
            corr_lbl += f", k: {k:+.6f})" if _rot else ")"
        ax.plot(positions, corrected, lw=2, color=_C["trimmed"], zorder=3, label=corr_lbl)
        ax.plot(positions, errors, "--", lw=1.1, color=_C["trimmed"], alpha=0.35,
                zorder=2, label=f"{_base} (as measured)")

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
    _date_label = "Test Date" if is_ft else "Trim Date"
    lines = [f"Model: {meta.get('model', 'N/A')}",
             f"Serial: {meta.get('serial', 'N/A')}",
             f"System: {meta.get('system', 'N/A')}",
             f"Track: {meta.get('track_id', 'N/A')}",
             f"{_date_label}: {meta.get('trim_date', 'N/A')}", ""]
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
    if is_ft:
        # No sigma at final test — show the linearity numbers + spec instead.
        # Linearity Pass is RE-GRADED on the applied offset so it agrees with the
        # Fail Points count above it; the station's own disposition is kept on a
        # separate 'Station Result' line so the two frames never contradict.
        ft_pass = (len(fail_points) == 0)
        _station = str(data.get("result") or "").upper() or (
            "PASS" if lin_pass else ("FAIL" if lin_pass is False else "N/A"))
        rows = [
            f"Optimal Offset: {_fmt(data.get('optimal_offset'))}",
            f"Linearity Error: {_fmt(data.get('linearity_error'))}",
            f"Linearity Spec: {_fmt(data.get('linearity_spec'))}",
            "",
            f"Fail Points: {len(fail_points)}",
            f"Linearity Pass: {'YES' if ft_pass else 'NO'}",
            f"Station Result: {_station}",
        ]
    else:
        # Same rule the FT branch above applies: Linearity Pass is RE-GRADED on
        # the fail points printed directly above it, so the two lines can never
        # contradict each other. Pairing a renderer-computed count with the
        # STORED verdict is what printed "Fail Points: 18 / Linearity Pass: YES"
        # on 8415-1 SN 26. When the re-grade disagrees with what the analyzer
        # stored, the stored verdict gets its own line rather than vanishing —
        # a zero-tolerance customer disposition is never silently overwritten.
        trim_pass = (len(fail_points) == 0) if has_trace else bool(lin_pass)
        rows = [
            f"Sigma Gradient: {_fmt(data.get('sigma_gradient'))}",
            f"Sigma Threshold: {_fmt(data.get('sigma_threshold'))}",
            f"Sigma Watch: {'no' if sig_pass else ('YES' if sig_pass is not None else 'N/A')}",
            "",
            f"Optimal Offset: {_fmt(data.get('optimal_offset'))}",
            f"Linearity Error: {_fmt(data.get('linearity_error'))}",
            f"Fail Points: {len(fail_points)}",
            f"Linearity Pass: {'YES' if trim_pass else ('NO' if (has_trace or lin_pass is not None) else 'N/A')}",
        ]
        if has_trace and lin_pass is not None and bool(lin_pass) != trim_pass:
            rows.append(f"Stored Verdict: {'PASS' if lin_pass else 'FAIL'}")
    ax_metrics.text(0.02, 0.98, "Final Test Metrics" if is_ft else "Analysis Metrics",
                    fontsize=11, fontweight="bold",
                    va="top", transform=ax_metrics.transAxes, color="black")
    for i, ln in enumerate(rows):
        col = "black"
        if ln.startswith("Stored Verdict:"):
            # Only printed when it disagrees with the re-grade — amber either
            # way, because either direction means "reprocess this unit".
            col = _C["warning"]
        elif ln.endswith(": NO") or ln.endswith("Watch: YES") or ln.endswith("Result: FAIL"):
            col = _C["fail"] if ("Linearity" in ln or "Result" in ln) else _C["warning"]
        elif ln.endswith(": YES"):
            col = _C["pass"]
        ax_metrics.text(0.02, 0.88 - i * 0.085, ln, fontsize=9.5, va="top",
                        transform=ax_metrics.transAxes, color=col)

    # ---- status panel (domain rule: linearity decides; sigma = watch flag) ----
    ax_status.axis("off")
    if is_ft:
        # Stamp is RE-GRADED on the applied offset so it agrees with the chart's
        # fail markers; when that clears a sweep the station recorded as FAIL we
        # flag it (PASS*, amber) and keep the station disposition in the note
        # rather than silently laundering a customer reject into a clean PASS.
        ft_pass = (len(fail_points) == 0)
        _result = str(data.get("result") or "").upper()
        station_fail = (_result == "FAIL") or (lin_pass is False)
        if not ft_pass:
            status, color, note = "FAIL", _C["fail"], "Final test — out of spec"
        elif station_fail:
            status, color, note = ("PASS*", _C["warning"],
                                   "In spec under best-fit offset · station: FAIL — reprocess")
        elif _result == "PASS" or lin_pass:
            status, color, note = "PASS", _C["pass"], "Final test — accepted"
        else:
            status, color, note = "NOT EVALUATED", "#7f8c8d", "No final-test grading stored"
    elif not has_trace:
        # No sweep to re-grade — the stored verdict stands alone.
        if lin_pass is False:
            status, color, note = "FAIL", _C["fail"], "Linearity out of spec (zero-tolerance)"
        elif lin_pass and sig_pass:
            status, color, note = "PASS", _C["pass"], "Linearity in spec · no sigma watch"
        elif lin_pass:
            status, color, note = "PASS (WATCH)", _C["warning"], "Linearity in spec · sigma drift-watch"
        else:
            status, color, note = "NOT EVALUATED", "#7f8c8d", "No linearity grading stored"
    else:
        # Stamp is RE-GRADED on the SAME fail points the chart marks, so the
        # markers, the count, the Linearity Pass line and this stamp are one
        # verdict. Disagreement with the stored value is surfaced, never
        # laundered — in EITHER direction (both are reprocess signals).
        trim_pass = (len(fail_points) == 0)
        stored_disagrees = lin_pass is not None and bool(lin_pass) != trim_pass
        if not trim_pass:
            status, color = "FAIL", _C["fail"]
            note = ("Linearity out of spec (zero-tolerance)"
                    + (" · stored verdict: PASS — reprocess" if stored_disagrees else ""))
        elif stored_disagrees:
            status, color, note = ("PASS*", _C["warning"],
                                   "In spec under the stored adjustment · stored "
                                   "verdict: FAIL — reprocess")
        elif sig_pass:
            status, color, note = "PASS", _C["pass"], "Linearity in spec · no sigma watch"
        else:
            status, color, note = "PASS (WATCH)", _C["warning"], "Linearity in spec · sigma drift-watch"
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
