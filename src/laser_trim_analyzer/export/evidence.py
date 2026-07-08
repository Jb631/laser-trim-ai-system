"""Spec 3c — Evidence export. The daily 'what I hand engineers' payoff (foundations Q8)."""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS, metric_label

# "Recent" window length (days of DATA, anchored to the model's latest file_date).
RECENT_DAYS = 30


# Shared with the drift engine (single source): values beyond this many
# baseline-σ are suspect data, not process signal. Evidence displays EXCLUDE
# and disclose them; the detector winsorizes them (drift_training).
from laser_trim_analyzer.ml.drift_types import SUSPECT_SIGMA_GATE  # noqa: E402,F401


def compute_recent_means(db, model: str, recent_days: int = RECENT_DAYS,
                         with_meta: bool = False):
    """Outlier-gated mean of each watched metric over the model's most recent
    `recent_days` of DATA.

    Anchored to the model's latest file_date, NOT wall-clock now: this is batch-loaded
    historical data (a loaded lot can be weeks old), so a now-anchored window leaves
    'recent' empty for nearly every model. The drift detector never persists a recent
    mean, so every evidence surface (UI table, copy-summary, Excel pack) must derive it
    here to populate the baseline-vs-recent comparison.

    Values beyond SUSPECT_SIGMA_GATE baseline-σ are excluded (only when a trained,
    non-degenerate baseline exists to judge against). Returns metric -> float|None;
    with_meta=True returns (means, meta) where meta[metric] =
    {"n": kept, "excluded": dropped} so every surface can disclose the gating.
    """
    from sqlalchemy import func
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SmoothnessResult as DBSR,
        ModelMetricState)
    from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
    from laser_trim_analyzer.ml.multi_metric_drift_detector import is_degenerate_baseline

    out: dict = {}
    meta: dict = {}
    with db.session() as s:
        anchor = (s.query(func.max(DBAR.file_date))
                  .filter(DBAR.model == model).scalar())
        cutoff = (anchor - timedelta(days=recent_days)) if anchor is not None else None
        baselines = {
            r.metric: (r.baseline_mean, r.baseline_std)
            for r in s.query(ModelMetricState).filter(
                ModelMetricState.model == model,
                ModelMetricState.is_trained == True).all()  # noqa: E712
        }
        for metric in WATCHED_METRICS:
            meta[metric] = {"n": 0, "excluded": 0}
            if cutoff is None:
                out[metric] = None
                continue
            if metric == "max_smoothness_value":
                rows = (s.query(DBSR.max_smoothness_value)
                        .filter(DBSR.model == model, DBSR.max_smoothness_value.isnot(None),
                                DBSR.file_date >= cutoff).all())
            elif metric in TRACK_METRIC_COLUMNS:
                col = TRACK_METRIC_COLUMNS[metric]
                rows = (s.query(col).join(DBAR, DBTR.analysis_id == DBAR.id)
                        .filter(DBAR.model == model, col.isnot(None),
                                DBAR.file_date >= cutoff).all())
            else:
                out[metric] = None
                continue
            values = [float(r[0]) for r in rows if r[0] is not None]
            bm, bs = baselines.get(metric, (None, None))
            if (bm is not None and bs is not None
                    and not is_degenerate_baseline(bm, bs)):
                kept = [v for v in values if abs(v - bm) <= SUSPECT_SIGMA_GATE * bs]
                meta[metric]["excluded"] = len(values) - len(kept)
            else:
                kept = values  # no trustworthy yardstick — no gating
            meta[metric]["n"] = len(kept)
            out[metric] = (sum(kept) / len(kept)) if kept else None
    return (out, meta) if with_meta else out


def _sigma_shift(ms, recent_val) -> Optional[float]:
    """Honest (recent − baseline) / baseline_std — the SAME number the Triage
    cards and Model-page pills show. MetricStatus.magnitude is alert-type
    scaled (step: √N·shift/σ; slow drift: cusum/h) and is NOT comparable
    across metrics; labeling it 'Δσ' made exports contradict the screen."""
    if recent_val is None or ms.baseline_std is None or ms.baseline_std <= 0:
        return None
    return (recent_val - ms.baseline_mean) / ms.baseline_std


def build_summary_text(model: str, status, recent_means: Optional[dict] = None,
                       recent_meta: Optional[dict] = None) -> str:
    """Paste-ready text: model + per-metric baseline-vs-recent + alert. Q8 traceable.

    `recent_means` (metric -> float|None) supplies the data-derived recent values; the
    hydrated detector's own recent_mean is always None, so without this the summary
    reads 'recent n/a' for every metric.
    """
    recent_means = recent_means or {}
    lines = [f"Drift summary — model {model}",
             f"Overall: {status.overall_tier.name.replace('_', ' ').title()}"
             + (f" (worst: {metric_label(status.worst_metric)})" if status.worst_metric else ""), ""]
    for m in WATCHED_METRICS:
        ms = status.per_metric.get(m)
        if ms is None:
            continue
        recent_val = recent_means.get(m) if recent_means.get(m) is not None else ms.recent_mean
        recent = f"{recent_val:.4g}" if recent_val is not None else "n/a"
        tier = ms.tier.name.replace("_", " ").title()
        shift = _sigma_shift(ms, recent_val)
        shift_txt = f"shift {shift:+.2f}σ" if shift is not None else "shift n/a"
        excl = (recent_meta or {}).get(m, {}).get("excluded") if recent_meta else None
        excl_txt = f" ({excl} suspect value{'s' if excl != 1 else ''} excluded)" if excl else ""
        lines.append(f"- {metric_label(m)}: baseline {ms.baseline_mean:.4g} ± {ms.baseline_std:.4g}, "
                     f"recent {recent}{excl_txt}, {shift_txt} [{tier}]")
    return "\n".join(lines)


def export_evidence_pack(db, model: str, out_path, window_days: Optional[int] = None) -> Path:
    """Write a traceable evidence workbook for `model`.

    Sheets:
      * Drift evidence — per-metric baseline vs recent, honest σ-shift, tier
      * Unit history  — one row per unit/track over the FULL record (default;
        pass window_days to restrict), every watched metric + pass flags.
        This is the 'export the model to Excel and read its whole history'
        workflow — judging whether the process is moving for better or worse.
      * Monthly summary — units, pass rate, and metric means per month
    """
    import pandas as pd
    from sqlalchemy import func, case
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType)
    from laser_trim_analyzer.ml.manager import get_model_drift_status

    status = get_model_drift_status(db, model)
    recent_means, recent_meta = compute_recent_means(db, model, with_meta=True)
    metric_rows = []
    for m in WATCHED_METRICS:
        ms = status.per_metric.get(m)
        if ms is None:
            continue
        recent_val = recent_means.get(m) if recent_means.get(m) is not None else ms.recent_mean
        mm = recent_meta.get(m, {})
        metric_rows.append({"Metric": metric_label(m), "Tier": ms.tier.name,
                            "Alert": ms.alert_type.value if ms.alert_type else "",
                            "Baseline mean": ms.baseline_mean, "Baseline std": ms.baseline_std,
                            "Recent mean": recent_val,
                            "Recent n": mm.get("n"),
                            # Traceability: how many recent values were gated
                            # out as suspect (scale anomalies) before the mean.
                            "Suspect excluded": mm.get("excluded"),
                            # Matches the UI (Triage cards / pills). The old
                            # 'Delta_sigma' column exported alert-scaled
                            # magnitude, which contradicted the screen.
                            "Shift (σ)": _sigma_shift(ms, recent_val),
                            "Alert magnitude (scaled)": ms.magnitude})

    cutoff = datetime.now() - timedelta(days=window_days) if window_days else None
    with db.session() as s:
        # Final-test outcome per trim unit (the other half of the trim-vs-FT
        # workflow — "did this unit pass at final test"; user request 2026-07-08).
        from laser_trim_analyzer.database.models import FinalTestResult as DBFT
        ft_by_trim: dict = {}
        for ft in (s.query(DBFT.linked_trim_id, DBFT.overall_status,
                           DBFT.file_date, DBFT.match_confidence)
                   .filter(DBFT.model == model, DBFT.linked_trim_id.isnot(None)).all()):
            # Keep the newest FT per trim record.
            prev = ft_by_trim.get(ft[0])
            if prev is None or (ft[2] and prev[1] and ft[2] > prev[1]):
                ft_by_trim[ft[0]] = (getattr(ft[1], "value", str(ft[1])), ft[2], ft[3])

        q = (s.query(DBAR.id, DBAR.serial, DBAR.file_date, DBAR.overall_status, DBAR.system,
                     DBTR.track_id, DBTR.sigma_gradient, DBTR.untrimmed_sigma_gradient,
                     DBTR.final_linearity_error_shifted, DBTR.untrimmed_error_max,
                     DBTR.untrimmed_resistance, DBTR.resistance_change_percent,
                     DBTR.measured_electrical_angle, DBTR.trim_pass_count,
                     DBTR.composite_trim_risk_score, DBTR.sigma_pass, DBTR.linearity_pass)
             .join(DBTR, DBTR.analysis_id == DBAR.id).filter(DBAR.model == model))
        if cutoff is not None:
            q = q.filter(DBAR.file_date >= cutoff)
        unit_rows = []
        for r in q.order_by(DBAR.file_date.desc()).all():
            ft = ft_by_trim.get(r[0])
            unit_rows.append({
                "Serial": r[1], "Date": r[2],
                "Status": getattr(r[3], "value", str(r[3])),
                "System": getattr(r[4], "value", str(r[4])),
                "Track": r[5],
                "Sigma gradient": r[6], "Untrimmed sigma": r[7],
                "Linearity error": r[8], "Untrimmed error max": r[9],
                "Untrimmed resistance": r[10], "Resistance change %": r[11],
                "Electrical angle": r[12], "Trim passes": r[13],
                "Composite risk": r[14],
                "Sigma pass": r[15], "Linearity pass": r[16],
                "FT result": ft[0] if ft else None,
                "FT date": ft[1] if ft else None,
                "FT match %": (round(ft[2] * 100) if ft and ft[2] is not None
                               else None)})

        # Monthly rollup: UNIT-level counts/pass rate (overall_status is the
        # per-unit verdict; UNTRIMMED test-sweeps excluded from the rate), plus
        # track-metric means for the better-or-worse read.
        month = func.strftime("%Y-%m", DBAR.file_date)
        unit_q = (s.query(
                    month.label("month"),
                    func.count(DBAR.id).label("units"),
                    func.sum(case((DBAR.overall_status == StatusType.PASS, 1), else_=0)).label("passed"),
                    func.sum(case((DBAR.overall_status == StatusType.FAIL, 1), else_=0)).label("failed"),
                    func.sum(case((DBAR.overall_status == StatusType.WARNING, 1), else_=0)).label("warned"),
                  )
                  .filter(DBAR.model == model,
                          DBAR.overall_status != StatusType.UNTRIMMED)
                  .group_by(month).order_by(month).all())
        mean_q = (s.query(
                    month.label("month"),
                    func.avg(DBTR.sigma_gradient), func.avg(DBTR.untrimmed_sigma_gradient),
                    func.avg(DBTR.final_linearity_error_shifted), func.avg(DBTR.untrimmed_error_max),
                    func.avg(DBTR.resistance_change_percent), func.avg(DBTR.measured_electrical_angle),
                  )
                  .join(DBTR, DBTR.analysis_id == DBAR.id)
                  .filter(DBAR.model == model)
                  .group_by(month).all())
        means_by_month = {r[0]: r[1:] for r in mean_q}
        monthly_rows = []
        for r in unit_q:
            mm = means_by_month.get(r.month, (None,) * 6)
            units = r.units or 0
            accepted = (r.passed or 0) + (r.warned or 0)
            monthly_rows.append({
                "Month": r.month, "Units": units,
                "Passed": r.passed or 0, "Watch (sigma)": r.warned or 0,
                "Failed": r.failed or 0,
                # Customer basis: linearity is the zero-tolerance requirement;
                # Watch units passed it (sigma = internal drift flag only).
                "Linearity yield %": (accepted / units * 100) if units else None,
                "Clean pass %": ((r.passed or 0) / units * 100) if units else None,
                "Mean sigma gradient": mm[0], "Mean untrimmed sigma": mm[1],
                "Mean linearity error": mm[2], "Mean untrimmed error max": mm[3],
                "Mean resistance change %": mm[4], "Mean electrical angle": mm[5],
            })

    out_path = Path(out_path)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        pd.DataFrame(metric_rows).to_excel(xl, sheet_name="Drift evidence", index=False)
        pd.DataFrame(unit_rows).to_excel(xl, sheet_name="Unit history", index=False)
        pd.DataFrame(monthly_rows).to_excel(xl, sheet_name="Monthly summary", index=False)
    return out_path
