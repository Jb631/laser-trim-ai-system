"""Spec 3c — Evidence export. The daily 'what I hand engineers' payoff (foundations Q8)."""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS, metric_label

# "Recent" window length (days of DATA, anchored to the model's latest file_date).
RECENT_DAYS = 30


def compute_recent_means(db, model: str, recent_days: int = RECENT_DAYS) -> dict:
    """Mean of each watched metric over the model's most recent `recent_days` of DATA.

    Anchored to the model's latest file_date, NOT wall-clock now: this is batch-loaded
    historical data (a loaded lot can be weeks old), so a now-anchored window leaves
    'recent' empty for nearly every model. The drift detector never persists a recent
    mean, so every evidence surface (UI table, copy-summary, Excel pack) must derive it
    here to populate the baseline-vs-recent comparison. Metric -> float | None.
    """
    from sqlalchemy import func
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SmoothnessResult as DBSR)
    from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS

    out = {}
    with db.session() as s:
        anchor = (s.query(func.max(DBAR.file_date))
                  .filter(DBAR.model == model).scalar())
        cutoff = (anchor - timedelta(days=recent_days)) if anchor is not None else None
        for metric in WATCHED_METRICS:
            if cutoff is None:
                out[metric] = None
                continue
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


def build_summary_text(model: str, status, recent_means: Optional[dict] = None) -> str:
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
        lines.append(f"- {metric_label(m)}: baseline {ms.baseline_mean:.4g} ± {ms.baseline_std:.4g}, "
                     f"recent {recent}, Δ {ms.magnitude:+.2f}σ [{tier}]")
    return "\n".join(lines)


def export_evidence_pack(db, model: str, out_path, window_days: Optional[int] = 365) -> Path:
    """Write a traceable evidence workbook for `model`. Sheets: Metrics, Units."""
    import pandas as pd
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR)
    from laser_trim_analyzer.ml.manager import get_model_drift_status

    status = get_model_drift_status(db, model)
    recent_means = compute_recent_means(db, model)
    metric_rows = []
    for m in WATCHED_METRICS:
        ms = status.per_metric.get(m)
        if ms is None:
            continue
        recent_val = recent_means.get(m) if recent_means.get(m) is not None else ms.recent_mean
        metric_rows.append({"Metric": metric_label(m), "Tier": ms.tier.name,
                            "Alert": ms.alert_type.value if ms.alert_type else "",
                            "Baseline mean": ms.baseline_mean, "Baseline std": ms.baseline_std,
                            "Recent mean": recent_val, "Delta_sigma": ms.magnitude})

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
