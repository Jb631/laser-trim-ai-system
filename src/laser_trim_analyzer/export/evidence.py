"""Spec 3c — Evidence export. The daily 'what I hand engineers' payoff (foundations Q8)."""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS, metric_label


def build_summary_text(model: str, status) -> str:
    """Paste-ready text: model + per-metric baseline-vs-recent + alert. Q8 traceable."""
    lines = [f"Drift summary — model {model}",
             f"Overall: {status.overall_tier.name.replace('_', ' ').title()}"
             + (f" (worst: {metric_label(status.worst_metric)})" if status.worst_metric else ""), ""]
    for m in WATCHED_METRICS:
        ms = status.per_metric.get(m)
        if ms is None:
            continue
        recent = f"{ms.recent_mean:.4g}" if ms.recent_mean is not None else "n/a"
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
    metric_rows = []
    for m in WATCHED_METRICS:
        ms = status.per_metric.get(m)
        if ms is None:
            continue
        metric_rows.append({"Metric": metric_label(m), "Tier": ms.tier.name,
                            "Alert": ms.alert_type.value if ms.alert_type else "",
                            "Baseline mean": ms.baseline_mean, "Baseline std": ms.baseline_std,
                            "Recent mean": ms.recent_mean, "Delta_sigma": ms.magnitude})

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
