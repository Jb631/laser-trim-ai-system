"""Windowed production-yield aggregation for the Dashboard.

Pure helpers: given a DatabaseManager and an ORM model class that has
`overall_status` + `file_date` (analysis_results or final_test_results), return
status-bucket counts, a pass-rate, and a per-day pass-rate trend. No Tk.
"""
from datetime import datetime
from typing import Optional

# overall_status (StatusType) name -> yield bucket.
_BUCKET = {
    "PASS": "passed", "WARNING": "warnings", "FAIL": "failed",
    "ERROR": "errors", "PROCESSING_FAILED": "errors", "UNTRIMMED": "untrimmed",
}


def _bucket(status) -> str:
    name = getattr(status, "name", None) or str(status)
    return _BUCKET.get(str(name).upper(), "errors")


def compute_yield(db, model_cls, cutoff: Optional[datetime]) -> dict:
    """Yield over `model_cls` rows with file_date >= cutoff (cutoff None = all time).

    pass_rate = passed / (passed + warnings + failed) * 100, i.e. a WARNING counts
    as not-a-clean-pass and ERROR/UNTRIMMED are excluded from the denominator.
    Returns counts, gradeable, total, pass_rate (None if no gradeable rows), and
    trend = [{"date": "YYYY-MM-DD", "pass_rate": float}] ascending by day.
    """
    counts = {"passed": 0, "warnings": 0, "failed": 0, "errors": 0, "untrimmed": 0}
    per_day: dict = {}
    with db.session() as s:
        q = s.query(model_cls.file_date, model_cls.overall_status)
        if cutoff is not None:
            q = q.filter(model_cls.file_date >= cutoff)
        rows = q.all()
    for file_date, status in rows:
        b = _bucket(status)
        counts[b] += 1
        if b in ("passed", "warnings", "failed") and file_date is not None:
            slot = per_day.setdefault(file_date.strftime("%Y-%m-%d"), [0, 0])
            slot[0] += 1
            if b == "passed":
                slot[1] += 1
    gradeable = counts["passed"] + counts["warnings"] + counts["failed"]
    pass_rate = (counts["passed"] / gradeable * 100.0) if gradeable else None
    trend = [{"date": d, "pass_rate": (p / t * 100.0 if t else 0.0)}
             for d, (t, p) in sorted(per_day.items())]
    return {**counts, "gradeable": gradeable, "total": sum(counts.values()),
            "pass_rate": pass_rate, "trend": trend}


def worst_models_by_yield(db, cutoff: Optional[datetime], min_units: int = 5, limit: int = 10):
    """Rank models by trim yield (worst first) over the window.

    'units' = gradeable trim records (pass+warning+fail) for the model; only models
    with units >= min_units are ranked (so a 1-unit 0% model can't dominate). Each
    row: {model, units, trim_rate, ft_rate}. trim_rate/ft_rate are % or None.
    Returns (rows[:limit], total_qualifying) so the caller can disclose the cap.
    """
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, FinalTestResult as DBFT)

    def _by_model(model_cls):
        acc: dict = {}
        with db.session() as s:
            q = s.query(model_cls.model, model_cls.overall_status)
            if cutoff is not None:
                q = q.filter(model_cls.file_date >= cutoff)
            rows = q.all()
        for model, status in rows:
            if not model:
                continue
            b = _bucket(status)
            if b not in ("passed", "warnings", "failed"):
                continue
            slot = acc.setdefault(model, [0, 0])   # [gradeable, passed]
            slot[0] += 1
            if b == "passed":
                slot[1] += 1
        return acc

    trim = _by_model(DBAR)
    ft = _by_model(DBFT)

    rows = []
    for model, (gradeable, passed) in trim.items():
        if gradeable < min_units:
            continue
        ftv = ft.get(model)
        rows.append({
            "model": model,
            "units": gradeable,
            "trim_rate": (passed / gradeable * 100.0) if gradeable else None,
            "ft_rate": (ftv[1] / ftv[0] * 100.0) if ftv and ftv[0] else None,
        })
    rows.sort(key=lambda r: (r["trim_rate"] is None, r["trim_rate"] if r["trim_rate"] is not None else 0.0))
    return rows[:limit], len(rows)
