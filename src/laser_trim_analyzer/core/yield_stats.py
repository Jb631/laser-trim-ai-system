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
