"""Windowed production-yield aggregation for the Dashboard.

Pure helpers: given a DatabaseManager and an ORM model class that has
`overall_status` + `file_date` (analysis_results or final_test_results), return
status-bucket counts, a pass-rate, and a per-day pass-rate trend. No Tk.
"""
import re
from datetime import datetime, timedelta
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

    TWO rates, two meanings (domain rule, 2026-07-07):
      * linearity_yield — the CUSTOMER basis: (passed + warnings) / gradeable.
        Linearity is the zero-tolerance customer requirement; WARNING units
        passed linearity (sigma is only an internal drift-watch flag), so they
        are accepted product. THIS is the headline yield.
      * pass_rate — the clean-pass rate: passed / gradeable. Internal process-
        health view (how many units also raised no sigma watch).
    ERROR/UNTRIMMED are excluded from both denominators. trend carries both
    rates per day; "rate" mirrors linearity_yield for the headline sparkline.
    """
    counts = {"passed": 0, "warnings": 0, "failed": 0, "errors": 0, "untrimmed": 0}
    per_day: dict = {}
    # Future-dated records are DATA ERRORS (a mistyped filename date put one
    # FT record five months ahead, stretching the dashboard trend and setting
    # its 'last day' value). Exclude beyond a 1-day clock-skew allowance and
    # DISCLOSE the count so the bad file gets fixed, not hidden.
    horizon = datetime.now() + timedelta(days=1)
    future_dated = 0
    with db.session() as s:
        q = s.query(model_cls.file_date, model_cls.overall_status)
        if cutoff is not None:
            q = q.filter(model_cls.file_date >= cutoff)
        rows = q.all()
    for file_date, status in rows:
        if file_date is not None and file_date > horizon:
            future_dated += 1
            continue
        b = _bucket(status)
        counts[b] += 1
        if b in ("passed", "warnings", "failed") and file_date is not None:
            slot = per_day.setdefault(file_date.strftime("%Y-%m-%d"), [0, 0, 0])
            slot[0] += 1                      # gradeable
            if b == "passed":
                slot[1] += 1                  # clean pass
            if b in ("passed", "warnings"):
                slot[2] += 1                  # linearity-accepted
    gradeable = counts["passed"] + counts["warnings"] + counts["failed"]
    pass_rate = (counts["passed"] / gradeable * 100.0) if gradeable else None
    accepted = counts["passed"] + counts["warnings"]
    linearity_yield = (accepted / gradeable * 100.0) if gradeable else None
    trend = [{"date": d,
              "pass_rate": (p / t * 100.0 if t else 0.0),
              "linearity_yield": (a / t * 100.0 if t else 0.0),
              "rate": (a / t * 100.0 if t else 0.0)}
             for d, (t, p, a) in sorted(per_day.items())]
    return {**counts, "gradeable": gradeable, "total": sum(counts.values()),
            "pass_rate": pass_rate, "linearity_yield": linearity_yield,
            "trend": trend, "future_dated": future_dated}


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
            slot = acc.setdefault(model, [0, 0])   # [gradeable, accepted]
            slot[0] += 1
            if b in ("passed", "warnings"):
                # Customer basis: WARNING = linearity-accepted, sigma-watch only.
                slot[1] += 1
        return acc

    trim = _by_model(DBAR)
    ft = _by_model(DBFT)

    rows = []
    for model, (gradeable, accepted) in trim.items():
        if gradeable < min_units:
            continue
        ftv = ft.get(model)
        rows.append({
            "model": model,
            "units": gradeable,
            "trim_rate": (accepted / gradeable * 100.0) if gradeable else None,
            "ft_rate": (ftv[1] / ftv[0] * 100.0) if ftv and ftv[0] else None,
        })
    rows.sort(key=lambda r: (r["trim_rate"] is None, r["trim_rate"] if r["trim_rate"] is not None else 0.0))
    return rows[:limit], len(rows)

_SECTION_RE = re.compile(r"^(.*?)_?te?st[ _]?data", re.IGNORECASE)
_FILETIME_RE = re.compile(
    r"(\d{1,2})-(\d{1,2})-(\d{4})_(\d{1,2})-(\d{2})\s?(AM|PM)", re.IGNORECASE)


def _section_of(filename: str, serial) -> str:
    """Section identity within a unit. Real conventions (verified 2026-07-13):
    8895 encodes sections in the SERIAL (1P/1R); 6607 encodes them in the
    FILENAME (_TA_/_TB_ with one serial); 8895 also uses _primary_/_REDUNDANT_
    tokens with one serial. The prefix before 'TEST DATA' captures all three;
    retrims differ only by the trailing timestamp, so they share it."""
    m = _SECTION_RE.match(filename or "")
    prefix = m.group(1).strip("_ ").lower() if m else ""
    return f"{serial}|{prefix}"


def _attempt_time(filename: str, file_date, row_id):
    """Ordering key for attempts. file_date is often day-only (00:00) while
    the filename carries '5-5-2026_10-07 AM' — parse it so first/last per
    section reflect the actual trim order, not processing order."""
    m = _FILETIME_RE.search(filename or "")
    if m:
        mo, d, y, hh, mm, ap = m.groups()
        h = int(hh) % 12 + (12 if ap.upper() == "PM" else 0)
        try:
            return (datetime(int(y), int(mo), int(d), h, int(mm)), row_id)
        except ValueError:
            pass
    return (file_date or datetime.min, row_id)


def compute_unit_yield(db, cutoff: Optional[datetime], model: Optional[str] = None) -> dict:
    """UNIT-basis yield: first-pass, final, and rework — the numbers the
    attempt-basis yield hides (QA audit, 2026-07-13).

    Identity (the tested unit-id rule): unit_id = model + shop number + DAY.
    Within a unit, rows group into SECTIONS via serial + filename section
    marker (see _section_of): 1P/1R, _TA_/_TB_, primary/REDUNDANT are
    parallel elements of one unit; repeats within a section are RETRIMS.
    Zero-tolerance rollup, same as tracks: a unit is accepted only when
    EVERY section is accepted.

    * first_pass_yield: all sections' FIRST attempts linearity-accepted.
    * final_yield:      all sections' LAST attempts linearity-accepted.
    * attempts_per_section: 1.00 = first-time-right (immune to dual-element
      inflation).
    * rework_units: units where ANY section needed >1 attempt.
    Cohort = units whose first attempt falls in the window. Attempt order
    inside a day comes from the filename timestamp where present.
    Yields in %, None when no gradeable units.
    """
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR

    accept = {"PASS", "WARNING"}     # linearity-accepted (customer basis)
    gradeable_status = {"PASS", "WARNING", "FAIL"}

    with db.session() as s:
        q = s.query(DBAR.unit_id, DBAR.serial, DBAR.file_date,
                    DBAR.overall_status, DBAR.id, DBAR.filename)
        if model is not None:
            q = q.filter(DBAR.model == model)
        rows = q.all()

    no_unit = 0
    by_unit: dict = {}
    for uid, serial, fdate, status, rid, fname in rows:
        if uid is None:
            no_unit += 1
            continue
        name = getattr(status, "name", None) or str(status)
        key = _section_of(fname, serial)
        by_unit.setdefault(uid, {}).setdefault(key, []).append(
            (_attempt_time(fname, fdate, rid), str(name).upper()))

    units = gradeable = first_ok = final_ok = rework = 0
    attempts_total = sections_total = 0
    for uid, sections in by_unit.items():
        first_seen = min(a[0][0] for grp in sections.values() for a in grp)
        # Cohort membership: unit's FIRST attempt inside the window.
        if cutoff is not None and first_seen < cutoff:
            continue
        units += 1
        sec_first, sec_last, sec_attempts = [], [], 0
        for grp in sections.values():
            grp.sort(key=lambda t: t[0])
            graded = [a for a in grp if a[1] in gradeable_status]
            if not graded:
                continue             # section has only ERROR/UNTRIMMED rows
            sec_attempts += len(graded)
            sec_first.append(graded[0][1])
            sec_last.append(graded[-1][1])
        if not sec_first:
            continue                 # nothing gradeable anywhere in the unit
        gradeable += 1
        attempts_total += sec_attempts
        sections_total += len(sec_first)
        if sec_attempts > len(sec_first):
            rework += 1              # some section needed more than one trim
        if all(st in accept for st in sec_first):
            first_ok += 1
        if all(st in accept for st in sec_last):
            final_ok += 1

    return {
        "units": units,
        "gradeable_units": gradeable,
        "first_pass_yield": (first_ok / gradeable * 100.0) if gradeable else None,
        "final_yield": (final_ok / gradeable * 100.0) if gradeable else None,
        "attempts_per_section": (attempts_total / sections_total) if sections_total else None,
        "rework_units": rework,
        "no_unit_id_rows": no_unit,
    }

def compute_unit_yield_monthly(db, model: str) -> dict:
    """Per-month unit cohorts for the evidence pack: month of a unit's FIRST
    attempt -> {units, first_pass_yield, final_yield, attempts_per_section,
    rework_units}. Same section/ordering rules as compute_unit_yield."""
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR

    accept = {"PASS", "WARNING"}
    gradeable_status = {"PASS", "WARNING", "FAIL"}
    with db.session() as s:
        rows = (s.query(DBAR.unit_id, DBAR.serial, DBAR.file_date,
                        DBAR.overall_status, DBAR.id, DBAR.filename)
                .filter(DBAR.model == model).all())
    by_unit: dict = {}
    for uid, serial, fdate, status, rid, fname in rows:
        if uid is None:
            continue
        name = getattr(status, "name", None) or str(status)
        key = _section_of(fname, serial)
        by_unit.setdefault(uid, {}).setdefault(key, []).append(
            (_attempt_time(fname, fdate, rid), str(name).upper()))
    out: dict = {}
    for uid, sections in by_unit.items():
        first_seen = min(a[0][0] for grp in sections.values() for a in grp)
        mo = first_seen.strftime("%Y-%m")
        sec_first, sec_last, sec_attempts = [], [], 0
        for grp in sections.values():
            grp.sort(key=lambda t: t[0])
            graded = [a for a in grp if a[1] in gradeable_status]
            if not graded:
                continue
            sec_attempts += len(graded)
            sec_first.append(graded[0][1])
            sec_last.append(graded[-1][1])
        if not sec_first:
            continue
        b = out.setdefault(mo, {"units": 0, "first_ok": 0, "final_ok": 0,
                                "attempts": 0, "sections": 0, "rework": 0})
        b["units"] += 1
        b["attempts"] += sec_attempts
        b["sections"] += len(sec_first)
        if sec_attempts > len(sec_first):
            b["rework"] += 1
        if all(st in accept for st in sec_first):
            b["first_ok"] += 1
        if all(st in accept for st in sec_last):
            b["final_ok"] += 1
    return {mo: {"units": b["units"],
                 "first_pass_yield": b["first_ok"] / b["units"] * 100.0,
                 "final_yield": b["final_ok"] / b["units"] * 100.0,
                 "attempts_per_section": (b["attempts"] / b["sections"]) if b["sections"] else None,
                 "rework_units": b["rework"]}
            for mo, b in out.items()}


def compute_trim_necessity(db, model: str,
                           cutoff: Optional[datetime] = None) -> Optional[dict]:
    """Did these units NEED the laser for linearity at all?

    James's question (2026-07-14): "can i tell if a model is passing
    linearity on its test run and we are only trimming the unit to bring
    the resistance into specification?" — wasted laser time when the
    as-fired resistance target is set too low.

    A unit counts as already-passing when EVERY track's raw pre-trim sweep
    worst point is inside the linearity spec (untrimmed_error_max <=
    linearity_spec — conservative: no offset correction even applied, so
    these are definite cases). Only units that were actually TRIMMED
    (any track with trim_pass_count >= 1) enter the denominator.

    Work-DB baseline (2yr, measured 2026-07-14): 12.6% overall; standouts
    8877-4 84%, 8167 84%, 8436 65%, 8755 59%, 8863 50%. Pre-passing units
    still moved resistance +11% on average — they were trimmed UP to a
    resistance target, confirming the hypothesis.

    Returns {trimmed_units, prepass_units, prepass_share (%),
             avg_resistance_change_prepass (%|None)} or None on no data.
    """
    from sqlalchemy import text

    where_date = "AND a.file_date >= :cutoff" if cutoff is not None else ""
    sql = text(f"""
        WITH unit AS (
          SELECT a.id,
                 MIN(CASE WHEN t.untrimmed_error_max <= t.linearity_spec
                          THEN 1 ELSE 0 END) AS prepass,
                 MAX(t.trim_pass_count) AS passes,
                 AVG(t.resistance_change_percent) AS rchg
          FROM analysis_results a
          JOIN track_results t ON t.analysis_id = a.id
          WHERE a.model = :model
            AND a.overall_status IN ('PASS','WARNING','FAIL')
            AND t.untrimmed_error_max IS NOT NULL
            AND t.linearity_spec IS NOT NULL
            {where_date}
          GROUP BY a.id)
        SELECT COUNT(*),
               COALESCE(SUM(prepass), 0),
               AVG(CASE WHEN prepass = 1 THEN rchg END)
        FROM unit WHERE passes >= 1""")
    params = {"model": model}
    if cutoff is not None:
        params["cutoff"] = cutoff
    with db.session() as s:
        n, pre, rchg = s.execute(sql, params).fetchone()
    if not n:
        return None
    return {"trimmed_units": int(n),
            "prepass_units": int(pre),
            "prepass_share": 100.0 * pre / n,
            "avg_resistance_change_prepass": (float(rchg) if rchg is not None else None)}

