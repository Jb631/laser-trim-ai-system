"""Run every analyzer for a model, rank what they found, and cache it.

Runs after an ingest, on the ingest worker thread (never the Tk thread), for
the models that ingest touched. The screens only ever read the cache.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import text

from .analyzers import (cut_setting, ink_target, limit_tables, pass_burden,
                        recipe_change, trim_effort)
from .data import load_model_tracks, yardstick_fidelity
from .model import Finding, rank

logger = logging.getLogger(__name__)


def _laser_label(system: str) -> str:
    # The shop's names, never the code's letters (A is laser TWO). Imported late
    # so this package stays importable in a bare test of the analyzers.
    from laser_trim_analyzer.core.models import laser_label
    return laser_label(system)


def _fleet_latest(db) -> Optional[datetime]:
    """The newest trim file in the database -- what 'now' means to a finding.

    Two kinds of row must not win this: a record whose PROCESSING failed carries
    `file_date = datetime.now()` from the moment the analyser gave up
    (`processor._create_minimal_metadata`), not a measurement -- counting it would mean
    today's crash always looks like "the latest data". And a mistyped filename date can
    put a file months in the future, which would make every model's real, current data
    look "stale" by comparison. Neither exists in the work database today (checked
    2026-09-23), so this changes nothing yet -- it is here for when one turns up.
    """
    from .data import _date
    from laser_trim_analyzer.core.model_stats import _FAILED_PROCESSING
    params: Dict[str, Any] = {f"failed{i}": name for i, name in enumerate(_FAILED_PROCESSING)}
    placeholders = ", ".join(f":{k}" for k in params)
    # Bound as a string, not a raw datetime: text() does not apply SQLAlchemy's own DATETIME
    # bind_processor (that only fires for ORM-typed columns), so a bare datetime falls back to
    # sqlite3's own adapter registry -- deprecated since 3.12. file_date is stored in exactly
    # this format ("%Y-%m-%d %H:%M:%S.%f", fixed-width and zero-padded, SQLAlchemy's sqlite
    # DATETIME default), so a plain string comparison sorts identically to a chronological one.
    cutoff = datetime.now() + timedelta(days=1)
    params["cutoff"] = f"{cutoff:%Y-%m-%d %H:%M:%S.%f}"
    with db.session() as s:
        v = s.execute(text(
            "SELECT MAX(file_date) FROM analysis_results "
            "WHERE system IN ('A','B','C') "
            f"AND overall_status NOT IN ({placeholders}) "
            "AND file_date <= :cutoff"), params).scalar()
    return _date(v)


def compute_for_model(db, model: str,
                      fleet_latest: Optional[datetime] = None) -> Tuple[Dict[str, Any], List[Finding]]:
    tracks = load_model_tracks(db, model)
    # Every documented key exists from the first line. None = not computed; an empty list =
    # computed, and there is nothing. An analyzer that CRASHED is named in facts["errors"]: on
    # these screens silence is itself a result, so a failure must never be able to look like one.
    facts: Dict[str, Any] = {"model": model, "tracks": len(tracks), "annual_volume": 0, "latest": None,
                             "yardstick": None, "recipe_history": None, "trim_effort": None,
                             "limit_tables": None, "cut_setting": None, "pass_burden": None, "errors": {}}
    if not tracks:
        return facts, []
    if fleet_latest is None:
        fleet_latest = _fleet_latest(db)
    latest = max(t.file_date for t in tracks)
    volume = sum(1 for t in tracks if t.file_date >= latest - timedelta(days=365))
    facts["annual_volume"] = volume
    facts["latest"] = latest.date().isoformat()
    fidelity = yardstick_fidelity(tracks)
    facts["yardstick"] = fidelity
    findings: List[Finding] = []

    def failed(name: str, exc: Exception) -> None:
        logger.exception("findings: %s failed for %s", name, model)
        facts["errors"][name] = f"{type(exc).__name__}: {exc}"

    # Each analyzer is independently guarded: one failing must not blank the rest.
    try:
        history, changes = recipe_change.analyze(model, tracks, _laser_label)  # stored verdicts only
        facts["recipe_history"] = history
        findings += changes
    except Exception as exc:
        failed("recipe_change", exc)
    try:
        findings += ink_target.analyze(model, tracks, _laser_label)         # stored verdicts only
    except Exception as exc:
        failed("ink_target", exc)
    try:
        table_history, table_findings = limit_tables.analyze(model, tracks, _laser_label)   # stored limits only
        facts["limit_tables"] = table_history
        findings += table_findings
    except Exception as exc:
        failed("limit_tables", exc)
    try:
        cut_facts, cut_findings = cut_setting.analyze(model, tracks, _laser_label,
                                                       now=fleet_latest)  # stored verdicts only
        facts["cut_setting"] = cut_facts
        findings += cut_findings
    except Exception as exc:
        failed("cut_setting", exc)
    try:
        burden_facts, burden_findings = pass_burden.analyze(model, tracks, _laser_label)  # pass counts only
        facts["pass_burden"] = burden_facts
        findings += burden_findings
    except Exception as exc:
        failed("pass_burden", exc)
    if fidelity["faithful"]:
        try:
            effort_facts, effort_findings = trim_effort.analyze(model, tracks, _laser_label)
            facts["trim_effort"] = effort_facts
            findings += effort_findings
        except Exception as exc:
            failed("trim_effort", exc)
    # else: grading intermediate sweeps is only honest where the yardstick reproduces the app's
    # own verdict. trim_effort stays None and facts["yardstick"] says why; nothing is guessed.
    for f in findings:
        f.annual_volume = volume
    return facts, rank(findings)


def refresh_findings(db, models: Optional[Iterable[str]] = None,
                     report: Optional[Dict[str, Any]] = None) -> int:
    """Recompute and cache findings for `models` (default: every model with trim data).

    Returns how many findings were stored. One model failing never stops the rest -- and it is
    never silent either. Pass a dict as `report` and it is filled with
        {"models": n, "stored": n,
         "failed_models": {model: "ExcType: message"},          # nothing stored; old cache kept
         "analyzer_errors": {model: {analyzer: "ExcType: message"}}}
    so a caller that tells a PERSON "done" can also tell them what did not get done.
    """
    if models is None:
        with db.session() as s:
            models = [r[0] for r in s.execute(text(
                "SELECT DISTINCT model FROM analysis_results "
                "WHERE system IN ('A','B','C') AND model IS NOT NULL ORDER BY model"))]
    fleet_latest = _fleet_latest(db)          # computed once -- "now" is the same instant for every model
    stored = n_models = 0
    failed_models: Dict[str, str] = {}
    analyzer_errors: Dict[str, Dict[str, str]] = {}
    for model in models:
        n_models += 1
        try:
            facts, findings = compute_for_model(db, model, fleet_latest=fleet_latest)
            stored += db.replace_process_findings(model, facts, [f.to_dict() for f in findings])
            if facts["errors"]:
                analyzer_errors[model] = dict(facts["errors"])
        except Exception as exc:
            logger.exception("findings: refresh failed for %s", model)
            failed_models[model] = f"{type(exc).__name__}: {exc}"
    if report is not None:
        report.update({"models": n_models, "stored": stored, "failed_models": failed_models,
                       "analyzer_errors": analyzer_errors})
    return stored
