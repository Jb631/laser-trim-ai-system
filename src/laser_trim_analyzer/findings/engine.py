"""Run every analyzer for a model, rank what they found, and cache it.

Runs after an ingest, on the ingest worker thread (never the Tk thread), for
the models that ingest touched. The screens only ever read the cache.
"""
import logging
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import text

from .analyzers import ink_target, recipe_change, trim_effort
from .data import load_model_tracks, yardstick_fidelity
from .model import Finding, rank

logger = logging.getLogger(__name__)


def _laser_label(system: str) -> str:
    # The shop's names, never the code's letters (A is laser TWO). Imported late
    # so this package stays importable in a bare test of the analyzers.
    from laser_trim_analyzer.core.models import laser_label
    return laser_label(system)


def compute_for_model(db, model: str) -> Tuple[Dict[str, Any], List[Finding]]:
    tracks = load_model_tracks(db, model)
    facts: Dict[str, Any] = {"model": model, "tracks": len(tracks)}
    if not tracks:
        return facts, []
    latest = max(t.file_date for t in tracks)
    volume = sum(1 for t in tracks if t.file_date >= latest - timedelta(days=365))
    facts["annual_volume"] = volume
    facts["latest"] = latest.date().isoformat()
    fidelity = yardstick_fidelity(tracks)
    facts["yardstick"] = fidelity
    findings: List[Finding] = []
    # Each analyzer is independently guarded: one failing must not blank the rest.
    try:
        history, changes = recipe_change.analyze(model, tracks, _laser_label)  # stored verdicts only
        facts["recipe_history"] = history
        findings += changes
    except Exception:
        logger.exception("findings: recipe_change failed for %s", model)
    try:
        findings += ink_target.analyze(model, tracks, _laser_label)         # stored verdicts only
    except Exception:
        logger.exception("findings: ink_target failed for %s", model)
    if fidelity["faithful"]:
        try:
            effort_facts, effort_findings = trim_effort.analyze(model, tracks, _laser_label)
            facts["trim_effort"] = effort_facts
            findings += effort_findings
        except Exception:
            logger.exception("findings: trim_effort failed for %s", model)
    else:
        # Grading intermediate sweeps is only honest where the yardstick reproduces
        # the app's own verdict. Where it does not, say nothing rather than guess.
        facts["trim_effort"] = None
    for f in findings:
        f.annual_volume = volume
    return facts, rank(findings)


def refresh_findings(db, models: Optional[Iterable[str]] = None) -> int:
    """Recompute and cache findings for `models` (default: every model with trim data).

    Returns how many findings were stored. One model failing never stops the rest.
    """
    if models is None:
        with db.session() as s:
            models = [r[0] for r in s.execute(text(
                "SELECT DISTINCT model FROM analysis_results "
                "WHERE system IN ('A','B','C') AND model IS NOT NULL ORDER BY model"))]
    stored = 0
    for model in models:
        try:
            facts, findings = compute_for_model(db, model)
            stored += db.replace_process_findings(model, facts, [f.to_dict() for f in findings])
        except Exception:
            logger.exception("findings: refresh failed for %s", model)
    return stored
