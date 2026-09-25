"""The FOCUS list's page-side loader — one function, two pages.

Home and Triage both open on "is anything drifting right now?", and they have
to answer it identically. `ml/spc.compute_focus_list` already owns the
membership rule, the ranking and the wording; what was about to get copied is
the wrapper around it — the failure posture and the "last processed" stamp the
empty state prints. So it lives here, called by both.

Worker-safe: no Tk, no widget, no page state. Callers run it on a thread and
marshal the result back through `safe_after`/`ui_dispatch`.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence, Tuple

from laser_trim_analyzer.ml.spc import FocusResult, compute_focus_list

logger = logging.getLogger(__name__)

EMPTY = FocusResult(focus=[], chronic=[], anchor=None)


@dataclass
class FocusLoadFailed(FocusResult):
    """What load_focus returns when the computation RAISED: empty lists, so a caller that only
    iterates still works, but a different thing from EMPTY -- `error` names the exception.
    Until the final review (2026-09-24) a crash came back as EMPTY itself, and the screens drew
    it as "0 drifting now", "Needs a look · 0" and "All models within tolerance": a failure
    looking like the best possible result. Every screen that draws the list checks
    focus_failed()."""
    error: str = ""


def focus_failed(result) -> Optional[str]:
    """The failure's "ExcType: message" when `result` is a failed load, else None."""
    return result.error if isinstance(result, FocusLoadFailed) else None


def load_focus(db, models: Optional[Sequence] = None
               ) -> Tuple[FocusResult, Optional[datetime]]:
    """(FocusResult, last_processed). Never raises.

    A compute crash is logged loudly and returned as a FocusLoadFailed -- empty, like a clean
    shop floor, but marked, so a screen can name the failure in a banner instead of reading it
    as "all models within tolerance".

    `models` is the caller's already-loaded model list (Triage has one for its
    browse list); pass it to avoid a second inventory query. Without it the
    stamp is read straight from the model inventory.
    """
    try:
        result = compute_focus_list(db)
    except Exception as exc:
        logger.exception("FOCUS computation failed")
        result = FocusLoadFailed(focus=[], chronic=[], anchor=None,
                                 error=f"{type(exc).__name__}: {exc}")
    return result, _last_processed(db, models)


def _last_processed(db, models: Optional[Sequence]) -> Optional[datetime]:
    """Newest data on record — the date an empty FOCUS list is 'as of'."""
    try:
        if models is None:
            from laser_trim_analyzer.ml.manager import list_known_models
            models = list_known_models(db)
        return max((m.last_processed for m in models if m.last_processed),
                   default=None)
    except Exception:
        logger.exception("Last-processed lookup failed")
        return None
