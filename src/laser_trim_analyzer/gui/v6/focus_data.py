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
from datetime import datetime
from typing import Optional, Sequence, Tuple

from laser_trim_analyzer.ml.spc import FocusResult, compute_focus_list

logger = logging.getLogger(__name__)

EMPTY = FocusResult(focus=[], chronic=[], anchor=None)


def load_focus(db, models: Optional[Sequence] = None
               ) -> Tuple[FocusResult, Optional[datetime]]:
    """(FocusResult, last_processed). Never raises.

    A compute crash degrades to an empty list — which reads as "all models
    within tolerance" — so it is logged loudly here. Without the log, a crash
    and a clean shop floor look identical on screen, which is exactly the bug
    this posture was written for.

    `models` is the caller's already-loaded model list (Triage has one for its
    browse list); pass it to avoid a second inventory query. Without it the
    stamp is read straight from the model inventory.
    """
    try:
        result = compute_focus_list(db)
    except Exception:
        logger.exception("FOCUS computation failed")
        result = EMPTY
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
