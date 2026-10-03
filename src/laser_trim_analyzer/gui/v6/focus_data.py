"""The FOCUS list's page-side loader — one function, for every screen that shows it.

Home and Triage both opened on "is anything drifting right now?" and had to answer it
identically; since 2026-10-02 (Graphite redesign) the Overview's cards are its one caller
(gui/v6/overview_data.py: the fail-rate half of the cards). `ml/spc.compute_focus_list` owns the
membership rule, the ranking and the wording; what lives here is the wrapper around it — the
failure posture and the "last processed" stamp.

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


def load_focus(db, models: Optional[Sequence] = None, *, stamp: bool = True
               ) -> Tuple[FocusResult, Optional[datetime]]:
    """(FocusResult, last_processed). Never raises.

    A compute crash is logged loudly and returned as a FocusLoadFailed -- empty, like a clean
    shop floor, but marked, so a screen can name the failure in a banner instead of reading it
    as "all models within tolerance".

    `models` is a caller's already-loaded model list; pass it to avoid a second
    inventory query. Without it the stamp is read straight from the model inventory -- which runs
    the drift detector over every model again, so a caller that never prints the stamp (the
    Overview) passes stamp=False and gets None.
    """
    try:
        result = compute_focus_list(db)
    except Exception as exc:
        logger.exception("FOCUS computation failed")
        result = FocusLoadFailed(focus=[], chronic=[], anchor=None,
                                 error=f"{type(exc).__name__}: {exc}")
    return result, (_last_processed(db, models) if stamp else None)


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
