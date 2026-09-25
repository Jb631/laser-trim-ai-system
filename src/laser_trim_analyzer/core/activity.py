"""Is a model still being trimmed? One definition -- its newest trim file -- for every screen.

James, 2026-09-25: "models that havnt been trimmed in 2 years should show inacative or something
but i dont want to hide them as the data is useful if we decide to start building the model
again." So a model is INACTIVE when its newest trim file is more than INACTIVE_AFTER (730 days)
before the newest trim file in the whole database. It is LABELLED, never hidden: every finding and
every number stays where it was.

A TRIM FILE is a laser file (system A, B or C) with at least one track that did not fail processing
(`core/model_stats.failed_processing_statuses()`: a failed record's file_date is the moment the
analyser gave up, not a measurement). An UNTRIMMED sweep counts -- the model was on the laser. Both
dates -- the model's and the fleet's -- refuse a file dated more than FUTURE_GRACE (a day) ahead of
now: a mistyped filename date would otherwise move a model's window, or make every other model look
stale. This module owns that guard; the findings engine's "now" (`engine._fleet_latest`) and
machine_compare's window anchor are this module's dates.

FLEET-ANCHORED, not wall-clock: two years before the newest trim file in the database, so a home
copy of an older database reads the same as work. And worked out when a screen LOADS
(`load_activity`, one query, cached for that load only), never from cached findings: a model turns
inactive when other models' newer files move the fleet forward, without its own findings being
refreshed. Measured on a writable copy of the 6.2 GB work database (2026-09-25): 54 ms warm, 326
models, 195 of them inactive.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Mapping, Optional

from sqlalchemy import text

from .model_stats import _FAILED_PROCESSING

FUTURE_GRACE = timedelta(days=1)       # a file dated further ahead of now than this is not believed
INACTIVE_AFTER = timedelta(days=730)   # James, 2026-09-25: "2 years"

# "A track that did not fail processing", asked as "has a track, and not every one of its tracks
# failed" -- the same set of files, answered from the indexes alone. Asking each track's own status
# reads every track row, sweep arrays and all: 121 ms against 54 ms on the 6.2 GB copy
# (tests/test_model_activity.py checks the two forms agree on files of every shape).
_SQL = """
SELECT a.model, MAX(a.file_date) FROM analysis_results a
WHERE a.system IN ('A', 'B', 'C') AND a.file_date <= :cutoff
  AND EXISTS (SELECT 1 FROM track_results t WHERE t.analysis_id = a.id)
  AND a.id NOT IN (
      SELECT f.analysis_id FROM track_results f
      WHERE f.status IN ({failed})
      GROUP BY f.analysis_id
      HAVING COUNT(*) = (SELECT COUNT(*) FROM track_results c WHERE c.analysis_id = f.analysis_id))
GROUP BY a.model
"""


def trusted_until(now: Optional[datetime] = None) -> datetime:
    """The latest file date believed: `now` (default: this moment) plus FUTURE_GRACE."""
    return (now if now is not None else datetime.now()) + FUTURE_GRACE


def newest_trim_file(dates: Iterable[Optional[datetime]],
                     now: Optional[datetime] = None) -> Optional[datetime]:
    """The newest of `dates` not after trusted_until(now), or None -- for trim files already in
    memory (machine_compare's tracks): the guard load_activity applies in SQL."""
    cutoff = trusted_until(now)
    return max((d for d in dates if d is not None and d <= cutoff), default=None)


def is_inactive(last: Optional[datetime], fleet: Optional[datetime]) -> bool:
    """More than INACTIVE_AFTER behind -- the whole gap, never floored to days. A model with no
    trim file, or a database with none, is never called inactive: there is nothing to say."""
    return last is not None and fleet is not None and fleet - last > INACTIVE_AFTER


@dataclass(frozen=True)
class Activity:
    """One screen load's answer: every model's newest trim file, and the fleet's."""
    newest: Mapping[str, datetime] = field(default_factory=dict)
    fleet: Optional[datetime] = None

    def last_trimmed(self, model: str) -> Optional[datetime]:
        return self.newest.get(model)

    def is_inactive(self, model: str) -> bool:
        return is_inactive(self.newest.get(model), self.fleet)

    def inactive(self) -> Dict[str, datetime]:
        """{model: its newest trim file} for the inactive models only -- what the screens take."""
        return {m: d for m, d in self.newest.items() if is_inactive(d, self.fleet)}


def _as_datetime(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("T", " "))
    except (TypeError, ValueError):
        return None


def load_activity(db, now: Optional[datetime] = None) -> Activity:
    """Every model's newest trim file, and the fleet's (the newest of them), in ONE query."""
    params: Dict[str, Any] = {f"failed{i}": name for i, name in enumerate(_FAILED_PROCESSING)}
    failed = ", ".join(f":{k}" for k in params)
    # Bound as a string, not a raw datetime: text() bypasses SQLAlchemy's DATETIME bind processor,
    # and sqlite3's own datetime adapter is deprecated since 3.12. file_date is stored in exactly
    # this fixed-width format, so the string comparison sorts as the dates do (engine.py said so
    # first, for the same bind).
    params["cutoff"] = f"{trusted_until(now):%Y-%m-%d %H:%M:%S.%f}"
    with db.session() as s:
        rows = s.execute(text(_SQL.format(failed=failed)), params).fetchall()
    newest = {}
    for model, last in rows:
        d = _as_datetime(last)
        if model and d is not None:
            newest[model] = d
    return Activity(newest=newest, fleet=max(newest.values(), default=None))


# ---- the words, one set for every screen ---------------------------------------------------

def inactive_tag(last: datetime) -> str:
    """A findings row's quiet tag, and the Triage list's status: "Inactive · last trimmed Mar 2016"."""
    return f"Inactive · last trimmed {last:%b %Y}"


def inactive_caption(last: datetime) -> str:
    """How the Model page's caption starts: "Inactive — last trimmed Mar 2016"."""
    return f"Inactive — last trimmed {last:%b %Y}"


def activity_unknown_notice(reason: str) -> str:
    """When which models are inactive could not be worked out: nothing is marked, and silence
    would claim every model is active."""
    return (f"Which models are inactive could not be worked out ({reason}), so no model is marked "
            f"Inactive here — this is an error, not a sign that every model is active. The log has "
            f"the details.")
