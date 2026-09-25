"""Is a model still being trimmed? One definition -- its newest trim file -- for every screen.

James, 2026-09-25: "models that havnt been trimmed in 2 years should show inacative or something
but i dont want to hide them as the data is useful if we decide to start building the model
again." So a model is INACTIVE when its newest trim file is more than INACTIVE_AFTER (730 days)
before the newest trim file in the whole database -- or when it was measured on a laser but never
cut ("no trims on record": never trimmed is certainly not trimmed in two years). It is LABELLED,
never hidden: every finding and every number stays where it was.

A TRIM FILE is a laser file (system A, B or C) with at least one track that was CUT and did not
fail processing. Two kinds of track are not a trim (`_NOT_A_TRIM`): one whose processing failed
(`core/model_stats.failed_processing_statuses()` -- its file_date is the moment the analyser gave
up, not a measurement), and a sweep the laser measured but did not cut, UNTRIMMED (controller
ruling, 2026-09-25: "havnt been trimmed"). Both dates -- the model's and the fleet's -- refuse a
file dated more than FUTURE_GRACE (a day) ahead of now: a mistyped filename date would otherwise
move a model's window, or make every other model look stale. This module owns that guard; the
findings engine's "now" (`engine._fleet_latest`) and machine_compare's window anchor (its tracks
with a cut, `t.passes`) are this module's dates.

FLEET-ANCHORED, not wall-clock: two years before the newest trim file in the database, so a home
copy of an older database reads the same as work. And worked out when a screen LOADS
(`load_activity`, one query, cached for that load only), never from cached findings: a model turns
inactive when other models' newer files move the fleet forward, without its own findings being
refreshed. Measured on the 6.2 GB work database (2026-09-25, read-only): about 80 ms warm, 1.7 s
on a cold file (the first load after a start; the first rule's query was 1.1 s cold, 50 ms warm);
326 models with laser files, 196 inactive -- 190 by date and 6 with no trim on record.

"No trims on record" needs a sweep that WAS read: a model whose files all failed processing (or
hold no track at all) gets no label. A record that failed processing is not a measurement
(CLAUDE.md) -- it may hold a perfectly good trim the app could not read -- and a failure must never
look like a result.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional

from sqlalchemy import text

from .model_stats import _FAILED_PROCESSING

FUTURE_GRACE = timedelta(days=1)       # a file dated further ahead of now than this is not believed
INACTIVE_AFTER = timedelta(days=730)   # James, 2026-09-25: "2 years"

# What does NOT count as a trim: a track that failed processing, or a sweep with no cut. Scoped to
# THIS module on purpose -- failed_processing_statuses() stays "failed processing" only, which its
# other callers (every average that must skip the 999.999 marker) need it to mean.
_NOT_A_TRIM = _FAILED_PROCESSING + ("UNTRIMMED",)

# One pass over every believable laser file (A/B/C, not dated in the future), per model: the newest
# that is a TRIM (NULL when none is), and whether ANY was measured -- read with a track that did not
# fail. "Has a track that is X" is asked as "has a track, and not every one of its tracks is
# not-X": the same set, answered from the indexes alone. Asking each track's own status reads every
# track row, sweep arrays and all: 3.7 s cold, 122 ms warm for the trim test alone
# (tests/test_model_activity.py checks the two forms agree on files of every shape).
_ALL_TRACKS_ARE = """
    SELECT f.analysis_id FROM track_results f WHERE f.status IN ({statuses})
    GROUP BY f.analysis_id
    HAVING COUNT(*) = (SELECT COUNT(*) FROM track_results c WHERE c.analysis_id = f.analysis_id)"""
_SQL = """
SELECT model,
       MAX(CASE WHEN tracked AND id NOT IN ({all_not_a_trim}) THEN file_date END),
       MAX(CASE WHEN tracked AND id NOT IN ({all_failed}) THEN 1 ELSE 0 END)
FROM (SELECT a.id AS id, a.model AS model, a.file_date AS file_date,
             EXISTS (SELECT 1 FROM track_results t WHERE t.analysis_id = a.id) AS tracked
      FROM analysis_results a
      WHERE a.system IN ('A', 'B', 'C') AND a.file_date <= :cutoff)
GROUP BY model
"""


def trusted_until(now: Optional[datetime] = None) -> datetime:
    """The latest file date believed: `now` (default: this moment) plus FUTURE_GRACE."""
    return (now if now is not None else datetime.now()) + FUTURE_GRACE


def newest_trim_file(dates: Iterable[Optional[datetime]],
                     now: Optional[datetime] = None) -> Optional[datetime]:
    """The newest of `dates` not after trusted_until(now), or None -- for trim files already in
    memory (machine_compare's cut tracks): the guard load_activity applies in SQL."""
    cutoff = trusted_until(now)
    return max((d for d in dates if d is not None and d <= cutoff), default=None)


def is_inactive(last: Optional[datetime], fleet: Optional[datetime]) -> bool:
    """A newest trim file more than INACTIVE_AFTER behind the fleet's -- the whole gap, never
    floored to days. (A model with laser files but no trim is `Activity.never_trimmed`.)"""
    return last is not None and fleet is not None and fleet - last > INACTIVE_AFTER


@dataclass(frozen=True)
class Activity:
    """One screen load's answer: every model's newest trim file, the fleet's, and the models that
    were measured on a laser but never cut."""
    newest: Mapping[str, datetime] = field(default_factory=dict)
    fleet: Optional[datetime] = None
    never_trimmed: FrozenSet[str] = frozenset()

    def last_trimmed(self, model: str) -> Optional[datetime]:
        return self.newest.get(model)

    def is_inactive(self, model: str) -> bool:
        """A model with no laser file at all is never called inactive: there is nothing to say."""
        if model in self.newest:
            return is_inactive(self.newest[model], self.fleet)
        return model in self.never_trimmed

    def inactive(self) -> Dict[str, Optional[datetime]]:
        """{model: its newest trim file, or None for "no trims on record"} for the inactive models
        only -- what the screens take (inactive_tag / inactive_caption read the None)."""
        out: Dict[str, Optional[datetime]] = {m: d for m, d in self.newest.items()
                                              if is_inactive(d, self.fleet)}
        out.update({m: None for m in self.never_trimmed})
        return out


def _as_datetime(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("T", " "))
    except (TypeError, ValueError):
        return None


def load_activity(db, now: Optional[datetime] = None) -> Activity:
    """Every model's newest trim file, the fleet's (the newest of them), and the models never
    trimmed, in ONE query."""
    params: Dict[str, Any] = {f"not_a_trim{i}": name for i, name in enumerate(_NOT_A_TRIM)}
    not_a_trim = ", ".join(f":{k}" for k in params)
    failed_params = {f"failed{i}": name for i, name in enumerate(_FAILED_PROCESSING)}
    failed = ", ".join(f":{k}" for k in failed_params)
    params.update(failed_params)
    # Bound as a string, not a raw datetime: text() bypasses SQLAlchemy's DATETIME bind processor,
    # and sqlite3's own datetime adapter is deprecated since 3.12. file_date is stored in exactly
    # this fixed-width format, so the string comparison sorts as the dates do (engine.py said so
    # first, for the same bind).
    params["cutoff"] = f"{trusted_until(now):%Y-%m-%d %H:%M:%S.%f}"
    with db.session() as s:
        rows = s.execute(text(_SQL.format(
            all_not_a_trim=_ALL_TRACKS_ARE.format(statuses=not_a_trim),
            all_failed=_ALL_TRACKS_ARE.format(statuses=failed))), params).fetchall()
    newest, never = {}, set()
    for model, last, measured in rows:
        if not model:
            continue
        d = _as_datetime(last) if last is not None else None
        if d is not None:
            newest[model] = d
        elif measured:
            never.add(model)
    return Activity(newest=newest, fleet=max(newest.values(), default=None),
                    never_trimmed=frozenset(never))


# ---- the words, one set for every screen ---------------------------------------------------

def inactive_tag(last: Optional[datetime]) -> str:
    """A findings row's quiet tag, and the Triage list's status: "Inactive · last trimmed Mar 2016",
    or "Inactive · no trims on record" (`last` None)."""
    return (f"Inactive · last trimmed {last:%b %Y}" if last is not None
            else "Inactive · no trims on record")


def inactive_caption(last: Optional[datetime]) -> str:
    """How the Model page's caption starts: "Inactive — last trimmed Mar 2016", or "Inactive — no
    trims on record" (`last` None)."""
    return (f"Inactive — last trimmed {last:%b %Y}" if last is not None
            else "Inactive — no trims on record")


def activity_unknown_notice(reason: str) -> str:
    """When which models are inactive could not be worked out: nothing is marked, and silence
    would claim every model is active."""
    return (f"Which models are inactive could not be worked out ({reason}), so no model is marked "
            f"Inactive here — this is an error, not a sign that every model is active. The log has "
            f"the details.")
