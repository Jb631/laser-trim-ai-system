"""The status bar's data -- ONE loader, `load_status(db)`, that never raises -- and the words the
bar prints (option B, 2026-10-04; spec docs/superpowers/specs/2026-10-04-option-b-design.md).

What the bar says about the DATABASE, left to right, and where each part comes from:

  * "Database OK" -- the database answered a read at all. When it did not, the bar says what
    failed ("Database: could not read (OperationalError)") and nothing else about the database:
    every count below would fail the same way, and one line says why;
  * "333 models" -- every model the Models picker can open: one with a laser file or a smoothness
    file (`ml.manager.list_known_models`; the Overview's "other models on file" uses the same
    set), each once;
  * "Newest file 29 Sep 2026" -- the newest trim file that COUNTS: graded, believed (not dated
    more than a day ahead of now -- `core.activity.trusted_until`) and not marked suspect. The
    Overview's windows end on the same file (`overview_data`'s anchor), so the bar and the
    Overview never name two different dates;
  * "70 files skipped" -- the files the ingest refuses to re-offer because they failed to read
    before (`count_failed_file_markers`; the Overview's own notice, Settings -> Retry unreadable
    files). Said only when there are some.

The rest of the bar is not the database's: the drift watch (DRIFT_*, V6App's startup rebuild and
catch-up) and the run in flight (`RunState`, V6App's run observer) -- their words are here too, so
every word the bar prints is tested without Tk.

Worker-safe: no Tk, no widget. Each part that fails is NAMED in `Status.failed` and the others still
load; a count that could not be read is None, never 0 (CLAUDE.md: a failure must never look like a
result).
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Optional, Tuple

from laser_trim_analyzer.gui.v6 import formats

logger = logging.getLogger(__name__)

# The parts that load (and fail) on their own.
PART_DATABASE = "database"
PART_MODELS = "models"
PART_NEWEST = "newest file"
PART_SKIPPED = "skipped files"

# The drift watch, as V6App holds it while its startup rebuild/catch-up runs.
DRIFT_CURRENT = "current"
DRIFT_UPDATING = "updating"
DRIFT_FAILED = "failed"

# How a word is drawn: plain (TEXT_SECONDARY), "check this" (CHECK), or -- the database's own
# word only -- with the healthy dot.
QUIET, CHECK, OK = "quiet", "check", "ok"

Words = Tuple[str, str]          # (text, tone); "" = nothing to say, the slot is not drawn

_GRADED = ("PASS", "WARNING", "FAIL")


@dataclass(frozen=True)
class Status:
    models: Optional[int] = None         # None: not read (failed, or the database could not be)
    newest: Optional[datetime] = None    # None: no trim file on record -- or not read (`failed`)
    skipped: Optional[int] = None
    failed: Mapping[str, str] = field(default_factory=dict)    # part -> the exception's class

    @property
    def database_ok(self) -> bool:
        return PART_DATABASE not in self.failed


@dataclass(frozen=True)
class RunState:
    """The long run in flight, as V6App.run_progress() reports it. `label` is None for a run that
    reports no progress of its own (a re-grade, a findings refresh): the bar names it instead.
    `total` 0 = not known yet (the pre-scan). `started` is time.monotonic() at its registration."""
    name: str
    label: Optional[str] = None
    done: int = 0
    total: int = 0
    started: float = 0.0


# ---- the loader ----------------------------------------------------------------------------------

def load_status(db, *, now: Optional[datetime] = None) -> Status:
    """What the bar says about the database. Never raises: each part that fails is named."""
    try:
        _probe(db)
    except Exception as exc:
        logger.exception("Status bar: the database could not be read")
        return Status(failed={PART_DATABASE: type(exc).__name__})
    failed = {}
    models = _part(failed, PART_MODELS, lambda: _count_models(db))
    newest = _part(failed, PART_NEWEST, lambda: _newest_trim_file(db, now))
    skipped = _part(failed, PART_SKIPPED, lambda: int(db.count_failed_file_markers()))
    return Status(models=models, newest=newest, skipped=skipped, failed=failed)


def _part(failed: dict, part: str, read):
    try:
        return read()
    except Exception as exc:
        logger.exception("Status bar: %s could not be read", part)
        failed[part] = type(exc).__name__
        return None


def _probe(db) -> None:
    """One row, or none: enough to know the database answers a read."""
    from sqlalchemy import text
    with db.session() as s:
        s.execute(text("SELECT 1 FROM analysis_results LIMIT 1")).first()


def _count_models(db) -> int:
    """Every model the Models picker can open (list_known_models' set: a laser file or a smoothness
    file), from the two tables' distinct names in one query."""
    from sqlalchemy import select, union
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SmoothnessResult as DBSR
    with db.session() as s:
        rows = s.execute(union(select(DBAR.model), select(DBSR.model))).fetchall()
    return len({r[0] for r in rows if r[0]})


def _newest_trim_file(db, now: Optional[datetime]) -> Optional[datetime]:
    """The newest trim file that counts -- the Overview's anchor (overview_data._rates_by_day): a
    graded status, a model and a date, not dated past trusted_until(now), not marked suspect."""
    from sqlalchemy import func, or_
    from laser_trim_analyzer.core.activity import trusted_until
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, StatusType
    with db.session() as s:
        value = (s.query(func.max(DBAR.file_date))
                 .filter(DBAR.overall_status.in_([StatusType[x] for x in _GRADED]),
                         DBAR.model.isnot(None), DBAR.file_date.isnot(None),
                         DBAR.file_date <= trusted_until(now),
                         or_(DBAR.data_quality.is_(None), DBAR.data_quality != "suspect"))
                 .scalar())
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("T", " ")[:26])


# ---- the words the bar prints (pure, so they are tested without Tk) ------------------------------

def database_words(status: Optional[Status]) -> Words:
    if status is None:
        return "Checking the database…", QUIET
    if not status.database_ok:
        return f"Database: could not read ({status.failed[PART_DATABASE]})", CHECK
    return "Database OK", OK


def _known(status: Optional[Status]) -> bool:
    """The database's counts are there to say: loaded, and the database could be read."""
    return status is not None and status.database_ok


def models_words(status: Optional[Status]) -> Words:
    if not _known(status):
        return "", QUIET
    if PART_MODELS in status.failed:
        return f"Models: could not count ({status.failed[PART_MODELS]})", CHECK
    n = status.models or 0
    return (f"{n:,} model" if n == 1 else f"{n:,} models"), QUIET


def newest_words(status: Optional[Status]) -> Words:
    if not _known(status):
        return "", QUIET
    if PART_NEWEST in status.failed:
        return f"Newest file: could not read ({status.failed[PART_NEWEST]})", CHECK
    if status.newest is None:
        return "No trim files yet", QUIET
    return f"Newest file {formats.day(status.newest)}", QUIET


def skipped_words(status: Optional[Status]) -> Words:
    if not _known(status):
        return "", QUIET
    if PART_SKIPPED in status.failed:
        return f"Skipped files: could not count ({status.failed[PART_SKIPPED]})", CHECK
    n = status.skipped or 0
    if n <= 0:
        return "", QUIET
    return (f"{n:,} file skipped" if n == 1 else f"{n:,} files skipped"), CHECK


def drift_words(state: str, error: Optional[str] = None) -> Words:
    if state == DRIFT_UPDATING:
        return "Drift watch updating…", QUIET
    if state == DRIFT_FAILED:
        return f"Drift watch: could not update ({error or 'unknown error'})", CHECK
    return "Drift watch current", QUIET


def run_words(run: Optional[RunState]) -> str:
    """"Processing new files · 34 of 120"; "Processing new files…" before the total is known (never
    "0 of 0", which reads as a stuck run); "A re-grade is running" for a run with no progress."""
    if run is None:
        return ""
    if not run.label:
        return f"{run.name} is running"
    if run.total > 0:
        return f"{run.label} · {run.done:,} of {run.total:,}"
    return f"{run.label}…"
