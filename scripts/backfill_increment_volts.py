"""Back-fill laser 1's TrimVolts response curves onto stored `Trim N` passes.

Task 4 (2026-09-24, commit `49f865e`) made the parser capture laser 1's (LTS, system code
`B` in `analysis_results.system`) `TrimVolts N` sheets onto every NEW `Trim N` pass row it
saves: `increment_volts` (JSON list of lists, one per engaged position), `increment_volts_
first_row` and `increment_volts_truncated`. A pass row already in the database before that
change carries a real SQL NULL in `increment_volts` -- Task 4's own chosen meaning of "not
captured", for a row written before the column existed and for one the parser tried and
found nothing for alike (design doc `2026-09-23-parse-fixes-design.md` ruling 4c).

This script re-opens each such file, read-only, and fills those three columns through the
SAME reader Task 4 uses (`core.trim_passes.read_increment_volts`, fed by `core.trim_passes.
increment_volts_frame` and `core.trim_setup.read_keyvalue`) -- so a back-filled row is
identical to a freshly-parsed one. It reads ONLY what it needs per file: the `Model
Parameters` (and `Track Parameters`, if present) sheet for `increment_volts_frame`'s fields
(`Points From Start`/`Points From End` when the file carries both, else `Initial`/`Ending
Points Ignored`, plus `Number of Readings (Lin)`), each `TrimVolts N` sheet a candidate
`Trim N` pass actually needs, and -- since Task 5 fix round 2 -- the workbook's own
`VOLTAGES` sheet, to check each capture before writing it (see "Placement self-check"
below). It never re-reads a `Trim N` sheet itself -- nothing on positions/errors/limits/
recipe/etc. is touched, on this row or any other.

A pass whose file has no `TrimVolts{N}` sheet (the one-in-4,972 touch-up case Task 4
measured), or whose sheet exists but fails to read, is left exactly as it was: NULL,
indistinguishable from "not yet attempted". That is Task 4's own representation, not
something this script can improve on without changing that ruling -- so a stopped-short
handful of files will be re-opened, harmlessly, on every future run. See the module's test
file for how small that set is expected to be.

Placement self-check (Task 5 fix round 2). Task 4's `first_row` rule (2026-09-24 review:
Points From Start/End when the file names both) was proven on 6,263 local sheets, but the
work database carries 4,311 laser-1 files, across many models, whose Points From Start
differs from its Initial Points Ignored -- more variety than the local corpus can promise to
have exercised. Before writing a pass, this script checks the capture against the workbook's
own `VOLTAGES` sheet the same way `scripts/app_qa_sweep.py`'s corpus sweep already does
(`core.trim_passes.voltages_placement`, shared by both so they can never disagree about a
file): if VOLTAGES CONTRADICTS the placement, that pass is refused -- not written -- and
named in the summary under "placement disagrees" (a wrong position is worse than none); if
the workbook has no `VOLTAGES` sheet (or it cannot settle this specific column), the pass is
written as it always was, and counted under "placement unverified".

Unlike the QA harnesses (`chart_qa_render_all.py`, `app_qa_sweep.py`), this script does NOT
refuse `data/analysis.db` -- filling in three columns of your own database, from your own
files, is the whole point, the same as `regrade_final_tests.py` and the other `backfill_*.py`
scripts in this folder. TAKE A SNAPSHOT FIRST anyway, onto LOCAL disk next to the database --
never a cloud-synced folder, which would queue the database's own size for upload just for a
safety net (`BRING_TO_WORK.md` already says to keep `data\\` out of OneDrive for exactly this
reason). `snapshot_db.py` refuses to overwrite an existing file:

    python scripts/snapshot_db.py data/analysis.db data/analysis_pre_tv_backfill.db

Usage
-----
    python scripts/backfill_increment_volts.py DB_PATH               # writes
    python scripts/backfill_increment_volts.py DB_PATH --dry-run     # reads and reports only
    python scripts/backfill_increment_volts.py DB_PATH --limit 2000  # first N files only

DB_PATH is REQUIRED -- there is no default, so this can never land on the wrong database
through an empty command line. Progress prints every 200 files (files/second and an ETA);
writes commit every 200 files, so a stopped run (Ctrl-C, a dropped VPN, going home for the
night) keeps everything already committed. Running the same command again continues: it
re-selects exactly the passes still missing their curves, NEWEST file first (`analysis_
results.file_date` descending, `id` descending as the tie-break). Deliberately NOT the
oldest-first order `id` ascending alone would give: `DatabaseManager.
get_final_tests_for_regrade`'s docstring learned this the hard way (2026-09-14) and abandoned
it for the same reason it applies here -- id/insertion order is oldest-file-first, and a run
measured in hours that walks the database that way spends its first stretch on 2010-era
workbooks while every screen in this app, and the cut-length model this capture feeds, reads
from the last year or two. A stopped or `--limit`-bounded run must leave the RECENT data
filled, not the oldest slice of it.
"""
import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import null as sql_null

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from laser_trim_analyzer.core import trim_passes as _tp          # noqa: E402
from laser_trim_analyzer.core import trim_setup as _ts           # noqa: E402
from laser_trim_analyzer.core.ingest_run import (                # noqa: E402
    EtaEstimator, format_clock, format_rate)
from laser_trim_analyzer.core.parser import ExcelParser           # noqa: E402
from laser_trim_analyzer.database.models import (                # noqa: E402
    AnalysisResult, SystemType, TrackResult, TrimPass)

logger = logging.getLogger(__name__)

PROGRESS_EVERY = 200
BATCH_SIZE = 200


class PassRef(NamedTuple):
    pass_id: int
    sheet: str


class FileWork(NamedTuple):
    analysis_id: int
    file_path: Optional[str]
    filename: Optional[str]
    passes: List[PassRef]


class CaptureResult(NamedTuple):
    """What `_capture_file` found for one file's candidate sheets.

    captures: {sheet: capture} -- cleared to write (placement confirmed, or unverifiable).
    problem: None, "missing" or "unreadable" (see `_capture_file`).
    disagreed: sheets whose capture was REFUSED -- the workbook's own VOLTAGES sheet
        contradicts the placement, so it was never added to `captures` and is not written.
    unverified: sheets that ARE in `captures` (written) but could not be checked against
        VOLTAGES (no VOLTAGES sheet, or it does not reach this sheet's column).
    empty: sheets that ARE in `captures` but whose TrimVolts sheet holds no reading at all
        -- a capture of nothing, so the writer stores NULL, exactly as a fresh parse does.
        Counted on their own line, never as filled (2026-09-24 final review: they were
        counted "Filled" while NULL was written).
    """
    captures: Dict[str, Dict[str, Any]]
    problem: Optional[str]
    disagreed: List[str]
    unverified: List[str]
    empty: Sequence[str] = ()     # the default for callers that build one by hand


class BackfillReport:
    """Everything a caller (the CLI, or a test) needs to know about one run."""

    def __init__(self, *, dry_run: bool, candidates_total: int,
                candidate_files_total: int, files_run: int) -> None:
        self.dry_run = dry_run
        self.candidates_total = candidates_total          # ALL laser-1 candidate passes, unbounded by --limit
        self.candidate_files_total = candidate_files_total  # ALL laser-1 candidate files, unbounded by --limit
        self.files_run = files_run                         # files this run actually attempted
        self.files_updated = 0
        self.passes_filled = 0
        self.passes_unfilled = 0                            # attempted, but no sheet / unreadable sheet
        self.passes_disagreed = 0                # refused: VOLTAGES contradicts the placement
        self.passes_unverified = 0            # written, but no VOLTAGES sheet to check against
        self.passes_empty = 0        # a TrimVolts sheet with no reading: NULL written, not filled
        self.files_missing = 0
        self.files_unreadable = 0
        self.missing_files: List[str] = []
        self.unreadable_files: List[str] = []
        self.disagreements: List[str] = []          # "<path> <sheet>", refused, never written
        self.unverified_placements: List[str] = []  # "<path> <sheet>", written, unchecked
        self.empty_sheets: List[str] = []           # "<path> <sheet>", NULL written, no reading
        self.elapsed_s = 0.0

    def __repr__(self) -> str:      # pragma: no cover -- debugging aid only
        return (f"BackfillReport(dry_run={self.dry_run}, files_run={self.files_run}, "
                f"files_updated={self.files_updated}, passes_filled={self.passes_filled}, "
                f"passes_unfilled={self.passes_unfilled}, "
                f"passes_disagreed={self.passes_disagreed}, "
                f"passes_unverified={self.passes_unverified}, "
                f"passes_empty={self.passes_empty}, "
                f"files_missing={self.files_missing}, "
                f"files_unreadable={self.files_unreadable})")


# ---------------------------------------------------------------------------
# Selection: exactly the candidates Task 4 could not have written yet.
# ---------------------------------------------------------------------------

def _select_candidates(db) -> List[FileWork]:
    """Every laser-1 `Trim N` pass with `increment_volts IS NULL`, grouped one entry per
    file, NEWEST file first (`file_date` descending, `id` descending as the tie-break for
    same-dated or undated rows -- stable across runs, which is what lets a stopped run
    resume somewhere sensible instead of re-shuffling).

    Newest first, not oldest: `id` ascending alone is INGEST order, which is oldest-file-
    first, and `DatabaseManager.get_final_tests_for_regrade`'s docstring already tells this
    exact story (2026-09-14) -- walking a multi-hour database pass that way spent the first
    hour and a half on 2010-era workbooks while every number the app shows, and the
    cut-length model this capture feeds, comes from the last year or two. A stopped or
    `--limit`-bounded run has to leave the USEFUL rows filled.

    `file_date` descending puts a NULL date LAST, not first: SQLite treats NULL as smaller
    than any value, so ASC sorts it first and DESC sorts it last -- which is right here too,
    the same as the regrade's own ordering: a row with no date is the one whose recency
    cannot be claimed, so it must never jump the queue ahead of a row that can.

    `sheet.ilike('trim %')` excludes `Lin Error` (which never carries TrimVolts and would
    otherwise be re-selected, forever, on every single run) and matches only what `core.
    trim_passes.pass_sheets` can ever have written into `trim_passes.sheet` for system B —
    real `Trim N` names, always. `increment_volts IS NULL` is exact here (never the JSON
    text `'null'`): see `_write_trim_passes`'s ruling in `database/manager.py`.
    """
    with db.session() as s:
        rows = (
            s.query(AnalysisResult.id, AnalysisResult.file_path, AnalysisResult.filename,
                    TrimPass.id, TrimPass.sheet)
            .join(TrackResult, TrimPass.track_result_id == TrackResult.id)
            .join(AnalysisResult, TrackResult.analysis_id == AnalysisResult.id)
            .filter(AnalysisResult.system == SystemType.B)
            .filter(TrimPass.increment_volts.is_(None))
            .filter(TrimPass.sheet.ilike("trim %"))
            .order_by(AnalysisResult.file_date.desc(), AnalysisResult.id.desc(),
                     TrimPass.id.asc())
            .all()
        )

    out: List[FileWork] = []
    current: Optional[FileWork] = None
    for analysis_id, file_path, filename, pass_id, sheet in rows:
        if current is None or current.analysis_id != analysis_id:
            current = FileWork(analysis_id=analysis_id, file_path=file_path,
                               filename=filename, passes=[])
            out.append(current)
        current.passes.append(PassRef(pass_id=pass_id, sheet=sheet))
    return out


# ---------------------------------------------------------------------------
# Reading: the same two things Task 4 reads, and nothing else.
# ---------------------------------------------------------------------------

# Mirrors core/parser.py ExcelParser.parse_file's own trim_setup loop EXACTLY: same three
# (sheet, label_col, value_col) candidates, tried in the same order, same "a layout that
# produced nothing was the wrong layout for this file" guard (>=3 keys), same setdefault
# (first occurrence wins). Not imported from there because parse_file does not expose this
# block separately -- it sits inline, beside reading the whole track, which this script must
# NOT do (that would mean re-reading every `Trim N` sheet this script has no business
# touching). The real work -- which layout is which, what a label normalises to -- is still
# 100% `core.trim_setup.read_keyvalue`, called exactly as the parser calls it.
_SETUP_CANDIDATES: Tuple[Tuple[str, int, int], ...] = (
    ("Track Parameters", 0, 1),     # System A layout (never matches a laser-1 file)
    ("Model Parameters", 0, 1),     # System A layout
    ("Model Parameters", 1, 0),     # System B layout -- laser 1's own
)


def _read_trim_setup(xl: pd.ExcelFile) -> Dict[str, Any]:
    setup: Dict[str, Any] = {}
    for sheet, label_col, value_col in _SETUP_CANDIDATES:
        if sheet not in xl.sheet_names:
            continue
        try:
            got = _ts.read_keyvalue(
                pd.read_excel(xl, sheet_name=sheet, header=None),
                label_col=label_col, value_col=value_col)
        except Exception:
            continue
        if len(got) >= 3:          # a layout that produced nothing was the wrong layout
            for k, v in got.items():
                setup.setdefault(k, v)
    return setup


def _capture_file(file_path: Optional[str], sheets_needed: List[str]) -> CaptureResult:
    """Open `file_path` read-only and read exactly what is needed for `sheets_needed`
    (each a `trim_passes.sheet` value, e.g. "Trim 1"): `Model Parameters`/`Track Parameters`,
    the specific `TrimVolts N` sheets those sheets need, and `VOLTAGES` (if present) to check
    each one before it is offered to be written.

    A capture that has anything live in it (`any(curves)`) is checked against `VOLTAGES`
    with `core.trim_passes.voltages_placement` -- the SAME comparison
    `scripts/app_qa_sweep.py`'s corpus sweep runs, not a second copy of it:
      - "misplaced": VOLTAGES contradicts where this capture says its readings are. Refused
        -- left out of `captures` entirely, named in `disagreed` -- a wrong position stored
        as data is worse than another NULL row a later run can still fill correctly.
      - "uncheckable": no VOLTAGES sheet, or it does not reach this sheet's column. Written
        as it always was (Task 4's own behaviour, unchanged), named in `unverified`.
      - "placed": written, nothing special recorded.
    A pass with no `TrimVolts N` sheet beside it, or one that fails to read, has no capture
    at all: absent from `captures` and from every list, left exactly as it was. A sheet that
    reads but holds no reading at all (`not any(curves)`) has nothing to check: it IS in
    `captures` -- its capture is all-None, so the writer stores NULL, exactly as a fresh
    parse would -- and is named in `empty`, so the run counts it on its own line and never
    as filled. (Until 2026-09-24 this docstring said such a sheet was absent from
    `captures` while the code added it, and the run reported it "Filled" over a NULL.)

    See `CaptureResult` for the full return shape. `problem` is None, "missing" (no such
    file here) or "unreadable" (the file exists but the workbook could not be opened, or
    something inside it raised unexpectedly). Never raises.
    """
    if not file_path:
        return CaptureResult({}, "missing", [], [])
    path = Path(file_path)
    try:
        exists = path.exists()
    except OSError:
        exists = False
    if not exists:
        return CaptureResult({}, "missing", [], [])

    try:
        opened = pd.ExcelFile(path)
    except Exception:
        logger.debug("backfill: could not open %s", file_path, exc_info=True)
        return CaptureResult({}, "unreadable", [], [])

    try:
        # A context manager, like every other ExcelFile in this codebase (parser.py's
        # _workbook): closes the underlying handle on the way out instead of leaving it to
        # the garbage collector across tens of thousands of files over several hours.
        with opened as xl:
            setup = _read_trim_setup(xl)
            first_row, window = _tp.increment_volts_frame(setup)
            trimvolts = _tp.trimvolts_sheets(xl.sheet_names)
            volts = None
            if "VOLTAGES" in xl.sheet_names:
                try:
                    volts = pd.read_excel(xl, sheet_name="VOLTAGES", header=None)
                except Exception:
                    logger.debug("backfill: could not read VOLTAGES in %s", file_path,
                                exc_info=True)
                    volts = None      # same as no VOLTAGES sheet: every capture uncheckable

            out: Dict[str, Dict[str, Any]] = {}
            disagreed: List[str] = []
            unverified: List[str] = []
            empty: List[str] = []
            for sheet in sheets_needed:
                m = ExcelParser._TRIM_N_RE.match(sheet.strip())
                if not m:
                    continue
                n = int(m.group(1))
                tv_sheet = trimvolts.get(n)
                if tv_sheet is None:
                    logger.debug("backfill: no TrimVolts sheet beside %r in %s", sheet, file_path)
                    continue
                try:
                    df = pd.read_excel(xl, sheet_name=tv_sheet, header=None)
                except Exception:
                    logger.debug("backfill: could not read %r beside %r in %s",
                                tv_sheet, sheet, file_path, exc_info=True)
                    continue
                cap = _tp.read_increment_volts(df, first_row, window)
                curves = cap["increment_volts"]
                if any(curves):
                    pc = _tp.voltages_placement(
                        curves, cap["increment_volts_first_row"], volts, n)
                    if pc.result == "misplaced":
                        logger.warning(
                            "backfill: %r in %s placed at first_row %s disagrees with "
                            "VOLTAGES (%d of %d off, %d matched) -- refusing to write it",
                            sheet, file_path, cap["increment_volts_first_row"],
                            pc.bad, pc.live_count, pc.matched)
                        disagreed.append(sheet)
                        continue                   # refused: never added to `out`
                    if pc.result == "uncheckable":
                        unverified.append(sheet)
                else:
                    empty.append(sheet)            # nothing read: written as NULL, not filled
                out[sheet] = cap
            return CaptureResult(out, None, disagreed, unverified, empty)
    except Exception:
        # Nothing inside core.trim_passes is documented to raise, but a file this script
        # has never seen before gets no benefit of the doubt: one bad workbook must cost
        # only itself, never the run.
        logger.warning("backfill: unexpected failure reading %s", file_path, exc_info=True)
        return CaptureResult({}, "unreadable", [], [])


# ---------------------------------------------------------------------------
# The run itself.
# ---------------------------------------------------------------------------

ProgressFn = Callable[[int, int, str, Optional[float], str], None]
SelectedFn = Callable[[int, int, int], None]


def backfill(db, *, dry_run: bool = False, limit: Optional[int] = None,
            batch_size: int = BATCH_SIZE, progress_every: int = PROGRESS_EVERY,
            progress: Optional[ProgressFn] = None,
            on_selected: Optional[SelectedFn] = None) -> BackfillReport:
    """Run the back-fill (or, with `dry_run=True`, compute and report without writing).

    `progress(done, total, filename, rate, eta_text)` is called every `progress_every`
    files and once more at the end. `on_selected(candidates_total, candidate_files_total,
    files_run)` is called once, right after the read-only selection, before any file is
    opened -- for a caller (the CLI) that wants to print "what this will touch" without a
    second, redundant selection query.
    """
    all_files = _select_candidates(db)
    candidates_total = sum(len(f.passes) for f in all_files)
    candidate_files_total = len(all_files)
    files = all_files[:limit] if limit is not None else all_files
    total = len(files)

    if on_selected is not None:
        on_selected(candidates_total, candidate_files_total, total)

    report = BackfillReport(dry_run=dry_run, candidates_total=candidates_total,
                            candidate_files_total=candidate_files_total, files_run=total)
    if total == 0:
        return report

    eta = EtaEstimator()
    started = time.monotonic()
    pending: List[Tuple[int, Dict[str, Any]]] = []     # (pass_id, capture)
    batch_files = 0

    def flush() -> None:
        nonlocal batch_files
        if pending and not dry_run:
            with db.session() as s:
                for pass_id, cap in pending:
                    row = s.get(TrimPass, pass_id)
                    if row is None:        # gone since selection -- nothing to write
                        continue
                    # Mirrors _write_trim_passes exactly: an empty curves list is stored as
                    # a real SQL NULL, never the JSON text "[]"; first_row/truncated are
                    # written unconditionally FOR EVERY PASS THAT REACHES `pending` -- which
                    # is not every candidate pass any more (Task 5 fix round 2): one whose
                    # placement disagreed with the workbook's own VOLTAGES sheet was refused
                    # earlier, in `_capture_file`/the loop below, and never added here at
                    # all. What Task 4's writer does for a pass it DOES write is unchanged.
                    row.increment_volts = cap["increment_volts"] or sql_null()
                    row.increment_volts_first_row = cap["increment_volts_first_row"]
                    row.increment_volts_truncated = cap["increment_volts_truncated"]
        pending.clear()
        batch_files = 0

    for i, work in enumerate(files):
        sheets_needed = [p.sheet for p in work.passes]
        result = _capture_file(work.file_path, sheets_needed)
        # Progress wants the short, friendly name; a missing/unreadable/disagreed report
        # wants the actual PATH -- that is the actionable fact (which share location is
        # unreachable, or which file's placement to go look at), and two files can share a
        # bare filename in different folders.
        display_name = work.filename or work.file_path or f"analysis {work.analysis_id}"
        problem_name = work.file_path or work.filename or f"analysis {work.analysis_id}"

        if result.problem == "missing":
            report.files_missing += 1
            report.missing_files.append(problem_name)
        elif result.problem == "unreadable":
            report.files_unreadable += 1
            report.unreadable_files.append(problem_name)
        else:
            touched = False
            disagreed_sheets = set(result.disagreed)
            unverified_sheets = set(result.unverified)
            empty_sheets = set(result.empty)
            for p in work.passes:
                cap = result.captures.get(p.sheet)
                if cap is not None and p.sheet in empty_sheets:
                    pending.append((p.pass_id, cap))       # NULL, as a fresh parse stores it
                    report.passes_empty += 1
                    report.empty_sheets.append(f"{problem_name} {p.sheet}")
                elif cap is not None:
                    pending.append((p.pass_id, cap))
                    report.passes_filled += 1
                    touched = True
                    if p.sheet in unverified_sheets:
                        report.passes_unverified += 1
                        report.unverified_placements.append(f"{problem_name} {p.sheet}")
                elif p.sheet in disagreed_sheets:
                    report.passes_disagreed += 1
                    report.disagreements.append(f"{problem_name} {p.sheet}")
                else:
                    report.passes_unfilled += 1
            if touched:
                report.files_updated += 1

        batch_files += 1
        done = i + 1
        eta.note(done)
        if batch_files >= batch_size:
            flush()
        if progress is not None and (done % progress_every == 0 or done == total):
            progress(done, total, display_name, eta.rate(), eta.eta_text(total - done))

    flush()
    report.elapsed_s = time.monotonic() - started
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_NAME = re.compile(r"^(?P<dir>.*[\\/])?(?P<name>[^\\/]+)$")


def _snapshot_hint(db_path) -> str:
    """The exact command to run first -- right next to the database, on LOCAL disk, never a
    cloud-synced folder: `BRING_TO_WORK.md` already tells James to keep `data\\` out of
    OneDrive, and queuing this database's own size for upload as a "just in case" copy is
    the same mistake in a new place. `snapshot_db.py` refuses to overwrite an existing
    file, which is worth saying here rather than let that refusal be the first anyone hears
    of it. `os.name` picks the form: PowerShell on the Windows machine this script actually
    runs on, plain POSIX for testing here on the Mac -- and automatically under a test that
    monkeypatches `os.name`, since this builds `dst` with a plain regex over `str(db_path)`
    rather than a `pathlib` instance method.

    That is deliberate, not merely convenient for testing: `pathlib.Path(...)` (the
    FACTORY) dispatches on `os.name` and happily returns a working `WindowsPath` on this
    Mac when `os.name` reads "nt" -- but `.with_name()`, `.with_suffix()` and friends
    reconstruct via the concrete class directly (`type(self)(...)`), which raises
    `UnsupportedOperation: cannot instantiate 'WindowsPath' on your system` regardless of
    what `os.name` says (verified on this Python, 3.14). A path STRING is already correct
    for wherever `main()` actually runs it; this only ever needs to read that string, never
    walk it as a real filesystem path.
    """
    s = str(db_path)
    m = _NAME.match(s)
    prefix, name = (m.group("dir") or "", m.group("name")) if m else ("", s)
    if "." in name and not name.startswith("."):
        stem, _, suffix = name.rpartition(".")
        dst_name = f"{stem}_pre_tv_backfill.{suffix}"
    else:
        dst_name = f"{name}_pre_tv_backfill"
    dst = f"{prefix}{dst_name}"
    note = "    (refuses to overwrite -- if that name is already there, pick another)"
    if os.name == "nt":
        return f"    .\\.venv\\Scripts\\python scripts\\snapshot_db.py {s} {dst}\n{note}"
    return f"    python scripts/snapshot_db.py {s} {dst}\n{note}"


def _print_names(label: str, names: List[str]) -> None:
    print(f"{len(names):,} file(s) {label}:")
    shown = names[:20]
    for n in shown:
        print(f"    {n}")
    if len(names) > len(shown):
        print(f"    ... and {len(names) - len(shown):,} more")


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="backfill_increment_volts.py",
        description="Back-fill laser 1's TrimVolts curves onto stored Trim N passes.")
    parser.add_argument("db_path", nargs="?", default=None,
                        help="Path to the database to update (required).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Touch at most this many files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read and report; write nothing.")
    args = parser.parse_args(argv)

    if not args.db_path:
        # No DB path at all: refuse, and print the full usage/snapshot advice above
        # rather than argparse's one-line error -- this is the one case where the
        # operator has given us nothing to say "what it will touch" about yet.
        print(__doc__)
        return 2

    db_path = Path(args.db_path)
    if not db_path.exists():
        # Checked BEFORE DatabaseManager: constructing it CREATES an empty database file,
        # and back-filling a fresh empty one would report "0 candidates" as if it had
        # succeeded (the same landmine scripts/regrade_final_tests.py guards against).
        print(f"FATAL | no database at {db_path}")
        return 1

    from laser_trim_analyzer.database.manager import DatabaseManager
    db = DatabaseManager(db_path)

    print(f"Laser 1 (LTS) TrimVolts back-fill{'  [DRY RUN -- nothing will be written]' if args.dry_run else ''}")
    print(f"Database: {db_path}")
    print("This WRITES to that database: three columns of trim_passes only "
          "(increment_volts, increment_volts_first_row, increment_volts_truncated). "
          "Nothing else changes.")
    print("Snapshot first if you have not already:")
    print(_snapshot_hint(db_path))
    print()

    def _on_selected(candidates_total: int, candidate_files_total: int, files_run: int) -> None:
        print(f"Laser-1 Trim-N passes still missing their curves: {candidates_total:,} "
              f"across {candidate_files_total:,} file(s).")
        touching = f"This run will touch: {files_run:,} file(s)"
        if args.limit is not None:
            touching += f" (--limit {args.limit:,})"
        print(touching + "\n")

    def _progress(done: int, total: int, name: str, rate: Optional[float], eta_text: str) -> None:
        bits = [f"{done:,} of {total:,}"]
        if rate:
            bits.append(format_rate(rate))
        if eta_text:
            bits.append(eta_text)
        print(" · ".join(bits) + f"  ({name})")

    report = backfill(db, dry_run=args.dry_run, limit=args.limit,
                      progress=_progress, on_selected=_on_selected)

    verb = "Would fill" if args.dry_run else "Filled"
    print(f"\n{verb} {report.passes_filled:,} pass(es) across {report.files_updated:,} "
          f"file(s) in {format_clock(report.elapsed_s)}.")
    if report.passes_unverified:
        print(f"({report.passes_unverified:,} of those {report.passes_filled:,} could not "
              f"be checked against the workbook's own VOLTAGES sheet -- written anyway, as "
              f"before this check existed.)")
    if report.passes_unfilled:
        print(f"{report.passes_unfilled:,} candidate pass(es) left exactly as they were "
              f"(no TrimVolts sheet found, or it could not be read).")
    if report.passes_empty:
        print(f"{report.passes_empty:,} candidate pass(es) whose TrimVolts sheet holds no "
              f"reading at all -- nothing to capture, written as NULL (not counted as "
              f"filled); a later run selects them again.")
        _print_names("with a TrimVolts sheet that holds no reading", report.empty_sheets)
    if report.passes_disagreed:
        print(f"{report.passes_disagreed:,} candidate pass(es) REFUSED -- the workbook's "
              f"own VOLTAGES sheet disagrees with where this run would have placed them. "
              f"Left exactly as they were; a wrong position is worse than none.")
        _print_names("placement disagrees", report.disagreements)
    if report.missing_files:
        _print_names("missing", report.missing_files)
    if report.unreadable_files:
        _print_names("that could not be opened", report.unreadable_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
