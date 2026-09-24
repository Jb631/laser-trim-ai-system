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
Parameters` (and `Track Parameters`, if present) sheet for the three ignored-point/reading-
count fields, and each `TrimVolts N` sheet a candidate `Trim N` pass actually needs. It never
re-reads a `Trim N` sheet itself -- nothing on positions/errors/limits/recipe/etc. is touched,
on this row or any other.

A pass whose file has no `TrimVolts{N}` sheet (the one-in-4,972 touch-up case Task 4
measured), or whose sheet exists but fails to read, is left exactly as it was: NULL,
indistinguishable from "not yet attempted". That is Task 4's own representation, not
something this script can improve on without changing that ruling -- so a stopped-short
handful of files will be re-opened, harmlessly, on every future run. See the module's test
file for how small that set is expected to be.

Unlike the QA harnesses (`chart_qa_render_all.py`, `app_qa_sweep.py`), this script does NOT
refuse `data/analysis.db` -- filling in three columns of your own database, from your own
files, is the whole point, the same as `regrade_final_tests.py` and the other `backfill_*.py`
scripts in this folder. TAKE A SNAPSHOT FIRST anyway; the columns this writes cannot be
recovered from the database itself if something goes wrong reading a share full of files over
several hours:

    python scripts/snapshot_db.py data\\analysis.db "$env:OneDrive\\analysis_pre_tv_backfill.db"

Usage
-----
    python scripts/backfill_increment_volts.py DB_PATH               # writes
    python scripts/backfill_increment_volts.py DB_PATH --dry-run     # reads and reports only
    python scripts/backfill_increment_volts.py DB_PATH --limit 2000  # first N files only

DB_PATH is REQUIRED -- there is no default, so this can never land on the wrong database
through an empty command line. Progress prints every 200 files (files/second and an ETA);
writes commit every 200 files, so a stopped run (Ctrl-C, a dropped VPN, going home for the
night) keeps everything already committed. Running the same command again continues: it
re-selects exactly the passes still missing their curves, oldest file first (`analysis_
results.id` ascending -- this database's own established meaning of "oldest first" for a
resumable repair pass; see `DatabaseManager.get_final_tests_for_regrade`'s docstring).
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

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
        self.files_missing = 0
        self.files_unreadable = 0
        self.missing_files: List[str] = []
        self.unreadable_files: List[str] = []
        self.elapsed_s = 0.0

    def __repr__(self) -> str:      # pragma: no cover -- debugging aid only
        return (f"BackfillReport(dry_run={self.dry_run}, files_run={self.files_run}, "
                f"files_updated={self.files_updated}, passes_filled={self.passes_filled}, "
                f"passes_unfilled={self.passes_unfilled}, files_missing={self.files_missing}, "
                f"files_unreadable={self.files_unreadable})")


# ---------------------------------------------------------------------------
# Selection: exactly the candidates Task 4 could not have written yet.
# ---------------------------------------------------------------------------

def _select_candidates(db) -> List[FileWork]:
    """Every laser-1 `Trim N` pass with `increment_volts IS NULL`, grouped one entry per
    file, oldest file first.

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
            .order_by(AnalysisResult.id.asc(), TrimPass.id.asc())
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


def _capture_file(file_path: Optional[str],
                  sheets_needed: List[str]) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """Open `file_path` read-only and read exactly what is needed for `sheets_needed`
    (each a `trim_passes.sheet` value, e.g. "Trim 1").

    Returns (captures, problem):
      captures: {sheet: {increment_volts, increment_volts_first_row,
                 increment_volts_truncated}} -- present only for a name that really is a
                 `Trim N` sheet AND has a readable `TrimVolts N` companion. A name missing
                 from the result is left exactly as it was in the database (see module
                 docstring) -- never guessed, never written as an empty capture.
      problem: None, "missing" (no such file here) or "unreadable" (the file exists but the
               workbook could not be opened, or something inside it raised unexpectedly).
               Never raises.
    """
    if not file_path:
        return {}, "missing"
    path = Path(file_path)
    try:
        exists = path.exists()
    except OSError:
        exists = False
    if not exists:
        return {}, "missing"

    try:
        opened = pd.ExcelFile(path)
    except Exception:
        logger.debug("backfill: could not open %s", file_path, exc_info=True)
        return {}, "unreadable"

    try:
        # A context manager, like every other ExcelFile in this codebase (parser.py's
        # _workbook): closes the underlying handle on the way out instead of leaving it to
        # the garbage collector across tens of thousands of files over several hours.
        with opened as xl:
            setup = _read_trim_setup(xl)
            first_row, window = _tp.increment_volts_frame(setup)
            trimvolts = _tp.trimvolts_sheets(xl.sheet_names)
            out: Dict[str, Dict[str, Any]] = {}
            for sheet in sheets_needed:
                m = ExcelParser._TRIM_N_RE.match(sheet.strip())
                if not m:
                    continue
                tv_sheet = trimvolts.get(int(m.group(1)))
                if tv_sheet is None:
                    logger.debug("backfill: no TrimVolts sheet beside %r in %s", sheet, file_path)
                    continue
                try:
                    df = pd.read_excel(xl, sheet_name=tv_sheet, header=None)
                except Exception:
                    logger.debug("backfill: could not read %r beside %r in %s",
                                tv_sheet, sheet, file_path, exc_info=True)
                    continue
                out[sheet] = _tp.read_increment_volts(df, first_row, window)
            return out, None
    except Exception:
        # Nothing inside core.trim_passes is documented to raise, but a file this script
        # has never seen before gets no benefit of the doubt: one bad workbook must cost
        # only itself, never the run.
        logger.warning("backfill: unexpected failure reading %s", file_path, exc_info=True)
        return {}, "unreadable"


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
                    # written unconditionally, exactly as Task 4's own writer does.
                    row.increment_volts = cap["increment_volts"] or sql_null()
                    row.increment_volts_first_row = cap["increment_volts_first_row"]
                    row.increment_volts_truncated = cap["increment_volts_truncated"]
        pending.clear()
        batch_files = 0

    for i, work in enumerate(files):
        sheets_needed = [p.sheet for p in work.passes]
        captures, problem = _capture_file(work.file_path, sheets_needed)
        # Progress wants the short, friendly name; a missing/unreadable report wants the
        # actual PATH -- that is the actionable fact (which share location is unreachable),
        # and two files can share a bare filename in different folders.
        display_name = work.filename or work.file_path or f"analysis {work.analysis_id}"
        problem_name = work.file_path or work.filename or f"analysis {work.analysis_id}"

        if problem == "missing":
            report.files_missing += 1
            report.missing_files.append(problem_name)
        elif problem == "unreadable":
            report.files_unreadable += 1
            report.unreadable_files.append(problem_name)
        else:
            touched = False
            for p in work.passes:
                cap = captures.get(p.sheet)
                if cap is None:
                    report.passes_unfilled += 1
                    continue
                pending.append((p.pass_id, cap))
                report.passes_filled += 1
                touched = True
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

def _snapshot_hint(db_path: Path) -> str:
    return f"    python scripts/snapshot_db.py {db_path} {db_path}.snapshot-<today>.db"


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
    if report.passes_unfilled:
        print(f"{report.passes_unfilled:,} candidate pass(es) left exactly as they were "
              f"(no TrimVolts sheet found, or it could not be read).")
    if report.missing_files:
        _print_names("missing", report.missing_files)
    if report.unreadable_files:
        _print_names("that could not be opened", report.unreadable_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
