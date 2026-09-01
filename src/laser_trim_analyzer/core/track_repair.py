"""Fix Missing Tracks — the one repair driver, shared by the V5 and V6 UIs.

Re-parses records whose track measurements are missing (see
DatabaseManager.get_trim_records_missing_tracks / get_final_tests_missing_tracks)
and writes the recovered tracks back.

No Tk, no widgets, no page imports: the caller supplies a `progress` callback
and runs this on a worker thread. That keeps the V5 Compare page and the V6
Settings maintenance section on identical behaviour instead of two drifting
copies of the same loop.

Source files live on a network share (\\\\192.168.66.9\\...). Off the work
network every record is unreachable, which is the NORMAL case on a dev
machine — so an unreachable source is a reported per-record outcome, never a
silent skip and never a reason to abort the batch.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Local fallbacks for records whose stored file_path points at the share.
_TRIM_FALLBACK_DIRS = (
    "test_files/System A test files",
    "test_files/System B test files",
    "test_files/Final Test files",
)
_FT_FALLBACK_DIRS = ("test_files/Final Test files",)

# Outcome kinds.
REPAIRED = "repaired"
SOURCE_UNREACHABLE = "source_unreachable"
NO_TRACKS_PARSED = "no_tracks_parsed"
WRITE_FAILED = "write_failed"
PARSE_ERROR = "parse_error"


@dataclass
class RepairOutcome:
    """What happened to one record. `result` is one of the constants above."""
    kind: str           # "final_test" | "trim"
    record_id: int
    filename: str
    result: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.result == REPAIRED


@dataclass
class RepairReport:
    outcomes: List[RepairOutcome] = field(default_factory=list)

    def _count(self, kind: Optional[str] = None, result: Optional[str] = None) -> int:
        return sum(1 for o in self.outcomes
                   if (kind is None or o.kind == kind)
                   and (result is None or o.result == result))

    @property
    def examined(self) -> int:
        return len(self.outcomes)

    @property
    def repaired(self) -> int:
        return self._count(result=REPAIRED)

    @property
    def unreachable(self) -> int:
        return self._count(result=SOURCE_UNREACHABLE)

    @property
    def failed(self) -> int:
        """Everything that was reachable but did not yield usable tracks."""
        return self.examined - self.repaired - self.unreachable

    def counts_by_kind(self, kind: str) -> Dict[str, int]:
        return {"examined": self._count(kind=kind),
                "repaired": self._count(kind=kind, result=REPAIRED),
                "unreachable": self._count(kind=kind, result=SOURCE_UNREACHABLE)}

    def summary(self) -> str:
        """One line for a status label. Names the unreachable count explicitly
        so 'nothing happened' never reads as 'nothing was wrong'."""
        if not self.outcomes:
            return "No records need fixing."
        ft, trim = self.counts_by_kind("final_test"), self.counts_by_kind("trim")
        parts = [f"Repaired {self.repaired} of {self.examined} "
                 f"(FT {ft['repaired']}/{ft['examined']}, "
                 f"Trim {trim['repaired']}/{trim['examined']})"]
        if self.unreachable:
            parts.append(f"{self.unreachable} source file(s) unreachable — "
                         f"run this on the work network")
        if self.failed:
            parts.append(f"{self.failed} failed to re-parse")
        return "; ".join(parts) + "."


def _resolve_source(record: Dict[str, Any], fallback_dirs) -> Optional[Path]:
    """Stored path first, then local test_files copies. None = unreachable."""
    raw = record.get("file_path")
    if raw:
        try:
            candidate = Path(raw)
            if candidate.exists():
                return candidate
        except (OSError, ValueError):
            pass  # malformed/UNC path off-network — fall through to fallbacks
    model, filename = record.get("model") or "", record.get("filename") or ""
    if filename:
        for base in fallback_dirs:
            alt = Path(base) / model / filename
            try:
                if alt.exists():
                    return alt
            except (OSError, ValueError):
                continue
    return None


def repair_missing_tracks(
    db=None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    linked_only: bool = True,
) -> RepairReport:
    """Re-parse every record with missing tracks and store what comes back.

    Args:
        db: DatabaseManager. Defaults to the shared instance.
        progress: called as progress(done, total, phase) where phase is
            "Final Test" or "Trim". Exceptions from it are swallowed — a
            progress callback must never kill a repair run.
        linked_only: passed through to get_trim_records_missing_tracks.

    Returns:
        RepairReport — one RepairOutcome per record examined. This function
        does not raise for per-record problems; only a failure to reach the
        database itself propagates.
    """
    from laser_trim_analyzer.core.final_test_parser import FinalTestParser
    from laser_trim_analyzer.core.processor import Processor

    if db is None:
        from laser_trim_analyzer.database.manager import get_database
        db = get_database()

    report = RepairReport()

    def _tick(done: int, total: int, phase: str) -> None:
        if progress is None:
            return
        try:
            progress(done, total, phase)
        except Exception:  # noqa: BLE001 - never let the UI kill the repair
            logger.debug("progress callback raised; continuing repair", exc_info=True)

    ft_parser = FinalTestParser()
    trim_processor = Processor(use_ml=False)

    # ---- Phase 1: Final Test records -------------------------------------
    ft_missing = db.get_final_tests_missing_tracks()
    for i, record in enumerate(ft_missing):
        _tick(i + 1, len(ft_missing), "Final Test")
        name = record.get("filename") or f"id={record.get('id')}"
        path = _resolve_source(record, _FT_FALLBACK_DIRS)
        if path is None:
            report.outcomes.append(RepairOutcome(
                "final_test", record["id"], name, SOURCE_UNREACHABLE,
                str(record.get("file_path") or "no stored path")))
            continue
        try:
            tracks = (ft_parser.parse_file(path) or {}).get("tracks") or []
            if not tracks:
                report.outcomes.append(RepairOutcome(
                    "final_test", record["id"], name, NO_TRACKS_PARSED,
                    "parser returned no tracks"))
            elif db.update_final_test_tracks(record["id"], tracks):
                report.outcomes.append(RepairOutcome(
                    "final_test", record["id"], name, REPAIRED,
                    f"{len(tracks)} track(s)"))
            else:
                report.outcomes.append(RepairOutcome(
                    "final_test", record["id"], name, WRITE_FAILED,
                    "database rejected the update"))
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the batch
            logger.error(f"Error fixing Final Test {name}: {e}")
            report.outcomes.append(RepairOutcome(
                "final_test", record["id"], name, PARSE_ERROR, f"{type(e).__name__}: {e}"))

    # ---- Phase 2: Trim records -------------------------------------------
    # Some "Trim" rows genuinely point at Final Test files (historical
    # mis-classification), so fall back to the FT parser when the trim
    # processor yields nothing.
    trim_missing = db.get_trim_records_missing_tracks(linked_only=linked_only)
    for i, record in enumerate(trim_missing):
        _tick(i + 1, len(trim_missing), "Trim")
        name = record.get("filename") or f"id={record.get('id')}"
        path = _resolve_source(record, _TRIM_FALLBACK_DIRS)
        if path is None:
            report.outcomes.append(RepairOutcome(
                "trim", record["id"], name, SOURCE_UNREACHABLE,
                str(record.get("file_path") or "no stored path")))
            continue
        try:
            analysis = trim_processor.process_file(path)
            if analysis and analysis.tracks:
                ok = db.update_trim_tracks(record["id"], analysis.tracks)
                report.outcomes.append(RepairOutcome(
                    "trim", record["id"], name,
                    REPAIRED if ok else WRITE_FAILED,
                    f"{len(analysis.tracks)} track(s)" if ok
                    else "database rejected the update"))
                continue

            ft_tracks = (ft_parser.parse_file(path) or {}).get("tracks") or []
            if ft_tracks and db.update_trim_tracks_from_final_test(record["id"], ft_tracks):
                report.outcomes.append(RepairOutcome(
                    "trim", record["id"], name, REPAIRED,
                    f"{len(ft_tracks)} track(s) via FT parser"))
            else:
                report.outcomes.append(RepairOutcome(
                    "trim", record["id"], name, NO_TRACKS_PARSED,
                    "neither parser returned tracks"))
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the batch
            logger.error(f"Error fixing Trim {name}: {e}")
            report.outcomes.append(RepairOutcome(
                "trim", record["id"], name, PARSE_ERROR, f"{type(e).__name__}: {e}"))

    return report
