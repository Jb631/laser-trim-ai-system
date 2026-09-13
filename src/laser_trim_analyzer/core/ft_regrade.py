"""Final-test grading — the ONE definition, and the driver that re-applies it.

THE DISPOSITION IS THE APP'S GRADE
----------------------------------
A final-test sheet carries its own PASSED/FAILED cell, and the app stores it
(``final_test_results.station_linearity_pass``) — as a REFERENCE. It is not the
disposition. The disposition is what it has always been: the analyzer's
per-point grade of the corrected error trace (``error + theory*k + offset``)
against the file's own limit columns, zero-tolerance, written to
``linearity_pass``. James, 2026-09-13: "i dont want to just copy the excel, i
want to grade the error and correct the offset but if cells should be ignored
then we should ignore those cells."

What changed is the last clause. The station grades a WINDOW of the sweep and
leaves the lead-in and run-out ungraded; the parser now recovers that window
(see ``FinalTestParser._graded_window``) and this module feeds the rows outside
it to the analyzer as excluded indices, merged with any ``exclude_points_ft``
the model spec already declares. Nothing else about the grade moves: same
analyzer, same arguments, same math.

WHY THE GRADING LIVES HERE AND NOT IN THE PROCESSOR
---------------------------------------------------
Two callers must produce byte-identical results or the database ends up with
two vintages of verdict in one column:

  * ``Processor._process_final_test`` — first-time grading, at ingest.
  * ``regrade_final_tests`` below — the repair pass for rows graded before the
    window existed, run once at the work machine against the real database.

So ``grade_ft_track`` is the single body both call. A second copy in the
re-grade driver is exactly the drift this file exists to prevent.

No Tk. ``progress`` is a plain callback and ``cancel`` a plain Event; the
driver runs on a worker thread and the caller posts to the UI itself.
"""
import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from laser_trim_analyzer.core.analyzer import parse_exclude_points
from laser_trim_analyzer.core.models import AnalysisStatus, TrackData

logger = logging.getLogger(__name__)


# ---- the shared grade -------------------------------------------------------

def out_of_window_indices(n_points: int,
                          window: Optional[Tuple[int, int]]) -> Set[int]:
    """The sweep indices the station did not grade.

    Empty when the file states no window — the app then grades the whole sweep,
    which is what it did before this existed.
    """
    if window is None or n_points <= 0:
        return set()
    low, high = window
    low = max(0, int(low))
    high = min(n_points - 1, int(high))
    return set(range(0, low)) | set(range(high + 1, n_points))


def merged_exclude_points(spec_exclude: Optional[str],
                          extra: Set[int]) -> Optional[str]:
    """One ``exclude_points`` JSON string carrying both sources.

    The per-model ``exclude_points_ft`` a user typed in Settings and the window
    the file itself declares are the same KIND of statement — "do not grade
    these indices" — so they go down the existing channel together rather than
    the analyzer growing a second one. Returns the spec string unchanged when
    there is nothing to add, so a file with no window produces the identical
    argument the analyzer received before.
    """
    if not extra:
        return spec_exclude
    indices = parse_exclude_points(spec_exclude) | set(extra)
    return json.dumps({"exclude": sorted(indices)})


def analyzer_errors(errors: Sequence) -> List[float]:
    """The error series in the form the analyzer's math accepts.

    A blank error cell is None in the parser's output (it is not a zero), but
    ``Analyzer._calculate_linearity`` builds the shifted trace with a plain
    ``e + offset``, which None cannot survive. NaN can, and means the same
    thing everywhere downstream: ``max_abs_measured`` skips it,
    ``_count_fail_points`` counts it as a fail unless excluded, and
    ``_calculate_optimal_offset`` already refuses to let it choose the offset
    for every other point. This is a representation change, not a policy one.
    """
    out: List[float] = []
    for value in errors or []:
        if value is None:
            out.append(float("nan"))
        else:
            out.append(value)
    return out


def grade_ft_track(analyzer, track: Dict[str, Any], ft_spec: Dict[str, Any], *,
                   model: Optional[str] = None,
                   ft_compensation: Optional[float] = None,
                   filename: str = "") -> TrackData:
    """Grade ONE final-test track and enrich `track` in place for the DB.

    Returns the TrackData the UI shows; mutates `track` with the values
    ``save_final_test`` / ``update_final_test_tracks`` persist (verdict, fail
    points, magnitude, offset, slope, linearity type).

    The verdict written here is the app's, corrected and per-point — the
    disposition. The station's own verdict rides alongside in the parser's
    reference fields and is never merged into it.
    """
    ft_linearity_type = ft_spec.get("linearity_type")

    # Handle None values explicitly (dict.get returns None if key exists with None value).
    # Format 2 FT files have no spec limits and the parser returns None — treat as
    # FAIL rather than silently calling unknown-status units PASS.
    linearity_pass = track.get("linearity_pass")
    if linearity_pass is None:
        logger.warning(
            f"FT track {track.get('track_id', '?')} of {filename}: "
            f"linearity_pass unknown (no spec limits) — defaulting to FAIL"
        )
        linearity_pass = False

    # Use analyzer for spec-aware optimization when we have error data
    positions = track.get("positions") or track.get("electrical_angles") or []
    errors = track.get("errors") or []
    upper_lims = track.get("upper_limits") or []
    lower_lims = track.get("lower_limits") or []

    if positions and errors and upper_lims and lower_lims:
        # Full analysis through analyzer
        track_dict = {
            "track_id": track.get("track_id", "default"),
            "positions": positions,
            "errors": analyzer_errors(errors),
            "upper_limits": upper_lims,
            "lower_limits": lower_lims,
            "travel_length": max(positions) - min(positions) if positions else 1.0,
            "linearity_spec": track.get("linearity_spec") or 0.01,
        }
        # Pass theory_values so the analyzer can run slope optimization
        # (adjusted = error + theory * k + offset)
        theory_vals = track.get("theory_values")
        if theory_vals:
            track_dict["theory_volts"] = theory_vals
        # The rows the station never graded join the per-model exclusions.
        ungraded = out_of_window_indices(len(errors), track.get("graded_window"))
        track_dict["exclude_points"] = merged_exclude_points(
            ft_spec.get("exclude_points"), ungraded)
        track_result = analyzer.analyze_track(
            track_dict,
            model=model,
            linearity_type=ft_linearity_type,
            angle_spec=ft_spec.get("angle_spec"),
            angle_tol=ft_spec.get("angle_tol"),
            angle_tol_type=ft_spec.get("angle_tol_type"),
            station_compensation=track.get("station_compensation") or ft_compensation,
        )

        # Enrich the raw parser track dict with spec-aware values so
        # save_final_test can persist them on final_test_tracks.
        track["optimal_offset"] = getattr(track_result, "optimal_offset", 0.0)
        track["optimal_slope"] = getattr(track_result, "optimal_slope", 0.0)
        track["linearity_type"] = (
            str(ft_linearity_type.value) if hasattr(ft_linearity_type, "value")
            else (str(ft_linearity_type) if ft_linearity_type else None)
        )
        # Overwrite the parser's raw-error fail count with the
        # analyzer's corrected-error count. Pass/fail is judged
        # on corrected errors (error + theory*k + offset), not raw,
        # so this is what should land in the DB.
        corrected_fail_points = getattr(track_result, "linearity_fail_points", None)
        if corrected_fail_points is not None:
            track["linearity_fail_points"] = corrected_fail_points
        corrected_lin_pass = getattr(track_result, "linearity_pass", None)
        if corrected_lin_pass is not None:
            track["linearity_pass"] = corrected_lin_pass
        corrected_lin_error = getattr(track_result, "linearity_error", None)
        if corrected_lin_error is not None:
            track["linearity_error"] = corrected_lin_error

        # A track with nothing left to grade is NOT a pass. The analyzer
        # counts zero fail points over zero graded indices and calls that
        # True, which is the correct answer to "how many failed" and the
        # wrong answer to "did it pass". This is reachable: a model spec
        # whose exclude_points_ft covers the whole window, or a file whose
        # limit columns are blank inside it.
        if _nothing_graded(len(errors), upper_lims, lower_lims,
                           track_dict["exclude_points"]):
            logger.warning(
                f"FT track {track.get('track_id', '?')} of {filename}: no "
                f"gradeable point inside the station's window — verdict NULL"
            )
            track["linearity_pass"] = None
            track["linearity_fail_points"] = 0
            track_result.linearity_pass = None
        return track_result

    # Minimal TrackData when no error data available
    track_data = TrackData(
        track_id=track.get("track_id", "default"),
        status=AnalysisStatus.PASS if linearity_pass else AnalysisStatus.FAIL,
        travel_length=1.0,
        linearity_spec=track.get("linearity_spec") or 0.01,
        sigma_gradient=0.0,
        sigma_threshold=0.01,
        sigma_pass=True,
        optimal_offset=0.0,
        linearity_error=track.get("linearity_error") or 0.0,
        linearity_pass=linearity_pass,
        linearity_fail_points=track.get("linearity_fail_points") or 0,
        linearity_type=ft_linearity_type,
        station_compensation=track.get("station_compensation") or ft_compensation,
        position_data=positions,
        error_data=analyzer_errors(errors),
    )

    # Enrich with identity correction so the columns are always
    # populated for every track.
    track["optimal_offset"] = 0.0
    track["optimal_slope"] = 0.0
    track["linearity_type"] = (
        str(ft_linearity_type.value) if hasattr(ft_linearity_type, "value")
        else (str(ft_linearity_type) if ft_linearity_type else None)
    )
    return track_data


def _nothing_graded(n_points: int, upper_limits, lower_limits,
                    exclude_json: Optional[str]) -> bool:
    """True when no index survives to be judged — mirrors _count_fail_points."""
    import numpy as np
    if not upper_limits or not lower_limits:
        return True
    excluded = parse_exclude_points(exclude_json)
    n = min(n_points, len(upper_limits), len(lower_limits))
    for i in range(n):
        if i in excluded:
            continue
        up, lo = upper_limits[i], lower_limits[i]
        if up is None or lo is None:
            continue
        if (isinstance(up, float) and np.isnan(up)) or \
           (isinstance(lo, float) and np.isnan(lo)):
            continue
        return False
    return True


def ft_reference_fields(track: Dict[str, Any]) -> Dict[str, Any]:
    """The station's own grading, as the DB columns name it.

    Reference only. Kept in one place so `save_final_test` and
    `update_final_test_tracks` cannot store different subsets of it.
    """
    window = track.get("graded_window")
    return {
        "station_flags": track.get("station_flags"),
        "station_fail_points": track.get("station_fail_points"),
        "graded_start": track.get("graded_start",
                                  window[0] if window else None),
        "graded_end": track.get("graded_end", window[1] if window else None),
        "ignore_start": track.get("ignore_start"),
        "ignore_end": track.get("ignore_end"),
    }


def graded_window_source(tracks: Sequence[Dict[str, Any]]) -> Optional[str]:
    """One source name for the file, for `final_test_results`.

    'flags' when any track found flags (the strongest evidence present),
    else 'ignore_cells', else 'all_rows'. Never None for a freshly graded
    row — NULL in that column is reserved to mean "graded before the window
    fix", which is what `count_legacy_ft_verdicts` counts.
    """
    sources = [t.get("graded_window_source") for t in tracks
               if t.get("graded_window_source")]
    for preferred in ("flags", "ignore_cells"):
        if preferred in sources:
            return preferred
    return "all_rows"


# ---- the re-grade driver ----------------------------------------------------

MISSING_FILE = "missing_file"
PARSE_ERROR = "parse_error"
REGRADED = "regraded"
UNCHANGED = "unchanged"


@dataclass
class RegradeOutcome:
    """What happened to one final_test_results row."""
    final_test_id: int
    model: str
    filename: str
    result: str
    before: Optional[bool] = None
    after: Optional[bool] = None
    detail: str = ""

    @property
    def changed(self) -> bool:
        return self.result == REGRADED and self.before != self.after


@dataclass
class RegradeReport:
    outcomes: List[RegradeOutcome] = field(default_factory=list)
    cancelled: bool = False
    applied: bool = False

    @property
    def examined(self) -> int:
        return len(self.outcomes)

    def _count(self, result: str) -> int:
        return sum(1 for o in self.outcomes if o.result == result)

    @property
    def missing(self) -> int:
        return self._count(MISSING_FILE)

    @property
    def errors(self) -> int:
        return self._count(PARSE_ERROR)

    @property
    def changed(self) -> int:
        return sum(1 for o in self.outcomes if o.changed)

    def transitions(self) -> Dict[str, int]:
        """PASS->FAIL / FAIL->PASS / ->NULL / NULL-> counts."""
        out = {"pass_to_fail": 0, "fail_to_pass": 0, "to_null": 0, "from_null": 0}
        for o in self.outcomes:
            if not o.changed:
                continue
            if o.after is None:
                out["to_null"] += 1
            elif o.before is None:
                out["from_null"] += 1
            elif o.before and not o.after:
                out["pass_to_fail"] += 1
            elif o.after and not o.before:
                out["fail_to_pass"] += 1
        return out

    def by_model(self) -> Dict[str, Dict[str, int]]:
        models: Dict[str, Dict[str, int]] = {}
        for o in self.outcomes:
            row = models.setdefault(o.model or "unknown", {
                "examined": 0, "changed": 0, "pass_to_fail": 0,
                "fail_to_pass": 0, "to_null": 0, "from_null": 0, "missing": 0})
            row["examined"] += 1
            if o.result == MISSING_FILE:
                row["missing"] += 1
            if not o.changed:
                continue
            row["changed"] += 1
            if o.after is None:
                row["to_null"] += 1
            elif o.before is None:
                row["from_null"] += 1
            elif o.before and not o.after:
                row["pass_to_fail"] += 1
            else:
                row["fail_to_pass"] += 1
        return models

    def summary(self) -> str:
        """One line for a status label. Says what did NOT happen, too."""
        if not self.outcomes:
            return "No final-test records need re-grading."
        t = self.transitions()
        verb = "Re-graded" if self.applied else "Would re-grade"
        parts = [f"{verb} {self.examined} record(s): {self.changed} verdict "
                 f"change(s) — {t['pass_to_fail']} PASS→FAIL, "
                 f"{t['fail_to_pass']} FAIL→PASS, {t['to_null']} →not graded"]
        if self.missing:
            parts.append(f"{self.missing} source file(s) unreachable")
        if self.errors:
            parts.append(f"{self.errors} failed to re-parse")
        if self.cancelled:
            parts.append("stopped early")
        return "; ".join(parts) + "."


def regrade_final_tests(
    db,
    *,
    progress: Optional[Callable[[int, int, str], None]] = None,
    cancel: Optional[threading.Event] = None,
    workers: int = 4,
    only_legacy: bool = True,
    apply: bool = False,
    limit: Optional[int] = None,
) -> RegradeReport:
    """Re-parse and re-grade stored final tests through `grade_ft_track`.

    Args:
        db: DatabaseManager.
        progress: progress(done, total, filename). Exceptions are swallowed —
            a progress callback must never kill a repair run.
        cancel: cooperative stop, checked between files. Everything already
            written stays written; the run is resumable because it is
            idempotent.
        workers: parse/analyze threads. The writes are serialized by the
            manager's own write lock.
        only_legacy: only rows whose `graded_window_source` IS NULL, i.e.
            graded before the window fix. False re-grades everything.
        apply: False (default) computes every verdict and writes NOTHING —
            the dry run the script prints its table from.
        limit: cap the number of rows examined (dry-run sampling).

    Returns:
        RegradeReport. Per-record problems are outcomes, never exceptions;
        only a failure to reach the database itself propagates.
    """
    from concurrent.futures import ThreadPoolExecutor
    from laser_trim_analyzer.core.analyzer import Analyzer
    from laser_trim_analyzer.core.final_test_parser import FinalTestParser
    from laser_trim_analyzer.core.ingest_run import INGEST_SWITCH_INTERVAL

    rows = db.get_final_tests_for_regrade(only_legacy=only_legacy, limit=limit)
    report = RegradeReport(applied=apply)
    if not rows:
        return report

    parser = FinalTestParser()
    analyzer = Analyzer()
    total = len(rows)
    done_lock = threading.Lock()
    done = 0

    def _tick(name: str) -> None:
        nonlocal done
        with done_lock:
            done += 1
            current = done
        if progress is None:
            return
        try:
            progress(current, total, name)
        except Exception:  # noqa: BLE001 - never let the UI kill the run
            logger.debug("progress callback raised; continuing", exc_info=True)

    def _one(row: Dict[str, Any]) -> RegradeOutcome:
        name = row.get("filename") or f"id={row.get('id')}"
        model = row.get("model") or "unknown"
        before = row.get("linearity_pass")
        raw_path = row.get("file_path")
        path = Path(raw_path) if raw_path else None
        try:
            if path is None or not path.exists():
                return RegradeOutcome(row["id"], model, name, MISSING_FILE,
                                      before, before,
                                      str(raw_path or "no stored path"))
            parsed = parser.parse_file(path)
            tracks = parsed.get("tracks") or []
            if not tracks:
                return RegradeOutcome(row["id"], model, name, PARSE_ERROR,
                                      before, before, "parser returned no tracks")
            spec = db.resolve_spec_for_ft(model, row.get("serial")) or {}
            ft_spec = {
                "linearity_type": spec.get("linearity_type"),
                "angle_spec": spec.get("electrical_angle"),
                "angle_tol": spec.get("electrical_angle_tol"),
                "angle_tol_type": spec.get("electrical_angle_tol_type"),
                "exclude_points": (spec.get("exclude_points_ft")
                                   or spec.get("exclude_points")),
            }
            compensation = (parsed.get("metadata") or {}).get("station_compensation")
            for track in tracks:
                grade_ft_track(analyzer, track, ft_spec, model=model,
                               ft_compensation=compensation, filename=name)
            after = db.resolve_final_test_linearity_pass(
                parsed.get("test_results") or {}, tracks)
            if apply:
                db.apply_final_test_regrade(
                    row["id"], tracks, parsed.get("test_results") or {})
            return RegradeOutcome(row["id"], model, name, REGRADED, before, after)
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the batch
            logger.error(f"Re-grade failed for {name}: {e}")
            return RegradeOutcome(row["id"], model, name, PARSE_ERROR,
                                  before, before, f"{type(e).__name__}: {e}")
        finally:
            _tick(name)

    # Same GIL courtesy the ingest uses: several Excel parsers at once will
    # starve a Tk window of the interpreter lock for hundreds of milliseconds
    # at a time unless CPython is asked to hand it around more often. Scoped to
    # this run and restored in `finally`, exactly as ingest_run does it.
    saved_interval = sys.getswitchinterval()
    sys.setswitchinterval(INGEST_SWITCH_INTERVAL)
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = []
            for row in rows:
                if cancel is not None and cancel.is_set():
                    report.cancelled = True
                    break
                futures.append(pool.submit(_one, row))
            for future in futures:
                report.outcomes.append(future.result())
    finally:
        sys.setswitchinterval(saved_interval)

    if cancel is not None and cancel.is_set():
        report.cancelled = True
    return report
