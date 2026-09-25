"""Build a deterministic snapshot of parser outputs for every sample file.

Used as a regression baseline: after each parser fix, re-run and diff against
the baseline. Only files in the targeted bug class should change.

Captures per file/track:
  - n_points  (length of position_data)
  - linearity_error  (rounded to 6 dp)
  - linearity_spec   (rounded to 6 dp)
  - sigma_gradient   (rounded to 6 dp)
  - linearity_fail_points
  - linearity_pass
  - overall_status

Runs on a throwaway database of its own: the processor reaches one through
get_database() (spec lookups, and it saves smoothness files and marks refused
ones), and the app's default -- the production database -- is refused outside
the app. That throwaway database starts empty, so before 2026-09-25 model_specs
was empty here too, and the snapshot's spec-aware analysis (linearity_type,
angle_spec/tol/tol_type, exclude_points -- see core/processor.py's
_get_spec_for_analysis) silently differed from a run beside a populated
database. Measured on real files: 8340 and 8340-3 (electrical_angle_tol_type=
'min') shift linearity_error/linearity_fail_points once their spec is loaded,
because 'min' grants a k allowance even with no explicit angle_tol (see
core/analyzer.py::_k_bounds_from_angle_tol). So this now loads model_specs from
the CONFIGURED DEFAULT database into the throwaway one first -- read-only
(sqlite3, mode=ro), never opened read-write, which stays the entire point of
the 2026-09-24 guard. No default database file (or one that is not a valid
database, or predates the model_specs table) leaves the throwaway database
exactly as empty as before -- never raises. save_final_test is still stubbed,
so nothing is written even there.

Usage:
    python scripts/parser_audit/snapshot.py [output_path]
"""
import sys
import json
import sqlite3
import tempfile
import warnings
import logging
from pathlib import Path

# Project src
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

from laser_trim_analyzer.core.processor import Processor
from laser_trim_analyzer.core.parser import detect_file_type
from laser_trim_analyzer.database import manager as mgr
import laser_trim_analyzer.database as dbpkg


PRECISION = 6  # decimal places for float comparison


def round_or_none(v, dp=PRECISION):
    if v is None:
        return None
    try:
        return round(float(v), dp)
    except (TypeError, ValueError):
        return None


def snapshot_track(track):
    return {
        "track_id": track.track_id,
        "n_points": len(track.position_data) if track.position_data else 0,
        "linearity_error": round_or_none(track.linearity_error),
        "linearity_spec": round_or_none(track.linearity_spec),
        "sigma_gradient": round_or_none(track.sigma_gradient),
        "linearity_fail_points": track.linearity_fail_points or 0,
        "linearity_pass": track.linearity_pass,
        "status": track.status.value if track.status else None,
    }


def _default_database_path() -> Path:
    """Where DatabaseManager() would open, per THIS process's config -- the
    SAME resolution DatabaseManager.__init__ uses (mgr.get_config()), so a
    caller that redirects the default for the guard (as the tests do)
    redirects this too."""
    return Path(mgr.get_config().database.path)


def load_model_specs_from_default_database(db) -> int:
    """Copy every `model_specs` row from the CONFIGURED DEFAULT database into
    `db` -- read-only, so the snapshot sees the same specs a pre-guard run
    would have (silently, against the real default) without ever opening it
    read-write, which stays the entire point of the 2026-09-24 guard.

    Returns how many rows were copied. 0 or the default path not existing, not
    a valid sqlite file, or predating the model_specs table -- never raises;
    `db` is left exactly as it was.
    """
    default_path = _default_database_path()
    if not default_path.exists():
        return 0
    try:
        con = sqlite3.connect(f"file:{default_path}?mode=ro", uri=True)
        try:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM model_specs").fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return 0
    for row in rows:
        data = dict(row)
        data.pop("id", None)
        data.pop("created_at", None)
        data.pop("updated_at", None)
        db.save_model_spec(data)
    return len(rows)


def build_snapshot(work_root: Path) -> dict:
    injected = mgr._db_manager
    injected_pkg = getattr(dbpkg, "_db_manager", None)
    with tempfile.TemporaryDirectory(prefix="parser_audit_") as scratch:
        db = mgr.DatabaseManager(Path(scratch) / "snapshot.db")
        mgr._db_manager = db
        dbpkg._db_manager = db
        n_specs = load_model_specs_from_default_database(db)
        print(f"Loaded {n_specs} model_specs row(s) from the default database "
              f"(read-only)", file=sys.stderr)
        try:
            return _snapshot(work_root, db)
        finally:
            mgr._db_manager = injected          # whatever was injected before, even None
            dbpkg._db_manager = injected_pkg
            db.close()


def _snapshot(work_root: Path, db) -> dict:
    orig_save = db.save_final_test
    db.save_final_test = lambda **kw: None
    try:
        proc = Processor(use_ml=False)
        files = sorted([*work_root.rglob("*.xls"), *work_root.rglob("*.xlsx")])
        snapshot = {}
        skipped = 0
        for i, f in enumerate(files):
            rel = str(f.relative_to(work_root))
            ft = detect_file_type(f)
            if ft == "non_trim":
                snapshot[rel] = {"file_type": ft, "skipped": True}
                skipped += 1
                continue
            try:
                r = proc.process_file(f, generate_plots=False)
            except Exception as e:
                snapshot[rel] = {
                    "file_type": ft,
                    "exception": f"{type(e).__name__}: {str(e)[:200]}",
                }
                continue
            if r is None:
                snapshot[rel] = {"file_type": ft, "result_none": True}
                continue
            snapshot[rel] = {
                "file_type": ft,
                "overall_status": r.overall_status.value if r.overall_status else None,
                "data_quality": r.data_quality,
                "tracks": [snapshot_track(t) for t in (r.tracks or [])],
            }
            if (i + 1) % 200 == 0:
                print(f"  ... {i+1}/{len(files)}", flush=True)
        return {
            "total_files": len(files),
            "skipped_non_trim": skipped,
            "files": snapshot,
        }
    finally:
        db.save_final_test = orig_save


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "scripts" / "parser_audit" / "baseline.json"
    )
    work_root = ROOT / "Work Files"
    if not work_root.exists():
        print(f"ERROR: {work_root} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Snapshotting parser outputs from: {work_root}")
    print(f"Output: {out_path}\n")
    snap = build_snapshot(work_root)
    out_path.write_text(json.dumps(snap, indent=2, default=str))
    print(f"\nSnapshot saved: {snap['total_files']} files "
          f"({snap['skipped_non_trim']} non_trim skipped)")
    print(f"Path: {out_path}")


if __name__ == "__main__":
    main()
