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

Avoids DB writes by stubbing save_final_test.

Usage:
    python scripts/parser_audit/snapshot.py [output_path]
"""
import sys
import json
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
from laser_trim_analyzer.database import get_database


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


def build_snapshot(work_root: Path) -> dict:
    db = get_database()
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
