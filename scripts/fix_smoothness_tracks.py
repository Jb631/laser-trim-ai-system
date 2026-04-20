"""
One-time repair: re-import per-position smoothness data for records that
were created before save_smoothness_result wrote the per-track arrays.

Symptom this fixes:
- Smoothness page lists 31 results but the chart panel is blank when you
  click on any of them.
- smoothness_tracks table has 0 rows even though smoothness_results has data.

Usage (from a shell, with the project venv active):
    python scripts/fix_smoothness_tracks.py

The script:
  1. Queries the DB for any SmoothnessResult that has 0 rows in smoothness_tracks.
  2. For each one, re-parses the original .xlsx via the smoothness parser.
  3. Writes the per-track data back via DatabaseManager.update_smoothness_tracks.

If the source file no longer exists at the recorded file_path, the script
just skips that record and reports it at the end.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make src importable when run from the repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from laser_trim_analyzer.database import get_database  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fix_smoothness_tracks")


def _load_smoothness_parser():
    """Return a parse(file_path) callable that yields a list of track dicts.

    Wraps SmoothnessParser.parse_file so the caller gets just the tracks
    list (which is all update_smoothness_tracks needs).
    """
    try:
        from laser_trim_analyzer.core.smoothness_parser import SmoothnessParser  # type: ignore
    except Exception as e:
        logger.error("Could not import SmoothnessParser: %s", e)
        return None

    sp = SmoothnessParser()

    def _parse(p):
        result = sp.parse_file(p)
        # parse_file returns {"metadata": ..., "tracks": [...], "file_hash": ...}
        if isinstance(result, dict):
            return result.get("tracks") or []
        return result or []

    return _parse


def main() -> int:
    db = get_database()
    parser = _load_smoothness_parser()
    if parser is None:
        logger.error(
            "Could not locate the smoothness parser. Look in "
            "src/laser_trim_analyzer/core/ for the smoothness parser module "
            "and update _load_smoothness_parser() in this script."
        )
        return 2

    missing = db.get_smoothness_files_missing_tracks()
    logger.info("Found %d smoothness records with 0 tracks", len(missing))
    if not missing:
        return 0

    fixed, skipped, failed = 0, 0, 0
    skipped_records = []

    for rec in missing:
        path_str = rec.get("file_path")
        rid = rec["id"]
        if not path_str:
            skipped += 1
            skipped_records.append((rid, rec.get("filename"), "no file_path"))
            continue
        p = Path(path_str)
        if not p.exists():
            skipped += 1
            skipped_records.append((rid, rec.get("filename"), f"missing on disk: {p}"))
            continue

        try:
            tracks = parser(str(p))
        except Exception as e:
            failed += 1
            logger.warning("Parse failed for ID %s (%s): %s", rid, p.name, e)
            continue

        if not tracks:
            failed += 1
            logger.warning("Parser returned no tracks for ID %s (%s)", rid, p.name)
            continue

        ok = db.update_smoothness_tracks(rid, tracks)
        if ok:
            fixed += 1
            logger.info("Repaired ID %s (%s) with %d tracks", rid, p.name, len(tracks))
        else:
            failed += 1

    logger.info(
        "Done. Repaired=%d  Skipped=%d  Failed=%d", fixed, skipped, failed
    )
    if skipped_records:
        logger.info("Skipped records:")
        for rid, fn, reason in skipped_records:
            logger.info("  - id=%s file=%s reason=%s", rid, fn, reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
