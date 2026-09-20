"""Work out process findings for a database and print them, ranked.

    python scripts/refresh_findings.py "Work Files/dev_db/slice.db"
    python scripts/refresh_findings.py "Work Files/dev_db/slice.db" 8232-1 8340-1

The app does this by itself after every ingest that saves trim files. This is
for a database that was built another way (scripts/build_dev_db.py), or to
recompute after the analyzers change. It WRITES the two cache tables, so it
refuses data/analysis.db unless --production is given.
"""
import os
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _is_production(path: Path) -> bool:
    """True for the work database under ANY spelling -- compared by identity, not by text.

    On a case-insensitive volume (macOS here, NTFS at work) `data/Analysis.db` IS
    `data/analysis.db`, but resolve() keeps the case as typed, and a hard link shares no
    path with its target at all. os.path.samefile compares the file itself. The name rule
    is the second net: REPO is THIS checkout, so from a git worktree the path rule points at
    the worktree's empty data folder, never at the real database.
    """
    if path.name.casefold() == "analysis.db":
        return True
    prod = REPO / "data" / "analysis.db"
    try:
        return path.exists() and prod.exists() and os.path.samefile(path, prod)
    except OSError:
        return True          # cannot prove it is NOT the work database -- refuse


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--production"]
    if not args:
        print(__doc__)
        return 2
    path = Path(args[0])
    is_production = _is_production(path)
    if is_production and "--production" not in sys.argv:
        print(f"REFUSED: {path} is the work database (named analysis.db, or the same file as "
              "data/analysis.db). Run this on a COPY under another name. The app refreshes its own "
              "findings after every ingest; pass --production only if you mean to write the cache "
              "tables into the work database by hand.")
        return 2
    if not path.exists():
        print(f"no such database: {path}")
        return 2

    import logging
    logging.disable(logging.WARNING)
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    from laser_trim_analyzer.findings.engine import refresh_findings

    db = mgr.DatabaseManager(path)
    mgr._db_manager = db                      # BOTH globals: nothing on this path may open
    dbpkg._db_manager = db                    # the configured database by accident.
    stored = refresh_findings(db, args[1:] or None)
    ranked = db.get_process_findings()
    print(f"{stored} findings stored; {len(ranked)} in the ranked list\n")
    for i, f in enumerate(ranked, 1):
        upy = f.get("units_per_year")
        gain = f"{upy:,.0f} units a year" if upy is not None else "no gain claimed"
        print(f"{i:>3}. {f['model']}  ·  {f['title']}")
        print(f"     {f['category']}  ·  lever: {f['lever_label']} ({f['lead_time']})  ·  {gain}")
        print(textwrap.fill(f["summary"], 100, initial_indent="     ", subsequent_indent="     "))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
