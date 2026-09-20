"""The one rule for "is this path the owner's work database?" -- shared by every tool in scripts/
that opens its target read-write and therefore must never be aimed at it.

Compared by IDENTITY, not by text: on a case-insensitive volume `data/Analysis.db` is the same file
as `data/analysis.db` but Path.resolve() keeps the case as typed, and a hard link shares no path with
its target at all. `repo` is a parameter so the rule can be tested against a decoy checkout --
NO TEST MAY EVER NAME THE REAL data/analysis.db: the refusal is the code under test, so a regression
would open the real database before the test went red.
"""
import os
from pathlib import Path


def is_production_db(path, repo, *, by_name: bool = True) -> bool:
    """True when `path` is, or cannot be told apart from, `<repo>/data/analysis.db`.

    by_name=True also refuses ANY file called analysis.db (any case), wherever it is: `repo` is the
    checkout the script lives in, so from a git worktree the identity rule points at the worktree's
    empty data folder, never at the real database -- the name rule is what catches it there.
    Pass by_name=False only for a tool whose documented usage takes a COPY that is itself named
    analysis.db.
    """
    path, prod = Path(path), Path(repo) / "data" / "analysis.db"
    if by_name and path.name.casefold() == "analysis.db":
        return True
    try:
        return path.exists() and prod.exists() and os.path.samefile(path, prod)
    except OSError:
        return True          # cannot prove it is NOT the work database -- refuse
