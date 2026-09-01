"""The guard that keeps the test suite off the production database.

A test does not have to do anything wrong to reach `./data/analysis.db`: any
code path that calls `get_database()` gets the module-global manager, and when
that global is unset the default `DatabaseManager()` resolves to the real work
database and opens it READ-WRITE. These pin the autouse fixture in conftest
that removes that fall-through.
"""
from pathlib import Path

from laser_trim_analyzer.database import manager as mgr


def _production_db() -> Path:
    return (Path(__file__).resolve().parents[1] / "data" / "analysis.db").resolve()


def test_the_global_manager_is_not_the_production_database():
    """`get_database()` inside a test must never hand back the real DB."""
    got = Path(mgr.get_database().get_database_path()).resolve()
    assert got != _production_db(), (
        f"get_database() resolved to the PRODUCTION database at {got}. The "
        "conftest guard is not active — a test can now write to real data."
    )


def test_the_global_is_pre_seeded_so_nothing_falls_through():
    """Pre-seeded, not merely different: an unset global would reconstruct the
    default manager (the production path) the moment anything asked for it."""
    assert mgr._db_manager is not None


def test_the_guard_survives_a_reset_by_the_code_under_test():
    """`reset_database()` exists and production code may call it. If it does,
    the NEXT `get_database()` must still not reach production — which is only
    true while the test process's own config points somewhere safe."""
    mgr.reset_database()
    got = Path(mgr.get_database().get_database_path()).resolve()
    assert got != _production_db()
