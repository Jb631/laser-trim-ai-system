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


def test_an_explicit_production_path_is_refused_not_redirected():
    """Redirecting the DEFAULT path leaves a hole: `__init__` only consults
    `get_config()` when the path argument is None, so a test that builds the
    production path by hand walks straight through. One did — see
    `test_5_8_2026_bugfixes.py`. Naming that file is a bug in the test, so it
    raises rather than being quietly pointed somewhere safe."""
    import pytest

    with pytest.raises(RuntimeError, match="production database"):
        mgr.DatabaseManager(_production_db())


def test_the_whole_production_data_directory_is_refused():
    """Not just analysis.db — the backups beside it are just as real."""
    import pytest

    beside_it = _production_db().parent / "analysis.db.bak-2026-08-30-pre-linspec-fix"
    with pytest.raises(RuntimeError, match="production database"):
        mgr.DatabaseManager(beside_it)


def test_an_ordinary_isolated_database_still_works(tmp_path):
    """The guard must refuse the work database without making the normal
    `DatabaseManager(tmp_path / ...)` that most tests use any harder."""
    db = mgr.DatabaseManager(tmp_path / "fine.db")
    assert Path(db.get_database_path()) == tmp_path / "fine.db"


def test_a_bare_config_does_not_resolve_to_the_production_database():
    """Patching `manager.get_config` fixes the manager and nothing else. Every
    default data path is built from `config.get_app_directory()`, so a bare
    `Config()` anywhere in a test would still hand out the production path —
    and something would eventually open it. The guard patches that choke point
    instead, so the default is safe at the source."""
    from laser_trim_analyzer.config import Config

    got = Path(Config().database.path).resolve()
    assert got != _production_db(), (
        f"a bare Config() resolved to the PRODUCTION database at {got}"
    )


def test_the_ml_models_directory_is_not_in_production_either():
    """Same choke point, and the reason to patch it rather than the database
    path alone: it also covers the ML model directory and config.yaml, so a
    test that trains a model cannot write into the real data/ directory."""
    from laser_trim_analyzer.config import Config

    got = Path(Config().models.path).resolve()
    production_data = _production_db().parent
    assert got != production_data and production_data not in got.parents, (
        f"ML models would be written to {got}, inside the production data "
        f"directory {production_data}"
    )
