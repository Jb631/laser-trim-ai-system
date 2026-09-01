"""The test suite must never write into the repository's own data/ directory.

Anything that reaches for "the app's database" on its own — Processor's
_mark_file_skipped(), ml.predictor, every GUI page — goes through the module
global get_database(), which lazily builds DatabaseManager() with NO path.
That falls back to get_app_directory()/data/analysis.db and CREATES it,
schema and all.

The damage is order-dependent and invisible: data/ is gitignored, so the
phantom database survives between runs untracked. A test that skips on
"local analysis.db not present" then stops skipping on the NEXT run and
queries an empty database, which reads as a flaky failure rather than the
pollution it is. scripts/app_qa_sweep.py guards the same hazard by pinning
the singleton before it can build its own.

The session-scoped _isolate_app_data fixture in conftest.py is what keeps
this true; these tests are its regression net.
"""
from pathlib import Path

REPO_DATA = Path(__file__).resolve().parents[1] / "data"


def test_global_database_does_not_live_in_the_repo():
    """get_database() must hand back a redirected DB, not <repo>/data/analysis.db."""
    from laser_trim_analyzer.database.manager import get_database

    resolved = Path(get_database().database_path).resolve()
    assert REPO_DATA.resolve() not in resolved.parents, (
        f"global database resolved into the repo: {resolved}"
    )


def test_default_config_paths_do_not_live_in_the_repo():
    """The config defaults every no-arg consumer inherits must be redirected too."""
    from laser_trim_analyzer.config import Config, get_config

    repo = REPO_DATA.resolve()
    for cfg in (Config(), get_config()):
        assert repo not in Path(cfg.database.path).resolve().parents
        assert repo not in Path(cfg.models.path).resolve().parents


def test_repo_data_dir_is_untouched():
    """Nothing the suite has run may have created a database in the repo.

    Rows are the discriminator, not existence — the same gate the 5-8 bugfix
    test now uses. A database the suite conjured is an empty schema; James's
    real work database has rows and predates the run, so a checkout carrying
    it sits this one out. Opened read-only so this can never write to it.

    Deliberately depends on no fixture: whichever conftest guard is in force,
    this check still runs.
    """
    import sqlite3

    import pytest

    db = REPO_DATA / "analysis.db"
    if not db.exists():
        return                                   # the good case
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT EXISTS(SELECT 1 FROM analysis_results)").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as exc:                 # no schema at all -> phantom
        raise AssertionError(f"phantom database created at {db} ({exc})") from None
    if rows:
        pytest.skip(f"{db} holds real data — the work database, not a phantom")
    raise AssertionError(
        f"phantom database created at {db}: it exists but holds no rows"
    )
