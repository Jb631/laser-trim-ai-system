import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def _never_touch_the_real_database(tmp_path, monkeypatch):
    """No test may reach ./data/analysis.db — James's production data.

    `Processor` (and several other call sites) resolve their database through
    the module-global `get_database()`, NOT through whatever manager the test
    handed them. When that global is unset, `get_database()` constructs a
    default `DatabaseManager`, which points at ./data/analysis.db and opens it
    READ-WRITE. A test doing nothing wrong can therefore write to the real
    3.6 GB work database; that happened during the HOME/shell work and was
    caught only because the row counts were checked afterwards.

    Autouse, so the guard cannot be forgotten. Pre-seeding the global alone is
    NOT enough — production code may legitimately call `reset_database()`, and
    the next `get_database()` would rebuild the default manager and land back
    on real data. So the DEFAULT PATH itself is redirected: `__init__` resolves
    an unset path through this module's `get_config()`, so patching that makes
    every fall-through — including a bare `DatabaseManager()` — land in tmp.
    A test that wants its own manager still just passes one.

    Redirecting the default is still not the whole job. `__init__` only
    consults `get_config()` when `database_path is None`, so a path handed in
    EXPLICITLY sails straight past the redirect — and a test can build the
    production path by hand, which is exactly what
    `test_5_8_2026_bugfixes.py::test_get_historical_data_light_load_returns_rows`
    did (`parents[1] / "data" / "analysis.db"`, skipped only when absent, so it
    ran against real data in a normal checkout). The last stanza closes that:
    any path under a repo `data/` directory is refused outright rather than
    redirected, because a test naming that file is a bug in the test, not a
    fall-through to paper over.
    """
    from laser_trim_analyzer import config as _cfg
    from laser_trim_analyzer.config import Config, get_app_directory
    from laser_trim_analyzer.database import manager as _mgr

    safe = Config()
    safe.database.path = tmp_path / "guard.db"
    monkeypatch.setattr(_mgr, "get_config", lambda: safe)

    # Both roots: the tree the tests live in and the tree the package was
    # imported from. They differ when the suite runs from a git worktree.
    protected = {
        (Path(__file__).resolve().parents[1] / "data").resolve(),
        (Path(get_app_directory()).resolve() / "data"),
    }
    _unguarded_init = _mgr.DatabaseManager.__init__

    def _guarded_init(self, database_path=None, *args, **kwargs):
        if database_path is not None:
            target = Path(database_path).resolve()
            for directory in protected:
                if target == directory or directory in target.parents:
                    raise RuntimeError(
                        f"refusing to open {target}: it is inside {directory}, "
                        "which holds the production database. Opening it "
                        "constructs an engine and runs PRAGMA journal_mode=WAL, "
                        "so it WRITES. Point the test at tmp_path instead."
                    )
        return _unguarded_init(self, database_path, *args, **kwargs)

    monkeypatch.setattr(_mgr.DatabaseManager, "__init__", _guarded_init)

    # `protected` is deliberately computed ABOVE, from the REAL app directory,
    # because the next patch moves it. Reversing these two would compute the
    # protected set from the tmp dir and guard nothing.
    #
    # `get_app_directory()` is the choke point for every default data path, not
    # just the database: config.py resolves the DB (:54), the ML model
    # directory (:96) and config.yaml (:243, :304) through it. Patching
    # `manager.get_config` alone leaves a bare `Config()` — and anything
    # anchored to the app directory, such as __main__'s log file — still
    # pointing into the real data/ directory.
    fake_app_root = tmp_path / "app_root"
    (fake_app_root / "data").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_cfg, "get_app_directory", lambda: fake_app_root)
    # get_config() memoises; without this it hands back a Config built before
    # the patch, still holding the production path.
    monkeypatch.setattr(_cfg, "_config", None)

    previous = _mgr._db_manager
    _mgr._db_manager = _mgr.DatabaseManager(safe.database.path)
    yield
    try:
        _mgr._db_manager.close()
    except Exception:
        pass
    _mgr._db_manager = previous


@pytest.fixture(scope="session")
def tk_root():
    """One headless CTk root for all widget-construction tests (no mainloop)."""
    import customtkinter as ctk
    try:
        ctk.deactivate_automatic_dpi_awareness()
    except Exception:
        pass
    root = ctk.CTk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def make_app(tmp_path):
    """Factory for a V6App on an ISOLATED tmp DB, auto-train OFF, destroyed on teardown.

    Use for every V6App test. Do NOT also request `tk_root` in an app test
    (that would create two live CTk roots). Retirement (Spec 3e Graduation)
    changes only the import line below.
    """
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.gui.v6.app import V6App

    created = []

    def _factory(db_name="v6.db"):
        cfg = Config()
        cfg.database.path = tmp_path / db_name
        app = V6App(
            cfg,
            db=DatabaseManager(cfg.database.path),
            auto_train_on_first_run=False,
        )
        app.withdraw()
        created.append(app)
        return app

    yield _factory
    for app in created:
        try:
            app.destroy()
        except Exception:
            pass
