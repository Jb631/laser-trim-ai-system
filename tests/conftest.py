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
    """
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.database import manager as _mgr

    safe = Config()
    safe.database.path = tmp_path / "guard.db"
    monkeypatch.setattr(_mgr, "get_config", lambda: safe)

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
