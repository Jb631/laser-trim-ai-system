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
    NOT enough — code may call `reset_database()`, and the next
    `get_database()` falls through to the default manager. Since 2026-09-24
    that fall-through is refused in code unless the APP asked for it
    (`manager.allow_default_database()`, called only by the entry point), and a
    test is not the app: the switch is pinned OFF here, per test, so a test that
    runs `main()` — which turns it on — cannot leave it on for the tests after
    it. The DEFAULT PATH itself is still redirected, for a test that turns the
    switch on as the app does: `__init__` resolves an unset path through this
    module's `get_config()`, so patching that makes every allowed fall-through
    — including a bare `DatabaseManager()` — land in tmp. A test that wants its
    own manager still just passes one.

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

    # Every test runs as a script would: the implicit default is refused.
    monkeypatch.setattr(_mgr, "_default_database_allowed", False)

    # The refusal's ERROR log is written once per PROCESS, not once per
    # refusal (C2 task 3b) -- so without this, whichever test happens to
    # refuse the default first in this process leaves the flag set and every
    # later test's own refusal logs silently. Reset per test, same as the
    # switch above.
    monkeypatch.setattr(_mgr, "_default_database_refusal_logged", False, raising=False)

    # Both roots: the tree the tests live in and the tree the package was
    # imported from. They differ when the suite runs from a git worktree.
    protected = {
        (Path(__file__).resolve().parents[1] / "data").resolve(),
        # Resolve the WHOLE path, as the first entry does. `target` below is
        # fully resolved, so resolving only the parent here would silently stop
        # matching if data/ were ever a symlink — leaving that root unguarded.
        (Path(get_app_directory()) / "data").resolve(),
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

    # Built AFTER the patch above, so every path on it — the ML models
    # directory included, not just the database — is already tmp-based. Built
    # before it, this object would hand `manager.get_config()` callers a
    # production models path.
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


@pytest.fixture(autouse=True)
def _never_read_the_real_ml_models(monkeypatch):
    """No test may score with a checkout's REAL trained models -- data/ml_models.

    2026-09-25: the composite-risk loader resolved `data/ml_models` against the WORKING
    directory, so a golden recorded in a worktree (no models) failed in the main checkout, where
    the trained models are: `composite_trim_risk_score: None -> 0.341`. A test's numbers must
    never depend on what happens to sit in the repo's data/ folder. The loaders now resolve
    through `get_app_directory()`, which the fixture above redirects to tmp; this is the guard
    behind that redirect. Any load of a composite-risk model or a predictor from inside a
    checkout's real data/ directory is REFUSED -- and, because the callers swallow a loader's
    error (scoring is never fatal), each refusal is also recorded and fails the test at teardown.

    Yields the list of refusals: the one test that proves the refusal acknowledges it by
    clearing the list.
    """
    from laser_trim_analyzer import config as _cfg
    from laser_trim_analyzer.ml import composite_risk as _cr
    from laser_trim_analyzer.ml import predictor as _pred

    # Both roots, computed from files -- NOT from get_app_directory(), which the fixture above
    # patches (in either order this set is the real one).
    protected = {
        (Path(__file__).resolve().parents[1] / "data").resolve(),
        (Path(_cfg.__file__).resolve().parents[2] / "data").resolve(),
    }
    refused = []

    def refuse_the_real_models(path):
        target = Path(path).resolve()
        for directory in protected:
            if target == directory or directory in target.parents:
                refused.append(str(target))
                raise RuntimeError(
                    f"refusing to load {target}: it is inside {directory}, which holds a "
                    "checkout's REAL trained ML models -- a test's numbers must not depend on "
                    "them. Point the loader at tmp_path instead.")

    real_composite_load = _cr.CompositeRiskModel.load.__func__

    def guarded_composite_load(cls, path):
        refuse_the_real_models(path)
        return real_composite_load(cls, path)

    real_predictor_load = _pred.ModelPredictor.load

    def guarded_predictor_load(self, path):
        refuse_the_real_models(path)
        return real_predictor_load(self, path)

    monkeypatch.setattr(_cr.CompositeRiskModel, "load", classmethod(guarded_composite_load))
    monkeypatch.setattr(_pred.ModelPredictor, "load", guarded_predictor_load)
    yield refused
    if refused:
        pytest.fail("this test read a checkout's real trained ML models (refused, but a caller "
                    f"swallowed the refusal): {refused[:3]}", pytrace=False)


@pytest.fixture
def tk_root():
    """One headless CTk root for THIS test only (no mainloop).

    FUNCTION-scoped on purpose, and it must stay that way. This was
    `scope="session"`, and that is what made the suite "take 2.5 hours" --
    it never did: it HUNG, at >100% CPU, which from the outside looks exactly
    like slow. A session root outlives every test, so in any process that also
    builds `V6App`s through `make_app` there are two or three live CTk roots at
    once. With more than one alive, `Misc.update()` inside a test's pump/settle
    helper stops returning: Tk's macOS idle handler keeps pumping the Cocoa
    event loop, the self-rearming `after` loops (CustomTkinter's appearance and
    scaling trackers, this app's UiDispatcher) keep firing there, and so Tcl's
    `update` never sees an empty queue. Measured during the hang: ~94 timer
    callbacks a second, forever, all on the main thread.

    That is why targets pass one file at a time and wedge when combined --
    which is exactly what a hand-picked gate file list can never catch.
    `tests/test_tk_thread_safety.py` holds the regression guard (in a
    subprocess, so a regression fails instead of hanging the suite).

    The cost of function scope is a fresh root per test (~0.03 s), plus a
    larger Tk teardown bill in files that build heavy pages -- see the H5
    report's summary in `docs/decisions/2026-09-ledger-decisions.md`.
    """
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
