"""Only the app opens the default database; everything else names one (2026-09-24).

`DatabaseManager()` with no path -- which is what `get_database()` builds when no manager
was injected -- falls back to the config's database path: on the owner's machines, the
production database. On 2026-09-24 two scratch scripts built a `Processor`, whose spec
lookup reached `get_database()`; it opened James's production copy and ran every start-up
migration on it. That fall-through is now refused unless the APP asked for it
(`manager.allow_default_database()`, called by the entry point the launchers run), and
every script names its database or injects a manager.

The conftest pins the switch OFF for every test -- a test is not the app -- and redirects
the app directory into tmp, so the tests here that turn it on still land in tmp.

NO TEST HERE NAMES THE REAL data/analysis.db, and none lets a regression reach it: the
refusal is the code under test, and a test aimed at the real path would open it read-write
before going red. Every default below is pointed into tmp first.
"""
import importlib.util
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from laser_trim_analyzer.config import Config
from laser_trim_analyzer.database import manager as mgr

REPO = Path(__file__).resolve().parents[1]


def _default_at(monkeypatch, path: Path) -> Path:
    """Make the implicit default resolve to `path`, a tmp file that does not exist yet."""
    cfg = Config()
    cfg.database.path = path
    monkeypatch.setattr(mgr, "get_config", lambda: cfg)
    return path


def _tables(path: Path) -> set:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


# ------------------------------------------------------------------ the guard

@pytest.mark.parametrize("build", [lambda: mgr.DatabaseManager(),
                                   lambda: mgr.DatabaseManager(None)],
                         ids=["no-argument", "None"])
def test_the_implicit_default_is_refused_before_anything_is_created(tmp_path, monkeypatch,
                                                                      build):
    target = _default_at(monkeypatch, tmp_path / "not_yet" / "analysis.db")
    with pytest.raises(RuntimeError, match="No database named"):
        build()
    assert not target.parent.exists(), "the refusal came after the database's folder was made"


def test_get_database_with_nothing_injected_refuses_and_caches_nothing(tmp_path, monkeypatch):
    target = _default_at(monkeypatch, tmp_path / "not_yet" / "analysis.db")
    mgr.reset_database()                       # nothing injected -- a bare script's position
    with pytest.raises(RuntimeError, match="only the app opens it"):
        mgr.get_database()
    assert mgr._db_manager is None, "a refused manager was cached for the next caller"
    assert not target.parent.exists()


# ------------------------------------------------ the refusal is a NAMED subclass (C2 task 3a)

@pytest.mark.parametrize("build", [lambda: mgr.DatabaseManager(),
                                   lambda: mgr.DatabaseManager(None)],
                         ids=["no-argument", "None"])
def test_the_refusal_is_the_named_subclass(tmp_path, monkeypatch, build):
    """`DefaultDatabaseRefused` -- so a test (or a reviewer) can tell this
    RuntimeError apart from any other, while every existing bare
    `except RuntimeError`/`except Exception` still catches it."""
    _default_at(monkeypatch, tmp_path / "not_yet" / "analysis.db")
    with pytest.raises(mgr.DefaultDatabaseRefused) as excinfo:
        build()
    assert isinstance(excinfo.value, RuntimeError)


def test_default_database_refused_is_exported_from_the_database_package():
    from laser_trim_analyzer.database import DefaultDatabaseRefused
    assert DefaultDatabaseRefused is mgr.DefaultDatabaseRefused
    assert issubclass(DefaultDatabaseRefused, RuntimeError)


def test_get_database_raises_the_named_subclass_too(tmp_path, monkeypatch):
    _default_at(monkeypatch, tmp_path / "not_yet" / "analysis.db")
    mgr.reset_database()
    with pytest.raises(mgr.DefaultDatabaseRefused):
        mgr.get_database()


# --------------------------------------------- the refusal is logged ONCE per process (C2 task 3b)

def test_the_refusal_is_logged_once_per_process_not_once_per_call(tmp_path, monkeypatch, caplog):
    """The guard (39461a2) logs at ERROR as well as raising, because the
    Processor's model-spec lookups swallow the raise at DEBUG. Each lookup
    calls get_database() -- so a run over many files, each refused the same
    way, used to write one ERROR line per file. Dedupe at the source: once
    per process, not once per refusal."""
    _default_at(monkeypatch, tmp_path / "not_yet" / "analysis.db")
    monkeypatch.setattr(mgr, "_default_database_refusal_logged", False)
    with caplog.at_level(logging.ERROR, logger="laser_trim_analyzer.database.manager"):
        for _ in range(4):
            with pytest.raises(mgr.DefaultDatabaseRefused):
                mgr.DatabaseManager()
    refusals = [r for r in caplog.records
                if r.levelno == logging.ERROR and "No database named" in r.message]
    assert len(refusals) == 1, [r.message for r in caplog.records]


def test_processor_model_spec_lookups_across_two_files_log_the_refusal_exactly_once(
        tmp_path, monkeypatch, caplog):
    """The brief's own framing: two files processed with no database named ->
    exactly one ERROR record. _get_spec_for_analysis is the Processor's
    model-spec lookup that reaches get_database() per file (without a snapshot;
    the dead _get_linearity_type was the other, deleted 2026-09-25); the
    Processor's own handling (swallow at DEBUG, degrade to no specs) is
    unchanged -- only the ERROR-level log at the source is deduped."""
    from laser_trim_analyzer.core.processor import Processor

    _default_at(monkeypatch, tmp_path / "not_yet" / "analysis.db")
    monkeypatch.setattr(mgr, "_default_database_refusal_logged", False)
    mgr.reset_database()
    proc = Processor(use_ml=False)

    with caplog.at_level(logging.ERROR, logger="laser_trim_analyzer.database.manager"):
        assert proc._get_spec_for_analysis("MODEL-FILE-1") is not None   # "file" 1
        assert proc._get_spec_for_analysis("MODEL-FILE-2") is not None   # "file" 2

    refusals = [r for r in caplog.records
                if r.levelno == logging.ERROR and "No database named" in r.message]
    assert len(refusals) == 1, [r.message for r in caplog.records]
    assert mgr._db_manager is None, "a refused manager must not be cached"


def test_with_a_tmp_manager_injected_the_processor_logs_no_refusal_at_all(
        tmp_path, monkeypatch, caplog):
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr_mod
    import laser_trim_analyzer.database as dbpkg

    monkeypatch.setattr(mgr, "_default_database_refusal_logged", False)
    injected = mgr.DatabaseManager(tmp_path / "injected.db")
    monkeypatch.setattr(mgr_mod, "_db_manager", injected, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", injected, raising=False)
    proc = Processor(use_ml=False)

    with caplog.at_level(logging.ERROR, logger="laser_trim_analyzer.database.manager"):
        proc._get_spec_for_analysis("MODEL-FILE-1")
        proc._get_spec_for_analysis("MODEL-FILE-2")

    refusals = [r for r in caplog.records
                if r.levelno == logging.ERROR and "No database named" in r.message]
    assert refusals == []


def test_an_explicit_path_is_always_allowed(tmp_path):
    """Named on purpose -- `data/analysis.db` included, the way James's work scripts name it."""
    assert mgr._default_database_allowed is False
    named = tmp_path / "data" / "analysis.db"
    db = mgr.DatabaseManager(named)
    try:
        assert Path(db.get_database_path()) == named
        assert "analysis_results" in _tables(named)
    finally:
        db.close()


def test_with_the_apps_switch_on_the_default_opens_as_it_always_did(tmp_path, monkeypatch):
    target = _default_at(monkeypatch, tmp_path / "app" / "analysis.db")
    monkeypatch.setattr(mgr, "_default_database_allowed", True)
    db = mgr.DatabaseManager()
    try:
        assert Path(db.get_database_path()) == target
        assert "analysis_results" in _tables(target)
    finally:
        db.close()


# A fresh interpreter: what a scratch script actually meets -- the SHIPPED switch, which the
# conftest's per-test pin would otherwise hide. The app directory is moved into tmp before
# anything is imported, so even a regressed guard could only open a file under tmp.
_BARE_SCRIPT = r"""
import sys
from pathlib import Path
import laser_trim_analyzer.config as config
config.get_app_directory = lambda: Path(sys.argv[1])
import laser_trim_analyzer.database.manager as m
for build in (m.DatabaseManager, m.get_database):
    try:
        build()
    except RuntimeError:
        print("REFUSED", build.__name__)
    else:
        print("OPENED", build.__name__)
"""


def test_a_bare_script_is_refused_in_a_fresh_process(tmp_path):
    root = tmp_path / "bare_root"
    (root / "data").mkdir(parents=True)
    # The same package this process imported -- from a worktree, a bare `python` would
    # otherwise find the main checkout's code.
    env = dict(os.environ, PYTHONPATH=str(Path(mgr.__file__).resolve().parents[2]))
    r = subprocess.run([sys.executable, "-c", _BARE_SCRIPT, str(root)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=120)
    said = [ln for ln in r.stdout.splitlines() if ln.startswith(("REFUSED", "OPENED"))]
    assert r.returncode == 0 and said == ["REFUSED DatabaseManager", "REFUSED get_database"], \
        r.stdout + r.stderr
    assert list((root / "data").iterdir()) == [], "a refused default still created a file"


# -------------------------------------------------------------------- the app

def _production_wiring(monkeypatch, tmp_path):
    """The app's own wiring inside the conftest's tmp app directory: the manager resolves its
    default through the SAME `get_config()` that main() loads (the conftest points the manager
    at a separate tmp file; the app has no such split), and nothing is injected, so the app
    has to open the default itself."""
    from laser_trim_analyzer import config as cfgmod
    monkeypatch.setattr(mgr, "get_config", cfgmod.get_config)
    mgr.reset_database()
    default = Path(cfgmod.get_config().database.path)
    # Never production: the conftest moved the app directory into this test's tmp_path.
    assert default.name == "analysis.db" and tmp_path in default.parents, default
    return default


def test_the_v6_entry_point_opens_the_configured_database(monkeypatch, tmp_path):
    """`python -m src --v6` (run_v6.bat, launch_v6.command) -> main(): the switch goes on
    before the app is built, and the real V6App opens config.database.path -- not a stand-in."""
    import laser_trim_analyzer.gui.v6.app as v6_mod
    from laser_trim_analyzer.__main__ import main

    default = _production_wiring(monkeypatch, tmp_path)
    before = set(threading.enumerate())
    seen = {}

    def _first_moments_then_close(app):
        """What mainloop would do first -- let HOME's loaders finish on the database the app
        just opened -- then close the window instead of waiting for a user."""
        app.withdraw()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            app.update()
            if not [t for t in threading.enumerate() if t not in before and t.is_alive()]:
                break
            time.sleep(0.02)
        seen["path"] = Path(app.db.get_database_path())
        seen["allowed"] = mgr._default_database_allowed
        seen["global"] = mgr._db_manager is app.db
        app.destroy()

    monkeypatch.setattr(v6_mod.V6App, "run", _first_moments_then_close)
    monkeypatch.setattr(sys, "argv", ["laser_trim_analyzer", "--v6"])
    main()
    assert seen["path"] == default
    assert seen["allowed"] is True
    assert seen["global"], "the app and the processor would be on different databases"
    assert "analysis_results" in _tables(default)


def test_the_v5_entry_point_turns_the_switch_on_before_its_pages_ask(monkeypatch, tmp_path):
    """`python -m src` (run_v5.bat). Every V5 page reaches its database through
    `get_database()`; the stand-in does exactly that from its constructor."""
    import laser_trim_analyzer.app as v5_mod
    from laser_trim_analyzer.__main__ import main

    default = _production_wiring(monkeypatch, tmp_path)
    seen = {}

    class _LikeTheV5Pages:
        def __init__(self, config):
            seen["path"] = Path(mgr.get_database().get_database_path())

        def run(self):
            pass

    monkeypatch.setattr(v5_mod, "LaserTrimApp", _LikeTheV5Pages)
    monkeypatch.setattr(sys, "argv", ["laser_trim_analyzer"])
    main()
    assert seen["path"] == default


def test_make_app_opens_the_database_it_is_handed_with_the_switch_off(make_app, tmp_path):
    """The tests' own construction path: an explicit, injected database needs no switch."""
    assert mgr._default_database_allowed is False
    app = make_app("proof.db")
    assert Path(app.db.get_database_path()) == tmp_path / "proof.db"
    assert "analysis_results" in _tables(tmp_path / "proof.db")


# -------------------------------------------------------------------- scripts

def _load(script: str):
    spec = importlib.util.spec_from_file_location(Path(script).stem, REPO / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exit_code(main) -> object:
    try:
        return main()
    except SystemExit as stop:
        return stop.code


# The scripts that WRITE and used to fall back to data/analysis.db -- through
# DatabaseManager() or a hard-coded default -- when no database was given.
NAME_THEIR_DATABASE = [
    # (script, other arguments, how the database is named)
    ("scripts/fix_smoothness_tracks.py", [], lambda db: [str(db)]),
    ("scripts/backfill_linearity_error.py", ["--dry-run"], lambda db: ["--db", str(db)]),
    ("scripts/backfill_untrimmed_sigma.py", ["--dry-run"], lambda db: ["--db", str(db)]),
    ("scripts/backfill_trim_effort.py", [], lambda db: [str(db)]),
    ("scripts/resummarize_smoothness.py", ["--dry-run"], lambda db: ["--db", str(db)]),
    ("scripts/fix_corrupt_linearity_spec.py", [], lambda db: ["--db", str(db)]),
]


@pytest.mark.parametrize("script,other,named", NAME_THEIR_DATABASE,
                         ids=[Path(s).stem for s, _o, _n in NAME_THEIR_DATABASE])
def test_a_script_that_writes_names_its_database_and_it_must_exist(script, other, named,
                                                                  tmp_path, monkeypatch,
                                                                  capsys):
    module = _load(script)

    def _opened(*_a, **_k):
        raise AssertionError(f"{script} opened a database")
    monkeypatch.setattr(sqlite3, "connect", _opened)
    monkeypatch.setattr(mgr.DatabaseManager, "__init__", _opened)

    monkeypatch.setattr(sys, "argv", [script, *other])                   # no database at all
    assert _exit_code(module.main) == 2
    said = capsys.readouterr().err
    assert "usage:" in said and "required" in said, said

    missing = tmp_path / "typo" / "analysis.db"                         # one that is not there
    monkeypatch.setattr(sys, "argv", [script, *other, *named(missing)])
    assert _exit_code(module.main) not in (0, None)
    assert not missing.exists() and not missing.parent.exists()


def test_the_pool_probe_gives_its_processor_a_throwaway_database(tmp_path):
    """Workers are fresh processes with nothing injected; each builds its Processor through
    `_processor()`, which must point `get_database()` at the probe's own scratch folder."""
    probe = _load("scripts/pool_probe.py")
    mgr.reset_database()                       # a fresh worker: nothing injected
    probe._use_scratch_db(str(tmp_path))
    try:
        probe._processor()
    finally:
        logging.disable(logging.NOTSET)        # _processor() mutes logging process-wide
    assert Path(mgr.get_database().get_database_path()).parent == tmp_path


def test_the_parser_snapshot_runs_on_a_throwaway_database(tmp_path, monkeypatch):
    root = logging.getLogger()
    level = root.level
    try:
        snapshot = _load("scripts/parser_audit/snapshot.py")      # mutes the root logger
    finally:
        root.setLevel(level)
    injected = mgr.get_database()
    seen = []
    monkeypatch.setattr(snapshot, "Processor",
                        lambda **_kw: seen.append(Path(mgr.get_database().get_database_path())))
    (tmp_path / "corpus").mkdir()
    snapshot.build_snapshot(tmp_path / "corpus")                     # empty corpus: no files
    assert seen and seen[0] != Path(injected.get_database_path())
    assert not seen[0].exists(), "the throwaway database outlived the snapshot"
    assert mgr.get_database() is injected, "the manager injected before it was not put back"
