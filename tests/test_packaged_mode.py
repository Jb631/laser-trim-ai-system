"""The packaged build is safe (2026-09-30).

The app is also handed out as a PyInstaller folder (`LaserTrimAnalyzer.exe`), where `sys.frozen`
is set and `sys.executable` is the app itself. What that changes, and what these tests pin:

  * NO worker PROCESSES: a spawned worker is a second copy of `sys.executable` -- in a packaged
    build, the app. `worker_count()` says none, with the reason, and the folder runs on threads:
    its batch line reads "N threads (the packaged build runs on threads)";
  * the entry point guards itself: `multiprocessing.freeze_support()` is the first thing `main()`
    does, so a stray spawn becomes that child and exits -- it can never open a second window;
  * a WINDOWED build has no console (`sys.stdout` is None): logging must not need one;
  * the app's folder is the executable's, so `data/` sits beside the .exe;
  * trained predictors copied from another computer do not load (their integrity key is the
    hostname and install path of the computer that trained them): the app says so in its log and
    goes on -- every view reads stored values.

`sys.frozen` is set by monkeypatch: nothing here needs a build.
"""
import importlib.util
import logging
import multiprocessing
import re
import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import worker_stubs

THREADS_WHEN_PACKAGED = "the packaged build runs on threads"


@pytest.fixture
def frozen(monkeypatch):
    """This process, as a packaged build sees itself."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)


class _Collect:
    """A writer that keeps every Outcome it is handed (the dispatch is what is under test)."""

    def __init__(self):
        self.outcomes = []

    def add(self, outcome):
        self.outcomes.append(outcome)
        return outcome.result

    def tick(self):
        pass

    def flush(self):
        pass


# ---- no worker processes ----------------------------------------------------------------------

def test_a_packaged_build_has_no_worker_processes_however_big_the_machine(frozen):
    from laser_trim_analyzer.core.ingest_worker import worker_count
    assert worker_count(cpus=14, free_gb=20.0) == (0, THREADS_WHEN_PACKAGED)
    assert worker_count() == (0, THREADS_WHEN_PACKAGED)


def test_from_source_the_same_machine_still_gets_its_worker_processes():
    """The control: the rule is the packaged build's alone."""
    from laser_trim_analyzer.core.ingest_worker import worker_count
    assert not getattr(sys, "frozen", False)
    assert worker_count(cpus=14, free_gb=20.0) == (8, "")


def _folder_run(tmp_path, monkeypatch):
    """A folder big enough for worker processes, run through the real dispatch: (how many pools
    were asked for, the batch line's `workers` text)."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core import ingest_worker
    from laser_trim_analyzer.database.specs import SpecSnapshot
    asked = []

    def recorded(cls, ctx, n, **k):
        asked.append(n)
        raise ingest_worker.PoolFailed("recorded here, never started")

    monkeypatch.setattr(ingest_worker.WorkerPool, "start", classmethod(recorded))
    monkeypatch.setattr(ingest_worker, "PROCESS_MIN_FILES", 1)
    cfg = Config()
    cfg.processing.turbo_mode_threshold = 1
    proc = worker_stubs.StubProcessor(config=cfg, snapshot=SpecSnapshot())
    files = worker_stubs.make_files(tmp_path / "in", [f"f{i:03d}.xls" for i in range(30)])
    writer = _Collect()
    got = list(proc.process_batch(files, incremental=False, writer=writer))
    assert len(got) == 30 and len(writer.outcomes) == 30
    return asked, proc.last_workers


def test_a_packaged_folder_runs_on_threads_and_its_batch_line_says_why(tmp_path, monkeypatch,
                                                                       frozen):
    asked, workers = _folder_run(tmp_path, monkeypatch)
    assert asked == [], "a packaged build asked for worker processes"
    assert re.fullmatch(r"\d+ threads? \(the packaged build runs on threads\)", workers), workers


def test_the_same_folder_from_source_does_ask_for_worker_processes(tmp_path, monkeypatch):
    """The control for the test above: unfrozen, this very folder asks for a pool -- so a
    packaged run that asked for none was the rule, not the folder."""
    asked, workers = _folder_run(tmp_path, monkeypatch)
    assert len(asked) == 1 and THREADS_WHEN_PACKAGED not in workers, (asked, workers)


# ---- the entry point --------------------------------------------------------------------------

def _run_main(monkeypatch, tmp_path, order):
    import laser_trim_analyzer.config as cmod
    import laser_trim_analyzer.gui.v6.app as v6_mod
    from laser_trim_analyzer.__main__ import main
    from laser_trim_analyzer.config import Config
    monkeypatch.setattr(v6_mod, "V6App",
                        MagicMock(side_effect=lambda *a, **k: order.append("app") or MagicMock()))
    cfg = Config()
    cfg.database.path = tmp_path / "t.db"
    monkeypatch.setattr(cmod, "get_config", lambda: order.append("config") or cfg)
    monkeypatch.setattr(sys, "argv", ["laser_trim_analyzer", "--v6"])
    main()


def test_main_asks_freeze_support_before_it_does_anything_else(monkeypatch, tmp_path):
    """In a packaged build a spawned child re-runs the app's own executable.
    `freeze_support()` is what turns that child into the worker it was meant to be -- and exits
    -- so it must run before the config is read, a database is opened or a window is built."""
    order = []
    monkeypatch.setattr(multiprocessing, "freeze_support",
                        lambda: order.append("freeze_support"))
    _run_main(monkeypatch, tmp_path, order)
    assert order[0] == "freeze_support" and order.count("freeze_support") == 1, order
    assert "config" in order and "app" in order, order


def test_freeze_support_is_the_first_statement_of_main():
    """Static, so a later edit cannot slide an import or a log line above it unnoticed."""
    import ast
    import laser_trim_analyzer.__main__ as entry
    tree = ast.parse(Path(entry.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = [s for s in main.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]   # docstring
    assert ast.unparse(body[0]) == "multiprocessing.freeze_support()", ast.unparse(body[0])


def test_a_windowed_build_has_no_console_and_logging_does_not_need_one(monkeypatch, tmp_path):
    """PyInstaller's windowed bootloader leaves sys.stdout and sys.stderr as None. A
    StreamHandler built on None would raise inside logging on every record."""
    from laser_trim_analyzer import config as cfg
    import laser_trim_analyzer.__main__ as entry
    app = tmp_path / "LaserTrimAnalyzer"
    monkeypatch.setattr(cfg, "get_app_directory", lambda: app)
    monkeypatch.setattr(entry, "get_app_directory", lambda: app)
    with_console = entry._log_handlers()
    with monkeypatch.context() as no_console:
        no_console.setattr(sys, "stdout", None)
        windowed = entry._log_handlers()
    try:
        assert [type(h).__name__ for h in with_console] == ["StreamHandler",
                                                           "RotatingFileHandler"]
        assert [type(h).__name__ for h in windowed] == ["RotatingFileHandler"]
        assert Path(windowed[0].baseFilename) == app / "data" / "laser_trim.log"
    finally:
        for h in with_console + windowed:
            h.close()


def test_a_packaged_apps_folder_is_the_executables(monkeypatch, tmp_path, frozen):
    """`data/` -- the database, config.yaml, the log -- sits beside LaserTrimAnalyzer.exe. Asked
    of a fresh copy of the module: the suite replaces `get_app_directory` in the imported one."""
    from laser_trim_analyzer import config as cfg
    exe = tmp_path / "LaserTrimAnalyzer" / "LaserTrimAnalyzer.exe"
    monkeypatch.setattr(sys, "executable", str(exe))
    spec = importlib.util.spec_from_file_location("config_as_packaged", cfg.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.get_app_directory() == exe.parent
    assert fresh.Config().database.path == exe.parent / "data" / "analysis.db"
    assert fresh.ml_models_directory() == exe.parent / "data" / "ml_models"


# ---- what the app reads from beside its own code ----------------------------------------------

# Every module that uses __file__, and why that is safe in a packaged build -- where a module's
# __file__ points INSIDE the bundle, and there is no source tree, no scripts/ and no
# pyproject.toml to find. A new entry here is a decision: if the module READS something from
# beside itself, that file must also be bundled (packaging/laser_trim_v6.spec, `datas`).
FILE_USERS = {
    "config.py": "get_app_directory(): only when NOT frozen -- the frozen branch is the "
                 "executable's folder",
    "core/ingest_worker.py": "_package_file(): part of a worker PROCESS's context, which a "
                             "packaged build never builds (it runs on threads)",
    "ml/predictor.py": "_get_hmac_key(): the install path as a STRING in the predictor "
                       "integrity key; nothing is read from it",
    "gui/v6/font_loader.py": "FONT_DIR: the bundled fonts -- read from beside the code, and "
                             "bundled by the spec",
    "gui/v6/icons.py": "ICON_DIR: the bundled icon images -- read from beside the code, and "
                       "bundled by the spec",
    "selfcheck.py": "package_directory(): the app's own modules, walked FROM SOURCE only, and "
                    "packaged_manifest.json, which the spec writes and bundles beside this "
                    "module",
}
# The only folders/files the app reaches through its own directory: `data` (the database,
# config.yaml, the log, the trained models), and beside the .exe of a packaged build the stamp
# the build wrote (read) and the self-check's result (written by `--check` alone). `path_value`
# is config.py resolving a RELATIVE database.path from config.yaml against the app's folder.
APP_DIRECTORY_CHILDREN = {"data", "BUILD_INFO_NAME", "CHECK_RESULT_NAME", "path_value"}


def _app_sources():
    from laser_trim_analyzer import config as cfg
    package = Path(cfg.__file__).resolve().parent
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(package).as_posix()
        if not rel.startswith("tests/"):
            yield rel, path


def test_every_use_of_dunder_file_in_the_app_is_a_known_one():
    users = {rel for rel, path in _app_sources() if "__file__" in path.read_text()}
    assert users == set(FILE_USERS), (
        f"new: {sorted(users - set(FILE_USERS))}; gone: {sorted(set(FILE_USERS) - users)} -- "
        "see FILE_USERS above")


def test_the_only_files_shipped_beside_the_code_are_the_fonts_and_icons():
    """Anything that is not Python inside the package has to be named in the spec's `datas`, or
    the packaged app will not have it. Today that is the fonts folder and the icons folder
    (2026-10-04), and nothing else."""
    from laser_trim_analyzer import config as cfg
    package = Path(cfg.__file__).resolve().parent
    others = {p.relative_to(package).as_posix() for p in package.rglob("*")
              if p.is_file() and p.suffix not in (".py", ".pyc") and p.name != ".DS_Store"}
    assert others and all(o.startswith(("gui/v6/fonts/", "gui/v6/icons/")) for o in others), \
        sorted(others)


def test_the_app_reaches_only_its_data_folder_through_its_own_directory():
    """No path is built from the app directory into src/, scripts/, a version file or
    pyproject.toml: in a packaged build the app directory holds the .exe and `data`, nothing
    else of the checkout."""
    joined = set()
    for _rel, path in _app_sources():
        joined |= set(re.findall(
            r"(?:get_app_directory\(\)|app_directory\(\)|app_dir\b[^/\n]*\))\s*/\s*"
            r"[\"']?(\w[\w.]*)", path.read_text()))
    assert joined == APP_DIRECTORY_CHILDREN, joined


# ---- trained predictors from another computer -------------------------------------------------

def _a_trained_predictor(name="INVENTED-1"):
    """A predictor with the least it needs to be saved (values invented; nothing is fitted)."""
    from datetime import datetime
    from laser_trim_analyzer.ml.predictor import ModelPredictor, PredictorMetrics
    p = ModelPredictor(name)
    p.is_trained = True
    p.training_date = datetime(2026, 1, 2)
    p.training_samples = 60
    p.metrics = PredictorMetrics(accuracy=0.5)
    return p


def test_a_predictor_trained_on_another_computer_does_not_load_and_nothing_raises(
        tmp_path, monkeypatch, caplog):
    """The integrity key is the hostname plus the install path, so a predictor file carried to
    the coworker's PC fails its check there: `load` returns False and says why at ERROR -- it
    never raises, and never loads the pickle."""
    from laser_trim_analyzer.ml.predictor import ModelPredictor
    path = tmp_path / "predictors" / "INVENTED-1.pkl"
    assert _a_trained_predictor().save(path)
    assert ModelPredictor("INVENTED-1").load(path) is True          # here, it loads

    monkeypatch.setattr(socket, "gethostname", lambda: "another-computer")
    there = ModelPredictor("INVENTED-1")
    with caplog.at_level(logging.ERROR, logger="laser_trim_analyzer.ml.predictor"):
        assert there.load(path) is False
    assert there.is_trained is False and there.classifier is None
    assert any("integrity check failed" in r.getMessage() for r in caplog.records), caplog.text


def test_the_ml_state_loads_without_the_predictors_it_could_not_verify(tmp_path, monkeypatch):
    """What the app does with that at start-up and at ingest: the manager finishes loading, the
    predictor is simply not trained there, and `load_ml_state` hands the analysis no predictor
    for the model -- no exception anywhere. The predictor panel then reads "No predictor"."""
    from laser_trim_analyzer.core.processor import load_ml_state
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml import invalidate_shared_ml_manager
    from laser_trim_analyzer.ml.manager import MLManager
    models = tmp_path / "ml_models"
    assert _a_trained_predictor().save(models / "predictors" / "INVENTED-1.pkl")
    db = DatabaseManager(tmp_path / "copy.db")
    try:
        here = MLManager(db, ml_storage_path=models)
        assert here.load_all() is True and here.predictors["INVENTED-1"].is_trained

        monkeypatch.setattr(socket, "gethostname", lambda: "another-computer")
        there = MLManager(db, ml_storage_path=models)
        assert there.load_all() is True and there.load_error is None
        assert there.predictors["INVENTED-1"].is_trained is False
        assert there.get_failure_probability("INVENTED-1", {"sigma_gradient": 0.1}) is None

        import laser_trim_analyzer.ml as ml
        monkeypatch.setattr(ml, "get_shared_ml_manager", lambda _db: there)
        thresholds, predictors = load_ml_state(db)
        assert predictors == {} and thresholds == {}
    finally:
        invalidate_shared_ml_manager()
        db.close()
