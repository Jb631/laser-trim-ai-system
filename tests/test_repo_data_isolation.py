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


def test_the_trained_model_folder_does_not_live_in_the_repo():
    """The folder the composite-risk models and the ML predictors load from -- for every consumer
    that does not name one -- is redirected like the database (2026-09-25: it was resolved against
    the working directory, so a run from the main checkout scored with its real models)."""
    from laser_trim_analyzer.config import ml_models_directory
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.ml.manager import MLManager

    repo = REPO_DATA.resolve()
    for folder in (ml_models_directory(), Processor(use_ml=False).ml_storage_path,
                   MLManager(None).storage_path):
        resolved = Path(folder).resolve()
        assert resolved != repo and repo not in resolved.parents, (
            f"the trained-model folder resolved into the repo: {resolved}")


def test_loading_a_checkouts_real_trained_models_is_refused_loudly(
        _never_read_the_real_ml_models):
    """The guard behind that redirect (conftest `_never_read_the_real_ml_models`): a load from
    the repo's real data/ml_models -- a path built by hand, or one resolved against the working
    directory in the main checkout -- is refused, and recorded so that a caller that swallows the
    refusal still fails the test at teardown. Checked here, then acknowledged."""
    import pytest
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
    from laser_trim_analyzer.ml.predictor import ModelPredictor

    real = REPO_DATA / "ml_models"
    refused = _never_read_the_real_ml_models
    with pytest.raises(RuntimeError, match="refusing to load"):
        CompositeRiskModel.load(real / "composite_risk" / "8232-1.pkl")
    with pytest.raises(RuntimeError, match="refusing to load"):
        ModelPredictor("8232-1").load(real / "predictors" / "8232-1.pkl")
    assert len(refused) == 2 and all("ml_models" in p for p in refused), refused
    refused.clear()          # acknowledged: the guard did its job, and nothing real was read


def test_a_refusal_a_caller_swallowed_still_fails_the_test(tmp_path):
    """The loud half of the guard: the scoring code swallows a loader's error (a score is never
    fatal), so a refusal alone would pass in silence. The guard records it and fails the test at
    teardown -- proved on an inner test, run in a subprocess with this suite's conftest loaded as a
    plugin, that swallows the refusal exactly as the scorer does."""
    import os
    import subprocess
    import sys

    repo = REPO_DATA.parent
    inner = tmp_path / "test_inner_swallows.py"
    inner.write_text(
        "from pathlib import Path\n"
        "def test_swallows_the_refusal():\n"
        "    from laser_trim_analyzer.ml.predictor import ModelPredictor\n"
        "    try:\n"
        f"        ModelPredictor('x').load(Path({str(REPO_DATA)!r}) / 'ml_models' / 'predictors' / 'x.pkl')\n"
        "    except Exception:\n"
        "        pass            # as the scoring code swallows it\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(repo / "src"), str(repo / "tests")]))
    run = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", str(inner), "-p", "conftest", "-q",
         "-p", "no:cacheprovider", "--rootdir", str(tmp_path)],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(tmp_path))
    out = run.stdout + run.stderr
    assert run.returncode != 0, out[-2000:]
    assert "read a checkout's real trained ML models" in out, out[-2000:]
