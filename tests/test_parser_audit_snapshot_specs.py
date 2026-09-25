"""The parser-audit snapshot sees the same model_specs a pre-guard run would
have (2026-09-25, C2 task 3c).

`scripts/parser_audit/snapshot.py::build_snapshot` runs the Processor on a
throwaway database of its own (39461a2), because the app's default -- the
production database -- is refused outside the app. That throwaway database
starts EMPTY, so `model_specs` is empty too: the Processor's per-file spec
lookups (`_get_spec_for_analysis`, which feeds `linearity_type`/`angle_spec`/
`angle_tol`/`angle_tol_type`/`exclude_points` into `analyze_track`) now find
nothing, where a pre-guard run -- reaching `get_database()` with nothing
injected -- would have silently opened the real default database and read
whatever specs were there.

Measured against real files in Work Files (2026-09-25): for 8340 and 8340-3,
whose model_specs row carries `electrical_angle_tol_type='min'`, having that
spec loaded changes `_k_bounds_from_angle_tol` from locked (0.0, 0.0) to
(0.0, 0.05) -- a real k allowance -- which measurably shifted both
`linearity_error` and `linearity_fail_points` on real sample files. So the
snapshot must load `model_specs` from the configured default database,
READ-ONLY (never open it read-write -- that is the entire point of the
guard), before running.
"""
import importlib.util
import sqlite3
from pathlib import Path

import pytest

from laser_trim_analyzer.database import manager as mgr

REPO = Path(__file__).resolve().parents[1]


def _load(script: str):
    spec = importlib.util.spec_from_file_location(Path(script).stem, REPO / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot(tmp_path):
    return _load("scripts/parser_audit/snapshot.py")


def _seed_default_database(monkeypatch, path: Path, spec_row: dict):
    """A "default database" -- as an app run would have left one -- carrying
    one model_specs row. Config-only: DatabaseManager(path) with an EXPLICIT
    path is always allowed, so seeding it is not itself the behaviour under
    test."""
    from laser_trim_analyzer import config as cfgmod

    seed = mgr.DatabaseManager(path)
    seed.save_model_spec(spec_row)
    seed.close()

    cfg = cfgmod.Config()
    cfg.database.path = path
    monkeypatch.setattr(mgr, "get_config", lambda: cfg)


# ---------------------------------------------------------------------------
# the headline case
# ---------------------------------------------------------------------------

def test_build_snapshot_loads_specs_from_the_default_database_read_only(
        tmp_path, monkeypatch, snapshot):
    default_path = tmp_path / "default" / "analysis.db"
    _seed_default_database(monkeypatch, default_path, {
        "model": "8340", "linearity_type": "Absolute",
        "electrical_angle": 0.3, "electrical_angle_tol_type": "min",
    })
    before_bytes = default_path.read_bytes()

    seen = {}

    def _fake_processor(**_kw):
        seen["spec"] = mgr.get_database().get_model_spec("8340")

        class _NoFilesExpected:
            def process_file(self, *_a, **_kw):
                raise AssertionError("empty corpus: no file should be processed")

        return _NoFilesExpected()

    monkeypatch.setattr(snapshot, "Processor", _fake_processor)
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    snapshot.build_snapshot(corpus)          # empty corpus -- returns fast

    assert seen.get("spec") is not None, (
        "the snapshot's own database has no '8340' spec -- specs were not loaded"
    )
    assert seen["spec"]["electrical_angle"] == pytest.approx(0.3)
    assert seen["spec"]["electrical_angle_tol_type"] == "min"
    assert default_path.read_bytes() == before_bytes, (
        "snapshot.py must never write to the default database -- only read it"
    )


def test_build_snapshot_restores_both_db_manager_globals_afterwards(
        tmp_path, monkeypatch, snapshot):
    import laser_trim_analyzer.database as dbpkg

    default_path = tmp_path / "default" / "analysis.db"
    _seed_default_database(monkeypatch, default_path, {"model": "8340"})

    injected = mgr.DatabaseManager(tmp_path / "already_injected.db")
    monkeypatch.setattr(mgr, "_db_manager", injected, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", injected, raising=False)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    snapshot.build_snapshot(corpus)

    assert mgr._db_manager is injected, "the manager injected before the snapshot ran was not put back"
    assert dbpkg._db_manager is injected


# ---------------------------------------------------------------------------
# absence / corruption of the default database must not crash the snapshot
# ---------------------------------------------------------------------------

def test_build_snapshot_with_no_default_database_file_runs_with_empty_specs(
        tmp_path, monkeypatch, snapshot):
    from laser_trim_analyzer import config as cfgmod
    cfg = cfgmod.Config()
    cfg.database.path = tmp_path / "does_not_exist" / "analysis.db"
    monkeypatch.setattr(mgr, "get_config", lambda: cfg)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    snap = snapshot.build_snapshot(corpus)       # must not raise
    assert snap["total_files"] == 0


def test_build_snapshot_with_a_zero_byte_default_database_runs_with_empty_specs(
        tmp_path, monkeypatch, snapshot):
    """The exact shape of a worktree's decoy data/analysis.db: present, 0
    bytes, not a valid sqlite file -- must degrade, not raise."""
    from laser_trim_analyzer import config as cfgmod
    decoy = tmp_path / "decoy" / "analysis.db"
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"")
    cfg = cfgmod.Config()
    cfg.database.path = decoy
    monkeypatch.setattr(mgr, "get_config", lambda: cfg)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    snap = snapshot.build_snapshot(corpus)       # must not raise
    assert snap["total_files"] == 0


def test_build_snapshot_with_a_default_database_missing_the_specs_table(
        tmp_path, monkeypatch, snapshot):
    """An old/partial database file without model_specs -- must degrade, not
    raise (sqlite3's own OperationalError for "no such table")."""
    from laser_trim_analyzer import config as cfgmod
    path = tmp_path / "no_specs_table" / "analysis.db"
    path.parent.mkdir(parents=True)
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()

    cfg = cfgmod.Config()
    cfg.database.path = path
    monkeypatch.setattr(mgr, "get_config", lambda: cfg)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    snap = snapshot.build_snapshot(corpus)       # must not raise
    assert snap["total_files"] == 0
