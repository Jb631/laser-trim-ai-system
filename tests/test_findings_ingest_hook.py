import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _db(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / "hook.db")


def test_findings_refresh_runs_after_a_batch_that_saved_trims(tmp_path, monkeypatch):
    from laser_trim_analyzer.core import ingest_run
    from laser_trim_analyzer.findings import engine
    seen = []
    monkeypatch.setattr(engine, "refresh_findings", lambda _db, models: seen.append(list(models)) or 0)
    phases = {}
    ingest_run._post_batch(_db(tmp_path), {"M2", "M1"}, 3, phases, None)
    assert seen == [["M1", "M2"]]
    assert "findings" in phases


def test_a_final_test_only_batch_does_not_recompute_findings(tmp_path, monkeypatch):
    from laser_trim_analyzer.core import ingest_run
    from laser_trim_analyzer.findings import engine
    seen = []
    monkeypatch.setattr(engine, "refresh_findings", lambda _db, models: seen.append(list(models)) or 0)
    phases = {}
    ingest_run._post_batch(_db(tmp_path), {"M1"}, 0, phases, None)
    assert seen == [] and "findings" not in phases


def test_a_failing_refresh_never_breaks_the_ingest(tmp_path, monkeypatch):
    from laser_trim_analyzer.core import ingest_run
    from laser_trim_analyzer.findings import engine

    def boom(_db, models):
        raise RuntimeError("findings exploded")
    monkeypatch.setattr(engine, "refresh_findings", boom)
    phases = {}
    ingest_run._post_batch(_db(tmp_path), {"M1"}, 3, phases, None)     # must not raise
    assert "findings" in phases


REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "scripts" / "refresh_findings.py"


def _tool(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          cwd=REPO, timeout=300)


# NOTE FOR WHOEVER EDITS THIS: no test here may name data/analysis.db. The refusal is the code under
# test -- if it regressed, a test aimed at the real path would open the owner's work database
# read-write before it went red. Every case below uses a path that does not exist, or a decoy.

@pytest.mark.parametrize("name", ["analysis.db", "Analysis.db", "ANALYSIS.DB"])
def test_the_tool_refuses_anything_named_like_the_work_database(tmp_path, name):
    r = _tool(TOOL, str(tmp_path / name))          # the file does not exist: refused BEFORE the existence check
    assert r.returncode == 2 and "REFUSED" in r.stdout, r.stdout + r.stderr


def test_the_tool_refuses_the_work_database_under_another_name(tmp_path):
    """The same FILE under a different path -- a hard link here; a case variant on a
    case-insensitive volume is the everyday version. REPO is derived from the script's own
    location, so a COPY of the script inside a decoy checkout treats the decoy as "production"."""
    decoy_repo = tmp_path / "repo"
    (decoy_repo / "scripts").mkdir(parents=True)
    (decoy_repo / "data").mkdir()
    script = decoy_repo / "scripts" / "refresh_findings.py"
    shutil.copy(TOOL, script)
    decoy = decoy_repo / "data" / "analysis.db"
    decoy.write_bytes(b"")
    alias = tmp_path / "alias.db"
    try:
        os.link(decoy, alias)
    except OSError as exc:                          # a filesystem with no hard links
        pytest.skip(f"cannot hard-link here: {exc}")
    r = _tool(script, str(alias))
    assert r.returncode == 2 and "REFUSED" in r.stdout, r.stdout + r.stderr
    assert decoy.stat().st_size == 0                # and nothing was written into it


def test_production_flag_gets_past_the_refusal(tmp_path):
    r = _tool(TOOL, str(tmp_path / "analysis.db"), "--production")
    assert "REFUSED" not in r.stdout and "no such database" in r.stdout, r.stdout + r.stderr


def test_an_ordinary_copy_is_not_refused(tmp_path):
    r = _tool(TOOL, str(tmp_path / "copy.db"))
    assert "REFUSED" not in r.stdout and "no such database" in r.stdout, r.stdout + r.stderr


def test_the_batch_phase_line_names_what_findings_cost(caplog):
    from laser_trim_analyzer.core import ingest_run

    def line(phases):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=ingest_run.logger.name):
            ingest_run.log_phases(phases, 5, object(), object())
        hits = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Batch phases:")]
        assert len(hits) == 1, hits
        return hits[0]

    assert "findings 12.3s" in line({"walk": 0.1, "process": 1.0, "findings": 12.3})
    assert "findings" not in line({"walk": 0.1, "process": 1.0})     # a batch with no new trims
