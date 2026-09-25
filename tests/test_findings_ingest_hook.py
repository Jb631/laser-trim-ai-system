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
    monkeypatch.setattr(engine, "refresh_findings",
                        lambda _db, models, report=None: seen.append(list(models)) or 0)
    phases = {}
    ingest_run._post_batch(_db(tmp_path), {"M2", "M1"}, 3, phases, None)
    assert seen == [["M1", "M2"]]
    assert "findings" in phases


def test_a_batch_that_saved_only_final_tests_recomputes_those_models_findings(tmp_path, monkeypatch):
    """M4 (final review, 2026-09-25): rework_load and station_setup read final tests, so a batch
    of final-test files changes what they would say -- it used to leave both stale until that
    model's next trim batch."""
    from laser_trim_analyzer.core import ingest_run
    from laser_trim_analyzer.findings import engine
    seen = []
    monkeypatch.setattr(engine, "refresh_findings",
                        lambda _db, models, report=None: seen.append(list(models)) or 0)
    phases = {}
    ingest_run._post_batch(_db(tmp_path), {"M2", "M1"}, 0, phases, None, new_final_tests=2)
    assert seen == [["M1", "M2"]] and "findings" in phases


def test_a_batch_that_saved_nothing_does_not_recompute_findings(tmp_path, monkeypatch):
    from laser_trim_analyzer.core import ingest_run
    from laser_trim_analyzer.findings import engine
    seen = []
    monkeypatch.setattr(engine, "refresh_findings",
                        lambda _db, models, report=None: seen.append(list(models)) or 0)
    phases = {}
    ingest_run._post_batch(_db(tmp_path), {"M1"}, 0, phases, None, new_final_tests=0)
    assert seen == [] and "findings" not in phases


def test_the_batch_counts_the_final_tests_it_saved_for_the_findings_gate(tmp_path, monkeypatch):
    """Only a final test that was SAVED counts (it carries its final_test_id); one the processor
    could not read comes back as an error result without one."""
    from types import SimpleNamespace
    from laser_trim_analyzer.core import ingest_run
    from laser_trim_analyzer.core.models import AnalysisStatus

    (tmp_path / "a.xls").write_bytes(b"junk")

    def ft(i, saved):
        r = SimpleNamespace(file_type="final_test",
                            metadata=SimpleNamespace(model="FT1", filename=f"ft{i}.xls"),
                            overall_status=AnalysisStatus.PASS if saved else AnalysisStatus.ERROR)
        if saved:
            r.final_test_id = 100 + i
        return r

    class _Proc:
        last_scan_stats = {}

        def __init__(self, *a, **k):
            pass

        def process_batch(self, *a, **k):
            yield ft(0, True)
            yield ft(1, True)
            yield ft(2, False)
            return SimpleNamespace(processed=3)

    calls = []
    monkeypatch.setattr(ingest_run, "Processor", _Proc)
    monkeypatch.setattr(ingest_run, "_post_batch",
                        lambda db, models, new_trims, phases, on_phase, **k:
                        calls.append((sorted(models), new_trims, k.get("new_final_tests"))))
    res = ingest_run.run_folder(str(tmp_path), db=SimpleNamespace(), config=None)
    assert res.ok and calls == [(["FT1"], 0, 2)]


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
    shutil.copy(REPO / "scripts" / "_db_guard.py", decoy_repo / "scripts" / "_db_guard.py")
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


def test_a_findings_phase_that_failed_for_every_model_does_not_log_as_success(tmp_path, monkeypatch, caplog):
    """Final review: refresh_findings catches per-model failures itself and returns a count, so the
    guard around it almost never fires. Without the report, every model failing logged as
    'refreshed for N models (0 findings)' -- indistinguishable from 'nothing to report'."""
    import logging
    from laser_trim_analyzer.core import ingest_run
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.findings import engine

    db = DatabaseManager(tmp_path / "hook.db")

    def all_fail(_db, models=None, report=None):
        if report is not None:
            report.update({"models": len(models or []), "stored": 0,
                           "failed_models": {m: "RuntimeError: boom" for m in (models or [])},
                           "analyzer_errors": {}})
        return 0
    monkeypatch.setattr(engine, "refresh_findings", all_fail)
    said = []
    phases = {}
    with caplog.at_level(logging.INFO):
        ingest_run._post_batch(db, {"A", "B"}, new_trims=3, phases=phases, on_phase=said.append)
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR
              and "could not be worked out" in r.getMessage()]
    assert len(errors) == 1 and "2 of 2 models" in errors[0], errors
    assert any("could not be worked out" in s for s in said), said
    assert not [r for r in caplog.records if "Process findings refreshed for" in r.getMessage()]
    assert "findings" in phases


def test_a_healthy_findings_phase_still_logs_one_quiet_line(tmp_path, monkeypatch, caplog):
    import logging
    from laser_trim_analyzer.core import ingest_run
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.findings import engine

    db = DatabaseManager(tmp_path / "hook2.db")
    monkeypatch.setattr(engine, "refresh_findings",
                        lambda _db, models=None, report=None: (report or {}).update(
                            {"models": 2, "stored": 4, "failed_models": {}, "analyzer_errors": {}}) or 4)
    with caplog.at_level(logging.INFO):
        ingest_run._post_batch(db, {"A", "B"}, new_trims=3, phases={}, on_phase=lambda _s: None)
    assert [r.getMessage() for r in caplog.records if "Process findings refreshed for" in r.getMessage()]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR and "findings" in r.getMessage().lower()]
