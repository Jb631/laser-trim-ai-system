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


def test_the_command_line_tool_refuses_the_production_database():
    import subprocess, sys
    r = subprocess.run([sys.executable, "scripts/refresh_findings.py", "data/analysis.db"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "REFUSED" in r.stdout
