"""The QA harnesses refuse the production database.

Both harnesses open their target READ-WRITE (DatabaseManager's engine setup
commits index creation into the file). Their old no-argument default was the
real `data/analysis.db`, and on 2026-08-31 three separate sessions opened the
production database by accident through exactly that default. These tests pin
the refusal via subprocess — the scripts stub `tkinter` into `sys.modules` at
import, so importing them inside pytest would poison this process.

NO TEST IN THIS FILE MAY NAME THE REAL data/analysis.db. The refusal under
test here is scripts/_db_guard.py, shared by every tool below — it is the
code under test, so if it ever regressed, a test that put the real path in a
subprocess argv would open the owner's 3.7 GB work database READ-WRITE before
the test could go red. Every case below uses a decoy literally named
`analysis.db` under tmp_path instead: all five guarded tools refuse by NAME
(not just by path identity), so a tmp_path decoy trips the same refusal the
real file would, without ever being it.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True,
                          text=True, cwd=REPO, timeout=120)


def test_sweep_refuses_the_production_database_by_name(tmp_path):
    r = _run("scripts/app_qa_sweep.py", str(tmp_path / "analysis.db"))
    assert r.returncode != 0
    assert "PRODUCTION" in r.stdout


def test_sweep_requires_a_db_argument():
    r = _run("scripts/app_qa_sweep.py")
    assert r.returncode != 0
    assert "required" in r.stdout


def test_chart_harness_refuses_the_production_database_by_name(tmp_path):
    r = _run("scripts/chart_qa_render_all.py", str(tmp_path / "out"), str(tmp_path / "analysis.db"))
    assert r.returncode != 0
    assert "PRODUCTION" in r.stdout


def test_chart_harness_requires_both_arguments():
    r = _run("scripts/chart_qa_render_all.py")
    assert r.returncode != 0
    assert "required" in r.stdout


def test_a_copy_is_still_accepted(tmp_path):
    """The guard must refuse ONE path, not make the harness unusable: a
    nonexistent copy path must get past the refusal and die on the
    existence check instead (proof the refusal matched by path, not mood)."""
    ghost = tmp_path / "copy.db"
    r = _run("scripts/app_qa_sweep.py", str(ghost))
    assert "PRODUCTION" not in r.stdout
    assert "no database at" in r.stdout


def test_ui_stall_probe_refuses_the_production_database_by_name(tmp_path):
    r = _run("scripts/ui_stall_probe.py", str(tmp_path / "analysis.db"))
    assert r.returncode != 0
    # this tool's refusal is a bare `raise SystemExit(message)`, which Python
    # prints to stderr (not stdout) for a non-integer exit code.
    assert "PRODUCTION" in r.stderr


def test_build_dev_db_refuses_the_production_database_by_name(tmp_path):
    r = _run("scripts/build_dev_db.py", str(tmp_path), str(tmp_path / "analysis.db"))
    assert r.returncode != 0
    assert "REFUSED" in r.stdout
