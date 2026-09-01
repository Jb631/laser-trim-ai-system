"""The QA harnesses refuse the production database.

Both harnesses open their target READ-WRITE (DatabaseManager's engine setup
commits index creation into the file). Their old no-argument default was the
real `data/analysis.db`, and on 2026-08-31 three separate sessions opened the
production database by accident through exactly that default. These tests pin
the refusal via subprocess — the scripts stub `tkinter` into `sys.modules` at
import, so importing them inside pytest would poison this process.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROD = REPO / "data" / "analysis.db"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True,
                          text=True, cwd=REPO, timeout=120)


def test_sweep_refuses_the_production_database_by_name():
    r = _run("scripts/app_qa_sweep.py", str(PROD))
    assert r.returncode != 0
    assert "PRODUCTION" in r.stdout


def test_sweep_requires_a_db_argument():
    r = _run("scripts/app_qa_sweep.py")
    assert r.returncode != 0
    assert "required" in r.stdout


def test_chart_harness_refuses_the_production_database_by_name(tmp_path):
    r = _run("scripts/chart_qa_render_all.py", str(tmp_path / "out"), str(PROD))
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
