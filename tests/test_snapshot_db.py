"""The snapshot tool must carry what a plain file copy loses.

2026-09-22: a finished rebuild was carried home through OneDrive by syncing the
database's folder. The database, its -wal journal and the logs arrived at three
different times, and what came down was an older database beside a journal from
another moment. `scripts/snapshot_db.py` exists so there is only ever ONE file to
carry. These tests pin the properties that make that true.
"""
import hashlib
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = str(REPO / "scripts" / "snapshot_db.py")


def _left_behind_db_with_rows_only_in_the_journal(tmp_path):
    """The state an app leaves when it exits WITHOUT checkpointing: analysis.db plus a
    -wal holding the newest committed rows, and no connection open on either.

    Built by writing through a live connection with auto-checkpoint off, then
    freezing the three files as they stand. It must be a frozen copy with NOTHING
    open on it: closing the last connection checkpoints the journal into the main
    file, and a writer left open would stop the snapshot's own connection from being
    the last -- which would hide a snapshot that opened the source read-WRITE.
    """
    live = tmp_path / "live.db"
    w = sqlite3.connect(live)
    w.execute("PRAGMA journal_mode=WAL")
    w.execute("PRAGMA wal_autocheckpoint=0")          # nothing folds in by itself
    w.execute("create table analysis_results (id integer primary key, v text)")
    w.execute("create table trim_passes (id integer primary key, v text)")
    w.commit()
    w.execute("PRAGMA wal_checkpoint(TRUNCATE)")        # the SCHEMA is in the main file
    w.executemany("insert into analysis_results(v) values (?)", [("a",)] * 500)
    w.executemany("insert into trim_passes(v) values (?)", [("p",)] * 300)
    w.commit()                                          # the ROWS are only in the journal
    src = tmp_path / "analysis_left_behind.db"
    for suffix in ("", "-wal", "-shm"):
        shutil.copy(str(live) + suffix, str(src) + suffix)
    w.close()
    assert Path(str(src) + "-wal").stat().st_size > 0
    return src


def _run(src, dst):
    return subprocess.run([sys.executable, SCRIPT, str(src), str(dst)],
                          capture_output=True, text=True, timeout=120)


def test_rows_that_live_only_in_the_journal_reach_the_snapshot(tmp_path):
    src = _left_behind_db_with_rows_only_in_the_journal(tmp_path)
    # Prove the premise: the main file ALONE does not have the rows. This is
    # exactly what a plain copy of analysis.db without its -wal would carry.
    bare = tmp_path / "bare.db"
    shutil.copy(src, bare)
    b = sqlite3.connect(bare)
    assert b.execute("select count(*) from analysis_results").fetchone()[0] == 0
    b.close()

    dst = tmp_path / "snap.db"
    r = _run(src, dst)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "GOOD: one file" in r.stdout

    s = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
    assert s.execute("select count(*) from analysis_results").fetchone()[0] == 500
    assert s.execute("select count(*) from trim_passes").fetchone()[0] == 300
    assert s.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    s.close()


def test_the_snapshot_is_one_file_with_no_journal_beside_it(tmp_path):
    src = _left_behind_db_with_rows_only_in_the_journal(tmp_path)
    dst = tmp_path / "snap.db"
    assert _run(src, dst).returncode == 0
    assert dst.exists()
    assert not dst.with_name(dst.name + "-wal").exists()
    assert not dst.with_name(dst.name + "-shm").exists()


def test_the_original_is_never_written(tmp_path):
    src = _left_behind_db_with_rows_only_in_the_journal(tmp_path)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    assert _run(src, tmp_path / "snap.db").returncode == 0
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before


def test_it_never_overwrites_an_existing_file(tmp_path):
    src = _left_behind_db_with_rows_only_in_the_journal(tmp_path)
    dst = tmp_path / "snap.db"
    dst.write_bytes(b"something precious")
    r = _run(src, dst)
    assert r.returncode == 2 and "already exists" in r.stdout
    assert dst.read_bytes() == b"something precious"


def test_it_refuses_to_snapshot_a_database_onto_itself(tmp_path):
    src = _left_behind_db_with_rows_only_in_the_journal(tmp_path)
    r = _run(src, src)
    assert r.returncode == 2 and "cannot be the database itself" in r.stdout
