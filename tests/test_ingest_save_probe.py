"""The work probe: what a save costs on THIS disk, and where the ingest loop's time goes.

`scripts/ingest_save_probe.py` runs on James's work laptop against the production database
(spec 2026-09-25-ingest-speed-design.md §8 is its contract). So what these tests pin is what
keeps that database untouched and that machine clean:

  * the source is opened ONLY read-only (`mode=ro`) and copied with SQLite's backup API -- its
    bytes and mtime do not move;
  * a copy may never land inside a `data/` folder, nor on a drive without twice the room;
  * every manager the probe builds is on its copy, injected into BOTH globals before any
    `Processor` exists -- the processor reaches the database through `get_database()`;
  * the copy and the temp files are gone at the end, and on Ctrl-C too;
  * its batch writer really is one transaction per batch (spec F4: pysqlite commits at the first
    RELEASE unless the batch opens with an explicit BEGIN -- a probe that fell into that trap
    would print batch numbers that are per-file commits).

Ruling 3 rides along: the older probes copy `model_specs` from a named database, read-only,
because an empty model_specs table makes the analysis a third cheaper than the ingest's (F8).

No test names the real data/analysis.db: every source is a tmp database built from the committed
fixtures.
"""
import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRIM_FIXTURES = ("dlts_8074_18.xls", "dlts_8232-1_242.xls", "lts_8232-1_193.xls")


@pytest.fixture
def probe(monkeypatch):
    """The probe module, imported BY NAME from scripts/ -- the LOOP block's spawned workers
    pickle its functions by reference, so it must be importable in a child as well."""
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import ingest_save_probe
    return ingest_save_probe


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def work(tmp_path):
    """A source database holding the three fixture files (as production holds its files), a
    folder with those files for the probe to take, and an empty temp dir for its copy."""
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg

    folder = tmp_path / "share" / "DLTS"
    folder.mkdir(parents=True)
    for name in TRIM_FIXTURES:
        shutil.copy2(REPO / "tests" / "fixtures" / "trim" / name, folder / name)
    src = tmp_path / "src" / "source.db"
    src.parent.mkdir()
    before = (mgr._db_manager, getattr(dbpkg, "_db_manager", None))
    db = mgr.DatabaseManager(src)
    mgr._db_manager = db
    dbpkg._db_manager = db
    try:
        db.save_model_spec({"model": "8074", "linearity_type": "independent"})
        proc = Processor(use_ml=False)
        for f in sorted(folder.iterdir()):
            r = proc.process_file(f)
            if r is not None:
                db.save_analysis(r)
    finally:
        db.close()
        mgr._db_manager, dbpkg._db_manager = before
    tmp = tmp_path / "probe_tmp"
    tmp.mkdir()
    return {"folder": folder, "src": src, "tmp": tmp}


def _run(probe, work, *extra):
    return probe.main([str(work["folder"]), "3", "--db", str(work["src"]),
                       "--tmp", str(work["tmp"]), *extra])


def _copy_path(work) -> Path:
    return work["tmp"] / f"ingest_save_probe_{os.getpid()}.db"


# ---- refusals --------------------------------------------------------------------------------

def test_a_copy_inside_a_data_folder_is_refused(probe, work, capsys):
    data = work["tmp"] / "data"
    data.mkdir()
    rc = probe.main([str(work["folder"]), "3", "--no-loop", "--db", str(work["src"]),
                     "--tmp", str(data)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "REFUSED" in out and "data" in out
    assert list(data.iterdir()) == []            # refused BEFORE anything was written


def test_a_drive_without_twice_the_database_free_is_refused(probe, work, capsys, monkeypatch):
    need = work["src"].stat().st_size * 2
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda p: usage(10 * need, 9 * need, need - 1))
    rc = _run(probe, work, "--no-loop")
    out = capsys.readouterr().out
    assert rc == 2
    assert "REFUSED" in out and "free" in out
    assert list(work["tmp"].iterdir()) == []


# ---- the source is only ever read ---------------------------------------------------------------

def test_the_source_is_opened_only_read_only_and_never_changes(probe, work, monkeypatch):
    seen = []
    real_connect, real_dbapi_connect = sqlite3.connect, sqlite3.dbapi2.connect

    def spy(real):
        def connect(database, *a, **k):
            seen.append((str(database), bool(k.get("uri"))))
            return real(database, *a, **k)
        return connect
    monkeypatch.setattr(sqlite3, "connect", spy(real_connect))
    monkeypatch.setattr(sqlite3.dbapi2, "connect", spy(real_dbapi_connect))
    sha, mtime = _sha(work["src"]), work["src"].stat().st_mtime_ns

    assert _run(probe, work, "--no-loop") == 0

    src_name = work["src"].name
    to_source = [(d, uri) for d, uri in seen if src_name in d]
    assert to_source, "the probe never opened its source -- this test would prove nothing"
    for d, uri in to_source:
        assert uri and "mode=ro" in d, f"source opened without mode=ro: {d}"
    assert _sha(work["src"]) == sha
    assert work["src"].stat().st_mtime_ns == mtime


def test_every_manager_is_on_the_copy_and_injected_before_any_processor(probe, work, monkeypatch):
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg

    copy = _copy_path(work).resolve()
    managers, processors, reached = [], [], []
    real_mgr_init, real_proc_init, real_get = (mgr.DatabaseManager.__init__, Processor.__init__,
                                               mgr.get_database)

    def mgr_init(self, database_path=None, *a, **k):
        managers.append(None if database_path is None else Path(database_path).resolve())
        return real_mgr_init(self, database_path, *a, **k)

    def on_copy(m):
        return m is not None and Path(m.database_path).resolve() == copy

    def proc_init(self, *a, **k):
        processors.append(on_copy(mgr._db_manager) and dbpkg._db_manager is mgr._db_manager)
        return real_proc_init(self, *a, **k)

    def get_database():
        # A fall-through would build a manager at the configured default; the processor swallows
        # what that raises, so RECORD it rather than rely on an exception reaching the test.
        reached.append(on_copy(mgr._db_manager))
        return real_get()

    monkeypatch.setattr(mgr.DatabaseManager, "__init__", mgr_init)
    monkeypatch.setattr(Processor, "__init__", proc_init)
    # Both names: the processor imports it from the PACKAGE (`from laser_trim_analyzer.database
    # import get_database`), which holds its own reference to the function.
    monkeypatch.setattr(mgr, "get_database", get_database)
    monkeypatch.setattr(dbpkg, "get_database", get_database)

    assert _run(probe, work, "--no-loop") == 0

    assert managers and all(p == copy for p in managers), managers
    assert processors and all(processors), processors
    assert reached and all(reached), reached


# ---- what it prints, and what it leaves behind ---------------------------------------------

ROW = re.compile(r"^\s+(1|10|50)  (FULL|NORMAL|OFF)\s+(2|64)MB(\s+\d+\.\d){5}")


def test_a_three_file_run_prints_every_save_line_and_leaves_nothing(probe, work, capsys):
    assert _run(probe, work, "--no-loop") == 0
    out = capsys.readouterr().out
    lines = out.splitlines()

    assert lines[0].startswith("ingest save probe  ")
    assert lines[1].startswith("copy   ") and "(read-only)" in lines[1]
    assert lines[2].startswith("files  3 of 3 from ") and "first in discovery order" in lines[2]
    assert "SAVE  3 new results per setting, median of 2 rounds, ms per file" in lines
    assert "batch  sync    cache   total  python   sql  commit    p99" in lines
    rows = [ln for ln in lines if ROW.match(ln)]
    got = {(m.group(1), m.group(2), m.group(3)) for m in map(ROW.match, rows)}
    want = {(b, s, c) for b in ("1", "10", "50") for s in ("FULL", "NORMAL") for c in ("2", "64")}
    assert want | {("1", "OFF", "2")} == got
    assert len(rows) == 13
    assert rows[0].split()[:3] == ["1", "FULL", "2MB"] and rows[0].endswith("<- today")
    assert rows[-1].split()[:3] == ["1", "OFF", "2MB"]
    assert rows[-1].endswith("<- reference only: no flush at all")
    assert "LOOP" not in out                      # --no-loop stops after SAVE
    assert lines[-1] == "copy and temp files deleted."
    assert list(work["tmp"].iterdir()) == []


def test_ctrl_c_still_deletes_the_copy_and_the_temp_files(probe, work, capsys, monkeypatch):
    def interrupted(*a, **k):
        assert _copy_path(work).exists()          # the copy was made; now it must go
        raise KeyboardInterrupt
    monkeypatch.setattr(probe, "_run_save_block", interrupted)
    rc = _run(probe, work, "--no-loop")
    out = capsys.readouterr().out
    assert rc == 130
    assert "interrupted" in out
    assert list(work["tmp"].iterdir()) == []


def test_keep_keeps_the_copy_and_says_where(probe, work, capsys):
    assert _run(probe, work, "--no-loop", "--keep") == 0
    out = capsys.readouterr().out
    copy = _copy_path(work)
    assert copy.exists() and str(copy) in out.splitlines()[-1]
    assert [p.name for p in work["tmp"].iterdir() if p.is_dir()] == []    # temp files still go


def test_the_full_run_prints_the_loop_block(probe, work, capsys):
    assert _run(probe, work, "--procs", "2") == 0
    lines = capsys.readouterr().out.splitlines()
    i = lines.index("LOOP  the same 3 files, local copies, model specs from the copy, ms per file")
    labels = ["parse+analyse, one at a time, without specs (what pool_probe measured)",
              "parse+analyse, one at a time, with specs (what the ingest does)",
              "parse+analyse, 4 threads, with specs",
              "parse+analyse, 2 processes, with specs (pool ready in ",
              "today's loop, headless (run_folder)"]
    for label, line in zip(labels, lines[i + 1:i + 6]):
        assert line.startswith(label), line
        assert re.search(r"\d+\.\d$", line), line
    assert re.match(r"^   of which save: wall \d+\.\d, cpu \d+\.\d \| GC pauses \d+\.\d \| rest -?\d+\.\d$",
                    lines[i + 6]), lines[i + 6]
    assert lines[-1] == "copy and temp files deleted."
    assert list(work["tmp"].iterdir()) == []


# ---- the batch writer the probe carries until Task 6 --------------------------------------

def test_the_probes_batch_writer_is_one_transaction_per_batch(probe, tmp_path):
    """Spec F4: without an explicit BEGIN, pysqlite's first RELEASE commits, and every later
    SAVEPOINT opens a fresh transaction -- a 'batch' that is really a commit per file."""
    from sqlalchemy import event
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg

    before = (mgr._db_manager, getattr(dbpkg, "_db_manager", None))
    db = mgr.DatabaseManager(tmp_path / "w.db")
    mgr._db_manager = db
    dbpkg._db_manager = db
    try:
        proc = Processor(use_ml=False)
        results = [proc.process_file(REPO / "tests" / "fixtures" / "trim" / n) for n in TRIM_FIXTURES]
        in_txn = []

        @event.listens_for(db._engine, "savepoint")
        def _sp(conn, name):
            in_txn.append(conn.connection.dbapi_connection.in_transaction)

        outcomes = probe._save_batch_probe(db, results)
        assert [o[0] for o in outcomes] == ["saved"] * 3
        # the 2nd and 3rd SAVEPOINTs must open INSIDE the batch's one transaction
        assert len(in_txn) == 3 and all(in_txn[1:]), in_txn
    finally:
        db.close()
        mgr._db_manager, dbpkg._db_manager = before


# ---- ruling 3: the older probes measure the analysis the ingest really runs -----------------

def test_copy_model_specs_brings_every_spec_and_only_reads_the_source(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import _probe_specs
    from laser_trim_analyzer.database import manager as mgr

    src = tmp_path / "named.db"
    s = mgr.DatabaseManager(src)
    s.save_model_spec({"model": "8074", "linearity_type": "independent", "aliases": "8074X"})
    s.save_model_spec({"model": "8232-1", "linearity_type": "absolute"})
    s.close()
    sha, mtime = _sha(src), src.stat().st_mtime_ns
    dest = tmp_path / "throwaway.db"
    d = mgr.DatabaseManager(dest)
    try:
        assert d.get_model_spec("8074") is None
        assert _probe_specs.copy_model_specs(src, dest) == 2
        assert d.get_model_spec("8074")["linearity_type"] == "independent"
        assert d.get_model_spec("8074X")["model"] == "8074"          # aliases travel too
        assert d.get_model_spec("8232-1")["linearity_type"] == "absolute"
    finally:
        d.close()
    assert _sha(src) == sha and src.stat().st_mtime_ns == mtime


@pytest.mark.parametrize("script", ["ingest_speed_probe.py", "parse_locality_probe.py"])
def test_the_older_probes_say_where_their_specs_came_from(script, work):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / script), str(work["folder"]), "2",
                        "--specs-from", str(work["src"])],
                       capture_output=True, text=True, timeout=300, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"model specs: 1 from {work['src']} (read-only)" in r.stdout, r.stdout


def test_pool_probe_workers_get_the_specs_too(tmp_path, work, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import pool_probe
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg

    before = (mgr._db_manager, getattr(dbpkg, "_db_manager", None))
    monkeypatch.setattr(pool_probe, "_PROC", None)
    try:
        pool_probe._use_scratch_db(str(tmp_path), str(work["src"]))
        proc = pool_probe._processor()
        assert mgr._db_manager is dbpkg._db_manager                  # BOTH globals
        assert Path(mgr._db_manager.database_path).parent == tmp_path
        assert mgr._db_manager.get_model_spec("8074") is not None
        assert proc is pool_probe._processor()
    finally:
        if mgr._db_manager is not before[0]:
            mgr._db_manager.close()
        mgr._db_manager, dbpkg._db_manager = before
