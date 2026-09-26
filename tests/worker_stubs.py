"""Stand-in processors a spawned worker PROCESS can build (ingest-speed Task 11, spec 4.9).

A worker process builds its processor from the class its parent used, by reference -- so a stub
for a process-mode test must live in a module a child can import, never inside a test function.
This one does (the suite puts tests/ on sys.path, and a spawned child is handed its parent's
sys.path). Everything here decides what to do from the FILE NAME, because a child cannot be
monkeypatched from the test:

    slow_*    sleeps a little first (a pool that is really busy)
    stuck_*   never finishes, in a worker process (close must terminate it)
    die_*     kills its worker process with os._exit (the pool breaks mid-run); in the PARENT --
              the thread fallback -- it analyses normally
    warn_*    logs a WARNING carrying the file name and the process id
    reach_get_database_*, reach_manager_get_database_*, reach_manager_*, reach_sqlite_*,
    reach_dbapi2_*
              reaches for a database the way a careless analysis would -- the package's
              get_database, the manager module's own, a manager of its own, sqlite3.connect,
              sqlite3.dbapi2.connect (what SQLAlchemy calls) -- and SWALLOWS the refusal, as the
              real analysis's `except Exception` would, then returns a verdict anyway: the worker
              must return `internal`, never that verdict

Every other file gets an ERROR result from minimal metadata, with the file's own (size, mtime)
and sha256, as `analyse_path` gives. Values are invented; no file is ever parsed.
"""
from __future__ import annotations

import hashlib
import logging
import multiprocessing
import os
import sqlite3
import time
from pathlib import Path

from laser_trim_analyzer.core.processor import Outcome, Processor

REPO = Path(__file__).resolve().parents[1]


def in_a_worker_process() -> bool:
    """True inside a spawned worker, False in the test's own process."""
    return multiprocessing.parent_process() is not None


def checkout_data_dirs():
    """A checkout's real data/ folders -- the ones conftest guards: this tree's and the one the
    package was imported from (they differ in a worktree)."""
    import laser_trim_analyzer.config as _cfg
    return tuple(sorted({str((REPO / "data").resolve()),
                         str((Path(_cfg.__file__).resolve().parents[2] / "data").resolve())}))


def package_file() -> str:
    """Where THIS process imported laser_trim_analyzer from (run in a worker, and compared)."""
    import laser_trim_analyzer
    return str(Path(laser_trim_analyzer.__file__).resolve())


def root_handlers() -> list:
    """The class names of THIS process's root logging handlers (run in a worker)."""
    return [type(h).__name__ for h in logging.getLogger().handlers]


def make_files(folder: Path, names) -> list:
    """Tiny files with these names (each its own bytes), for a folder a run walks."""
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name in names:
        p = folder / name
        p.write_bytes(f"invented: {name}".encode())
        out.append(p)
    return out


class StubProcessor(Processor):
    """The REAL dispatch, pools and writer, with an analysis decided by the file's name."""

    def __init__(self, *a, **k):
        k.setdefault("use_ml", False)
        super().__init__(*a, **k)

    def analyse_path(self, file_path, disk_stat=None):
        path = Path(file_path)
        name = path.name
        if name.startswith("slow"):
            time.sleep(0.05)
        if name.startswith("stuck") and in_a_worker_process():
            time.sleep(600)
        if name.startswith("die") and in_a_worker_process():
            os._exit(3)
        if name.startswith("warn"):
            logging.getLogger("laser_trim_analyzer.tests.worker_stubs").warning(
                "invented warning for %s from process %d", name, os.getpid())
        if name.startswith("reach_"):
            self._reach(path)
        result = self._create_error_result(self._create_minimal_metadata(path), "stub",
                                           time.time())
        stat = file_hash = None
        try:
            st = os.stat(path)
            stat = (st.st_size, st.st_mtime)
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            pass
        return Outcome(path=str(path), result=result, stat=stat, file_hash=file_hash,
                       started=time.time())

    @staticmethod
    def _reach(path: Path) -> None:
        """What a careless analysis would do -- each refusal swallowed, as `except Exception`
        does in the real analysis."""
        name = path.name
        try:
            if name.startswith("reach_get_database"):
                from laser_trim_analyzer.database import get_database
                get_database()
            elif name.startswith("reach_manager_get_database"):
                from laser_trim_analyzer.database import manager
                manager.get_database()
            elif name.startswith("reach_dbapi2"):
                # `as`: a bare `import sqlite3.dbapi2` would make `sqlite3` a local name of this
                # whole function, and the reach_sqlite branch below would never reach anything
                import sqlite3.dbapi2 as dbapi2
                dbapi2.connect(str(path.with_suffix(".dbapi2.db"))).close()
            elif name.startswith("reach_manager"):
                from laser_trim_analyzer.database.manager import DatabaseManager
                DatabaseManager(path.with_suffix(".manager.db"))
            elif name.startswith("reach_sqlite"):
                sqlite3.connect(str(path.with_suffix(".raw.db"))).close()
        except Exception:
            pass


class SlowStartProcessor(StubProcessor):
    """A processor that takes 20 s to build in a worker process -- a worker start as slow as an
    endpoint scanner can make it at work."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if in_a_worker_process():
            time.sleep(20)


class LoudInitProcessor(StubProcessor):
    """A processor whose construction itself reaches for a database: a worker must refuse to
    start with it, by name, rather than analyse anything."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        try:
            from laser_trim_analyzer.database import get_database
            get_database()
        except Exception:
            pass
