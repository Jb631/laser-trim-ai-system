"""SQLite pragmas every connection carries (ingest-speed spec 3.8, ruling 12).

`_set_sqlite_pragma` (database/manager.py) must never be deleted or renamed -- CLAUDE.md:
it is how foreign keys are enforced. This adds `synchronous=NORMAL` (WAL's documented safe
setting: the WAL is synced before every checkpoint, not at each commit, so a power cut can
roll back only the batches since the last checkpoint, and never corrupts the file -- a
re-runnable ingest is what makes that trade free) and `cache_size=-65536` (64 MiB) beside it.

*Where it actually has to run.* The app has exactly one SQLite connection (F14: StaticPool).
Registering `_set_sqlite_pragma` as a "connect" event listener only matters if that listener
is attached BEFORE the pool creates its one-and-only DBAPI connection -- StaticPool creates it
lazily, on the first checkout, and fires "connect" at that moment. Registered afterward (the
order this code had until this fix), the listener is provably never invoked in production: this
project has exactly one `create_engine()` call, always StaticPool, so there was never a second
connection for it to catch either. These tests exercise the REAL DatabaseManager, not the
listener function in isolation, so a regression that moves the registration back below the
first checkout -- restoring the old dead-listener bug -- turns them red.
"""
from pathlib import Path


def test_a_fresh_database_manager_sets_all_three_pragmas_on_its_real_connection(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "x.db")
    try:
        with db._engine.connect() as conn:
            raw = conn.connection.dbapi_connection
            assert raw.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert raw.execute("PRAGMA synchronous").fetchone()[0] == 1       # NORMAL
            assert raw.execute("PRAGMA cache_size").fetchone()[0] == -65536   # 64 MiB
    finally:
        db.close()


def test_the_session_the_app_actually_writes_through_carries_the_same_pragmas(tmp_path):
    """session() is what every real read and write goes through. With StaticPool this is
    the SAME one connection as the test above, but asserting it again here pins the path
    that matters -- not an incidental side connection nobody uses."""
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "x.db")
    try:
        with db.session() as session:
            raw = session.connection().connection.dbapi_connection
            assert raw.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert raw.execute("PRAGMA synchronous").fetchone()[0] == 1
            assert raw.execute("PRAGMA cache_size").fetchone()[0] == -65536
    finally:
        db.close()


def test_pragmas_hold_across_repeated_checkouts_not_just_the_first(tmp_path):
    """Not a one-shot fluke of the very first connect: later checkouts see the same
    settings too, without assuming StaticPool's one-connection detail by name."""
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "x.db")
    try:
        for _ in range(3):
            with db._engine.connect() as conn:
                raw = conn.connection.dbapi_connection
                assert raw.execute("PRAGMA synchronous").fetchone()[0] == 1
                assert raw.execute("PRAGMA cache_size").fetchone()[0] == -65536
    finally:
        db.close()


def test_two_independent_database_managers_each_get_all_three(tmp_path):
    """Every manager passes through __init__ -- a second one, on a second file, must not
    silently ride on whatever the first happened to set (no shared global pragma state)."""
    from laser_trim_analyzer.database.manager import DatabaseManager

    db_a = DatabaseManager(tmp_path / "a.db")
    db_b = DatabaseManager(tmp_path / "b.db")
    try:
        for db in (db_a, db_b):
            with db._engine.connect() as conn:
                raw = conn.connection.dbapi_connection
                assert raw.execute("PRAGMA foreign_keys").fetchone()[0] == 1
                assert raw.execute("PRAGMA synchronous").fetchone()[0] == 1
                assert raw.execute("PRAGMA cache_size").fetchone()[0] == -65536
    finally:
        db_a.close()
        db_b.close()


def test_set_sqlite_pragma_listener_is_registered_before_the_first_checkout():
    """Pins the ORDERING fix directly: if this ever regresses back to registering the
    listener after the constructor's first checkout, this is the test that names it
    rather than leaving it to a silent pragma mismatch somewhere downstream.

    Reads the FILE, not `inspect.getsource(DatabaseManager.__init__)` -- the suite's
    own autouse `_never_touch_the_real_database` fixture (conftest.py) replaces
    `__init__` with a guard wrapper for every test, so the live method object's
    source is that wrapper's, not this one's.
    """
    from laser_trim_analyzer.database import manager as mgr

    src = Path(mgr.__file__).read_text()
    start = src.index("    def __init__(self")
    end = src.index("\n    def _init_database")
    assert start < end
    body = src[start:end]

    listener_pos = body.index("event.listens_for")
    # the first real checkout the constructor makes after wiring the engine
    first_connect_pos = body.index("self._engine.connect()")
    assert listener_pos < first_connect_pos, (
        "the pragma listener must be registered before the constructor's first "
        "engine.connect() -- StaticPool creates its one connection lazily on that "
        "first checkout, and a listener registered after it never fires")
