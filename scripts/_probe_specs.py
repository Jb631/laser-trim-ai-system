"""Give a probe's throwaway database the model specs of a NAMED one -- reading the source only.

Why (spec docs/superpowers/specs/2026-09-25-ingest-speed-design.md, F8 and ruling 3): a fresh
DatabaseManager database has an EMPTY model_specs table, and the analysis the ingest runs depends
on it. On the same 240 DLTS files the spec-driven analysis costs ~48% more (67 -> 99 ms/file on
the Mac) and stores different numbers for 156 of 356 results. Every probe that timed "parsing" on
a throwaway database was timing a cheaper, different analysis -- which is how A0's 339 ms/file at
work came to be read as the parse cost when the ingest's real one is higher.

The source is opened with `mode=ro` and never written; rows go into the throwaway database through
its own plain connection. Columns are matched by name, so a source older or newer than this code
still copies every column the two schemas share.
"""
import sqlite3
from pathlib import Path


def default_source(repo) -> Path:
    """What the probes read specs from unless told otherwise: this checkout's own database."""
    return Path(repo) / "data" / "analysis.db"


def _open_readonly(source: Path) -> sqlite3.Connection:
    return sqlite3.connect(Path(source).resolve().as_uri() + "?mode=ro", uri=True)


def copy_model_specs(source, dest) -> int:
    """Copy every model_specs row from `source` (read-only) into `dest`. Returns how many.

    0 when the source does not exist or has no model_specs table -- the caller says so (see
    `describe`), because a probe running spec-less must never look like one that is not.
    """
    source, dest = Path(source), Path(dest)
    if not source.is_file():
        return 0
    src = _open_readonly(source)
    try:
        tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "model_specs" not in tables:
            return 0
        src_cols = [r[1] for r in src.execute("PRAGMA table_info(model_specs)")]
        rows = src.execute("SELECT " + ", ".join(f'"{c}"' for c in src_cols)
                           + " FROM model_specs").fetchall()
    finally:
        src.close()
    if not rows:
        return 0
    dst = sqlite3.connect(str(dest))
    try:
        dst_cols = {r[1] for r in dst.execute("PRAGMA table_info(model_specs)")}
        cols = [c for c in src_cols if c in dst_cols]
        keep = [src_cols.index(c) for c in cols]
        dst.executemany(
            "INSERT OR REPLACE INTO model_specs (" + ", ".join(f'"{c}"' for c in cols) + ") VALUES ("
            + ", ".join("?" * len(cols)) + ")",
            [tuple(r[i] for i in keep) for r in rows])
        dst.commit()
    finally:
        dst.close()
    return len(rows)


def describe(count: int, source) -> str:
    """The one line a probe prints about its specs -- loud when there are none."""
    if count:
        return f"model specs: {count} from {source} (read-only)"
    return (f"model specs: NONE from {source} -- this times a CHEAPER analysis than the ingest "
            "runs; pass --specs-from <a copy of the work database>")


def specs_from_argv(argv, repo):
    """(`--specs-from` value or the default, argv without it) -- the probes keep their positional
    usage, so this is pulled out by hand rather than through argparse."""
    argv = list(argv)
    if "--specs-from" in argv:
        i = argv.index("--specs-from")
        if i + 1 >= len(argv):
            raise SystemExit("--specs-from needs a database path")
        value = Path(argv[i + 1])
        del argv[i:i + 2]
        return value, argv
    return default_source(repo), argv
