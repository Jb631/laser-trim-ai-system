"""Every field that existed before capture must be byte-identical after it.

The method: run the same real files through the tree as it was before this
work and through the tree as it is now, into two throwaway databases, and
diff every stored column. This caught two real bugs in September 2026 and is
the only reason a parser change of this size is safe.

How the "before" side is produced
---------------------------------
NOT with `git stash`. The capture work is committed, so a stash captures the
NEW tree and the proof would compare the new tree against itself: green, and
worthless. Worktrees are banned in this project. So the pre-capture tree is
extracted from the commit before Task 1 and imported from there:

    mkdir -p /tmp/precap && git archive adcc1ce | tar -x -C /tmp/precap
    .venv/bin/python tests/test_trim_capture_noop.py --write-baseline \\
        --root /tmp/precap

The FIXTURES and the baseline file both come from THIS tree; only the code
under test is swapped. `adcc1ce` predates the fixtures, so it has none.

Why `--write-baseline` refuses to guess where it imported from
--------------------------------------------------------------
The venv carries an editable install of this repo's `src`. If that ever wins
over the injected path, the baseline is silently taken from the NEW tree, the
test passes trivially, and the whole no-op proof reads green while proving
nothing. So the capture asserts its own provenance and aborts loudly when the
import did not come from the root it was told to use. It prints the file it
imported, which belongs in the record of any run that regenerates this.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# NOT parse_baseline.json: that is the 645-real-file parse baseline driven by
# tests/test_parse_all_models.py, a different and valuable artifact.
BASELINE = REPO / "tests" / "fixtures" / "parse_baseline_trim.json"
FIXTURES = sorted((REPO / "tests" / "fixtures" / "trim").glob("*.xls"))

PRE_EXISTING = [
    "track_id", "status", "linearity_spec", "linearity_pass", "linearity_fail_points",
    "sigma_gradient", "sigma_threshold", "sigma_pass", "optimal_offset", "optimal_slope",
    "untrimmed_resistance", "trimmed_resistance", "resistance_change",
    "resistance_change_percent", "travel_length", "unit_length", "trim_pass_count",
    "untrimmed_rms_error", "untrimmed_error_max", "trimmed_rms_error",
    "position_data", "error_data", "upper_limits", "lower_limits",
    "untrimmed_positions", "untrimmed_errors",
]


def test_no_pre_existing_field_changed():
    """Compares against a baseline captured from the pre-capture tree.

    Regenerate ONLY from a commit before this work — see the module docstring.
    A failure here is a real regression. Never regenerate to make it green.

    Neither missing fixtures nor a missing baseline is a skip. Both artifacts
    are TRACKED, so absence means a broken checkout, and this is the one test
    whose entire value IS the artifact: skipping it turns the proof into a
    green line that proves nothing.
    """
    assert FIXTURES, (
        f"tests/fixtures/trim holds no .xls files; they are tracked, so this "
        f"is a broken checkout, not an optional extra")
    assert BASELINE.exists(), (
        f"{BASELINE.name} is missing. It is tracked. Regenerate it ONLY from "
        f"the pre-capture tree — see the module docstring — never from HEAD.")
    expected = json.loads(BASELINE.read_text())
    actual = _normalise(_dump())
    # Weak-assertion trap: an empty dump must never read green. If the dump
    # silently produced nothing, every per-field comparison below would be
    # vacuous, so pin the population first.
    assert len(actual) >= len(FIXTURES), (
        f"only {len(actual)} track rows from {len(FIXTURES)} fixtures")
    assert set(actual) == set(expected), "a fixture file appeared or vanished"
    diffs = []
    for name, fields in expected.items():
        for key in PRE_EXISTING:
            if not _same(fields.get(key), actual[name].get(key)):
                diffs.append(f"{name}.{key}: {fields.get(key)!r} -> {actual[name].get(key)!r}")
    assert not diffs, "capture changed pre-existing values:\n" + "\n".join(diffs[:20])


def _same(a, b):
    """Equality that treats NaN as equal to NaN.

    `nan != nan`, so a plain `!=` would report a permanent, unfixable diff on
    any column that legitimately stores NaN — and these exact 8232-1 files are
    the ones that produce it: `max()` returns NaN when element 0 is NaN, and
    they open with an unmeasured lead-in. A baseline NaN compared against the
    identical NaN must read "unchanged", because it IS unchanged.
    """
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:       # both NaN
            return True
    return a == b


def _normalise(dump):
    """Both sides through the same JSON round-trip, so neither is favoured.

    The baseline is stored as JSON; comparing a live Python value against a
    JSON-decoded one would report differences that are only encoding. Running
    the live side through the identical transform removes that without
    loosening anything — it is the same function applied to both.
    """
    return json.loads(json.dumps(dump, default=str))


def _check_import_root(expected_root: Path) -> Path:
    """Abort unless `laser_trim_analyzer` really came from `expected_root`.

    Ruling 19 of Task 9: without this the editable install can win and the
    baseline is captured from the tree it is supposed to be compared against.
    """
    import laser_trim_analyzer
    src = Path(laser_trim_analyzer.__file__).resolve()
    root = expected_root.resolve()
    if root not in src.parents:
        raise SystemExit(f"ABORT: imported {src}, expected under {root}")
    print(f"importing from: {src}")
    return src


def _dump():
    """{filename::track_id: {field: value}} for every fixture, via the pipeline."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg

    out = {}
    # Both globals are RESTORED afterwards, the same saved/finally idiom the
    # sweep uses. Leaving them pointing at a deleted tmp database is how the
    # next thing in the process ends up constructing a fresh manager at the
    # config default — the production path — which is the exact hazard the
    # injection exists to prevent. `dbpkg._db_manager` is restored by nobody
    # else, since nothing but this file sets it.
    saved_mgr = getattr(mgr, "_db_manager", None)
    saved_pkg = getattr(dbpkg, "_db_manager", None)
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "noop.db"
        db = mgr.DatabaseManager(db_path)
        try:
            # BOTH globals. `get_database()` ignores a manager handed in
            # elsewhere and, when its global is unset, CONSTRUCTS one at the
            # config default — which in a normal checkout is the production
            # work database, opened read-write. Hit three times in this project.
            mgr._db_manager = db
            dbpkg._db_manager = db
            # A bare Config(), not get_config(): get_config() reads config.yaml
            # from the app directory, which differs between the extracted
            # pre-capture tree and this one. Dataclass defaults are identical in
            # both (config.py is untouched by the capture work), so this makes
            # the two runs differ ONLY in the code being proved.
            cfg = Config()
            cfg.database.path = db_path
            proc = Processor(config=cfg, use_ml=False)
            for f in FIXTURES:
                result = proc.process_file(f, generate_plots=False)
                assert result is not None, f"pipeline returned nothing for {f.name}"
                db.save_analysis(result)
            import sqlalchemy as sa
            with db.session() as s:
                cols = ", ".join(f"t.{c}" for c in PRE_EXISTING)
                for row in s.execute(sa.text(
                        f"SELECT a.filename, {cols} FROM track_results t "
                        "JOIN analysis_results a ON a.id = t.analysis_id")):
                    out[f"{row[0]}::{row[1]}"] = dict(zip(PRE_EXISTING, row[1:]))
        finally:
            mgr._db_manager = saved_mgr
            dbpkg._db_manager = saved_pkg
            db.close()
    return out


USAGE = (
    "usage: python tests/test_trim_capture_noop.py --write-baseline "
    "--root /tmp/precap\n"
    "  mkdir -p /tmp/precap && git archive adcc1ce | tar -x -C /tmp/precap\n"
    "--root is REQUIRED and must NOT be this repo: a baseline taken from the\n"
    "tree it is meant to be compared against proves nothing, and reads green\n"
    "forever after."
)


if __name__ == "__main__":
    if "--write-baseline" in sys.argv:
        # `--root` is mandatory, and defaulting it to REPO was the whole hole:
        # dropping the flag — the brief's own Step 2 command minus one
        # argument — used to write a self-referential baseline that passed the
        # provenance check, passed the test, and proved nothing. The likelier
        # operator error by far, since this file's own rule is "never
        # regenerate the baseline to make it green" and the bare command is
        # the obvious thing to reach for.
        if "--root" not in sys.argv:
            raise SystemExit("ABORT: --root is required.\n" + USAGE)
        try:
            root = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()
        except IndexError:
            raise SystemExit("ABORT: --root needs a path.\n" + USAGE)
        if root == REPO.resolve():
            raise SystemExit(
                f"ABORT: --root is this repo ({root}). The baseline must come "
                f"from the tree BEFORE the capture work, not from HEAD.\n"
                + USAGE)
        if not (root / "src" / "laser_trim_analyzer").is_dir():
            raise SystemExit(f"ABORT: no src/laser_trim_analyzer under {root}\n"
                             + USAGE)
        # Ahead of the editable install's .pth entry, and ahead of anything
        # else on the path. Nothing from laser_trim_analyzer may be imported
        # before this line.
        sys.path.insert(0, str(root / "src"))
        _check_import_root(root)
        BASELINE.write_text(json.dumps(_normalise(_dump()), indent=1) + "\n")
        print(f"baseline written: {BASELINE}")
    else:
        raise SystemExit(USAGE)
