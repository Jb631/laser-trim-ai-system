"""The walk's stat dict must be keyed the way the processor looks it up.

Work incident, 2026-09-14. Every incremental run at the work machine logged

    check 1.4s (0 known in memory) | verify 171,006 files 542.0s

— a nine-minute stat() storm over the SMB share on a folder where nothing had
changed, on every run, before and after the pull. Nothing was wrong with the
index: `discover_excel_files` keyed its (size, mtime) dict by `entry.path`,
and the configured folders come from Tk's `askdirectory`, which hands back
FORWARD slashes on Windows. So the walk produced the mixed form

    //192.168.66.9/…/Test Station\\6607\\file.xls

while `Processor._classify_scan` looks up `str(Path(file))` — the all-backslash
form, which is also the form stored in `analysis_results.file_path`:

    \\\\192.168.66.9\\…\\Test Station\\6607\\file.xls

The key never matched, so every file classified "needs_hash" and paid a real
stat() over the share.

macOS cannot reproduce the slash flavour, but it reproduces the SHAPE of the
bug exactly: any root that is not already in normal form (a doubled separator,
a trailing separator, a "/./" segment) makes `entry.path` disagree with
`str(Path(entry.path))`. That disagreement is the whole fault, and it is what
these tests pin — on every platform, in the one form the lookup uses.
"""
from pathlib import Path

from laser_trim_analyzer.core.ingest_run import discover_excel_files


def _make_tree(root: Path):
    made = []
    for rel in ("a.xls", "sub/b.xlsx", "sub/deep/c.XLS"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * (len(rel) + 10))
        made.append(str(p))
    (root / "notes.txt").write_text("ignored")
    return sorted(made)


def test_every_returned_path_is_its_own_lookup_key(tmp_path):
    """`stats[str(Path(p))]` is THE lookup `_classify_scan` performs."""
    _make_tree(tmp_path)
    files, stats = discover_excel_files(str(tmp_path))
    assert files
    for p in files:
        assert str(Path(p)) == p, f"walk returned a non-normal path: {p!r}"
        assert str(Path(p)) in stats, f"no stat under the processor's key: {p!r}"


def test_a_root_with_a_doubled_separator_still_keys_normally(tmp_path):
    """Tk's `askdirectory` form on Windows, in the only shape macOS can show.

    A doubled separator inside the root is exactly what a forward-slash UNC
    root does to `entry.path` on Windows: it survives into every child path,
    and `str(Path(...))` removes it.
    """
    made = _make_tree(tmp_path)
    files, stats = discover_excel_files(str(tmp_path) + "//")
    assert sorted(str(Path(p)) for p in files) == made
    for p in files:
        assert str(Path(p)) == p, f"walk returned a non-normal path: {p!r}"
        assert str(Path(p)) in stats, f"no stat under the processor's key: {p!r}"


def test_a_root_with_a_dot_segment_still_keys_normally(tmp_path):
    made = _make_tree(tmp_path)
    files, stats = discover_excel_files(str(tmp_path) + "/./")
    assert sorted(str(Path(p)) for p in files) == made
    for p in files:
        assert str(Path(p)) == p, f"walk returned a non-normal path: {p!r}"
        assert str(Path(p)) in stats, f"no stat under the processor's key: {p!r}"


def test_a_trailing_separator_on_the_root_still_keys_normally(tmp_path):
    made = _make_tree(tmp_path)
    files, stats = discover_excel_files(str(tmp_path) + "/")
    assert sorted(str(Path(p)) for p in files) == made
    for p in files:
        assert str(Path(p)) in stats


def test_the_walk_and_the_classifier_agree_on_an_odd_root(tmp_path):
    """The end of the bug, not just its shape: a known file must settle in
    MEMORY — decision "processed", no stat(), no hash — when the folder is
    reached through a root that is not in normal form.

    `_classify_scan` is driven directly with the caches the processor loads
    from the database, because those are stored as `str(Path(...))` (see
    `save_final_test` / `_save_result`) and that is the half of the mismatch
    the walk has to meet. Before the fix this returned "needs_hash" — the
    542-second verify pass, on a folder where nothing had changed.
    """
    from laser_trim_analyzer.core.processor import Processor

    made = _make_tree(tmp_path)
    files, stats = discover_excel_files(str(tmp_path) + "//")

    proc = object.__new__(Processor)          # no DB, no config: caches only
    proc._processed_filenames = set(made)     # what the database holds
    proc._processed_stat = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime)
                            for p in made}
    proc._processed_basename = {}
    proc._disk_stats = stats
    proc._scan_rebound = 0
    proc._scan_adopted = 0

    decisions = [proc._classify_scan(Path(f)) for f in files]
    assert decisions == ["processed"] * len(files), (
        f"{decisions.count('needs_hash')} of {len(files)} files would pay a "
        f"stat() over the share")
