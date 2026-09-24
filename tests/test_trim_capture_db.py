from laser_trim_analyzer.database.manager import DatabaseManager


def test_new_tables_are_created_on_a_fresh_database(tmp_path):
    db = DatabaseManager(tmp_path / "t.db")
    with db.session() as s:
        names = {r[0] for r in s.execute(
            __import__("sqlalchemy").text(
                "SELECT name FROM sqlite_master WHERE type='table'"))}
    assert "trim_passes" in names
    assert "trim_setup" in names


def test_trim_pass_columns_are_what_the_engine_needs(tmp_path):
    db = DatabaseManager(tmp_path / "t.db")
    import sqlalchemy as sa
    with db.session() as s:
        cols = {r[1] for r in s.execute(sa.text("PRAGMA table_info(trim_passes)"))}
    for c in ("track_result_id", "pass_index", "sheet", "positions", "errors",
              "upper_limits", "lower_limits", "laser_cut_length", "laser_speed_high",
              "trim_voltage", "cut_lengths", "trim_currents", "used_deltas"):
        assert c in cols, f"missing {c}"


# append to tests/test_trim_capture_db.py
from pathlib import Path
import sqlalchemy as sa
import pytest


def test_pipeline_writes_passes_and_setup(tmp_path, monkeypatch):
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.core.processor import Processor
    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    import laser_trim_analyzer.database as dbpkg
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)

    proc = Processor(use_ml=False)
    result = proc.process_file(Path("tests/fixtures/trim/dlts_8232-1_243.xls"))
    db.save_analysis(result)

    with db.session() as s:
        passes = s.execute(sa.text(
            "SELECT pass_index, laser_cut_length FROM trim_passes ORDER BY pass_index")).all()
        setup = s.execute(sa.text(
            "SELECT initial_resistance_low, final_resistance_high FROM trim_setup")).first()
    assert [p[0] for p in passes] == [1, 2, 3]
    assert passes[0][1] == 0.75
    assert setup == (4200.0, 5500.0)


def test_saving_twice_does_not_duplicate_passes(tmp_path, monkeypatch):
    """A reprocess must refresh, not accumulate."""
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.core.processor import Processor
    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    proc = Processor(use_ml=False)
    p = Path("tests/fixtures/trim/dlts_8232-1_243.xls")
    db.save_analysis(proc.process_file(p))
    db.save_analysis(proc.process_file(p))
    with db.session() as s:
        n = s.execute(sa.text("SELECT COUNT(*) FROM trim_passes")).scalar()
    assert n == 3, f"expected 3 pass rows after two saves, got {n}"


def test_trim_pass_unique_index_rejects_duplicate_insert(tmp_path, monkeypatch):
    """Findings review (trim-capture Task 7, concern #1 -- see
    docs/decisions/2026-09-ledger-decisions.md): the schema test only ever proved
    the (track_result_id, pass_index) index exists, never that it actually
    rejects a duplicate row. This is the first task to write pass rows, so
    prove the DB-level behaviour, not just the DDL.
    """
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.database.models import TrimPass
    from laser_trim_analyzer.core.processor import Processor

    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    proc = Processor(use_ml=False)
    result = proc.process_file(Path("tests/fixtures/trim/dlts_8232-1_243.xls"))
    db.save_analysis(result)

    with db.session() as s:
        track_result_id = s.execute(sa.text(
            "SELECT track_result_id FROM trim_passes WHERE pass_index = 1 LIMIT 1"
        )).scalar()
    assert track_result_id is not None

    with pytest.raises(sa.exc.IntegrityError):
        with db.session() as s:
            s.add(TrimPass(track_result_id=track_result_id, pass_index=1))


def test_write_trim_passes_tolerates_duplicate_pass_index(tmp_path, monkeypatch):
    """Findings review (trim-capture Task 7, concern #2 -- see
    docs/decisions/2026-09-ledger-decisions.md): pass_sheets has no dedup guard,
    so two differently-named sheets that normalise to the same leading
    number would produce two passes with the same pass_index and collide on
    the unique index at write time. Never observed in a fixture, but a
    production file that fails to save entirely (verdict and all) would be
    worse than one that loses a single duplicate pass row -- so the write
    path must tolerate the collision, not let it raise.
    """
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.core.processor import Processor

    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    proc = Processor(use_ml=False)
    result = proc.process_file(Path("tests/fixtures/trim/dlts_8232-1_243.xls"))

    track = result.tracks[0]
    assert len(track.trim_passes) >= 2
    # Force the collision the parser is not known to produce but cannot rule
    # out: relabel pass 2 as pass 1's index.
    track.trim_passes[1]["pass_index"] = track.trim_passes[0]["pass_index"]

    db_id = db.save_analysis(result)  # must not raise
    assert db_id > 0

    with db.session() as s:
        n = s.execute(sa.text("SELECT COUNT(*) FROM trim_passes")).scalar()
        indices = [r[0] for r in s.execute(
            sa.text("SELECT pass_index FROM trim_passes ORDER BY pass_index"))]
    assert n == 2, f"expected the colliding pass to be dropped, got {n} rows"
    assert indices == sorted(set(indices)), "no duplicate pass_index made it to storage"


def _two_track_result(proc, src="tests/fixtures/trim/dlts_8232-1_243.xls"):
    """A real analysed result, widened to two tracks with distinguishable passes.

    All four trim fixtures are single-track, so nothing in the suite exercised
    the zip in `save_analysis`. Rather than wait for a real TRK1+TRK2 workbook
    to fixture, clone the real track and label the copy, which is enough to
    prove the PAIRING -- the thing that would fail silently.
    """
    import copy
    result = proc.process_file(Path(src))
    assert len(result.tracks) == 1, "fixture is expected to be single-track"
    first = result.tracks[0]
    second = copy.deepcopy(first)
    second.track_id = "TRK2"
    # Stamp every pass so a mispairing is visible in the stored rows rather
    # than hidden behind identical payloads.
    for p in second.trim_passes:
        p["sheet"] = f"SEC1 TRK2 {p.get('pass_index')}"
    for p in first.trim_passes:
        p["sheet"] = f"SEC1 TRK1 {p.get('pass_index')}"
    result.tracks.append(second)
    result.metadata.has_multi_tracks = True
    return result


def test_passes_land_on_the_right_track_in_a_multi_track_save(tmp_path, monkeypatch):
    """The riskiest line Tasks 1-8 added, and the one nothing covered.

    `save_analysis` writes pass rows INSIDE the analysis transaction, pairing
    parser tracks with freshly-flushed DB rows by `zip(analysis.tracks,
    db_analysis.tracks)`. Two things can go wrong there and neither raises:
    a mispairing files TRK2's cuts under TRK1, and a length mismatch makes
    zip stop early, dropping the tail track's passes entirely. Every trim
    fixture is single-track, so `zip` of one against one could not express
    either bug and the whole path read green.
    """
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.core.processor import Processor

    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    import laser_trim_analyzer.database as dbpkg
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)

    proc = Processor(use_ml=False)
    result = _two_track_result(proc)
    n_passes = sum(len(t.trim_passes) for t in result.tracks)
    db.save_analysis(result)

    with db.session() as s:
        rows = s.execute(sa.text(
            "SELECT t.track_id, p.pass_index, p.sheet FROM trim_passes p "
            "JOIN track_results t ON t.id = p.track_result_id "
            "ORDER BY t.track_id, p.pass_index")).all()
        track_ids = [r[0] for r in s.execute(sa.text(
            "SELECT track_id FROM track_results ORDER BY track_id"))]

    # 1. Both tracks were stored -- zip's inputs really were the same length.
    assert track_ids == ["TRK1", "TRK2"], track_ids
    # 2. Nothing was truncated: every pass of every track has a row.
    assert len(rows) == n_passes, (
        f"{len(rows)} pass rows stored, {n_passes} passes in the result -- "
        f"zip(analysis.tracks, db_analysis.tracks) dropped data")
    # 3. And each row is filed under the track it actually came from. This is
    #    the assertion a mispairing fails; the counts above would still pass.
    misfiled = [r for r in rows if r[0] not in r[2]]
    assert not misfiled, f"passes filed under the wrong track: {misfiled}"
    # 4. Both tracks carry a full set, not one track holding everything.
    per_track = {}
    for tid, idx, _sheet in rows:
        per_track.setdefault(tid, []).append(idx)
    assert per_track["TRK1"] == per_track["TRK2"] == [1, 2, 3], per_track


def test_every_parser_track_gets_a_db_track_row(tmp_path, monkeypatch):
    """The precondition `zip` in save_analysis silently depends on.

    zip stops at the shorter input, so if `_map_analysis_to_db` ever dropped
    or reordered a track, the tail track's passes would vanish with no error
    and no log line. Mapping is 1:1 today -- this pins that, so a future
    change to the mapping breaks here rather than in the data.
    """
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.core.processor import Processor

    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    proc = Processor(use_ml=False)
    result = _two_track_result(proc)

    db_analysis = db._map_analysis_to_db(result)
    assert len(db_analysis.tracks) == len(result.tracks), (
        f"{len(result.tracks)} parser tracks mapped to "
        f"{len(db_analysis.tracks)} DB tracks; zip would truncate")
    assert [t.track_id for t in db_analysis.tracks] == \
           [t.track_id for t in result.tracks], "mapping reordered the tracks"


def test_write_trim_passes_tolerates_a_missing_pass_index(tmp_path, monkeypatch):
    """The sibling hazard to the duplicate-index case above, and the one the dedup
    guard does NOT catch (found 2026-09-20 while reading the build ledgers).

    `None` is a perfectly good set member, so a pass with no index survives the
    duplicate check and then meets `pass_index nullable=False` at flush -- raising
    the very IntegrityError the tolerate-fix exists to avoid, and rolling back the
    whole analysis save. During an unattended rebuild that costs every track's
    verdict for that file, to save one pass row.
    """
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.core.processor import Processor

    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    proc = Processor(use_ml=False)
    result = proc.process_file(Path("tests/fixtures/trim/dlts_8232-1_243.xls"))

    track = result.tracks[0]
    assert len(track.trim_passes) >= 2
    keep = len(track.trim_passes) - 1
    track.trim_passes[0]["pass_index"] = None

    db_id = db.save_analysis(result)          # must not raise
    assert db_id > 0

    with db.session() as s:
        n = s.execute(sa.text("SELECT COUNT(*) FROM trim_passes")).scalar()
        nulls = s.execute(sa.text(
            "SELECT COUNT(*) FROM trim_passes WHERE pass_index IS NULL")).scalar()
        verdicts = s.execute(sa.text(
            "SELECT COUNT(*) FROM track_results WHERE analysis_id = :a"), {"a": db_id}).scalar()
    assert nulls == 0
    assert n == keep, f"expected the indexless pass dropped and {keep} kept, got {n}"
    assert verdicts >= 1, "the track's verdict must survive a dropped pass row"
