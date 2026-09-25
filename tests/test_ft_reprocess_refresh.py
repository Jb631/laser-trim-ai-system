"""A reprocessed final-test file refreshes its stored row (2026-09-17).

Before this change, `save_final_test`'s content-hash early return handed back
the existing row's id and updated nothing else. Trim rows have always updated
in place on a re-read (`_update_existing_analysis`); final-test rows took the
early return and stayed frozen, so reprocessing the ~151,000-row work database
would have silently refreshed only half of what anyone would expect.

The fix reuses `apply_final_test_regrade` — the SAME writer the re-grade
repair pass already uses — so a refreshed row is indistinguishable from a
freshly graded one. These tests also pin the two structural guarantees named
in the task brief: the per-path duplicate marker (shipped earlier the same
day) must still be written when the incoming path differs from the one on
record, and a refresh failure must be logged and swallowed rather than
propagated, because this path runs unattended, overnight, 151,000 times.
"""
import logging
from datetime import datetime

import pytest
import sqlalchemy as sa

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import FinalTestTrack, ProcessedFile


def _meta(name="x.xls", path=None):
    return {"filename": name, "file_path": path or f"/tmp/{name}", "model": "8232-1",
            "serial": "1", "file_date": datetime(2026, 1, 1)}


def _tracks(**overrides):
    track = {"track_id": "default", "linearity_pass": False,
              "linearity_fail_points": 4, "linearity_error": 0.02}
    track.update(overrides)
    return [track]


# ---------------------------------------------------------------------------
# the headline case, as specified
# ---------------------------------------------------------------------------

def test_resaving_the_same_content_refreshes_the_verdict(tmp_path):
    db = DatabaseManager(tmp_path / "t.db")
    tracks = _tracks()
    first = db.save_final_test(metadata=_meta(), tracks=tracks,
                               test_results={"linearity_pass": False},
                               file_hash="abc123", file_size=10,
                               file_modified_date=None)
    tracks[0].update(linearity_pass=True, linearity_fail_points=0, linearity_error=0.004)
    second = db.save_final_test(metadata=_meta(), tracks=tracks,
                                test_results={"linearity_pass": True},
                                file_hash="abc123", file_size=10,
                                file_modified_date=None)
    assert second == first, "must update the same row, not create another"
    with db.session() as s:
        verdict = s.execute(sa.text(
            "SELECT linearity_pass FROM final_test_results WHERE id=:i"),
            {"i": first}).scalar()
        rows = s.execute(sa.text("SELECT COUNT(*) FROM final_test_results")).scalar()
    assert bool(verdict) is True, "the stored verdict was not refreshed"
    assert rows == 1


def test_resaving_refreshes_the_track_row_not_just_the_header(tmp_path):
    """`apply_final_test_regrade` replaces track rows wholesale. Pin that a
    reprocess leaves exactly one, freshly-graded track row behind — not the
    stale one, and not a second one appended beside it."""
    db = DatabaseManager(tmp_path / "t.db")
    tracks = _tracks()
    ft_id = db.save_final_test(metadata=_meta(), tracks=tracks,
                               test_results={"linearity_pass": False},
                               file_hash="abc123", file_size=10,
                               file_modified_date=None)
    tracks[0].update(linearity_pass=True, linearity_fail_points=0, linearity_error=0.004)
    db.save_final_test(metadata=_meta(), tracks=tracks,
                       test_results={"linearity_pass": True},
                       file_hash="abc123", file_size=10,
                       file_modified_date=None)

    with db.session() as s:
        track_rows = s.query(FinalTestTrack).filter(
            FinalTestTrack.final_test_id == ft_id).all()
        assert len(track_rows) == 1, "the stale track row must be replaced, not appended to"
        assert track_rows[0].linearity_pass is True
        assert track_rows[0].linearity_fail_points == 0
        assert track_rows[0].linearity_error == pytest.approx(0.004)


# ---------------------------------------------------------------------------
# the duplicate-path marker (shipped earlier the same day) must survive
# ---------------------------------------------------------------------------

def test_duplicate_under_a_new_path_still_gets_marked_and_also_refreshes(tmp_path):
    """Same content, resubmitted under a DIFFERENT path: the second path must
    still get a per-path skip marker (otherwise the scan re-offers it every
    run forever), and now the stored verdict must ALSO refresh — the two
    behaviours are independent and this change must not trade one for the
    other."""
    db = DatabaseManager(tmp_path / "t.db")
    tracks = _tracks()
    first_path = str(tmp_path / "first" / "x.xls")
    first = db.save_final_test(
        metadata=_meta(path=first_path), tracks=tracks,
        test_results={"linearity_pass": False},
        file_hash="dupcontent", file_size=10, file_modified_date=None)

    second_path = str(tmp_path / "second" / "copy of x.xls")
    tracks[0].update(linearity_pass=True, linearity_fail_points=0, linearity_error=0.001)
    second = db.save_final_test(
        metadata=_meta(name="copy of x.xls", path=second_path), tracks=tracks,
        test_results={"linearity_pass": True},
        file_hash="dupcontent", file_size=10, file_modified_date=None)

    assert second == first, "the content is stored once, under the first path"
    with db.session() as s:
        # Snapshot into plain tuples inside the session — ProcessedFile rows
        # expire on commit, and reading an attribute after the `with` block
        # closes raises DetachedInstanceError rather than answering stale.
        markers = {r.file_path: (r.file_hash, r.error_message)
                   for r in s.query(ProcessedFile).all()}
        verdict = s.execute(sa.text(
            "SELECT linearity_pass FROM final_test_results WHERE id=:i"),
            {"i": first}).scalar()

    assert second_path in markers, "the second path must still be recorded with a skip marker"
    marker_hash, marker_reason = markers[second_path]
    assert marker_hash.startswith("skip:")
    assert f"final_test_results id {first}" in (marker_reason or "")
    assert first_path not in markers, "the original path is a real record, not a marker"
    assert bool(verdict) is True, "the verdict must refresh even when reached via a duplicate path"


def test_resaving_under_the_same_path_writes_no_marker(tmp_path):
    """The plain re-read case (same path both times) must not grow a
    duplicate-path marker for its own path — only a genuinely different path
    earns one."""
    db = DatabaseManager(tmp_path / "t.db")
    tracks = _tracks()
    db.save_final_test(metadata=_meta(), tracks=tracks,
                       test_results={"linearity_pass": False},
                       file_hash="abc123", file_size=10,
                       file_modified_date=None)
    tracks[0].update(linearity_pass=True, linearity_fail_points=0)
    db.save_final_test(metadata=_meta(), tracks=tracks,
                       test_results={"linearity_pass": True},
                       file_hash="abc123", file_size=10,
                       file_modified_date=None)
    with db.session() as s:
        assert s.query(ProcessedFile).count() == 0


# ---------------------------------------------------------------------------
# the refresh-failure judgement call: log and swallow, never propagate
# ---------------------------------------------------------------------------

def test_refresh_failure_is_logged_and_swallowed_not_propagated(tmp_path, monkeypatch, caplog):
    """apply_final_test_regrade can fail on a malformed record. This save
    path runs unattended, overnight, ~151,000 times, so one bad record must
    not raise out of save_final_test (which would risk the whole run) — it
    must be logged and the previously stored verdict left in place."""
    db = DatabaseManager(tmp_path / "t.db")
    tracks = _tracks()
    first = db.save_final_test(metadata=_meta(), tracks=tracks,
                               test_results={"linearity_pass": False},
                               file_hash="abc123", file_size=10,
                               file_modified_date=None)

    def boom(self, *a, **k):
        raise RuntimeError("malformed record")

    # The re-grade WRITER, shared by apply_final_test_regrade and (since ingest-speed Task 7)
    # the save path's own session-taking body -- the save no longer goes through
    # apply_final_test_regrade, whose own session a write batch would refuse.
    monkeypatch.setattr(DatabaseManager, "_write_final_test_regrade", boom)

    tracks[0].update(linearity_pass=True, linearity_fail_points=0, linearity_error=0.004)
    with caplog.at_level(logging.WARNING):
        second = db.save_final_test(metadata=_meta(), tracks=tracks,
                                    test_results={"linearity_pass": True},
                                    file_hash="abc123", file_size=10,
                                    file_modified_date=None)

    assert second == first, "the file must still resolve to the existing row, not raise"
    assert any(r.levelno >= logging.WARNING for r in caplog.records), (
        "a swallowed refresh failure must leave a trace in the log")

    with db.session() as s:
        verdict = s.execute(sa.text(
            "SELECT linearity_pass FROM final_test_results WHERE id=:i"),
            {"i": first}).scalar()
    assert bool(verdict) is False, (
        "a failed refresh must leave the prior verdict standing, not a half-applied one")


# ---------------------------------------------------------------------------
# guard rail: the genuinely-new-file path is untouched by this change
# ---------------------------------------------------------------------------

def test_different_content_still_creates_a_second_row(tmp_path):
    db = DatabaseManager(tmp_path / "t.db")
    first = db.save_final_test(metadata=_meta(), tracks=_tracks(),
                               test_results={"linearity_pass": False},
                               file_hash="hash-one", file_size=10,
                               file_modified_date=None)
    second = db.save_final_test(
        metadata=_meta(name="y.xls", path="/tmp/y.xls"), tracks=_tracks(linearity_pass=True),
        test_results={"linearity_pass": True},
        file_hash="hash-two", file_size=11, file_modified_date=None)
    assert second != first
    with db.session() as s:
        assert s.execute(sa.text(
            "SELECT COUNT(*) FROM final_test_results")).scalar() == 2
