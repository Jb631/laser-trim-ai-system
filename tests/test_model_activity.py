"""Is a model still being trimmed? `core/activity.py` -- James, 2026-09-25: "models that havnt been
trimmed in 2 years should show inacative or something but i dont want to hide them".

A model is INACTIVE when its newest trim file is more than 730 days before the newest trim file in
the whole database. A trim file is a laser file with a track that did not fail processing, and
neither date believes a file dated more than a day in the future. Every rule below was made to
fail first. Example data is invented.
"""
from datetime import datetime, timedelta

import pytest

NEWEST = datetime(2026, 9, 22, 17, 15)


def _db(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / "activity.db")


_serial = iter(range(1, 10 ** 6))


def _file(db, model, when, *, statuses=("PASS",), system="B", overall=None):
    """One laser file for `model`, dated `when`, with one track per status (none: no track). The
    file's own status is its first track's unless `overall` says otherwise."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult, StatusType, SystemType, TrackResult)
    n = next(_serial)
    overall = overall or (statuses[0] if statuses else "ERROR")
    with db.session() as s:
        a = AnalysisResult(model=model, serial=f"{model}-{n}", system=SystemType[system],
                           filename=f"{model}_{n}.xls", file_date=when,
                           overall_status=StatusType[overall])
        s.add(a)
        s.flush()
        for k, status in enumerate(statuses):
            s.add(TrackResult(analysis_id=a.id, track_id=f"TRK{k + 1}", status=StatusType[status]))


def _activity(db):
    from laser_trim_analyzer.core.activity import load_activity
    return load_activity(db)


# ---- the rule: more than 730 days behind the fleet's newest trim file --------------------------

def test_731_days_behind_the_newest_file_is_inactive_with_its_month(tmp_path):
    from laser_trim_analyzer.core.activity import inactive_caption, inactive_tag
    db = _db(tmp_path)
    _file(db, "LIVE", NEWEST)
    _file(db, "OLD", NEWEST - timedelta(days=731))
    act = _activity(db)
    assert act.fleet == NEWEST
    assert act.is_inactive("OLD") and not act.is_inactive("LIVE")
    assert act.inactive() == {"OLD": NEWEST - timedelta(days=731)}
    last = act.last_trimmed("OLD")
    assert inactive_tag(last) == "Inactive · last trimmed Sep 2024"
    assert inactive_caption(last) == "Inactive — last trimmed Sep 2024"


@pytest.mark.parametrize("behind,inactive", [
    (timedelta(days=729), False),
    (timedelta(days=730), False),                     # MORE than 730 days, not 730
    (timedelta(days=730, hours=1), True),             # the whole gap, never floored days
])
def test_the_line_is_more_than_730_days(tmp_path, behind, inactive):
    db = _db(tmp_path)
    _file(db, "LIVE", NEWEST)
    _file(db, "M", NEWEST - behind)
    assert _activity(db).is_inactive("M") is inactive


def test_the_fleet_moving_forward_makes_a_model_inactive_without_touching_its_own_data(tmp_path):
    """Worked out when a screen loads: another model's newer file is enough."""
    db = _db(tmp_path)
    _file(db, "OLD", NEWEST - timedelta(days=800))
    _file(db, "LIVE", NEWEST - timedelta(days=100))            # 700 days apart: active
    assert not _activity(db).is_inactive("OLD")
    _file(db, "LIVE", NEWEST)                                  # 800 days apart now
    assert _activity(db).is_inactive("OLD")


def test_a_model_with_no_trim_file_is_never_called_inactive(tmp_path):
    db = _db(tmp_path)
    _file(db, "LIVE", NEWEST)
    act = _activity(db)
    assert act.last_trimmed("NEVER-SEEN") is None and not act.is_inactive("NEVER-SEEN")


def test_an_empty_database_has_no_fleet_and_nothing_inactive(tmp_path):
    act = _activity(_db(tmp_path))
    assert act.fleet is None and act.inactive() == {}


# ---- what a trim file is ------------------------------------------------------------------------

def test_a_file_dated_in_the_future_moves_neither_date(tmp_path):
    """A mistyped filename date. More than a day ahead of now is not believed -- for the model's
    own newest file and for the fleet's. Within the day it is (a clock a few hours apart)."""
    db = _db(tmp_path)
    _file(db, "LIVE", NEWEST)
    _file(db, "OLD", NEWEST - timedelta(days=900))
    _file(db, "OLD", datetime.now() + timedelta(days=30))           # its own future file
    _file(db, "TYPO", datetime.now() + timedelta(days=400))         # another model's
    act = _activity(db)
    assert act.fleet == NEWEST
    assert act.last_trimmed("OLD") == NEWEST - timedelta(days=900) and act.is_inactive("OLD")
    assert act.last_trimmed("TYPO") is None
    soon = datetime.now() + timedelta(hours=12)
    _file(db, "SOON", soon)
    assert _activity(db).fleet == soon


def test_a_file_whose_tracks_all_failed_processing_moves_neither_date(tmp_path):
    """A failed record's file_date is when the analyser gave up, not a measurement."""
    db = _db(tmp_path)
    _file(db, "LIVE", NEWEST - timedelta(days=10))
    _file(db, "OLD", NEWEST - timedelta(days=800))
    _file(db, "OLD", NEWEST, statuses=("ERROR", "PROCESSING_FAILED"))
    _file(db, "LIVE", NEWEST, statuses=())                          # a record with no track at all
    act = _activity(db)
    assert act.fleet == NEWEST - timedelta(days=10)
    assert act.last_trimmed("OLD") == NEWEST - timedelta(days=800)


def test_a_file_with_one_good_track_counts(tmp_path):
    db = _db(tmp_path)
    _file(db, "M", NEWEST, statuses=("ERROR", "WARNING"))
    assert _activity(db).last_trimmed("M") == NEWEST


def test_an_untrimmed_sweep_counts_as_being_run(tmp_path):
    """The brief's definition: a laser file whose track did not fail processing -- a sweep the
    laser measured and did not cut is still the model on the laser (machine_compare's "graded or
    not")."""
    db = _db(tmp_path)
    _file(db, "M", NEWEST, statuses=("UNTRIMMED",))
    assert _activity(db).last_trimmed("M") == NEWEST


def test_the_index_only_query_is_the_plain_definition(tmp_path):
    """load_activity asks "has a track, and not every track failed" -- the same set as "has a track
    that did not fail", answered from the indexes. Checked here against the plain form, on files
    of every shape."""
    import sqlalchemy as sa
    db = _db(tmp_path)
    shapes = [("PASS",), ("ERROR",), ("ERROR", "PASS"), ("PROCESSING_FAILED", "ERROR"), (),
              ("UNTRIMMED",), ("FAIL", "ERROR"), ("WARNING",)]
    for m in range(6):
        for k, shape in enumerate(shapes):
            _file(db, f"M{m}", NEWEST - timedelta(days=37 * m + 11 * k), statuses=shape,
                  system="ABC"[(m + k) % 3])
    with db.session() as s:
        plain = dict(s.execute(sa.text(
            "SELECT a.model, MAX(a.file_date) FROM analysis_results a "
            "WHERE a.system IN ('A','B','C') AND EXISTS (SELECT 1 FROM track_results t "
            "WHERE t.analysis_id = a.id AND t.status NOT IN ('ERROR', 'PROCESSING_FAILED')) "
            "GROUP BY a.model")).fetchall())
    act = _activity(db)
    assert {m: d.isoformat(" ") for m, d in act.newest.items()} == \
        {m: str(d)[:19] for m, d in plain.items()}


def test_newest_trim_file_applies_the_same_guard_to_dates_in_memory():
    from laser_trim_analyzer.core.activity import newest_trim_file
    now = datetime(2026, 9, 25, 9, 0)
    dates = [datetime(2026, 9, 1), None, datetime(2026, 9, 26, 8, 0), datetime(2027, 1, 1)]
    assert newest_trim_file(dates, now=now) == datetime(2026, 9, 26, 8, 0)     # within the day
    assert newest_trim_file([datetime(2027, 1, 1)], now=now) is None
    assert newest_trim_file([], now=now) is None


# ---- one definition: the findings engine's "now" is this module's fleet date ----------------

def test_the_findings_engines_now_is_this_modules_fleet_date(tmp_path):
    """Including both shapes where a FILE's own status and its tracks' disagree (the work database
    has one of the first: a two-track file stored ERROR with one good track). The engine used to
    read the file's status, and would pick the other date here."""
    from laser_trim_analyzer.findings.engine import _fleet_latest
    db = _db(tmp_path)
    _file(db, "LIVE", NEWEST)
    _file(db, "M", NEWEST + timedelta(days=1), statuses=("ERROR", "WARNING"))   # a good track: counts
    _file(db, "M", NEWEST + timedelta(days=2), statuses=("ERROR",), overall="PASS")  # none: does not
    _file(db, "M", datetime.now() + timedelta(days=90))                         # the future: does not
    assert _fleet_latest(db) == _activity(db).fleet == NEWEST + timedelta(days=1)
