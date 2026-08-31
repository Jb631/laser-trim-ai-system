"""Re-trim confound in the escape/overkill metric (2026-08-30).

A unit that fails linearity is re-trimmed until it passes. Those attempts land
in separate analysis rows on the SAME DAY, and the clock time that orders them
lives only in the filename — the parser used to drop it, so every attempt tied
on `file_date` and `_find_matching_trim` linked an arbitrary one (in practice
the earliest-ingested). When that arbitrary pick was a failing early attempt,
the pair scored as an "overkill": the trim station blamed for rejecting a unit
that final test then passed.

The unit's disposition is its LAST attempt before it left the station, so:
  * fail@10:01 then pass@11:17, FT passes -> agreement, NOT overkill
  * fail@10:01 then pass@11:17, FT fails  -> ESCAPE (the old code hid this one)
  * pass@10:01 then fail@11:17, FT fails  -> agreement, NOT escape

The correction is deliberately bounded to the trim day. Shop serial numbers are
REUSED across production lots (repeated serials span ~6 years in the real DB),
so "the best/last run for this serial" over all history would credit a 2012
unit's pass to a 2026 unit. test_recycled_serial_* pins that guard.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _trim(s, model, serial, when, lin_pass, i, DBAR, DBTR, StatusType, SystemType,
          filename=None, legacy=True):
    """One trim attempt: analysis row + a single trimmed track.

    `legacy=True` reproduces the state of every row already in the production
    DB: `file_date` truncated to midnight, with the attempt's clock time
    present ONLY in the filename. That is what makes same-day attempts tie and
    is the condition the fix has to survive — tests that write the time
    straight into file_date would pass against the broken code.
    """
    stamp = f"{when.month}-{when.day}-{when.year}_{when.hour % 12 or 12}-" \
            f"{when.minute:02d} {'PM' if when.hour >= 12 else 'AM'}"
    a = DBAR(filename=filename or f"{model}_{serial}_TA_Test Data_{stamp}Trimmed Correct.xls",
             file_path=f"/t/{model}/{i}", file_hash=f"{model}{serial}{i}".ljust(64, "0"),
             model=model, serial=serial, system=SystemType.A,
             file_date=when.replace(hour=0, minute=0, second=0, microsecond=0)
                        if legacy else when,
             timestamp=when,
             overall_status=StatusType.PASS if lin_pass else StatusType.FAIL,
             has_multi_tracks=False, processing_time=0.1)
    s.add(a)
    s.flush()
    s.add(DBTR(analysis_id=a.id, track_id="default",
               status=StatusType.PASS if lin_pass else StatusType.FAIL,
               linearity_pass=lin_pass, sigma_gradient=0.01, sigma_pass=True))
    return a


def _repair(db):
    """The production repair path: recover clock times for legacy date-only
    rows, then re-point every FT link at the day's final attempt."""
    db.backfill_trim_file_times()
    db.rematch_final_tests()


def _ft(s, model, serial, when, lin_pass, i, DBFT, StatusType):
    s.add(DBFT(filename=f"{model} final {serial}_{when.month}-{when.day}-"
                        f"{when.year}_7-38 PM.xls",
               model=model, serial=serial, test_date=when, file_date=when,
               timestamp=when, linearity_pass=lin_pass,
               overall_status=StatusType.PASS if lin_pass else StatusType.FAIL))


@pytest.fixture
def models():
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, FinalTestResult as DBFT, TrackResult as DBTR,
        StatusType, SystemType)
    return DBAR, DBFT, DBTR, StatusType, SystemType


# --------------------------------------------------------------------------
# Source fix 1: the parser must keep the clock time that orders same-day runs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("8340-1_1_TA_Test Data_2-17-2026_12-32 PMTrimmed Correct.xls",
     datetime(2026, 2, 17, 12, 32)),
    ("8922_33_TEST DATA_5-12-2026_11-17 AM.xls", datetime(2026, 5, 12, 11, 17)),
    ("8340-1_1_TA_Test Data_8-22-2012_10-01 AMTrimmed Correct.xls",
     datetime(2012, 8, 22, 10, 1)),
    # midnight/noon boundaries
    ("X_1_TA_Test Data_1-2-2026_12-05 AMTrimmed Correct.xls",
     datetime(2026, 1, 2, 0, 5)),
    ("X_1_TA_Test Data_1-2-2026_12-05 PMTrimmed Correct.xls",
     datetime(2026, 1, 2, 12, 5)),
    # H-MM-SS form seen on shop-serial files
    ("8444-shop0_12-7-2021 10-14-41 AM.xlsx", datetime(2021, 12, 7, 10, 14, 41)),
    # no time in the name -> midnight, exactly as before
    ("8340-1_1_TA_Test Data_2-17-2026_Trimmed Correct.xls", datetime(2026, 2, 17)),
])
def test_parser_keeps_clock_time_from_filename(filename, expected):
    from laser_trim_analyzer.core.parser import ExcelParser

    got = ExcelParser()._extract_date_from_filename(filename)
    assert got == expected, f"{filename!r} -> {got!r}, expected {expected!r}"


# --------------------------------------------------------------------------
# Source fix 2: the matcher must link the day's LAST attempt
# --------------------------------------------------------------------------

def test_matcher_links_last_attempt_of_the_day(tmp_path, models):
    """fail@10:01, fail@10:07, pass@11:17 all on one day -> link the 11:17 pass."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    DBAR, DBFT, DBTR, StatusType, SystemType = models

    db = DatabaseManager(tmp_path / "lastwins.db")
    day = datetime(2026, 3, 2)
    with db.session() as s:
        _trim(s, "M1", "7", day.replace(hour=10, minute=1), False, 0, DBAR, DBTR, StatusType, SystemType)
        _trim(s, "M1", "7", day.replace(hour=10, minute=7), False, 1, DBAR, DBTR, StatusType, SystemType)
        winner = _trim(s, "M1", "7", day.replace(hour=11, minute=17), True, 2, DBAR, DBTR, StatusType, SystemType)
        s.commit()
        want = winner.id
        _ft(s, "M1", "7", day + timedelta(days=3), True, 0, DBFT, StatusType)
        s.commit()

    _repair(db)
    with db.session() as s:
        assert s.query(DBFT).first().linked_trim_id == want


def test_matcher_still_links_same_day_final_test(tmp_path, models):
    """Regression guard: 7,263 real linked rows have days_since_trim == 0. Adding
    a clock time to the trim must not push it past a midnight FT timestamp."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    DBAR, DBFT, DBTR, StatusType, SystemType = models

    db = DatabaseManager(tmp_path / "sameday.db")
    day = datetime(2026, 3, 2)
    with db.session() as s:
        _trim(s, "M1", "7", day.replace(hour=14, minute=30), True, 0, DBAR, DBTR, StatusType, SystemType)
        s.commit()
        _ft(s, "M1", "7", day, True, 0, DBFT, StatusType)   # FT stored at midnight
        s.commit()

    _repair(db)
    with db.session() as s:
        ft = s.query(DBFT).first()
        assert ft.linked_trim_id is not None, "same-day trim must still match"
        assert ft.days_since_trim == 0


# --------------------------------------------------------------------------
# The metric itself
# --------------------------------------------------------------------------

def _agreement(db, model):
    return db.get_model_trim_ft_agreement(model)


def test_retrim_fail_then_pass_is_not_overkill(tmp_path, models):
    """THE headline case: a unit re-trimmed until it passed, which FT then
    passed, is the process working — not the trim station over-rejecting."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    DBAR, DBFT, DBTR, StatusType, SystemType = models

    db = DatabaseManager(tmp_path / "ov.db")
    day = datetime(2026, 3, 2)
    with db.session() as s:
        _trim(s, "M1", "7", day.replace(hour=10, minute=1), False, 0, DBAR, DBTR, StatusType, SystemType)
        _trim(s, "M1", "7", day.replace(hour=11, minute=17), True, 1, DBAR, DBTR, StatusType, SystemType)
        s.commit()
        _ft(s, "M1", "7", day + timedelta(days=3), True, 0, DBFT, StatusType)
        s.commit()
    _repair(db)

    out = _agreement(db, "M1")
    assert out["linked"] == 1
    assert out["overkills"] == 0, "re-trimmed-then-passed unit must not be an overkill"
    assert out["escapes"] == 0
    assert out["agreements"] == 1


def test_retrim_fail_then_pass_with_ft_fail_is_an_escape(tmp_path, models):
    """The mirror image the old code HID: the day's final attempt passed and FT
    failed, so the unit escaped. Linking the early failing attempt scored this
    as an agreement and undercounted escapes."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    DBAR, DBFT, DBTR, StatusType, SystemType = models

    db = DatabaseManager(tmp_path / "esc.db")
    day = datetime(2026, 3, 2)
    with db.session() as s:
        _trim(s, "M1", "7", day.replace(hour=10, minute=1), False, 0, DBAR, DBTR, StatusType, SystemType)
        _trim(s, "M1", "7", day.replace(hour=11, minute=17), True, 1, DBAR, DBTR, StatusType, SystemType)
        s.commit()
        _ft(s, "M1", "7", day + timedelta(days=3), False, 0, DBFT, StatusType)
        s.commit()
    _repair(db)

    out = _agreement(db, "M1")
    assert out["escapes"] == 1, "day's final attempt passed + FT failed = escape"
    assert out["overkills"] == 0
    assert out["escape_units"] == ["7"]


def test_pass_then_fail_same_day_is_not_an_escape(tmp_path, models):
    """Reverse order: the station's LAST word was a rejection, so an FT failure
    is agreement, not an escape."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    DBAR, DBFT, DBTR, StatusType, SystemType = models

    db = DatabaseManager(tmp_path / "pf.db")
    day = datetime(2026, 3, 2)
    with db.session() as s:
        _trim(s, "M1", "7", day.replace(hour=10, minute=1), True, 0, DBAR, DBTR, StatusType, SystemType)
        _trim(s, "M1", "7", day.replace(hour=11, minute=17), False, 1, DBAR, DBTR, StatusType, SystemType)
        s.commit()
        _ft(s, "M1", "7", day + timedelta(days=3), False, 0, DBFT, StatusType)
        s.commit()
    _repair(db)

    out = _agreement(db, "M1")
    assert out["escapes"] == 0, "last attempt failed -> station caught it"
    assert out["overkills"] == 0
    assert out["agreements"] == 1


def test_genuine_overkill_still_counts(tmp_path, models):
    """The fix must not zero the metric out: a unit whose only/last trim attempt
    failed and which FT then passed IS a genuine overkill."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    DBAR, DBFT, DBTR, StatusType, SystemType = models

    db = DatabaseManager(tmp_path / "genuine.db")
    day = datetime(2026, 3, 2)
    with db.session() as s:
        _trim(s, "M1", "7", day.replace(hour=10, minute=1), True, 0, DBAR, DBTR, StatusType, SystemType)
        _trim(s, "M1", "7", day.replace(hour=11, minute=17), False, 1, DBAR, DBTR, StatusType, SystemType)
        s.commit()
        _ft(s, "M1", "7", day + timedelta(days=3), True, 0, DBFT, StatusType)
        s.commit()
    _repair(db)

    out = _agreement(db, "M1")
    assert out["overkills"] == 1, "last attempt failed + FT passed = real overkill"
    assert out["overkill_units"] == ["7"]


# --------------------------------------------------------------------------
# The guard: shop serial numbers are REUSED across lots
# --------------------------------------------------------------------------

def test_recycled_serial_does_not_rescue_an_overkill(tmp_path, models):
    """Shop numbers get reused. A passing trim on the same serial YEARS earlier
    is a different physical unit and must not cancel today's overkill.

    In the real DB this is the whole of 8340-1's 145 "artifact" overkills:
    serial '1' has runs from 2012 through 2026.
    """
    from laser_trim_analyzer.database.manager import DatabaseManager
    DBAR, DBFT, DBTR, StatusType, SystemType = models

    db = DatabaseManager(tmp_path / "recycled.db")
    with db.session() as s:
        _trim(s, "M1", "1", datetime(2012, 6, 18, 15, 10), True, 0, DBAR, DBTR, StatusType, SystemType)
        _trim(s, "M1", "1", datetime(2026, 2, 17, 12, 32), False, 1, DBAR, DBTR, StatusType, SystemType)
        s.commit()
        _ft(s, "M1", "1", datetime(2026, 3, 12), True, 0, DBFT, StatusType)
        s.commit()
    _repair(db)

    out = _agreement(db, "M1")
    assert out["overkills"] == 1, (
        "a 2012 pass on a recycled shop number must not excuse a 2026 rejection")


def test_earlier_day_pass_does_not_rescue_an_overkill(tmp_path, models):
    """Even within one lot's lifetime, a pass on an EARLIER day followed by a
    later failing trim means the unit came back and was rejected again."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    DBAR, DBFT, DBTR, StatusType, SystemType = models

    db = DatabaseManager(tmp_path / "earlier.db")
    with db.session() as s:
        _trim(s, "M1", "7", datetime(2026, 2, 1, 9, 0), True, 0, DBAR, DBTR, StatusType, SystemType)
        _trim(s, "M1", "7", datetime(2026, 3, 1, 9, 0), False, 1, DBAR, DBTR, StatusType, SystemType)
        s.commit()
        _ft(s, "M1", "7", datetime(2026, 3, 5), True, 0, DBFT, StatusType)
        s.commit()
    _repair(db)

    assert _agreement(db, "M1")["overkills"] == 1


# --------------------------------------------------------------------------
# One definition, one place
# --------------------------------------------------------------------------

def test_company_and_per_model_definitions_agree(tmp_path, models):
    """get_escape_overkill_analysis (the Gap) and get_model_trim_ft_agreement
    (the trim-vs-FT tab) must not be able to disagree."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    DBAR, DBFT, DBTR, StatusType, SystemType = models

    db = DatabaseManager(tmp_path / "agree.db")
    day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=10)
    with db.session() as s:
        # re-trim -> not overkill
        _trim(s, "M1", "7", day.replace(hour=10, minute=1), False, 0, DBAR, DBTR, StatusType, SystemType)
        _trim(s, "M1", "7", day.replace(hour=11, minute=17), True, 1, DBAR, DBTR, StatusType, SystemType)
        # genuine overkill
        _trim(s, "M1", "8", day.replace(hour=9, minute=0), False, 2, DBAR, DBTR, StatusType, SystemType)
        s.commit()
        _ft(s, "M1", "7", day + timedelta(days=2), True, 0, DBFT, StatusType)
        _ft(s, "M1", "8", day + timedelta(days=2), True, 1, DBFT, StatusType)
        s.commit()
    _repair(db)

    per_model = db.get_model_trim_ft_agreement("M1")
    company = db.get_escape_overkill_analysis(days_back=90)
    assert per_model["overkills"] == company["overkills"] == 1
    assert per_model["escapes"] == company["escapes"] == 0
    assert per_model["linked"] == company["total_linked"] == 2
