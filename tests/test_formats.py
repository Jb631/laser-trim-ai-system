"""One date format on every screen (finish pass, 2026-10-04)."""
from datetime import date, datetime

from laser_trim_analyzer.gui.v6 import formats as F


def test_one_way_to_write_a_date_whatever_it_arrives_as():
    for d in (datetime(2026, 9, 29, 17, 22), date(2026, 9, 29), "2026-09-29 17:22:00.000000",
              "2026-09-29T17:22:00"):
        assert F.day(d) == "29 Sep 2026"
        assert F.day_short(d) == "29 Sep"
        assert F.month(d) == "Sep 2026"
    assert F.stamp(datetime(2026, 9, 29, 17, 22)) == "29 Sep 2026 17:22"
    assert F.day(datetime(2026, 7, 7)) == "7 Jul 2026"                    # no leading zero


def test_nothing_or_nonsense_reads_as_a_dash_never_a_crash():
    for bad in (None, "", "not a date"):
        assert F.day(bad) == F.month(bad) == F.day_short(bad) == F.stamp(bad) == "—"
