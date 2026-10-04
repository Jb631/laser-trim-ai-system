"""Company yield trend query (2026-07-06) — the Dashboard company-trend section.

Company-wide pass-rate per week/month with per-system split: (PASS + WARNING) /
graded. UNTRIMMED test-sweeps and files that failed processing are out of numerator
AND denominator (no trim verdict); since 2026-10-04 suspect files and future-dated
files are out too -- the Overview's own definition (it charts each laser with this).
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _add(db, model, system, status, when, n=1):
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SystemType, StatusType)
    tag = f"{status[:2]}{system}"  # keep (filename, file_date, model, serial) unique
    with db.session() as s:
        for i in range(n):
            s.add(DBAR(filename=f"{model}-{tag}-{when:%Y%m%d}-{i}.xls",
                       file_path=f"/f/{model}/{tag}/{when:%Y%m%d}/{i}",
                       file_hash=f"{model}{tag}{when:%Y%m%d}{i}".ljust(64, "0"),
                       model=model, serial=f"s{tag}{i}", system=SystemType(system),
                       file_date=when, timestamp=when,
                       overall_status=StatusType[status],
                       has_multi_tracks=False, processing_time=0.1))
        s.commit()


def test_company_trend_monthly_with_system_split(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "t.db")
    jan, feb = datetime(2026, 1, 10), datetime(2026, 2, 10)
    _add(db, "M1", "A", "PASS", jan, n=8)
    # WARNING = linearity-accepted, sigma-watch only (customer basis): counts
    # in the yield numerator. Domain rule 2026-07-07.
    _add(db, "M1", "A", "WARNING", jan, n=2)
    _add(db, "M1", "A", "FAIL", jan, n=2)
    _add(db, "M2", "B", "PASS", jan, n=5)
    _add(db, "M3", "C", "PASS", feb, n=3)      # LTS3 ramping in Feb
    _add(db, "M3", "C", "FAIL", feb, n=1)
    _add(db, "M1", "A", "UNTRIMMED", feb, n=4)  # must not count anywhere

    out = db.get_company_yield_trend(days_back=36500, period="month")

    assert out["periods"] == ["2026-01", "2026-02"]
    jan_c, feb_c = out["company"]
    # Jan: 17 gradeable, 15 linearity-accepted (8+2 WARN A, 5 B).
    assert (jan_c["total"], jan_c["accepted"]) == (17, 15)
    assert abs(jan_c["linearity_yield"] - 15 / 17 * 100) < 1e-9
    # Feb: UNTRIMMED excluded -> only the 4 C units count.
    assert (feb_c["total"], feb_c["accepted"]) == (4, 3)

    assert set(out["by_system"]) == {"A", "B", "C"}
    # System C: no Jan data (total 0, rate None), Feb 4/3.
    c_jan, c_feb = out["by_system"]["C"]
    assert c_jan["total"] == 0 and c_jan["linearity_yield"] is None
    assert (c_feb["total"], c_feb["accepted"]) == (4, 3)
    # System A: Feb has ONLY the untrimmed rows -> zero gradeable.
    a_feb = out["by_system"]["A"][1]
    assert a_feb["total"] == 0


def test_company_trend_weekly_and_window(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "w.db")
    _add(db, "M1", "A", "PASS", datetime(2020, 6, 1), n=2)   # far outside window
    _add(db, "M1", "A", "PASS", datetime.now(), n=3)

    out = db.get_company_yield_trend(days_back=30, period="week")
    assert len(out["periods"]) == 1
    assert out["company"][0]["total"] == 3

    out_all = db.get_company_yield_trend(days_back=36500, period="week")
    assert sum(r["total"] for r in out_all["company"]) == 5
    # Honesty metadata present: vintage + partial flag computed.
    assert out_all["data_through"] is not None
    assert isinstance(out_all["partial_last"], bool)


def test_a_failed_file_a_suspect_file_and_a_future_date_never_count(tmp_path):
    """The company trend is the Overview's own yield since 2026-10-04 (the Overview charts each
    laser with it): (PASS + WARNING) / graded. A file that failed processing is not a measurement
    (CLAUDE.md) -- it used to count as a failure; a file marked suspect is left out (the rule the
    drift watch, the FOCUS list and the Overview's cards follow since 2026-10-02); and a file
    dated more than a day ahead is not believed -- it used to draw a month in the future."""
    from datetime import timedelta
    from sqlalchemy import text
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "clean.db")
    jan = datetime(2026, 1, 10)
    _add(db, "M1", "A", "PASS", jan, n=6)
    _add(db, "M1", "A", "FAIL", jan, n=2)
    _add(db, "M1", "A", "ERROR", jan, n=3)                    # failed processing
    _add(db, "M1", "A", "PROCESSING_FAILED", jan, n=1)
    _add(db, "M2", "A", "FAIL", jan, n=4)                     # marked suspect below
    _add(db, "M3", "A", "PASS", datetime.now() + timedelta(days=40), n=5)   # a mistyped date
    with db.session() as s:
        s.execute(text("UPDATE analysis_results SET data_quality = 'suspect' WHERE model = 'M2'"))
        s.commit()

    out = db.get_company_yield_trend(days_back=36500, period="month")
    assert out["periods"] == ["2026-01"]
    (jan_c,) = out["company"]
    assert (jan_c["total"], jan_c["accepted"]) == (8, 6)
    assert out["data_through"].date() == jan.date()           # the newest file it believes
