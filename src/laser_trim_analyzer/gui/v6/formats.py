"""One way to write a date on every screen (finish pass, 2026-10-04: the app said "29 Sep 2026",
"Jun 26, 2026" and "2026-07-07" on three different pages).

    day(d)    "29 Sep 2026"         the default, anywhere a date stands alone
    day_short "29 Sep"              a chart axis, a tight column
    month(d)  "Sep 2026"            "last trimmed Sep 2026"
    stamp(d)  "29 Sep 2026 17:22"   a time matters (a file, a run)

Each takes a datetime, a date, an ISO string as SQLite returns it, or None (-> "—"). Never
`%-d`: it raises on Windows; the day is formatted as a plain int.
"""
from datetime import date, datetime
from typing import Optional, Union

Day = Union[datetime, date, str, None]
NONE = "—"


def _as_date(d: Day) -> Optional[datetime]:
    if d is None or d == "":
        return None
    if isinstance(d, datetime):
        return d
    if isinstance(d, date):
        return datetime(d.year, d.month, d.day)
    try:
        return datetime.fromisoformat(str(d).replace("T", " ")[:26])
    except ValueError:
        return None


def day(d: Day) -> str:
    x = _as_date(d)
    return NONE if x is None else f"{x.day} {x:%b %Y}"


def day_short(d: Day) -> str:
    x = _as_date(d)
    return NONE if x is None else f"{x.day} {x:%b}"


def month(d: Day) -> str:
    x = _as_date(d)
    return NONE if x is None else f"{x:%b %Y}"


def stamp(d: Day) -> str:
    x = _as_date(d)
    return NONE if x is None else f"{x.day} {x:%b %Y %H:%M}"
