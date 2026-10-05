"""One way to write a date on every screen (finish pass, 2026-10-04: the app said "29 Sep 2026",
"Jun 26, 2026" and "2026-07-07" on three different pages).

    day(d)    "29 Sep 2026"         the default, anywhere a date stands alone
    day_short "29 Sep"              a chart axis, a tight column
    month(d)  "Sep 2026"            "last trimmed Sep 2026"
    stamp(d)  "29 Sep 2026 17:22"   a time matters (a file, a run)
    axis_labels(dates, unit)        a chart axis: "22 Dec", "29 Dec", "5 Jan 2026"

Each takes a datetime, a date, an ISO string as SQLite returns it, or None (-> "—"). Never
`%-d`: it raises on Windows; the day is formatted as a plain int.
"""
from datetime import date, datetime
from typing import Iterable, List, Optional, Union

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


def axis_labels(dates: Iterable[Day], unit: str = "day") -> List[str]:
    """Labels for a run of dates along a chart axis, oldest first, in the same words:

        unit "day"    "29 Sep"   lots, weeks, a few weeks' days
        unit "month"  "Sep"      a chart by month
        unit "year"   "2026"

    The year is said where it changes -- on January's label by month ("Jan 2026"), on the first
    day of a new year by day ("5 Jan 2026") -- so an axis that crosses a new year never reads
    backwards (8887's lots once ran "08/20" then "07/23", eleven months later). An axis that
    never crosses one says its year on its first label instead, so the year is always said once.
    Said on the first label as well, it ran into the next year's label on a narrow chart ("Nov
    2025" beside "Jan 2026"). A date that cannot be read is "—"."""
    days = [_as_date(d) for d in dates]
    crosses = len({x.year for x in days if x is not None}) > 1
    out: List[str] = []
    year = None
    for x in days:
        if x is None:
            out.append(NONE)
            continue
        if unit == "year":
            out.append(f"{x.year}")
        else:
            short = day_short(x) if unit == "day" else f"{x:%b}"
            said = (x.year != year) if year is not None else not crosses
            out.append(f"{short} {x.year}" if said else short)
        year = x.year
    return out
