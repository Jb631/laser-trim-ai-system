"""How the Findings page and tab arrange findings: group, number, merge, order.

Pure data, no Tk -- so every rule those screens follow can be tested without a window. The
screens draw what arrange() returns and decide nothing themselves.
Spec: docs/superpowers/specs/2026-09-23-design-system-and-findings-page-design.md, section 3.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ROWS_PER_GROUP = 5


@dataclass(frozen=True)
class GroupSpec:
    key: str
    title: str
    meaning: str
    column: str          # the unit of the readout column, printed in the group header
    tone: str            # "act" -> teal count pill, "check" -> coral
    empty: str            # what the group says when it has nothing


GROUPS: Tuple[GroupSpec, ...] = (
    GroupSpec("yield", "Change a setting to raise yield",
              "A different setting did better on the same test", "tracks a year", "act",
              "Nothing here yet. A finding appears when a model ran two cut settings, or two "
              "incoming-resistance windows, on the same test and one did clearly better."),
    GroupSpec("laser_time", "Laser time you could save",
              "Units cut that didn't need it, or given more cuts than planned", "tracks", "act",
              "Nothing here yet. A finding appears when units arrive already inside their limits, "
              "or take more cuts than their recipe asks for."),
    GroupSpec("check", "Check the test",
              "Graded against more than one limit table, so pass rates across the change don't compare",
              "tracks", "check",
              "Nothing to check. A finding appears when a model is graded against more than one "
              "limit table."),
    GroupSpec("history", "What changed", "Recipe changes, newest first", "pass-rate move", "act",
              "No recipe changes found."),
)
OTHER = GroupSpec("other", "Other findings",
                  "From an analyzer this page does not know how to group yet", "", "check", "")

# Every analyzer -> its group. A finding from an analyzer NOT listed here goes under OTHER
# and is logged -- never dropped: on these screens silence is itself a result, so a finding
# must never be able to vanish. A test asserts this covers every module in findings/analyzers.
ANALYZER_GROUP: Dict[str, str] = {
    "cut_setting": "yield",
    "ink_target": "yield",
    "trim_effort": "laser_time",
    "pass_burden": "laser_time",
    "limit_tables": "check",
    "recipe_change": "history",
}

_GRADE_TAG = {"same_days": "same days", "side_by_side": "side by side",
              "two_periods": "two periods · test first"}
_GRADE_ORDER = ("two_periods", "side_by_side", "same_days")        # weakest first
_LASER_TIME_FIELD = {"Trim avoidance": "arrive_in_spec_n", "Pass effectiveness": "multi_cut_n",
                     "Multi-pass burden": "tracks_over_recipe"}
_RECIPE = re.compile(r"^(?P<laser>[^:]+): recipe changed from (?P<a>.+) to (?P<b>.+)$")
_CUTS = re.compile(r"^(?P<n>\d+) cuts?(?: \(cut length (?P<len>[^)]+)\))?$")


@dataclass
class Row:
    group: str
    model: str
    statement: str
    value: Optional[float]
    findings: List[Dict[str, Any]]
    tags: List[str] = field(default_factory=list)
    when: Optional[str] = None                      # ISO date, history rows

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.group, self.model, self.statement)

    @property
    def merged(self) -> bool:
        return len(self.findings) > 1


@dataclass
class Group:
    spec: GroupSpec
    rows: List[Row]


def group_key(finding: Dict[str, Any]) -> str:
    return ANALYZER_GROUP.get(finding.get("analyzer"), OTHER.key)


def _num(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _laser(finding) -> str:
    from laser_trim_analyzer.core.models import laser_label
    systems = finding.get("systems") or ()
    return laser_label(systems[0]) if systems else ""


def _month(iso) -> str:
    try:
        return datetime.fromisoformat(str(iso)[:10]).strftime("%b %Y")
    except (TypeError, ValueError):
        return ""


def readout(finding: Dict[str, Any]) -> Optional[float]:
    g = group_key(finding)
    ev = finding.get("evidence") or {}
    if g == "yield":
        return _num(finding.get("tracks_per_year"))
    if g == "laser_time":
        name = _LASER_TIME_FIELD.get(finding.get("category"))
        v = _num((ev.get("facts") or {}).get(name)) if name else None
        return v if v is not None else _num(finding.get("n_units"))
    if g == "history":
        pb = _num((ev.get("before") or {}).get("trim_pass_pct"))
        pa = _num((ev.get("after") or {}).get("trim_pass_pct"))
        return None if pb is None or pa is None else pa - pb
    return _num(finding.get("n_units"))


def _recipe_move(a: str, b: str) -> str:
    ma, mb = _CUTS.match(a.strip()), _CUTS.match(b.strip())
    if not (ma and mb):
        return f"{a} → {b}"
    na, nb = int(ma["n"]), int(mb["n"])
    if na != nb:
        return f"{na} cut{'s' if na != 1 else ''} → {nb} cut{'s' if nb != 1 else ''}"
    if ma["len"] and mb["len"]:
        return f"cut {ma['len']} → {mb['len']}"
    return f"{a} → {b}"


def statement(finding: Dict[str, Any]) -> str:
    title = str(finding.get("title") or "")
    ev = finding.get("evidence") or {}
    if finding.get("analyzer") == "cut_setting" and _num(ev.get("best")) is not None \
            and _num(ev.get("current")) is not None:
        best, current = float(ev["best"]), float(ev["current"])
        if ev.get("stale"):
            return f"{_laser(finding)}: {best:g} did better than {current:g}, last run {_month(ev.get('last_ran'))}"
        return f"{_laser(finding)}: cut {current:g} → try {best:g}"
    if group_key(finding) == "history":
        when = _month((ev.get("after") or {}).get("first"))
        m = _RECIPE.match(title)
        body = f"{m['laser']}: {_recipe_move(m['a'], m['b'])}" if m else title
        return f"{when} · {body}" if when else body
    return title


def _merge_key(finding) -> Optional[Tuple]:
    """Same model, analyzer and recommendation, differing only by track -> one row."""
    if finding.get("analyzer") != "cut_setting":
        return None
    ev = finding.get("evidence") or {}
    if _num(ev.get("best")) is None or _num(ev.get("current")) is None:
        return None
    return (finding.get("model"), tuple(finding.get("systems") or ()),
            float(ev["best"]), float(ev["current"]), bool(ev.get("stale")))


def _tags(members: Sequence[Dict[str, Any]]) -> List[str]:
    tags: List[str] = []
    if len(members) == 2:
        tags.append("both tracks")
    elif len(members) > 2:
        tags.append(f"{len(members)} tracks")
    grades = [(m.get("evidence") or {}).get("grade") for m in members]
    grades = [g for g in grades if g in _GRADE_TAG]
    if grades:
        tags.append(_GRADE_TAG[min(grades, key=_GRADE_ORDER.index)])   # the weakest, honestly
    return tags


def _sum(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


def _size(r: Row) -> int:
    return sum(int(f.get("n_units") or 0) for f in r.findings)


def _sort(key: str, rows: List[Row]) -> None:
    if key == "history":
        rows.sort(key=lambda r: r.model)
        rows.sort(key=lambda r: r.when or "", reverse=True)          # newest first; undated last
    elif key == "yield":
        rows.sort(key=lambda r: (r.value is None, -(r.value or 0.0), -_size(r), r.model))
    else:
        rows.sort(key=lambda r: (r.value is None, -(r.value or 0.0), r.model))


def arrange(findings: Sequence[Dict[str, Any]], *, include_empty: bool = True) -> List[Group]:
    buckets: Dict[str, List[Row]] = {s.key: [] for s in (*GROUPS, OTHER)}
    merged: Dict[Tuple, Row] = {}
    unmapped = set()
    for fnd in findings:
        g = group_key(fnd)
        if g == OTHER.key:
            unmapped.add(str(fnd.get("analyzer")))
        mk = _merge_key(fnd)
        if mk is not None and mk in merged:
            merged[mk].findings.append(fnd)
            continue
        ev = fnd.get("evidence") or {}
        r = Row(group=g, model=str(fnd.get("model") or ""), statement=statement(fnd), value=None,
                findings=[fnd], when=(ev.get("after") or {}).get("first") if g == "history" else None)
        if mk is not None:
            merged[mk] = r
        buckets[g].append(r)
    if unmapped:
        logger.error("Findings page: no group for analyzer(s) %s -- shown under 'Other findings'",
                     sorted(unmapped))
    out: List[Group] = []
    for spec in (*GROUPS, OTHER):
        rows = buckets[spec.key]
        for r in rows:
            r.value = _sum(readout(x) for x in r.findings)
            r.tags = _tags(r.findings)
        _sort(spec.key, rows)
        if rows or (include_empty and spec is not OTHER):
            out.append(Group(spec, rows))
    return out


def _plural(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def caption(groups: Sequence[Group], findings: Sequence[Dict[str, Any]]) -> str:
    n = {g.spec.key: len(g.rows) for g in groups}
    parts = [_plural(n.get("yield", 0), "change worth testing", "changes worth testing"),
             _plural(n.get("laser_time", 0), "way to save laser time", "ways to save laser time"),
             _plural(n.get("check", 0), "test to check", "tests to check")]
    stamps = [str(x.get("computed_at")) for x in findings if x.get("computed_at")]
    if stamps:
        try:
            dt = datetime.fromisoformat(max(stamps)[:19])
            parts.append(f"worked out {dt.day} {dt:%b}")          # NOT %-d: it raises on Windows
        except ValueError:
            pass
    return " · ".join(parts)


def value_text(group: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    if group == "yield":
        return f"~{value:,.0f}"
    if group == "history":
        return f"{value:+.0f}"
    return f"{value:,.0f}"


def value_tone(group: str, value: Optional[float]) -> Optional[str]:
    if group != "history" or value is None or value == 0:
        return None
    return "up" if value > 0 else "down"
