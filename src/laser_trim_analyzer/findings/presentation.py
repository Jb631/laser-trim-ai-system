"""How the Findings page and tab arrange findings: group, number, merge, order.

Pure data, no Tk -- so every rule those screens follow can be tested without a window. The
screens draw what arrange() returns and decide nothing themselves.
Spec: docs/superpowers/specs/2026-09-23-design-system-and-findings-page-design.md, section 3.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
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
    "machine_compare": "yield",
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
    base: str = ""          # the statement as first worded, before arrange() named a track in it
    ident: Tuple = ()       # what tells this row's findings apart from another's (_identity)

    @property
    def key(self) -> Tuple:
        """Unique within one arrange(), and the same across renders of the same findings.

        Not the displayed text alone (final review, 2026-09-24): limit_tables writes one finding
        per track with a title naming neither, so two tracks gave two identical rows under ONE
        key -- clicking the first opened the second's detail, and the first's evidence could never
        be opened at all. The findings' own identity (laser, track, table...) is part of the key,
        and so is the statement as first worded, not after a track was named in it -- so a row's
        key does not change when a same-reading sibling comes or goes.
        """
        return (self.group, self.model, self.base or self.statement, self.ident)

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


def _track(finding) -> Optional[str]:
    """The track a finding is about, from its evidence -- None when it does not say (an analyzer
    that is not per track, or a cache written before the track was stored)."""
    t = (finding.get("evidence") or {}).get("track")
    return None if t in (None, "") else str(t)


def _identity(finding) -> Tuple:
    """What tells one finding from another of the same model and analyzer that reads the same:
    its laser and track, and where an analyzer splits a track further, the limit table
    (cut_setting) or the recipe's first cut (pass_burden); a recipe change's first date. None
    wherever a finding -- or a cache written before these were stored -- does not carry one."""
    ev = finding.get("evidence") or {}
    facts = ev.get("facts") if isinstance(ev.get("facts"), dict) else {}
    return (tuple(finding.get("systems") or ()), _track(finding), ev.get("table"),
            facts.get("cut_setting"), (ev.get("after") or {}).get("first"))


def _ran_on_laser_since(ev) -> bool:
    """cut_setting: the model kept running on this laser after this (track, limit table) went
    quiet -- the analyzer's own decision, stored with the finding."""
    return bool(ev.get("ran_on_laser_since"))


def _different_test(finding) -> bool:
    """A recipe change whose two sides were graded against different limit tables, or a mix of
    them (recipe_change stores both). CLAUDE.md: never compare pass rates across a table change."""
    ev = finding.get("evidence") or {}
    return bool(ev.get("limit_table_changed") or ev.get("limit_tables_mixed"))


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
        if ev.get("stale") and _ran_on_laser_since(ev) and _track(finding):
            # Not "last run <month>": the MODEL is still running on this laser, only this track
            # on this limit table stopped (final review, 2026-09-24).
            return (f"{_laser(finding)}: {best:g} did better than {current:g}, {_track(finding)} "
                    f"last ran on that limit table {_month(ev.get('last_ran'))}")
        if ev.get("stale"):
            return f"{_laser(finding)}: {best:g} did better than {current:g}, last run {_month(ev.get('last_ran'))}"
        return f"{_laser(finding)}: cut {current:g} → try {best:g}"
    if group_key(finding) == "history":
        when = _month((ev.get("after") or {}).get("first"))
        m = _RECIPE.match(title)
        body = f"{m['laser']}: {_recipe_move(m['a'], m['b'])}" if m else title
        return f"{when} · {body}" if when else body
    return title


def _merged_statement(members: Sequence[Dict[str, Any]]) -> str:
    """The statement for a MERGED row (arrange() found more than one finding under one
    _merge_key) -- the same wording as statement(), except it never names a single track: a
    merged row is about every track in it, and its "both tracks"/"N tracks" tag (_tags) already
    says how many. Computed from every member, not from member order -- a lone `text =
    statement(fnd)` computed before merging (final review, 2026-09-24) named whichever finding's
    track happened to be first in `findings`, so it could read "Track A ... last ran on that
    limit table" on a row tagged "both tracks". Uses the LATEST `last_ran` among the members.

    Only cut_setting findings ever merge (see _merge_key), so this only has to handle that shape,
    and every member is guaranteed the same laser/best/current/stale/ran_on_laser_since -- those
    are the merge key itself.
    """
    fnd = members[0]
    ev = fnd.get("evidence") or {}
    best, current = float(ev["best"]), float(ev["current"])
    if not ev.get("stale"):
        return f"{_laser(fnd)}: cut {current:g} → try {best:g}"
    last_ran_values = [(m.get("evidence") or {}).get("last_ran") for m in members]
    last_ran = max((v for v in last_ran_values if v), default=None)
    if _ran_on_laser_since(ev):
        return (f"{_laser(fnd)}: {best:g} did better than {current:g}, "
                f"last ran on that limit table {_month(last_ran)}")
    return f"{_laser(fnd)}: {best:g} did better than {current:g}, last run {_month(last_ran)}"


def _merge_key(finding) -> Optional[Tuple]:
    """Same model, analyzer and recommendation, differing only by track -> one row.

    "Only by track" includes the TEST: cut_setting groups by (laser, track, limit table), so the
    table is part of the key -- two tracks graded against different tables are two rows, and
    one track's two tables are never shown as "both tracks". A cache written before the table
    was stored has None there for every finding, which keeps the old behaviour for it.
    """
    if finding.get("analyzer") != "cut_setting":
        return None
    ev = finding.get("evidence") or {}
    if _num(ev.get("best")) is None or _num(ev.get("current")) is None:
        return None
    return (finding.get("model"), tuple(finding.get("systems") or ()),
            float(ev["best"]), float(ev["current"]), bool(ev.get("stale")),
            _ran_on_laser_since(ev), ev.get("table"))


def _shares_a_track(row: "Row", finding) -> bool:
    """Would merging `finding` into `row` put one track in it twice? Never merge that: a merged
    row claims "both tracks". (With the table in _merge_key this can only happen on a cache that
    stored the track but not yet the table.) An unknown track cannot be said to repeat."""
    t = _track(finding)
    return t is not None and any(_track(f) == t for f in row.findings)


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
    if group_key(members[0]) == "history" and any(_different_test(m) for m in members):
        tags.append("different test")        # its move is not coloured either -- see value_tone
    return tags


def _scope(r: "Row") -> str:
    """What a row is about below its laser: its track(s) -- and for a multi-pass burden finding
    the recipe's first cut as well, since pass_burden splits one track by cut (its facts label
    reads "Laser 1 (LTS) · Track A · cut 4000")."""
    tracks = sorted({t for t in (_track(f) for f in r.findings) if t})
    parts = [" · ".join(tracks)] if tracks else []
    first = r.findings[0]
    if first.get("analyzer") == "pass_burden":
        cut = _num(((first.get("evidence") or {}).get("facts") or {}).get("cut_setting"))
        if cut is not None:
            parts.append(f"cut {cut:g}")
    return " · ".join(parts)


def _name_what_differs(rows: List["Row"]) -> None:
    """Rows of one group and model that would read IDENTICALLY -- limit_tables writes one
    finding per track and its title names neither -- say what tells them apart, right after the
    laser: "Laser 1 (LTS) · Track A: the limit table changed". A row that reads uniquely is left
    exactly as worded."""
    same: Dict[Tuple[str, str], List[Row]] = {}
    for r in rows:
        same.setdefault((r.model, r.statement), []).append(r)
    for twins in same.values():
        if len(twins) < 2:
            continue
        for r in twins:
            scope = _scope(r)
            if not scope:
                continue
            laser = _laser(r.findings[0])
            if laser and r.statement.startswith(laser + ":"):
                r.statement = f"{laser} · {scope}:{r.statement[len(laser) + 1:]}"
            else:
                r.statement = f"{scope} · {r.statement}"


def _unique_keys(rows: List["Row"]) -> None:
    """Last resort. Two rows that NOTHING in their evidence tells apart (a cache written before the
    track was stored, an analyzer this page does not know) still get a key each -- numbered in the
    order they arrived -- so clicking one opens that one, never its twin."""
    seen: Dict[Tuple, int] = {}
    for r in rows:
        n = seen.get(r.key, 0)
        seen[r.key] = n + 1
        if n:
            r.ident = r.ident + (("twin", n),)


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
        if mk is not None and mk in merged and not _shares_a_track(merged[mk], fnd):
            merged[mk].findings.append(fnd)
            continue
        ev = fnd.get("evidence") or {}
        text = statement(fnd)
        r = Row(group=g, model=str(fnd.get("model") or ""), statement=text, value=None,
                findings=[fnd], when=(ev.get("after") or {}).get("first") if g == "history" else None,
                base=text)
        if mk is not None and mk not in merged:
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
            # Sorted, so a refresh that returns the merged findings in another order keeps the key.
            r.ident = tuple(sorted((_identity(x) for x in r.findings), key=repr))
            if len(r.findings) > 1:
                # Recompute explicitly from every member -- see _merged_statement. r.statement/base
                # were set from whichever finding created the row, which must not stand once a
                # second one has merged into it.
                r.statement = r.base = _merged_statement(r.findings)
        _name_what_differs(rows)
        _unique_keys(rows)
        _sort(spec.key, rows)
        if rows or (include_empty and spec is not OTHER):
            out.append(Group(spec, rows))
    return out


def _plural(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def caption(groups: Sequence[Group], findings: Sequence[Dict[str, Any]], *,
            tz: Optional[tzinfo] = None) -> str:
    """`tz` is for tests; the page passes nothing and gets the machine's own time zone."""
    n = {g.spec.key: len(g.rows) for g in groups}
    parts = [_plural(n.get("yield", 0), "change worth testing", "changes worth testing"),
             _plural(n.get("laser_time", 0), "way to save laser time", "ways to save laser time"),
             _plural(n.get("check", 0), "test to check", "tests to check")]
    stamps = [str(x.get("computed_at")) for x in findings if x.get("computed_at")]
    if stamps:
        try:
            # computed_at is written by utc_now() (database/models.py) and read back without its
            # zone, so it is UTC: an evening refresh in the US is already TOMORROW in UTC. Say the
            # date in the reader's own time (final review, 2026-09-24).
            dt = (datetime.fromisoformat(max(stamps)[:19])
                  .replace(tzinfo=timezone.utc).astimezone(tz))
            parts.append(f"worked out {dt.day} {dt:%b}")          # NOT %-d: it raises on Windows
        except ValueError:
            pass
    return " · ".join(parts)


def errors_notice(errors: Dict[str, Any]) -> str:
    """The banner for models whose last refresh had an analyzer fail ({model: {analyzer: msg}},
    DatabaseManager.get_process_errors) -- one wording for every screen that lists findings (the
    Findings page, and Home's "Worth changing" since the final review of 2026-09-24): a finding
    those analyzers would have made is MISSING, so the list must say it may be short."""
    names = sorted(errors)
    shown = ", ".join(names[:10]) + (" …" if len(names) > 10 else "")
    return (f"{len(names)} model(s) could not be fully worked out on the last refresh, so they may "
            f"be missing from this list: {shown}. Open one to see what failed.")


def errors_unknown_notice(reason: str) -> str:
    """The banner for when WHICH models failed could not itself be read -- the list is still
    true, but it may be missing models, and silence would claim it is not."""
    return (f"Whether any model failed on the last refresh could not be checked ({reason}), so "
            f"this list may be missing models.")


def value_text(group: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    if group == "yield":
        return f"~{value:,.0f}"
    if group == "history":
        return f"{value:+.0f}"
    return f"{value:,.0f}"


def value_tone(group: str, value: Optional[float],
               findings: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Green up, coral down -- for a pass-rate move measured on ONE test. When the recipe change's
    evidence says the limit table changed (or was mixed) across it, the row keeps its recorded
    move but gets no colour and a "different test" tag (see _tags): part of that move may be a
    change of test, not of parts, and CLAUDE.md's rule -- never compare pass rates across a
    table change -- outranks the mockup's colour. `findings` is required on purpose: a caller
    that forgot it would colour every move."""
    if group != "history" or value is None or value == 0:
        return None
    if any(_different_test(f) for f in findings):
        return None
    return "up" if value > 0 else "down"
