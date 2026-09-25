"""How the Findings page and tab arrange findings: group, number, merge, order.

Pure data, no Tk -- so every rule those screens follow can be tested without a window. The
screens draw what arrange() returns and decide nothing themselves.
Spec: docs/superpowers/specs/2026-09-23-design-system-and-findings-page-design.md, section 3.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from laser_trim_analyzer.core.activity import inactive_tag

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
              "A different setting or laser did better on the same test, or the loss starts "
              "before the laser", "tracks a year", "act",
              "Nothing here yet. A finding appears when a model ran two cut settings, two "
              "incoming-resistance windows or two lasers on the same test and one did clearly "
              "better, or when a unit's incoming linearity already predicts the laser's verdict."),
    GroupSpec("laser_time", "Laser time you could save",
              "Units cut that didn't need it, given more cuts than planned, or hand-trimmed "
              "after failing at the laser", "tracks", "act",
              "Nothing here yet. A finding appears when units arrive already inside their limits, "
              "take more cuts than their recipe asks for, or fail at the laser and pass final "
              "test after hand trim."),
    GroupSpec("check", "Check the test",
              "Graded against more than one limit table, or to different limits than final test, "
              "so pass rates across the change don't compare",
              "tracks", "check",
              "Nothing to check. A finding appears when a model is graded against more than one "
              "limit table, or when the laser and final test grade it to different limits."),
    GroupSpec("history", "What changed", "Recipe and setting changes, newest first",
              "pass-rate move", "act",
              "No recipe or setting changes found. A row appears when a model's cut recipe or "
              "laser setup changed between two stable runs, with the pass rate either side."),
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
    "loss_origin": "yield",
    "trim_effort": "laser_time",
    "pass_burden": "laser_time",
    "rework_load": "laser_time",
    "limit_tables": "check",
    "station_setup": "check",
    "recipe_change": "history",
    "setup_change": "history",
}

_GRADE_TAG = {"same_days": "same days", "side_by_side": "side by side",
              "two_periods": "two periods · test first"}
_GRADE_ORDER = ("two_periods", "side_by_side", "same_days")        # weakest first
_LASER_TIME_FIELD = {"Trim avoidance": "arrive_in_spec_n", "Pass effectiveness": "multi_cut_n",
                     "Multi-pass burden": "tracks_over_recipe", "Rework load": "rework_unit_days"}
# A readout whose unit is not its group's column names that unit on its own row. The laser-time
# column counts TRACKS; rework load counts unit-days -- a two-track unit final-tested once per
# track is one unit-day (review of 85222c4, 2026-09-25).
_READOUT_UNIT = {"rework_load": "unit-days"}
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
    # Whether core/activity calls the model INACTIVE (James, 2026-09-25: labelled, never hidden), and
    # its newest trim file -- None when it has none on record, and for every active row.
    inactive: bool = False
    inactive_since: Optional[datetime] = None

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


def _configured_disagrees(finding) -> bool:
    """ink_target: the recommended incoming-resistance window lies wholly outside the station's
    own configured window (evidence["configured_disagrees"] -- None when nothing is configured)."""
    ev = finding.get("evidence") or {}
    return bool(ev.get("configured_disagrees"))


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
        when, body = _history_when(finding), _history_body(finding)
        return f"{when} · {body}" if when else body
    return title


def _history_when(finding) -> str:
    """When a history row's change happened: the month the new run began -- and for a setup
    change, the month the stable setup before it ended too, when that differs: the change happened
    somewhere in the up-to-60 days between the two (setup_change's MIN_RUN_DAYS cap), and a row
    names every month that span touches rather than only its last."""
    ev = finding.get("evidence") or {}
    when = _month((ev.get("after") or {}).get("first"))
    if finding.get("analyzer") == "setup_change":
        since = _month((ev.get("before") or {}).get("last"))
        if since and when and since != when:
            when = f"{since} – {when}"
    return when


def _history_body(finding) -> str:
    """What a history row says changed, after its date: "Laser 1 (LTS): cut 6800 → 6900"."""
    title = str(finding.get("title") or "")
    m = _RECIPE.match(title)
    return f"{m['laser']}: {_recipe_move(m['a'], m['b'])}" if m else title


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
    if any(_configured_disagrees(m) for m in members):
        tags.append("outside the configured window")
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


def _name_what_differs(rows: List["Row"], history: bool = False) -> None:
    """Rows of one group and model that would read IDENTICALLY -- limit_tables writes one
    finding per track and its title names neither -- say what tells them apart, right after the
    laser: "Laser 1 (LTS) · Track A: the limit table changed". A row that reads uniquely is left
    exactly as worded.

    A history row leads with its date, so its twins are the rows that say the same thing AFTER
    the date (setup_change writes one finding per track, and two tracks' stable setups can meet
    in different months), and the track goes after the laser and before the date (final review,
    2026-09-25, M8): "Laser 1 (LTS) · Track A · Jun 2026: Laser PRR 2000 → 3000"."""
    same: Dict[Tuple[str, str], List[Row]] = {}
    for r in rows:
        said = _history_body(r.findings[0]) if history else r.statement
        same.setdefault((r.model, said), []).append(r)
    for twins in same.values():
        if len(twins) < 2:
            continue
        for r in twins:
            scope = _scope(r)
            if not scope:
                continue
            laser = _laser(r.findings[0])
            if history:
                when, body = _history_when(r.findings[0]), _history_body(r.findings[0])
                if laser and body.startswith(laser + ":"):
                    at = f" · {when}" if when else ""
                    r.statement = f"{laser} · {scope}{at}:{body[len(laser) + 1:]}"
                    continue
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


def arrange(findings: Sequence[Dict[str, Any]], *, include_empty: bool = True,
            inactive: Optional[Mapping[str, Optional[datetime]]] = None) -> List[Group]:
    """`inactive` = {model: newest trim file, or None for none on record} for the models
    core/activity calls inactive (the screen's own load_activity; F5, 2026-09-25): each of their
    rows gets the quiet tag "Inactive · last trimmed Mon YYYY" (or "... no trims on record"),
    `inactive` and `inactive_since`. Nothing else changes -- every row stays, in the same order, so
    every count stays too; `preview` is where active rows come first."""
    inactive = inactive or {}
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
            if r.model in inactive:
                r.inactive, r.inactive_since = True, inactive[r.model]
                r.tags.append(inactive_tag(r.inactive_since))
            # Sorted, so a refresh that returns the merged findings in another order keeps the key.
            r.ident = tuple(sorted((_identity(x) for x in r.findings), key=repr))
            if len(r.findings) > 1:
                # Recompute explicitly from every member -- see _merged_statement. r.statement/base
                # were set from whichever finding created the row, which must not stand once a
                # second one has merged into it.
                r.statement = r.base = _merged_statement(r.findings)
        _name_what_differs(rows, history=spec.key == "history")
        _unique_keys(rows)
        _sort(spec.key, rows)
        if rows or (include_empty and spec is not OTHER):
            out.append(Group(spec, rows))
    return out


def preview(group: Group, n: int) -> List[Row]:
    """The rows a group shows before "Show all": its first `n`, ACTIVE models' rows first and then
    the inactive ones, each part in the group's own order (James, 2026-09-25: a model not trimmed
    in two years is labelled, never hidden -- so it can fall down a short list, never off the long
    one). The full list (`group.rows`) keeps its order. "What changed" is a history, newest first
    by date, and stays that way."""
    rows = group.rows
    if group.spec.key != "history":
        rows = [r for r in rows if not r.inactive] + [r for r in rows if r.inactive]
    return rows[:n]


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


def _move_text(value: float) -> str:
    """A pass-rate move, signed -- "±0" when it rounds to nothing: "+0" and "-0" claim a direction
    the number does not have (re-review, 2026-09-25: "Laser Power 52 → 60 · -0")."""
    text = f"{value:+.0f}"
    return "±0" if text in ("+0", "-0") else text


def value_text(group: str, value: Optional[float],
               findings: Sequence[Dict[str, Any]] = ()) -> str:
    """`findings` = the row's own findings: a readout counted in something other than its group's
    column (_READOUT_UNIT) says so -- "261 unit-days", never a bare 261 under "tracks"."""
    if value is None:
        return "—"
    if group == "yield":
        return f"~{value:,.0f}"
    if group == "history":
        return _move_text(value)
    units = {_READOUT_UNIT.get(f.get("analyzer")) for f in findings}
    unit = next(iter(units)) if len(units) == 1 else None
    return f"{value:,.0f} {unit}" if unit else f"{value:,.0f}"


def value_tone(group: str, value: Optional[float],
               findings: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Green up, coral down -- for a pass-rate move measured on ONE test. When the recipe change's
    evidence says the limit table changed (or was mixed) across it, the row keeps its recorded
    move but gets no colour and a "different test" tag (see _tags): part of that move may be a
    change of test, not of parts, and CLAUDE.md's rule -- never compare pass rates across a
    table change -- outranks the mockup's colour. `findings` is required on purpose: a caller
    that forgot it would colour every move. A move that reads "±0" (_move_text) is neither."""
    if group != "history" or value is None or _move_text(value) == "±0":
        return None
    if any(_different_test(f) for f in findings):
        return None
    return "up" if value > 0 else "down"
