"""Setup-change detection: a laser setting BEYOND the cut recipe changed -- did the pass rate follow?

`recipe_change` covers the cut recipe (how many cuts, and each one's cut-length setting -- the
per-pass `trim_passes` log). Everything else the laser was told to do for a file -- power,
duration, pulse rate, frequency, tolerances, indexing method, and the rest of the 40-odd keys
`trim_setup.parameters` carries -- is this analyzer's job (TRACKER B1c: "captured and analysed by
nothing"). Same shape as `recipe_change`, applied to a dict instead of a recipe tuple: per (model,
laser, track), order a key's values by date, find runs of a constant value, and report each
boundary where the run either side is long and large enough to trust, held to ONE limit table.

**Never claims the setting caused the move.** Like `recipe_change`, this is a "what changed"
history entry, not a recommendation -- `expected_gain_points` is always None, and median incoming
resistance travels beside the pass-rate move for the same reason recipe_change discloses it: a
setting and the material often change together.

**Never across a limit table change.** A pass rate is a verdict against a test (CLAUDE.md). Unlike
`recipe_change`, which still reports a like-for-like figure when the table moved too, this
analyzer says nothing when the before and after runs are not graded against the exact same single
table -- ruling 6 asks for the move "with the limit table held constant", not a disclosed blend.

**No move-size filter.** Every boundary that clears `MIN_RUN_DAYS` and `MIN_TRACKS_SIDE` on both
sides, on one table, is reported, whether the pass rate moved a lot, a little, or not at all --
the question this analyzer answers is "did it follow", and a "no" is as informative as a "yes".
Ruling 6's own containment against a noisy history group is exactly these two floors and the
table rule, not a move threshold.

**`EXCLUDED`** holds the identity-like keys actually present in the six local fixtures'
`trim_setup.parameters` (`dlts_7553_10B`, `dlts_8074_18`, `dlts_8232-1_242/243`,
`lts_8232-1_193/194` -- see test_findings_data.py's `fixture_db`), not a guess at a wider corpus
that is not on disk here:
  - `alias` -- names the TRACK ("Outer Track" / "Inner Track" / "Track"), not a laser setting.
  - `model` / `model_number` -- restates the model number the file is already filtered to.
  - `track_parameters` -- the block's own section header ("SEC1-TRK1" / "SEC1-TRK2"), naming
    which track's block this is, the same thing `track_name` already carries.
None of serial, date, file name, operator or comment keys are present in these two sheets at all
(a per-unit serial/date lives on `analysis_results`, not in the laser's own parameter block) --
so the narrower set above is the whole of what needs excluding here, not an oversight.

**`ALIASES`** folds laser 1's `Response` (key `response`) and laser 2's `Response (Linear or
Function)` (key `response_linear_or_function`) onto one canonical key, ruled one setting on
2026-09-23. Confirmed against `core.trim_setup.normalise_key` on the fixtures directly:
`normalise_key("Response") == "response"`, `normalise_key("Response (Linear or Function)") ==
"response_linear_or_function"`. Grouping is already per (system, track) -- so aliasing does not
change which tracks get compared, only the KEY NAME a finding reports, so the same setting reads
the same regardless of which laser wrote it.

**Values are compared per their own kind.** A numeric value (`int`/`float`, `bool` excluded) is
rounded to 6dp and compared as a number, so `200` and `200.0` -- an int one file and a float the
next, both meaning the same reading -- are never reported as a "change". Anything else is compared
as its `str()` form (the brief's "non-numeric values are compared as strings"), which also means a
value that arrived typed inconsistently (e.g. `"Linear"` one file, `Linear` -- already a str --
the next) still compares equal instead of manufacturing a boundary out of a parsing accident.

**Cut length stays in scope, worded with no direction.** `laser_cut_length` (System B/laser 1,
unitless machine counts) and any resistance-window key (`initial_resistance_lower_limit`, etc.,
System A/laser 2's `..._mm` cut-length variant does not appear in these fixtures at all) are NOT
identity-like, so `EXCLUDED` leaves them in scope -- CLAUDE.md's "cut length is two different
quantities" trap is about POOLING laser 1's and laser 2's numbers together, which never happens
here (grouping is per (model, laser, track), so the two lasers' own runs are never compared to
each other), and about calling a laser-1 move "longer"/"shorter", which this module's title and
summary never do for ANY key: they always say "changed from A to B", never a direction or a unit.
A `laser_cut_length` boundary can therefore also surface here alongside whatever `recipe_change`
already says about the same period from the per-pass log -- two analyzers looking at two different
captures of a related setting, not a bug to suppress.

**Track 2 setup** (`findings/data.py::load_model_tracks`) is `{**parameters, **track2_parameters}`
on a TRK2 track when a Track 2 block was captured (Task 9, System A two-track files only), else
Track 1's own `parameters` -- resolved once in the loader, read here as `track.setup` without
knowing which case it was.
"""
from statistics import median
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from ..model import Finding
from ..stats import pct

MIN_RUN_DAYS = 60          # each side of a change must span at least this many days
MIN_TRACKS_SIDE = 100       # graded tracks needed on EACH side before a change is reported

# Laser 1's `Response` and laser 2's `Response (Linear or Function)` are one setting (2026-09-23).
# {alias-of -> canonical}; extend here, never by special-casing a laser in the analyzer body.
ALIASES: Dict[str, str] = {"response_linear_or_function": "response"}

# Identity-like keys actually present in the fixtures' trim_setup.parameters -- see the module
# docstring for what each one is and why serial/date/file-name/operator/comment keys are not here.
EXCLUDED: FrozenSet[str] = frozenset({"alias", "model", "model_number", "track_parameters"})


def _normalise(v: Any):
    """One parameter value, ready to compare: a bool or non-numeric as its `str()`, a number
    rounded to 6dp so an int one file and an equal float the next never look like a change."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    return str(v)


def _canonical_setup(setup: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One track's `setup` dict, alias-mapped and identity-filtered, values normalised. `None`
    values are dropped -- a key a file did not capture is absent, not a value of "None"."""
    out: Dict[str, Any] = {}
    if not setup:
        return out
    for k, v in setup.items():
        if v is None:
            continue
        key = ALIASES.get(k, k)
        if key in EXCLUDED:
            continue
        out[key] = _normalise(v)
    return out


def _fmt(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _span_days(tracks) -> int:
    return (max(t.file_date for t in tracks) - min(t.file_date for t in tracks)).days


def _single_table_key(tracks) -> Optional[str]:
    """The one limit-table key common to every track in `tracks`, or None when there is not
    exactly one (no table at all, or more than one in play) -- "held constant" means singular."""
    keys = {t.limit_table.key for t in tracks if t.limit_table is not None}
    return keys.pop() if len(keys) == 1 else None


def _side(tracks) -> Dict[str, Any]:
    graded = [t.linearity_pass for t in tracks if t.linearity_pass is not None]
    rs = [t.untrimmed_resistance for t in tracks if t.untrimmed_resistance]
    return {"n": len(tracks), "trim_pass_pct": pct(graded), "graded_n": len(graded),
            "median_incoming_r": median(rs) if rs else None,
            "first": min(t.file_date for t in tracks).date().isoformat(),
            "last": max(t.file_date for t in tracks).date().isoformat()}


def _finding(model: str, system: str, track_name: str, key: str,
            before_value: Any, after_value: Any, b: Dict[str, Any], a: Dict[str, Any],
            laser_label) -> Finding:
    setting = key.replace("_", " ")
    where = laser_label(system) if track_name == "default" else f"{laser_label(system)} · {track_name}"
    title = f"{laser_label(system)}: {setting} changed from {_fmt(before_value)} to {_fmt(after_value)}"
    summary = (
        f"Between {b['last']} and {a['first']} {setting} on {where} changed from "
        f"{_fmt(before_value)} to {_fmt(after_value)}. Units leaving the laser inside their "
        f"linearity limits went from {b['trim_pass_pct']:.0f}% ({b['graded_n']:,} tracks) to "
        f"{a['trim_pass_pct']:.0f}% ({a['graded_n']:,} tracks).")
    if b["median_incoming_r"] and a["median_incoming_r"]:
        summary += (f" Median incoming resistance was {b['median_incoming_r']:,.0f} before and "
                    f"{a['median_incoming_r']:,.0f} after, so {setting} is not the only thing "
                    "that changed.")
    return Finding(
        model=model, analyzer="setup_change", category="Setting change",
        lever="laser_settings", systems=(system,),
        title=title, summary=summary,
        n_units=b["n"] + a["n"],
        strength_name="tracks on the smaller side of the change",
        strength_value=float(min(b["graded_n"], a["graded_n"])),
        expected_gain_points=None,             # a detection, never a recommendation
        evidence={"before": {**b, "value": before_value}, "after": {**a, "value": after_value},
                 "track": track_name, "setting": key})


def analyze(model: str, tracks, laser_label) -> Tuple[List[Dict[str, Any]], List[Finding]]:
    """(one entry per reported boundary, for the facts strip; findings -- the same population,
    Finding-shaped). Below `MIN_RUN_DAYS`/`MIN_TRACKS_SIDE` on either side, off a single shared
    limit table, or ungraded throughout: not a comparable measurement, so neither a fact nor a
    finding (ruling 6's spec is silent on "always"; this analyzer's floors gate both alike)."""
    history: List[Dict[str, Any]] = []
    findings: List[Finding] = []
    cut = [t for t in tracks if t.passes and t.setup]      # a real cut, with a captured setup block
    for system in sorted({t.system for t in cut}):
        on_laser = [t for t in cut if t.system == system]
        for track_name in sorted({t.track_name for t in on_laser}):
            rows = sorted((t for t in on_laser if t.track_name == track_name),
                          key=lambda t: t.file_date)
            canon = {t.track_id: _canonical_setup(t.setup) for t in rows}
            keys = set()
            for c in canon.values():
                keys.update(c)
            for key in sorted(keys):
                seq = [(t, canon[t.track_id][key]) for t in rows if key in canon[t.track_id]]
                if len(seq) < 2:
                    continue
                runs: List[Dict[str, Any]] = []            # [{"value": v, "tracks": [...]}, ...]
                for t, v in seq:
                    if runs and runs[-1]["value"] == v:
                        runs[-1]["tracks"].append(t)
                    else:
                        runs.append({"value": v, "tracks": [t]})
                for before, after in zip(runs, runs[1:]):
                    bt, at = before["tracks"], after["tracks"]
                    b, a = _side(bt), _side(at)
                    if b["graded_n"] < MIN_TRACKS_SIDE or a["graded_n"] < MIN_TRACKS_SIDE:
                        continue                            # too thin a side to call
                    if _span_days(bt) < MIN_RUN_DAYS or _span_days(at) < MIN_RUN_DAYS:
                        continue                             # too short a run to trust
                    tb, ta = _single_table_key(bt), _single_table_key(at)
                    if tb is None or ta is None or tb != ta:
                        continue                             # never across a limit-table change
                    if b["trim_pass_pct"] is None or a["trim_pass_pct"] is None:
                        continue                             # nothing graded on one side
                    history.append({"system": system, "track": track_name, "setting": key,
                                    "before": {**b, "value": before["value"]},
                                    "after": {**a, "value": after["value"]}})
                    findings.append(_finding(model, system, track_name, key,
                                             before["value"], after["value"], b, a, laser_label))
    return history, findings
