"""Setup-change detection: the laser's setup beyond the cut recipe changed -- did the pass rate follow?

`recipe_change` covers the cut recipe. Everything else the laser was told to do for a file -- power,
pulse duration, pulse rate, speeds, tolerances, indexing, ignored points and the rest of the block
`trim_setup.parameters` carries -- is this analyzer's (TRACKER B1c: "captured and analysed by
nothing").

**A change is a boundary between two consecutive STABLE SETUPS** on one (model, laser, track) --
the final review of 2026-09-25 (C1, risk 10). The first version followed each setting's own run of
constant value, and that run spanned every other change on the track: 6607's pulse-duration change
of 2025-06-19 read 75.5% -> 64.4% (a fall, coloured coral) because its "before" pooled five
setups; setup against setup the move was 38.6% -> 67.6%. And one day's seven changed settings on
6644-04 were seven rows. So:

  * The track's files are cut into SETUPS on the whole setup: every captured key except identity
    keys (`EXCLUDED`), per-unit readings (`READINGS`) and mirrored label rows (below), plus the cut
    recipe -- `recipe_change.RECIPE_PARAMETER_KEYS` from the block and the first cut's setting from
    the pass log (laser 2's cut length lives only there) -- which bounds a setup but is never named
    here: recipe_change owns it. A file starts a new setup when any key it carries holds a value
    different from that key's last captured value; a key a file does not carry keeps its last
    value, and a key's first capture is not a change (nothing was known before it).
  * A setup is STABLE when it has `MIN_TRACKS_SIDE` graded tracks spanning `MIN_RUN_DAYS`.
  * Each pair of consecutive stable setups is compared, stable against stable. A short-lived setup
    between them (one that does not meet the floors) is absorbed: its tracks are in neither side,
    and the settings that differ between the two stable setups are one compound change (6607's
    power change of 2026-01-06 and pulse change of 2026-01-12 become one). A change with no stable
    setup on a side is not reported, and neither is one whose two stable setups are more than
    MIN_RUN_DAYS apart, last file of the one to first file of the other (controller ruling,
    2026-09-25): a transition longer than a stable setup's own minimum is a period, not one change
    -- it stays a fact, never a finding. (With no cap, 8340-3 compared two setups 9.6 years
    apart.) The summary gives both sides' dates and says how many short-lived setups (and
    tracks) sat between them.
  * One finding per (laser, track) change, naming EVERY setting that differs, with the file's own
    label (`LABELS`). A setting that moved and came back inside the absorbed stretch is no change.
  * Facts keep every per-setting boundary, reported or not, each with the reason it was not.

**Never across a limit-table change.** A pass rate is a verdict against a test (CLAUDE.md): each
side must be graded against ONE limit table, the same one, or the change is not reported.

**Never claims the setting caused the move.** A history entry, not a recommendation --
`expected_gain_points` is always None; median incoming resistance travels beside the move because
a setting and the material often change together; a cut recipe that changed at the same point is
said too.

**Track 2's own settings** (I2). A TRK2 track whose file captured no Track 2 block carries Track 1's
block (`TrackView.setup_inherited`). Laser 2's 'Track Parameters' sheet holds every laser setting
per track -- its 'Model Parameters' holds only identity, counts and axis limits -- so such a track
has no known setup of its own and is left out here entirely: an inherited block contributes no
key, so no setup ever spans an inherited stretch and a captured one. (The work database has 0 of
1,868 two-track analyses captured; without this, the day capture starts, Track 1's 10,350-ohm
incoming limit giving way to Track 2's 4,000 would read as a setting change.)

**`EXCLUDED` -- identity, never a setting:** `alias` (names the track), `model` / `model_number`
(restate the model), `track_parameters` (the block's own header), `customer` and `drawing` (whose
part it is), `template_updated` (the template's revision date), `report_info` (a section header).

**`READINGS` -- what the machine measured or found, not what anyone set** (curated on a copy of
the work database, 2026-09-25: per key, how often it differs between consecutive files of a
(model, laser, track), and how often a change lands on a value that track never held before -- a
setting returns to known values or changes rarely, a reading keeps producing new ones. The rates
are not bimodal, so no threshold decides this alone):
  * laser 2 / laser 3 format -- `length_theoretical`, `starting_position`, and the four
    `error_split_{low,high}_{voltage,position}`: the element's measured length and the coordinates
    derived from it, per unit. 5,000+ distinct values over 57,909 laser-2 files; they differ on
    20-22% of consecutive files, in lockstep; 78% of their changes land on a value never held
    before (the review: 874-914 distinct values over 1,434-1,875 files on 8204-3 and 6126).
    `angle_theoretical` is the rotary twin (8614: 14 distinct values over 113 files).
  * `laser_height`: a machine coordinate found at each setup (the focus), not a dialed setting --
    1,583 distinct values across 294 laser-2 groups; 83% of its changes land on a new value and
    55% happen with no other setting moving. Counted as a setting it splits a setup at every
    re-focus: 57 findings on 19 models fall to 48 on 16.
  * `index_position`, `index_voltage`: found per unit where the machine indexes the element itself
    -- 6828 holds 390 distinct index positions over 669 files, differing on 66% of consecutive
    files; constant on 163 of 166 laser-2 groups (so leaving them out moves no finding today).
  * laser 1 -- `low_/high_error_split_volts`, `low_/high_error_split_position`, `start_position`,
    `pot_angle`, `stop_angle`: the error-split points found on the unit and the travel derived
    from them (6126: 671-692 distinct values over 1,945 files, differing on 47% of consecutive
    files; counted as settings, 57 findings fall to 53); `low_/high_end_volts`,
    `low_/high_end_position`: found per unit on some models (8340: 35 distinct values over 75
    files; 8531-1 Track B: 46 over 106).
The review's list alone (the six laser-2 readings, and the angle twin) gives 44 findings on
14 models. The whole rule, on the copy of 2026-09-25: 57 findings on 19 models.
Kept as settings although they flip on a few models, because a change in them is a change of
what the sweep measures: `num_of_lin_positions` and laser 1's `number_of_readings_lin` / `_trim`
(how many points the sweep takes -- 8232-1's 111 -> 57 readings at the same -55 degree travel is
the known limit-table density change; the count follows the measured length on a few models, such
as 8824's 81 distinct values over 150 files, and there the tables move with it, so the table rule
silences them either way) and `points_from_start` / `points_from_end` (the trim window). A changed
point count changes the limit table, so it only ever BOUNDS a setup -- which splits a two-table
setup into two one-table ones (57 findings on 19 models; 54 on 18 counting it a reading).

**Mirrored label rows.** The parser tries each parameter sheet in both layouts and keeps the
first value it sees, so a laser-1 value-first sheet read label-first leaves rows like
`{"no": "Use Table Theory?"}` beside the real `{"use_table_theory": "NO"}` -- the key is a value,
the value is the label of a real key in the same block (laser 2's label-first sheet read
value-first does the same, and there the key can be a customer or drawing name). Such a row
renames and re-points whenever the real setting moves, so it is dropped: an unknown key whose
text value normalises to a KNOWN key. On the copy this caught every unknown key but two real ones
(`circle_cut_radius`, `trim_delta_source`, both in `LABELS`), and flagged no known key.

**`LABELS`** are the files' own row labels (read with the parser's layouts from the local corpus,
2026-09-25), with two cleanups for a sentence: a numeric-range parenthetical is dropped, as
`normalise_key` drops it ('Laser Power (0-255)' -> 'Laser Power'; a unit such as '(ns)' stays), and
a trailing '?' or ':' goes. `circle_cut_radius` is the one label not read from a file (no local file
carries it; named by its sibling 'Circle Cut Length'). A captured key the map does not know yet is
still a setting -- it bounds a setup, so a change is never pooled -- and is named by its stored key
and "(no label yet)", never as if that were a label.

**`ALIASES`** fold one setting's two names onto one: laser 1's `Response` and laser 2's `Response
(Linear or Function)` (ruled one setting, 2026-09-23), and the laser-2 template's own typo 'Length
Mamimum', later fixed to 'Length Maximum' (25,782 files carry the one, 32,014 the other).

**Values are compared per their own kind**: a number rounded to 6 dp (an int one file and the
equal float the next are no change), anything else as its `str()`.
"""
from statistics import median
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from ...core.trim_setup import normalise_key
from ..data import tables_of
from ..model import Finding
from ..stats import pct, plausible_resistance
from .recipe_change import RECIPE_PARAMETER_KEYS

MIN_RUN_DAYS = 60          # a stable setup spans at least this many days...
MIN_TRACKS_SIDE = 100       # ...with at least this many graded tracks

ALIASES: Dict[str, str] = {"response_linear_or_function": "response",
                           "length_mamimum": "length_maximum"}

EXCLUDED: FrozenSet[str] = frozenset({
    "alias", "model", "model_number", "track_parameters", "customer", "drawing",
    "template_updated", "report_info"})

READINGS: FrozenSet[str] = frozenset({
    # laser 2 / laser 3 format
    "length_theoretical", "angle_theoretical", "starting_position",
    "error_split_low_voltage", "error_split_high_voltage",
    "error_split_low_position", "error_split_high_position",
    "laser_height", "index_position", "index_voltage",
    # laser 1 format
    "low_error_split_volts", "high_error_split_volts",
    "low_error_split_position", "high_error_split_position",
    "start_position", "pot_angle", "stop_angle",
    "low_end_volts", "high_end_volts", "low_end_position", "high_end_position"})

LABELS: Dict[str, str] = {
    # laser 2 / laser 3 format ('Track Parameters', 'Model Parameters')
    "a_axis_current_limit": "A-Axis Current Limit",
    "angle_maximum": "Angle Maximum",
    "angle_minimum": "Angle Minimum",
    "center_tap_lower_limit": "Center-Tap Lower Limit",
    "center_tap_target": "Center-Tap Target",
    "center_tap_upper_limit": "Center-Tap Upper Limit",
    "end_collet_release": "End Collet Release",
    "final_resistance_lower_limit": "Final Resistance Lower Limit",
    "final_resistance_upper_limit": "Final Resistance Upper Limit",
    "high_tap": "HIGH-TAP",
    "indexing_method": "Indexing Method",
    "initial_resistance_lower_limit": "Initial Resistance Lower Limit",
    "initial_resistance_upper_limit": "Initial Resistance Upper Limit",
    "inner_edge_position": "Inner Edge Position",
    "laser_duration_ns": "Laser Duration (ns)",
    "laser_pulse_repetition_rate_hz": "Laser Pulse Repetition Rate (Hz)",
    "length_maximum": "Length Maximum",
    "length_minimum": "Length Minimum",
    "linearity_velocity": "Linearity Velocity",
    "low_tap": "LOW-TAP",
    "next_position": "Next Position",
    "next_position_index": "Next Position Index",
    "num_of_lin_positions": "# of Lin Positions",
    "num_of_sections": "# of Sections",
    "num_of_tracks": "# of Tracks",
    "num_of_trim_parameters": "# of Trim Parameters",
    "outer_edge_position": "Outer Edge Position",
    "plate_move_during_setup": "Plate Move During Setup",
    "resistance_threshold_pct": "Resistance Threshold %",
    "test_voltage": "Test Voltage",
    "theoretical_resistance": "Theoretical Resistance",
    "type": "Type",
    "x_axis_current_limit": "X-Axis Current Limit",
    "y_axis_current_limit": "Y-Axis Current Limit",
    "z_axis_current_limit": "Z-Axis Current Limit",
    # both formats
    "end_position": "End Position",
    "laser_power": "Laser Power",
    "response": "Response",
    # laser 1 format ('Model Parameters', value first)
    "absolute_end_position": "Absolute End Position",
    "auto_retrim": "Auto-Retrim",
    "balance_ends": "Balance Ends",
    "circle_cut_length": "Circle Cut Length",
    "circle_cut_radius": "Circle Cut Radius",
    "coarse_lower_tolerance": "Coarse Lower Tolerance",
    "coarse_trim_voltage": "Coarse Trim Voltage",
    "coarse_upper_tolerance": "Coarse Upper Tolerance",
    "delta_rank": "Delta Rank",
    "element_configuration": "Element Configuration",
    "end_balance_limit": "End-Balance Limit",
    "end_degree_increment": "End Degree Increment",
    "end_degree_start": "End Degree Start",
    "end_point": "End point",
    "ending_points_ignored": "Ending points Ignored",
    "endset": "Endset",
    "endset_voltage_tolerance": "Endset Voltage Tolerance",
    "endsetpct": "Endset%",
    "establish_coordinates": "Establish Coordinates",
    "fine_lower_tolerance": "Fine lower Tolerance",
    "fine_trim_voltage": "Fine Trim Voltage",
    "fine_upper_tolerance": "Fine Upper Tolerance",
    "initial_end_balance": "Initial End-Balance",
    "initial_points_ignored": "Initial Points Ignored",
    "laser_back_length": "Laser Back Length",
    "laser_current": "Laser Current",
    "laser_cut_segments": "Laser Cut Segments",
    "laser_frequency": "Laser Frequency",
    "laser_prr": "Laser PRR",
    "laser_speed_high": "Laser Speed High",
    "laser_speed_slow": "Laser Speed Slow",
    "laser_start_position": "Laser Start Position",
    "max_elec_angle": "max elec. Angle",
    "max_resistance": "Max resistance",
    "max_stroke": "Max Stroke",
    "min_elec_angle": "min elec. Angle",
    "min_resistance": "Min Resistance",
    "min_stroke": "Min Stroke",
    "move_plate_during_setup": "Move plate during setup",
    "number_of_readings_lin": "Number of Readings (Lin)",
    "number_of_readings_trim": "Number of Readings (trim)",
    "points_from_end": "Points From End",
    "points_from_start": "Points From Start",
    "pot_type": "Pot Type",
    "pulse_duration": "Pulse Duration",
    "readings_deg": "Readings/Deg.",
    "reserved": "<Reserved>",
    "source_of_trim_deltas": "Source of trim deltas",
    "speed_crossover": "Speed Crossover",
    "start_point": "Start Point",
    "theo_resistance": "Theo. Resistance",
    "theory_delta_multiplier": "Theory Delta Multiplier",
    "trim_delta_source": "Trim Delta Source",
    "trim_range": "Trim Range",
    "use_table_theory": "Use Table Theory",
    "vfinder_increment_multiplier": "Vfinder Increment Multiplier",
}

# Every key this module has a rule for; a mirrored label row points at one of these.
_KNOWN: FrozenSet[str] = frozenset(LABELS) | frozenset(ALIASES) | EXCLUDED | READINGS | \
    RECIPE_PARAMETER_KEYS
# The first cut's setting from the pass log, as one more recipe component. Bounds, never named.
FIRST_CUT = "__first_cut__"
_RECIPE: FrozenSet[str] = RECIPE_PARAMETER_KEYS | {FIRST_CUT}
_NORMALISED: Dict[str, str] = {}     # normalise_key of a text value, memoised (values repeat heavily)


def _normalise(v: Any):
    """One parameter value, ready to compare: a bool or non-numeric as its `str()`, a number
    rounded to 6dp so an int one file and an equal float the next never look like a change."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    return str(v)


def _mirrored_label(key: str, value: Any) -> bool:
    """A row whose key is a VALUE and whose value is the LABEL of a known key -- a parameter sheet
    read in the other layout (module docstring). Only ever asked of a key this module does not know."""
    if not isinstance(value, str):
        return False
    target = _NORMALISED.get(value)
    if target is None:
        target = _NORMALISED[value] = normalise_key(value)
    return target != key and target in _KNOWN


def _signature(track) -> Dict[str, Any]:
    """What one file tells us of the setup: every captured key but identity, readings and mirrored
    label rows, alias-mapped and normalised -- plus the pass log's first cut. None values are
    dropped: a key a file did not capture is absent, not a value of "None"."""
    out: Dict[str, Any] = {}
    for k, v in (track.setup or {}).items():
        if v is None:
            continue
        key = ALIASES.get(k, k)
        if key in EXCLUDED or key in READINGS:
            continue
        if key not in _KNOWN and _mirrored_label(key, v):
            continue
        out[key] = _normalise(v)
    if track.passes and track.passes[0].cut_setting is not None:
        out[FIRST_CUT] = round(float(track.passes[0].cut_setting), 6)
    return out


def _fmt(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _name(key: str) -> str:
    """The file's own label -- or, for a key the map does not know yet, its stored name AS such."""
    return LABELS.get(key) or f"'{key}' (no label yet)"


def _and(names: List[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _span_days(tracks) -> int:
    return (max(t.file_date for t in tracks) - min(t.file_date for t in tracks)).days


def _single_table_key(tracks) -> Optional[str]:
    """The one limit-table key common to every track in `tracks`, or None when there is not
    exactly one (no table at all, or more than one in play) -- "held constant" means singular."""
    keys = tables_of(tracks)
    return next(iter(keys)) if len(keys) == 1 else None


def _side(tracks) -> Dict[str, Any]:
    graded = [t.linearity_pass for t in tracks if t.linearity_pass is not None]
    rs = [t.untrimmed_resistance for t in tracks if plausible_resistance(t.untrimmed_resistance)]
    return {"n": len(tracks), "trim_pass_pct": pct(graded), "graded_n": len(graded),
            "median_incoming_r": median(rs) if rs else None,
            "first": tracks[0].file_date.date().isoformat(),
            "last": tracks[-1].file_date.date().isoformat(),
            "limit_table": _single_table_key(tracks),
            "track_ids": [tracks[0].track_id, tracks[-1].track_id]}


def _setups(rows) -> List[Dict[str, Any]]:
    """`rows` (one laser and track, in file order) cut into setups. Each: its tracks, the settings
    that changed where it starts [(key, old, new)], and the setup it ran at (every key's last value)."""
    setups: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {}
    for t in rows:
        sig = _signature(t)
        moved = [(k, state[k], v) for k, v in sorted(sig.items()) if k in state and state[k] != v]
        if not setups or moved:
            if setups:
                setups[-1]["state"] = dict(state)
            setups.append({"tracks": [], "moved": moved})
        state.update(sig)
        setups[-1]["tracks"].append(t)
    if setups:
        setups[-1]["state"] = dict(state)
    for s in setups:
        s["side"] = _side(s["tracks"])
        s["stable"] = (s["side"]["graded_n"] >= MIN_TRACKS_SIDE
                       and _span_days(s["tracks"]) >= MIN_RUN_DAYS)
    return setups


def _compare(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Two consecutive stable setups: which settings differ, whether the recipe moved too,
    whether both sides were graded against one and the same limit table, and whether they meet
    within MIN_RUN_DAYS (the last file of the one to the first file of the other)."""
    bs, as_ = before["state"], after["state"]
    diff = [k for k in sorted(bs.keys() & as_.keys()) if bs[k] != as_[k]]
    tb, ta = before["side"]["limit_table"], after["side"]["limit_table"]
    gap = (after["tracks"][0].file_date - before["tracks"][-1].file_date).days
    return {"settings": sorted((k for k in diff if k not in _RECIPE),
                               key=lambda k: (_name(k).lower(), k)),
            "recipe_changed": any(k in _RECIPE for k in diff),
            "gap_days": gap, "within": gap <= MIN_RUN_DAYS,
            "same_table": tb is not None and tb == ta,
            "graded": (before["side"]["trim_pass_pct"] is not None
                       and after["side"]["trim_pass_pct"] is not None)}


def _title(laser: str, settings: List[Dict[str, Any]]) -> str:
    if len(settings) <= 2:
        return f"{laser}: " + ", ".join(f"{_name(s['setting'])} {_fmt(s['from'])} → {_fmt(s['to'])}"
                                        for s in settings)
    return f"{laser}: {len(settings)} settings changed ({_and([_name(s['setting']) for s in settings])})"


def _finding(model: str, system: str, track_name: str, settings: List[Dict[str, Any]],
             b: Dict[str, Any], a: Dict[str, Any], absorbed: List[Dict[str, Any]],
             recipe_changed: bool, laser_label) -> Finding:
    where = laser_label(system) if track_name == "default" else f"{laser_label(system)} · {track_name}"
    moved = "; ".join(f"{_name(s['setting'])} went from {_fmt(s['from'])} to {_fmt(s['to'])}"
                      for s in settings)
    absorbed_tracks = sum(len(s["tracks"]) for s in absorbed)
    summary = (
        f"Between {b['last']} and {a['first']}, on {where}: {moved}. Units leaving the laser inside "
        f"their linearity limits went from {b['trim_pass_pct']:.0f}% ({b['graded_n']:,} tracks, "
        f"{b['first']} to {b['last']}) to {a['trim_pass_pct']:.0f}% ({a['graded_n']:,} tracks, "
        f"{a['first']} to {a['last']}) -- one setup on each side, both graded against the same "
        "limit table.")
    if absorbed:
        summary += (f" {len(absorbed)} short-lived setup{'s' if len(absorbed) != 1 else ''} ran in "
                    f"between ({absorbed_tracks:,} tracks); neither side includes "
                    f"{'them' if len(absorbed) != 1 else 'it'}.")
    if recipe_changed:
        summary += " The cut recipe changed at the same point too, so this move is not the settings' alone."
    if b["median_incoming_r"] and a["median_incoming_r"]:
        which = "this setting is" if len(settings) == 1 else "these settings are"
        summary += (f" Median incoming resistance was {b['median_incoming_r']:,.0f} before and "
                    f"{a['median_incoming_r']:,.0f} after, so {which} not the only thing that changed.")
    return Finding(
        model=model, analyzer="setup_change", category="Setting change",
        lever="laser_settings", systems=(system,),
        title=_title(laser_label(system), settings), summary=summary,
        n_units=b["n"] + a["n"],
        strength_name="tracks on the smaller side of the change",
        strength_value=float(min(b["graded_n"], a["graded_n"])),
        expected_gain_points=None,             # a detection, never a recommendation
        evidence={"before": b, "after": a, "track": track_name, "settings": settings,
                  "absorbed_setups": len(absorbed), "absorbed_tracks": absorbed_tracks,
                  "recipe_changed": recipe_changed})


def analyze(model: str, tracks, laser_label) -> Tuple[List[Dict[str, Any]], List[Finding]]:
    """(every per-setting boundary, for the facts; one finding per reported change)."""
    boundaries: List[Dict[str, Any]] = []
    findings: List[Finding] = []
    own = [t for t in tracks if t.passes and t.setup and not t.setup_inherited]
    for system in sorted({t.system for t in own}):
        for track_name in sorted({t.track_name for t in own if t.system == system}):
            rows = sorted((t for t in own if t.system == system and t.track_name == track_name),
                          key=lambda t: (t.file_date, t.track_id))
            setups = _setups(rows)
            stable = [i for i, s in enumerate(setups) if s["stable"]]
            changes: Dict[Tuple[int, int], Dict[str, Any]] = {}
            for i, j in zip(stable, stable[1:]):
                c = changes[(i, j)] = _compare(setups[i], setups[j])
                if not (c["settings"] and c["within"] and c["same_table"] and c["graded"]):
                    continue
                bstate, astate = setups[i]["state"], setups[j]["state"]
                settings = [{"setting": k, "label": LABELS.get(k), "from": bstate[k], "to": astate[k]}
                            for k in c["settings"]]
                findings.append(_finding(model, system, track_name, settings, setups[i]["side"],
                                         setups[j]["side"], setups[i + 1:j], c["recipe_changed"],
                                         laser_label))
            for idx, s in enumerate(setups):
                for key, old, new in s["moved"]:
                    if key in _RECIPE:
                        continue                          # recipe_change's, never named here
                    boundaries.append({"system": system, "track": track_name, "setting": key,
                                       "label": LABELS.get(key),
                                       "date": s["tracks"][0].file_date.date().isoformat(),
                                       "from": old, "to": new,
                                       **_why(idx, key, setups, stable, changes)})
    return boundaries, findings


def _why(idx: int, key: str, setups, stable: List[int],
         changes: Dict[Tuple[int, int], Dict[str, Any]]) -> Dict[str, Any]:
    """Whether the boundary where setup `idx` starts (for `key`) is part of a reported change,
    and if not, why not."""
    before = next((i for i in reversed(stable) if i < idx), None)
    after = idx if setups[idx]["stable"] else next((j for j in stable if j > idx), None)
    if before is None:
        return {"reported": False, "why_not": "no stable setup before it"}
    if after is None:
        return {"reported": False, "why_not": "no stable setup after it yet"}
    c = changes[(before, after)]
    if not c["within"]:
        return {"reported": False,
                "why_not": (f"the stable setups either side are {c['gap_days']} days apart -- longer "
                            f"than a stable setup's own {MIN_RUN_DAYS}-day minimum, so a period, not "
                            "one change")}
    if key not in setups[before]["state"]:
        return {"reported": False, "why_not": "not captured in the stable setup before it"}
    if key not in c["settings"]:
        return {"reported": False, "why_not": "undone before the next stable setup"}
    if not c["same_table"]:
        return {"reported": False,
                "why_not": "the setups either side were not graded against one and the same limit table"}
    if not c["graded"]:
        return {"reported": False, "why_not": "nothing graded on one side"}
    return {"reported": True, "why_not": None}
