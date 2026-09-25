"""Where the loss is made: does the incoming sweep already predict the laser's verdict?

A track arrives at the laser with an untrimmed sweep of its own -- its own error before any cut,
and its own resistance. If tracks that go on to FAIL the laser already looked worse coming in,
the loss was made upstream (deposition), not at the laser: no amount of cutting recovers a mark
the wafer never had room to reach. If incoming and outcome are unrelated, the laser's own
settings are still the first place to look -- the other analyzers already cover that ground.

This is always a fact, per laser, for the Model page's Findings tab, for ANY laser that graded at
least one track in the window -- Ruling 2 says so explicitly ("Stored in facts["loss_origin"]...
always"), and that is what this module does: how well the incoming measurement separates a laser
PASS from a laser FAIL, as an AUC (the probability a FAIL's score is worse than a PASS's; 0.5 = no
better than chance, 1.0 = perfect separation, half credit on a tie). Two scores are tracked -- the
untrimmed sweep's largest error magnitude, and the untrimmed resistance -- but only the error AUC
can ever produce a finding: resistance is `ink_target`'s question, with its own holdout and its
own floor.

**One test per laser** (final review, 2026-09-25, M2). The score separates PASS from FAIL, and a
pass is a verdict against a test: pooling two limit tables would let a difference between the
TESTS read as incoming linearity. So each laser is scored on its busiest limit table in the window
(`findings.data.tables_of`) and nothing else; `other_tables_n` counts the graded tracks on its
other tables, left out. On the copy of 2026-09-25 four lasers that clear the floors graded against
two to four tables in the year (6126, 8232-1, 8340 and 8340-1 on laser 1); scoring the busiest
table changes no call (6607 laser 1 remains the one finding).

`facts[laser]` = {"n", "fails", "auc_error", "auc_resistance", "limit_table": {"key",
"graded_points", "tracks"}, "other_tables_n"} for every laser with at least one graded track on a
limit table in the window -- unconditionally (fix, 2026-09-24: an earlier pass here copied
`machine_compare`'s "facts only for a comparable population" rule for consistency across this
plan's analyzers, which was right for `machine_compare` -- its spec is silent on the point -- but
wrong here, where Ruling 2 says "always" in as many words). "n"/"fails" count the error-scored
population; `auc_error`/`auc_resistance` are `None` only when they truly cannot be computed -- no
scored track on one side at all (no fails, or no passes, with a usable reading), which `auc()`
itself already reports as `None`. A FINDING additionally needs the population to be trustworthy --
MIN_TRACKS scored tracks in the window, with at least MIN_PER_OUTCOME of each outcome -- AND the
error AUC to clear STRONG_AUC: a real-but-thin AUC (8202-1, 0.71 over only 27 failures) or a
real-but-weak one (8232-1 and 8340-1, both 0.57) is a fact, sitting on the tab beside its own
"n"/"fails" so a sub-floor number is never mistaken for a call -- never a finding. The other
analyzers already cover the laser's own levers, and this one names deposition only when the
incoming measurement is both trustworthy and a strong predictor. `machine_compare` keeps its own
"facts only for a comparable pair" rule -- this fix does not touch it.

Score = the untrimmed sweep's largest error magnitude (max|error| across its recorded points)
and, separately, the untrimmed resistance. A track with no usable untrimmed error reading is
skipped from the error AUC entirely -- never scored 0, which is a false floor no laser earns.
Same for resistance: None and the work database's 1e9+ junk readings are skipped, valid range
0 < r < 1e9. "n" and "fails" in facts count the error-scored population specifically -- what the
error AUC is actually computed over.

The window is the MODEL's own latest year -- LOOKBACK_DAYS back from the latest `file_date`
among ALL its tracks, across every laser (the engine's own `annual_volume` convention), computed
once before the per-laser split. Never each laser's own latest date: a laser that stopped running
the model early must not quietly widen its own window relative to the others.

Measured (latest year, 2026-09-24): 6607 on laser 1 (LTS) AUC 0.74 over 1,290 tracks, 465 failing
-- the only one of eight models checked that clears every floor. 8232-1 0.57 and 8340-1 0.57:
made at the laser, as the 2026-09-17 rework study already found. 8202-1 0.71 but only 27 failures,
under the 50-per-outcome floor -- a real, visible AUC with too little of one outcome to call a
finding.

This routes the question; it never grades a part, and the lever it names -- deposition -- is
upstream of anything the laser or its settings can fix.
"""
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..data import tables_of
from ..model import Finding
from ..stats import auc, plausible_resistance   # auc: the one AUC, shared with rework_load's rank test

MIN_TRACKS = 300            # scored tracks for a laser, below this an AUC is a taste, not a rate
MIN_PER_OUTCOME = 50         # each outcome (fail, pass) must clear this or the rarer one is noise
STRONG_AUC = 0.70             # below this the laser's own settings are still the first thing to check
LOOKBACK_DAYS = 365


def _score_error(t) -> Optional[float]:
    """The untrimmed sweep's largest error magnitude, or None when there is nothing usable to
    read -- a track with no untrimmed sweep, or one whose every point is None, is skipped, never
    scored 0."""
    if not t.untrimmed_errors:
        return None
    vals = [abs(e) for e in t.untrimmed_errors if e is not None]
    return max(vals) if vals else None


def _score_resistance(t) -> Optional[float]:
    r = t.untrimmed_resistance
    return r if plausible_resistance(r) else None


def _split(rows, score_fn) -> Tuple[List[float], List[float]]:
    """(fail scores, pass scores) for `rows` -- already graded (linearity_pass is not None) --
    skipping any track `score_fn` cannot score."""
    fails: List[float] = []
    passes: List[float] = []
    for t in rows:
        s = score_fn(t)
        if s is None:
            continue
        (fails if not t.linearity_pass else passes).append(s)
    return fails, passes


def analyze(model: str, tracks, laser_label) -> Tuple[Dict[str, Any], List[Finding]]:
    facts: Dict[str, Any] = {}
    findings: List[Finding] = []
    dated = [t for t in tracks if t.file_date is not None]
    if not dated:
        return facts, findings
    latest = max(t.file_date for t in dated)
    recent = [t for t in dated if t.file_date >= latest - timedelta(days=LOOKBACK_DAYS)]

    by_laser: Dict[str, List] = {}
    for t in recent:
        by_laser.setdefault(t.system, []).append(t)

    for system, rows in sorted(by_laser.items()):
        tables = tables_of(t for t in rows if t.linearity_pass is not None)
        if not tables:
            continue                                    # nothing graded on a test -- not even a fact
        busiest = max(tables, key=lambda k: (tables[k], k))
        graded = [t for t in rows if t.linearity_pass is not None
                  and t.limit_table is not None and t.limit_table.key == busiest]
        fails_e, passes_e = _split(graded, _score_error)
        n_fails, n_passes = len(fails_e), len(passes_e)
        n = n_fails + n_passes
        auc_error = auc(fails_e, passes_e)              # None only when one side scored nothing
        fails_r, passes_r = _split(graded, _score_resistance)
        auc_resistance = auc(fails_r, passes_r)
        # A fact for every laser that graded at least one track here -- Ruling 2 says "always";
        # the thresholds below gate only whether it is ALSO a finding (fix, 2026-09-24: an
        # earlier pass wrongly excluded a sub-floor laser from facts too, copying machine_compare's
        # rule where this module's own spec says otherwise -- see the module docstring).
        facts[laser_label(system)] = {
            "n": n, "fails": n_fails, "auc_error": auc_error, "auc_resistance": auc_resistance,
            "limit_table": {"key": busiest, "graded_points": graded[0].limit_table.graded,
                            "tracks": tables[busiest]},
            "other_tables_n": sum(tables.values()) - tables[busiest]}

        if n < MIN_TRACKS or min(n_fails, n_passes) < MIN_PER_OUTCOME:
            continue                                    # not enough of one outcome to trust a CALL
        # auc_error cannot be None here: clearing MIN_PER_OUTCOME (> 0) means both fails_e and
        # passes_e are non-empty, the only case auc() returns None for.
        if auc_error < STRONG_AUC:
            continue                                    # a real, weak AUC: a fact, never a finding
        findings.append(Finding(
            model=model, analyzer="loss_origin", category="Where the loss is made",
            lever="deposition", systems=(system,),
            title=(f"{laser_label(system)}: incoming linearity predicts the laser verdict "
                   f"(AUC {auc_error:.2f})"),
            summary=(
                "Tracks that fail at the laser already arrived with worse linearity: the loss "
                "starts before the laser, so deposition is the lever to look at first. This "
                "routes the question; it never grades a part."),
            n_units=n,
            strength_name="AUC, incoming max|error| vs laser verdict",
            strength_value=round(auc_error, 3),
            expected_gain_points=None,          # routes the question; claims no yield gain
            evidence={"facts": facts[laser_label(system)]}))
    return facts, findings
