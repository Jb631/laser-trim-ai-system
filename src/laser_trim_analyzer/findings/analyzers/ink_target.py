"""Ink target: does incoming resistance move the result -- with everything else held still?

8232-1's yield history looked like a resistance story (corr -0.63) until two
other changes turned up inside it: a move between lasers and a change from
one cut to two. So this analyzer never pools. It works inside ONE laser, ONE
recipe and ONE limit table -- the most recent such group with enough units --
and only there asks whether incoming resistance separates good from bad.

The limit table is held still because it IS the test: the same model on laser 1
was graded at 89 points until 2025 and at 45 since, and a laxer table passes
more units whatever their resistance. If the incoming target drifted over the
same months, pooling the two tables would credit the resistance with a change
that was really a change of test.

The station's FINAL-RESISTANCE window is held still for exactly the same reason,
and it took a second review to see it: `_success` below asks whether the trimmed
resistance landed inside that window, so the window is part of the test too. With
it left out, a group in which only the window moved produced "aim lower", a
correlation of -0.87 and a claimed gain of 50 yield points -- under a sentence
saying nothing else explained it. Anything a verdict is measured AGAINST belongs
in the group key.

Success is the app's stored trim verdict AND the final resistance landing
inside the station's own final limits (when the file carries them). Final-
test verdicts are deliberately not used: units may be hand-trimmed between
the laser and final test, so final test does not grade the laser's work.
"""
from datetime import timedelta
from typing import List, Optional

from ..model import Finding
from ..stats import pct, spearman
from .recipe_change import describe

# 600, not 200. At 200 this analyzer fires on PURE NOISE 14-15% of the time: the gain is an
# in-sample maximum over seven overlapping candidate windows, with no holdout and no correction for
# having tried seven. A 400-trial Monte Carlo with resistance drawn independently of the outcome
# (final review, 2026-09-20) measured 13.8% at n=200, 10.8% at 300, 1.5% at 600 and 0% at 1200 --
# and MIN_GAIN_POINTS=3.0 filters nothing at 200, where the median noise "gain" is 7 points.
# The lever here is the ink formulation, which costs a lot; it must not be pulled on noise.
MIN_N = 600
BINS = 5
MIN_ABS_RHO = 0.10
MIN_GAIN_POINTS = 3.0
MIN_WINDOW_SHARE = 0.40        # a window nobody can hit is not a recommendation
OUT_OF_SAMPLE = 1.0            # ...and it must survive IN FULL on data the window was not chosen on.
                               # Measured over 300 pure-noise draws: 14% of groups produced a finding at
                               # n=200 with none of this, 2.0% at n=600, 1.0% with the two-fold test
                               # below, 0.0% at n=900 -- while a real 20-point separation is still found
                               # at 700 tracks. The residual 1% is why a finding says how it was measured.
MAX_R = 1e7                    # ohms. The work database holds 1e12 readings; one of them drags the
                               # quantile edges and can put a trillion ohms in a recommended window.


def _usable_r(t) -> bool:
    r = t.untrimmed_resistance
    return isinstance(r, (int, float)) and not isinstance(r, bool) and 0 < r < MAX_R


def _success(t) -> Optional[bool]:
    if t.linearity_pass is None:
        return None
    if t.final_r_low and t.final_r_high and t.trimmed_resistance:
        return bool(t.linearity_pass and t.final_r_low <= t.trimmed_resistance <= t.final_r_high)
    return bool(t.linearity_pass)


def _bins(ts) -> List[dict]:
    """`ts` sorted by resistance, cut into BINS equal counts."""
    n = len(ts)
    edges = [round(i * n / BINS) for i in range(BINS + 1)]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        chunk = ts[lo:hi]
        out.append({"r_low": chunk[0].untrimmed_resistance, "r_high": chunk[-1].untrimmed_resistance,
                    "n": len(chunk), "success_pct": pct([_success(t) for t in chunk])})
    return out


def _best_window(ts) -> Optional[dict]:
    """The contiguous 2- or 3-bin window of `ts` (sorted by resistance) with the best success rate."""
    n = len(ts)
    if n < BINS * 2:
        return None
    edges = [round(i * n / BINS) for i in range(BINS + 1)]
    best = None
    for width in (2, 3):
        for i in range(0, BINS - width + 1):
            chunk = ts[edges[i]:edges[i + width]]
            if len(chunk) / n < MIN_WINDOW_SHARE:
                continue
            rate = pct([_success(t) for t in chunk])
            if best is None or rate > best["success_pct"]:
                best = {"r_low": chunk[0].untrimmed_resistance, "r_high": chunk[-1].untrimmed_resistance,
                        "n": len(chunk), "success_pct": rate}
    return best


def _overlaps(a, b) -> bool:
    """Do the two CLOSED intervals `(low, high)` share at least one point? Touching at a single
    boundary point counts as overlap -- closed intervals include their endpoints."""
    return a[0] <= b[1] and b[0] <= a[1]


def analyze(model: str, tracks, laser_label) -> List[Finding]:
    groups = {}
    for t in tracks:
        if t.passes and _usable_r(t) and _success(t) is not None:
            table = t.limit_table
            groups.setdefault((t.system, t.recipe, table.key if table else None,
                               t.final_r_low, t.final_r_high), []).append(t)
    eligible = [(k, v) for k, v in groups.items() if len(v) >= MIN_N]
    if not eligible:
        return []                                       # thin sample: say nothing
    (system, recipe, _table_key, final_r_low, final_r_high), ts = max(
        eligible, key=lambda kv: max(t.file_date for t in kv[1]))
    table = ts[0].limit_table
    # The biggest eligible group is not always the CURRENT one: raising MIN_N to 600 means a model
    # that has since moved to a new recipe, laser or limit table can have its advice drawn from the
    # older, larger setup. That is honest only if it SAYS so -- advice about a retired setup acted on
    # as if it were current is exactly the kind of wrong action this app exists to prevent.
    newest_overall = max(t.file_date for v in groups.values() for t in v)
    superseded = newest_overall > max(t.file_date for t in ts) + timedelta(days=30)
    ts = sorted(ts, key=lambda t: t.untrimmed_resistance)
    rs = [t.untrimmed_resistance for t in ts]
    ok = [1.0 if _success(t) else 0.0 for t in ts]
    rho = spearman(rs, ok)
    if rho is None or abs(rho) < MIN_ABS_RHO:
        return []                                       # resistance is not the lever here
    n = len(ts)
    overall = pct([bool(x) for x in ok])
    bins = _bins(ts)
    best = _best_window(ts)
    if best is None or best["success_pct"] - overall < MIN_GAIN_POINTS:
        return []
    # OUT OF SAMPLE, because the window above is a maximum over seven overlapping candidates with no
    # correction for having tried seven. Measured on pure noise (final review, 2026-09-20): every gate
    # above passed 14% of the time at n=200 and 2% at n=600, with a median "gain" of 7 points. So the
    # window is now CHOSEN on one period and MEASURED on the other -- the claim a process
    # recommendation actually makes ("had you set this window then, it would have helped since") --
    # and the number reported is the out-of-sample one. Both directions must hold, so neither period
    # can carry it alone.
    by_date = sorted(ts, key=lambda t: t.file_date)
    mid = len(by_date) // 2
    early, late = by_date[:mid], by_date[mid:]
    folds = []
    for pick_on, judge_on in ((early, late), (late, early)):
        chosen = _best_window(sorted(pick_on, key=lambda t: t.untrimmed_resistance))
        if chosen is None:
            return []
        inside = [t for t in judge_on if chosen["r_low"] <= t.untrimmed_resistance <= chosen["r_high"]]
        if len(inside) < MIN_N // 10:                      # too few to judge on
            return []
        fold = pct([_success(t) for t in inside]) - pct([_success(t) for t in judge_on])
        if fold < MIN_GAIN_POINTS * OUT_OF_SAMPLE:
            return []                                      # it does not survive off its own data
        folds.append(round(fold, 1))
    gain = round(sum(folds) / len(folds), 1)               # the honest, out-of-sample number
    direction = "lower" if rho < 0 else "higher"
    # The station's CURRENT incoming window: from the most recent track that carries one.
    # `ts` was sorted by RESISTANCE for the binning above, so walking it backwards finds the
    # highest-resistance track, not the newest -- a stale window reported as the current one.
    with_window = [t for t in ts if t.initial_r_low and t.initial_r_high]
    configured = None
    if with_window:
        newest = max(with_window, key=lambda t: t.file_date)
        configured = (newest.initial_r_low, newest.initial_r_high)
    # Does the station's own setting disagree with what the data says did best? Only asked when
    # there IS a configured window; "disagrees" means no overlap at all -- a window that merely
    # differs from the configured one (but still catches some of it) is not a disagreement.
    configured_disagrees = None if configured is None else \
        not _overlaps(configured, (best["r_low"], best["r_high"]))
    # Branch on the configured window FIRST, then on whether it disagrees: the sentence must not
    # rest on configured_disagrees happening to be None exactly when configured is (review of
    # aa8b20c) -- only the first two branches may read configured[0] / configured[1].
    if configured is None:
        configured_sentence = ("These files carry no configured incoming window, so this is a "
                               "computed target, not a comparison with a setting. ")
    elif configured_disagrees:
        configured_sentence = (f"The station is set to accept {configured[0]:,.0f} to "
                               f"{configured[1]:,.0f} Ω, and the window that did best lies outside "
                               "it. ")
    else:
        configured_sentence = (f"The station is set to accept {configured[0]:,.0f} to "
                               f"{configured[1]:,.0f} Ω incoming. ")
    first, last = min(t.file_date for t in ts).date(), max(t.file_date for t in ts).date()
    # The rate is claimed over THIS group only -- the tracks it was measured on, in the last year of
    # the group's own data. Scaling it by the whole model overstated one real two-laser case 14x.
    newest = max(t.file_date for t in ts)
    scope_year = sum(1 for t in ts if t.file_date >= newest - timedelta(days=365))
    return [Finding(
        model=model, analyzer="ink_target", category="Ink target",
        lever="ink", systems=(system,),
        title=f"Incoming resistance: aim {direction} -- {best['r_low']:,.0f} to {best['r_high']:,.0f} Ω does best",
        summary=(f"Within {laser_label(system)} running {describe(recipe)} ({first} to {last}, {n:,} tracks), units "
                 f"arriving between {best['r_low']:,.0f} and {best['r_high']:,.0f} Ω left the laser good "
                 f"{best['success_pct']:.0f}% of the time against {overall:.0f}% overall "
                 f"({scope_year:,} of those tracks are from the last year, which is what the rate is "
                 f"scaled by). The {gain:+.1f} points is measured OUT OF SAMPLE: the window was chosen "
                 f"on one half of the period and scored on the other, both ways round "
                 f"({folds[0]:+.1f} and {folds[1]:+.1f} points). "
                 + (f"This is NOT the setup the model runs today -- the laser, recipe, limit table or "
                    f"resistance window has changed since {last} (the current setup has too few tracks "
                    f"to judge). Treat it as history unless you go back to it. "
                    if superseded else "")
                 + configured_sentence
                 + "Laser, recipe, limit table and the station's final-resistance window are held constant, "
                   "so none of them explains the difference. Period, operator and lot are not."),
        n_units=n, scope_annual_tracks=scope_year,
        strength_name="Spearman correlation, incoming resistance vs good-at-laser",
        strength_value=rho, expected_gain_points=gain,
        gain_definition=("good-at-laser rate inside the recommended window minus the rate over the rest of "
                         "the period, measured OUT OF SAMPLE (window chosen on one half, scored on the "
                         "other, both ways, averaged) and scaled by the group's own last-year track count; "
                         "good = the app's trim linearity verdict AND final resistance inside the station's "
                         "final limits"),
        evidence={"bins": bins, "window": best, "overall_pct": overall, "recipe": describe(recipe),
                  "out_of_sample_gain_each_fold": folds,
                  "configured_incoming": configured, "configured_disagrees": configured_disagrees,
                  "period": [first.isoformat(), last.isoformat()],
                  "limit_table": None if table is None else {"rows": table.rows, "graded": table.graded},
                  "final_resistance_window": [final_r_low, final_r_high], "superseded": superseded})]
