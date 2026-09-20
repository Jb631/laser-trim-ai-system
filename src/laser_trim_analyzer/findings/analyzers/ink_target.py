"""Ink target: does incoming resistance move the result -- with everything else held still?

8232-1's yield history looked like a resistance story (corr -0.63) until two
other changes turned up inside it: a move between lasers and a change from
one cut to two. So this analyzer never pools. It works inside ONE laser and
ONE recipe -- the most recent such group with enough units -- and only there
asks whether incoming resistance separates good from bad.

Success is the app's stored trim verdict AND the final resistance landing
inside the station's own final limits (when the file carries them). Final-
test verdicts are deliberately not used: units may be hand-trimmed between
the laser and final test, so final test does not grade the laser's work.
"""
from typing import List, Optional

from ..model import Finding
from ..stats import pct, spearman
from .recipe_change import describe

MIN_N = 200
BINS = 5
MIN_ABS_RHO = 0.10
MIN_GAIN_POINTS = 3.0
MIN_WINDOW_SHARE = 0.40        # a window nobody can hit is not a recommendation


def _success(t) -> Optional[bool]:
    if t.linearity_pass is None:
        return None
    if t.final_r_low and t.final_r_high and t.trimmed_resistance:
        return bool(t.linearity_pass and t.final_r_low <= t.trimmed_resistance <= t.final_r_high)
    return bool(t.linearity_pass)


def analyze(model: str, tracks, laser_label) -> List[Finding]:
    groups = {}
    for t in tracks:
        if t.passes and t.untrimmed_resistance and _success(t) is not None:
            groups.setdefault((t.system, t.recipe), []).append(t)
    eligible = [(k, v) for k, v in groups.items() if len(v) >= MIN_N]
    if not eligible:
        return []                                       # thin sample: say nothing
    (system, recipe), ts = max(eligible, key=lambda kv: max(t.file_date for t in kv[1]))
    ts = sorted(ts, key=lambda t: t.untrimmed_resistance)
    rs = [t.untrimmed_resistance for t in ts]
    ok = [1.0 if _success(t) else 0.0 for t in ts]
    rho = spearman(rs, ok)
    if rho is None or abs(rho) < MIN_ABS_RHO:
        return []                                       # resistance is not the lever here
    n = len(ts)
    edges = [round(i * n / BINS) for i in range(BINS + 1)]
    bins = []
    for lo, hi in zip(edges, edges[1:]):
        chunk = ts[lo:hi]
        bins.append({"r_low": chunk[0].untrimmed_resistance, "r_high": chunk[-1].untrimmed_resistance,
                     "n": len(chunk), "success_pct": pct([_success(t) for t in chunk])})
    overall = pct([bool(x) for x in ok])
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
    if best is None:
        return []
    gain = best["success_pct"] - overall
    if gain < MIN_GAIN_POINTS:
        return []
    direction = "lower" if rho < 0 else "higher"
    configured = next(((t.initial_r_low, t.initial_r_high) for t in reversed(ts)
                       if t.initial_r_low and t.initial_r_high), None)
    first, last = min(t.file_date for t in ts).date(), max(t.file_date for t in ts).date()
    return [Finding(
        model=model, analyzer="ink_target", category="Ink target",
        lever="ink", systems=(system,),
        title=f"Incoming resistance: aim {direction} -- {best['r_low']:,.0f} to {best['r_high']:,.0f} Ω does best",
        summary=(f"Within {laser_label(system)} running {describe(recipe)} ({first} to {last}, {n:,} tracks), units "
                 f"arriving between {best['r_low']:,.0f} and {best['r_high']:,.0f} Ω left the laser good "
                 f"{best['success_pct']:.0f}% of the time against {overall:.0f}% overall. "
                 + (f"The station is set to accept {configured[0]:,.0f} to {configured[1]:,.0f} Ω incoming. "
                    if configured else "These files carry no configured incoming window, so this is a computed "
                    "target, not a comparison with a setting. ")
                 + "Laser and recipe are held constant, so neither explains the difference."),
        n_units=n, strength_name="Spearman correlation, incoming resistance vs good-at-laser",
        strength_value=rho, expected_gain_points=gain,
        gain_definition=("good-at-laser rate inside the recommended window minus the rate over the whole group; "
                         "good = the app's trim linearity verdict AND final resistance inside the station's "
                         "final limits"),
        evidence={"bins": bins, "window": best, "overall_pct": overall, "recipe": describe(recipe),
                  "configured_incoming": configured, "period": [first.isoformat(), last.isoformat()]})]
