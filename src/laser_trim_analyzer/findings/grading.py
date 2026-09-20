"""Grade ANY sweep the way the app grades the final one.

`margin_ratio` is the app's linearity rule as one number: slide the whole
curve by the best single offset, then take the worst point as a fraction of
its own half-band. <= 1.0 means every graded point fits inside its per-point
limits -- the zero-tolerance rule. Points with a blank error or blank limit
are ungraded (never 0.0), which is how the stations mark ignored rows.

Checked 2026-09-20 against the app's stored verdict: 8,984 of 8,984 tracks on
the 8232-1/8340-1 slice, and 542 of 544 across 255 models in the sample base.
The engine re-checks this per model (`yardstick_fidelity`) and stays silent
where it does not hold.
"""
from typing import Optional, Sequence


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def margin_ratio(errors: Sequence, upper: Sequence, lower: Sequence,
                 min_points: int = 3) -> Optional[float]:
    pts = [(e - (u + l) / 2.0, (u - l) / 2.0)
           for e, u, l in zip(errors or (), upper or (), lower or ())
           if _num(e) and _num(u) and _num(l) and u > l]
    if len(pts) < min_points:
        return None
    lo = min(d for d, _ in pts)
    hi = max(d for d, _ in pts)

    def worst(offset: float) -> float:
        return max(abs(d - offset) / h for d, h in pts)

    for _ in range(64):                       # worst() is convex in the offset; (2/3)^64 ~ 5e-12
        a = lo + (hi - lo) / 3.0
        b = hi - (hi - lo) / 3.0
        if worst(a) < worst(b):
            hi = b
        else:
            lo = a
    return worst((lo + hi) / 2.0)


_EDGE = 1e-9      # a sweep lying exactly ON its limit passes (the app's rule is inclusive);
                  # without this the search's last few ulps decide a knife-edge case.


def in_limits(errors: Sequence, upper: Sequence, lower: Sequence) -> Optional[bool]:
    r = margin_ratio(errors, upper, lower)
    return None if r is None else r <= 1.0 + _EDGE
