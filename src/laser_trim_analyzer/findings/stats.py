"""Small, dependency-free statistics the analyzers share."""
import math
from collections import Counter
from statistics import mean
from typing import Any, List, Optional, Sequence, Tuple


def pct(flags: Sequence[bool]) -> Optional[float]:
    flags = list(flags)
    return (100.0 * sum(1 for f in flags if f) / len(flags)) if flags else None


def plausible_resistance(r: Any) -> bool:
    """Is `r` a plausible ohms reading, not the tester's open-circuit rail?

    The work database holds 1e12-ohm (and other 1e9+) readings from the open-circuit rail
    (CLAUDE.md, "What must never be stored"; `core/model_stats.py`'s own docstring names the same
    corruption). `core/model_stats.py` has a richer plausibility system (`_plausible`/`_band`), but
    it is model-median-relative and needs the whole population computed first -- a heavier tool
    than a single-value guard on one side of one comparison needs, and its `_plausible` is private.
    This is the ONE definition of the simple bound (`0 < r < 1e9`, originally `loss_origin`'s
    `_score_resistance`), shared by every analyzer that filters `untrimmed_resistance` before
    taking a median or scoring an AUC over it -- lifted out on fix round 1, 2026-09-25
    (task-6-review.md, Minor #3), after `recipe_change`'s and `setup_change`'s own `_side()`
    helpers were found using a bare truthy filter that let a 1e12 reading through.
    """
    return isinstance(r, (int, float)) and not isinstance(r, bool) and 0 < r < 1e9


def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0       # ties share the average rank
        i = j + 1
    return ranks


def auc(fails: Sequence[float], passes: Sequence[float]) -> Optional[float]:
    """The probability a value from the first group is greater than one from the second (a tie
    counts half) -- the Mann-Whitney U statistic scaled to [0, 1]. None when either side is empty:
    there is no separation to report without at least one example of each.

    Lives here, not in loss_origin where it was written, since rework_load's rank test builds its
    U from it too (fix round 2, 2026-09-25) -- one rank computation, never a second copy."""
    if not fails or not passes:
        return None
    n1, n2 = len(fails), len(passes)
    ranks = _ranks(list(fails) + list(passes))
    r1 = sum(ranks[:n1])                      # the first group occupies the first n1 slots
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n2)


def mann_whitney_lower(lower: Sequence[float],
                       higher: Sequence[float]) -> Optional[Tuple[float, Optional[float]]]:
    """(U, one-sided p) for the claim that values in `lower` tend to be LOWER than in `higher`.

    U is the first group's Mann-Whitney statistic -- the number of (lower, higher) pairs with the
    first value greater, a tie counting half -- built from `auc` (U = AUC x n1 x n2), so the ranks
    are computed once. p is the normal approximation with the tie correction:
        var(U) = n1 n2 / 12 * [(N + 1) - sum(t^3 - t) / (N (N - 1))],  z = (U - n1 n2 / 2) / sd,
    p = Phi(z) -- small when U is small, i.e. when the first group sits low. No continuity
    correction (the ruling names the tie correction only; at the >= 20 per group the analyzers
    require, the two differ in the third significant figure). Matches
    scipy.stats.mannwhitneyu(..., alternative="less", method="asymptotic", use_continuity=False).

    None when either group is empty; p is None when every value is tied (no variation, no test).
    """
    a = auc(lower, higher)
    if a is None:
        return None
    n1, n2 = len(lower), len(higher)
    u = a * n1 * n2
    n = n1 + n2
    ties = sum(t ** 3 - t for t in Counter(list(lower) + list(higher)).values())
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return u, None
    z = (u - n1 * n2 / 2.0) / math.sqrt(var)
    return u, 0.5 * math.erfc(-z / math.sqrt(2.0))


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = mean(rx), mean(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None                                   # no variation: no relationship to report
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sxx * syy) ** 0.5
