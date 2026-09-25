"""Small, dependency-free statistics the analyzers share."""
from statistics import mean
from typing import Any, List, Optional, Sequence


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
