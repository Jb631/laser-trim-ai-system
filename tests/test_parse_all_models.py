"""Systematic per-model parse regression across all four file types.

Covers trim sweeps, final-test, and output smoothness — one representative
sample file per (type, model). Two guarantees:

1. NO REGRESSION: a file that currently parses cleanly ("ok") must keep
   parsing cleanly. A model can only get better (empty/error -> ok), never
   worse. The frozen state lives in fixtures/parse_baseline.json; regenerate it
   deliberately with `python tests/gen_parse_baseline.py` after intended changes.

2. GOLDEN VALUES: a handful of key models have their exact parsed metrics
   pinned, so any numeric drift fails loudly. This includes 1844205/1844202,
   whose pre-trim (untrimmed) sigma regressed to ~1.6 from a 2x-resolution
   measured column before the parser fix — the golden values lock that fix in.
"""
import json
from pathlib import Path

import pytest

from model_sweep import classify, discover_representatives

BASELINE_PATH = Path(__file__).resolve().parent / "fixtures" / "parse_baseline.json"
BASELINE = json.loads(BASELINE_PATH.read_text())

# ok is strictly better than empty, which is better than error.
_RANK = {"error": 0, "empty": 1, "ok": 2}

# Exact expected metrics for key models (the golden-value regression set).
# untr_sigma is the pre-trim sigma; the 1844 pair must stay ~0.1, NOT ~1.6.
GOLDEN = {
    "1844205": {"untr_sigma": 0.097321, "untr_n": 32, "post_sigma": 0.003675, "n_points": 63},
    "1844202": {"untr_sigma": 0.093948, "untr_n": 32, "post_sigma": 0.008324, "n_points": 63},
    "1844204": {"untr_sigma": 0.004923, "untr_n": 63, "post_sigma": 0.004947, "n_points": 63},
    "8340-1":  {"untr_sigma": 0.03067,  "untr_n": 63, "post_sigma": 0.007185, "n_points": 63},
    "8232-1":  {"untr_sigma": 0.065373, "untr_n": 56, "post_sigma": 0.025829, "n_points": 56},
}

# Baseline-only bookkeeping keys: `status` is compared by RANK above (a model may get better),
# `reason` is free text, `type`/`model` are how the entry is looked up, not parse output.
_NOT_VALUES = {"status", "reason", "type", "model"}


def _moved_fields(base: dict, now: dict) -> list:
    """[(key, baseline value, value now)] for every frozen value that no longer matches.
    Floats to 1e-6 (absolute and relative): tight enough that a moved sigma cannot hide,
    loose enough for another platform's BLAS."""
    import math
    moved = []
    for key, was in base.items():
        if key in _NOT_VALUES:
            continue
        is_now = now.get(key)
        if isinstance(was, float) or isinstance(is_now, float):
            same = (was is not None and is_now is not None
                    and not isinstance(was, bool) and not isinstance(is_now, bool)
                    and math.isclose(was, is_now, rel_tol=1e-6, abs_tol=1e-6))
            same = same or (was is None and is_now is None)
        else:
            same = was == is_now
        if not same:
            moved.append((key, was, is_now))
    return moved


# Current representatives, keyed by relpath for lookup against the baseline.
_CURRENT = {rel: (ftype, model) for ftype, model, rel in discover_representatives()}


@pytest.mark.parametrize("relpath", sorted(BASELINE.keys()))
def test_no_parse_regression(relpath):
    """Every model that parsed cleanly before must still parse at least as well."""
    base = BASELINE[relpath]
    if relpath not in _CURRENT:
        pytest.skip(f"sample file no longer present: {relpath}")
    ftype = base["type"]
    now = classify(ftype, relpath)
    base_rank = _RANK[base["status"]]
    now_rank = _RANK[now["status"]]
    assert now_rank >= base_rank, (
        f"{base['model']} ({ftype}) regressed: was '{base['status']}', "
        f"now '{now['status']}' ({now.get('reason', '')}) for {relpath}"
    )
    # ... and every VALUE the baseline froze must still come out the same.
    #
    # Until 2026-09-20 this test stopped at the rank above, and was described (by its own
    # readers, in two documents) as "diffs 645 real files". It did not: only the five GOLDEN
    # models had numbers compared. A parser change that moved a sigma, a point count or a
    # final-test verdict on any of the other 640 files stayed green -- found when a fix rerouted
    # a whole class of files and this gate never blinked. An audit that day compared every
    # stored field: 633 of 641 present files were identical and the other 8 were explained,
    # deliberate changes, so the comparison was switched on and those 8 entries refreshed.
    if base["status"] == "ok" and now["status"] == "ok":
        moved = _moved_fields(base, now)
        assert not moved, (
            f"{base['model']} ({ftype}) parses to different VALUES than the baseline for {relpath}: "
            + "; ".join(f"{k}: {b!r} -> {n!r}" for k, b, n in moved)
            + ". If the change is intended, refresh THIS entry deliberately "
              "(tests/gen_parse_baseline.py) and say why in the commit."
        )


@pytest.mark.parametrize("model", sorted(GOLDEN.keys()))
def test_golden_values(model):
    """Key models must reproduce their exact pinned parsed metrics."""
    rel = next((r for r, (t, m) in _CURRENT.items() if m == model and t == "trim"), None)
    if rel is None:
        pytest.skip(f"sample data not present for golden model {model}")
    res = classify("trim", rel)
    assert res["status"] == "ok", f"{model} did not parse cleanly: {res}"
    for key, expected in GOLDEN[model].items():
        actual = res.get(key)
        if isinstance(expected, float):
            assert actual == pytest.approx(expected, abs=1e-4), (
                f"{model}.{key}: expected {expected}, got {actual}"
            )
        else:
            assert actual == expected, f"{model}.{key}: expected {expected}, got {actual}"


def test_pretrim_resolution_fix_locked():
    """Regression lock: the 2x-resolution pre-trim bug stays fixed.

    Before the fix, 1844205/1844202 pre-trim sigma was ~1.6 (measured column
    recorded at double the position/theory grid resolution, subtracted
    row-by-row). It must stay well under 0.5.
    """
    for model in ("1844205", "1844202"):
        rel = next((r for r, (t, m) in _CURRENT.items() if m == model and t == "trim"), None)
        if rel is None:
            pytest.skip(f"sample data not present for {model}")
        res = classify("trim", rel)
        assert res["status"] == "ok"
        assert res["untr_sigma"] is not None and res["untr_sigma"] < 0.5, (
            f"{model} pre-trim sigma {res['untr_sigma']} looks like the "
            f"2x-resolution bug (was ~1.6) — the fix regressed"
        )


def test_baseline_covers_every_model():
    """Sanity: the baseline isn't accidentally empty or shrunk to a stub."""
    assert len(BASELINE) > 500, f"baseline only has {len(BASELINE)} entries"
