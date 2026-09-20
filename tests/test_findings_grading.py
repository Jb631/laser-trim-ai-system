import pytest

from laser_trim_analyzer.findings.grading import margin_ratio, in_limits
from laser_trim_analyzer.findings.stats import spearman, pct

BAND = 0.10
N_POINTS = 12


def sweep(worst: float):
    """A sweep whose best-offset worst point is `worst` x the band (1.0 = exactly at the limit)."""
    half = worst * BAND
    errors = tuple(half if i % 2 else -half for i in range(N_POINTS))
    return errors, tuple([BAND] * N_POINTS), tuple([-BAND] * N_POINTS)


def test_margin_ratio_is_one_exactly_at_the_limit():
    e, u, l = sweep(1.0)
    assert margin_ratio(e, u, l) == pytest.approx(1.0, abs=1e-6)
    assert in_limits(e, u, l) is True

def test_a_constant_offset_is_free():
    e, u, l = sweep(0.5)
    shifted = tuple(x + 5 * BAND for x in e)             # far outside the band before the offset
    assert margin_ratio(shifted, u, l) == pytest.approx(0.5, abs=1e-6)

def test_a_bowtie_waist_is_graded_point_by_point():
    # Two neighbouring points in the narrow waist that need OPPOSITE offsets: no single
    # offset fits both, however wide the band is elsewhere. (One such point alone can be
    # rescued by an offset -- the first draft of this test got that wrong.)
    e = [0.0] * 12; u = [0.10] * 12; l = [-0.10] * 12
    u[5] = u[6] = 0.01; l[5] = l[6] = -0.01
    e[5], e[6] = 0.03, -0.03
    assert in_limits(e, u, l) is False
    e[5], e[6] = 0.005, -0.005
    assert in_limits(e, u, l) is True

def test_blank_cells_are_ungraded_never_zero():
    e, u, l = map(list, sweep(0.5))
    e[3] = None; u[4] = None                              # ignored rows
    assert margin_ratio(e, u, l) == pytest.approx(0.5, abs=1e-6)
    assert margin_ratio([None] * 12, u, l) is None        # nothing measured: no grade, not a pass

def test_booleans_are_not_measurements():
    assert margin_ratio([True] * 12, [0.1] * 12, [-0.1] * 12) is None


def test_spearman_basics():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4], [1, 1, 1, 1]) is None   # no variation: nothing to report
    assert pct([]) is None and pct([True, False]) == 50.0


def test_arrays_that_do_not_line_up_are_ungradeable_not_quietly_truncated():
    """zip() stops at the shortest input. A limit array shorter than its error array would drop the
    tail of the sweep without a word -- and the tail is where a bowtie band is widest and the verdict
    is usually decided. 28,887 real sweeps were checked 2026-09-20: every one lines up, so a
    mismatch is corrupt data, and corrupt data is ungradeable, never "graded on what was left"."""
    from laser_trim_analyzer.findings.grading import in_limits, margin_ratio
    e, u, l = [0.0, 0.0, 0.0, 0.0, 9.9], [1.0] * 4, [-1.0] * 4      # the out-of-limits point has no limit
    assert margin_ratio(e, u, l) is None and in_limits(e, u, l) is None
    assert margin_ratio([0.0] * 4, [1.0] * 5, [-1.0] * 5) is None
    assert margin_ratio([0.0] * 4, [1.0] * 4, [-1.0] * 4) == 0.0          # aligned still grades
