"""Tap-filtering diagnostics for output smoothness.

Potentiometers with intermediate taps show a sharp step at each tap that
inflates the smoothness deviation and can cause a false failure. The parser now
reports a diagnostic tap-excluded estimate alongside the file's authoritative
value (it never overrides the file's verdict). These tests pin that behavior
against real samples:

  - 8202-1 (tapped): a unit FAILED at 0.0227 (spec 0.0083) with many tap steps;
    excluding taps must materially lower the deviation estimate.
  - 1844205 (no taps, curvature failure): excluding taps must NOT meaningfully
    change the estimate — its failure isn't tap-driven.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the package importable without relying on a conftest or pip install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laser_trim_analyzer.core.smoothness_parser import SmoothnessParser, analyze_taps

ROOT = Path(__file__).resolve().parent.parent
SMOOTH_DIR = ROOT / "Work Files/Sample_Base_2026-04-10/Smoothness_Sample_2026-04-10/Test Station"


def _largest_file(model: str) -> Path:
    files = sorted(
        glob.glob(str(SMOOTH_DIR / model / "*.xlsx")),
        key=lambda f: Path(f).stat().st_size,
        reverse=True,
    )
    if not files:
        pytest.skip(f"smoothness sample data not present for {model}")
    return Path(files[0])


def test_end_swing_excluded_middle_defect_preserved():
    """A broad swing at each end is excluded; a MIDDLE defect is kept."""
    n = 2000
    ez = int(n * 0.05)
    v = 0.0005 * np.sin(np.arange(n))                    # quiet baseline
    # Broad half-cosine swings filling each end zone (not a narrow spike).
    bump = 0.08 * np.sin(np.linspace(0, np.pi, ez))
    v[:ez] += bump                                       # swing up at start
    v[-ez:] += bump                                      # swing down at end
    v[1000] += 0.03                                      # MIDDLE defect (must survive)
    res = analyze_taps(v.tolist(), spec=0.01)
    assert res["left_swing"] and res["right_swing"]
    assert res["tap_excluded_max_deviation"] < res["recomputed_max_deviation"]
    assert res["tap_excluded_max_deviation"] >= 0.02     # middle defect preserved


def test_no_swing_means_no_filtering():
    """If there's no end swing, nothing is filtered — even a near-end defect."""
    n = 2000
    v = 0.0005 * np.sin(np.arange(n))
    v[20] += 0.05                                        # localized defect near start
    res = analyze_taps(v.tolist(), spec=0.01)
    assert res["swing_detected"] is False
    # Nothing excluded → tap-excluded equals the whole-sweep estimate.
    assert res["tap_excluded_max_deviation"] == res["recomputed_max_deviation"]


def test_tap_driven_false_failure_revealed():
    """8202 failed on the end swing: excluding the swinging end drops below spec."""
    track = SmoothnessParser().parse_file(_largest_file("8202"))["tracks"][0]
    assert track["swing_detected"] is True
    assert track["tap_excluded_max_deviation"] < track["recomputed_max_deviation"]
    assert track["tap_excluded_max_deviation"] <= track["smoothness_spec"]
    # The file's own verdict is preserved untouched (we never override it).
    assert track["smoothness_pass"] is False


def test_middle_defect_unit_not_rescued():
    """8202-1's worst point is mid-signal: end exclusion must NOT rescue it."""
    track = SmoothnessParser().parse_file(_largest_file("8202-1"))["tracks"][0]
    # Even if an end swing is excluded, the middle defect keeps it failing.
    assert track["tap_excluded_max_deviation"] > track["smoothness_spec"]


def test_curvature_failure_not_tap_driven():
    """1844205 fails throughout (tight spec), no end swing: no filtering applied."""
    track = SmoothnessParser().parse_file(_largest_file("1844205"))["tracks"][0]
    assert track["swing_detected"] is False
    assert track["tap_excluded_max_deviation"] == track["recomputed_max_deviation"]


def test_no_override_of_file_verdict():
    """Tap fields are additive diagnostics; file's max/verdict are still present."""
    track = SmoothnessParser().parse_file(_largest_file("8202"))["tracks"][0]
    for key in ("max_smoothness", "smoothness_pass", "smoothness_spec"):
        assert key in track
    for key in ("swing_detected", "tap_excluded_max_deviation", "recomputed_max_deviation"):
        assert key in track
