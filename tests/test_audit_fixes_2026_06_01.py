"""Regression tests for the 2026-06-01 whole-app audit fixes.

Plan: docs/AUDIT_FIX_PLAN_2026-06-01.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Batch A / H2 -- UNTRIMMED is its own bucket and is excluded from the yield
# denominator (was previously dropped, diluting pass_rate and making counts
# fail to sum to `processed`).
# ---------------------------------------------------------------------------


def test_batch_summary_buckets_untrimmed_separately():
    from laser_trim_analyzer.core.models import AnalysisStatus, BatchSummary

    s = BatchSummary(total_files=10, processed=10)
    statuses = (
        [AnalysisStatus.PASS] * 5
        + [AnalysisStatus.WARNING] * 1
        + [AnalysisStatus.FAIL] * 2
        + [AnalysisStatus.UNTRIMMED] * 2
    )
    for st in statuses:
        s.record_status(st)

    assert s.passed == 5
    assert s.warnings == 1
    assert s.failed == 2
    assert s.untrimmed == 2
    assert s.errors == 0  # UNTRIMMED must NOT land in the error bucket
    # Every processed file is now accounted for in exactly one bucket.
    assert s.passed + s.warnings + s.failed + s.untrimmed + s.errors == s.processed


def test_batch_summary_untrimmed_not_an_error():
    from laser_trim_analyzer.core.models import AnalysisStatus, BatchSummary

    s = BatchSummary()
    s.record_status(AnalysisStatus.UNTRIMMED)
    assert s.untrimmed == 1
    assert s.errors == 0


def test_gradeable_count_excludes_untrimmed():
    from laser_trim_analyzer.core.models import BatchSummary

    s = BatchSummary(processed=10, untrimmed=2)
    assert s.gradeable_count == 8
    # Never negative even if bookkeeping is odd.
    assert BatchSummary(processed=0, untrimmed=5).gradeable_count == 0


def test_yield_denominator_uses_gradeable_not_processed():
    """5 pass / 1 warn / 2 fail / 2 untrimmed -> 62.5% over the gradeable 8,
    NOT 50% over the processed 10."""
    from laser_trim_analyzer.core.models import AnalysisStatus, BatchSummary

    s = BatchSummary(processed=10)
    for st in ([AnalysisStatus.PASS] * 5 + [AnalysisStatus.WARNING]
               + [AnalysisStatus.FAIL] * 2 + [AnalysisStatus.UNTRIMMED] * 2):
        s.record_status(st)
    pass_rate = (s.passed / s.gradeable_count) * 100
    assert round(pass_rate, 1) == 62.5
