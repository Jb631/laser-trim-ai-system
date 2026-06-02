"""Regression tests for the 2026-06-01 whole-app audit fixes.

Plan: docs/AUDIT_FIX_PLAN_2026-06-01.md
"""
import sys
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# Batch A / H1 -- DB yield denominators exclude UNTRIMMED.
# ---------------------------------------------------------------------------


def _seed_mixed(db):
    """3 PASS, 1 FAIL, 2 UNTRIMMED (one model 'M'), each with one track."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import (
        AnalysisResult as AR, TrackResult as TR, SystemType, StatusType,
    )
    now = datetime.now()
    rows = [
        ("p1", StatusType.PASS, True, True, StatusType.PASS),
        ("p2", StatusType.PASS, True, True, StatusType.PASS),
        ("p3", StatusType.PASS, True, True, StatusType.PASS),
        ("f1", StatusType.FAIL, False, False, StatusType.FAIL),
        ("u1", StatusType.UNTRIMMED, None, None, StatusType.UNTRIMMED),
        ("u2", StatusType.UNTRIMMED, None, None, StatusType.UNTRIMMED),
    ]
    with db.session() as s:
        for serial, status, sig, lin, tstatus in rows:
            ar = AR(filename=f"{serial}.xls", file_path=f"/f/{serial}", file_hash=serial,
                    model="M", serial=serial, system=SystemType.A, file_date=now,
                    timestamp=now, overall_status=status, has_multi_tracks=False,
                    processing_time=0.1)
            s.add(ar); s.flush()
            s.add(TR(analysis_id=ar.id, track_id="T1", status=tstatus,
                     sigma_pass=sig, linearity_pass=lin))
        s.commit()


def test_get_overall_stats_excludes_untrimmed(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    db = DatabaseManager(tmp_path / "overall.db")
    _seed_mixed(db)
    stats = db.get_overall_stats()
    # 3 pass over 4 GRADEABLE files = 75% (NOT 3/6 = 50%).
    assert stats["trimmed_total"] == 4
    assert round(stats["pass_rate"], 1) == 75.0
    # Track rates over the 4 trimmed tracks (UNTRIMMED tracks excluded).
    assert stats["total_tracks"] == 4
    assert round(stats["sigma_pass_rate"], 1) == 75.0
    assert round(stats["linearity_pass_rate"], 1) == 75.0


def test_get_model_stats_excludes_untrimmed(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    db = DatabaseManager(tmp_path / "modelstats.db")
    _seed_mixed(db)
    row = next(r for r in db.get_model_stats() if r["model"] == "M")
    assert row["count"] == 6            # total throughput keeps every file
    assert row["trimmed_count"] == 4    # gradeable denominator excludes UNTRIMMED
    assert row["passed"] == 3
    assert row["failed"] == 1           # NOT 3 (was count-passed = 6-3)
    assert round(row["pass_rate"], 1) == 75.0


# ---------------------------------------------------------------------------
# Batch B / H3 -- linearity Cpk/Ppk is one-sided upper (max-abs-error >= 0),
# not a symmetric two-sided spec.
# ---------------------------------------------------------------------------


def test_linearity_cpk_is_one_sided_upper():
    from laser_trim_analyzer.core.cpk import calculate_cpk

    devs = [0.10, 0.12, 0.18, 0.09, 0.15, 0.20, 0.11, 0.13, 0.17, 0.14]
    r = calculate_cpk(devs, spec_limit_pct=0.5, subgroup_size=1)

    assert r.lsl == 0.0           # one-sided floor, NOT -0.5
    assert r.usl == 0.5
    # Cpk == Cpu computed from the result's own mean/within-sigma.
    expected_cpu = (r.usl - r.mean) / (3 * r.std_within)
    assert r.cpk == pytest.approx(expected_cpu)
    assert r.cp == pytest.approx(expected_cpu)   # one-sided: Cp == Cpu
    assert r.ppk == pytest.approx((r.usl - r.mean) / (3 * r.std_overall))
    # We are NOT reporting the old inflated two-sided Cp = (usl-(-spec))/(6*std).
    two_sided_cp = (r.usl - (-0.5)) / (6 * r.std_within)
    assert r.cp < two_sided_cp


# ---------------------------------------------------------------------------
# Batch C / C2 -- generic smoothness uses deviation MAGNITUDE, so a worst
# NEGATIVE excursion beyond spec cannot pass a failing unit.
# ---------------------------------------------------------------------------


def test_smoothness_generic_uses_magnitude_not_signed_max(tmp_path):
    import pandas as pd
    from laser_trim_analyzer.core.smoothness_parser import SmoothnessParser

    path = tmp_path / "smooth_generic.xlsx"
    rows = [
        ["Position", "Smoothness", "Spec"],
        [0.00, 0.01, 0.05],
        [0.25, -0.08, 0.05],   # worst excursion is negative, magnitude 0.08 > 0.05
        [0.50, 0.02, 0.05],
        [0.75, -0.03, 0.05],
        [1.00, 0.01, 0.05],
    ]
    pd.DataFrame(rows).to_excel(path, index=False, header=False, sheet_name="Test Data")

    tracks = SmoothnessParser().parse_file(path).get("tracks", [])
    assert tracks, "expected at least one parsed track"
    t = tracks[0]
    # Magnitude, not signed peak: 0.08 (abs of -0.08), NOT 0.02.
    assert round(t["max_smoothness"], 3) == 0.08
    # 0.08 > 0.05 spec must never read as a pass (signed max gave 0.02 -> false pass).
    assert t["smoothness_pass"] in (False, None)
    assert t["smoothness_pass"] is not True


# ---------------------------------------------------------------------------
# Batch D / H4 -- incremental skip confirms by CONTENT hash, so a re-export of
# new content to a reused filename is processed, not silently skipped.
# ---------------------------------------------------------------------------


def test_incremental_skip_confirms_by_content_hash(tmp_path):
    import os
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.utils.hashing import calculate_file_hash

    f = tmp_path / "test.xls"
    f.write_text("original content")
    os.utime(f, (1000, 1000))

    p = Processor(use_ml=False)
    p._processed_filenames = {str(f)}
    p._processed_hashes = {calculate_file_hash(f)}

    # Same path + same content -> already processed (skip).
    assert p._is_processed(f) is True

    # NEW content re-exported to the SAME path -> must NOT be skipped.
    f.write_text("new re-trimmed content -- different bytes entirely")
    os.utime(f, (2000, 2000))  # distinct mtime so the hash cache recomputes
    assert p._is_processed(f) is False

    # A brand-new path we've never seen -> not processed (cheap early-out).
    g = tmp_path / "brand_new.xls"
    g.write_text("x")
    assert p._is_processed(g) is False
