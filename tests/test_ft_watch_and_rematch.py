"""Final-test watch + FT→trim rematch (2026-07-13).

Covers the two additions that point the lot detector at the last station:
  * ft_fail_fraction — FT lot fail rate trains/advances like any lot metric
  * escape_fraction — only CONFIDENT links to ACCEPTED trims count
and the matcher work the escape metric depends on:
  * rematch_unlinked_final_tests links FT records saved before their trims
  * the 180-day window and its confidence decay
  * the "7953-1A" glued-letter model-variant normalization
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _add_trim(s, model, serial, when, status, i, DBAR, SystemType):
    s.add(DBAR(filename=f"{model}-{serial}-{i}.xls", file_path=f"/t/{model}/{i}",
               file_hash=f"{model}{serial}{i}".ljust(64, "0"), model=model,
               serial=serial, system=SystemType.A, file_date=when,
               timestamp=when, overall_status=status,
               has_multi_tracks=False, processing_time=0.1))


def _add_ft(s, model, serial, when, status, i, DBFT, linked=None, conf=None):
    s.add(DBFT(filename=f"FT-{model}-{serial}-{i}.xls", model=model,
               serial=serial, test_date=when, file_date=when,
               timestamp=when, overall_status=status,
               linked_trim_id=linked, match_confidence=conf))


def test_ft_fail_fraction_trains_on_ft_lots(tmp_path):
    """A model with clean historical FT lots and a 40%-fail newest closed lot
    trains on FT data and the last closed lot's fraction reads 0.4."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        FinalTestResult as DBFT, ModelMetricState, StatusType)
    from laser_trim_analyzer.ml.drift_training import train_drift_detector
    from laser_trim_analyzer.ml.lots import get_model_lots

    db = DatabaseManager(tmp_path / "ftw.db")
    t0 = datetime(2025, 1, 6)
    with db.session() as s:
        i = 0
        for lot in range(14):
            lot_day = t0 + timedelta(weeks=lot)
            fails = 4 if lot == 13 else 0
            for u in range(10):
                st_ = StatusType.FAIL if u < fails else StatusType.PASS
                _add_ft(s, "FTW", f"s{i}", lot_day + timedelta(hours=u), st_, i, DBFT)
                i += 1
        s.commit()
    train_drift_detector(db, model="FTW")
    with db.session() as s:
        ms = s.query(ModelMetricState).filter_by(
            model="FTW", metric="ft_fail_fraction").first()
        assert ms is not None and ms.is_trained
        assert ms.baseline_mean is not None and ms.baseline_mean < 0.05
        assert ms.last_row_id is None            # lot-mode row
    lots = [l for l in get_model_lots(db, "FTW", "ft_fail_fraction")
            if not l.is_open()]
    assert abs(lots[-1].median - 0.4) < 1e-9     # mean-aggregated fraction


def test_ft_fail_fraction_ignores_pre2000_garbage_dates(tmp_path):
    """1899-12-30 epoch defaults in FT files must not create phantom lots."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import FinalTestResult as DBFT, StatusType
    from laser_trim_analyzer.ml.lots import get_model_lots

    db = DatabaseManager(tmp_path / "old.db")
    with db.session() as s:
        _add_ft(s, "OLD", "1", datetime(1899, 12, 30), StatusType.FAIL, 0, DBFT)
        _add_ft(s, "OLD", "2", datetime(2025, 3, 2), StatusType.PASS, 1, DBFT)
        s.commit()
    lots = get_model_lots(db, "OLD", "ft_fail_fraction")
    assert len(lots) == 1 and lots[0].n == 1     # only the 2025 record


def test_escape_fraction_requires_confident_link_and_accepted_trim(tmp_path):
    """Escapes = trim ACCEPTED (PASS/WARNING) then FT FAIL, link conf ≥ 0.5.
    Low-confidence links and trim-FAILED units must not count."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, FinalTestResult as DBFT, StatusType, SystemType)
    from laser_trim_analyzer.ml.drift_training import _load_samples_with_dates

    db = DatabaseManager(tmp_path / "esc.db")
    t0 = datetime(2025, 4, 6)
    with db.session() as s:
        for i, st_ in enumerate([StatusType.PASS, StatusType.PASS,
                                 StatusType.WARNING, StatusType.FAIL]):
            _add_trim(s, "ESC", f"s{i}", t0 - timedelta(days=5), st_, i,
                      DBAR, SystemType)
        s.commit()
    with db.session() as s:
        ids = {r.serial: r.id for r in s.query(DBAR).filter_by(model="ESC")}
    with db.session() as s:
        # confident link, trim PASS, FT FAIL -> escape (1.0)
        _add_ft(s, "ESC", "s0", t0, StatusType.FAIL, 0, DBFT, ids["s0"], 0.9)
        # confident link, trim PASS, FT PASS -> 0.0
        _add_ft(s, "ESC", "s1", t0, StatusType.PASS, 1, DBFT, ids["s1"], 0.9)
        # confident link, trim WARNING (accepted), FT FAIL -> escape (1.0)
        _add_ft(s, "ESC", "s2", t0, StatusType.FAIL, 2, DBFT, ids["s2"], 0.8)
        # trim FAILED -> not an escape candidate at all
        _add_ft(s, "ESC", "s3", t0, StatusType.FAIL, 3, DBFT, ids["s3"], 0.9)
        # LOW-confidence link -> excluded
        _add_ft(s, "ESC", "s0", t0, StatusType.FAIL, 4, DBFT, ids["s0"], 0.3)
        # unlinked -> excluded
        _add_ft(s, "ESC", "s1", t0, StatusType.FAIL, 5, DBFT, None, None)
        s.commit()
    samples = _load_samples_with_dates(db, "ESC", "escape_fraction")
    vals = sorted(v for _d, v, _r in samples)
    assert vals == [0.0, 1.0, 1.0]


def test_rematch_links_ft_saved_before_trim(tmp_path):
    """The batch-ordering hole: FT processed before its trim stays NULL at
    save time; rematch_unlinked_final_tests closes it."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, FinalTestResult as DBFT, StatusType, SystemType)

    db = DatabaseManager(tmp_path / "rm.db")
    ftd = datetime(2025, 5, 10)
    with db.session() as s:
        _add_ft(s, "8340", "17", ftd, StatusType.PASS, 0, DBFT)          # unlinked
        _add_trim(s, "8340", "17", ftd - timedelta(days=5), StatusType.PASS,
                  0, DBAR, SystemType)
        # same-serial trim >180d out: must NOT be chosen over the 5-day one
        _add_trim(s, "8340", "17", ftd - timedelta(days=400), StatusType.PASS,
                  1, DBAR, SystemType)
        s.commit()
    stats = db.rematch_unlinked_final_tests()
    assert stats["new_matches"] == 1
    with db.session() as s:
        ft = s.query(DBFT).filter_by(model="8340").first()
        assert ft.linked_trim_id is not None
        assert ft.match_method == "exact"
        assert ft.days_since_trim == 5
        assert ft.match_confidence > 0.9


def test_rematch_never_links_trim_after_ft(tmp_path):
    """Domain rule (James, 2026-07-13): trim ALWAYS precedes final test.
    An FT record whose only same-serial trim is dated AFTER it must stay
    unmatched — a later trim is a different build or a bad date, never
    this unit's trim."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, FinalTestResult as DBFT, StatusType, SystemType)

    db = DatabaseManager(tmp_path / "order.db")
    ftd = datetime(2025, 5, 10)
    with db.session() as s:
        _add_ft(s, "8340", "17", ftd, StatusType.PASS, 0, DBFT)
        _add_trim(s, "8340", "17", ftd + timedelta(days=2), StatusType.PASS,
                  0, DBAR, SystemType)
        s.commit()
    stats = db.rematch_unlinked_final_tests()
    assert stats["new_matches"] == 0 and stats["still_unmatched"] == 1
    with db.session() as s:
        assert s.query(DBFT).first().linked_trim_id is None


def test_rematch_respects_180_day_window(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, FinalTestResult as DBFT, StatusType, SystemType)

    db = DatabaseManager(tmp_path / "win.db")
    ftd = datetime(2025, 5, 10)
    with db.session() as s:
        _add_ft(s, "2511", "9", ftd, StatusType.PASS, 0, DBFT)
        _add_trim(s, "2511", "9", ftd - timedelta(days=200), StatusType.PASS,
                  0, DBAR, SystemType)
        s.commit()
    stats = db.rematch_unlinked_final_tests()
    assert stats["new_matches"] == 0 and stats["still_unmatched"] == 1


def test_rematch_never_links_sibling_variants(tmp_path):
    """Code-review BLOCKER (2026-07-13): FT '7953-1A' must NOT link to a
    '7953-1B' trim — sibling variants share a base but are different
    products. Variant matching requires one side to BE the base form,
    exactly like _find_matching_trim."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, FinalTestResult as DBFT, StatusType, SystemType)

    db = DatabaseManager(tmp_path / "sib.db")
    ftd = datetime(2025, 6, 1)
    with db.session() as s:
        _add_ft(s, "7953-1A", "42", ftd, StatusType.FAIL, 0, DBFT)
        _add_trim(s, "7953-1B", "42", ftd - timedelta(days=10), StatusType.PASS,
                  0, DBAR, SystemType)
        s.commit()
    stats = db.rematch_unlinked_final_tests()
    assert stats["new_matches"] == 0
    with db.session() as s:
        assert s.query(DBFT).first().linked_trim_id is None


def test_rematch_aggressive_gate_matches_save_time_semantics(tmp_path):
    """Code-review finding #2: the aggressive serial stage only runs when the
    FT serial itself changes under aggressive normalization — FT serial '123'
    must not link to trim serial '123X' (the save-time matcher never does)."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, FinalTestResult as DBFT, StatusType, SystemType)

    db = DatabaseManager(tmp_path / "aggr.db")
    ftd = datetime(2025, 6, 1)
    with db.session() as s:
        _add_ft(s, "8340", "123", ftd, StatusType.PASS, 0, DBFT)
        _add_trim(s, "8340", "123X", ftd - timedelta(days=5), StatusType.PASS,
                  0, DBAR, SystemType)
        s.commit()
    stats = db.rematch_unlinked_final_tests()
    assert stats["new_matches"] == 0
    # And the direction that SHOULD work still does: FT '123X' → trim '123'.
    with db.session() as s:
        _add_ft(s, "8340", "123X", ftd, StatusType.PASS, 1, DBFT)
        _add_trim(s, "8340", "123", ftd - timedelta(days=5), StatusType.PASS,
                  1, DBAR, SystemType)
        s.commit()
    stats = db.rematch_unlinked_final_tests()
    assert stats["new_matches"] >= 1


def test_rematch_glued_letter_model_variant(tmp_path):
    """FT files say '7953-1', trim files say '7953-1A' — 197 recent records
    were unlinkable before the glued-letter normalization."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, FinalTestResult as DBFT, StatusType, SystemType)

    db = DatabaseManager(tmp_path / "var.db")
    ftd = datetime(2025, 6, 1)
    with db.session() as s:
        _add_ft(s, "7953-1", "42", ftd, StatusType.FAIL, 0, DBFT)
        _add_trim(s, "7953-1A", "42", ftd - timedelta(days=10), StatusType.PASS,
                  0, DBAR, SystemType)
        s.commit()
    stats = db.rematch_unlinked_final_tests()
    assert stats["new_matches"] == 1
    with db.session() as s:
        ft = s.query(DBFT).filter_by(model="7953-1").first()
        assert ft.match_method == "model_variant"


def test_confidence_decay_spans_full_window():
    """0.002/day beyond 30d: a 100-day link must stay distinguishable from a
    175-day one (the old 0.007/day floored both at 0.40 by day ~73)."""
    from laser_trim_analyzer.database.manager import DatabaseManager

    c = DatabaseManager._calculate_match_confidence
    assert c(7) > c(30) > c(100) > c(175) >= 0.40
    assert abs(c(100) - (0.7 - 70 * 0.002)) < 1e-9


def test_normalize_model_glued_letter():
    from laser_trim_analyzer.database.manager import DatabaseManager
    n = DatabaseManager._normalize_model
    assert n("7953-1A") == "7953-1"
    assert n("7953-1B") == "7953-1"
    assert n("8275A") == "8275"
    assert n("8508-A") == "8508"
    assert n("8340-1") == "8340-1"      # numeric configs stay distinct
    assert n("2475-08") == "2475-8"
