"""Evidence pack v2 (2026-07-06): full unit history + monthly summary.

James' workflow: analyze units on screen, export the MODEL to Excel to read
its full history and judge process direction, unit charts for the team. The
old pack exported a 365-day window with 7 columns and labeled the alert-scaled
magnitude 'Delta_sigma' (contradicting the UI's honest σ-shift).
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _seed_units(db, model, n=8, start=datetime(2024, 1, 15)):
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    with db.session() as s:
        for i in range(n):
            when = start + timedelta(days=40 * i)  # spans multiple months/years
            ar = DBAR(filename=f"{model}-{i}.xls", file_path=f"/f/{model}/{i}",
                      file_hash=f"{model}h{i}".ljust(64, "0"), model=model,
                      serial=f"sn{i}", system=SystemType.A, file_date=when,
                      timestamp=when,
                      overall_status=StatusType.PASS if i % 4 else StatusType.FAIL,
                      has_multi_tracks=False, processing_time=0.1)
            s.add(ar); s.flush()
            s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                       sigma_gradient=0.01 + i * 0.001,
                       untrimmed_sigma_gradient=0.02 + i * 0.001,
                       untrimmed_resistance=1000 + i,
                       resistance_change_percent=1.5,
                       measured_electrical_angle=340.0,
                       trim_pass_count=1 + (i % 2),
                       sigma_pass=True, linearity_pass=bool(i % 4)))
        s.commit()


def test_export_full_history_and_monthly(tmp_path):
    import pandas as pd
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.export.evidence import export_evidence_pack

    db = DatabaseManager(tmp_path / "e.db")
    _seed_units(db, "EXP-1", n=8)

    out = export_evidence_pack(db, "EXP-1", tmp_path / "pack.xlsx")
    sheets = pd.read_excel(out, sheet_name=None)

    # Stable workbook shape: all five sheets ALWAYS exist (empty = headers
    # only) so "no records" can't be mistaken for a broken export.
    assert set(sheets) == {"Drift evidence", "Unit history", "Monthly summary",
                           "Final test units", "Smoothness"}

    units = sheets["Unit history"]
    # FULL record: oldest row is from 2024 — the old 365-day window would drop it.
    assert len(units) == 8
    for col in ("Serial", "Date", "Status", "System", "Track", "Sigma gradient",
                "Trimmed resistance", "Sigma threshold", "Linearity spec",
                "Fail points", "Optimal offset", "Trim improvement %", "Filename",
                "Untrimmed sigma", "Untrimmed resistance", "Resistance change %",
                "Trim passes", "Sigma pass", "Linearity pass",
                "FT result", "FT date", "FT match %"):
        assert col in units.columns, f"missing column {col}"

    monthly = sheets["Monthly summary"]
    assert len(monthly) >= 6  # 8 units spaced 40 days apart span many months
    # Customer basis (linearity yield) AND internal clean-pass, both labeled.
    assert "Linearity yield %" in monthly.columns
    assert "Clean pass %" in monthly.columns
    assert "Mean sigma gradient" in monthly.columns
    # Unit counts across months must total the seeded units.
    assert int(monthly["Units"].sum()) == 8


def test_summary_text_uses_honest_sigma_shift(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.export.evidence import build_summary_text, compute_recent_means
    from laser_trim_analyzer.ml.drift_training import train_drift_detector
    from laser_trim_analyzer.ml.manager import get_model_drift_status
    from test_drift_live_advance import _seed

    db = DatabaseManager(tmp_path / "t.db")
    _seed(db, "TXT", [0.010 + (i % 3) * 0.0002 for i in range(60)])
    train_drift_detector(db, sensitivity_preset="standard")

    status = get_model_drift_status(db, "TXT")
    text = build_summary_text("TXT", status, recent_means=compute_recent_means(db, "TXT"))
    assert "shift" in text  # honest σ-shift wording, not the alert-scaled 'Δ'
    assert "Drift summary — model TXT" in text
