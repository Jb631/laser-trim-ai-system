"""Rework load: laser-FAIL -> final-test-PASS pairs, confirmed as hand trim, never overkill.

Every silence test here was made to FAIL first by relaxing the control it names -- a green test
that cannot go red proves nothing. The unit-day disposition rule itself (per track, the day's
last attempt; every track must pass) is already pinned exhaustively against
`DatabaseManager.get_model_trim_ft_agreement` elsewhere (`test_retrim_disposition.py`,
`scripts/app_qa_sweep.py`'s own independent re-derivation); these tests use simple,
one-track-per-day fixtures (no `unit_id` set, so `_linked_pairs`' unit-day CTE falls back to the
linked file's own verdict, exactly as `get_model_trim_ft_agreement` itself does for the same
fixtures) and test only what rework_load adds on top: the readout, the ratio confirmation, the
facts/finding split, and that a crash is never swallowed here.
"""
from datetime import datetime

import pytest

from laser_trim_analyzer.database.models import (
    AnalysisResult, FinalTestResult, StatusType, SystemType, TrackResult)
from laser_trim_analyzer.findings.analyzers import rework_load
from laser_trim_analyzer.findings import presentation as P
from findings_helpers import START, days, label, make_track

MODEL = "M"


@pytest.fixture()
def db(tmp_path):
    """A real, empty DatabaseManager -- both globals injected per global-constraints.md, even
    though rework_load is always called with `db` passed explicitly and never reaches
    get_database() itself; the rule is unconditional for anything that builds one."""
    import laser_trim_analyzer.database as _d
    import laser_trim_analyzer.database.manager as _m
    d = _m.DatabaseManager(tmp_path / "rework.db")
    _m._db_manager = d
    _d._db_manager = d
    return d


def _pair(db, tag, day, *, trim_pass, trim_error, ft_error, ft_pass=True,
         system=SystemType.B, confidence=1.0, model=MODEL, extra_error_track=None):
    """One linked (trim, final-test) pair: an AnalysisResult + one TrackResult on the laser
    side, and a FinalTestResult linked to it (`linked_trim_id`) on the final-test side.
    `extra_error_track`, when given, adds a SECOND TrackResult on the same analysis with
    status=ERROR and that final_linearity_error_shifted -- a corrupting outlier that must never
    reach the MAX() the ratio is computed from."""
    with db.session() as s:
        a = AnalysisResult(model=model, serial=f"{model}-{tag}", system=system, file_date=day,
                           filename=f"{model}_{tag}_trim.xls",
                           overall_status=StatusType.PASS if trim_pass else StatusType.FAIL)
        s.add(a)
        s.flush()
        s.add(TrackResult(analysis_id=a.id, track_id="TRK1",
                          status=StatusType.PASS if trim_pass else StatusType.FAIL,
                          linearity_pass=trim_pass,
                          final_linearity_error_shifted=trim_error))
        if extra_error_track is not None:
            s.add(TrackResult(analysis_id=a.id, track_id="TRK2", status=StatusType.ERROR,
                              linearity_pass=None,
                              final_linearity_error_shifted=extra_error_track))
        s.add(FinalTestResult(model=model, serial=f"{model}-{tag}",
                              filename=f"{model}_{tag}_ft.xls", file_date=day,
                              overall_status=StatusType.PASS,
                              linearity_pass=ft_pass, linearity_error=ft_error,
                              linked_trim_id=a.id, match_confidence=confidence))
        return a.id


def _seed(db, n_rework, n_control, *, rework_ft_error=0.10, control_ft_error=0.30,
          trim_error=0.30, start=START, model=MODEL, system=SystemType.B, extra_error_track=None,
          tag_prefix=""):
    """`n_rework` trim-FAIL/FT-PASS pairs and `n_control` trim-PASS/FT-PASS pairs, one a day.
    `tag_prefix` keeps a second `_seed` call on the same model/dates from colliding with the
    first on AnalysisResult's own (filename, file_date, model, serial) uniqueness."""
    d = list(days(start, n_rework + n_control))
    for k in range(n_rework):
        _pair(db, f"{tag_prefix}R{k}", d[k], trim_pass=False, trim_error=trim_error,
             ft_error=rework_ft_error, model=model, system=system,
             extra_error_track=extra_error_track)
    for k in range(n_control):
        _pair(db, f"{tag_prefix}C{k}", d[n_rework + k], trim_pass=True, trim_error=trim_error,
             ft_error=control_ft_error, model=model, system=system,
             extra_error_track=extra_error_track)


def tracks_for(n, model=MODEL, system="B"):
    """The window population: only file_date matters to rework_load (it sets `latest`/`cutoff`
    and, when no rework pair carries a laser, the systems fallback) -- independent of the DB rows
    seeded for the ratio itself, exactly as station_setup treats its two samples."""
    return [make_track(k, date=d, system=system) for k, d in enumerate(days(START, n))]


def only(findings):
    assert len(findings) == 1, [f.title for f in findings]
    return findings[0]


class _Boom:
    def session(self, *a, **k):
        raise AssertionError("must not query the database with nothing to report on")

    def get_model_trim_ft_agreement(self, *a, **k):
        raise AssertionError("must not query the database with nothing to report on")


# ---- a confirmed signature is one finding; readout is the unit-day count, untouched -----------

def test_a_confirmed_signature_is_one_finding_with_the_unit_day_readout(db):
    _seed(db, 40, 40)          # rework ratio 0.10/0.30 = 1/3; control ratio 0.30/0.30 = 1.0
    facts, findings = rework_load.analyze(MODEL, db, tracks_for(5), label)
    f = only(findings)
    assert f.analyzer == "rework_load" and f.category == "Rework load"
    assert f.lever == "laser_settings"
    assert f.expected_gain_points is None and f.gain_definition == ""
    assert f.n_units == 40                                        # get_model_trim_ft_agreement's overkills
    assert f.title == "Laser 1 (LTS): 40 units a year fail here and pass final test after rework"
    assert "hand trim" in f.summary and "33%" in f.summary and "100%" in f.summary
    assert f.systems == ("B",)
    assert facts["rework_unit_days"] == 40 and facts["linked"] == 80
    assert facts["confirmed"] is True
    # Rounded to 3dp by the analyzer (round(median, 3)) before it ever reaches facts/evidence.
    assert facts["median_ratio_rework"] == pytest.approx(1 / 3, abs=1e-3)
    assert facts["median_ratio_control"] == pytest.approx(1.0, abs=1e-3)


def test_evidence_facts_are_exactly_the_four_documented_keys(db):
    _seed(db, 40, 40)
    f = only(rework_load.analyze(MODEL, db, tracks_for(5), label)[1])
    assert set(f.evidence) == {"facts"}
    assert f.evidence["facts"] == pytest.approx(
        {"rework_unit_days": 40, "control_n": 40,
         "median_ratio_rework": 1 / 3, "median_ratio_control": 1.0}, abs=1e-3)


# ---- no improvement: the signature is not confirmed, so nothing -- CONFIRM_RATIO's own test ---

def test_ft_error_close_to_laser_error_is_not_confirmed(db):
    # Same ratio (1.0) either side: final test was not looser, but the error never fell either.
    _seed(db, 40, 40, rework_ft_error=0.30, control_ft_error=0.30)
    facts, findings = rework_load.analyze(MODEL, db, tracks_for(5), label)
    assert findings == []
    assert facts["rework_unit_days"] == 40                # the count is real...
    assert facts["confirmed"] is False                     # ...but the signature is not confirmed
    assert facts["median_ratio_rework"] == pytest.approx(1.0, abs=1e-6)
    assert facts["median_ratio_control"] == pytest.approx(1.0, abs=1e-6)


# ---- below MIN_UNIT_DAYS: nothing, and the ratio query never even runs ------------------------

def test_below_min_unit_days_is_nothing(db):
    assert rework_load.MIN_UNIT_DAYS == 30
    _seed(db, 20, 40)          # 20 rework unit-days, comfortably below the 30 floor
    facts, findings = rework_load.analyze(MODEL, db, tracks_for(5), label)
    assert findings == []
    assert facts["rework_unit_days"] == 20
    assert facts["confirmed"] is False
    # The floor gates BEFORE the ratio query -- no median was ever computed.
    assert "median_ratio_rework" not in facts and "control_n" not in facts


# ---- below MIN_CONTROL: nothing, even with a comfortable rework group -------------------------

def test_below_min_control_is_nothing(db):
    assert rework_load.MIN_CONTROL == 30
    _seed(db, 40, 10)          # 40 rework (plenty), only 10 pass/pass control pairs
    facts, findings = rework_load.analyze(MODEL, db, tracks_for(5), label)
    assert findings == []
    assert facts["control_n"] == 10
    assert facts["confirmed"] is False
    assert "median_ratio_rework" not in facts        # gated before a median is trusted either way


# ---- no linked final tests: nothing, and facts say so (the brief's ({...}, []) case) -----------

def test_no_linked_final_tests_is_nothing_with_facts_saying_so(db):
    # Trim data only -- no FinalTestResult rows at all, so nothing is linked.
    with db.session() as s:
        a = AnalysisResult(model=MODEL, serial=f"{MODEL}-1", system=SystemType.B, file_date=START,
                           filename=f"{MODEL}_1_trim.xls", overall_status=StatusType.FAIL)
        s.add(a)
        s.flush()
        s.add(TrackResult(analysis_id=a.id, track_id="TRK1", status=StatusType.FAIL,
                          linearity_pass=False, final_linearity_error_shifted=0.30))
    facts, findings = rework_load.analyze(MODEL, db, tracks_for(5), label)
    assert findings == []
    assert facts != {}
    assert facts["linked"] == 0 and facts["rework_unit_days"] == 0
    assert facts["confirmed"] is False and "note" in facts and facts["note"]


# ---- failed-processing tracks are excluded from the MAX() the ratio is built from --------------

def test_failed_processing_tracks_never_reach_the_ratio(db):
    # Every REWORK analysis carries a SECOND, ERROR-status track with a wildly larger
    # final_linearity_error_shifted (5.0 against the real 0.30). If it were not excluded from the
    # MAX() the ratio is built from, the rework ratio would collapse toward 0.10/5.0 instead of
    # 1/3. Left off the control group on purpose: get_model_trim_ft_agreement's own "every track
    # must pass" rule does not itself exclude ERROR-status tracks (out of this analyzer's scope to
    # change -- see rework_load's docstring), so adding one to an otherwise-passing control
    # analysis would flip ITS readout classification too, muddying what this test pins. A rework
    # analysis is unaffected either way: its real track already fails it on its own.
    d = days(START, 70)
    for k in range(35):
        _pair(db, f"R{k}", d[k], trim_pass=False, trim_error=0.30, ft_error=0.10,
             extra_error_track=5.0)
    for k in range(35):
        _pair(db, f"C{k}", d[35 + k], trim_pass=True, trim_error=0.30, ft_error=0.30)
    facts, findings = rework_load.analyze(MODEL, db, tracks_for(5), label)
    f = only(findings)
    assert facts["rework_unit_days"] == 35             # unaffected: the real track already fails it
    assert facts["median_ratio_rework"] == pytest.approx(1 / 3, abs=1e-3)
    assert facts["median_ratio_control"] == pytest.approx(1.0, abs=1e-3)
    assert f.n_units == 35


# ---- nothing to report on: the database is never even queried ---------------------------------

def test_no_tracks_the_database_is_never_queried():
    assert rework_load.analyze(MODEL, _Boom(), [], label) == ({}, [])


def test_undated_tracks_the_database_is_never_queried():
    from dataclasses import replace
    tracks = [replace(t, file_date=None) for t in tracks_for(5)]
    assert rework_load.analyze(MODEL, _Boom(), tracks, label) == ({}, [])


# ---- a crash is never swallowed here -----------------------------------------------------------

def test_a_raising_agreement_lookup_is_not_caught_here(db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(db, "get_model_trim_ft_agreement", boom)
    with pytest.raises(RuntimeError, match="database is locked"):
        rework_load.analyze(MODEL, db, tracks_for(5), label)


# ---- systems: the laser(s) whose OWN failures are being reworked, not the model's every laser --

def test_systems_are_the_lasers_the_rework_pairs_actually_ran_on(db):
    _seed(db, 20, 40, system=SystemType.A, model=MODEL, tag_prefix="A")
    _seed(db, 20, 0, system=SystemType.B, model=MODEL, tag_prefix="B")   # more rework on a 2nd laser
    f = only(rework_load.analyze(MODEL, db, tracks_for(5), label)[1])
    assert f.systems == ("A", "B")
    assert f.title.startswith("Laser 2 (DLTS): 40 ")             # laser_label(systems[0])


# ---- presentation wiring: laser_time group, rework_unit_days readout, unmapped-analyzer test ---

def test_group_and_readout_through_presentation(db):
    _seed(db, 40, 40)
    f = only(rework_load.analyze(MODEL, db, tracks_for(5), label)[1])
    d = f.to_dict()
    assert P.group_key(d) == "laser_time"
    assert P.readout(d) == 40.0
    assert P.statement(d) == f.title
