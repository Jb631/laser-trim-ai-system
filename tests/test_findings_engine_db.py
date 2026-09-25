from dataclasses import replace

from findings_helpers import START, era, ink_tracks


def _db(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / "engine.db")


def _hot():
    return ink_tracks(600, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25)


def test_the_two_cache_tables_exist_on_a_fresh_database(tmp_path):
    import sqlalchemy as sa
    db = _db(tmp_path)
    with db.session() as s:
        names = {r[0] for r in s.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table'"))}
    assert {"process_findings", "model_process_facts"} <= names


def test_refresh_stores_ranked_findings_and_always_stores_facts(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    tracks = {"HOT": _hot(), "QUIET": ink_tracks(600, START, (1.0, 2.0), lambda r: 0.5)}
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: tracks[m])
    assert engine.refresh_findings(db, ["HOT", "QUIET"]) == 1
    ranked = db.get_process_findings()
    assert [d["model"] for d in ranked] == ["HOT"]
    hot = ranked[0]
    assert hot["lever"] == "ink" and hot["lever_label"].startswith("Ink") and hot["lead_time"] == "next lot"
    assert hot["tracks_per_year"] > 0
    assert len(hot["evidence"]["bins"]) == 5           # the evidence survived JSON, not a silent "null"
    assert db.get_process_findings("QUIET") == []
    quiet = db.get_process_facts("QUIET")              # silence still leaves the measurements
    assert quiet["tracks"] == 600 and quiet["yardstick"]["faithful"] is True
    assert db.get_process_facts("never-seen") is None


def test_refresh_replaces_rather_than_appends(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    hot = _hot()
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)
    engine.refresh_findings(db, ["HOT"])
    engine.refresh_findings(db, ["HOT"])
    assert len(db.get_process_findings("HOT")) == 1


def test_one_model_failing_does_not_stop_the_rest(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    hot = _hot()

    def loader(_db, model):
        if model == "BROKEN":
            raise RuntimeError("boom")
        return hot
    monkeypatch.setattr(engine, "load_model_tracks", loader)
    assert engine.refresh_findings(db, ["BROKEN", "HOT"]) == 1
    assert len(db.get_process_findings("HOT")) == 1


def test_an_unfaithful_yardstick_silences_the_grading_analyzer(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    # Stored verdicts that contradict the sweeps: the yardstick cannot vouch for this model.
    liars = [replace(t, linearity_pass=not t.linearity_pass) for t in era(0, START, 200, (1.0,), 0.5)]
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: liars)
    facts, _ = engine.compute_for_model(db, "LIAR")
    assert facts["yardstick"]["faithful"] is False
    assert facts["trim_effort"] is None


def test_a_date_in_the_evidence_is_flattened_not_silently_nulled(tmp_path):
    from datetime import datetime
    db = _db(tmp_path)
    finding = {"model": "X", "analyzer": "a", "category": "c", "lever": "ink", "title": "t", "summary": "s",
               "expected_gain_points": None, "tracks_per_year": None, "annual_volume": 5, "n_units": 5,
               "evidence": {"when": datetime(2026, 1, 6)}}
    db.replace_process_findings("X", {"tracks": 1, "when": datetime(2026, 1, 6)}, [finding])
    assert db.get_process_findings("X")[0]["evidence"]["when"].startswith("2026-01-06")
    assert db.get_process_facts("X")["when"].startswith("2026-01-06")


def test_every_finding_carries_the_models_annual_volume(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    hot = _hot()                                       # 600 tracks over 300 days: all inside one year
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)
    _, findings = engine.compute_for_model(db, "HOT")
    assert findings and all(f.annual_volume == 600 for f in findings)


# ---- fix round 1: a failure is never silence; the ranking is tested with more than one row ----

def test_a_crashed_analyzer_is_named_not_shown_as_silence(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    hot = _hot()
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)

    def boom(*a, **k):
        raise RuntimeError("analyzer exploded")
    monkeypatch.setattr(engine.recipe_change, "analyze", boom)
    monkeypatch.setattr(engine.trim_effort, "analyze", boom)
    # A fixed instant so this stays a test of load_model_tracks() being mocked, not the DB (no
    # fleet_latest would make compute_for_model query it from db itself). A real, if empty, db is
    # still needed -- station_setup queries it directly (spec_alignment.sample_and_compare), and
    # an empty database just answers "insufficient" (no crash) for a model it has never heard of.
    facts, findings = engine.compute_for_model(_db(tmp_path), "HOT", fleet_latest=START)
    assert set(facts["errors"]) == {"recipe_change", "trim_effort"}
    assert facts["errors"]["recipe_change"] == "RuntimeError: analyzer exploded"
    assert facts["recipe_history"] is None and facts["trim_effort"] is None   # not computed...
    assert facts["yardstick"]["faithful"] is True                             # ...and NOT because of the yardstick
    assert [f.analyzer for f in findings] == ["ink_target"]                   # the survivor still reports


def test_a_healthy_run_has_no_errors_and_an_explicit_history(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    hot = _hot()
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)
    facts, _ = engine.compute_for_model(_db(tmp_path), "HOT", fleet_latest=START)
    assert facts["errors"] == {}
    assert isinstance(facts["recipe_history"], list) and isinstance(facts["trim_effort"], dict)
    assert isinstance(facts["setup_change"], list)


def test_every_documented_key_exists_even_for_a_model_with_no_tracks(monkeypatch):
    from laser_trim_analyzer.findings import engine
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: [])
    facts, findings = engine.compute_for_model(None, "EMPTY")
    assert findings == [] and facts["tracks"] == 0
    assert set(facts) == {"model", "tracks", "annual_volume", "latest", "yardstick",
                          "recipe_history", "trim_effort", "limit_tables", "cut_setting", "pass_burden",
                          "machine_compare", "loss_origin", "station_setup", "rework_load",
                          "setup_change", "errors"}


def test_refresh_reports_what_did_not_get_done(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    hot = _hot()

    def loader(_db, model):
        if model == "BROKEN":
            raise RuntimeError("boom")
        return hot
    monkeypatch.setattr(engine, "load_model_tracks", loader)

    def bad_effort(*a, **k):
        raise ValueError("no sweeps")
    monkeypatch.setattr(engine.trim_effort, "analyze", bad_effort)
    report = {}
    stored = engine.refresh_findings(db, ["BROKEN", "HOT"], report)
    assert report == {"models": 2, "stored": stored,
                      "failed_models": {"BROKEN": "RuntimeError: boom"},
                      "analyzer_errors": {"HOT": {"trim_effort": "ValueError: no sweeps"}}}
    assert db.get_process_facts("HOT")["errors"] == {"trim_effort": "ValueError: no sweeps"}   # and it is cached
    assert db.get_process_facts("BROKEN") is None


def _row(model, tpy, size, title="t"):
    """`tpy` = tracks a year the gain is worth (None = the finding claims no rate); `size` = its own sample."""
    return {"model": model, "analyzer": "a", "category": "c", "lever": "ink", "title": title, "summary": "s",
            "expected_gain_points": None if tpy is None else 1.0, "tracks_per_year": tpy,
            "scope_annual_tracks": 0 if tpy is None else int(tpy * 100), "annual_volume": size,
            "n_units": size, "evidence": {}}


def test_findings_come_back_ranked_across_models(tmp_path):
    db = _db(tmp_path)
    db.replace_process_findings("SMALL", {"tracks": 1}, [_row("SMALL", 5.0, 9000)])
    db.replace_process_findings("NOGAIN", {"tracks": 1}, [_row("NOGAIN", None, 99999),
                                                           _row("NOGAIN", None, 10, "second")])
    db.replace_process_findings("BIG", {"tracks": 1}, [_row("BIG", 400.0, 50)])
    ranked = db.get_process_findings()
    # recoverable tracks a year first (NOT sample size); findings that claim no rate last, by size
    assert [(d["model"], d["title"]) for d in ranked] == [("BIG", "t"), ("SMALL", "t"),
                                                          ("NOGAIN", "t"), ("NOGAIN", "second")]


def test_an_empty_model_name_matches_nothing_not_every_model(tmp_path):
    db = _db(tmp_path)
    db.replace_process_findings("X", {"tracks": 1}, [_row("X", 1.0, 5)])
    assert db.get_process_findings("") == []
    assert len(db.get_process_findings(None)) == 1


def test_findings_say_when_they_were_computed_and_the_clock_is_not_deprecated(tmp_path):
    import warnings
    db = _db(tmp_path)
    with warnings.catch_warnings():
        # ONLY the old clock: a blanket "error" filter would fail this test the day some
        # library on the save path deprecates something unrelated.
        warnings.filterwarnings("error", message=r".*utcnow.*", category=DeprecationWarning)
        db.replace_process_findings("X", {"tracks": 1}, [_row("X", 1.0, 5)])
    got = db.get_process_findings("X")[0]
    assert got["computed_at"][:2] == "20" and got["title"] == "t"


def test_models_whose_analyzers_failed_can_be_listed(tmp_path):
    db = _db(tmp_path)
    db.replace_process_findings("FINE", {"tracks": 5, "errors": {}}, [])
    db.replace_process_findings("OLDROW", {"tracks": 5}, [])                       # a cache row from before "errors" existed
    db.replace_process_findings("HURT", {"tracks": 5, "errors": {"trim_effort": "ValueError: no sweeps"}}, [])
    assert db.get_process_errors() == {"HURT": {"trim_effort": "ValueError: no sweeps"}}


def test_the_limit_table_analyzer_runs_inside_the_engine_and_its_history_is_cached(tmp_path, monkeypatch):
    from findings_helpers import table, table_era
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    tracks = table_era(0, START, 200, table(12, 0.10), 0.6) + table_era(1000, START, 200, table(23, 0.10), 0.3)
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: tracks)
    assert engine.refresh_findings(db, ["TWO"]) == 1
    (found,) = db.get_process_findings("TWO")
    assert found["analyzer"] == "limit_tables" and found["lever"] == "laser_limit_table"
    assert found["tracks_per_year"] is None and found["evidence"]["comparison"]["kind"] == "same_band_other_density"
    cached = db.get_process_facts("TWO")["limit_tables"]
    assert [(h["graded"], h["n"]) for h in cached] == [(12, 200), (23, 200)] and db.get_process_errors() == {}


def test_a_crash_in_the_limit_table_analyzer_is_named_like_any_other(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    hot = _hot()
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)

    def boom(*a, **k):
        raise RuntimeError("bad table")
    monkeypatch.setattr(engine.limit_tables, "analyze", boom)
    facts, findings = engine.compute_for_model(_db(tmp_path), "HOT", fleet_latest=START)
    assert facts["errors"] == {"limit_tables": "RuntimeError: bad table"} and facts["limit_tables"] is None
    assert [f.analyzer for f in findings] == ["ink_target"]


# ---- Task 3: station_setup is the first analyzer that reads the database itself -- its read
# failure must reach the engine's guard uncaught, unlike compare_station_specs (the banner's
# function), which turns the same failure into "insufficient" on purpose.

def test_a_raising_sampler_is_named_by_the_engine_like_any_other_crash(tmp_path, monkeypatch):
    """station_setup calls spec_alignment.sample_and_compare directly -- made to raise instead of
    degrading, precisely so a read failure is never hidden from the findings engine. Patching the
    SAMPLER itself (not station_setup.analyze) proves the analyzer does not catch it before the
    engine's own guard ever sees it."""
    from laser_trim_analyzer.core import spec_alignment
    from laser_trim_analyzer.findings import engine
    hot = _hot()
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)

    def boom(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(spec_alignment, "sample_and_compare", boom)
    facts, findings = engine.compute_for_model(_db(tmp_path), "HOT", fleet_latest=START)
    assert facts["errors"] == {"station_setup": "RuntimeError: database is locked"}
    assert facts["station_setup"] is None
    assert [f.analyzer for f in findings] == ["ink_target"]                   # the rest still ran


# ---- Task 4: rework_load is the second analyzer that reads the database itself (the unit-day
# rework count, plus its own query for the ratio evidence -- see that module's own docstring for
# why get_model_trim_ft_agreement alone is not enough). A crash there must reach the engine's
# guard uncaught, exactly like station_setup's.

def test_a_crash_in_rework_load_is_named_by_the_engine_like_any_other(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    hot = _hot()
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)

    def boom(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(engine.rework_load, "analyze", boom)
    facts, findings = engine.compute_for_model(_db(tmp_path), "HOT", fleet_latest=START)
    assert facts["errors"] == {"rework_load": "RuntimeError: database is locked"}
    assert facts["rework_load"] is None
    assert [f.analyzer for f in findings] == ["ink_target"]                   # the rest still ran


def test_rework_load_runs_inside_the_engine_and_its_facts_are_cached(tmp_path, monkeypatch):
    """A real, confirmed rework signature, seeded straight into the database as stored sweeps on
    both stations (not through load_model_tracks, which is mocked here to supply only the window
    population -- exactly as rework_load's own tests keep the two samples independent)."""
    from laser_trim_analyzer.findings import engine
    from findings_helpers import days, make_track
    from test_findings_rework_load import seed

    db = _db(tmp_path)
    seed(db, 40, 90, model="REWORK")        # 40 reworked unit-days at 1/3, 90 pass/pass at 1.0
    tracks = [make_track(9000 + k, date=dt, system="B") for k, dt in enumerate(days(START, 5))]
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: tracks)
    assert engine.refresh_findings(db, ["REWORK"]) == 1
    (found,) = db.get_process_findings("REWORK")
    assert found["analyzer"] == "rework_load" and found["lever"] == "laser_settings"
    assert found["n_units"] == 40 and found["tracks_per_year"] is None       # no gain claimed
    cached = db.get_process_facts("REWORK")["rework_load"]
    assert cached["rework_unit_days"] == 40 and cached["confirmed"] is True
    assert cached["by_laser"]["Laser 1 (LTS)"]["control_top_third_n"] == 30
    assert db.get_process_errors() == {}


# ---- Task 6: setup_change is a laser setting BEYOND the cut recipe -- takes no db, so wiring it
# needs nothing beyond the tracks population, unlike station_setup/rework_load before it.

def test_a_crash_in_setup_change_is_named_like_any_other(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    hot = _hot()
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)

    def boom(*a, **k):
        raise RuntimeError("bad setup block")
    monkeypatch.setattr(engine.setup_change, "analyze", boom)
    facts, findings = engine.compute_for_model(_db(tmp_path), "HOT", fleet_latest=START)
    assert facts["errors"] == {"setup_change": "RuntimeError: bad setup block"}
    assert facts["setup_change"] is None
    assert [f.analyzer for f in findings] == ["ink_target"]                   # the rest still ran


def test_setup_change_runs_inside_the_engine_and_its_history_is_cached(tmp_path, monkeypatch):
    from datetime import datetime
    from test_findings_setup_change import setup_era
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    tracks = (setup_era(0, START, 200, {"laser_power": 50}, 0.80)
             + setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50))
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: tracks)
    assert engine.refresh_findings(db, ["SETUP"]) == 1
    (found,) = db.get_process_findings("SETUP")
    assert found["analyzer"] == "setup_change" and found["lever"] == "laser_settings"
    assert found["tracks_per_year"] is None                       # no gain claimed
    cached = db.get_process_facts("SETUP")["setup_change"]
    assert len(cached) == 1 and cached[0]["setting"] == "laser_power"
    assert db.get_process_errors() == {}


# ---- fix round 1, Important #1: one real change, one finding (task-6-review.md) ----------------

def test_a_real_laser1_cut_length_change_is_only_recipe_changes_finding(tmp_path, monkeypatch):
    """The exact regression fix round 1 exists for: a genuine laser-1 cut-length change is
    readable from BOTH trim_passes (recipe_change's own per-pass log, via TrackView.recipe/passes)
    and trim_setup.parameters['laser_cut_length'] (the file-level nominal value, via
    TrackView.setup) -- through the whole engine, it must produce exactly ONE finding
    (recipe_change's), never a second, redundantly-worded one from setup_change."""
    from datetime import datetime
    from dataclasses import replace
    from laser_trim_analyzer.findings import engine
    from findings_helpers import days, make_track

    def track(i, date, cut_length, good):
        t = make_track(i, date=date, system="B", passes=((0.8 if good else 1.5, cut_length),))
        return replace(t, setup={"laser_cut_length": cut_length})

    before = [track(k, d, 4100.0, (k % 100) < 80) for k, d in enumerate(days(START, 200))]
    after = [track(1000 + k, d, 4000.0, (k % 100) < 40)
            for k, d in enumerate(days(datetime(2024, 9, 1), 200))]
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: before + after)
    facts, findings = engine.compute_for_model(_db(tmp_path), "M", fleet_latest=START)
    setup_findings = [f for f in findings if f.analyzer == "setup_change"]
    recipe_findings = [f for f in findings if f.analyzer == "recipe_change"]
    assert setup_findings == []
    assert len(recipe_findings) == 1
    assert "4100" in recipe_findings[0].title and "4000" in recipe_findings[0].title


# ---- Task 5: _fleet_latest -- what "now" means, and what cannot be trusted to say so ----

def test_a_failed_processing_row_does_not_move_fleet_latest(tmp_path):
    """A PROCESSING_FAILED/ERROR row's file_date is when the analyser gave up
    (_create_minimal_metadata sets it to datetime.now()), not a measurement -- so a fresh
    crash must never be able to make itself "the latest data"."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import AnalysisResult, StatusType, SystemType
    from laser_trim_analyzer.findings.engine import _fleet_latest
    db = _db(tmp_path)
    with db.session() as s:
        s.add(AnalysisResult(model="M", serial="M-1", system=SystemType.B,
                             filename="m1.xls", file_date=START, overall_status=StatusType.PASS))
    before = _fleet_latest(db)
    with db.session() as s:
        s.add(AnalysisResult(model="M", serial="M-2", system=SystemType.B,
                             filename="m2.xls", file_date=datetime.now(), overall_status=StatusType.ERROR))
    assert _fleet_latest(db) == before == START


def test_a_row_dated_more_than_a_day_in_the_future_does_not_move_fleet_latest(tmp_path):
    """A mistyped filename date can put a file months out -- which would make every OTHER
    model's real, current data look "stale" by comparison if it were allowed to set "now"."""
    from datetime import datetime, timedelta
    from laser_trim_analyzer.database.models import AnalysisResult, StatusType, SystemType
    from laser_trim_analyzer.findings.engine import _fleet_latest
    db = _db(tmp_path)
    with db.session() as s:
        s.add(AnalysisResult(model="M", serial="M-1", system=SystemType.B,
                             filename="m1.xls", file_date=START, overall_status=StatusType.PASS))
    before = _fleet_latest(db)
    with db.session() as s:
        s.add(AnalysisResult(model="M", serial="M-2", system=SystemType.B,
                             filename="m2.xls", file_date=datetime.now() + timedelta(days=400),
                             overall_status=StatusType.PASS))
    assert _fleet_latest(db) == before == START


def test_fleet_latest_does_not_use_the_deprecated_datetime_adapter(tmp_path):
    """_fleet_latest binds its cutoff as a formatted string, not a raw datetime, so it never
    falls back to sqlite3's own adapter registry (deprecated as of Python 3.12 -- text()
    bypasses SQLAlchemy's own DATETIME bind_processor, which is what the ORM path uses)."""
    import warnings
    from laser_trim_analyzer.findings.engine import _fleet_latest
    db = _db(tmp_path)
    with warnings.catch_warnings():
        # ONLY the datetime adapter: a blanket "error" filter would fail this test the day
        # some library on this path deprecates something unrelated (e.g. pydantic's).
        warnings.filterwarnings("error", message=r".*datetime adapter.*", category=DeprecationWarning)
        assert _fleet_latest(db) is None                  # empty db -- exercises the bind either way


# ---- the findings engine loads no plotting code (fix round 2, 2026-09-25) --------------------------
# rework_load reads both stations' sweeps through the ONE graded-trace definition,
# corrected_errors. While that lived in export/unit_chart (a matplotlib module), importing the
# engine -- on the ingest worker, in scripts, in every analyzer test -- loaded matplotlib.figure.

def test_importing_the_findings_engine_loads_no_plotting_code():
    """In a FRESH interpreter, so nothing this test session already imported can hide it."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src"
    code = ("import sys, laser_trim_analyzer.findings.engine as e; "
            "print(e.__file__); "
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(src)}, timeout=120)
    assert out.returncode == 0, out.stderr
    engine_file, loaded = out.stdout.strip().splitlines()[-2:]
    assert engine_file.startswith(str(src))            # the code under test, not another checkout
    assert loaded == "[]", f"importing the findings engine loaded {loaded}"


# ---- I1 (final review, 2026-09-25): machine_compare's window ends at the FLEET's newest file ----

def test_the_engine_anchors_machine_compare_to_the_fleets_newest_file(tmp_path, monkeypatch):
    """8081-4's shape: a model last run in 2015-16, compared on its own 'latest year', read as a
    current recommendation. Anchored to the fleet's newest file it is a dated fact only."""
    from datetime import datetime
    from laser_trim_analyzer.findings import engine
    from test_findings_machine_compare import block
    old = (block(0, datetime(2015, 5, 1), 200, "A", 0.95)
           + block(1000, datetime(2015, 5, 1), 200, "B", 0.60))
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: old)
    facts, findings = engine.compute_for_model(_db(tmp_path), "OLD", fleet_latest=datetime(2026, 9, 22))
    assert [f for f in findings if f.analyzer == "machine_compare"] == []
    (c,) = facts["machine_compare"]["comparisons"]
    assert c["in_window"] is False and c["months"] == ["2015-05"]
    assert facts["machine_compare"]["window"]["anchored_to"] == "2026-09-22"


# ---- I4 (final review, 2026-09-25): a crash in machine_compare or loss_origin is named too ------
# With both guards changed to swallow silently, every findings test that could run still passed.

def test_a_crash_in_machine_compare_is_named_like_any_other(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    hot = _hot()
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)

    def boom(*a, **k):
        raise RuntimeError("bad month cell")
    monkeypatch.setattr(engine.machine_compare, "analyze", boom)
    facts, findings = engine.compute_for_model(_db(tmp_path), "HOT", fleet_latest=START)
    assert facts["errors"] == {"machine_compare": "RuntimeError: bad month cell"}
    assert facts["machine_compare"] is None
    assert facts["loss_origin"] is not None and facts["recipe_history"] is not None   # the rest ran
    assert [f.analyzer for f in findings] == ["ink_target"]


def test_a_crash_in_loss_origin_is_named_like_any_other(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    hot = _hot()
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)

    def boom(*a, **k):
        raise RuntimeError("bad incoming sweep")
    monkeypatch.setattr(engine.loss_origin, "analyze", boom)
    facts, findings = engine.compute_for_model(_db(tmp_path), "HOT", fleet_latest=START)
    assert facts["errors"] == {"loss_origin": "RuntimeError: bad incoming sweep"}
    assert facts["loss_origin"] is None
    assert facts["machine_compare"] is not None and facts["station_setup"] is not None  # the rest ran
    assert [f.analyzer for f in findings] == ["ink_target"]


def test_the_stored_predictor_auc_is_read_without_loading_the_ml_manager(tmp_path):
    """The Findings tab sets loss_origin's AUC beside the final-test predictor's (spec ruling 2);
    the predictor's is the one its last training stored in model_ml_state."""
    from laser_trim_analyzer.database.models import ModelMLState
    db = _db(tmp_path)
    assert db.get_predictor_auc("NONE") is None
    with db.session() as s:
        s.add(ModelMLState(model="UNTRAINED", predictor_trained=False, predictor_auc=0.9))
        s.add(ModelMLState(model="TRAINED", predictor_trained=True, predictor_auc=0.8123))
    assert db.get_predictor_auc("UNTRAINED") is None
    assert db.get_predictor_auc("TRAINED") == 0.8123
