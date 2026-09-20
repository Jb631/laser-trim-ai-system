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
    assert hot["units_per_year"] > 0
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
               "expected_gain_points": None, "units_per_year": None, "annual_volume": 5, "n_units": 5,
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
