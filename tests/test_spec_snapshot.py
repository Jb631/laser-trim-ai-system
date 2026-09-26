"""A spec snapshot the worker can carry, answering exactly as the database does (ingest-speed
Task 8; spec §4, ruling 16).

What the ingest's ANALYSIS reads from the database is three things: the model spec, looked up per
file (`get_model_spec`, or `resolve_spec_for_ft` for a final test, whose serial may name a section),
and the ML thresholds and predictors, loaded when a Processor is built. A worker process must never
open a database (ruling 15), so all three reach it as a SpecSnapshot taken at folder start -- and
through ONE resolver shared with `get_model_spec` / `resolve_spec_for_ft`, so the snapshot and the
database cannot answer differently. The trap that makes "one resolver" matter: `get_model_spec`
searches aliases in the table's own row order, and a copy that searched them in model order (the
work probe's did) answers differently whenever two specs share an alias.
"""
import pickle
import shutil
import sqlite3
from pathlib import Path

import pytest

import save_rows

REPO = Path(__file__).resolve().parents[1]

# Invented specs. The models are the fixtures' own, plus sectioned and aliased parts; the values
# are invented. Inserted in THIS order, so the ids -- the alias search order -- are known.
SPECS = [
    {"model": "8232-1", "linearity_type": "independent", "electrical_angle": 340.0,
     "electrical_angle_tol": 5.0, "electrical_angle_tol_type": "bilateral"},
    {"model": "8074", "linearity_type": "absolute", "exclude_points": "1-2"},
    {"model": "7553", "linearity_type": "terminal", "exclude_points_ft": "3"},
    {"model": "7458", "linearity_type": "independent", "aliases": "7458X"},
    {"model": "7539-2", "linearity_type": "absolute"},
    # AMBIG-1 is listed by 8639-30 (id 6) AND 1621501 (id 13): the older row answers -- in model
    # order 1621501 would, which is how a copy of the rule sorted by model gets it wrong.
    {"model": "8639-30", "linearity_type": "independent", "aliases": "AMBIG-1"},
    {"model": "8340", "linearity_type": "independent"},
    {"model": "8340-3", "linearity_type": "terminal"},
    {"model": "8508", "linearity_type": "independent"},
    {"model": "8508-A", "linearity_type": "absolute"},
    {"model": "8508-B", "linearity_type": "terminal", "electrical_angle": 120.0},
    {"model": "8508-D", "linearity_type": "independent"},
    {"model": "1621501", "linearity_type": "absolute",
     "aliases": " 2001621501 | 1621501X |AMBIG-1"},
    {"model": "9101", "linearity_type": "terminal", "aliases": "9101Y"},
    {"model": "9102", "aliases": ""},
]
MODELS = ["8232-1", "8074", "7553", "7458", "7539-2", "8639-30", "8340", "8340-3", "8508",
          "8508-A", "8508-B", "8508-D", "1621501", "9101", "9102", "8434", "9990",
          "7458X", "AMBIG-1", "9101Y", "2001621501", "1621501X", " 8232-1 ", "8232-1 ",
          "8232-1-", "8508-C", "8508-b", "unknown", "", None, "   "]
FT_QUESTIONS = [("8508", s) for s in ("31B", "31b", "31B ", "31 B", "31", "31Z", "B", None, "")] + [
    ("8508-A", "7A"), ("7458", "7"), ("7458", "7A"), ("2001621501", "5A"), ("AMBIG-1", "9"),
    (None, "31B"), ("", "31B"), ("8340", "12"), ("unknown", "3C"), (" 8508", "31B")]


@pytest.fixture
def db(tmp_path, monkeypatch):
    from laser_trim_analyzer.database.manager import DatabaseManager
    d = DatabaseManager(tmp_path / "specs.db")
    save_rows.inject(d, monkeypatch)
    for spec in SPECS:
        d.save_model_spec(dict(spec))
    yield d
    d.close()


def _invented_ml():
    """One invented sigma threshold and one trained predictor, both for 8232-1 (invented data)."""
    import random
    import pandas as pd
    from laser_trim_analyzer.ml.predictor import FEATURE_COLUMNS, ModelPredictor
    random.seed(7)
    trained = ModelPredictor("8232-1")
    X = pd.DataFrame([{c: random.random() + (0.4 if i % 3 == 0 else 0.0) for c in FEATURE_COLUMNS}
                      for i in range(60)])
    trained.train(X, pd.Series([1 if i % 3 == 0 else 0 for i in range(60)]))
    assert trained.is_trained
    return {"8232-1": 0.0042}, {"8232-1": trained}


@pytest.fixture
def fresh_ml_cache():
    """The shared ML manager is a five-minute PROCESS cache: drop it before and after, so no other
    test's database leaks into this one's ML state, and none of this one's into the next."""
    from laser_trim_analyzer.ml import invalidate_shared_ml_manager
    invalidate_shared_ml_manager()
    yield
    invalidate_shared_ml_manager()


def _snapshot(db, use_ml=False):
    from laser_trim_analyzer.core.processor import take_spec_snapshot
    return take_spec_snapshot(use_ml=use_ml)


def test_the_snapshot_answers_every_question_exactly_as_the_database(db):
    """Every fixture model, every alias, sections by trailing serial letter (upper, lower, with a
    trailing space, a missing section falling back to the plain model), an alias two specs share
    (the lower id wins), whitespace, unknown and empty models: the same dict, or the same None."""
    snap = _snapshot(db)
    for model in MODELS:
        assert snap.get_model_spec(model) == db.get_model_spec(model), model
    for model, serial in FT_QUESTIONS:
        assert (snap.resolve_spec_for_ft(model, serial)
                == db.resolve_spec_for_ft(model, serial)), (model, serial)
    # ...and the answers are the interesting ones, not all None:
    assert db.get_model_spec("AMBIG-1")["model"] == "8639-30", "two specs share it: the older row wins"
    assert db.resolve_spec_for_ft("8508", "31b")["model"] == "8508-B"
    assert db.resolve_spec_for_ft("8508", "31Z")["model"] == "8508"
    assert db.get_model_spec(" 2001621501 ") is not None and db.get_model_spec("9102") is not None


def test_one_resolver_answers_for_the_database_and_the_snapshot(db, monkeypatch):
    """Not two copies of one rule: both ask the same function."""
    from laser_trim_analyzer.database import specs
    asked = []
    real = specs.resolve_model_spec

    def spy(model, primary, alias_rows):
        asked.append(model)
        return real(model, primary, alias_rows)

    monkeypatch.setattr(specs, "resolve_model_spec", spy)
    snap = _snapshot(db)
    db.get_model_spec("7458X")
    snap.get_model_spec("7458X")
    db.resolve_spec_for_ft("8508", "31B")
    snap.resolve_spec_for_ft("8508", "31B")
    assert asked == ["7458X", "7458X", "8508-B", "8508-B"], asked


def test_the_snapshot_is_plain_picklable_data(db, monkeypatch):
    """Task 11 sends it to spawned processes: it pickles, and comes back answering the same --
    ML thresholds and a trained predictor included (invented training data)."""
    from laser_trim_analyzer.core import processor
    from laser_trim_analyzer.ml.predictor import FEATURE_COLUMNS
    thresholds, predictors = _invented_ml()
    trained = predictors["8232-1"]
    monkeypatch.setattr(processor, "load_ml_state",
                        lambda db_: (dict(thresholds), dict(predictors)))
    snap = _snapshot(db, use_ml=True)
    back = pickle.loads(pickle.dumps(snap))     # our own object, made above: the spawn transport
    for model in MODELS:
        assert back.get_model_spec(model) == snap.get_model_spec(model), model
    assert back.ml_thresholds == {"8232-1": 0.0042}
    features = {c: 0.3 for c in FEATURE_COLUMNS}
    assert (back.ml_predictors["8232-1"].predict_failure_probability(features)
            == trained.predict_failure_probability(features))
    assert all(type(v) in (str, int, float, bool, type(None)) for row in back.specs
               for v in row.values()), "a spec row is plain values"


def test_a_processor_with_a_snapshot_never_asks_the_database(db, monkeypatch):
    """Given a snapshot, the analysis reads specs and ML state from it and NOTHING from the
    database -- `get_database` raises here -- and stores exactly what the database-backed
    Processor stores (the 8232-1 spec moves the offset, error and fail points)."""
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    fixtures = sorted((REPO / "tests" / "fixtures" / "trim").glob("*.xls"))
    snap = _snapshot(db)
    reference = Processor(use_ml=False)
    reference.ml_storage_path = REPO / "no_ml_models_here"
    want = [reference.process_file(f).model_dump(exclude={"processing_time"}) for f in fixtures]

    asked = []

    def refused():
        asked.append("get_database")          # recorded: a reach the caller swallows still shows
        raise AssertionError("the analysis asked the database")

    monkeypatch.setattr(mgr, "get_database", refused)
    monkeypatch.setattr(dbpkg, "get_database", refused)
    carried = Processor(use_ml=True, snapshot=snap)        # use_ml: from the snapshot, not the db
    carried.ml_storage_path = REPO / "no_ml_models_here"
    got = [carried.process_file(f).model_dump(exclude={"processing_time"}) for f in fixtures]
    diffs = save_rows.differences(got, want, exact=True)     # exact, and a stored NaN is a value
    assert not diffs, diffs
    assert asked == [], "the analysis asked the database for something the snapshot carries"
    assert want[2]["tracks"][0]["linearity_fail_points"] == 1, "the 8232-1 spec was in force"


def test_with_ml_on_the_snapshot_stores_exactly_what_the_database_backed_processor_stores(
        db, monkeypatch):
    """Review I-1: the snapshot's ML HALF. With invented ML state -- one sigma threshold and one
    trained predictor, for 8232-1 -- a Processor carrying a PICKLED snapshot (the spawn transport,
    Task 11) stores exactly what the database-backed Processor stores, the ML columns included
    (sigma_threshold, sigma_pass, failure_probability, risk_category), and asks the database
    nothing. And the ML state is in force: with ML off, stored values move -- so a snapshot, or a
    Processor, that dropped it could not pass."""
    from laser_trim_analyzer.core import processor
    from laser_trim_analyzer.core.processor import Processor, take_spec_snapshot
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.database.specs import SpecSnapshot
    import laser_trim_analyzer.database as dbpkg
    thresholds, predictors = _invented_ml()
    monkeypatch.setattr(processor, "load_ml_state",
                        lambda db_: (dict(thresholds), dict(predictors)))
    fixtures = sorted((REPO / "tests" / "fixtures" / "trim").glob("*.xls"))

    def run(proc):
        proc.ml_storage_path = REPO / "no_ml_models_here"
        return [proc.process_file(f).model_dump(exclude={"processing_time"}) for f in fixtures]

    by_database = run(Processor(use_ml=True))
    snap = pickle.loads(pickle.dumps(take_spec_snapshot(use_ml=True)))  # our own object, made here
    asked = []

    def refused():
        asked.append("get_database")
        raise AssertionError("the analysis asked the database")

    monkeypatch.setattr(mgr, "get_database", refused)
    monkeypatch.setattr(dbpkg, "get_database", refused)
    carried = run(Processor(use_ml=True, snapshot=snap))
    ml_off = run(Processor(use_ml=False, snapshot=SpecSnapshot(specs=snap.specs)))
    diffs = save_rows.differences(carried, by_database, exact=True)
    assert not diffs, diffs
    assert asked == [], "the analysis asked the database for something the snapshot carries"
    moved = [f.name for f, a, b in zip(fixtures, by_database, ml_off)
             if save_rows.differences([a], [b], exact=True)]
    assert moved, "the invented ML state moved no stored value -- this test could not see it lost"
    tracks = [t for r in by_database if r["metadata"]["model"] == "8232-1" for t in r["tracks"]]
    assert tracks and all(t["sigma_threshold"] == 0.0042 for t in tracks), "the ML threshold ruled"
    assert any(t["failure_probability"] is not None for t in tracks), "the predictor ran"


def test_a_failed_ml_load_warns_once_per_folder_and_the_folder_still_runs(
        tmp_path, monkeypatch, caplog, fresh_ml_cache):
    """Ruling of 2026-09-25 (review m-3): a failed ML load WARNS, once per folder, naming what
    failed and that the thresholds fall back to the defaults -- and does NOT refuse the folder.
    (Predictors never load on the Mac by design; sigma is a drift signal, never a rejection.)"""
    import laser_trim_analyzer.ml as ml_pkg
    from laser_trim_analyzer.core.ingest_run import run_folders
    from laser_trim_analyzer.database.manager import DatabaseManager

    def cannot(db_, *a, **k):
        raise RuntimeError("invented: the ML state cannot be read")

    monkeypatch.setattr(ml_pkg, "get_shared_ml_manager", cannot)
    db = DatabaseManager(tmp_path / "run.db")
    save_rows.inject(db, monkeypatch)
    src = REPO / "tests" / "fixtures" / "trim" / "dlts_8232-1_242.xls"
    folders = [tmp_path / "laser 1", tmp_path / "laser 2"]
    for folder, serials in zip(folders, ((901, 902), (903,))):   # 3 files: per FILE would say 3
        folder.mkdir()
        for s in serials:
            shutil.copyfile(src, folder / f"dlts_8232-1_{s}.xls")
    with caplog.at_level("WARNING"):
        report = run_folders([str(f) for f in folders], db=db, config=None, incremental=True)
    assert [r.ok for r in report.results] == [True, True], [r.error for r in report.results]
    said = [r.getMessage() for r in caplog.records
            if r.levelname == "WARNING" and "ML state could not be loaded" in r.getMessage()]
    assert len(said) == 2, said                        # once per FOLDER: not per file, not never
    assert all("invented: the ML state cannot be read" in m and "fall back to the formula "
               "defaults" in m for m in said), said
    con = sqlite3.connect(f"file:{db.database_path}?mode=ro", uri=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0] == 3
    finally:
        con.close()


def test_a_manager_that_could_not_read_its_state_is_named_too(tmp_path, caplog, fresh_ml_cache):
    """The ML manager swallows its own failure to read model_ml_state (and the shared cache then
    serves the empty manager for five minutes); load_ml_state is where it is said."""
    from laser_trim_analyzer.core.processor import load_ml_state
    from laser_trim_analyzer.database.manager import DatabaseManager
    db = DatabaseManager(tmp_path / "ml.db")
    try:
        con = sqlite3.connect(str(tmp_path / "ml.db"))
        con.execute("DROP TABLE model_ml_state")        # invented breakage
        con.commit()
        con.close()
        with caplog.at_level("WARNING"):
            assert load_ml_state(db) == ({}, {})
    finally:
        db.close()
    said = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("ML state could not be loaded" in m and "model_ml_state could not be read" in m
               for m in said), said


def test_a_spec_edited_mid_run_takes_effect_at_the_next_folder(tmp_path, monkeypatch):
    """Ruling 16: the snapshot is taken at FOLDER start. A spec edited while folder 1 runs (here,
    right after its first file is analysed -- when the old per-file lookup would have seen it)
    does not reach folder 1's second file -- it takes effect at folder 2. The 8232-1 angle spec
    is what moves the stored numbers."""
    from laser_trim_analyzer.core.ingest_run import run_folders
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database.manager import DatabaseManager
    db = DatabaseManager(tmp_path / "run.db")
    save_rows.inject(db, monkeypatch)
    db.save_model_spec({"model": "8232-1", "linearity_type": "independent"})
    src = REPO / "tests" / "fixtures" / "trim" / "dlts_8232-1_242.xls"
    one, two = tmp_path / "laser 1", tmp_path / "laser 2"
    for folder, serials in ((one, (901, 902)), (two, (903,))):
        folder.mkdir()
        for s in serials:
            shutil.copyfile(src, folder / f"dlts_8232-1_{s}.xls")
    real_analyse = Processor.analyse_path
    edited = []

    def analyse_then_edit(self, file_path, disk_stat=None):
        out = real_analyse(self, file_path, disk_stat)
        if not edited:
            edited.append(out.result.metadata.serial)
            db.save_model_spec({"model": "8232-1", "electrical_angle": 340.0,
                                "electrical_angle_tol": 5.0,
                                "electrical_angle_tol_type": "bilateral"})
        return out

    monkeypatch.setattr(Processor, "analyse_path", analyse_then_edit)
    report = run_folders([str(one), str(two)], db=db, config=None, incremental=True)
    assert [r.ok for r in report.results] == [True, True], [r.error for r in report.results]
    con = sqlite3.connect(f"file:{db.database_path}?mode=ro", uri=True)
    try:
        fail_points = dict(con.execute(
            "SELECT a.serial, t.linearity_fail_points FROM analysis_results a "
            "JOIN track_results t ON t.analysis_id = a.id").fetchall())
    finally:
        con.close()
    assert len(edited) == 1 and edited[0] in ("901", "902"), edited   # folder 1's first file
    assert fail_points == {"901": 5, "902": 5, "903": 1}, fail_points


def test_a_folder_whose_specs_cannot_be_read_fails_by_name(tmp_path, monkeypatch):
    """Without its snapshot a folder would be analysed spec-less -- different stored numbers,
    without a word (spec §4.3). So a folder whose specs cannot be read at its start fails, and says
    why, instead of running."""
    from laser_trim_analyzer.core.ingest_run import run_folder
    from laser_trim_analyzer.database.manager import DatabaseManager
    db = DatabaseManager(tmp_path / "run.db")
    save_rows.inject(db, monkeypatch)
    folder = tmp_path / "laser"
    folder.mkdir()
    shutil.copyfile(REPO / "tests" / "fixtures" / "trim" / "dlts_8232-1_242.xls",
                    folder / "dlts_8232-1_242.xls")

    def unreadable(self):
        raise RuntimeError("invented: the model_specs table cannot be read")

    monkeypatch.setattr(DatabaseManager, "get_all_model_specs", unreadable)
    res = run_folder(str(folder), db=db, config=None)
    assert res.ok is False and "model specs" in res.error and "cannot be read" in res.error
    with db.session() as s:
        from sqlalchemy import text
        assert s.execute(text("SELECT COUNT(*) FROM analysis_results")).scalar() == 0
