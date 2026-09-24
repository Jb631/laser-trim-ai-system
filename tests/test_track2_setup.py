"""Track 2's own setup, on two-track System A (laser 2 DLTS / laser 3 LTS3) files.

`core/trim_setup.read_keyvalue` has only ever read column B (`value_col=1`) of the
'Track Parameters' sheet. On a two-track file, column C carries TRACK 2's entire own
setup block under the same labels -- its own resistance limits, laser power, alias, etc.
(see the research brief, `research-dlts-unread-sheets.md` Sec.4). Dropping it meant
`findings/data.py` judged a TRK2 track against TRACK 1's resistance limits. Fixed
2026-09-24: `trim_setup.read_track2_keyvalue` reads column C when it is a REAL Track 2
block (never a guess -- see its docstring), `core/parser.py` carries it as
`trim_setup["_track2"]`, `_write_trim_setup` stores it in a new, nullable
`trim_setup.track2_parameters` column (NOT a new promoted column -- `trim_setup` stays
one row per analysis), and `findings/data.py::load_model_tracks` gives a TRK2 track its
own limits from it when present. No back-fill: existing two-track analyses are judged
against Track 1's limits until reprocessed.

Fixtures (both real files, sanitized -- see task-9-report.md for what was checked):
`dlts_8074_18.xls` -- BOTH tracks in one file, with genuinely DIFFERENT initial
resistance limits (Track 1: 10350/11650, Track 2: 4000/6000) -- the strong case.
`dlts_7553_10B.xls` -- only TRK2 is cut in this particular file (its TRK1 sibling is a
separate file, shop 10A, not copied here); this file's 'Track Parameters' sheet still
carries a real Track 2 block, proving the read does not depend on `has_multi_tracks`.
"""
import json
from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa

from laser_trim_analyzer.core.trim_setup import read_track2_keyvalue, resistance_limits

DLTS_BOTH_TRACKS = Path("tests/fixtures/trim/dlts_8074_18.xls")
DLTS_TRK2_ONLY = Path("tests/fixtures/trim/dlts_7553_10B.xls")
DLTS_SINGLE_TRACK = Path("tests/fixtures/trim/dlts_8232-1_242.xls")


# ---------------------------------------------------------------------------
# read_track2_keyvalue -- pure, DataFrame in, dict out. No Excel I/O.
# ---------------------------------------------------------------------------

def _sheet(track1_values, track2_values, header2="SEC1-TRK2", pad=10):
    """A minimal 'Track Parameters'-shaped DataFrame: row 0 is the header row,
    then one (label, track1 value, track2 value) row per entry, padded with
    extra rows so a real block clears `_TRACK2_MIN_VALUES` the same way the
    real ~38-row sheet does."""
    rows = [["TRACK PARAMETERS", "SEC1-TRK1", header2]]
    for i, (v1, v2) in enumerate(zip(track1_values, track2_values)):
        rows.append([f"Field {i}", v1, v2])
    for i in range(pad):
        rows.append([f"Pad {i}", i, i + 1000])
    return pd.DataFrame(rows)


def test_a_real_header_named_block_is_read():
    df = _sheet([11650, 10350, "Outer Track"], [6000, 4000, "Inner Track"])
    out = read_track2_keyvalue(df)
    assert out["field_0"] == 6000
    assert out["field_1"] == 4000
    assert out["field_2"] == "Inner Track"
    # the header cell itself becomes a (harmless) key too, same as read_keyvalue's
    # normal behaviour on column B -- never special-cased away.
    assert out["track_parameters"] == "SEC1-TRK2"


def test_column_c_entirely_blank_is_not_a_track2_block():
    """A genuinely single-track model (e.g. 2475-10): the sheet has a 3rd column
    reserved by the template, but every cell in it -- including the header -- is
    blank."""
    rows = [["TRACK PARAMETERS", "SEC1-TRK1", None]]
    for i in range(15):
        rows.append([f"Field {i}", i, None])
    df = pd.DataFrame(rows)
    assert read_track2_keyvalue(df) == {}


def test_no_column_c_at_all_is_not_a_track2_block():
    df = pd.DataFrame([["TRACK PARAMETERS", "SEC1-TRK1"], ["Field 0", 1]])
    assert read_track2_keyvalue(df) == {}


def test_a_handful_of_stray_comments_is_not_a_track2_block():
    """Verified on the corpus: a few single-track files carry 1-4 stray hand-typed
    comments in column C/D (e.g. 'degrees from end') -- real cells, but not a
    parameter block. Below `_TRACK2_MIN_VALUES` and the header does not name TRK2."""
    rows = [["TRACK PARAMETERS", "SEC1-TRK1", None]]
    for i in range(15):
        rows.append([f"Field {i}", i, None])
    rows[4][2] = "use an odd number for a center at 5v 0 degrees"
    rows[9][2] = 0.5
    df = pd.DataFrame(rows)
    assert read_track2_keyvalue(df) == {}


def test_a_mislabelled_header_with_real_values_is_still_read():
    """Verified on the corpus (e.g. shop 8532-8A): column C's own header cell
    wrongly repeats 'SEC1-TRK1' (a copy-paste in the source workbook), but the
    column still carries a full, genuinely different value set. The header text
    plays no part in the decision at all (ruling, 2026-09-24) -- only the value
    count does, so a wrong, right, or absent header all read identically here."""
    df = _sheet([11650, 10350, "Outer Track"], [6000, 4000, "Inner Track"],
                header2="SEC1-TRK1")
    out = read_track2_keyvalue(df)
    assert out["field_0"] == 6000 and out["field_2"] == "Inner Track"


@pytest.mark.parametrize("track2_values", [
    [],                              # entirely empty under the header
    [0.5, "a comment", 6000],        # a few values -- still below the threshold
], ids=["empty", "a-few-values"])
def test_a_trk2_named_header_with_too_few_values_is_not_a_block(track2_values):
    """Controller's ruling (2026-09-24): the header cell is not part of the
    decision at all. A column C that says 'SEC1-TRK2' at row 0 is not enough on
    its own -- it still has to clear `_TRACK2_MIN_VALUES` real values, the same
    bar a column with no header, or a wrong one, has to clear. Not observed on
    the local corpus (every header-named block there also clears the bar), but
    the header was never trustworthy on its own -- a template's column C could
    say "TRK2" with nothing real underneath, and this must still return {}."""
    rows = [["TRACK PARAMETERS", "SEC1-TRK1", "SEC1-TRK2"]]
    for i in range(15):
        v2 = track2_values[i] if i < len(track2_values) else None
        rows.append([f"Field {i}", i, v2])
    df = pd.DataFrame(rows)
    assert read_track2_keyvalue(df) == {}


def test_first_occurrence_wins_same_as_read_keyvalue():
    df = pd.DataFrame([
        ["TRACK PARAMETERS", "SEC1-TRK1", "SEC1-TRK2"],
        ["Dup", 1, 111],
        ["Dup", 2, 222],
    ] + [[f"Pad {i}", i, i + 1000] for i in range(10)])
    assert read_track2_keyvalue(df)["dup"] == 111


# ---------------------------------------------------------------------------
# resistance_limits -- the four PROMOTED fields, picked out of a raw block.
# ---------------------------------------------------------------------------

def test_resistance_limits_reads_the_four_canonical_fields():
    block = {"initial_resistance_lower_limit": 4000, "initial_resistance_upper_limit": 6000,
             "final_resistance_lower_limit": 12350, "final_resistance_upper_limit": 13650,
             "alias": "Inner Track"}
    assert resistance_limits(block) == {
        "initial_resistance_low": 4000.0, "initial_resistance_high": 6000.0,
        "final_resistance_low": 12350.0, "final_resistance_high": 13650.0}


@pytest.mark.parametrize("empty", [None, {}, []])
def test_resistance_limits_on_nothing_capture_is_all_none(empty):
    out = resistance_limits(empty)
    assert out == {"initial_resistance_low": None, "initial_resistance_high": None,
                    "final_resistance_low": None, "final_resistance_high": None}


def test_resistance_limits_partial_block_leaves_the_rest_none():
    out = resistance_limits({"initial_resistance_lower_limit": 100})
    assert out["initial_resistance_low"] == 100.0
    assert out["initial_resistance_high"] is None
    assert out["final_resistance_low"] is None


def test_resistance_limits_falls_back_to_the_promoted_alias_keys():
    """Same alias precedence `_write_trim_setup` uses: min_resistance/max_resistance
    feed final_resistance_low/high when the canonical labels are absent."""
    out = resistance_limits({"min_resistance": 5, "max_resistance": 6})
    assert out["final_resistance_low"] == 5.0 and out["final_resistance_high"] == 6.0


def test_resistance_limits_ignores_a_non_numeric_value():
    out = resistance_limits({"initial_resistance_lower_limit": "not a number"})
    assert out["initial_resistance_low"] is None


# ---------------------------------------------------------------------------
# Step 1 (task-9-brief.md): the real fixtures through the real pipeline.
# ---------------------------------------------------------------------------

def _process_into(tmp_path, monkeypatch, *files):
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    from laser_trim_analyzer.core.processor import Processor

    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)
    proc = Processor(use_ml=False)
    for f in files:
        db.save_analysis(proc.process_file(f))
    return db


@pytest.mark.skipif(not DLTS_BOTH_TRACKS.exists(), reason="two-track DLTS fixture")
def test_two_track_file_gets_its_own_track2_block_and_parameters_is_unchanged(
        tmp_path, monkeypatch):
    db = _process_into(tmp_path, monkeypatch, DLTS_BOTH_TRACKS)
    with db.session() as s:
        row = s.execute(sa.text(
            "SELECT s.parameters, s.track2_parameters, s.initial_resistance_low, "
            "s.initial_resistance_high FROM trim_setup s "
            "JOIN analysis_results a ON a.id = s.analysis_id "
            "WHERE a.filename = :f"), {"f": DLTS_BOTH_TRACKS.name}).one()
    parameters = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    track2 = json.loads(row[1]) if isinstance(row[1], str) else row[1]

    # parameters (Track 1's block) is untouched by this feature.
    assert "_track2" not in parameters
    assert parameters["initial_resistance_lower_limit"] == 10350
    assert parameters["initial_resistance_upper_limit"] == 11650
    assert row[2] == 10350.0 and row[3] == 11650.0     # the PROMOTED columns: still Track 1's

    # track2_parameters holds Track 2's own block: real resistance limits + alias,
    # genuinely different from Track 1's (not a copy).
    assert track2 is not None
    assert track2["initial_resistance_lower_limit"] == 4000
    assert track2["initial_resistance_upper_limit"] == 6000
    assert track2["alias"] == "Inner Track"
    assert parameters["alias"] == "Outer Track"


@pytest.mark.skipif(not DLTS_SINGLE_TRACK.exists(), reason="single-track DLTS fixture")
def test_single_track_file_stores_a_real_null(tmp_path, monkeypatch):
    db = _process_into(tmp_path, monkeypatch, DLTS_SINGLE_TRACK)
    with db.session() as s:
        row = s.execute(sa.text(
            "SELECT s.track2_parameters, typeof(s.track2_parameters) FROM trim_setup s "
            "JOIN analysis_results a ON a.id = s.analysis_id "
            "WHERE a.filename = :f"), {"f": DLTS_SINGLE_TRACK.name}).one()
    assert row[0] is None
    assert row[1] == "null", "must be a real SQL NULL, not the JSON text 'null'"


@pytest.mark.skipif(not DLTS_BOTH_TRACKS.exists(), reason="two-track DLTS fixture")
def test_load_model_tracks_gives_trk1_and_trk2_their_own_limits(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings.data import load_model_tracks

    db = _process_into(tmp_path, monkeypatch, DLTS_BOTH_TRACKS)
    tracks = {t.track_name: t for t in load_model_tracks(db, "8074")}
    assert set(tracks) == {"TRK1", "TRK2"}
    assert (tracks["TRK1"].initial_r_low, tracks["TRK1"].initial_r_high) == (10350.0, 11650.0)
    assert (tracks["TRK2"].initial_r_low, tracks["TRK2"].initial_r_high) == (4000.0, 6000.0)
    # final limits happen to be equal in this real file -- still each track's OWN
    # reading, not a coincidence of the code path (proven by initial_* above, which
    # genuinely differ).
    assert (tracks["TRK1"].final_r_low, tracks["TRK1"].final_r_high) == (12350.0, 13650.0)
    assert (tracks["TRK2"].final_r_low, tracks["TRK2"].final_r_high) == (12350.0, 13650.0)


def _edit_track2_block(db, filename, edit):
    """Rewrite one analysis's stored Track 2 block in place -- `edit(block)` mutates it."""
    with db.session() as s:
        sid, raw = s.execute(sa.text(
            "SELECT s.id, s.track2_parameters FROM trim_setup s "
            "JOIN analysis_results a ON a.id = s.analysis_id WHERE a.filename = :f"),
            {"f": filename}).one()
        block = json.loads(raw) if isinstance(raw, str) else dict(raw)
        edit(block)
        s.execute(sa.text("UPDATE trim_setup SET track2_parameters = :b WHERE id = :i"),
                  {"b": json.dumps(block), "i": sid})


def _drop_and_set(drop, **values):
    def edit(block):
        block.pop(drop)
        block.update(values)
    return edit


# Track 1's (the file's) pairs on this fixture: initial 10350/11650, final 12350/13650.
# Track 2's own block: initial 4000/6000, final 12350/13650 -- so the final cases give the
# block invented, DIFFERENT finals, or an override could not be told from no override.
@pytest.mark.skipif(not DLTS_BOTH_TRACKS.exists(), reason="two-track DLTS fixture")
@pytest.mark.parametrize("edit, want_initial, want_final", [
    # Only the LOW initial limit: the old per-field override paired Track 2's 4000 with
    # Track 1's 11650 -- a window neither track was ever graded against.
    (_drop_and_set("initial_resistance_upper_limit",
                   final_resistance_lower_limit=12000, final_resistance_upper_limit=14000),
     (10350.0, 11650.0), (12000.0, 14000.0)),
    # Only the HIGH initial limit: per field, Track 1's 10350 over Track 2's 6000 -- an
    # INVERTED window, low above high.
    (_drop_and_set("initial_resistance_lower_limit",
                   final_resistance_lower_limit=12000, final_resistance_upper_limit=14000),
     (10350.0, 11650.0), (12000.0, 14000.0)),
    # The same two cases on the FINAL pair, the initial pair complete (so still Track 2's).
    (_drop_and_set("final_resistance_upper_limit", final_resistance_lower_limit=7000),
     (4000.0, 6000.0), (12350.0, 13650.0)),
    (_drop_and_set("final_resistance_lower_limit", final_resistance_upper_limit=9000),
     (4000.0, 6000.0), (12350.0, 13650.0)),
], ids=["initial-low-only", "initial-high-only", "final-low-only", "final-high-only"])
def test_a_half_pair_in_the_track2_block_keeps_the_files_pair(
        tmp_path, monkeypatch, edit, want_initial, want_final):
    """Track 2's limits replace the file's as PAIRS: a block that gives only one side of
    a window leaves that window exactly as the file has it, never half Track 2's and
    half Track 1's; a complete pair beside it is still Track 2's own."""
    from laser_trim_analyzer.findings.data import load_model_tracks

    db = _process_into(tmp_path, monkeypatch, DLTS_BOTH_TRACKS)
    _edit_track2_block(db, DLTS_BOTH_TRACKS.name, edit)
    tracks = {t.track_name: t for t in load_model_tracks(db, "8074")}
    trk2 = tracks["TRK2"]
    assert (trk2.initial_r_low, trk2.initial_r_high) == want_initial
    assert (trk2.final_r_low, trk2.final_r_high) == want_final
    assert (tracks["TRK1"].initial_r_low, tracks["TRK1"].initial_r_high) == (10350.0, 11650.0)


@pytest.mark.skipif(not DLTS_TRK2_ONLY.exists(), reason="TRK2-only DLTS fixture")
def test_a_file_whose_only_track_is_trk2_still_gets_its_own_block(tmp_path, monkeypatch):
    """Track 1 and Track 2 are cut in SEPARATE files for this shop number (10A/10B);
    only 10B (TRK2) is a fixture here. `has_multi_tracks` is False for this file (it
    carries only TRK2 sweep sheets) -- the capture must not be gated on that flag, or
    this exact, real, common pattern (verified: 63 of 4,871 local files) would silently
    keep reading Track 1's limits for a track that IS Track 2."""
    from laser_trim_analyzer.findings.data import load_model_tracks

    db = _process_into(tmp_path, monkeypatch, DLTS_TRK2_ONLY)
    with db.session() as s:
        row = s.execute(sa.text(
            "SELECT a.has_multi_tracks, s.track2_parameters FROM trim_setup s "
            "JOIN analysis_results a ON a.id = s.analysis_id "
            "WHERE a.filename = :f"), {"f": DLTS_TRK2_ONLY.name}).one()
    assert not row[0], "this fixture is expected to carry only TRK2 sweep sheets"
    track2 = json.loads(row[1]) if isinstance(row[1], str) else row[1]
    assert track2 is not None and track2["alias"] == "Inner Track"

    tracks = load_model_tracks(db, "7553")
    assert [t.track_name for t in tracks] == ["TRK2"]
    assert tracks[0].initial_r_low == 3000.0 and tracks[0].initial_r_high == 4000.0


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_the_migration_adds_the_column_to_an_existing_database(tmp_path):
    """Same pattern as the increment_volts/initial_trim_value migrations: an
    existing DB missing the column gets it added (metadata only)."""
    from laser_trim_analyzer.database import manager as mgr

    db_path = tmp_path / "pre.db"
    db = mgr.DatabaseManager(db_path)
    with db.session() as s:
        s.execute(sa.text("ALTER TABLE trim_setup DROP COLUMN track2_parameters"))
        s.commit()
    db.close()

    db2 = mgr.DatabaseManager(db_path)   # start-up migration must add it back
    with db2.session() as s:
        cols = {r[1] for r in s.execute(sa.text("PRAGMA table_info(trim_setup)"))}
    assert "track2_parameters" in cols
    db2.close()
