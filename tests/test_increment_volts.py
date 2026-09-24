"""Laser 1 (LTS) records how each position responds to each laser increment, in its
`TrimVolts N` sheets. The parser never read them (2026-09-23).

Layout, measured on 4,972 real laser-1 files (design doc section 4): one COLUMN per
engaged position, one ROW per laser increment, each cell the output voltage after it.
Column k is the position at data row (Initial Points Ignored + k) of `Trim N`. Zeros are
end padding only. The capture lands on laser 1's `Trim N` pass rows as `increment_volts`
(never `trim_volts`: that key already carries the Trim Parameters' "Trim Volts" SETTING).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

from laser_trim_analyzer.core.trim_passes import (
    XLS_MAX_COLUMNS, increment_volts_frame, read_increment_volts, trimvolts_sheets)

LTS = Path("tests/fixtures/trim/lts_8232-1_193.xls")
LTS_194 = Path("tests/fixtures/trim/lts_8232-1_194.xls")
DLTS = Path("tests/fixtures/trim/dlts_8232-1_243.xls")
# A real laser-1 touch-up (8340-1, 2026) whose file names a trim window that differs from
# its ignored-point counts: Initial/Ending Points Ignored 1/1, Points From Start/End 2/2,
# 123 readings, a 120-column TrimVolts1. Kept out of tests/fixtures/trim, which the noop
# baseline and the sweep's pinned table glob.
PFS = Path("tests/fixtures/trimvolts/lts_8340-1_32.xls")
KEYS = ("increment_volts", "increment_volts_first_row", "increment_volts_truncated")

# Readings per pass, counted cell by cell on the fixtures' own TrimVolts sheets
# (2026-09-24). Pinned, not floored: a reader that stops one row early, or reads the
# zero padding as readings, moves these.
READINGS = {"lts_8232-1_193.xls": {1: 199, 2: 719},
            "lts_8232-1_194.xls": {1: 880, 2: 795}}


def _trim_passes(parsed):
    return [p for t in parsed["tracks"] for p in t.get("trim_passes", [])
            if p["sheet"].lower().startswith("trim ")]


# ---------------------------------------------------------------- the pure reader


def test_zero_padding_is_dropped_and_each_position_keeps_its_readings():
    df = pd.DataFrame([[0.27, 0.40, 0.50],
                       [0.28, 0.41, 0.00],
                       [0.29, 0.00, 0.00]])
    out = read_increment_volts(df, first_row=2, window=3)
    assert out["increment_volts"] == [[0.27, 0.28, 0.29], [0.40, 0.41], [0.50]]
    assert out["increment_volts_first_row"] == 2
    assert out["increment_volts_truncated"] is False


def test_a_sheet_at_the_256_column_limit_is_marked_truncated():
    df = pd.DataFrame([[0.3] * 256])
    assert XLS_MAX_COLUMNS == 256
    assert read_increment_volts(df, first_row=0, window=718)["increment_volts_truncated"] is True
    assert read_increment_volts(df, first_row=0, window=None)["increment_volts_truncated"] is True


def test_a_sheet_narrower_than_its_window_is_marked_truncated():
    """The 1x1 sheets the research found are real, and honest about not covering the window."""
    df = pd.DataFrame([[0.31]])
    out = read_increment_volts(df, first_row=2, window=49)
    assert out["increment_volts"] == [[0.31]]
    assert out["increment_volts_truncated"] is True
    # ...while a sheet that covers its window, or whose window is unknown, is not.
    assert read_increment_volts(df, first_row=2, window=1)["increment_volts_truncated"] is False
    assert read_increment_volts(df, first_row=None, window=None)["increment_volts_truncated"] is False


def test_a_blank_cell_is_never_a_reading():
    df = pd.DataFrame([[0.27, float("nan")], [0.28, float("nan")]])
    assert read_increment_volts(df, first_row=0, window=2)["increment_volts"] == [[0.27, 0.28], []]


def test_text_and_flags_are_never_readings():
    """Mirrors `_col`: a TRUE cell is not a measurement, and neither is a label."""
    df = pd.DataFrame([[0.27, True, "Volts"],
                       [0.28, 0.5, 0.6]], dtype=object)
    out = read_increment_volts(df, first_row=0, window=3)["increment_volts"]
    assert out == [[0.27, 0.28], [], []]
    assert all(isinstance(v, float) for curve in out for v in curve)


def test_the_window_comes_from_the_files_own_model_parameters():
    base = {"initial_points_ignored": 2, "ending_points_ignored": 7, "number_of_readings_lin": 57}
    assert increment_volts_frame(base) == (2, 49)            # 57 - 2 - 7 + 1
    assert increment_volts_frame({**base, "initial_points_ignored": 2.0}) == (2, 49)
    assert increment_volts_frame({**base, "number_of_readings_lin": "57"}) == (2, 49)
    # first_row needs only its own field; the window needs all three.
    assert increment_volts_frame({k: v for k, v in base.items()
                                  if k != "number_of_readings_lin"}) == (2, None)
    assert increment_volts_frame({k: v for k, v in base.items()
                                  if k != "initial_points_ignored"}) == (None, None)
    # Nothing is guessed from a value that is not a whole, non-negative count.
    assert increment_volts_frame({**base, "initial_points_ignored": 2.5}) == (None, None)
    assert increment_volts_frame({**base, "initial_points_ignored": -1}) == (None, None)
    assert increment_volts_frame({**base, "initial_points_ignored": True}) == (None, None)
    assert increment_volts_frame({**base, "ending_points_ignored": "n/a"}) == (2, None)
    assert increment_volts_frame({**base, "number_of_readings_lin": 5}) == (2, None)  # window < 1
    assert increment_volts_frame(None) == (None, None)
    assert increment_volts_frame({}) == (None, None)
    # It runs outside any try in parse_file: nothing a cell can hold may make it raise.
    assert increment_volts_frame({**base, "initial_points_ignored": 10 ** 400}) == (None, None)
    assert increment_volts_frame({**base, "ending_points_ignored": float("inf")}) == (2, None)
    assert increment_volts_frame({**base, "number_of_readings_lin": [57]}) == (2, None)


def test_the_window_is_the_files_points_from_start_and_end_when_it_names_them():
    """Laser 1 starts TrimVolts at Points From Start ("how many points from start to begin
    reading/trimming"), not at Initial Points Ignored, whenever the file carries both
    Points From Start and Points From End -- measured against the machine's own VOLTAGES
    placement on 6,263 of 6,263 local sheets, where the ignored count alone placed 36 passes
    one position off (2026-09-24 review)."""
    ignored = {"initial_points_ignored": 1, "ending_points_ignored": 1,
               "number_of_readings_lin": 123}
    named = {**ignored, "points_from_start": 2, "points_from_end": 2}
    assert increment_volts_frame(named) == (2, 120)             # 123 - 2 - 2 + 1
    assert increment_volts_frame(ignored) == (1, 122)           # today's rule without them
    # One pair or the other, never a mix: both fields, readable as counts, or neither.
    assert increment_volts_frame({**ignored, "points_from_start": 2}) == (1, 122)
    assert increment_volts_frame({**ignored, "points_from_end": 2}) == (1, 122)
    assert increment_volts_frame({**named, "points_from_end": "n/a"}) == (1, 122)
    assert increment_volts_frame({**named, "points_from_start": 2.0}) == (2, 120)
    assert increment_volts_frame({k: v for k, v in named.items()
                                  if k != "number_of_readings_lin"}) == (2, None)
    # The older template's `Start Point` is NOT the trim window: it is never read.
    assert increment_volts_frame({**ignored, "start_point": 5, "end_point": 5}) == (1, 122)


def test_a_sheet_with_no_reading_at_all_is_no_capture():
    """Never (NULL, first_row, truncated): a first_row and a truncated flag for nothing."""
    empty = {"increment_volts": [], "increment_volts_first_row": None,
             "increment_volts_truncated": None}
    assert read_increment_volts(pd.DataFrame(), first_row=2, window=49) == empty
    assert read_increment_volts(pd.DataFrame([[0.0, float("nan")], [0.0, 0.0]]),
                                first_row=2, window=49) == empty
    # One reading anywhere is a capture, with its trailing empty curves kept.
    got = read_increment_volts(pd.DataFrame([[0.3, float("nan")]]), first_row=2, window=2)
    assert got == {"increment_volts": [[0.3], []], "increment_volts_first_row": 2,
                   "increment_volts_truncated": False}


def test_trimvolts_sheets_are_found_by_pass_number_and_an_ambiguous_number_is_not_guessed():
    names = ["Model Parameters", "TrimVolts1", "Trim 1", "trimvolts 2", "Trim 2",
             "VOLTAGES", "TrimVolts3", "TrimVolts 3", "TrimVoltsX"]
    assert trimvolts_sheets(names) == {1: "TrimVolts1", 2: "trimvolts 2"}


# ------------------------------------------------------------- the real laser-1 file


@pytest.mark.parametrize("path", [LTS, LTS_194], ids=lambda p: p.name)
def test_the_real_laser_1_file_carries_its_response_curves(path):
    from laser_trim_analyzer.core.parser import ExcelParser
    assert path.exists(), f"{path} is a tracked fixture"
    parsed = ExcelParser().parse_file(path)
    passes = _trim_passes(parsed)
    assert [p["sheet"] for p in passes] == ["Trim 1", "Trim 2"]
    for p in passes:
        curves = p["increment_volts"]
        # 57 readings - 2 - 7 + 1, measured on BOTH TrimVolts1 and TrimVolts2 of both
        # fixtures 2026-09-24 (the brief measured TrimVolts2 of 193 only).
        assert len(curves) == 49
        assert all(c and c[0] != 0.0 for c in curves)  # row 0 always read
        assert all(v != 0.0 for c in curves for v in c)  # padding never read
        assert p["increment_volts_first_row"] == 2
        assert p["increment_volts_truncated"] is False
        n = int(p["sheet"].split()[-1])
        assert sum(len(c) for c in curves) == READINGS[path.name][n]
    lin = [p for t in parsed["tracks"] for p in t.get("trim_passes", [])
           if p["sheet"].lower().startswith("lin error")]
    assert lin, "the fixture has a Lin Error pass"
    assert all(not (set(KEYS) & set(p)) for p in lin)   # no TrimVolts for Lin Error


def test_the_first_curve_is_the_sheets_first_column_cell_for_cell():
    from laser_trim_analyzer.core.parser import ExcelParser
    passes = _trim_passes(ExcelParser().parse_file(LTS))
    tv2 = pd.read_excel(LTS, sheet_name="TrimVolts2", header=None)
    col0 = tv2.iloc[:, 0].tolist()
    k = next(i for i, v in enumerate(col0) if v == 0.0)
    assert passes[1]["increment_volts"][0] == [float(v) for v in col0[:k]]
    assert passes[1]["increment_volts"][0][:3] == [0.273896, 0.274935, 0.275958]


def test_the_mapping_is_the_one_the_machine_itself_uses():
    """Column k of `TrimVolts N` is the position at data row (first_row + k) of `Trim N`.

    Two proofs. The brief's: the LAST reading of column k tracks `Trim N`'s measured volts
    at that row (r > 0.999). They are NOT the same number -- the live reading during the
    cut vs the verification sweep after it, ratio 0.94-1.23 across the corpus -- so this
    test never asserts equality there. But r alone cannot pin the offset: a straight ramp
    correlates with any shift of itself, and on these fixtures offsets first_row-1 and
    first_row+1 ALSO score r > 0.999 (on 194 pass 2, offset 1 even scores higher).

    So the offset is pinned by the machine's own bookkeeping: its `VOLTAGES` sheet places
    `TrimVolts N` column k's last reading at row (first_row + k), EXACTLY, in the same row
    space as the `test` sweep (VOLTAGES column 0 is `test`'s measured column, exactly) --
    and one row either side it is off by ~0.2 V.
    """
    from laser_trim_analyzer.core.parser import ExcelParser
    passes = _trim_passes(ExcelParser().parse_file(LTS))
    xl = pd.ExcelFile(LTS)
    volts = pd.read_excel(xl, sheet_name="VOLTAGES", header=None)
    test = pd.read_excel(xl, sheet_name="test", header=None)
    rows = volts.shape[0]
    # VOLTAGES rows ARE the position rows: column 0 is the test sweep's measured column.
    assert np.array_equal(volts.iloc[:, 0].to_numpy(dtype=float),
                          test.iloc[:rows, 0].to_numpy(dtype=float))
    for p in passes:
        n = int(p["sheet"].split()[-1])
        fr = p["increment_volts_first_row"]
        last = np.array([c[-1] for c in p["increment_volts"]])
        k = len(last)

        sweep = pd.read_excel(xl, sheet_name=p["sheet"], header=None)
        start = next(i for i in range(20)
                     if isinstance(sweep.iat[i, 4], (int, float)) and sweep.iat[i, 4] == sweep.iat[i, 4])
        # The pass's own positions agree with the rows read here.
        assert p["positions"][fr] == float(sweep.iat[start + fr, 4])
        measured = sweep.iloc[start + fr:start + fr + k, 0].to_numpy(dtype=float)
        assert np.corrcoef(last, measured)[0, 1] > 0.999

        placed = volts.iloc[fr:fr + k, n].to_numpy(dtype=float)
        assert np.array_equal(placed, last), f"{p['sheet']}: VOLTAGES col {n} rows {fr}.."
        for shift in (-1, 1):
            off = volts.iloc[fr + shift:fr + shift + k, n].to_numpy(dtype=float)
            assert np.nanmax(np.abs(off - last)) > 0.1, f"offset {fr + shift} also matched"


def test_a_file_that_names_points_from_start_is_placed_where_the_machine_put_it():
    """The 2026-09-24 review's case, on a real file. Laser 1 put TrimVolts1 column 0 at
    Points From Start (2), not Initial Points Ignored (1) -- its own VOLTAGES sheet says so,
    120 curves of 120 exact at row 2 + k and none at 1 + k -- and the sheet's 120 columns
    are its whole window (123 - 2 - 2 + 1), not a truncation of 122."""
    from laser_trim_analyzer.core.parser import ExcelParser
    assert PFS.exists(), f"{PFS} is a tracked fixture"
    parsed = ExcelParser().parse_file(PFS)
    setup = parsed["trim_setup"]
    assert (setup["initial_points_ignored"], setup["ending_points_ignored"]) == (1, 1)
    assert (setup["points_from_start"], setup["points_from_end"]) == (2, 2)
    passes = _trim_passes(parsed)
    assert [p["sheet"] for p in passes] == ["Trim 1"]
    p = passes[0]
    curves = p["increment_volts"]
    assert len(curves) == 120 and all(curves)
    assert sum(len(c) for c in curves) == 929
    assert p["increment_volts_first_row"] == 2
    assert p["increment_volts_truncated"] is False

    volts = pd.read_excel(PFS, sheet_name="VOLTAGES", header=None)
    last = [c[-1] for c in curves]

    def exact_at(first_row):
        col = volts.iloc[first_row:first_row + len(last), 1].tolist()
        return sum(1 for a, b in zip(last, col) if a == b)

    assert exact_at(2) == 120
    assert exact_at(1) == 0 and exact_at(3) == 0


# ------------------------------------------------------ never at the cost of the pass


def test_laser_2_passes_never_carry_increment_volts():
    from laser_trim_analyzer.core.parser import ExcelParser
    parsed = ExcelParser().parse_file(DLTS)
    passes = [p for t in parsed["tracks"] for p in t.get("trim_passes", [])]
    assert len(passes) == 3
    assert all(not (set(KEYS) & set(p)) for p in passes)


def test_a_trimvolts_read_that_raises_never_costs_the_pass(monkeypatch):
    """The capture sits inside the per-pass try; a failure there must drop only the new keys,
    never the `Trim N` sweep that was already stored before this capture existed."""
    from laser_trim_analyzer.core import trim_passes as tp
    from laser_trim_analyzer.core.parser import ExcelParser
    clean = _trim_passes(ExcelParser().parse_file(LTS))

    def boom(*a, **k):
        raise ValueError("synthetic TrimVolts failure")

    monkeypatch.setattr(tp, "read_increment_volts", boom)
    broken = _trim_passes(ExcelParser().parse_file(LTS))
    assert [p["sheet"] for p in broken] == ["Trim 1", "Trim 2"]
    for before, after in zip(clean, broken):
        assert not (set(KEYS) & set(after))
        assert {k: v for k, v in before.items() if k not in KEYS} == after


def test_a_pass_whose_trimvolts_sheet_is_missing_keeps_everything_else(tmp_path):
    """The touch-up case: a real `Trim 1` with no `TrimVolts1` (1 file in 4,972). Built by
    copying the fixture sheet for sheet into two .xlsx workbooks, one without TrimVolts1,
    so the only difference between them is the missing sheet."""
    from laser_trim_analyzer.core.parser import ExcelParser
    xl = pd.ExcelFile(LTS)
    sheets = {name: pd.read_excel(xl, sheet_name=name, header=None) for name in xl.sheet_names}

    def write(path, drop=()):
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for name, df in sheets.items():
                if name not in drop:
                    df.to_excel(w, sheet_name=name, header=False, index=False)
        return ExcelParser().parse_file(path)

    whole = _trim_passes(write(tmp_path / "lts_8232-1_193_whole.xlsx"))
    missing = _trim_passes(write(tmp_path / "lts_8232-1_193_touchup.xlsx", drop=("TrimVolts1",)))
    assert [p["sheet"] for p in missing] == ["Trim 1", "Trim 2"]
    assert not (set(KEYS) & set(missing[0]))                     # no sheet, no keys
    assert {k: v for k, v in whole[0].items() if k not in KEYS} == missing[0]
    assert missing[1]["increment_volts"] == whole[1]["increment_volts"]
    assert len(missing[1]["increment_volts"]) == 49


# ------------------------------------------------------------------- the database


def _inject(monkeypatch, db):
    """BOTH globals: a Processor must never reach the configured database."""
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)


def test_the_curves_round_trip_through_the_database(tmp_path, monkeypatch):
    from laser_trim_analyzer.core.parser import ExcelParser
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    db = mgr.DatabaseManager(tmp_path / "t.db")
    _inject(monkeypatch, db)
    proc = Processor(use_ml=False)
    for f in (LTS, DLTS):
        db.save_analysis(proc.process_file(f))

    parsed = {p["sheet"]: p for p in _trim_passes(ExcelParser().parse_file(LTS))}
    with db.session() as s:
        rows = s.execute(sa.text(
            "SELECT a.filename, p.sheet, p.increment_volts, p.increment_volts_first_row, "
            "p.increment_volts_truncated, p.recipe FROM trim_passes p "
            "JOIN track_results t ON t.id = p.track_result_id "
            "JOIN analysis_results a ON a.id = t.analysis_id "
            "ORDER BY a.filename, p.pass_index")).all()
    lts_rows = [r for r in rows if r[0] == LTS.name]
    dlts_rows = [r for r in rows if r[0] == DLTS.name]
    assert [r[1] for r in lts_rows] == ["Trim 1", "Trim 2", "Lin Error"]
    assert len(dlts_rows) == 3
    for _, sheet, volts, first_row, truncated, _ in lts_rows[:2]:
        assert json.loads(volts) == parsed[sheet]["increment_volts"]   # exact, float for float
        assert first_row == 2
        assert truncated == 0
    for _, sheet, volts, first_row, truncated, _ in [lts_rows[2]] + dlts_rows:
        assert (volts, first_row, truncated) == (None, None, None), sheet
    # The recipe blob is a column that existed before this capture: it must not grow the keys.
    for r in rows:
        assert not (set(KEYS) & set(json.loads(r[5] or "{}"))), r[1]

    # And the ORM reads it back as the same lists.
    from laser_trim_analyzer.database.models import TrimPass
    with db.session() as s:
        orm = s.query(TrimPass).filter(TrimPass.sheet == "Trim 2").one()
        assert orm.increment_volts == parsed["Trim 2"]["increment_volts"]
        assert orm.increment_volts_first_row == 2 and orm.increment_volts_truncated is False


# The table exactly as the 2026-09-20 rebuild created it (sqlite_master of the work
# database, read-only, 2026-09-24): no increment_volts columns.
_PRE_CAPTURE_TRIM_PASSES = """
CREATE TABLE trim_passes (
    id INTEGER NOT NULL, track_result_id INTEGER NOT NULL, pass_index INTEGER NOT NULL,
    sheet VARCHAR(64), label VARCHAR(64), positions JSON, errors JSON, upper_limits JSON,
    lower_limits JSON, cut_lengths JSON, trim_currents JSON, pred_deltas JSON,
    used_deltas JSON, trim_target JSON, final_trim_value JSON, laser_cut_length FLOAT,
    laser_speed_high FLOAT, laser_speed_low FLOAT, trim_voltage FLOAT,
    trim_upper_tolerance FLOAT, trim_lower_tolerance FLOAT, recipe JSON,
    created_date DATETIME NOT NULL, PRIMARY KEY (id),
    FOREIGN KEY(track_result_id) REFERENCES track_results (id) ON DELETE CASCADE
);
CREATE INDEX idx_trimpass_track ON trim_passes (track_result_id);
CREATE UNIQUE INDEX idx_trimpass_track_idx ON trim_passes (track_result_id, pass_index);
"""


def test_the_start_up_migration_adds_the_columns_to_an_existing_database(tmp_path, monkeypatch):
    import sqlite3
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(_PRE_CAPTURE_TRIM_PASSES)
    conn.execute("INSERT INTO trim_passes (id, track_result_id, pass_index, sheet, positions, "
                 "recipe, created_date) VALUES (1, 999, 1, 'Trim 1', '[1.0, 2.0]', "
                 "'{\"label\": \"pass1\"}', '2026-09-20 14:09:53.805395')")
    conn.commit()
    conn.close()

    db = mgr.DatabaseManager(path)
    _inject(monkeypatch, db)
    conn = sqlite3.connect(path)
    cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(trim_passes)")}
    old = conn.execute("SELECT sheet, positions, recipe, created_date, increment_volts, "
                       "increment_volts_first_row, increment_volts_truncated "
                       "FROM trim_passes WHERE id = 1").fetchone()
    conn.close()
    assert cols["increment_volts"] == "JSON"
    assert cols["increment_volts_first_row"] == "INTEGER"
    assert cols["increment_volts_truncated"] == "BOOLEAN"
    # The row that was already there is untouched, and simply has no capture.
    assert old == ("Trim 1", "[1.0, 2.0]", '{"label": "pass1"}',
                   "2026-09-20 14:09:53.805395", None, None, None)

    # The database recorded WHEN it started capturing, in created_date's own UTC format,
    # so the old row compares as before it -- a string compare is a time compare.
    since = _since(path)
    assert since is not None and len(since) == len("2026-09-20 14:09:53.805395")
    assert old[3] < since

    # Idempotent (the record is not moved by a second start-up), and the migrated table
    # takes a real save whose rows compare as after the record.
    mgr.DatabaseManager(path).close()
    assert _since(path) == since
    db.save_analysis(Processor(use_ml=False).process_file(LTS))
    conn = sqlite3.connect(path)
    got = conn.execute("SELECT COUNT(*) FROM trim_passes WHERE increment_volts IS NOT NULL "
                       "AND increment_volts_first_row = 2 AND created_date >= ?",
                       (since,)).fetchone()[0]
    conn.close()
    assert got == 2


def _since(path):
    import sqlite3
    from laser_trim_analyzer.database.manager import INCREMENT_VOLTS_SINCE_KEY
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT value FROM app_meta WHERE key = ?",
                           (INCREMENT_VOLTS_SINCE_KEY,)).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def test_a_new_database_records_when_it_started_capturing(tmp_path):
    """create_all makes the columns, so the migration's ALTER adds nothing -- the record is
    still written, at the first start-up, before any pass exists."""
    from laser_trim_analyzer.database import manager as mgr
    path = tmp_path / "new.db"
    mgr.DatabaseManager(path).close()
    first = _since(path)
    assert first is not None
    mgr.DatabaseManager(path).close()
    assert _since(path) == first


def test_the_start_is_not_recorded_until_the_column_really_exists(tmp_path, monkeypatch):
    """An ALTER that fails (for any reason but "duplicate column") leaves the column
    missing, and recording a start then would claim a capture the database cannot hold.
    The next start-up that does add the column records it."""
    import sqlite3
    import sqlalchemy
    from laser_trim_analyzer.database import manager as mgr
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(_PRE_CAPTURE_TRIM_PASSES)
    conn.close()

    real_text = mgr.text

    def sabotaged(sql, *a, **k):
        if str(sql).startswith("ALTER TABLE trim_passes ADD COLUMN"):
            return real_text("SELECT no_such_function()")          # fails, not a duplicate
        return real_text(sql, *a, **k)

    monkeypatch.setattr(mgr, "text", sabotaged)
    mgr.DatabaseManager(path).close()
    conn = sqlite3.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trim_passes)")}
    conn.close()
    assert "increment_volts" not in cols
    assert _since(path) is None

    monkeypatch.setattr(mgr, "text", real_text)
    mgr.DatabaseManager(path).close()
    assert _since(path) is not None


def test_a_pass_with_no_curves_stores_three_nulls(tmp_path, monkeypatch):
    """Whatever the pass dict says beside it, no curves means no capture in ALL three columns."""
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    db = mgr.DatabaseManager(tmp_path / "t.db")
    _inject(monkeypatch, db)
    result = Processor(use_ml=False).process_file(LTS)
    trim1 = next(p for t in result.tracks for p in t.trim_passes if p["sheet"] == "Trim 1")
    trim1.update({"increment_volts": [], "increment_volts_first_row": 2,
                  "increment_volts_truncated": True})
    db.save_analysis(result)
    with db.session() as s:
        rows = dict((r[0], r[1:]) for r in s.execute(sa.text(
            "SELECT sheet, increment_volts, increment_volts_first_row, "
            "increment_volts_truncated FROM trim_passes")).all())
    assert rows["Trim 1"] == (None, None, None)
    assert rows["Trim 2"][1:] == (2, 0)        # the pass beside it is untouched
