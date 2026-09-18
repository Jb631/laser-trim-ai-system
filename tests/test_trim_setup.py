import pandas as pd
import pytest
from laser_trim_analyzer.core.trim_setup import (
    normalise_key, read_keyvalue, read_per_pass, PROMOTED)

LTS = "tests/fixtures/trim/lts_8232-1_194.xls"
DLTS = "tests/fixtures/trim/dlts_8232-1_243.xls"


def test_normalise_key_is_stable_across_wording():
    assert normalise_key("Initial Resistance Upper Limit") == "initial_resistance_upper_limit"
    assert normalise_key("Laser Power (0-255)") == "laser_power"
    assert normalise_key("  Min Resistance ") == "min_resistance"


def test_system_b_model_parameters_are_value_first():
    df = pd.read_excel(LTS, sheet_name="Model Parameters", header=None)
    kv = read_keyvalue(df, label_col=1, value_col=0)
    assert kv["model_number"] == "8232-1"
    assert kv["max_resistance"] == 5500
    assert kv["min_resistance"] == 5000
    assert kv["theo_resistance"] == 5250
    assert kv["initial_points_ignored"] == 2
    assert kv["ending_points_ignored"] == 7


def test_system_a_track_parameters_are_label_first():
    df = pd.read_excel(DLTS, sheet_name="Track Parameters", header=None)
    kv = read_keyvalue(df, label_col=0, value_col=1)
    assert kv["initial_resistance_upper_limit"] == 4600
    assert kv["initial_resistance_lower_limit"] == 4200
    assert kv["final_resistance_upper_limit"] == 5500
    assert kv["final_resistance_lower_limit"] == 5000
    assert kv["test_voltage"] == 10
    assert kv["indexing_method"] == "ERROR-SPLIT"


def test_system_a_trim_parameters_give_one_dict_per_pass():
    df = pd.read_excel(DLTS, sheet_name="Trim Parameters", header=None)
    passes = read_per_pass(df)
    assert len(passes) == 2
    assert passes[0]["label"] == "SEC1-TRK1-TRM1"
    assert passes[0]["laser_cut_length_mm"] == 0.75
    assert passes[1]["laser_cut_length_mm"] == 0.88
    assert passes[0]["laser_speed_high"] == 0.35
    assert passes[1]["laser_speed_high"] == 0.25


def test_promoted_keys_all_exist_in_a_real_sheet():
    """Every key we promote to a real column must be readable from a real file,
    or the column is dead weight."""
    a = read_keyvalue(pd.read_excel(DLTS, sheet_name="Track Parameters", header=None),
                      label_col=0, value_col=1)
    b = read_keyvalue(pd.read_excel(LTS, sheet_name="Model Parameters", header=None),
                      label_col=1, value_col=0)
    seen = set(a) | set(b)
    missing = [k for k in PROMOTED if k not in seen]
    assert not missing, f"promoted but never seen in a real file: {missing}"


def test_unreadable_sheet_yields_empty_not_an_exception():
    assert read_keyvalue(pd.DataFrame(), label_col=0, value_col=1) == {}
    assert read_per_pass(pd.DataFrame()) == []


def test_a_datetime_cell_does_not_take_the_whole_block_down():
    """One unserialisable cell used to cost the ENTIRE parameter block.

    `trim_setup.parameters` is one SafeJSON column. SafeJSON test-serialises
    and, on TypeError, logs and substitutes None -- so a single `Template
    Updated` datetime wrote the whole block as the literal string "null",
    which reads back as `[]`: a list where a dict was promised, so a consumer
    calling `.get()` raises AttributeError. 44 of 522 real DLTS files across
    23 models, with the promoted columns landing correctly the whole time.
    """
    import json
    from datetime import datetime
    df = pd.DataFrame([
        ["Template Updated:", datetime(2026, 1, 6)],
        ["Laser Power (0-255)", 28],
        ["Initial Resistance Upper Limit", 11000.0],
        ["Indexing Method", "Rotary"],
    ])
    got = read_keyvalue(df, label_col=0, value_col=1)
    assert got["template_updated"] == "2026-01-06T00:00:00"
    assert got["laser_power"] == 28
    assert got["initial_resistance_upper_limit"] == 11000.0
    json.dumps(got)          # the whole point: this used to raise


def test_a_date_in_the_label_column_is_not_a_key():
    """_clean coerces VALUES so they survive JSON. Labels must not go through
    it: `8251-1`'s value-first sheet has the date in the label column, and
    coercing it invented the key '2026_01_06t00_00_00'."""
    from datetime import datetime
    df = pd.DataFrame([[datetime(2026, 1, 6), "Template Updated:"],
                       [28, "Laser Power (0-255)"]])
    got = read_keyvalue(df, label_col=1, value_col=0)
    assert got == {"template_updated": "2026-01-06T00:00:00", "laser_power": 28}


def test_the_real_file_that_found_this_stores_a_usable_dict():
    import json
    from pathlib import Path
    from laser_trim_analyzer.core.parser import ExcelParser
    src = Path("tests/fixtures/trim_setup/8251-1_29_template_updated.xls")
    assert src.exists(), "the fixture is tracked; this is a broken checkout"
    setup = ExcelParser().parse_file(src)["trim_setup"]
    assert isinstance(setup, dict) and len(setup) > 3, setup
    assert isinstance(setup["template_updated"], str)
    json.dumps(setup)
    # Not "starts with a digit": `8251_1` and `05bf8251_1` are real labels on
    # this sheet (the model number, with the value "Model"). The shape being
    # ruled out is a key derived from a coerced DATE.
    import re
    dated = [k for k in setup if re.match(r"^\d{4}_\d{2}_\d{2}", k)]
    assert not dated, f"a date became a key: {dated}"
    assert len(setup) == 54, f"{len(setup)} keys, expected 54"
