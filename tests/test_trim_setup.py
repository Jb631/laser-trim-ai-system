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
