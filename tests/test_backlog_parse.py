import pandas as pd
import pytest

from laser_trim_analyzer.core.backlog import Backlog, BacklogFormatError, parse_backlog

KNOWN = {"8506", "8232-1", "1844205", "7844"}


def _df(rows):
    return pd.DataFrame(rows, columns=["Customer Name", "PO Number", "Order Date", "Need Date",
                                       "Item ID", "Balance", "Unit Price"])


def test_active_models_are_known_items_with_an_open_balance():
    b = parse_backlog(_df([
        ["ACME", "PO1", "2026-01-05", "2026-10-01", "8506", 4, 120.0],
        ["ACME", "PO2", "2026-02-01", "2026-09-15", "8506", 6, 120.0],
        ["BOLT", "PO3", "2026-03-01", "2026-11-01", "8232-1", 10, 410.0],
        ["BOLT", "PO4", "2026-03-02", "2026-11-02", "7844", 0, 99.0],          # nothing left open
        ["CORP", "PO5", "2026-03-03", "2026-11-03", "9999-X", 3, 50.0],        # not a model the app knows
    ]), KNOWN)
    assert b.models == ["8232-1", "8506"]
    assert b.open_qty == {"8232-1": 10, "8506": 10}
    assert b.earliest_need == {"8232-1": "2026-11-01", "8506": "2026-09-15"}
    assert b.unmatched == ["9999-X"]
    assert (b.open_lines, b.open_units, b.matched_units) == (4, 23, 20)


def test_the_price_is_the_latest_orders_not_the_commonest_or_the_average():
    b = parse_backlog(_df([
        ["A", "P", "2025-01-10", "2026-10-01", "8506", 5, 2735.0],
        ["A", "P", "2025-06-10", "2026-10-01", "8506", 5, 2735.0],
        ["A", "P", "2026-08-19", "2026-10-01", "8506", 1, 4195.0],            # the newest order
        ["A", "P", "2025-03-10", "2026-10-01", "8506", 5, 2735.0],
    ]), KNOWN)
    assert b.prices == {"8506": 4195.0}


def test_addons_are_never_counted_and_never_folded_into_the_model():
    b = parse_backlog(_df([
        ["A", "P", "2026-01-01", "2026-10-01", "1844205", 2, 100.0],
        ["A", "P", "2026-06-01", "2026-10-01", "1844205 FAI", 1, 900.0],       # newer, pricier, NOT the model
        ["A", "P", "2026-06-02", "2026-10-01", "1844205 TEST UNITS", 7, 5.0],
    ]), KNOWN)
    assert b.models == ["1844205"]
    assert b.prices == {"1844205": 100.0} and b.open_qty == {"1844205": 2}
    assert b.unmatched == ["1844205 FAI", "1844205 TEST UNITS"]


def test_a_numeric_item_id_is_the_same_item_however_excel_typed_it():
    b = parse_backlog(_df([
        ["A", "P", "2026-01-01", "2026-10-01", 8506, 1, 10.0],
        ["A", "P", "2026-01-02", "2026-10-01", 8506.0, 2, 12.0],
        ["A", "P", "2026-01-03", "2026-10-01", " 8506 ", 3, 14.0],
    ]), KNOWN)
    assert b.models == ["8506"] and b.open_qty == {"8506": 6} and b.prices == {"8506": 14.0}


def test_a_model_with_no_usable_price_is_active_but_unpriced():
    b = parse_backlog(_df([["A", "P", "2026-01-01", "2026-10-01", "7844", 5, 0.0],
                           ["A", "P", "2026-01-02", "2026-10-01", "7844", 5, None]]), KNOWN)
    assert b.models == ["7844"] and b.prices == {}


def test_customer_names_and_po_numbers_never_reach_the_result():
    b = parse_backlog(_df([["SECRET CUSTOMER", "PO-SECRET", "2026-01-01", "2026-10-01", "8506", 1, 10.0]]), KNOWN)
    assert "SECRET" not in repr(b)


def test_a_file_that_is_not_a_backlog_says_what_is_missing():
    with pytest.raises(BacklogFormatError) as e:
        parse_backlog(pd.DataFrame({"Item ID": ["8506"], "foo": [1]}), KNOWN)
    assert "Balance" in str(e.value) and "Unit Price" in str(e.value) and "Item ID column" not in str(e.value)


def test_an_empty_backlog_is_empty_not_an_error():
    b = parse_backlog(_df([]), KNOWN)
    assert b == Backlog()
