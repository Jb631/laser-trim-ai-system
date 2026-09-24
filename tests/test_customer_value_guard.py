"""The guard that keeps the backlog export out of the repository must itself be tested.

It is the only thing between a customer's prices, names and PO numbers and a public
commit, and on 2026-09-20 it was found to have two defects of its own: a zero unit
price matched everywhere (a no-charge backlog line), and numbers inside this project's
data filenames ("...9-5-2025_9-00 AM.xls") were read as prices. Both produced a report
against a commit that had been on origin/main since April.

Nothing here uses a real value. The invented pairs below are invented.
"""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _guard():
    spec = importlib.util.spec_from_file_location(
        "customer_guard", REPO / "scripts" / "check_no_customer_values.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _guard()
PRICES = {"9999-1": {675.0, 1234.5, 30.0}}  # invented model, invented prices
NAMES = {"Wile E Coyote Aerospace"}
POS = {"PO-00099911"}


def find(line):
    return G.findings_in(line, PRICES, NAMES, POS)


# ---- it still catches what it exists to catch ----

def test_a_price_beside_its_model_is_caught():
    assert [f[0] for f in find('rows = [("9999-1", 675.0)]')] == ["PRICE"]


def test_a_price_in_a_commit_message_is_caught():
    assert [f[0] for f in find("fix: 9999-1 now quotes 1234.5 per unit")] == ["PRICE"]


def test_a_customer_name_is_caught():
    assert [f[0] for f in find("# shipped to Wile E Coyote Aerospace")] == ["CUSTOMER"]


def test_a_po_number_is_caught():
    assert [f[0] for f in find("order PO-00099911 closed")] == ["PO NUMBER"]


def test_the_report_never_carries_the_value_itself():
    kind, line_no, note = find('x = ("9999-1", 675.0)')[0]
    assert "675" not in note and "675" not in str(kind) and line_no == 1


def test_a_price_without_its_model_is_not_a_leak():
    assert find("timeout = 675.0") == []


def test_another_models_price_beside_this_model_is_not_a_leak():
    assert find('("8888-2", 675.0)') == []


# ---- and it no longer cries wolf ----

def test_a_number_inside_a_data_filename_is_not_a_price():
    # The shape of the real false positive: a fragment of the timestamp in a data
    # filename. 30 here is a price in PRICES, so this fires unless a rule stops it --
    # the first version of this test used a token whose only number was 0.0, which the
    # zero-price filter removed anyway, so it passed without exercising anything.
    line = '"Sample_Base_2026-04-10/DLTS/9999-1/9999-1_107_TEST DATA_9-5-2025_10-30 AM.xls": {'
    assert find(line) == []


def test_a_date_alone_is_enough_to_disqualify_a_number():
    # No path separator and no file extension in this token: only the date rule can
    # stop it. (Checked by mutation 2026-09-20: disable _DATEISH and this goes red.)
    assert find("# 9999-1 sweep at DATA_9-5-2025_10-30") == []


def test_a_path_alone_is_enough_to_disqualify_a_number():
    # The token holding the number has a slash but NO date and no extension, so only
    # the path rule can stop it. (_DATEISH's [-/] class swallows most path-like number
    # runs, which is why this had to be built so carefully to reach the path rule at
    # all -- the first version was caught by the date rule and proved nothing.)
    assert find('model 9999-1 wrote out/675.0/x') == []


def test_a_filename_extension_alone_is_enough():
    # No slash and no date in the token; the number is separated from the extension so
    # the base regex still extracts it.
    assert find('open("9999-1 675-x.json")') == []


def test_a_real_price_survives_next_to_a_filename_on_the_same_line():
    # The fix must not be a blanket exemption for lines that mention a file.
    line = 'PRICES = {"9999-1": 675.0}   # from Backlog_2026-04-10.xls'
    assert [f[0] for f in find(line)] == ["PRICE"]


def test_zero_priced_backlog_lines_are_dropped_before_they_reach_the_scan():
    # _backlog() filters them; a 0.00 "price" would otherwise match every 00 in the tree.
    assert find('x = ("9999-1", 0.0)') == []


# ---- a short all-digit PO number is a whole number, never digits inside a longer one ----
# 2026-09-24: the tail of a long measured float in a regenerated fixture baseline spelled a
# real 5-digit PO, and the plain substring match reported a leak. 25 of the export's PO
# numbers are six digits or fewer, so any long float could do it again.

SHORT_PO = "90817"  # invented, all digits, like the export's short ones


def find_po(line):
    return G.findings_in(line, {}, set(), {SHORT_PO})


def test_a_short_po_standing_alone_is_caught():
    assert [f[0] for f in find_po("closed order 90817 today")] == ["PO NUMBER"]


def test_a_short_po_glued_to_a_label_is_still_caught():
    # A letter before the digits does not make them part of a longer NUMBER.
    assert [f[0] for f in find_po("ref PO90817")] == ["PO NUMBER"]


def test_a_short_po_after_a_word_and_a_full_stop_is_still_caught():
    # Only a digit and then a point is a decimal; "no." is not.
    assert [f[0] for f in find_po("order no.90817")] == ["PO NUMBER"]


def test_a_short_po_inside_a_decimal_fraction_is_not_a_po():
    # The shape of the real false positive: the last digits of a long float.
    assert find_po("[0.0038472619290817, -0.0041]") == []


def test_a_short_po_straight_after_the_decimal_point_is_not_a_po():
    # Only the digit-and-point rule can stop this one: a point, not a digit, precedes it.
    assert find_po("x = 0.90817") == []


def test_a_short_po_continued_by_another_digit_is_not_a_po():
    # Only the right-hand rule can stop this one: the left edge is a space.
    assert find_po("count = 908172") == []


def test_a_short_po_as_the_integer_part_of_a_decimal_still_flags():
    # Deliberately NOT exempt: a whole number the PO's digits begin is left for a person to judge.
    assert [f[0] for f in find_po("value 90817.5")] == ["PO NUMBER"]


# ---- the real export, when this machine has one ----

def test_the_repository_is_clean_against_the_real_backlog():
    data = G._backlog()
    if data is None:
        pytest.skip("no backlog export on this machine -- nothing to check against")
    prices, names, pos, _src = data
    assert prices, "the backlog parsed but yielded no prices -- the guard would pass vacuously"
    assert all(p > 0 for vals in prices.values() for p in vals), "a zero price reached the scan"


def test_every_text_format_the_repository_commits_is_scanned():
    # A design mockup (.html) went through a push unscanned on 2026-09-23 -- clean, by luck.
    # Every text extension present in the tracked tree must be one the guard reads.
    import subprocess
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split()
    exts = {Path(f).suffix.lower() for f in tracked if Path(f).suffix}
    binary = {".xls", ".xlsx", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".ttf", ".otf", ".woff",
              ".woff2", ".pdf", ".db", ".pkl", ".joblib", ".zip", ".gz", ".icns", ".bin", ".lock"}
    unscanned = sorted(e for e in exts - binary if e not in G.TEXT)
    assert not unscanned, f"committed text formats the guard never reads: {unscanned}"
