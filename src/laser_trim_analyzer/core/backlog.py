"""Read an open-order backlog export into what the app needs -- and nothing else.

One file answers two questions Settings used to ask separately: which models are
ACTIVE (they have open orders) and what each one SELLS for. Rules, from James
(2026-09-20):

  * price   = the unit price on the model's LATEST order ("that's the current price")
  * add-ons = items like "1844205 FAI", "6655-10 LAT", "... TEST UNITS" are separate
              line items, not the model: only an Item ID that EXACTLY matches a model
              the app knows counts. Nothing is folded into a base model.
  * upload  = replaces the previous backlog's list entirely.

The export carries customer names and PO numbers. They are never read into the
result: only model, open quantity, price and the earliest need date leave here.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd

ITEM_COLS = ("item id", "item_id", "itemid", "item", "model", "part", "part number")
BALANCE_COLS = ("balance", "open qty", "open quantity", "qty open", "backlog qty")
PRICE_COLS = ("unit price", "unit_price", "unitprice", "price")
ORDER_DATE_COLS = ("order date", "order_date", "so date", "date ordered")
NEED_DATE_COLS = ("need date", "need_date", "due date", "required date")


class BacklogFormatError(ValueError):
    """The file is not a backlog this reader understands; the message says what is missing."""


@dataclass
class Backlog:
    models: List[str] = field(default_factory=list)            # known models with an open balance
    prices: Dict[str, float] = field(default_factory=dict)     # model -> unit price on its latest order
    open_qty: Dict[str, int] = field(default_factory=dict)     # model -> open units
    earliest_need: Dict[str, str] = field(default_factory=dict)  # model -> ISO date
    unmatched: List[str] = field(default_factory=list)         # open items that are not known models
    open_lines: int = 0
    open_units: int = 0
    matched_units: int = 0


def _find(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    return next((lookup[n] for n in names if n in lookup), None)


def _item_id(value) -> str:
    """'8506', 8506 and 8506.0 are the same item; keep everything else verbatim (trimmed)."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def parse_backlog(df: pd.DataFrame, known_models: Iterable[str]) -> Backlog:
    item_c, bal_c, price_c = _find(df, ITEM_COLS), _find(df, BALANCE_COLS), _find(df, PRICE_COLS)
    missing = [label for label, col in (("an Item ID column", item_c), ("a Balance (open quantity) column", bal_c),
                                        ("a Unit Price column", price_c)) if col is None]
    if missing:
        raise BacklogFormatError("This does not look like a backlog export. It is missing " + " and ".join(missing)
                                 + f". Columns found: {', '.join(str(c) for c in df.columns)}")
    order_c, need_c = _find(df, ORDER_DATE_COLS), _find(df, NEED_DATE_COLS)
    known = {str(m).strip() for m in known_models}

    work = pd.DataFrame({
        "item": df[item_c].map(_item_id),
        "balance": pd.to_numeric(df[bal_c], errors="coerce"),
        "price": pd.to_numeric(df[price_c], errors="coerce"),
        "ordered": pd.to_datetime(df[order_c], errors="coerce") if order_c else pd.NaT,
        "need": pd.to_datetime(df[need_c], errors="coerce") if need_c else pd.NaT,
        "row": range(len(df)),
    })
    work = work[(work["item"] != "") & (work["balance"] > 0)]      # only lines with something still open
    out = Backlog(open_lines=len(work), open_units=int(work["balance"].sum()) if len(work) else 0)
    out.unmatched = sorted(set(work["item"]) - known)
    work = work[work["item"].isin(known)]
    out.matched_units = int(work["balance"].sum()) if len(work) else 0
    for model, lines in work.groupby("item"):
        out.models.append(model)
        out.open_qty[model] = int(lines["balance"].sum())
        priced = lines[lines["price"] > 0]
        if len(priced):
            # Latest ORDER wins; undated lines sort first; the file's own row order breaks ties.
            latest = priced.sort_values(["ordered", "row"], na_position="first").iloc[-1]
            out.prices[model] = float(latest["price"])
        needs = lines["need"].dropna()
        if len(needs):
            out.earliest_need[model] = needs.min().date().isoformat()
    out.models.sort()
    return out
