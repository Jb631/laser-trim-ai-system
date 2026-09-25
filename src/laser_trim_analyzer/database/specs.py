"""Model-spec resolution: ONE rule, for the database and for the snapshot a worker carries.

The ingest's analysis looks a model spec up per file -- `get_model_spec`, or `resolve_spec_for_ft`
for a final test, whose serial may name a section. A worker PROCESS never opens a database (ingest-
speed ruling 15), so it answers the same questions from a `SpecSnapshot` taken at folder start
(ruling 16). The two must never answer differently, so they share the rule itself -- the functions
below -- and differ only in where the rows come from: `DatabaseManager` hands them the table's rows
through two queries, the snapshot from memory.

One detail makes that sharing load-bearing rather than tidy: an alias is searched in the TABLE'S
ROW ORDER (id order), so when two specs list the same alias the older row answers. A copy of the
rule that searched in model order answered differently whenever that happened.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# A trailing section letter on a final test's serial ('31B', '1004a', '31B '): 8508's sections are
# stored as 8508-A .. 8508-D, while its final-test files say model 8508 and carry the section in
# the serial.
_SECTION_LETTER = re.compile(r'^.*?([A-Za-z])\s*$')


def parse_aliases(aliases_str: Optional[str]) -> List[str]:
    """Pipe-separated aliases as a trimmed list of non-empty tokens."""
    if not aliases_str:
        return []
    return [a.strip() for a in aliases_str.split("|") if a.strip()]


def resolve_model_spec(model: Optional[str],
                       primary: Callable[[str], Optional[Dict[str, Any]]],
                       alias_rows: Callable[[], Iterable[Dict[str, Any]]]
                       ) -> Optional[Dict[str, Any]]:
    """THE model-spec rule (`get_model_spec` and `SpecSnapshot.get_model_spec` alike).

    The model, stripped, matched exactly against the spec rows' `model` (`primary`); failing that,
    the first row -- in id order (`alias_rows`) -- whose pipe-separated aliases contain it
    exactly. None if neither, or for no model at all.
    """
    if not model:
        return None
    model = model.strip()
    row = primary(model)
    if row is not None:
        return row
    for row in alias_rows():
        if model in parse_aliases(row.get("aliases")):
            return row
    return None


def resolve_ft_spec(model: Optional[str], serial: Optional[str],
                    lookup: Callable[[str], Optional[Dict[str, Any]]]
                    ) -> Optional[Dict[str, Any]]:
    """THE final-test rule (`resolve_spec_for_ft` and `SpecSnapshot.resolve_spec_for_ft` alike).

    A serial ending in a letter asks for the section's spec first (`8508` + `31b` -> `8508-B`,
    upper-cased); failing that, or without one, the model's own. `lookup` is the matching
    `get_model_spec`.
    """
    if not model:
        return None
    if serial:
        m = _SECTION_LETTER.match(str(serial))
        if m:
            section_spec = lookup(f"{model}-{m.group(1).upper()}")
            if section_spec:
                return section_spec
    return lookup(model)


@dataclass(frozen=True)
class SpecSnapshot:
    """What the ingest's analysis reads from the database, read ONCE at folder start.

    `specs`: every model_specs row, as `get_model_spec` returns it, kept in id order.
    `ml_thresholds` (model -> sigma threshold) and `ml_predictors` (model -> trained
    ModelPredictor): the ML state a Processor otherwise loads from the database when it is built.

    Plain data -- dicts of plain values, and predictors that pickle -- so it can go to a spawned
    worker process (ingest-speed Task 11). It answers `get_model_spec` and `resolve_spec_for_ft`
    through the same functions `DatabaseManager` does, so it and the database cannot disagree; a
    spec edited while a folder runs reaches the next folder's snapshot.
    """
    specs: Tuple[Dict[str, Any], ...] = ()
    ml_thresholds: Dict[str, float] = field(default_factory=dict)
    ml_predictors: Dict[str, Any] = field(default_factory=dict)
    _by_model: Dict[str, Dict[str, Any]] = field(init=False, repr=False, compare=False)
    _alias_rows: Tuple[Dict[str, Any], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        rows = tuple(sorted((dict(r) for r in self.specs), key=lambda r: r["id"]))
        object.__setattr__(self, "specs", rows)
        object.__setattr__(self, "_by_model", {r["model"]: r for r in rows})
        # The database's alias query: `aliases IS NOT NULL AND aliases != ''`, in id order.
        object.__setattr__(self, "_alias_rows", tuple(r for r in rows if r.get("aliases")))

    def get_model_spec(self, model: Optional[str]) -> Optional[Dict[str, Any]]:
        row = resolve_model_spec(model, self._by_model.get, lambda: self._alias_rows)
        return None if row is None else dict(row)      # a copy, like the database's own

    def resolve_spec_for_ft(self, model: Optional[str],
                            serial: Optional[str]) -> Optional[Dict[str, Any]]:
        return resolve_ft_spec(model, serial, self.get_model_spec)
