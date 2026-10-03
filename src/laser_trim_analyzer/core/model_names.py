"""What the shop knows about a model's NAME and its process -- in one place.

MODEL_ALIASES (James, 2026-10-02: "yea you can mearge 7953A with 7953-A. and 7953B with 7953-B"):
the same parts, named both ways by their files -- the only two such twins in the fleet (checked on
the 30 Sep work copy: no other pair of names differs only by a hyphen). Every name a parser reads
and every name a save writes goes through `canonical_model`, so the glued spelling is never stored
again; the rows stored before are renamed once at start-up (database/migrations.py,
`_merge_model_aliases`).

HAND_TRIM_MODELS (James: 8232-1 and 8340-1 are hand-trimmed after the laser; 2026-10-02: "yes
those are hand trim models and we should use 20X"): on these, a laser-stage linearity error up to
20x the spec band is a REAL error -- their ordinary tracks run up to 5-10x -- not a scale fault, so
a file is only "suspect" for its size beyond 20x (`suspect_error_factor`). Add a model here when
James names another one.
"""
from typing import Optional

MODEL_ALIASES = {
    "7953A": "7953-A",
    "7953B": "7953-B",
}

HAND_TRIM_MODELS = frozenset({"8232-1", "8340-1"})

# How many times the spec band a linearity error may be before the file is "suspect" (a scale or
# unit fault, not a measurement -- core/processor.py, `_validate_track_data`).
SUSPECT_ERROR_FACTOR = 10.0
HAND_TRIM_SUSPECT_ERROR_FACTOR = 20.0


def canonical_model(name: Optional[str]) -> Optional[str]:
    """The one stored spelling of a model name (unchanged unless it is a known alias)."""
    if not name:
        return name
    return MODEL_ALIASES.get(name, name)


def suspect_error_factor(model: Optional[str]) -> float:
    """The multiple of the spec band beyond which a linearity error marks its file suspect."""
    if canonical_model(model) in HAND_TRIM_MODELS:
        return HAND_TRIM_SUSPECT_ERROR_FACTOR
    return SUSPECT_ERROR_FACTOR
