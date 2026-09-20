"""What a finding is, which levers it may name, and how findings are ranked."""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# The ONLY levers a finding may name, with their lead times (spec: "The levers,
# and their lead times"). The ATP linearity and resistance specs are the
# customer's drawing: they are deliberately absent, so no analyzer can ever
# recommend changing them -- Finding() raises on any other key.
LEVERS: Dict[str, Tuple[str, str]] = {
    "laser_settings":    ("Laser settings", "same day"),
    "laser_limit_table": ("Laser limit table", "same day"),
    "ink":               ("Ink formulation (incoming resistance)", "next lot"),
    "deposition":        ("Deposition / upstream", "next lot, or ECN"),
}


@dataclass
class Finding:
    model: str
    analyzer: str
    category: str
    lever: str
    title: str
    summary: str
    systems: Tuple[str, ...]
    n_units: int
    strength_name: str
    strength_value: Optional[float]
    expected_gain_points: Optional[float] = None   # yield points; None = cannot say honestly
    gain_definition: str = ""
    # Tracks a year in the population THIS finding was computed on -- set by the analyzer that
    # claims the gain, because only it knows its own group. A final review demonstrated why it
    # cannot be the model's total: a model with 4,000 tracks on one laser and 300 on another had a
    # finding computed inside the 300 and published a rate off all 4,300, a 14x overstatement.
    # 0 means "this finding does not know its own population", and then NO rate is claimed.
    scope_annual_tracks: int = 0
    annual_volume: int = 0                        # the whole model, for ranking findings that claim no rate
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.lever not in LEVERS:
            raise ValueError(f"{self.lever!r} is not a lever anyone can move; allowed: {sorted(LEVERS)}")
        if self.expected_gain_points is not None and not self.gain_definition:
            raise ValueError("a finding that claims a gain must say how the gain is defined")

    @property
    def lever_label(self) -> str:
        return LEVERS[self.lever][0]

    @property
    def lead_time(self) -> str:
        return LEVERS[self.lever][1]

    @property
    def tracks_per_year(self) -> Optional[float]:
        """Tracks a year the gain is worth, over the population the gain was measured on.

        TRACKS, not units, and deliberately so: a unit trimmed twice is two tracks (1.72 tracks per
        distinct serial in the last year of the work database), and a multi-track part is one track
        per track. Calling them units would overstate the count on both counts.
        """
        if self.expected_gain_points is None or not self.scope_annual_tracks:
            return None
        return self.expected_gain_points / 100.0 * self.scope_annual_tracks

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(lever_label=self.lever_label, lead_time=self.lead_time,
                 tracks_per_year=self.tracks_per_year, systems=list(self.systems))
        return d


def rank(findings: List[Finding]) -> List[Finding]:
    """Recoverable tracks a year first; findings that claim no rate last, by their own size."""
    with_rate = [f for f in findings if f.tracks_per_year is not None]
    without = [f for f in findings if f.tracks_per_year is None]
    with_rate.sort(key=lambda f: (-f.tracks_per_year, f.model, f.analyzer))
    without.sort(key=lambda f: (-f.n_units, f.model, f.analyzer))
    return with_rate + without
