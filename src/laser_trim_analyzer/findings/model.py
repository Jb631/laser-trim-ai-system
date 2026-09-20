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
    annual_volume: int = 0
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
    def units_per_year(self) -> Optional[float]:
        if self.expected_gain_points is None:
            return None
        return self.expected_gain_points / 100.0 * self.annual_volume

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(lever_label=self.lever_label, lead_time=self.lead_time,
                 units_per_year=self.units_per_year, systems=list(self.systems))
        return d


def rank(findings: List[Finding]) -> List[Finding]:
    """Recoverable units per year first; findings that cannot state a gain last, by volume."""
    with_gain = [f for f in findings if f.units_per_year is not None]
    without = [f for f in findings if f.units_per_year is None]
    with_gain.sort(key=lambda f: (-f.units_per_year, f.model, f.analyzer))
    without.sort(key=lambda f: (-f.annual_volume, f.model, f.analyzer))
    return with_gain + without
