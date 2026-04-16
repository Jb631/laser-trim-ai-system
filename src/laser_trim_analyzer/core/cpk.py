"""
Cpk/Ppk process capability analysis for linearity data.

Cpk uses within-subgroup variation (short-term capability).
Ppk uses overall variation (long-term performance).
Both require a spec limit — sourced from model_specs.linearity_spec_pct.

For linearity, the spec is symmetric: +/-spec_pct, so:
  USL = +spec_pct, LSL = -spec_pct
  Cpk = min((USL - mean) / (3*sigma), (mean - LSL) / (3*sigma))
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CpkResult:
    """Result of a Cpk/Ppk calculation."""
    cpk: Optional[float] = None
    ppk: Optional[float] = None
    cp: Optional[float] = None
    pp: Optional[float] = None
    mean: Optional[float] = None
    std_within: Optional[float] = None
    std_overall: Optional[float] = None
    usl: Optional[float] = None
    lsl: Optional[float] = None
    n_samples: int = 0
    rating: str = "Unknown"

    def to_dict(self):
        return {
            "cpk": self.cpk, "ppk": self.ppk, "cp": self.cp, "pp": self.pp,
            "mean": self.mean, "std_within": self.std_within,
            "std_overall": self.std_overall, "usl": self.usl, "lsl": self.lsl,
            "n_samples": self.n_samples, "rating": self.rating,
        }


def rate_cpk(cpk_value: Optional[float]) -> str:
    """Rate a Cpk value according to industry standards."""
    if cpk_value is None:
        return "Unknown"
    if cpk_value >= 1.67:
        return "Excellent"
    elif cpk_value >= 1.33:
        return "Capable"
    elif cpk_value >= 1.0:
        return "Marginal"
    else:
        return "Incapable"


def calculate_cpk(
    deviations: List[float],
    spec_limit_pct: float,
    subgroup_size: int = 5,
) -> CpkResult:
    """
    Calculate Cpk and Ppk for a set of linearity deviation values.

    Args:
        deviations: List of linearity deviation values (max error per unit).
        spec_limit_pct: The spec limit as a percentage (e.g., 0.5 for +/-0.5%).
                        Symmetric spec: USL = +spec_limit_pct, LSL = -spec_limit_pct.
        subgroup_size: Size of rational subgroups for within-group sigma estimation.

    Returns:
        CpkResult with Cpk, Ppk, Cp, Pp, and supporting statistics.
    """
    import numpy as np

    result = CpkResult()

    if not deviations or len(deviations) < 2:
        return result

    data = np.array(deviations, dtype=float)
    data = data[~np.isnan(data)]

    if len(data) < 2:
        return result

    result.n_samples = len(data)
    result.usl = spec_limit_pct
    result.lsl = -spec_limit_pct
    result.mean = float(np.mean(data))

    # Overall standard deviation (for Ppk)
    result.std_overall = float(np.std(data, ddof=1))

    # Within-subgroup standard deviation (for Cpk)
    if subgroup_size <= 1 or len(data) < subgroup_size * 2:
        # Moving range method (MR-bar / d2)
        moving_ranges = np.abs(np.diff(data))
        mr_bar = np.mean(moving_ranges)
        d2 = 1.128  # d2 constant for n=2
        result.std_within = float(mr_bar / d2) if mr_bar > 0 else result.std_overall
    else:
        # Pooled standard deviation from subgroups
        n_subgroups = len(data) // subgroup_size
        subgroups = [
            data[i * subgroup_size : (i + 1) * subgroup_size]
            for i in range(n_subgroups)
        ]
        ranges = [np.max(sg) - np.min(sg) for sg in subgroups]
        r_bar = np.mean(ranges)
        d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
                    6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
        d2 = d2_table.get(subgroup_size, 2.326)
        result.std_within = float(r_bar / d2) if r_bar > 0 else result.std_overall

    # Calculate capability indices
    usl = result.usl
    lsl = result.lsl
    mean = result.mean

    if result.std_within and result.std_within > 0:
        result.cp = float((usl - lsl) / (6 * result.std_within))
        cpu = (usl - mean) / (3 * result.std_within)
        cpl = (mean - lsl) / (3 * result.std_within)
        result.cpk = float(min(cpu, cpl))

    if result.std_overall and result.std_overall > 0:
        result.pp = float((usl - lsl) / (6 * result.std_overall))
        ppu = (usl - mean) / (3 * result.std_overall)
        ppl = (mean - lsl) / (3 * result.std_overall)
        result.ppk = float(min(ppu, ppl))

    result.rating = rate_cpk(result.cpk)

    return result


def calculate_cpk_trend(
    deviations_by_period: List[Tuple[str, List[float]]],
    spec_limit_pct: float,
) -> List[dict]:
    """
    Calculate Cpk for each time period to show capability trend.

    Args:
        deviations_by_period: List of (period_label, deviations) tuples.
        spec_limit_pct: The spec limit percentage.

    Returns:
        List of dicts with keys: period, cpk, ppk, n_samples, rating
    """
    results = []
    for period_label, devs in deviations_by_period:
        r = calculate_cpk(devs, spec_limit_pct, subgroup_size=1)
        results.append({
            "period": period_label,
            "cpk": r.cpk,
            "ppk": r.ppk,
            "n_samples": r.n_samples,
            "rating": r.rating,
        })
    return results
