"""'This week's priorities' — models ranked by money leaking at final test.

Final-test failure is the most expensive escape: maximum labor and material are
already in the unit by the time it fails there. So the dollar signal is

    dollar_impact = FT_fail_units × unit_price × cost_ratio

where cost_ratio (config, default 0.5) is the share of unit price lost to a
late failure, and unit_price comes from config.model_prices (Settings →
Pricing). Models with no loaded price rank counts-only (dollar_impact = None) so
they never silently drop off the board — add a price and the dollars light up.
Clicking a row on the dashboard drills into the Model page for the reason
(escapes vs rework vs avoidable trims) and the evidence.

James, 2026-07-14: the prices were loaded the whole time (121 models); V6 just
never consumed them. This is the consumption the V5 dashboard/trends had and the
V6 port dropped.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def compute_cost_priorities(db, model_prices: Optional[Dict[str, float]],
                            cost_ratio: float, recent_days: int = 90,
                            limit: int = 25) -> List[dict]:
    """Rank models by money leaking at final test in the recent window.

    Returns a list of dicts sorted by dollar_impact desc (priced models first,
    then counts-only by FT-fail volume):
        {model, ft_total, ft_fails, ft_fail_rate (%), price, dollar_impact}
    dollar_impact and price are None for models without a loaded price.
    """
    from sqlalchemy import text

    cutoff = datetime.now() - timedelta(days=recent_days)
    prices = {str(k): float(v) for k, v in (model_prices or {}).items()}
    try:
        ratio = float(cost_ratio)
    except (TypeError, ValueError):
        ratio = 0.5

    with db.session() as s:
        rows = s.execute(text("""
            SELECT model,
                   COUNT(*)                                             AS ft_total,
                   SUM(CASE WHEN overall_status='FAIL' THEN 1 ELSE 0 END) AS ft_fails
            FROM final_test_results
            WHERE file_date >= :cutoff
            GROUP BY model"""), {"cutoff": cutoff}).fetchall()

    out: List[dict] = []
    for r in rows:
        model, ft_total, ft_fails = r[0], int(r[1] or 0), int(r[2] or 0)
        if ft_fails <= 0:
            continue
        price = prices.get(str(model))
        dollar = (ft_fails * price * ratio) if price else None
        out.append({
            "model": model,
            "ft_total": ft_total,
            "ft_fails": ft_fails,
            "ft_fail_rate": (100.0 * ft_fails / ft_total) if ft_total else 0.0,
            "price": price,
            "dollar_impact": dollar,
        })
    # Priced models ranked by dollars; counts-only models after, by FT-fail volume.
    out.sort(key=lambda d: (d["dollar_impact"] is None,
                            -(d["dollar_impact"] or 0.0),
                            -d["ft_fails"]))
    return out[:limit]
