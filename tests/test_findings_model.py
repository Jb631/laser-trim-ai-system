import pytest

from laser_trim_analyzer.findings.model import Finding, LEVERS, rank


def _f(**kw):
    base = dict(model="M", analyzer="a", category="c", lever="ink", title="t", summary="s",
                systems=("A",), n_units=10, strength_name="n", strength_value=1.0)
    base.update(kw); return Finding(**base)

def test_the_atp_spec_can_never_be_a_lever():
    for bad in ("atp", "ATP linearity spec", "customer_spec", "drawing", ""):
        with pytest.raises(ValueError):
            _f(lever=bad)
    assert "atp" not in " ".join(LEVERS).lower()

def test_a_claimed_gain_must_be_defined():
    with pytest.raises(ValueError):
        _f(expected_gain_points=5.0)
    assert _f(expected_gain_points=5.0, gain_definition="x", annual_volume=1000).units_per_year == 50.0

def test_ranking_is_units_per_year_then_volume():
    a = _f(model="A", expected_gain_points=10.0, gain_definition="x", annual_volume=100)    # 10 units/yr
    b = _f(model="B", expected_gain_points=2.0, gain_definition="x", annual_volume=5000)    # 100 units/yr
    c = _f(model="C", annual_volume=9000)                                                   # no gain
    d = _f(model="D", annual_volume=50)
    assert [f.model for f in rank([a, c, d, b])] == ["B", "A", "C", "D"]
