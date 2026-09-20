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
    assert _f(expected_gain_points=5.0, gain_definition="x", scope_annual_tracks=1000).tracks_per_year == 50.0

def test_ranking_is_tracks_per_year_then_sample_size():
    a = _f(model="A", expected_gain_points=10.0, gain_definition="x", scope_annual_tracks=100)   # 10 a year
    b = _f(model="B", expected_gain_points=2.0, gain_definition="x", scope_annual_tracks=5000)   # 100 a year
    c = _f(model="C", n_units=9000)                                                              # claims no rate
    d = _f(model="D", n_units=50)
    assert [f.model for f in rank([a, c, d, b])] == ["B", "A", "C", "D"]


def _gain_finding(**kw):
    from laser_trim_analyzer.findings.model import Finding
    base = dict(model="M", analyzer="ink_target", category="Ink target", lever="ink", title="t",
                summary="s", systems=("B",), n_units=300, strength_name="rho", strength_value=-0.2,
                expected_gain_points=5.0, gain_definition="how the gain is defined")
    base.update(kw)
    return Finding(**base)


def test_a_rate_is_only_claimed_over_the_population_the_finding_was_computed_on():
    """Final review, demonstrated: a model with 4,000 tracks on one laser and 300 on another had an
    ink-target finding computed inside the 300-track group and published a rate off the model's whole
    4,300 -- a 14x overstatement. A gain measured on one group may only be scaled by THAT group."""
    f = _gain_finding(scope_annual_tracks=300, annual_volume=4300)
    assert f.tracks_per_year == 15.0                      # 5% of its own 300, not of 4,300
    assert f.to_dict()["tracks_per_year"] == 15.0


def test_a_finding_that_does_not_know_its_own_population_claims_no_rate():
    f = _gain_finding(scope_annual_tracks=0, annual_volume=4300)
    assert f.tracks_per_year is None                      # silence beats a number from the wrong denominator
    assert _gain_finding(expected_gain_points=None, gain_definition="", scope_annual_tracks=300).tracks_per_year is None


def test_findings_that_claim_no_rate_rank_last_by_their_own_size():
    from laser_trim_analyzer.findings.model import rank
    big = _gain_finding(model="BIG", scope_annual_tracks=1000)          # 50 a year
    small = _gain_finding(model="SMALL", scope_annual_tracks=100)       # 5 a year
    none_big = _gain_finding(model="NB", expected_gain_points=None, gain_definition="", n_units=900)
    none_small = _gain_finding(model="NS", expected_gain_points=None, gain_definition="", n_units=50)
    assert [f.model for f in rank([small, none_small, big, none_big])] == ["BIG", "SMALL", "NB", "NS"]
