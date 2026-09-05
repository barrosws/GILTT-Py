import dataclasses
import math
import pytest

from gilttpy.physics.aerosol_collection import (
    AerosolCollectionState,
    CollectionEfficiencies,
    Zhang2001CollectionEfficiencies,
    Zhang2001SurfaceCollectionParameters,
    zhang2001_brownian_efficiency,
    zhang2001_impaction_efficiency,
    zhang2001_interception_efficiency,
    zhang2001_smooth_impaction_efficiency,
)


def surface(**overrides):
    base = dict(
        surface_label="QA_REFERENCE_VEGETATION",
        brownian_exponent=2.0 / 3.0,
        impaction_alpha=0.8,
        collector_radius_m=2.0e-3,
        provenance="QA_SOURCE_TAGGED_PARAMETER_SET",
    )
    base.update(overrides)
    return Zhang2001SurfaceCollectionParameters(**base)


def state(**overrides):
    base = dict(
        schmidt_number=1.0e5,
        stokes_number=0.2,
        particle_diameter_m=2.0e-6,
    )
    base.update(overrides)
    return AerosolCollectionState(**base)


def test_brownian_efficiency_matches_exact_zhang_slinn_power_law():
    sc = 1.0e6
    for gamma in (0.5, 0.56, 2.0 / 3.0):
        got = zhang2001_brownian_efficiency(schmidt_number=sc, exponent=gamma)
        assert got == pytest.approx(sc ** (-gamma), rel=2e-15)


def test_brownian_efficiency_has_exact_inverse_schmidt_power_scaling():
    gamma = 0.6
    e1 = zhang2001_brownian_efficiency(schmidt_number=10.0, exponent=gamma)
    e2 = zhang2001_brownian_efficiency(schmidt_number=40.0, exponent=gamma)
    assert e2 / e1 == pytest.approx(4.0 ** (-gamma), rel=2e-15)
    assert e2 < e1


def test_impaction_efficiency_matches_exact_peters_eiden_zhang_form():
    st = 0.2
    alpha = 0.8
    got = zhang2001_impaction_efficiency(stokes_number=st, alpha=alpha)
    assert got == pytest.approx((st / (alpha + st)) ** 2, rel=2e-15)
    assert got == pytest.approx(0.04, rel=2e-15)


def test_impaction_branches_have_correct_zero_and_high_stokes_limits():
    # Vegetated/rough Peters-Eiden branch.
    assert zhang2001_impaction_efficiency(stokes_number=0.0, alpha=0.8) == 0.0
    low = zhang2001_impaction_efficiency(stokes_number=0.1, alpha=0.8)
    high = zhang2001_impaction_efficiency(stokes_number=10.0, alpha=0.8)
    asym = zhang2001_impaction_efficiency(stokes_number=1.0e12, alpha=0.8)
    assert 0.0 < low < high < asym < 1.0
    assert asym == pytest.approx(1.0, rel=2e-12)
    # Smooth Slinn-Slinn branch is distinct: 10**(-3/St).
    assert zhang2001_smooth_impaction_efficiency(stokes_number=0.0) == 0.0
    assert zhang2001_smooth_impaction_efficiency(stokes_number=1.0) == pytest.approx(1e-3)
    assert zhang2001_smooth_impaction_efficiency(stokes_number=3.0) == pytest.approx(0.1)
    assert zhang2001_smooth_impaction_efficiency(stokes_number=1.0e12) == pytest.approx(1.0, rel=1e-11)


def test_interception_efficiency_is_exact_quadratic_in_particle_to_collector_ratio():
    a = 2.0e-3
    e1 = zhang2001_interception_efficiency(particle_diameter_m=1.0e-6, collector_radius_m=a)
    e2 = zhang2001_interception_efficiency(particle_diameter_m=2.0e-6, collector_radius_m=a)
    assert e1 == pytest.approx(0.5 * (1.0e-6 / a) ** 2, rel=2e-15)
    assert e2 / e1 == pytest.approx(4.0, rel=2e-15)


def test_interception_is_scale_invariant_for_equal_dimensionless_geometry():
    e1 = zhang2001_interception_efficiency(particle_diameter_m=2.0e-6, collector_radius_m=2.0e-3)
    e2 = zhang2001_interception_efficiency(particle_diameter_m=2.0e-5, collector_radius_m=2.0e-2)
    assert e1 == pytest.approx(e2, rel=0, abs=0)


def test_source_equations_are_not_silently_clipped_to_probability_bounds():
    # The source algebra itself can exceed one outside its normal atmospheric
    # geometry domain.  QA-030E preserves the equation instead of hiding that
    # condition with min(1,E); domain validity belongs to parameter provenance.
    e = zhang2001_interception_efficiency(particle_diameter_m=2.0, collector_radius_m=1.0)
    assert e == pytest.approx(2.0, rel=0, abs=0)


def test_composite_provider_preserves_mechanism_decomposition_and_exact_sum():
    p = surface()
    s = state()
    provider = Zhang2001CollectionEfficiencies(s, p)
    eff = provider.efficiencies()
    assert eff.brownian == pytest.approx(s.schmidt_number ** (-p.brownian_exponent), rel=2e-15)
    assert eff.impaction == pytest.approx((s.stokes_number / (p.impaction_alpha + s.stokes_number)) ** 2, rel=2e-15)
    assert eff.interception == pytest.approx(0.5 * (s.particle_diameter_m / p.collector_radius_m) ** 2, rel=2e-15)
    assert eff.total == pytest.approx(eff.brownian + eff.impaction + eff.interception, rel=2e-15)


def test_mechanism_limits_are_independent_and_no_rebound_is_hidden():
    p = surface()
    base = Zhang2001CollectionEfficiencies(state(stokes_number=0.0, particle_diameter_m=0.0), p).efficiencies()
    assert base.impaction == 0.0
    assert base.interception == 0.0
    assert base.brownian > 0.0
    assert base.total == pytest.approx(base.brownian)
    # CollectionEfficiencies has only the three capture mechanisms; rebound is QA-030G.
    assert {f.name for f in dataclasses.fields(CollectionEfficiencies)} == {
        "brownian", "impaction", "interception"
    }


def test_parameter_provenance_contract_and_invalid_inputs_fail_explicitly():
    names = {f.name for f in dataclasses.fields(Zhang2001SurfaceCollectionParameters)}
    assert names == {
        "surface_label", "brownian_exponent", "impaction_alpha", "collector_radius_m", "provenance"
    }
    with pytest.raises(ValueError):
        surface(surface_label="")
    with pytest.raises(ValueError):
        surface(provenance="")
    with pytest.raises(ValueError):
        surface(brownian_exponent=0.49)
    with pytest.raises(ValueError):
        surface(brownian_exponent=0.68)
    with pytest.raises(ValueError):
        surface(impaction_alpha=0.0)
    with pytest.raises(ValueError):
        surface(collector_radius_m=0.0)
    with pytest.raises(ValueError):
        state(schmidt_number=0.0)
    with pytest.raises(ValueError):
        state(stokes_number=-1.0)
    with pytest.raises(ValueError):
        state(particle_diameter_m=-1e-6)
