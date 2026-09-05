import dataclasses
import math
import pytest

from gilttpy.physics.aerosol_sticking import (
    ZHANG2001_REBOUND_ACTIVATION_DIAMETER_M,
    AerosolSurfaceState,
    Zhang2001ReboundSticking,
    sticking_adjusted_collection_efficiency,
    zhang2001_dry_sticking_fraction,
    zhang2001_sticking_fraction,
)


def dry_surface():
    return AerosolSurfaceState("QA_DRY", False, "QA_REFERENCE")


def wet_surface():
    return AerosolSurfaceState("QA_WET", True, "QA_REFERENCE")


def test_dry_coarse_particle_matches_exact_zhang_slinn_formula():
    st = 4.0
    got = zhang2001_sticking_fraction(
        stokes_number=st,
        particle_diameter_m=10e-6,
        surface_is_wet=False,
    )
    assert got == pytest.approx(math.exp(-math.sqrt(st)), rel=2e-15)
    assert got == pytest.approx(zhang2001_dry_sticking_fraction(stokes_number=st), rel=0, abs=0)


def test_wet_surface_has_no_rebound_for_any_tested_stokes_number_or_size():
    for st in (0.0, 0.01, 1.0, 100.0, math.inf):
        for dp in (0.05e-6, 5e-6, 100e-6):
            assert zhang2001_sticking_fraction(
                stokes_number=st,
                particle_diameter_m=dp,
                surface_is_wet=True,
            ) == 1.0


def test_source_threshold_is_strictly_greater_than_five_micrometres():
    threshold = ZHANG2001_REBOUND_ACTIVATION_DIAMETER_M
    assert threshold == 5.0e-6
    assert zhang2001_sticking_fraction(
        stokes_number=1.0,
        particle_diameter_m=threshold,
        surface_is_wet=False,
    ) == 1.0
    above = zhang2001_sticking_fraction(
        stokes_number=1.0,
        particle_diameter_m=math.nextafter(threshold, math.inf),
        surface_is_wet=False,
    )
    assert above == pytest.approx(math.exp(-1.0), rel=2e-15)


def test_dry_rebound_exact_stokes_limits_are_exposed():
    assert zhang2001_dry_sticking_fraction(stokes_number=0.0) == 1.0
    assert zhang2001_dry_sticking_fraction(stokes_number=math.inf) == 0.0


def test_dry_coarse_sticking_fraction_decreases_monotonically_with_stokes_number():
    values = [
        zhang2001_sticking_fraction(
            stokes_number=st,
            particle_diameter_m=20e-6,
            surface_is_wet=False,
        )
        for st in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
    ]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[0] == 1.0
    assert values[-1] < 1e-4


def test_sticking_adjusted_collection_is_exact_multiplication_only():
    raw = 0.03125
    r1 = math.exp(-math.sqrt(0.5))
    assert sticking_adjusted_collection_efficiency(
        total_collection_efficiency=raw,
        sticking_fraction=r1,
    ) == pytest.approx(raw * r1, rel=2e-15)


def test_surface_wetness_and_provenance_are_explicit_not_inferred_from_rh():
    assert {f.name for f in dataclasses.fields(AerosolSurfaceState)} == {
        "surface_label", "is_wet", "provenance"
    }
    assert wet_surface().is_wet is True
    assert dry_surface().is_wet is False
    with pytest.raises(TypeError):
        AerosolSurfaceState("bad", provenance="QA_REFERENCE")


def test_typed_provider_preserves_explicit_surface_state_and_threshold():
    provider = Zhang2001ReboundSticking(
        surface=dry_surface(),
        rebound_activation_diameter_m=8e-6,
        provenance="QA_CUSTOM_THRESHOLD",
    )
    assert provider.sticking_fraction(stokes_number=4.0, particle_diameter_m=7e-6) == 1.0
    assert provider.sticking_fraction(stokes_number=4.0, particle_diameter_m=9e-6) == pytest.approx(math.exp(-2.0))
    wet_provider = Zhang2001ReboundSticking(surface=wet_surface())
    assert wet_provider.sticking_fraction(stokes_number=100.0, particle_diameter_m=50e-6) == 1.0


def test_threshold_is_not_smoothed_or_hidden():
    threshold = 5e-6
    st = 0.25
    at = zhang2001_sticking_fraction(
        stokes_number=st,
        particle_diameter_m=threshold,
        surface_is_wet=False,
        rebound_activation_diameter_m=threshold,
    )
    above = zhang2001_sticking_fraction(
        stokes_number=st,
        particle_diameter_m=math.nextafter(threshold, math.inf),
        surface_is_wet=False,
        rebound_activation_diameter_m=threshold,
    )
    assert at == 1.0
    assert above == pytest.approx(math.exp(-0.5), rel=2e-15)


def test_invalid_states_fail_explicitly():
    with pytest.raises(ValueError):
        zhang2001_dry_sticking_fraction(stokes_number=-1.0)
    with pytest.raises(ValueError):
        zhang2001_sticking_fraction(stokes_number=1.0, particle_diameter_m=0.0, surface_is_wet=False)
    with pytest.raises(ValueError):
        zhang2001_sticking_fraction(stokes_number=1.0, particle_diameter_m=10e-6, surface_is_wet=False, rebound_activation_diameter_m=0.0)
    with pytest.raises(ValueError):
        zhang2001_sticking_fraction(stokes_number=1.0, particle_diameter_m=10e-6, surface_is_wet=1)
    with pytest.raises(ValueError):
        sticking_adjusted_collection_efficiency(total_collection_efficiency=-1.0, sticking_fraction=1.0)
    with pytest.raises(ValueError):
        sticking_adjusted_collection_efficiency(total_collection_efficiency=1.0, sticking_fraction=1.01)
    with pytest.raises(ValueError):
        AerosolSurfaceState("", False, "QA")
    with pytest.raises(ValueError):
        AerosolSurfaceState("dry", False, "")
