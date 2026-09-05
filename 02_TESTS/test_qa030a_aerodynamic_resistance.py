import math
import pytest

from gilttpy.physics.aerodynamic import (
    MOSTAerodynamicResistance,
    businger_dyer_phi_h,
    businger_dyer_psi_h,
    most_aerodynamic_resistance,
)


def test_neutral_log_law_exact():
    got = most_aerodynamic_resistance(
        friction_velocity_m_s=0.4,
        reference_height_m=10.0,
        scalar_roughness_length_m=0.1,
        monin_obukhov_length_m=math.inf,
    )
    expected = math.log(100.0) / (0.4 * 0.4)
    assert got == pytest.approx(expected, rel=0, abs=2e-14)


def test_stable_increases_ra_and_unstable_decreases_ra():
    base = dict(
        friction_velocity_m_s=0.35,
        reference_height_m=10.0,
        scalar_roughness_length_m=0.05,
    )
    neutral = most_aerodynamic_resistance(**base, monin_obukhov_length_m=math.inf)
    stable = most_aerodynamic_resistance(**base, monin_obukhov_length_m=80.0)
    unstable = most_aerodynamic_resistance(**base, monin_obukhov_length_m=-80.0)
    assert unstable < neutral < stable


def test_neutral_continuity_from_both_sides():
    base = dict(
        friction_velocity_m_s=0.5,
        reference_height_m=20.0,
        scalar_roughness_length_m=0.2,
    )
    neutral = most_aerodynamic_resistance(**base, monin_obukhov_length_m=math.inf)
    stable = most_aerodynamic_resistance(**base, monin_obukhov_length_m=1e12)
    unstable = most_aerodynamic_resistance(**base, monin_obukhov_length_m=-1e12)
    assert stable == pytest.approx(neutral, rel=2e-10)
    assert unstable == pytest.approx(neutral, rel=2e-10)


def test_displacement_height_uses_effective_reference_height():
    a = most_aerodynamic_resistance(
        friction_velocity_m_s=0.4,
        reference_height_m=12.0,
        displacement_height_m=2.0,
        scalar_roughness_length_m=0.1,
        monin_obukhov_length_m=math.inf,
    )
    b = most_aerodynamic_resistance(
        friction_velocity_m_s=0.4,
        reference_height_m=10.0,
        displacement_height_m=0.0,
        scalar_roughness_length_m=0.1,
        monin_obukhov_length_m=math.inf,
    )
    assert a == pytest.approx(b, rel=0, abs=1e-14)


def test_ra_scales_exactly_as_inverse_ustar_when_other_inputs_fixed():
    kwargs = dict(
        reference_height_m=10.0,
        scalar_roughness_length_m=0.1,
        monin_obukhov_length_m=-100.0,
    )
    a = most_aerodynamic_resistance(friction_velocity_m_s=0.25, **kwargs)
    b = most_aerodynamic_resistance(friction_velocity_m_s=0.50, **kwargs)
    assert a / b == pytest.approx(2.0, rel=2e-15)


def test_integrated_psi_matches_local_phi_derivative_identity():
    # Integrated MOST correction obeys d psi_h / d ln(z) = 1 - phi_h.
    for zeta in (-1.2, -0.2, 0.2, 1.2):
        eps = 2e-7
        zp = zeta * math.exp(eps)
        zm = zeta * math.exp(-eps)
        derivative = (businger_dyer_psi_h(zp) - businger_dyer_psi_h(zm)) / (2 * eps)
        assert derivative == pytest.approx(1.0 - businger_dyer_phi_h(zeta), rel=3e-8, abs=3e-8)


def test_provider_exposes_dimensionless_stability_without_clipping():
    p = MOSTAerodynamicResistance(
        friction_velocity_m_s=0.3,
        reference_height_m=10.0,
        displacement_height_m=1.0,
        scalar_roughness_length_m=0.1,
        monin_obukhov_length_m=-30.0,
    )
    assert p.zeta_reference == pytest.approx(-0.3)
    assert p.zeta_roughness == pytest.approx(-1.0 / 300.0)
    assert p.resistance_s_m() > 0.0


def test_zero_thickness_transfer_limit_is_zero():
    got = most_aerodynamic_resistance(
        friction_velocity_m_s=0.4,
        reference_height_m=0.1,
        scalar_roughness_length_m=0.1,
        monin_obukhov_length_m=math.inf,
    )
    assert got == 0.0


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        MOSTAerodynamicResistance(
            friction_velocity_m_s=0.4,
            reference_height_m=2.0,
            displacement_height_m=1.95,
            scalar_roughness_length_m=0.1,
        )


def test_no_silent_calm_or_zero_L_clipping():
    with pytest.raises(ValueError):
        MOSTAerodynamicResistance(
            friction_velocity_m_s=0.0,
            reference_height_m=10.0,
            scalar_roughness_length_m=0.1,
        )
    with pytest.raises(ValueError):
        MOSTAerodynamicResistance(
            friction_velocity_m_s=0.4,
            reference_height_m=10.0,
            scalar_roughness_length_m=0.1,
            monin_obukhov_length_m=0.0,
        )
