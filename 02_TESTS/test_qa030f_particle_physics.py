import dataclasses
import math
import pytest

from gilttpy.physics.aerosol_collection import AerosolCollectionState
from gilttpy.physics.deposition import G, cunningham_slip_correction, stokes_settling_velocity
from gilttpy.physics.particle_physics import (
    BOLTZMANN_CONSTANT_J_K,
    WILLEKE_REFERENCE_MEAN_FREE_PATH_M,
    WILLEKE_REFERENCE_PRESSURE_PA,
    WILLEKE_REFERENCE_TEMPERATURE_K,
    AerosolAirState,
    AerosolParticleProperties,
    AerosolParticleTransportState,
    air_mean_free_path_willeke_m,
    brownian_diffusivity_m2_s,
    particle_relaxation_time_s,
    particle_schmidt_number,
    stokes_settling_velocity_from_relaxation_m_s,
    zhang2001_smooth_stokes_number_corrected,
    zhang2001_vegetated_stokes_number,
)


def particle(dp=1.0e-6, rho=1500.0, basis="dry_geometric", provenance="QA_REFERENCE"):
    return AerosolParticleProperties(dp, rho, basis, provenance)


def air(t=298.15, p=101325.0):
    return AerosolAirState(t, p)


def transport(dp=1.0e-6, rho=1500.0, t=298.15, p=101325.0):
    return AerosolParticleTransportState(particle(dp, rho), air(t, p))


def test_willeke_mean_free_path_reference_and_pressure_scaling_are_exact():
    lam0 = air_mean_free_path_willeke_m(
        temperature_k=WILLEKE_REFERENCE_TEMPERATURE_K,
        pressure_pa=WILLEKE_REFERENCE_PRESSURE_PA,
    )
    assert lam0 == pytest.approx(WILLEKE_REFERENCE_MEAN_FREE_PATH_M, rel=2e-15)
    half_p = air_mean_free_path_willeke_m(
        temperature_k=WILLEKE_REFERENCE_TEMPERATURE_K,
        pressure_pa=WILLEKE_REFERENCE_PRESSURE_PA / 2.0,
    )
    assert half_p / lam0 == pytest.approx(2.0, rel=2e-15)


def test_cunningham_path_is_exactly_compatible_with_qa029_kernel():
    tr = transport(dp=0.2e-6)
    got = tr.slip_correction
    expected = cunningham_slip_correction(tr.particle.diameter_m, tr.air.mean_free_path_m)
    assert got == pytest.approx(expected, rel=0, abs=0)
    # Equivalent Zhang/Davies form 1 + 2 lambda/d [1.257 + 0.4 exp(-0.55 d/lambda)]
    dp = tr.particle.diameter_m
    lam = tr.air.mean_free_path_m
    zhang = 1.0 + 2.0 * lam / dp * (1.257 + 0.4 * math.exp(-0.55 * dp / lam))
    assert got == pytest.approx(zhang, rel=2e-15)


def test_brownian_diffusivity_matches_stokes_einstein_cunningham_exactly():
    tr = transport(dp=0.5e-6)
    got = tr.brownian_diffusivity_m2_s
    expected = (
        BOLTZMANN_CONSTANT_J_K
        * tr.air.temperature_k
        * tr.slip_correction
        / (3.0 * math.pi * tr.air.dynamic_viscosity_pa_s * tr.particle.diameter_m)
    )
    assert got == pytest.approx(expected, rel=2e-15)
    direct = brownian_diffusivity_m2_s(
        particle_diameter_m=tr.particle.diameter_m,
        temperature_k=tr.air.temperature_k,
        air_dynamic_viscosity_pa_s=tr.air.dynamic_viscosity_pa_s,
        slip_correction=tr.slip_correction,
    )
    assert direct == pytest.approx(got, rel=0, abs=0)


def test_relaxation_time_and_settling_velocity_identity_and_qa029_equivalence():
    tr = transport(dp=2.0e-6, rho=2000.0)
    tau = tr.relaxation_time_s
    expected_tau = (
        tr.particle.density_kg_m3
        * tr.particle.diameter_m**2
        * tr.slip_correction
        / (18.0 * tr.air.dynamic_viscosity_pa_s)
    )
    assert tau == pytest.approx(expected_tau, rel=2e-15)
    assert tr.settling_velocity_m_s == pytest.approx(G * tau, rel=2e-15)
    qa029 = stokes_settling_velocity(
        tr.particle.diameter_m,
        tr.particle.density_kg_m3,
        tr.air.dynamic_viscosity_pa_s,
        tr.air.mean_free_path_m,
    )
    assert tr.settling_velocity_m_s == pytest.approx(qa029, rel=2e-15)


def test_particle_schmidt_number_is_exact_nu_over_brownian_diffusivity():
    tr = transport(dp=1.0e-6)
    sc = tr.schmidt_number
    assert sc == pytest.approx(
        tr.air.kinematic_viscosity_m2_s / tr.brownian_diffusivity_m2_s,
        rel=2e-15,
    )
    assert particle_schmidt_number(
        air_kinematic_viscosity_m2_s=tr.air.kinematic_viscosity_m2_s,
        brownian_diffusivity_m2_s=tr.brownian_diffusivity_m2_s,
    ) == pytest.approx(sc, rel=0, abs=0)


def test_vegetated_stokes_number_matches_both_vg_and_tau_forms():
    tr = transport(dp=3e-6)
    ustar = 0.4
    radius = 2.0e-3
    st = zhang2001_vegetated_stokes_number(
        settling_velocity_m_s=tr.settling_velocity_m_s,
        friction_velocity_m_s=ustar,
        collector_radius_m=radius,
    )
    assert st == pytest.approx(tr.settling_velocity_m_s * ustar / (G * radius), rel=2e-15)
    assert st == pytest.approx(tr.relaxation_time_s * ustar / radius, rel=2e-15)
    assert tr.stokes_number(
        friction_velocity_m_s=ustar, surface_regime="vegetated", collector_radius_m=radius
    ) == pytest.approx(st, rel=0, abs=0)


def test_smooth_stokes_number_is_dimensionless_corrected_form():
    tr = transport(dp=3e-6)
    ustar = 0.4
    st = zhang2001_smooth_stokes_number_corrected(
        settling_velocity_m_s=tr.settling_velocity_m_s,
        friction_velocity_m_s=ustar,
        air_kinematic_viscosity_m2_s=tr.air.kinematic_viscosity_m2_s,
    )
    assert st == pytest.approx(
        tr.settling_velocity_m_s * ustar**2 / (G * tr.air.kinematic_viscosity_m2_s),
        rel=2e-15,
    )
    assert st == pytest.approx(tr.relaxation_time_s * ustar**2 / tr.air.kinematic_viscosity_m2_s, rel=2e-15)
    assert tr.stokes_number(friction_velocity_m_s=ustar, surface_regime="smooth") == pytest.approx(st)


def test_transport_state_feeds_qa030e_without_hidden_property_reconstruction():
    tr = transport(dp=1.5e-6, rho=1800.0)
    st = tr.stokes_number(
        friction_velocity_m_s=0.35, surface_regime="vegetated", collector_radius_m=2e-3
    )
    collection_state = AerosolCollectionState(
        schmidt_number=tr.schmidt_number,
        stokes_number=st,
        particle_diameter_m=tr.particle.diameter_m,
    )
    assert collection_state.schmidt_number == tr.schmidt_number
    assert collection_state.stokes_number == st
    assert collection_state.particle_diameter_m == tr.particle.diameter_m


def test_size_regime_has_expected_brownian_to_settling_transition():
    small = transport(dp=0.05e-6, rho=1500.0)
    medium = transport(dp=1.0e-6, rho=1500.0)
    large = transport(dp=10.0e-6, rho=1500.0)
    assert small.brownian_diffusivity_m2_s > medium.brownian_diffusivity_m2_s > large.brownian_diffusivity_m2_s
    assert small.schmidt_number < medium.schmidt_number < large.schmidt_number
    assert small.settling_velocity_m_s < medium.settling_velocity_m_s < large.settling_velocity_m_s


def test_explicit_diameter_provenance_and_invalid_states_fail():
    assert {f.name for f in dataclasses.fields(AerosolParticleProperties)} == {
        "diameter_m", "density_kg_m3", "diameter_basis", "provenance"
    }
    with pytest.raises(ValueError):
        particle(dp=0.0)
    with pytest.raises(ValueError):
        particle(rho=0.0)
    with pytest.raises(ValueError):
        particle(basis="")
    with pytest.raises(ValueError):
        particle(provenance="")
    with pytest.raises(ValueError):
        air(p=0.0)
    with pytest.raises(ValueError):
        transport().stokes_number(friction_velocity_m_s=0.4, surface_regime="vegetated")
    with pytest.raises(ValueError):
        transport().stokes_number(friction_velocity_m_s=0.4, surface_regime="smooth", collector_radius_m=1e-3)
    with pytest.raises(ValueError):
        transport().stokes_number(friction_velocity_m_s=0.4, surface_regime="other")
