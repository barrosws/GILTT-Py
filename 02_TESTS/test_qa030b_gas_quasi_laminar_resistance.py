import math
import pytest

from gilttpy.physics.quasi_laminar import (
    DEFAULT_AIR_PRANDTL,
    MASSMAN_1998_AIR_DIFFUSIVITY,
    DEPACGasQuasiLaminarResistance,
    MassmanDEPACGasQuasiLaminarResistance,
    air_dynamic_viscosity_sutherland_pa_s,
    air_kinematic_viscosity_sutherland_m2_s,
    depac_gas_quasi_laminar_resistance,
    massman_1998_air_diffusivity_m2_s,
    schmidt_number,
)


def test_schmidt_number_exact_ratio():
    assert schmidt_number(
        air_kinematic_viscosity_m2_s=1.5e-5,
        molecular_diffusivity_m2_s=2.0e-5,
    ) == pytest.approx(0.75, rel=0, abs=1e-15)


def test_depac_wesely_formula_exact_reference():
    ustar = 0.4
    nu = 1.50e-5
    d = 1.20e-5
    pr = 0.72
    expected = 2.0 / (0.4 * ustar) * ((nu / d) / pr) ** (2.0 / 3.0)
    got = depac_gas_quasi_laminar_resistance(
        friction_velocity_m_s=ustar,
        molecular_diffusivity_m2_s=d,
        air_kinematic_viscosity_m2_s=nu,
        prandtl_number=pr,
    )
    assert got == pytest.approx(expected, rel=2e-15)


def test_rb_inverse_friction_velocity_scaling():
    kw = dict(molecular_diffusivity_m2_s=1.4e-5, air_kinematic_viscosity_m2_s=1.5e-5)
    r1 = depac_gas_quasi_laminar_resistance(friction_velocity_m_s=0.25, **kw)
    r2 = depac_gas_quasi_laminar_resistance(friction_velocity_m_s=0.50, **kw)
    assert r1 / r2 == pytest.approx(2.0, rel=2e-15)


def test_rb_diffusivity_and_prandtl_power_laws():
    base = dict(friction_velocity_m_s=0.4, air_kinematic_viscosity_m2_s=1.5e-5)
    r_d1 = depac_gas_quasi_laminar_resistance(molecular_diffusivity_m2_s=1.0e-5, prandtl_number=0.72, **base)
    r_d2 = depac_gas_quasi_laminar_resistance(molecular_diffusivity_m2_s=2.0e-5, prandtl_number=0.72, **base)
    assert r_d1 / r_d2 == pytest.approx(2.0 ** (2.0 / 3.0), rel=2e-15)

    r_p1 = depac_gas_quasi_laminar_resistance(molecular_diffusivity_m2_s=1.4e-5, prandtl_number=0.72, **base)
    r_p2 = depac_gas_quasi_laminar_resistance(molecular_diffusivity_m2_s=1.4e-5, prandtl_number=1.44, **base)
    assert r_p1 / r_p2 == pytest.approx(2.0 ** (2.0 / 3.0), rel=2e-15)


def test_sutherland_air_viscosity_reference_near_300k():
    mu = air_dynamic_viscosity_sutherland_pa_s(300.0)
    assert mu == pytest.approx(1.8460e-5, rel=3e-4)
    nu = air_kinematic_viscosity_sutherland_m2_s(temperature_k=300.0, pressure_pa=101325.0)
    assert 1.55e-5 < nu < 1.60e-5


def test_massman_reference_values_and_provenance_are_explicit():
    assert massman_1998_air_diffusivity_m2_s("SO2") == pytest.approx(0.1089e-4, rel=0, abs=1e-16)
    assert massman_1998_air_diffusivity_m2_s("NH3") == pytest.approx(0.1978e-4, rel=0, abs=1e-16)
    assert massman_1998_air_diffusivity_m2_s("O3") == pytest.approx(0.1444e-4, rel=0, abs=1e-16)
    assert "model_estimate" in MASSMAN_1998_AIR_DIFFUSIVITY["O3"].provenance
    assert "model_estimate" in MASSMAN_1998_AIR_DIFFUSIVITY["NO"].provenance
    assert "model_estimate" in MASSMAN_1998_AIR_DIFFUSIVITY["NO2"].provenance


def test_massman_temperature_pressure_scaling_exact():
    d0 = massman_1998_air_diffusivity_m2_s("CO2")
    d2p = massman_1998_air_diffusivity_m2_s("CO2", pressure_pa=2.0 * 101325.0)
    assert d2p / d0 == pytest.approx(0.5, rel=2e-15)

    t2 = 2.0 * 273.15
    dt = massman_1998_air_diffusivity_m2_s("CO2", temperature_k=t2)
    assert dt / d0 == pytest.approx(2.0 ** 1.81, rel=2e-15)


def test_species_dependence_ordering_follows_diffusivity_not_molar_mass_shortcut():
    # At common air state and u*, larger D -> smaller Sc -> smaller Rb.
    so2 = MassmanDEPACGasQuasiLaminarResistance("SO2", 0.4, temperature_k=298.15)
    o3 = MassmanDEPACGasQuasiLaminarResistance("O3", 0.4, temperature_k=298.15)
    nh3 = MassmanDEPACGasQuasiLaminarResistance("NH3", 0.4, temperature_k=298.15)
    assert nh3.molecular_diffusivity_m2_s > o3.molecular_diffusivity_m2_s > so2.molecular_diffusivity_m2_s
    assert nh3.resistance_s_m() < o3.resistance_s_m() < so2.resistance_s_m()


def test_explicit_and_massman_provider_paths_agree_when_properties_match():
    ref = MassmanDEPACGasQuasiLaminarResistance("O3", 0.37, temperature_k=293.15, pressure_pa=98000.0)
    explicit = DEPACGasQuasiLaminarResistance(
        friction_velocity_m_s=0.37,
        molecular_diffusivity_m2_s=ref.molecular_diffusivity_m2_s,
        air_kinematic_viscosity_m2_s=ref.air_kinematic_viscosity_m2_s,
        prandtl_number=DEFAULT_AIR_PRANDTL,
    )
    assert explicit.schmidt_number == pytest.approx(ref.schmidt_number, rel=2e-15)
    assert explicit.resistance_s_m() == pytest.approx(ref.resistance_s_m(), rel=2e-15)


def test_invalid_inputs_and_unknown_species_fail_explicitly():
    with pytest.raises(ValueError):
        schmidt_number(air_kinematic_viscosity_m2_s=0.0, molecular_diffusivity_m2_s=1e-5)
    with pytest.raises(ValueError):
        depac_gas_quasi_laminar_resistance(
            friction_velocity_m_s=0.0,
            molecular_diffusivity_m2_s=1e-5,
            air_kinematic_viscosity_m2_s=1.5e-5,
        )
    with pytest.raises(ValueError):
        depac_gas_quasi_laminar_resistance(
            friction_velocity_m_s=0.3,
            molecular_diffusivity_m2_s=1e-5,
            air_kinematic_viscosity_m2_s=1.5e-5,
            prandtl_number=0.0,
        )
    with pytest.raises(KeyError):
        massman_1998_air_diffusivity_m2_s("HNO3")
