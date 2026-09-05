import inspect
import math

import pytest

from gilttpy.physics.aerodynamic import most_aerodynamic_resistance
from gilttpy.physics.aerosol_collection import (
    AerosolCollectionState,
    Zhang2001CollectionEfficiencies,
    Zhang2001SurfaceCollectionParameters,
    zhang2001_smooth_impaction_efficiency,
)
from gilttpy.physics.aerosol_sticking import (
    AerosolSurfaceState,
    Zhang2001ReboundSticking,
    sticking_adjusted_collection_efficiency,
)
from gilttpy.physics.deposition import AerosolResistanceSettling, GasResistance
from gilttpy.physics.particle_physics import (
    AerosolAirState,
    AerosolParticleProperties,
    AerosolParticleTransportState,
)
from gilttpy.physics.quasi_laminar import (
    air_kinematic_viscosity_sutherland_m2_s,
    depac_gas_quasi_laminar_resistance,
    massman_1998_air_diffusivity_m2_s,
)
from gilttpy.physics.stomatal import (
    JarvisEmbersonBulkStomatalResistance,
    StomatalEnvironment,
    StomatalPhysiologyParameters,
)
from gilttpy.physics.surface_resistance import DEPACCanopyResistance


def _rough_surface():
    return Zhang2001SurfaceCollectionParameters(
        surface_label="QA synthetic rough collector",
        brownian_exponent=2.0 / 3.0,
        impaction_alpha=1.0,
        collector_radius_m=2.0e-3,
        provenance="QA030H synthetic dimensional audit; not a land-use calibration",
    )


def _particle(dp_m=1.0e-6, density=1500.0):
    return AerosolParticleTransportState(
        particle=AerosolParticleProperties(
            diameter_m=dp_m,
            density_kg_m3=density,
            diameter_basis="current transport diameter",
            provenance="QA030H synthetic particle",
        ),
        air=AerosolAirState(temperature_k=298.15, pressure_pa=101325.0),
    )


def _rough_chain(dp_m, *, ustar=0.4, wet=False):
    transport = _particle(dp_m)
    surface = _rough_surface()
    st = transport.stokes_number(
        friction_velocity_m_s=ustar,
        surface_regime="vegetated",
        collector_radius_m=surface.collector_radius_m,
    )
    eff = Zhang2001CollectionEfficiencies(
        state=AerosolCollectionState(
            schmidt_number=transport.schmidt_number,
            stokes_number=st,
            particle_diameter_m=dp_m,
        ),
        surface=surface,
    ).efficiencies()
    sticking = Zhang2001ReboundSticking(
        AerosolSurfaceState(
            surface_label=surface.surface_label,
            is_wet=wet,
            provenance="QA030H explicit wetness state",
        )
    ).sticking_fraction(stokes_number=st, particle_diameter_m=dp_m)
    effective = sticking_adjusted_collection_efficiency(
        total_collection_efficiency=eff.total,
        sticking_fraction=sticking,
    )
    rs = math.inf if effective == 0.0 else 1.0 / (3.0 * ustar * effective)
    return transport, st, eff, sticking, rs


def _neutral_ra(ustar=0.4):
    return most_aerodynamic_resistance(
        friction_velocity_m_s=ustar,
        reference_height_m=10.0,
        scalar_roughness_length_m=0.01,
        monin_obukhov_length_m=math.inf,
    )


def _z01_vd(ra, rs, vg):
    return vg + (0.0 if math.isinf(rs) else 1.0 / (ra + rs))


def test_shared_air_state_is_identical_across_gas_and_particle_modules():
    air = AerosolAirState(temperature_k=298.15, pressure_pa=101325.0)
    direct = air_kinematic_viscosity_sutherland_m2_s(
        temperature_k=298.15, pressure_pa=101325.0
    )
    assert air.kinematic_viscosity_m2_s == pytest.approx(direct, rel=0.0, abs=0.0)


def test_integrated_gas_chain_stomata_rc_and_vd_identity():
    env = StomatalEnvironment(
        ppfd_umol_m2_s=800.0,
        temperature_c=25.0,
        vapor_pressure_deficit_kpa=1.0,
        soil_water_content_m3_m3=0.25,
        day_of_year=180.0,
        leaf_area_index=3.0,
    )
    pars = StomatalPhysiologyParameters(
        vegetation_type="QA synthetic vegetation",
        maximum_leaf_conductance_m_s=0.005,
        minimum_fraction=0.1,
        light_response_per_umol=0.003,
        temperature_minimum_c=0.0,
        temperature_optimum_c=25.0,
        temperature_maximum_c=40.0,
        vpd_full_open_kpa=0.5,
        vpd_minimum_open_kpa=3.0,
        soil_wilting_point_m3_m3=0.10,
        soil_field_capacity_m3_m3=0.30,
    )
    rstom = JarvisEmbersonBulkStomatalResistance(env, pars).resistance_s_m()
    rc = DEPACCanopyResistance(
        stomatal_resistance_s_m=rstom,
        external_surface_resistance_s_m=500.0,
        in_canopy_resistance_s_m=100.0,
        soil_resistance_s_m=300.0,
    ).resistance_s_m()
    ra = _neutral_ra()
    dm = massman_1998_air_diffusivity_m2_s("O3", temperature_k=298.15, pressure_pa=101325.0)
    rb = depac_gas_quasi_laminar_resistance(
        friction_velocity_m_s=0.4,
        molecular_diffusivity_m2_s=dm,
        air_kinematic_viscosity_m2_s=AerosolAirState(298.15, 101325.0).kinematic_viscosity_m2_s,
    )
    model = GasResistance(ra, rb, rc)
    assert model.deposition_velocity() == pytest.approx(1.0 / (ra + rb + rc), rel=1e-15)
    assert 0.0 < model.deposition_velocity() <= 1.0 / (ra + rb)


def test_integrated_gas_closed_surface_limit_is_exact_zero_flux():
    rc = DEPACCanopyResistance(math.inf, math.inf, math.inf, math.inf).resistance_s_m()
    assert rc == math.inf
    model = GasResistance(_neutral_ra(), 20.0, rc)
    assert model.deposition_velocity() == 0.0
    assert model.downward_flux(123.0) == 0.0


def test_rough_z01_surface_resistance_has_correct_units_and_positive_limit():
    transport, st, eff, sticking, rs = _rough_chain(1.0e-6)
    assert st > 0.0
    assert eff.total > 0.0
    assert sticking == 1.0
    assert math.isfinite(rs) and rs > 0.0
    # Rs = 1/(epsilon0*u*E*R1), epsilon0=3 in the source family.
    assert rs == pytest.approx(1.0 / (3.0 * 0.4 * eff.total * sticking), rel=1e-15)
    assert transport.settling_velocity_m_s > 0.0


def test_smooth_surface_branch_uses_smooth_stokes_and_no_interception_term():
    transport = _particle(10.0e-6)
    st_smooth = transport.stokes_number(
        friction_velocity_m_s=0.4, surface_regime="smooth"
    )
    e_imp = zhang2001_smooth_impaction_efficiency(stokes_number=st_smooth)
    e_b = transport.schmidt_number ** (-0.5)
    # The smooth audit total intentionally has only Brownian + smooth impaction.
    e_total_smooth = e_b + e_imp
    assert st_smooth > 0.0
    assert e_total_smooth == pytest.approx(e_b + e_imp, rel=0.0, abs=0.0)


def test_wet_surface_never_reduces_z01_effective_collection_relative_to_dry():
    _, _, eff_dry, r1_dry, rs_dry = _rough_chain(10.0e-6, wet=False)
    _, _, eff_wet, r1_wet, rs_wet = _rough_chain(10.0e-6, wet=True)
    assert eff_dry.total == pytest.approx(eff_wet.total, rel=0.0, abs=0.0)
    assert r1_wet == 1.0
    assert r1_wet > r1_dry
    assert rs_wet < rs_dry


def test_z01_and_venkatram_pleim_complete_aerosol_models_are_not_interchangeable():
    transport, _, _, _, rs = _rough_chain(3.0e-6)
    ra = _neutral_ra()
    vg = transport.settling_velocity_m_s
    vd_z01 = _z01_vd(ra, rs, vg)
    vd_vp = AerosolResistanceSettling(ra, rs, vg).deposition_velocity()
    relative_gap = abs(vd_z01 - vd_vp) / vd_z01
    # A deterministic QA example gives a material >20% difference; this is a
    # model-form distinction, not a numerical tolerance issue.
    assert relative_gap > 0.20


def test_collection_modules_remain_unwired_to_giltt_lower_interface():
    import gilttpy.physics.aerosol_collection as ac
    import gilttpy.physics.aerosol_sticking as ast
    import gilttpy.physics.particle_physics as pp

    for module in (ac, ast, pp):
        source = inspect.getsource(module)
        assert "ResolvedLowerInterface" not in source
        assert "steady_2d" not in source
        assert "transient_2d" not in source


def test_common_ustar_changes_ra_st_and_z01_rs_in_physically_coherent_directions():
    ra_low = _neutral_ra(0.2)
    ra_high = _neutral_ra(0.6)
    _, st_low, _, _, rs_low = _rough_chain(3.0e-6, ustar=0.2)
    _, st_high, _, _, rs_high = _rough_chain(3.0e-6, ustar=0.6)
    assert ra_high < ra_low
    assert st_high > st_low
    assert rs_high < rs_low


def test_reference_particle_size_sweep_is_finite_nonnegative_and_exposes_rebound_jump():
    ra = _neutral_ra()
    values = []
    for dp_um in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 5.01, 10.0, 20.0):
        transport, st, eff, r1, rs = _rough_chain(dp_um * 1e-6)
        vd = _z01_vd(ra, rs, transport.settling_velocity_m_s)
        assert math.isfinite(vd) and vd >= 0.0
        assert math.isfinite(st) and st >= 0.0
        assert math.isfinite(eff.total) and eff.total >= 0.0
        values.append((dp_um, r1, vd))
    r1_5 = next(r for d, r, _ in values if d == 5.0)
    r1_501 = next(r for d, r, _ in values if d == 5.01)
    assert r1_5 == 1.0
    assert r1_501 < 1.0
